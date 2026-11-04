"""Unit tests for PCI discovery, flock leasing, and FIFO queue ordering."""

import fcntl
import os
import pathlib
from unittest import mock

from absl.testing import absltest

from xprof.xloop import chip_broker


def _write_pci_device(
    root: pathlib.Path,
    address: str,
    *,
    vendor: str = "0x1ae0",
    device: str = "0x0076",
    iommu_group: str | None = "12",
) -> None:
  """Materializes one fake PCI device under a fake sysfs root."""
  device_dir = root / address
  device_dir.mkdir(parents=True)
  (device_dir / "vendor").write_text(f"{vendor}\n")
  (device_dir / "device").write_text(f"{device}\n")
  if iommu_group is not None:
    group_dir = root / ".iommu_groups" / iommu_group
    group_dir.mkdir(parents=True, exist_ok=True)
    (device_dir / "iommu_group").symlink_to(group_dir)


def _dead_pid() -> int:
  """Returns a pid that is certainly not running.

  Probes above the usual `pid_max` (2^22 on modern Linux). The kernel reports
  ESRCH both for an out-of-range pid and for an unused in-range one, so this
  works whatever `pid_max` is set to. Spawning a real process and waiting for
  it would be the obvious alternative, but `sys.executable` inside a py_test is
  the test binary itself.

  Returns:
    An unused pid.
  """
  for candidate in range(4_000_000, 4_000_100):
    try:
      os.kill(candidate, 0)
    except ProcessLookupError:
      return candidate
    except PermissionError:
      continue
  raise AssertionError("could not find an unused pid to test reaping with")


class DiscoverChipsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.root = pathlib.Path(self.create_tempdir().full_path)
    # Discovery consults the environment first; keep tests hermetic.
    self.enter_context(
        mock.patch.dict(os.environ, {"XLOOP_NUM_CHIPS": ""}, clear=False)
    )
    os.environ.pop("XLOOP_NUM_CHIPS", None)
    os.environ.pop("XLOOP_TPU_PCI_DEVICE_IDS", None)

  def test_collapses_functions_of_one_chip(self):
    # Sibling cores share a slot and differ only in function index.
    _write_pci_device(self.root, "0000:00:05.0", iommu_group="7")
    _write_pci_device(self.root, "0000:00:05.1", iommu_group="7")
    chips = chip_broker.discover_chips(str(self.root))
    self.assertLen(chips, 1)
    self.assertEqual(chips[0].pci_slot, "0000:00:05")
    self.assertEqual(chips[0].vfio_path, "/dev/vfio/7")

  def test_indices_are_dense_and_slot_ordered(self):
    for i, address in enumerate(["0000:00:07.0", "0000:00:05.0"]):
      _write_pci_device(self.root, address, iommu_group=str(10 + i))
    chips = chip_broker.discover_chips(str(self.root))
    self.assertEqual([c.index for c in chips], [0, 1])
    self.assertEqual([c.pci_slot for c in chips], ["0000:00:05", "0000:00:07"])

  def test_ignores_other_vendors(self):
    _write_pci_device(self.root, "0000:00:05.0", vendor="0x8086")
    self.assertEmpty(chip_broker.discover_chips(str(self.root)))

  def test_ignores_google_devices_without_iommu_group(self):
    # Incidental Google-vendor devices are not passed through to the guest and
    # so have no IOMMU group; accelerators always do.
    _write_pci_device(self.root, "0000:00:05.0", iommu_group=None)
    self.assertEmpty(chip_broker.discover_chips(str(self.root)))

  def test_allowlist_restricts_by_device_id(self):
    _write_pci_device(self.root, "0000:00:05.0", device="0x0076")
    _write_pci_device(self.root, "0000:00:06.0", device="0x00ff")
    with mock.patch.dict(os.environ, {"XLOOP_TPU_PCI_DEVICE_IDS": "0x0076"}):
      chips = chip_broker.discover_chips(str(self.root))
    self.assertLen(chips, 1)
    self.assertEqual(chips[0].pci_slot, "0000:00:05")

  def test_allowlist_waives_iommu_requirement(self):
    _write_pci_device(self.root, "0000:00:05.0", iommu_group=None)
    with mock.patch.dict(os.environ, {"XLOOP_TPU_PCI_DEVICE_IDS": "0x0076"}):
      chips = chip_broker.discover_chips(str(self.root))
    self.assertLen(chips, 1)
    self.assertIsNone(chips[0].vfio_path)

  def test_chip_count_override(self):
    with mock.patch.dict(os.environ, {"XLOOP_NUM_CHIPS": "4"}):
      chips = chip_broker.discover_chips(str(self.root))
    self.assertLen(chips, 4)
    self.assertEqual([c.index for c in chips], [0, 1, 2, 3])

  def test_malformed_chip_count_override_is_ignored(self):
    _write_pci_device(self.root, "0000:00:05.0")
    with mock.patch.dict(os.environ, {"XLOOP_NUM_CHIPS": "many"}):
      chips = chip_broker.discover_chips(str(self.root))
    self.assertLen(chips, 1)

  def test_missing_pci_root_yields_no_chips(self):
    self.assertEmpty(chip_broker.discover_chips("/nonexistent/pci"))


class ChipEnvTest(absltest.TestCase):

  def test_env_pins_process_to_chip_by_device_path(self):
    env = chip_broker.Chip(index=3, pci_slot="s", vfio_path="/dev/vfio/9").env()
    self.assertEqual(env["TPU_VISIBLE_DEVICE_PATHS"], "/dev/vfio/9")
    # Without this libtpu refuses a second load, which is what would otherwise
    # prevent concurrent workers entirely.
    self.assertEqual(env["ALLOW_MULTIPLE_LIBTPU_LOAD"], "1")

  def test_env_clears_every_selection_variable_it_does_not_set(self):
    # Omitting them is not enough. The worker inherits the caller's
    # environment, and libtpu applies the first of the three that is set, so
    # an exported TPU_VISIBLE_CHIPS would outrank a lease expressed as
    # TPU_VISIBLE_DEVICES and send the worker to the wrong chip.
    env = chip_broker.Chip(index=2, pci_slot="s", vfio_path=None).env()
    self.assertEqual(env["TPU_VISIBLE_DEVICES"], "2")
    self.assertIsNone(env["TPU_VISIBLE_DEVICE_PATHS"])
    self.assertIsNone(env["TPU_VISIBLE_CHIPS"])

  def test_env_clears_the_index_when_pinning_by_path(self):
    env = chip_broker.Chip(index=3, pci_slot="s", vfio_path="/dev/vfio/9").env()
    self.assertIsNone(env["TPU_VISIBLE_DEVICES"])
    self.assertIsNone(env["TPU_VISIBLE_CHIPS"])

  def test_every_known_selection_variable_is_accounted_for(self):
    env = chip_broker.Chip(index=0, pci_slot="s", vfio_path="/dev/vfio/1").env()
    for name in chip_broker.SELECTION_ENV_VARS:
      self.assertIn(name, env, f"{name} is neither set nor cleared")

  def test_env_does_not_inject_libtpu_flags(self):
    # libtpu aborts on an unknown flag and the flag set varies between builds,
    # so the engine must not add any of its own. Regression test for a flag
    # that killed every worker on a Cloud TPU v6e.
    env = chip_broker.Chip(index=0, pci_slot="s", vfio_path="/dev/vfio/1").env()
    self.assertNotIn("LIBTPU_INIT_ARGS", env)

  def test_env_falls_back_to_index_when_not_vfio_bound(self):
    env = chip_broker.Chip(index=2, pci_slot="s", vfio_path=None).env()
    self.assertIsNone(env["TPU_VISIBLE_DEVICE_PATHS"])
    self.assertEqual(env["TPU_VISIBLE_DEVICES"], "2")


def _chips(n: int) -> list[chip_broker.Chip]:
  return [
      chip_broker.Chip(index=i, pci_slot=f"slot{i}", vfio_path=None)
      for i in range(n)
  ]


class ChipBrokerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.lease_dir = self.create_tempdir().full_path

  def _broker(self, n: int = 2, num_chips: int | None = None):
    return chip_broker.ChipBroker(
        num_chips=num_chips, lease_dir=self.lease_dir, chips=_chips(n)
    )

  def test_raises_when_no_chips_present(self):
    with self.assertRaisesRegex(
        chip_broker.ChipBrokerError, "no accelerators found"
    ):
      chip_broker.ChipBroker(lease_dir=self.lease_dir, chips=[])

  def test_num_chips_caps_capacity(self):
    self.assertEqual(self._broker(n=8, num_chips=3).capacity, 3)

  def test_num_chips_above_available_is_clamped(self):
    self.assertEqual(self._broker(n=2, num_chips=8).capacity, 2)

  def test_num_chips_must_be_positive(self):
    with self.assertRaises(chip_broker.ChipBrokerError):
      self._broker(n=2, num_chips=0)

  def test_leases_are_exclusive(self):
    broker = self._broker(n=1)
    with broker.lease():
      with self.assertRaisesRegex(
          chip_broker.ChipBrokerError, "no chip available"
      ):
        broker.acquire(timeout_sec=0.0)

  def test_lease_is_released_on_exit(self):
    broker = self._broker(n=1)
    with broker.lease() as first:
      pass
    with broker.lease() as second:
      self.assertEqual(second.chip.index, first.chip.index)

  def test_lease_is_released_on_exception(self):
    broker = self._broker(n=1)
    with self.assertRaises(ValueError):
      with broker.lease():
        raise ValueError("boom")
    # Capacity must come back; otherwise one crash wedges the host.
    with broker.lease():
      pass

  def test_distinct_chips_are_handed_out(self):
    broker = self._broker(n=2)
    with broker.lease() as a, broker.lease() as b:
      self.assertNotEqual(a.chip.index, b.chip.index)

  def test_separate_broker_instances_share_state(self):
    # Two brokers over one directory stand in for two processes.
    first = self._broker(n=1)
    second = self._broker(n=1)
    with first.lease():
      with self.assertRaises(chip_broker.ChipBrokerError):
        second.acquire(timeout_sec=0.0)

  def test_double_release_is_harmless(self):
    broker = self._broker(n=1)
    claim = broker.acquire(timeout_sec=1.0)
    broker.release(claim)
    broker.release(claim)
    with broker.lease():
      pass

  def test_lock_file_without_a_held_lock_is_claimable(self):
    # The critical case: a SIGKILLed worker runs no atexit handler and leaves
    # its lock file behind. The kernel dropped its flock when it died, so the
    # chip must come straight back into service.
    broker = self._broker(n=1)
    stale = pathlib.Path(self.lease_dir) / "chip_0.lock"
    stale.write_text(f"{_dead_pid()}\n")
    with broker.lease() as claim:
      self.assertEqual(claim.chip.index, 0)

  def test_lock_held_by_another_process_blocks_acquisition(self):
    broker = self._broker(n=1)
    path = pathlib.Path(self.lease_dir) / "chip_0.lock"
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o666)
    self.addCleanup(os.close, fd)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with self.assertRaises(chip_broker.ChipBrokerError):
      broker.acquire(timeout_sec=0.0)

  def test_a_recycled_pid_does_not_keep_a_dead_lease_alive(self):
    # The bug flock exists to prevent: the file names a live pid, but that
    # process is not the one that took the lease and holds no lock. Judging
    # liveness from the recorded pid would strand the chip indefinitely.
    broker = self._broker(n=1)
    path = pathlib.Path(self.lease_dir) / "chip_0.lock"
    path.write_text(f"{os.getpid()}\n")
    with broker.lease() as claim:
      self.assertEqual(claim.chip.index, 0)

  def test_corrupt_lock_file_is_claimable(self):
    broker = self._broker(n=1)
    stale = pathlib.Path(self.lease_dir) / "chip_0.lock"
    stale.write_text("not-a-pid\n")
    with broker.lease():
      pass

  def test_released_lease_can_be_reacquired(self):
    broker = self._broker(n=1)
    with broker.lease() as first:
      self.assertEqual(first.chip.index, 0)
    with broker.lease() as second:
      self.assertEqual(second.chip.index, 0)

  def test_reap_all_clears_stale_tickets(self):
    broker = self._broker(n=2)
    (pathlib.Path(self.lease_dir) / "ticket_000000000000").write_text(
        f"{_dead_pid()}\n"
    )
    self.assertEqual(broker.reap_all(), 1)
    self.assertEqual(broker.capacity, 2)

  def test_reap_all_leaves_chip_locks_alone(self):
    # Unlinking a lock file would detach any process already blocked on it.
    broker = self._broker(n=1)
    path = pathlib.Path(self.lease_dir) / "chip_0.lock"
    path.write_text(f"{_dead_pid()}\n")
    self.assertEqual(broker.reap_all(), 0)
    self.assertTrue(path.exists())

  def test_reap_all_spares_live_tickets(self):
    broker = self._broker(n=1)
    (pathlib.Path(self.lease_dir) / "ticket_000000000000").write_text(
        f"{os.getpid()}\n"
    )
    self.assertEqual(broker.reap_all(), 0)

  def test_older_live_ticket_blocks_acquisition(self):
    # FIFO: a later arrival must not overtake a waiting earlier one.
    broker = self._broker(n=1)
    (pathlib.Path(self.lease_dir) / "ticket_000000000000").write_text(
        f"{os.getpid()}\n"
    )
    with self.assertRaises(chip_broker.ChipBrokerError):
      broker.acquire(timeout_sec=0.0)

  def test_older_dead_ticket_does_not_block(self):
    broker = self._broker(n=1)
    (pathlib.Path(self.lease_dir) / "ticket_000000000000").write_text(
        f"{_dead_pid()}\n"
    )
    with broker.lease():
      pass

  def test_tickets_are_released_after_acquisition(self):
    broker = self._broker(n=1)
    with broker.lease():
      leftover = list(pathlib.Path(self.lease_dir).glob("ticket_*"))
    self.assertEmpty(leftover)

  def test_tickets_are_released_after_timeout(self):
    # A waiter that gives up must not leave a ticket behind blocking others.
    broker = self._broker(n=1)
    with broker.lease():
      with self.assertRaises(chip_broker.ChipBrokerError):
        broker.acquire(timeout_sec=0.0)
    self.assertEmpty(list(pathlib.Path(self.lease_dir).glob("ticket_*")))

  def test_on_wait_reports_queue_position(self):
    broker = self._broker(n=1)
    seen = []
    with broker.lease():
      with self.assertRaises(chip_broker.ChipBrokerError):
        broker.acquire(
            timeout_sec=0.3, on_wait=lambda pos, cap: seen.append((pos, cap))
        )
    self.assertNotEmpty(seen)
    self.assertEqual(seen[0][1], 1)

  def test_lease_dir_is_created(self):
    nested = os.path.join(self.lease_dir, "a", "b")
    broker = chip_broker.ChipBroker(lease_dir=nested, chips=_chips(1))
    self.assertTrue(os.path.isdir(broker.lease_dir))


class DefaultLeaseDirTest(absltest.TestCase):

  def test_is_absolute_and_shared_across_users(self):
    # Must not be keyed by user: chips are host-global, so a per-user
    # namespace would let two users each lease the same chip.
    path = chip_broker.default_lease_dir()
    self.assertTrue(os.path.isabs(path))
    self.assertEqual(os.path.basename(path), "xloop_chips")

  def test_directory_is_writable_by_other_users(self):
    directory = os.path.join(self.create_tempdir().full_path, "chips")
    chip_broker.ChipBroker(lease_dir=directory, chips=_chips(1))
    mode = os.stat(directory).st_mode
    self.assertTrue(mode & 0o002, "others must be able to add a lock")
    self.assertTrue(mode & 0o1000, "sticky bit must protect others' locks")


class ForeignManagerTest(absltest.TestCase):

  def test_no_manager_on_an_ordinary_host(self):
    with mock.patch.object(os.path, "exists", return_value=False):
      self.assertIsNone(chip_broker.detect_foreign_manager())

  def test_foreign_runtime_socket_is_detected(self):
    with mock.patch.object(
        os.path,
        "exists",
        side_effect=lambda p: p == "/var/google/services/xos/xos.sock",
    ):
      reason = chip_broker.detect_foreign_manager()
    self.assertIsNotNone(reason)
    self.assertIn("/var/google/services/xos/xos.sock", reason)

  def test_broker_refuses_to_start_under_a_foreign_manager(self):
    with mock.patch.object(
        chip_broker, "detect_foreign_manager", return_value="daemon present"
    ):
      with self.assertRaises(chip_broker.ChipBrokerError):
        chip_broker.ChipBroker(chips=_chips(1))


if __name__ == "__main__":
  absltest.main()
