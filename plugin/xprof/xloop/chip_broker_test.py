# Copyright 2026 The XProf Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
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


def _write_accel_device(
    class_root: pathlib.Path,
    name: str,
    address: str,
    *,
    with_pci_address_file: bool = True,
    pci_root: pathlib.Path | None = None,
) -> None:
  """Materializes one fake `accel` class entry pointing at a PCI function."""
  entry = class_root / name
  entry.mkdir(parents=True)
  if with_pci_address_file:
    (entry / "pci_address").write_text(f"{address}\n")
  if pci_root is not None:
    (entry / "device").symlink_to(pci_root / address)


def _add_node(dev_root: pathlib.Path, relative: str) -> None:
  """Creates a stand-in for a device node, e.g. `vfio/12`, under `dev_root`."""
  node = dev_root / relative
  node.parent.mkdir(parents=True, exist_ok=True)
  node.write_text("")


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
    self.dev = pathlib.Path(self.create_tempdir().full_path)
    self.accel = pathlib.Path(self.create_tempdir().full_path)
    # Discovery consults the environment first; keep tests hermetic.
    self.enter_context(mock.patch.dict(os.environ))
    for name in (
        "XLOOP_NUM_CHIPS",
        "XLOOP_TPU_PCI_DEVICE_IDS",
        *chip_broker.SELECTION_ENV_VARS,
    ):
      os.environ.pop(name, None)

  def _discover(self) -> list[chip_broker.Chip]:
    return chip_broker.discover_chips(
        str(self.root), dev_root=str(self.dev), accel_class_root=str(self.accel)
    )

  def _write_vfio_chip(self, address: str, group: str, **kwargs) -> None:
    _write_pci_device(self.root, address, iommu_group=group, **kwargs)
    _add_node(self.dev, f"vfio/{group}")

  def test_collapses_functions_of_one_chip(self):
    # Sibling cores share a slot and differ only in function index.
    _write_pci_device(self.root, "0000:00:05.0", iommu_group="7")
    _write_pci_device(self.root, "0000:00:05.1", iommu_group="7")
    _add_node(self.dev, "vfio/7")
    chips = self._discover()
    self.assertLen(chips, 1)
    self.assertEqual(chips[0].pci_slot, "0000:00:05")
    self.assertEqual(chips[0].device_path, "/dev/vfio/7")

  def test_indices_are_dense_and_slot_ordered(self):
    for i, address in enumerate(["0000:00:07.0", "0000:00:05.0"]):
      self._write_vfio_chip(address, str(10 + i))
    chips = self._discover()
    self.assertEqual([c.index for c in chips], [0, 1])
    self.assertEqual([c.pci_slot for c in chips], ["0000:00:05", "0000:00:07"])

  def test_ignores_other_vendors(self):
    self._write_vfio_chip("0000:00:05.0", "7", vendor="0x8086")
    self.assertEmpty(self._discover())

  def test_ignores_google_devices_that_are_not_tpus(self):
    # Hosts without accelerators still enumerate other Google-vendor devices,
    # some with an IOMMU group; only known TPU device IDs count.
    self._write_vfio_chip("0000:00:05.0", "7", device="0x0042")
    self.assertEmpty(self._discover())

  def test_every_known_tpu_generation_is_found(self):
    ids = ["0x0027", "0x0056", "0x005e", "0x0062", "0x0063", "0x006f", "0x0076"]
    for i, device_id in enumerate(ids):
      self._write_vfio_chip(f"0000:00:{i + 4:02x}.0", str(i), device=device_id)
    self.assertLen(self._discover(), len(ids))

  def test_keeps_only_chips_whose_node_is_present(self):
    # A container that holds 4 of the host's 8 chips: sysfs lists all 8, but
    # only 4 nodes are in this process's /dev.
    for i in range(8):
      _write_pci_device(
          self.root, f"0000:00:{i + 4:02x}.0", iommu_group=str(20 + i)
      )
    for group in ("21", "23", "24", "26"):
      _add_node(self.dev, f"vfio/{group}")
    chips = self._discover()
    self.assertEqual([c.index for c in chips], [0, 1, 2, 3])
    self.assertEqual(
        [c.device_path for c in chips],
        ["/dev/vfio/21", "/dev/vfio/23", "/dev/vfio/24", "/dev/vfio/26"],
    )

  def test_no_chip_is_kept_when_no_node_is_present(self):
    for i in range(4):
      _write_pci_device(
          self.root, f"0000:00:{i + 4:02x}.0", iommu_group=str(20 + i)
      )
    self.assertEmpty(self._discover())

  def test_an_accel_chip_is_pinned_by_its_accel_node(self):
    _write_pci_device(self.root, "0000:00:05.0", iommu_group=None)
    _write_accel_device(self.accel, "accel0", "0000:00:05.0")
    _add_node(self.dev, "accel0")
    [chip] = self._discover()
    self.assertEqual(chip.device_path, "/dev/accel0")

  def test_an_accel_chip_with_an_iommu_group_uses_its_accel_node(self):
    # A host with an IOMMU gives every device a group, also one that is not
    # bound to vfio. Then there is no /dev/vfio node, and the accel node is
    # the one that works.
    _write_pci_device(self.root, "0000:00:05.0", iommu_group="7")
    _write_accel_device(self.accel, "accel3", "0000:00:05.0")
    _add_node(self.dev, "accel3")
    [chip] = self._discover()
    self.assertEqual(chip.device_path, "/dev/accel3")

  def test_accel_mapping_falls_back_to_the_device_link(self):
    _write_pci_device(self.root, "0000:00:05.0", iommu_group=None)
    _write_accel_device(
        self.accel,
        "accel1",
        "0000:00:05.0",
        with_pci_address_file=False,
        pci_root=self.root,
    )
    _add_node(self.dev, "accel1")
    [chip] = self._discover()
    self.assertEqual(chip.device_path, "/dev/accel1")

  def test_accel_chips_in_a_container_subset(self):
    for i in range(4):
      address = f"0000:00:{i + 4:02x}.0"
      _write_pci_device(self.root, address, iommu_group=None)
      _write_accel_device(self.accel, f"accel{i}", address)
    _add_node(self.dev, "accel2")
    _add_node(self.dev, "accel3")
    chips = self._discover()
    self.assertEqual(
        [(c.index, c.device_path) for c in chips],
        [(0, "/dev/accel2"), (1, "/dev/accel3")],
    )

  def test_a_chip_without_any_device_node_is_skipped(self):
    # No vfio group and no accel entry: nothing can pin a worker to it.
    _write_pci_device(self.root, "0000:00:05.0", iommu_group=None)
    self.assertEmpty(self._discover())

  def test_platform_device_paths_are_intersected(self):
    for i, group in enumerate(("21", "22", "23")):
      self._write_vfio_chip(f"0000:00:{i + 4:02x}.0", group)
    with mock.patch.dict(
        os.environ,
        # /dev/vfio/99 is not present and must not appear.
        {"TPU_VISIBLE_DEVICE_PATHS": "/dev/vfio/23,/dev/vfio/21,/dev/vfio/99"},
    ):
      chips = self._discover()
    self.assertEqual(
        [(c.index, c.device_path) for c in chips],
        [(0, "/dev/vfio/21"), (1, "/dev/vfio/23")],
    )

  def test_platform_index_follows_the_libtpu_scanner_order(self):
    # libtpu counts readable nodes in sorted path order, so "/dev/vfio/10"
    # comes before "/dev/vfio/9" even though its slot is later.
    self._write_vfio_chip("0000:00:04.0", "9")
    self._write_vfio_chip("0000:00:05.0", "10")
    with mock.patch.dict(os.environ, {"TPU_VISIBLE_CHIPS": "0"}):
      [chip] = self._discover()
    self.assertEqual(chip.device_path, "/dev/vfio/10")
    self.assertEqual(chip.index, 0)

  def test_platform_device_paths_take_precedence_over_indices(self):
    self._write_vfio_chip("0000:00:04.0", "21")
    self._write_vfio_chip("0000:00:05.0", "22")
    with mock.patch.dict(
        os.environ,
        {"TPU_VISIBLE_DEVICE_PATHS": "/dev/vfio/22", "TPU_VISIBLE_CHIPS": "0"},
    ):
      [chip] = self._discover()
    self.assertEqual(chip.device_path, "/dev/vfio/22")

  def test_platform_chips_take_precedence_over_devices(self):
    self._write_vfio_chip("0000:00:04.0", "21")
    self._write_vfio_chip("0000:00:05.0", "22")
    with mock.patch.dict(
        os.environ, {"TPU_VISIBLE_CHIPS": "1", "TPU_VISIBLE_DEVICES": "0"}
    ):
      [chip] = self._discover()
    self.assertEqual(chip.device_path, "/dev/vfio/22")

  def test_an_empty_platform_variable_is_ignored(self):
    self._write_vfio_chip("0000:00:04.0", "21")
    with mock.patch.dict(os.environ, {"TPU_VISIBLE_DEVICE_PATHS": " "}):
      self.assertLen(self._discover(), 1)

  def test_a_malformed_platform_index_selects_nothing(self):
    self._write_vfio_chip("0000:00:04.0", "21")
    with mock.patch.dict(os.environ, {"TPU_VISIBLE_DEVICES": "first"}):
      self.assertEmpty(self._discover())

  def test_allowlist_restricts_by_device_id(self):
    self._write_vfio_chip("0000:00:05.0", "7", device="0x0076")
    self._write_vfio_chip("0000:00:06.0", "8", device="0x00ff")
    with mock.patch.dict(os.environ, {"XLOOP_TPU_PCI_DEVICE_IDS": "0x0076"}):
      chips = self._discover()
    self.assertLen(chips, 1)
    self.assertEqual(chips[0].pci_slot, "0000:00:05")

  def test_allowlist_admits_a_new_device_id(self):
    self._write_vfio_chip("0000:00:05.0", "7", device="0x00ff")
    with mock.patch.dict(os.environ, {"XLOOP_TPU_PCI_DEVICE_IDS": "0x00FF"}):
      chips = self._discover()
    self.assertLen(chips, 1)

  def test_chip_count_override(self):
    with mock.patch.dict(os.environ, {"XLOOP_NUM_CHIPS": "4"}):
      chips = self._discover()
    self.assertLen(chips, 4)
    self.assertEqual([c.index for c in chips], [0, 1, 2, 3])
    self.assertTrue(all(c.device_path is None for c in chips))

  def test_malformed_chip_count_override_is_ignored(self):
    self._write_vfio_chip("0000:00:05.0", "7")
    with mock.patch.dict(os.environ, {"XLOOP_NUM_CHIPS": "many"}):
      chips = self._discover()
    self.assertLen(chips, 1)

  def test_missing_pci_root_yields_no_chips(self):
    self.assertEmpty(
        chip_broker.discover_chips("/nonexistent/pci", dev_root=str(self.dev))
    )

  def test_unreadable_pci_root_yields_no_chips(self):
    # Sandboxes can deny even a stat of the PCI tree.
    with mock.patch.object(
        pathlib.Path, "is_dir", side_effect=PermissionError("denied")
    ):
      self.assertEmpty(self._discover())

  def test_unlistable_pci_root_yields_no_chips(self):
    self._write_vfio_chip("0000:00:05.0", "7")
    with mock.patch.object(
        pathlib.Path, "iterdir", side_effect=PermissionError("denied")
    ):
      self.assertEmpty(self._discover())

  def test_an_unreadable_device_is_skipped(self):
    self._write_vfio_chip("0000:00:05.0", "7")
    self._write_vfio_chip("0000:00:06.0", "8")
    real_iommu_group = chip_broker._iommu_group

    def flaky_iommu_group(device_path):
      if device_path.name == "0000:00:06.0":
        raise PermissionError("denied")
      return real_iommu_group(device_path)

    with mock.patch.object(
        chip_broker, "_iommu_group", side_effect=flaky_iommu_group
    ):
      chips = self._discover()
    self.assertEqual([c.pci_slot for c in chips], ["0000:00:05"])

  def test_a_device_without_permissions_does_not_break_discovery(self):
    self._write_vfio_chip("0000:00:05.0", "7")
    self._write_vfio_chip("0000:00:06.0", "8")
    locked = self.root / "0000:00:06.0"
    locked.chmod(0)
    self.addCleanup(locked.chmod, 0o755)
    chips = self._discover()
    # The readable chip is always found; the locked one only if the test runs
    # with privileges that ignore the mode.
    self.assertIn("0000:00:05", [c.pci_slot for c in chips])

  def test_undecodable_sysfs_contents_are_skipped(self):
    self._write_vfio_chip("0000:00:05.0", "7")
    self._write_vfio_chip("0000:00:06.0", "8")
    (self.root / "0000:00:06.0" / "vendor").write_bytes(b"\xff\xfe\xfa")
    chips = self._discover()
    self.assertEqual([c.pci_slot for c in chips], ["0000:00:05"])


class ChipEnvTest(absltest.TestCase):

  def test_env_pins_process_to_chip_by_device_path(self):
    env = chip_broker.Chip(
        index=3, pci_slot="s", device_path="/dev/vfio/9"
    ).env()
    self.assertEqual(env["TPU_VISIBLE_DEVICE_PATHS"], "/dev/vfio/9")
    # Without this libtpu refuses a second load, which is what would otherwise
    # prevent concurrent workers entirely.
    self.assertEqual(env["ALLOW_MULTIPLE_LIBTPU_LOAD"], "1")

  def test_env_clears_every_selection_variable_it_does_not_set(self):
    # Omitting them is not enough. The worker inherits the caller's
    # environment, and libtpu applies the first of the three that is set, so
    # an exported TPU_VISIBLE_CHIPS would outrank a lease expressed as
    # TPU_VISIBLE_DEVICES and send the worker to the wrong chip.
    env = chip_broker.Chip(index=2, pci_slot="s", device_path=None).env()
    self.assertEqual(env["TPU_VISIBLE_DEVICES"], "2")
    self.assertIsNone(env["TPU_VISIBLE_DEVICE_PATHS"])
    self.assertIsNone(env["TPU_VISIBLE_CHIPS"])

  def test_env_clears_the_index_when_pinning_by_path(self):
    env = chip_broker.Chip(
        index=3, pci_slot="s", device_path="/dev/vfio/9"
    ).env()
    self.assertIsNone(env["TPU_VISIBLE_DEVICES"])
    self.assertIsNone(env["TPU_VISIBLE_CHIPS"])

  def test_every_known_selection_variable_is_accounted_for(self):
    env = chip_broker.Chip(
        index=0, pci_slot="s", device_path="/dev/vfio/1"
    ).env()
    for name in chip_broker.SELECTION_ENV_VARS:
      self.assertIn(name, env, f"{name} is neither set nor cleared")

  def test_env_does_not_inject_libtpu_flags(self):
    # libtpu aborts on an unknown flag and the flag set varies between builds,
    # so the engine must not add any of its own. Regression test for a flag
    # that killed every worker on a Cloud TPU v6e.
    env = chip_broker.Chip(
        index=0, pci_slot="s", device_path="/dev/vfio/1"
    ).env()
    self.assertNotIn("LIBTPU_INIT_ARGS", env)

  def test_env_falls_back_to_index_for_a_synthetic_chip(self):
    env = chip_broker.Chip(index=2, pci_slot="s", device_path=None).env()
    self.assertIsNone(env["TPU_VISIBLE_DEVICE_PATHS"])
    self.assertEqual(env["TPU_VISIBLE_DEVICES"], "2")

  def test_env_pins_an_accel_chip_by_its_node(self):
    env = chip_broker.Chip(
        index=5, pci_slot="s", device_path="/dev/accel1"
    ).env()
    self.assertEqual(env["TPU_VISIBLE_DEVICE_PATHS"], "/dev/accel1")
    self.assertIsNone(env["TPU_VISIBLE_DEVICES"])

  def test_env_overrides_a_platform_selection(self):
    # Discovery already intersected with the platform variable, so the lease
    # is inside the platform's set. The worker must get exactly the lease.
    with mock.patch.dict(os.environ, {"TPU_VISIBLE_CHIPS": "0,1"}):
      env = chip_broker.Chip(
          index=1, pci_slot="s", device_path="/dev/vfio/22"
      ).env()
    self.assertEqual(env["TPU_VISIBLE_DEVICE_PATHS"], "/dev/vfio/22")
    self.assertIsNone(env["TPU_VISIBLE_CHIPS"])


class LeaseKeyTest(absltest.TestCase):

  def test_is_derived_from_the_pci_slot(self):
    chip = chip_broker.Chip(index=0, pci_slot="0000:00:05", device_path=None)
    self.assertEqual(chip.lease_key, "0000_00_05")

  def test_ignores_the_index(self):
    first = chip_broker.Chip(index=0, pci_slot="0000:00:05", device_path=None)
    second = chip_broker.Chip(index=3, pci_slot="0000:00:05", device_path=None)
    self.assertEqual(first.lease_key, second.lease_key)


def _chips(n: int) -> list[chip_broker.Chip]:
  return [
      chip_broker.Chip(index=i, pci_slot=f"slot{i}", device_path=None)
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

  def _lock_path(self, index: int) -> pathlib.Path:
    return pathlib.Path(self.lease_dir) / f"chip_slot{index}.lock"

  def test_one_chip_seen_at_different_indices_is_leased_once(self):
    # Two processes that see different subsets of the host's chips give one
    # chip different indices. The lease must still be exclusive.
    first = chip_broker.ChipBroker(
        lease_dir=self.lease_dir,
        chips=[
            chip_broker.Chip(index=1, pci_slot="0000:00:05", device_path=None)
        ],
    )
    second = chip_broker.ChipBroker(
        lease_dir=self.lease_dir,
        chips=[
            chip_broker.Chip(index=0, pci_slot="0000:00:05", device_path=None)
        ],
    )
    with first.lease():
      with self.assertRaises(chip_broker.ChipBrokerError):
        second.acquire(timeout_sec=0.0)

  def test_is_chip_leased_reports_a_held_lease(self):
    broker = self._broker(n=2)
    chip, other = broker.chips
    with broker.lease() as claim:
      self.assertEqual(claim.chip, chip)
      self.assertTrue(chip_broker.is_chip_leased(self.lease_dir, chip))
      self.assertFalse(chip_broker.is_chip_leased(self.lease_dir, other))
    self.assertFalse(chip_broker.is_chip_leased(self.lease_dir, chip))

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
    stale = self._lock_path(0)
    stale.write_text(f"{_dead_pid()}\n")
    with broker.lease() as claim:
      self.assertEqual(claim.chip.index, 0)

  def test_lock_held_by_another_process_blocks_acquisition(self):
    broker = self._broker(n=1)
    path = self._lock_path(0)
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
    path = self._lock_path(0)
    path.write_text(f"{os.getpid()}\n")
    with broker.lease() as claim:
      self.assertEqual(claim.chip.index, 0)

  def test_corrupt_lock_file_is_claimable(self):
    broker = self._broker(n=1)
    stale = self._lock_path(0)
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
    path = self._lock_path(0)
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

  def test_lock_file_is_open_to_other_users_despite_the_umask(self):
    self.addCleanup(os.umask, os.umask(0o022))
    broker = self._broker(n=1)
    with broker.lease():
      pass
    self.assertEqual(self._lock_path(0).stat().st_mode & 0o777, 0o666)

  def test_lock_file_without_write_access_can_still_be_claimed(self):
    # Another user's lock file that kept a umask-masked mode. `flock` needs
    # only a read-only descriptor.
    broker = self._broker(n=1)
    self._lock_path(0).write_text("")
    real_open = os.open

    def open_without_write_access(path, flags, *args):
      if str(path).endswith(".lock") and flags & (os.O_WRONLY | os.O_RDWR):
        raise PermissionError(13, "Permission denied", path)
      return real_open(path, flags, *args)

    with mock.patch.object(os, "open", side_effect=open_without_write_access):
      with broker.lease() as claim:
        self.assertEqual(claim.chip.index, 0)
        self.assertTrue(chip_broker.is_chip_leased(self.lease_dir, claim.chip))

  def test_dead_ticket_that_cannot_be_removed_does_not_block(self):
    # The lease directory has the sticky bit, so another user's dead ticket
    # cannot be unlinked. It must not hold up the queue.
    broker = self._broker(n=1)
    stuck = pathlib.Path(self.lease_dir) / "ticket_000000000000"
    stuck.write_text(f"{_dead_pid()}\n")
    real_unlink = pathlib.Path.unlink

    def unlink(path, *args, **kwargs):
      if path == stuck:
        raise PermissionError(1, "Operation not permitted", str(path))
      return real_unlink(path, *args, **kwargs)

    with mock.patch.object(pathlib.Path, "unlink", autospec=True) as patched:
      patched.side_effect = unlink
      with broker.lease() as claim:
        self.assertEqual(claim.chip.index, 0)
    self.assertTrue(stuck.exists())

  def test_no_staging_file_is_left_behind(self):
    broker = self._broker(n=1)
    with broker.lease():
      pass
    self.assertEqual(os.listdir(self.lease_dir), ["chip_slot0.lock"])


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
    with mock.patch.dict(os.environ, {"XLOOP_FOREIGN_MANAGER_PATHS": ""}):
      self.assertIsNone(chip_broker.detect_foreign_manager())

  def test_absent_marker_is_not_a_manager(self):
    missing = os.path.join(self.create_tempdir().full_path, "runtime.sock")
    with mock.patch.dict(os.environ, {"XLOOP_FOREIGN_MANAGER_PATHS": missing}):
      self.assertIsNone(chip_broker.detect_foreign_manager())

  def test_foreign_runtime_socket_is_detected(self):
    tmpdir = self.create_tempdir()
    marker = tmpdir.create_file("runtime.sock").full_path
    missing = os.path.join(tmpdir.full_path, "absent.sock")
    with mock.patch.dict(
        os.environ,
        {"XLOOP_FOREIGN_MANAGER_PATHS": os.pathsep.join([missing, marker])},
    ):
      reason = chip_broker.detect_foreign_manager()
    self.assertIsNotNone(reason)
    self.assertIn(marker, reason)

  def test_broker_refuses_to_start_under_a_foreign_manager(self):
    with mock.patch.object(
        chip_broker, "detect_foreign_manager", return_value="daemon present"
    ):
      with self.assertRaises(chip_broker.ChipBrokerError):
        chip_broker.ChipBroker(chips=_chips(1))


if __name__ == "__main__":
  absltest.main()
