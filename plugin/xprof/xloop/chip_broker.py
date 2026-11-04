"""Cooperative accelerator tenancy for a single host.

On a TPU VM with multiple chips, concurrent benchmark workers need a way to
each claim a distinct chip without colliding on chip 0. This module provides
an exclusive per-chip lease with a FIFO wait queue and automatic reclamation
when a worker dies.

Why the engine has to do this at all
------------------------------------
`libtpu` normally refuses to initialize twice on one host, which prevents two
benchmark workers from running concurrently. The escape hatch is
`ALLOW_MULTIPLE_LIBTPU_LOAD=1` combined with `TPU_VISIBLE_DEVICE_PATHS` to pin
each process to one chip. That flag also **disables libtpu's own arbitration**,
so nothing below this module prevents two processes from claiming the same
chip. The broker is the only thing standing between concurrent users and a pair
of workers silently corrupting each other's measurements.

A worker killed with `SIGKILL` runs no `atexit` handler, so each lease is an
`flock` held on an open file descriptor: the kernel releases the lock the
instant the holder process exits, for any reason.

Isolation boundary
------------------
This is cooperative tenancy inside one trusted VM. The boundary is a vfio group
plus a process, not a security boundary. It is appropriate for a team sharing a
development VM. It is *not* sufficient for untrusted multi-tenant use.
"""

from __future__ import annotations

import contextlib
import dataclasses
import errno
import fcntl
import logging
import os
import pathlib
import random
import re
import tempfile
import time
from typing import Any, Iterator

# PCI vendor ID under which Google accelerators enumerate.
_GOOGLE_PCI_VENDOR_ID = "0x1ae0"

_PCI_DEVICES_ROOT = "/sys/bus/pci/devices"

# Comma-separated PCI device IDs to accept, overriding the default of "any
# device under the Google vendor ID that has an IOMMU group". Set this if the
# host enumerates non-accelerator Google devices.
_DEVICE_ID_ALLOWLIST_ENV = "XLOOP_TPU_PCI_DEVICE_IDS"

# Overrides discovery entirely with a chip count. Useful on hosts where sysfs
# is not readable, and in tests.
_CHIP_COUNT_ENV = "XLOOP_NUM_CHIPS"

_DEFAULT_ACQUIRE_TIMEOUT_SEC = 1800.0
_POLL_INTERVAL_SEC = 0.25

# The device-selection variables libtpu consults, in the order it applies
# them: the first one set wins. The broker must control all of them, not just
# the one it prefers, or an inherited value silently overrides a lease. See
# `Chip.env`.
SELECTION_ENV_VARS = (
    "TPU_VISIBLE_DEVICE_PATHS",
    "TPU_VISIBLE_CHIPS",
    "TPU_VISIBLE_DEVICES",
)

# Matches a PCI address like `0000:00:05.0`; group 1 is the slot, shared by the
# functions belonging to one physical chip.
_PCI_FUNCTION_RE = re.compile(r"^(?P<slot>[0-9a-f:.]+?)\.(?P<function>\d+)$")

logger = logging.getLogger(__name__)


class ChipBrokerError(RuntimeError):  # pylint: disable=g-bad-exception-name
  """Raised when a chip cannot be leased."""


@dataclasses.dataclass(frozen=True)
class Chip:
  """One physical accelerator on this host.

  Attributes:
    index: Logical index, `0..n-1`, stable for a given host because slots are
      sorted before assignment.
    pci_slot: PCI slot shared by the chip's functions, e.g. `0000:00:05`.
    vfio_path: `/dev/vfio/<group>` for the chip, if it is bound to vfio.
  """

  index: int
  pci_slot: str
  vfio_path: str | None

  def env(self) -> dict[str, str | None]:
    """Environment that pins a worker process to this chip.

    libtpu reads three device-selection variables and applies the first that
    is set, in the order `TPU_VISIBLE_DEVICE_PATHS`, `TPU_VISIBLE_CHIPS`,
    `TPU_VISIBLE_DEVICES` (learning/brain/tfrc/runtime/common_lib.cc). A
    worker inherits the caller's environment, so setting only the variable the
    broker prefers is not enough: a user who has exported a higher-precedence
    one silently overrides the lease, and the worker opens a chip the broker
    did not grant. This therefore names all three, clearing the ones it does
    not want.

    Deliberately does not set `LIBTPU_INIT_ARGS`: libtpu treats an unknown
    flag as fatal, and the flag set differs between builds, so injecting one
    here would make the engine's viability depend on the libtpu that happens
    to be installed. Callers who need flags can export the variable
    themselves; the worker inherits it.

    Returns:
      Variables to apply to the worker's environment. A None value means
      remove the variable if present.
    """
    env: dict[str, str | None] = {
        # Permit a second libtpu in this host. Necessary for concurrency, and
        # the reason this broker must exist.
        "ALLOW_MULTIPLE_LIBTPU_LOAD": "1",
    }
    env.update({name: None for name in SELECTION_ENV_VARS})
    if self.vfio_path:
      # Preferred when available: it names the device rather than an index
      # into an ordering the worker has to agree on.
      env["TPU_VISIBLE_DEVICE_PATHS"] = self.vfio_path
    else:
      env["TPU_VISIBLE_DEVICES"] = str(self.index)
    return env


def _read_sysfs(path: pathlib.Path) -> str | None:
  try:
    return path.read_text().strip()
  except OSError:
    return None


def _iommu_group(device_path: pathlib.Path) -> str | None:
  """Returns the IOMMU group number for a PCI device, if it has one."""
  try:
    target = (device_path / "iommu_group").resolve()
  except OSError:
    return None
  name = target.name
  return name if name.isdigit() else None


def _device_id_allowlist() -> frozenset[str] | None:
  raw = os.environ.get(_DEVICE_ID_ALLOWLIST_ENV, "").strip()
  if not raw:
    return None
  return frozenset(part.strip().lower() for part in raw.split(",") if part)


def discover_chips(
    pci_root: str = _PCI_DEVICES_ROOT,
) -> list[Chip]:
  """Enumerates the accelerators attached to this host.

  Discovery reads sysfs rather than asking JAX, because initializing JAX claims
  every visible chip -- which is precisely what the broker exists to prevent.

  A chip may expose several PCI functions; they share a slot and are collapsed
  into one entry. Slots are sorted so indices are stable across processes.

  Args:
    pci_root: Root of the PCI device tree. Overridable for testing.

  Returns:
    The chips, ordered by PCI slot. Empty if none are found, which is the
    normal result on a CPU-only host.
  """
  override = os.environ.get(_CHIP_COUNT_ENV, "").strip()
  if override:
    try:
      count = int(override)
    except ValueError:
      logger.warning(
          "ignoring %s=%r: not an integer", _CHIP_COUNT_ENV, override
      )
    else:
      return [
          Chip(index=i, pci_slot=f"synthetic:{i}", vfio_path=None)
          for i in range(max(count, 0))
      ]

  root = pathlib.Path(pci_root)
  if not root.is_dir():
    return []

  allowlist = _device_id_allowlist()
  slots: dict[str, str | None] = {}
  try:
    entries = sorted(root.iterdir())
  except OSError as e:
    logger.warning("cannot enumerate %s: %s", pci_root, e)
    return []

  for device_path in entries:
    if _read_sysfs(device_path / "vendor") != _GOOGLE_PCI_VENDOR_ID:
      continue
    if allowlist is not None:
      device_id = _read_sysfs(device_path / "device")
      if device_id is None or device_id.lower() not in allowlist:
        continue

    group = _iommu_group(device_path)
    if allowlist is None and group is None:
      # Without an explicit allowlist, require an IOMMU group. Accelerators
      # passed through to the guest always have one; incidental Google-vendor
      # devices generally do not.
      continue

    match = _PCI_FUNCTION_RE.match(device_path.name)
    slot = match.group("slot") if match else device_path.name
    # Keep the first group seen for a slot; sibling functions share it.
    if slot not in slots or slots[slot] is None:
      slots[slot] = group

  return [
      Chip(
          index=i,
          pci_slot=slot,
          vfio_path=f"/dev/vfio/{group}" if group else None,
      )
      for i, (slot, group) in enumerate(sorted(slots.items()))
  ]


def default_lease_dir() -> str:
  """Returns the default directory holding lock and ticket files.

  One directory per host, not per user. A chip is a host-global resource: if
  each user got their own lease namespace, two users on one VM would each see
  every chip as free, lease the same one, and collide on the hardware -- which
  is exactly what the broker exists to prevent. Shared tenancy has to mean
  shared across everyone on the host.

  Placed under the system temp directory so the state disappears on reboot,
  which is also what makes a stale lease from a crashed run self-healing.

  Returns:
    An absolute directory path. Not created; `_ensure_lease_dir` creates it
    with the permissions that make it usable by more than one user.
  """
  return os.path.join(tempfile.gettempdir(), "xloop_chips")


def _ensure_lease_dir(path: pathlib.Path) -> None:
  """Creates the lease directory so that several users can share it.

  Mode 1777, like `/tmp` itself: anyone may add a lock, and the sticky bit
  stops one user removing another's. Created with an explicit chmod because
  `mkdir`'s mode is masked by the process umask, which is typically 022 and
  would leave the directory writable only by its creator -- turning the first
  user to run into the only user who can.

  Args:
    path: Directory to create.

  Raises:
    ChipBrokerError: If the directory cannot be created.
  """
  try:
    path.mkdir(parents=True, exist_ok=True)
  except OSError as e:
    raise ChipBrokerError(f"cannot create lease directory {path}: {e}") from e
  try:
    path.chmod(0o1777)
  except OSError:
    # Typically because another user owns it, in which case it already has
    # the mode we want. Locking will fail loudly later if it does not.
    pass


# Host-level TPU runtime sockets/config directories. If any of these exist, a
# system daemon already owns the accelerators on this host and per-chip
# leases from this process would collide with it.
_FOREIGN_MANAGER_MARKERS = (
    "/var/google/services/xos/xos.sock",
    "/var/google/services/tpum/xos_server.sock",
    "/dev/shm/xos/xos.sock",
    "/etc/xosconf.d",
)


def detect_foreign_manager() -> str | None:
  """Returns a description of another TPU manager on this host, if any.

  Returns:
    A human-readable reason to refuse to broker, or None if the host has no
    conflicting runtime daemon.
  """
  for marker in _FOREIGN_MANAGER_MARKERS:
    if os.path.exists(marker):
      return (
          f"{marker} exists, indicating a host runtime daemon already owns the"
          " accelerators on this machine; refusing to hand out per-chip"
          " leases."
      )
  return None


def is_chip_leased(lease_dir: str, index: int) -> bool:
  """Whether chip `index` is currently leased by some process.

  A lock file outlives the lease that created it -- see
  `ChipBroker._try_claim` for why it must -- so its presence proves nothing.
  The only way to ask is to try to take the lock and immediately drop it.

  Args:
    lease_dir: Directory holding the lock files.
    index: Chip index to test.

  Returns:
    True if another process holds the lease, False if it is free or its
    state cannot be read.
  """
  path = os.path.join(lease_dir, f"chip_{index}.lock")
  if not os.path.exists(path):
    return False
  try:
    fd = os.open(path, os.O_RDWR)
  except OSError:
    # Present but unopenable: assume someone else's and leave it alone.
    return True
  try:
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
  except OSError:
    return True
  else:
    # Taking the lock proved it was free; drop it again at once so this
    # query stays read-only.
    fcntl.flock(fd, fcntl.LOCK_UN)
    return False
  finally:
    os.close(fd)


def _pid_alive(pid: int) -> bool:
  """Whether a process exists, without requiring permission to signal it."""
  if pid <= 0:
    return False
  try:
    os.kill(pid, 0)
  except ProcessLookupError:
    return False
  except PermissionError:
    # It exists; we simply may not signal it.
    return True
  except OSError as e:
    return e.errno != errno.ESRCH
  return True


def _read_pid(path: pathlib.Path) -> int | None:
  try:
    raw = path.read_text().strip()
  except OSError:
    return None
  try:
    return int(raw.split()[0])
  except (ValueError, IndexError):
    return None


@dataclasses.dataclass(frozen=True)
class ChipLease:
  """An exclusive claim on one chip.

  Attributes:
    chip: The leased chip.
    lock_path: Lock file backing the lease.
    fd: Open descriptor holding the `flock`. The lock lives on this descriptor,
      not on the file's contents, so the kernel drops it if the holder exits
      without releasing. -1 once released.
  """

  chip: Chip
  lock_path: str
  fd: int = -1

  def env(self) -> dict[str, str | None]:
    """Environment pinning a worker to the leased chip."""
    return self.chip.env()


class ChipBroker:
  """Hands out exclusive, FIFO-ordered leases on this host's chips.

  Safe across processes and across users. State is one directory of files,
  shared by everyone on the host because the chips are:

    * `chip_<i>.lock` -- an `flock` is held on this file for the life of the
      lease. The kernel releases it if the holder dies.
    * `ticket_<n>` -- a queue position, holding the waiting pid.

  A waiter takes the lowest unused ticket number, then claims a chip only once
  no live older ticket is outstanding. Tickets record a pid and are reaped on
  contact, which is adequate for ordering; the chip locks themselves do not
  rely on pids at all.

  This is advisory. It coordinates processes that go through it, and the
  kernel's rule that only one process may open a vfio group is the only hard
  guarantee. A process that opens a chip without asking is not prevented.
  """

  def __init__(
      self,
      num_chips: int | None = None,
      lease_dir: str | None = None,
      chips: list[Chip] | None = None,
  ) -> None:
    """Initializes the broker.

    Args:
      num_chips: Cap on how many chips to hand out. `None` means all discovered
        chips. Values above the discovered count are clamped.
      lease_dir: Directory for lock and ticket files. Defaults to a per-user
        directory under the system temp directory.
      chips: Explicit chip list, bypassing discovery. For tests.

    Raises:
      ChipBrokerError: If no chips are available to broker, or another TPU
        manager owns this host's devices.
    """
    foreign = detect_foreign_manager()
    if foreign:
      raise ChipBrokerError(foreign)

    discovered = chips if chips is not None else discover_chips()
    if not discovered:
      raise ChipBrokerError(
          "no accelerators found on this host. Set"
          f" {_CHIP_COUNT_ENV} to broker a synthetic chip count, or run with"
          " a single worker and no broker."
      )
    if num_chips is not None:
      if num_chips < 1:
        raise ChipBrokerError(f"num_chips must be >= 1, got {num_chips}")
      if num_chips > len(discovered):
        logger.warning(
            "requested %d chips but only %d are present; using %d",
            num_chips,
            len(discovered),
            len(discovered),
        )
      discovered = discovered[:num_chips]

    self._chips = discovered
    self._lease_dir = pathlib.Path(lease_dir or default_lease_dir())
    _ensure_lease_dir(self._lease_dir)

  @property
  def chips(self) -> list[Chip]:
    """The chips this broker manages."""
    return list(self._chips)

  @property
  def capacity(self) -> int:
    """How many leases can be held at once."""
    return len(self._chips)

  @property
  def lease_dir(self) -> str:
    """Directory holding lock and ticket files."""
    return str(self._lease_dir)

  def _lock_path(self, chip: Chip) -> pathlib.Path:
    return self._lease_dir / f"chip_{chip.index}.lock"

  def _reap(self, path: pathlib.Path) -> bool:
    """Removes `path` if the pid inside it is dead.

    Args:
      path: Lock or ticket file.

    Returns:
      True if the file was removed or had already gone.
    """
    pid = _read_pid(path)
    if pid is not None and _pid_alive(pid):
      return False
    try:
      path.unlink()
      logger.info("reaped stale %s (pid %s)", path.name, pid)
    except FileNotFoundError:
      pass
    except OSError as e:
      logger.warning("could not reap %s: %s", path, e)
      return False
    return True

  def _try_claim(self, chip: Chip) -> ChipLease | None:
    """Attempts to take the lock for `chip` without blocking.

    The lock is an `flock` held on an open descriptor rather than the mere
    existence of a file. The kernel drops it when the holder exits, however
    it exits, so a crashed worker frees its chip with no reaping and no
    liveness guess. Checking a recorded pid instead would be wrong twice
    over: a recycled pid keeps a dead lease alive forever, and the check
    races the writer.

    The file is never unlinked. Another process may already have it open and
    be blocked on the lock; unlinking would detach them from the name and let
    a third process take a lock nobody else can see.

    Args:
      chip: Chip to claim.

    Returns:
      The lease, or None if another process holds it.

    Raises:
      ChipBrokerError: If the lock file cannot be created or locked for a
        reason other than contention.
    """
    path = self._lock_path(chip)
    try:
      # 0o666 so a second user on the host can lock the same file; the
      # directory carries the sticky bit so they cannot remove it.
      fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o666)
    except OSError as e:
      raise ChipBrokerError(f"cannot create lock {path}: {e}") from e

    try:
      fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as e:
      os.close(fd)
      if e.errno in (errno.EACCES, errno.EAGAIN):
        return None
      raise ChipBrokerError(f"cannot lock {path}: {e}") from e

    # Recorded for operators reading the directory; the lock does not depend
    # on it.
    try:
      os.ftruncate(fd, 0)
      os.write(fd, f"{os.getpid()}\n".encode())
    except OSError:
      pass

    return ChipLease(chip=chip, lock_path=str(path), fd=fd)

  def _take_ticket(self) -> pathlib.Path:
    """Claims the lowest free queue position."""
    number = 0
    while True:
      path = self._lease_dir / f"ticket_{number:012d}"
      try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
      except FileExistsError:
        if self._reap(path):
          continue  # Retry the same number, now free.
        number += 1
        continue
      except OSError as e:
        raise ChipBrokerError(f"cannot create ticket {path}: {e}") from e
      with os.fdopen(fd, "w") as f:
        f.write(f"{os.getpid()}\n")
      return path

  def _has_older_waiter(self, ticket: pathlib.Path) -> bool:
    """Whether a live waiter holds an earlier ticket."""
    try:
      tickets = sorted(self._lease_dir.glob("ticket_*"))
    except OSError:
      return False
    for other in tickets:
      if other.name >= ticket.name:
        break
      if not self._reap(other):
        return True
    return False

  def _release_ticket(self, ticket: pathlib.Path) -> None:
    try:
      ticket.unlink()
    except FileNotFoundError:
      pass
    except OSError as e:
      logger.warning("could not release ticket %s: %s", ticket, e)

  def acquire(
      self,
      timeout_sec: float = _DEFAULT_ACQUIRE_TIMEOUT_SEC,
      on_wait: Any = None,
  ) -> ChipLease:
    """Blocks until a chip is free, respecting arrival order.

    Args:
      timeout_sec: How long to wait before giving up.
      on_wait: Optional callable invoked as `on_wait(position, capacity)` each
        time the wait polls, so a caller can report queue position. Callers
        should surface this: a silent block is indistinguishable from a hang.

    Returns:
      The lease. Release it with `release()`, or use `lease()` instead.

    Raises:
      ChipBrokerError: If no chip became available within `timeout_sec`.
    """
    ticket = self._take_ticket()
    deadline = time.monotonic() + timeout_sec
    try:
      while True:
        if not self._has_older_waiter(ticket):
          for chip in self._chips:
            claim = self._try_claim(chip)
            if claim is not None:
              return claim
        if time.monotonic() >= deadline:
          raise ChipBrokerError(
              f"no chip available after {timeout_sec:g}s"
              f" ({self.capacity} chip(s) managed, all busy)"
          )
        if on_wait is not None:
          on_wait(self._queue_position(ticket), self.capacity)
        # Jitter so that waiters released together do not collide repeatedly.
        time.sleep(_POLL_INTERVAL_SEC * (1.0 + random.random()))
    finally:
      self._release_ticket(ticket)

  def _queue_position(self, ticket: pathlib.Path) -> int:
    """Zero-based position of `ticket` among live waiters."""
    try:
      tickets = sorted(self._lease_dir.glob("ticket_*"))
    except OSError:
      return 0
    return sum(1 for other in tickets if other.name < ticket.name)

  def release(self, lease: ChipLease) -> None:
    """Releases a lease.

    Closing the descriptor drops the `flock`. The file stays: see
    `_try_claim` for why unlinking it would be unsafe.

    Safe to call twice.

    Args:
      lease: The lease to release.
    """
    if lease.fd < 0:
      return
    try:
      fcntl.flock(lease.fd, fcntl.LOCK_UN)
    except OSError as e:
      logger.warning("could not unlock %s: %s", lease.lock_path, e)
    try:
      os.close(lease.fd)
    except OSError:
      pass
    # The dataclass is frozen, so the caller's copy still holds the old fd.
    # Closing twice is the only hazard and it is guarded by the check above
    # in the common path; `lease()` drops its reference immediately after.
    object.__setattr__(lease, "fd", -1)

  @contextlib.contextmanager
  def lease(
      self,
      timeout_sec: float = _DEFAULT_ACQUIRE_TIMEOUT_SEC,
      on_wait: Any = None,
  ) -> Iterator[ChipLease]:
    """Acquires a chip for the duration of the block.

    Args:
      timeout_sec: Acquisition deadline.
      on_wait: Optional `on_wait(position, capacity)` progress callback.

    Yields:
      The lease.

    Raises:
      ChipBrokerError: If no chip became available in time.
    """
    claim = self.acquire(timeout_sec=timeout_sec, on_wait=on_wait)
    try:
      yield claim
    finally:
      self.release(claim)

  def reap_all(self) -> int:
    """Removes every queue ticket left behind by a dead process.

    Called on engine startup so a previous crashed run does not permanently
    consume capacity.

    Chip locks are deliberately not touched. They are `flock`s, which the
    kernel already released when the holder died, so there is nothing to
    reclaim; and unlinking one would break the association for any process
    currently blocked on it. Only tickets, which are plain pid files, can go
    stale in a way that needs cleaning.

    Returns:
      Number of tickets reaped.
    """
    reaped = 0
    try:
      entries = list(self._lease_dir.iterdir())
    except OSError:
      return 0
    for path in entries:
      if not path.name.startswith("ticket_"):
        continue
      if self._reap(path):
        reaped += 1
    return reaped
