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

# Class directory of accelerators bound to the `accel` kernel driver rather
# than to vfio. Each `accelN` entry names its PCI function in a `pci_address`
# file, and its device node is `/dev/accelN`.
_ACCEL_CLASS_ROOT = "/sys/class/accel"

# Device nodes as seen by this process. Sysfs is not namespaced, so in a
# container it lists every chip on the host. Only the device nodes in this
# process's `/dev` show which of them the process can open.
_DEV_ROOT = "/dev"

_ACCEL_NAME_RE = re.compile(r"^accel\d+$")

# PCI device IDs of the TPU chips a guest sees, by generation. Other devices
# under the Google vendor ID (network adapters, security chips and the like)
# are common on hosts without accelerators, so the vendor alone is not enough.
_KNOWN_TPU_DEVICE_IDS = frozenset({
    "0x0027",  # v2, v3
    "0x0056",  # v4 lite
    "0x005e",  # v4
    "0x0062",  # v5p
    "0x0063",  # v5e
    "0x006f",  # v6e
    "0x0076",  # 7x
})

# Comma-separated PCI device IDs to accept instead of `_KNOWN_TPU_DEVICE_IDS`.
# Set this for a chip generation newer than the list above.
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
  """One physical accelerator that this process can open.

  Attributes:
    index: Logical index, `0..n-1`, in PCI slot order among the chips this
      process can open. In a container that holds a subset of the host's chips,
      this is not the host-wide position of the chip.
    pci_slot: PCI slot shared by the chip's functions, e.g. `0000:00:05`. This
      is the same in every process on the host, so leases are keyed on it.
    device_path: The device node that pins a worker to the chip:
      `/dev/vfio/<group>` for a chip bound to vfio, `/dev/accelN` for a chip
      bound to the `accel` driver. None only for the synthetic chips of
      `XLOOP_NUM_CHIPS`.
  """

  index: int
  pci_slot: str
  device_path: str | None

  @property
  def lease_key(self) -> str:
    """Name of the chip in the lease directory.

    Keyed on the PCI slot, not on `index`. Two processes on one host can see
    different subsets of the chips, for example when each has its own
    `TPU_VISIBLE_DEVICE_PATHS`. Then one chip can have a different index in
    each process, and index-keyed locks would let both lease it.

    Returns:
      A string that is safe to use in a file name.
    """
    return re.sub(r"[^0-9A-Za-z]", "_", self.pci_slot)

  def env(self) -> dict[str, str | None]:
    """Environment that pins a worker process to this chip.

    libtpu reads three device-selection variables and applies the first that
    is set, in the order `TPU_VISIBLE_DEVICE_PATHS`, `TPU_VISIBLE_CHIPS`,
    `TPU_VISIBLE_DEVICES`. A worker inherits the caller's environment, so
    setting only the variable the broker prefers is not enough: a user who has
    exported a higher-precedence one silently overrides the lease, and the
    worker opens a chip the broker did not grant. This therefore names all
    three, clearing the ones it does not want.

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
    if self.device_path:
      # Names the device rather than an index into an ordering the worker has
      # to agree on. libtpu accepts `/dev/vfio/<group>` and `/dev/accelN`
      # here.
      env["TPU_VISIBLE_DEVICE_PATHS"] = self.device_path
    else:
      # Only synthetic chips have no device node; the index is all there is.
      env["TPU_VISIBLE_DEVICES"] = str(self.index)
    return env


def _read_sysfs(path: pathlib.Path) -> str | None:
  try:
    return path.read_text().strip()
  except (OSError, ValueError):
    # ValueError covers undecodable contents.
    return None


def _iommu_group(device_path: pathlib.Path) -> str | None:
  """Returns the IOMMU group number for a PCI device, if it has one."""
  try:
    target = (device_path / "iommu_group").resolve()
  except (OSError, RuntimeError):
    # RuntimeError is what older Pythons raise on a symlink loop.
    return None
  name = target.name
  return name if name.isdigit() else None


def _device_id_allowlist() -> frozenset[str]:
  raw = os.environ.get(_DEVICE_ID_ALLOWLIST_ENV, "").strip()
  if not raw:
    return _KNOWN_TPU_DEVICE_IDS
  return frozenset(
      part.strip().lower() for part in raw.split(",") if part.strip()
  )


def _probe_device(
    device_path: pathlib.Path, allowlist: frozenset[str]
) -> tuple[str, str | None] | None:
  """Classifies one PCI device.

  Args:
    device_path: The device's directory under the PCI device tree.
    allowlist: Accepted PCI device IDs, lower case.

  Returns:
    `(slot, iommu_group)` if the device is an accelerator, else None. The
    group is None when the chip is not bound to vfio.
  """
  if _read_sysfs(device_path / "vendor") != _GOOGLE_PCI_VENDOR_ID:
    return None
  device_id = _read_sysfs(device_path / "device")
  if device_id is None or device_id.lower() not in allowlist:
    return None

  group = _iommu_group(device_path)
  match = _PCI_FUNCTION_RE.match(device_path.name)
  slot = match.group("slot") if match else device_path.name
  return slot, group


def _accel_nodes_by_slot(accel_class_root: str) -> dict[str, str]:
  """Maps a PCI slot to the `accelN` device of a chip on the `accel` driver.

  Chips that are not bound to vfio (for example v4 and v5p) have a device node
  `/dev/accelN`. The class entry `accelN` names its PCI function in a
  `pci_address` file. libtpu's device scanner reads that same file. If the file
  is absent, the standard `device` link to the PCI function is used.

  Args:
    accel_class_root: Root of the `accel` class tree.

  Returns:
    `accelN` names keyed by PCI slot. Empty if the tree is absent or
    unreadable.
  """
  root = pathlib.Path(accel_class_root)
  try:
    entries = sorted(root.iterdir()) if root.is_dir() else []
  except OSError as e:
    logger.warning("cannot enumerate %s: %s", accel_class_root, e)
    return {}
  nodes: dict[str, str] = {}
  for entry in entries:
    if not _ACCEL_NAME_RE.match(entry.name):
      continue
    address = _read_sysfs(entry / "pci_address")
    if not address:
      try:
        address = (entry / "device").resolve().name
      except (OSError, RuntimeError):
        continue
    match = _PCI_FUNCTION_RE.match(address.lower())
    if match:
      nodes.setdefault(match.group("slot"), entry.name)
  return nodes


def _node_is_open_to_us(dev_root: str, device_path: str) -> bool:
  """Whether this process can read `device_path` under `dev_root`.

  libtpu's device scanner skips a node that the process cannot read, so this
  uses the same test.

  Args:
    dev_root: Directory that stands for `/dev`.
    device_path: Absolute node path under `/dev`, e.g. `/dev/vfio/12`.

  Returns:
    True if the node exists and is readable.
  """
  relative = os.path.relpath(device_path, _DEV_ROOT)
  return os.access(os.path.join(dev_root, relative), os.R_OK)


def _platform_selection() -> tuple[str, list[str]] | None:
  """Returns the selection variable that libtpu would apply, if one is set.

  libtpu applies the first non-empty variable of `SELECTION_ENV_VARS`.

  Returns:
    `(name, entries)` for that variable, or None if none is set.
  """
  for name in SELECTION_ENV_VARS:
    raw = os.environ.get(name, "").strip()
    if raw:
      return name, [part.strip() for part in raw.split(",") if part.strip()]
  return None


def _apply_platform_selection(
    candidates: list[tuple[str, str]],
) -> list[tuple[str, str]]:
  """Keeps only the chips that a platform selection variable allows.

  A platform (for example a container runtime) can set a selection variable
  before XLoop starts. The broker must not hand out a chip outside that set.

  `TPU_VISIBLE_DEVICE_PATHS` lists device nodes, so it is compared by path. The
  index variables count the device nodes that libtpu's scanner finds: the nodes
  this process can read, in sorted path order. The same order is used here.

  Args:
    candidates: `(pci_slot, device_path)` for each chip this process can open.

  Returns:
    The candidates that the platform allows, in their original order.
  """
  selection = _platform_selection()
  if selection is None:
    return candidates
  name, entries = selection
  if name == "TPU_VISIBLE_DEVICE_PATHS":
    allowed_paths = {os.path.normpath(entry) for entry in entries}
    kept = [c for c in candidates if os.path.normpath(c[1]) in allowed_paths]
  else:
    by_scanner_order = sorted(candidates, key=lambda c: c[1])
    allowed_slots = set()
    for entry in entries:
      try:
        position = int(entry)
      except ValueError:
        logger.warning("ignoring %s entry %r: not an integer", name, entry)
        continue
      if 0 <= position < len(by_scanner_order):
        allowed_slots.add(by_scanner_order[position][0])
    kept = [c for c in candidates if c[0] in allowed_slots]
  logger.info(
      "%s=%s allows %d of %d chip(s)",
      name,
      ",".join(entries),
      len(kept),
      len(candidates),
  )
  return kept


def discover_chips(
    pci_root: str = _PCI_DEVICES_ROOT,
    dev_root: str = _DEV_ROOT,
    accel_class_root: str = _ACCEL_CLASS_ROOT,
) -> list[Chip]:
  """Enumerates the accelerators that this process can open.

  Discovery reads sysfs rather than asking JAX, because initializing JAX claims
  every visible chip -- which is precisely what the broker exists to prevent.

  A chip may expose several PCI functions; they share a slot and are collapsed
  into one entry. Slots are sorted, so processes that see the same chips agree
  on the indices.

  Sysfs is not namespaced: in a container that holds 4 of a host's 8 chips,
  the PCI tree still lists 8. A chip is therefore kept only if its device
  node (`/dev/vfio/<group>`, else `/dev/accelN`) is readable in this
  process's `/dev`. A chip with neither node is skipped, and so is a chip
  that a selection variable set by the platform does not allow (see
  `_apply_platform_selection`).

  Args:
    pci_root: Root of the PCI device tree. Overridable for testing.
    dev_root: Directory to look for device nodes in. Overridable for testing.
      The returned paths always start with `/dev`.
    accel_class_root: Root of the `accel` class tree. Overridable for testing.

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
          Chip(index=i, pci_slot=f"synthetic:{i}", device_path=None)
          for i in range(max(count, 0))
      ]

  # Sysfs can be absent, unreadable or only partly readable (containers,
  # sandboxes, hosts without a PCI bus). None of that is an error: it means
  # this host has no chips the broker can see, so discovery degrades to
  # skipping what it cannot read rather than failing the run.
  root = pathlib.Path(pci_root)
  try:
    if not root.is_dir():
      return []
    entries = sorted(root.iterdir())
  except OSError as e:
    logger.warning("cannot enumerate %s: %s", pci_root, e)
    return []

  allowlist = _device_id_allowlist()
  slots: dict[str, str | None] = {}
  for device_path in entries:
    try:
      found = _probe_device(device_path, allowlist)
    except OSError as e:
      logger.warning("skipping %s: %s", device_path, e)
      continue
    if found is None:
      continue
    slot, group = found
    # Keep the first group seen for a slot; sibling functions share it.
    if slot not in slots or slots[slot] is None:
      slots[slot] = group

  accel_nodes = _accel_nodes_by_slot(accel_class_root)
  candidates: list[tuple[str, str]] = []
  for slot, group in sorted(slots.items()):
    # A chip has an IOMMU group whenever the host has an IOMMU, also when it
    # is bound to the `accel` driver. So try each node and keep the first
    # that this process can read.
    options = []
    if group:
      options.append(f"/dev/vfio/{group}")
    if slot in accel_nodes:
      options.append(f"/dev/{accel_nodes[slot]}")
    usable = [p for p in options if _node_is_open_to_us(dev_root, p)]
    if not usable:
      logger.info(
          "skipping chip %s: no readable device node among %s",
          slot,
          options or "(no vfio group and no accel device)",
      )
      continue
    candidates.append((slot, usable[0]))

  return [
      Chip(index=i, pci_slot=slot, device_path=path)
      for i, (slot, path) in enumerate(_apply_platform_selection(candidates))
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


# Colon-separated paths (sockets, config directories) whose existence means a
# host runtime daemon already owns the accelerators, so per-chip leases from
# this process would collide with it. Empty by default: a stock TPU VM has no
# such daemon.
_FOREIGN_MANAGER_PATHS_ENV = "XLOOP_FOREIGN_MANAGER_PATHS"


def _foreign_manager_markers() -> list[str]:
  raw = os.environ.get(_FOREIGN_MANAGER_PATHS_ENV, "")
  return [part for part in raw.split(os.pathsep) if part.strip()]


def detect_foreign_manager() -> str | None:
  """Returns a description of another TPU manager on this host, if any.

  The paths to check come from `XLOOP_FOREIGN_MANAGER_PATHS`.

  Returns:
    A human-readable reason to refuse to broker, or None if the host has no
    conflicting runtime daemon.
  """
  for marker in _foreign_manager_markers():
    if os.path.exists(marker):
      return (
          f"{marker} exists, indicating a host runtime daemon already owns the"
          " accelerators on this machine; refusing to hand out per-chip"
          " leases."
      )
  return None


def _lock_file_name(chip: Chip) -> str:
  return f"chip_{chip.lease_key}.lock"


def _open_lock_file(path: str | os.PathLike[str]) -> int:
  """Opens a chip lock file for `flock`, and creates it if it is absent.

  The lease directory is shared by every user on the host, so the file can
  belong to another user. The mode given to `open` is masked by the creator's
  umask, so the creator sets 0o666 again with `fchmod`. A file that does not
  have that mode can still be locked: `flock` does not need write access, so
  this falls back to a read-only descriptor.

  Args:
    path: The lock file.

  Returns:
    An open descriptor. The caller closes it.

  Raises:
    OSError: If the file can be neither opened nor created.
  """
  try:
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o666)
  except PermissionError:
    return os.open(path, os.O_RDONLY)
  try:
    os.fchmod(fd, 0o666)
  except OSError:
    # Only the owner can change the mode. The read-only fallback above
    # covers a file whose owner did not.
    pass
  return fd


def is_chip_leased(lease_dir: str, chip: Chip) -> bool:
  """Whether `chip` is currently leased by some process.

  A lock file outlives the lease that created it -- see
  `ChipBroker._try_claim` for why it must -- so its presence proves nothing.
  The only way to ask is to try to take the lock and immediately drop it.

  Args:
    lease_dir: Directory holding the lock files.
    chip: Chip to test.

  Returns:
    True if another process holds the lease, False if it is free or its
    state cannot be read.
  """
  path = os.path.join(lease_dir, _lock_file_name(chip))
  if not os.path.exists(path):
    return False
  try:
    # Read-only is enough for `flock`, and works on another user's file.
    fd = os.open(path, os.O_RDONLY)
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

    * `chip_<slot>.lock` -- an `flock` is held on this file for the life of
      the lease. The kernel releases it if the holder dies. Named after the
      PCI slot, see `Chip.lease_key`.
    * `ticket_<n>` -- a queue position, holding the waiting pid.

  A waiter takes the lowest unused ticket number, then claims a chip only once
  no live older ticket is outstanding. Tickets record a pid. A ticket whose
  process is dead is skipped, and removed if the sticky directory permits.
  This is adequate for ordering; the chip locks themselves do not rely on
  pids at all.

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
    return self._lease_dir / _lock_file_name(chip)

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
      # Shared with the other users on the host. The directory carries the
      # sticky bit, so they cannot remove it.
      fd = _open_lock_file(path)
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
    # on it. This fails on a read-only descriptor, which is harmless.
    try:
      os.ftruncate(fd, 0)
      os.write(fd, f"{os.getpid()}\n".encode())
    except OSError:
      pass

    return ChipLease(chip=chip, lock_path=str(path), fd=fd)

  def _take_ticket(self) -> pathlib.Path:
    """Claims the lowest free queue position.

    The ticket is written in full under a private name first, and then linked
    to its queue name. `link` fails if the name exists, as `O_EXCL` does. So
    no process ever sees a ticket that does not hold a pid yet: such a ticket
    would look dead and be reaped.

    Returns:
      The ticket file.

    Raises:
      ChipBrokerError: If the ticket cannot be written.
    """
    try:
      fd, staging = tempfile.mkstemp(prefix=".staging_", dir=self._lease_dir)
      with os.fdopen(fd, "w") as f:
        f.write(f"{os.getpid()}\n")
      # Readable by other users, who check whether this waiter is alive.
      os.chmod(staging, 0o644)
    except OSError as e:
      raise ChipBrokerError(
          f"cannot write a ticket in {self._lease_dir}: {e}"
      ) from e
    try:
      number = 0
      while True:
        path = self._lease_dir / f"ticket_{number:012d}"
        try:
          os.link(staging, path)
        except FileExistsError:
          if self._reap(path):
            continue  # Retry the same number, now free.
          number += 1
          continue
        except OSError as e:
          raise ChipBrokerError(f"cannot create ticket {path}: {e}") from e
        return path
    finally:
      try:
        os.unlink(staging)
      except OSError:
        pass

  def _has_older_waiter(self, ticket: pathlib.Path) -> bool:
    """Whether a live waiter holds an earlier ticket.

    The test is whether the ticket's process is alive, not whether its file
    can be removed. The directory carries the sticky bit, so a dead ticket of
    another user stays in place. It must not block the queue.

    Args:
      ticket: This waiter's ticket.

    Returns:
      True if an earlier ticket belongs to a live process.
    """
    try:
      tickets = sorted(self._lease_dir.glob("ticket_*"))
    except OSError:
      return False
    for other in tickets:
      if other.name >= ticket.name:
        break
      pid = _read_pid(other)
      if pid is not None and _pid_alive(pid):
        return True
      self._reap(other)
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
