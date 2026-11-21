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
"""Per-sample device execution times, read back out of an emitted trace.

Wall-clock timing around a jitted call cannot measure a microbenchmark. The
timed region spans a Python call, JAX dispatch, the driver, a host-device
round trip and a synchronisation, and on a v6e those cost ~160us in total
against ~2.5us for a 1024-cube matmul. Fitting a chain of K such matmuls gives
`160 + 2.5K` microseconds: the floor is roughly 64x the operation under test,
so a real 128x improvement shows up as 1.7x.

The device plane of the trace the harness already writes carries the number we
actually want. The TPU timestamps each XLA module execution itself, so a
duration taken from there contains no host dispatch at all. This module reads
those durations back and hands them to the summarizer.

Matching is positional. `eval_jax_lib.profile` interposes
`eval_jax_lib.dummy_op` between samples, so a run of N samples appears on the
`XLA Modules` line as N module executions separated by N-1 dummies. Dropping
the dummies leaves exactly the samples, in order. That is deterministic, and
unlike correlating against the host-side `xloop_sample_N` annotations it does
not depend on the host and device clocks being comparable -- on GCE those
annotations land in a different plane and no common time base is guaranteed.

Only the benchmarked computation counts. A sample can run more than one XLA
module: for example, JAX can run a small `jit__multi_slice` module to move a
sharded input before the computation itself. The harness records the name of
the benchmarked module, and only that module's executions are matched to
samples. If the name is not known, the module that runs on the most devices
is used. Ties go to the module with the most executions, then to the larger
total device time.

A chip can expose more than one device, each with its own plane (for example
one per TensorCore). Every device plane that holds executions of the module is
read. A computation that spans several devices is done only when its last
device is done, so the time reported for a sample is the slowest device's time
for that sample. Each device's own times are kept as well.

A device that cannot be matched to the samples is left out, and the reason is
recorded as a note in the results, so a reader can see that the reported time
does not include that device.

When several implementations are compared, `eval_jax_lib.profile_funcs` runs
the samples of each one in blocks, and each block writes a trace of its own.
Each block is aligned on its own with the number of samples it ran, and the
blocks are joined in order. If one block cannot be aligned, the role gets no
trace times at all: a missing block would move every later sample out of its
position, and sample i of one role must stay paired with sample i of the
other.
"""

from __future__ import annotations

import collections
from collections.abc import Iterable, Sequence
import dataclasses
import os
from typing import Any

from absl import logging

# Mirrors the C++ XPlane schema (xplane_schema.cc).
_DEVICE_PLANE_PREFIX = "/device:TPU:"
_MODULE_LINE_NAME = "XLA Modules"

# XLA names a module after the jitted callable that produced it, suffixed with
# a program id: `jit_dummy_op(3223885384320152505)`. The separator interposed
# by `eval_jax_lib.dummy_op` is therefore identified by this prefix.
_SEPARATOR_MODULE_PREFIX = "jit_dummy_op"

_TRACE_SUFFIX = ".xplane.pb"

_NANOSECONDS_PER_SECOND = 1e9


def module_base_name(event_name: str) -> str:
  """Returns an XLA module event name without its program id.

  Args:
    event_name: A name from the `XLA Modules` line, such as
      `jit__matmul(11694009845663015448)`.

  Returns:
    The module name, such as `jit__matmul`. A name without a trailing
    `(<digits>)` is returned unchanged.
  """
  head, separator, tail = event_name.rpartition("(")
  if separator and tail.endswith(")") and tail[:-1].isdigit():
    return head
  return event_name


def find_trace_file(trace_dir: str) -> str | None:
  """Locates the trace `eval_jax_lib.trace_context` wrote under `trace_dir`.

  Args:
    trace_dir: A role's profile directory, as returned by
      `xloop_lib.profile_dir`. The layout beneath it is
      `plugins/profile/<run>/<host>.xplane.pb`.

  Returns:
    The path of the newest trace file, or None if there is none. Newest rather
    than first because a directory reused across runs accumulates one run
    subdirectory per invocation, and the current run is the one wanted.
  """
  if not trace_dir or not os.path.isdir(trace_dir):
    return None
  candidates = []
  for root, _, files in os.walk(trace_dir):
    for name in files:
      if name.endswith(_TRACE_SUFFIX):
        candidates.append(os.path.join(root, name))
  if not candidates:
    return None
  return max(candidates, key=os.path.getmtime)


def _module_events_by_device(
    planes: Iterable[Any],
) -> dict[str, list[tuple[str, float, float]]]:
  """Returns `(name, start_ns, duration_ns)` per XLA module, per device.

  Args:
    planes: The planes of a parsed trace, as `profile_data.ProfilePlane`s.

  Returns:
    Events in execution order, keyed by device id: the part of the plane name
    after `/device:TPU:`. Only devices that hold module executions are
    present. A chip can have more than one device (for example one per
    TensorCore), so a worker pinned to one chip can still produce several.
  """
  by_device: dict[str, list[tuple[str, float, float]]] = {}
  for plane in planes:
    if not plane.name.startswith(_DEVICE_PLANE_PREFIX):
      continue
    device = plane.name[len(_DEVICE_PLANE_PREFIX) :]
    for line in plane.lines:
      if line.name != _MODULE_LINE_NAME:
        continue
      # `start_ns` already folds the line's `timestamp_ns` into the event's
      # `offset_ps`, so events on one line can be ordered directly.
      events = [
          (event.name, event.start_ns, event.duration_ns)
          for event in line.events
      ]
      if len(events) > len(by_device.get(device, [])):
        by_device[device] = events
  for events in by_device.values():
    events.sort(key=lambda item: item[1])
  return by_device


@dataclasses.dataclass(frozen=True)
class DeviceTiming:
  """Per-sample device execution times for one role.

  Attributes:
    samples_sec: The reported time of each sample, in seconds: the slowest
      device's time for that sample.
    per_device_sec: Each device's time for each sample, keyed by device id.
      Holds only the devices whose samples were used.
    timing_device: The device that set the reported time in the most samples.
      Ties go to the device that sorts first.
    module_name: The XLA module whose executions were timed, without its program
      id.
    notes: Why a device or a module was not used, one sentence each. Empty when
      the trace matched the samples without any exclusion.
  """

  samples_sec: list[float]
  per_device_sec: dict[str, list[float]]
  timing_device: str
  module_name: str = ""
  notes: tuple[str, ...] = ()


def _device_sort_key(device: str) -> tuple[int, int | str]:
  return (0, int(device)) if device.isdigit() else (1, device)


def _combine_devices(
    samples_by_device: dict[str, list[float]],
    expected_samples: int,
    path: str,
    module_name: str,
    notes: list[str],
) -> DeviceTiming | None:
  """Aligns each device's samples and takes the slowest per sample.

  A multi-device computation is done only when its last device is done, so the
  time of a sample is the largest of its device times.

  Each device is aligned on its own, by the rule in
  `read_device_timing`. A device that holds fewer samples than the target
  cannot be aligned and is left out, with a warning and a note. The target is
  `expected_samples` if it is positive. Only when the harness did not say how
  many samples ran is it the largest count of any device.

  Args:
    samples_by_device: Durations in seconds, in execution order, keyed by device
      id.
    expected_samples: Number of timed samples the harness ran, or 0.
    path: The trace file, for log messages.
    module_name: The module whose executions these are.
    notes: Receives one sentence per device that is left out.

  Returns:
    The combined timing, or None if no device could be aligned.
  """
  target = expected_samples or max(len(s) for s in samples_by_device.values())
  aligned: dict[str, list[float]] = {}
  for device in sorted(samples_by_device, key=_device_sort_key):
    samples = samples_by_device[device]
    if len(samples) < target:
      # A short trace cannot be aligned with the samples that produced it: the
      # profiler may have dropped events anywhere, so position i in the trace
      # is not necessarily sample i. Reject rather than silently mismatch.
      note = (
          f"device {device} holds {len(samples)} executions of {module_name}"
          f" but {target} samples ran; it is left out"
      )
      logging.warning("Trace %s: %s.", path, note)
      notes.append(note)
      continue
    if len(samples) > target:
      # Extra events are attributable: profiling starts before the first
      # sample and stops after the last, so any surplus is at the edges.
      # Keeping the last `target` drops residual warmup.
      logging.info(
          "Trace %s: device %s holds %d module executions for %d samples;"
          " keeping the last %d.",
          path,
          device,
          len(samples),
          target,
          target,
      )
      samples = samples[-target:]
    aligned[device] = samples

  if not aligned:
    return None

  return _slowest_device_timing(aligned, module_name, notes)


def _slowest_device_timing(
    aligned: dict[str, list[float]], module_name: str, notes: list[str]
) -> DeviceTiming:
  """Takes the slowest device per sample.

  Args:
    aligned: Each device's time for each sample, keyed by device id, in the
      order of `_device_sort_key`. Not empty, and every list has the same
      length.
    module_name: The module whose executions these are.
    notes: The notes to record.

  Returns:
    The combined timing.
  """
  devices = list(aligned)
  samples_sec: list[float] = []
  slowest: list[str] = []
  for row in zip(*(aligned[device] for device in devices)):
    # `max` keeps the first of equal values, so ties go to the device that
    # sorts first.
    position = max(range(len(row)), key=row.__getitem__)
    samples_sec.append(row[position])
    slowest.append(devices[position])
  return DeviceTiming(
      samples_sec=samples_sec,
      per_device_sec=aligned,
      timing_device=max(devices, key=slowest.count),
      module_name=module_name,
      notes=tuple(notes),
  )


def _choose_module(
    events_by_device: dict[str, list[tuple[str, float, float]]],
    module_name: str,
    notes: list[str],
) -> str | None:
  """Chooses the module whose executions are the samples.

  Args:
    events_by_device: Module executions without separators, keyed by device.
    module_name: The benchmarked module as the harness recorded it, or "".
    notes: Receives a sentence if `module_name` is given but not in the trace.

  Returns:
    `module_name` if any device ran it. Otherwise the module that ran on the
    most devices. Ties go to the module with the most executions, then to the
    larger total device time, then to the name that sorts first. None if there
    are no executions.
  """
  device_count: collections.Counter[str] = collections.Counter()
  executions: collections.Counter[str] = collections.Counter()
  total_ns: collections.defaultdict[str, float] = collections.defaultdict(float)
  for events in events_by_device.values():
    names = set()
    for name, _, duration_ns in events:
      base = module_base_name(name)
      names.add(base)
      executions[base] += 1
      total_ns[base] += duration_ns
    device_count.update(names)
  if not device_count:
    return None
  if module_name in device_count:
    return module_name
  chosen = max(
      sorted(device_count),
      key=lambda name: (device_count[name], executions[name], total_ns[name]),
  )
  if module_name:
    notes.append(
        f"module {module_name} is not in the trace; timing {chosen} instead"
    )
  return chosen


def read_device_timing(
    trace_dir: str, expected_samples: int = 0, module_name: str = ""
) -> DeviceTiming | None:
  """Reads per-sample device execution times out of a role's trace.

  Every `/device:TPU:<id>` plane that holds executions of the benchmarked
  module is read. With one device, its times are the result. With several,
  each sample's time is the slowest device's time for that sample (see
  `_combine_devices`).

  Args:
    trace_dir: The role's profile directory.
    expected_samples: Number of timed samples the harness ran. When positive, a
      device that holds more executions keeps its last `expected_samples`, and a
      device that holds fewer is left out. Pass it whenever it is known.
    module_name: The benchmarked XLA module, without its program id, as
      `eval_jax_lib.profile` records it. Executions of other modules, such as
      input moves that JAX runs before a sharded computation, are not samples.
      When empty, the module is chosen as `_choose_module` describes.

  Returns:
    The timing, or None if the trace is missing, unreadable, holds no device
    plane, or no device can be reconciled with `expected_samples`. Callers
    treat None as "fall back to wall clock" rather than as an error: a trace
    is a diagnostic artifact, and a run that produced valid outputs should
    still report something.
  """
  timing, _ = _read_device_timing(trace_dir, expected_samples, module_name)
  return timing


def _read_device_timing(
    trace_dir: str, expected_samples: int, module_name: str
) -> tuple[DeviceTiming | None, list[str]]:
  """Does the work of `read_device_timing`, and also returns its notes.

  Args:
    trace_dir: See `read_device_timing`.
    expected_samples: See `read_device_timing`.
    module_name: See `read_device_timing`.

  Returns:
    The timing or None, and one sentence for each device or module that was
    not used, or for why there is no timing.
  """
  notes: list[str] = []
  path = find_trace_file(trace_dir)
  if path is None:
    logging.warning("No %s under %s.", _TRACE_SUFFIX, trace_dir)
    notes.append(f"no {_TRACE_SUFFIX} file was written")
    return None, notes

  try:
    # Imported at call time so that a missing or unloadable native extension
    # degrades to wall-clock timing instead of failing the worker at import.
    from xprof import profile_data  # pylint: disable=g-import-not-at-top
  except (ImportError, OSError) as error:
    logging.warning(
        "Cannot load the XSpace reader (%s); falling back to wall-clock"
        " timing.",
        error,
    )
    notes.append(f"the trace reader cannot be loaded: {error}")
    return None, notes

  try:
    with open(path, "rb") as handle:
      data = handle.read()
    with profile_data.ProfileData.from_serialized_xspace(data) as profile:
      events_by_device = _module_events_by_device(profile.planes)
  except (OSError, ValueError) as error:
    logging.warning("Could not read trace %s: %s", path, error)
    notes.append(f"the trace cannot be read: {error}")
    return None, notes

  events_by_device = {
      device: [
          e for e in events if not e[0].startswith(_SEPARATOR_MODULE_PREFIX)
      ]
      for device, events in events_by_device.items()
  }
  chosen = _choose_module(events_by_device, module_name, notes)
  if chosen is None:
    logging.warning(
        "No %r events on a device plane in %s.", _MODULE_LINE_NAME, path
    )
    notes.append(f"no device plane holds {_MODULE_LINE_NAME!r} events")
    return None, notes

  samples_by_device = {
      device: [
          duration_ns / _NANOSECONDS_PER_SECOND
          for name, _, duration_ns in events
          if module_base_name(name) == chosen
      ]
      for device, events in events_by_device.items()
  }
  # A device that never ran the module took no part in the computation, so it
  # is not a device that was left out.
  samples_by_device = {d: s for d, s in samples_by_device.items() if s}

  timing = _combine_devices(
      samples_by_device, expected_samples, path, chosen, notes
  )
  if timing is None:
    logging.warning(
        "Trace %s cannot be aligned with %d samples; falling back to"
        " wall-clock timing.",
        path,
        expected_samples,
    )
  return timing, notes


def _read_block_timing(
    trace_dirs: Sequence[str],
    block_sample_counts: Sequence[int],
    module_name: str,
) -> tuple[DeviceTiming | None, list[str]]:
  """Reads the traces of the blocks of one role and joins them in order.

  Each block is aligned on its own, with the number of samples it ran. The
  joined timing keeps only the devices that every block used, and each
  sample's time is the slowest of those devices.

  Args:
    trace_dirs: The trace directory of each block, in block order.
    block_sample_counts: The number of samples of each block, in block order.
    module_name: See `read_device_timing`.

  Returns:
    The timing, or None if a block cannot be aligned, if the blocks timed
    different modules, or if no device was used by every block. Also one
    sentence for each device, module or block that was not used, with the
    block named when there is more than one block.
  """
  if len(trace_dirs) == 1:
    return _read_device_timing(
        trace_dirs[0], block_sample_counts[0], module_name
    )

  notes: list[str] = []
  timings: list[DeviceTiming] = []
  for block, (trace_dir, count) in enumerate(
      zip(trace_dirs, block_sample_counts)
  ):
    timing, block_notes = _read_device_timing(trace_dir, count, module_name)
    notes.extend(f"block {block}: {note}" for note in block_notes)
    if timing is None:
      notes.append(
          f"block {block} cannot be aligned with its {count} samples; no"
          " block is used, so that samples stay paired by position"
      )
      return None, notes
    timings.append(timing)

  chosen = timings[0].module_name
  for block, timing in enumerate(timings):
    if timing.module_name != chosen:
      notes.append(
          f"block {block} timed {timing.module_name} but block 0 timed"
          f" {chosen}; no block is used"
      )
      return None, notes

  joined = _join_shared_devices(timings, notes)
  if not joined:
    notes.append("no device is used in every block")
    return None, notes
  return _slowest_device_timing(joined, chosen, notes), notes


def _join_shared_devices(
    timings: Sequence[DeviceTiming], notes: list[str]
) -> dict[str, list[float]]:
  """Joins in order the per-sample times of the devices that every block used.

  Args:
    timings: The timing of each block, in block order.
    notes: Receives one sentence for each device that a block used but another
      block did not.

  Returns:
    Each shared device's times over all blocks, keyed by device id in the order
    of `_device_sort_key`. Empty if no device is shared by every block.
  """
  shared = set(timings[0].per_device_sec)
  for timing in timings[1:]:
    shared &= set(timing.per_device_sec)
  for block, timing in enumerate(timings):
    for device in sorted(
        set(timing.per_device_sec) - shared, key=_device_sort_key
    ):
      notes.append(
          f"device {device} is used in block {block} but not in every block;"
          " it is left out"
      )
  joined: dict[str, list[float]] = {}
  for device in sorted(shared, key=_device_sort_key):
    joined[device] = []
    for timing in timings:
      joined[device].extend(timing.per_device_sec[device])
  return joined


def _block_traces(
    results: dict[str, Any], role: str, expected_samples: int
) -> tuple[list[str], list[int], list[str]]:
  """Returns the trace directory and sample count of each block of a role.

  Args:
    results: See `annotate_results`.
    role: The role.
    expected_samples: See `annotate_results`.

  Returns:
    The trace directories, the sample counts and the notes. The lists are
    empty if the role has no usable trace directory. A payload without block
    entries, as written before blocks existed, has one block that holds
    `expected_samples` samples.
  """
  trace_dirs = results.get(f"{role}_trace_dirs")
  counts = results.get(f"{role}_block_sample_counts")
  if not (
      isinstance(trace_dirs, list)
      and trace_dirs
      and all(isinstance(path, str) for path in trace_dirs)
      and isinstance(counts, list)
      and len(counts) == len(trace_dirs)
      and all(isinstance(count, int) for count in counts)
  ):
    trace_dir = results.get(f"{role}_trace_dir")
    if not isinstance(trace_dir, str):
      return [], [], []
    return [trace_dir], [expected_samples], []
  if expected_samples and sum(counts) != expected_samples:
    note = (
        f"the blocks hold {sum(counts)} samples but {expected_samples} ran;"
        " no block is used"
    )
    return [], [], [note]
  return list(trace_dirs), list(counts), []


def read_device_execution_times_sec(
    trace_dir: str, expected_samples: int = 0, module_name: str = ""
) -> list[float]:
  """Returns `read_device_timing(...).samples_sec`, or [] if there is none.

  Args:
    trace_dir: The role's profile directory.
    expected_samples: Number of timed samples the harness ran, or 0.
    module_name: The benchmarked module, or "".

  Returns:
    One duration in seconds per sample, in execution order.
  """
  timing = read_device_timing(trace_dir, expected_samples, module_name)
  return timing.samples_sec if timing else []


def annotate_results(
    results: dict[str, Any],
    roles: Sequence[str],
    expected_samples: int | None = None,
) -> None:
  """Adds trace-derived timing to a results payload.

  For each role whose trace can be read, this adds:

    * `{role}_trace_matched_execution_times_sec`: the reported time of each
      sample.
    * `{role}_timing_device`: the device that set those times most often.
    * `{role}_timing_module`: the XLA module whose executions were timed.
    * `{role}_device_execution_times_sec`: each device's times, keyed by
      device id. Only when more than one device was used.

  For each role with a trace directory, this also adds
  `{role}_timing_notes` when a device, module or block was not used, or when
  the trace gave no timing at all. Roles whose trace cannot be read have no
  times, and the summarizer falls back to wall clock for them.

  A role that ran in blocks has one trace per block. Each block is aligned
  with its own sample count, and the blocks are joined in order (see
  `_read_block_timing`). A block that cannot be aligned leaves the role
  without trace times.

  Args:
    results: The dictionary from `eval_jax_lib.profile_funcs`, modified in
      place. Each role's traces are read from its `{role}_trace_dirs` and
      `{role}_block_sample_counts` entries, or from `{role}_trace_dir` when
      those are absent, so this stays consistent with wherever the harness chose
      to write. Its module is read from `{role}_module_name`.
    roles: Roles that were profiled.
    expected_samples: Number of timed samples the harness ran. If None, it is
      read from `results["num_executions"]`. Pass it explicitly: without it, a
      device that holds extra executions sets the count, and the devices that
      hold the right count are left out.
  """
  if expected_samples is None:
    expected_samples = int(results.get("num_executions", 0) or 0)
  for role in roles:
    trace_dirs, counts, notes = _block_traces(results, role, expected_samples)
    timing = None
    if trace_dirs:
      module_name = results.get(f"{role}_module_name")
      timing, notes = _read_block_timing(
          trace_dirs,
          counts,
          module_name if isinstance(module_name, str) else "",
      )
    if notes:
      results[f"{role}_timing_notes"] = notes
    if timing is None:
      continue
    results[f"{role}_trace_matched_execution_times_sec"] = timing.samples_sec
    results[f"{role}_timing_device"] = timing.timing_device
    results[f"{role}_timing_module"] = timing.module_name
    if len(timing.per_device_sec) > 1:
      results[f"{role}_device_execution_times_sec"] = timing.per_device_sec
