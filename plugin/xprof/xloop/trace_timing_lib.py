# Copyright 2025 The XProf Authors. All Rights Reserved.
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
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import os
from typing import Any

from absl import logging

# Mirrors the C++ XPlane schema (xplane_schema.cc).
_DEVICE_PLANE_PREFIX = "/device:TPU"
_MODULE_LINE_NAME = "XLA Modules"

# XLA names a module after the jitted callable that produced it, suffixed with
# a program id: `jit_dummy_op(3223885384320152505)`. The separator interposed
# by `eval_jax_lib.dummy_op` is therefore identified by this prefix.
_SEPARATOR_MODULE_PREFIX = "jit_dummy_op"

_TRACE_SUFFIX = ".xplane.pb"

_NANOSECONDS_PER_SECOND = 1e9


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


def _module_events(planes: Iterable[Any]) -> list[tuple[str, float, float]]:
  """Returns `(name, start_ns, duration_ns)` for each XLA module executed.

  Args:
    planes: The planes of a parsed trace, as `profile_data.ProfilePlane`s.

  Returns:
    Events from the device plane holding the most module executions, in
    execution order. A worker is pinned to one chip so there is normally one
    such plane; picking the busiest keeps the reading well-defined if a future
    caller profiles several.
  """
  best: list[tuple[str, float, float]] = []
  for plane in planes:
    if not plane.name.startswith(_DEVICE_PLANE_PREFIX):
      continue
    for line in plane.lines:
      if line.name != _MODULE_LINE_NAME:
        continue
      # `start_ns` already folds the line's `timestamp_ns` into the event's
      # `offset_ps`, so events on one line can be ordered directly.
      events = [
          (event.name, event.start_ns, event.duration_ns)
          for event in line.events
      ]
      if len(events) > len(best):
        best = events
  best.sort(key=lambda item: item[1])
  return best


def read_device_execution_times_sec(
    trace_dir: str, expected_samples: int = 0
) -> list[float]:
  """Reads per-sample device execution times out of a role's trace.

  Args:
    trace_dir: The role's profile directory.
    expected_samples: Number of timed samples the harness ran. When positive and
      the trace holds a different number, the result is reconciled or rejected
      as described below.

  Returns:
    One duration in seconds per sample, in execution order. Empty if the trace
    is missing, unreadable, holds no device plane, or cannot be reconciled
    with `expected_samples`. Callers treat an empty result as "fall back to
    wall clock" rather than as an error: a trace is a diagnostic artifact, and
    a run that produced valid outputs should still report something.
  """
  path = find_trace_file(trace_dir)
  if path is None:
    logging.warning("No %s under %s.", _TRACE_SUFFIX, trace_dir)
    return []

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
    return []

  try:
    with open(path, "rb") as handle:
      data = handle.read()
    with profile_data.ProfileData.from_serialized_xspace(data) as profile:
      events = _module_events(profile.planes)
  except (OSError, ValueError) as error:
    logging.warning("Could not read trace %s: %s", path, error)
    return []

  if not events:
    logging.warning(
        "No %r events on a device plane in %s.", _MODULE_LINE_NAME, path
    )
    return []

  samples = [
      duration_ns / _NANOSECONDS_PER_SECOND
      for name, _, duration_ns in events
      if not name.startswith(_SEPARATOR_MODULE_PREFIX)
  ]

  if expected_samples and len(samples) != expected_samples:
    if len(samples) < expected_samples:
      # A short trace cannot be aligned with the samples that produced it: the
      # profiler may have dropped events anywhere, so position i in the trace
      # is not necessarily sample i. Reject rather than silently mismatch.
      logging.warning(
          "Trace %s holds %d module executions but %d samples ran; falling"
          " back to wall-clock timing.",
          path,
          len(samples),
          expected_samples,
      )
      return []
    # Extra events are attributable: profiling starts before the first sample
    # and stops after the last, so any surplus is at the edges. Keeping the
    # last `expected_samples` drops residual warmup.
    logging.info(
        "Trace %s holds %d module executions for %d samples; keeping the last"
        " %d.",
        path,
        len(samples),
        expected_samples,
        expected_samples,
    )
    samples = samples[-expected_samples:]

  return samples


def annotate_results(results: dict[str, Any], roles: Sequence[str]) -> None:
  """Adds `{role}_trace_matched_execution_times_sec` to a results payload.

  Roles whose trace cannot be read are left absent, and the summarizer falls
  back to wall clock for them.

  Args:
    results: The dictionary from `eval_jax_lib.profile_funcs`, modified in
      place. Each role's trace directory is read from its `{role}_trace_dir`
      entry, so this stays consistent with wherever the harness chose to write.
    roles: Roles that were profiled.
  """
  expected = int(results.get("num_executions", 0) or 0)
  for role in roles:
    trace_dir = results.get(f"{role}_trace_dir")
    if not isinstance(trace_dir, str):
      continue
    times = read_device_execution_times_sec(trace_dir, expected)
    if times:
      results[f"{role}_trace_matched_execution_times_sec"] = times
