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
"""Tests for trace_timing_lib.

The fixtures here reproduce the structure observed on a real Cloud TPU v6e-8:
a `/device:TPU:0` plane carrying an `XLA Modules` line, on which samples of the
benchmarked computation alternate with `jit_dummy_op` separators.
"""

import os
from typing import Any

from absl.testing import absltest

from tensorflow.tsl.profiler.protobuf import xplane_pb2  # pylint: disable=g-direct-tensorflow-import
from xprof.xloop import trace_timing_lib


def _add_device_plane(
    space: xplane_pb2.XSpace,
    module_durations_ps,
    *,
    plane_name: str = "/device:TPU:0",
    line_name: str = "XLA Modules",
    module_name: str = "jit__softmax(123)",
    separators: bool = True,
    line_timestamp_ns: int = 1_000,
) -> None:
  """Adds one device plane in the layout `trace_context` produces.

  Args:
    space: The trace to add the plane to.
    module_durations_ps: Duration of each benchmarked sample, in picoseconds.
    plane_name: Name of the plane to place the events on.
    line_name: Name of the line to place the events on.
    module_name: Event name used for the benchmarked samples.
    separators: Whether to interpose `jit_dummy_op` events between samples, as
      the harness does.
    line_timestamp_ns: Line base timestamp, which offsets are relative to.
  """
  plane = space.planes.add()
  plane.name = plane_name
  plane.event_metadata[1].id = 1
  plane.event_metadata[1].name = module_name
  plane.event_metadata[2].id = 2
  plane.event_metadata[2].name = "jit_dummy_op(999)"

  line = plane.lines.add()
  line.name = line_name
  line.timestamp_ns = line_timestamp_ns

  offset = 0
  for index, duration in enumerate(module_durations_ps):
    if separators and index:
      event = line.events.add()
      event.metadata_id = 2
      event.offset_ps = offset
      event.duration_ps = 28_000_000  # ~28us, as measured on v6e.
      offset += 28_000_000
    event = line.events.add()
    event.metadata_id = 1
    event.offset_ps = offset
    event.duration_ps = duration
    offset += duration


def _save(directory: str, space: xplane_pb2.XSpace) -> str:
  """Writes `space` where `trace_context` would, and returns the path."""
  run_dir = os.path.join(directory, "plugins", "profile", "run0")
  os.makedirs(run_dir, exist_ok=True)
  path = os.path.join(run_dir, "host.xplane.pb")
  with open(path, "wb") as handle:
    handle.write(space.SerializeToString())
  return path


def _write_trace(directory: str, module_durations_ps, **kwargs) -> str:
  """Writes a synthetic one-device trace; see `_add_device_plane`."""
  space = xplane_pb2.XSpace()
  _add_device_plane(space, module_durations_ps, **kwargs)
  return _save(directory, space)


def _write_multi_device_trace(directory: str, durations_by_device) -> str:
  """Writes a trace with one plane per device id in `durations_by_device`."""
  space = xplane_pb2.XSpace()
  for device, durations in durations_by_device.items():
    _add_device_plane(space, durations, plane_name=f"/device:TPU:{device}")
  return _save(directory, space)


def _add_plane_events(
    space: xplane_pb2.XSpace,
    device: str,
    events,
) -> None:
  """Adds one device plane that holds the given module executions in order.

  Args:
    space: The trace to add the plane to.
    device: The device id, the part of the plane name after `/device:TPU:`.
    events: `(module event name, duration in picoseconds)` pairs, in execution
      order.
  """
  plane = space.planes.add()
  plane.name = f"/device:TPU:{device}"
  ids: dict[str, int] = {}
  line = plane.lines.add()
  line.name = "XLA Modules"
  line.timestamp_ns = 1_000
  offset = 0
  for name, duration in events:
    if name not in ids:
      ids[name] = len(ids) + 1
      plane.event_metadata[ids[name]].id = ids[name]
      plane.event_metadata[ids[name]].name = name
    event = line.events.add()
    event.metadata_id = ids[name]
    event.offset_ps = offset
    event.duration_ps = duration
    offset += duration + 1_000_000


_MATMUL = "jit__matmul(11694009845663015448)"
_SLICE = "jit__multi_slice(1516952095172282025)"
_DUMMY = "jit_dummy_op(4086281167137480546)"


def _sharded_sample_events(matmul_durations_ps, with_slice: bool):
  """Events of one device for a sharded run, as recorded on a TPU7x host.

  Args:
    matmul_durations_ps: Duration of the benchmarked module in each sample.
    with_slice: Whether each sample also runs a `jit__multi_slice` input move
      before the benchmarked module, as device 0 does on that host.

  Returns:
    `(name, duration)` pairs with a separator between samples.
  """
  events = []
  for index, duration in enumerate(matmul_durations_ps):
    if index:
      events.append((_DUMMY, 26_000_000))
    if with_slice:
      events.append((_SLICE, 44_000_000))
    events.append((_MATMUL, duration))
  return events


class ModuleBaseNameTest(absltest.TestCase):

  def test_removes_the_program_id(self):
    self.assertEqual(trace_timing_lib.module_base_name(_MATMUL), "jit__matmul")

  def test_keeps_a_name_without_a_program_id(self):
    self.assertEqual(
        trace_timing_lib.module_base_name("jit_f(abc)"), "jit_f(abc)"
    )
    self.assertEqual(trace_timing_lib.module_base_name("jit_f"), "jit_f")


class FindTraceFileTest(absltest.TestCase):

  def test_returns_none_for_missing_directory(self):
    self.assertIsNone(
        trace_timing_lib.find_trace_file("/nonexistent/profile/dir")
    )

  def test_returns_none_when_no_trace_present(self):
    self.assertIsNone(
        trace_timing_lib.find_trace_file(self.create_tempdir().full_path)
    )

  def test_finds_trace_nested_under_run_directory(self):
    directory = self.create_tempdir().full_path
    path = _write_trace(directory, [1_000_000])
    self.assertEqual(trace_timing_lib.find_trace_file(directory), path)

  def test_prefers_newest_when_directory_reused_across_runs(self):
    directory = self.create_tempdir().full_path
    old = os.path.join(directory, "plugins", "profile", "run_old")
    os.makedirs(old)
    stale = os.path.join(old, "host.xplane.pb")
    with open(stale, "wb") as handle:
      handle.write(b"")
    os.utime(stale, (1, 1))
    fresh = _write_trace(directory, [1_000_000])
    self.assertEqual(trace_timing_lib.find_trace_file(directory), fresh)


class ReadDeviceExecutionTimesTest(absltest.TestCase):

  def test_reads_durations_in_seconds(self):
    directory = self.create_tempdir().full_path
    _write_trace(directory, [13_010_000, 12_990_000, 13_000_000])
    self.assertSequenceAlmostEqual(
        trace_timing_lib.read_device_execution_times_sec(directory),
        [13.01e-6, 12.99e-6, 13.0e-6],
    )

  def test_excludes_separator_modules(self):
    directory = self.create_tempdir().full_path
    _write_trace(directory, [1_000_000] * 4)
    # Four samples with three interposed separators: only the samples count.
    self.assertLen(
        trace_timing_lib.read_device_execution_times_sec(directory), 4
    )

  def test_orders_by_start_not_by_position(self):
    directory = self.create_tempdir().full_path
    path = _write_trace(directory, [1_000_000])
    space = xplane_pb2.XSpace()
    with open(path, "rb") as handle:
      space.ParseFromString(handle.read())
    line = space.planes[0].lines[0]
    # Append an event that started earlier than the one already present.
    event = line.events.add()
    event.metadata_id = 1
    event.offset_ps = -500_000
    event.duration_ps = 7_000_000
    with open(path, "wb") as handle:
      handle.write(space.SerializeToString())
    self.assertSequenceAlmostEqual(
        trace_timing_lib.read_device_execution_times_sec(directory),
        [7e-6, 1e-6],
    )

  def test_returns_empty_when_no_device_plane(self):
    directory = self.create_tempdir().full_path
    _write_trace(directory, [1_000_000], plane_name="/host:CPU")
    self.assertEmpty(
        trace_timing_lib.read_device_execution_times_sec(directory)
    )

  def test_returns_empty_when_no_module_line(self):
    directory = self.create_tempdir().full_path
    _write_trace(directory, [1_000_000], line_name="XLA Ops")
    self.assertEmpty(
        trace_timing_lib.read_device_execution_times_sec(directory)
    )

  def test_returns_empty_for_unreadable_trace(self):
    directory = self.create_tempdir().full_path
    run_dir = os.path.join(directory, "plugins", "profile", "run0")
    os.makedirs(run_dir)
    with open(os.path.join(run_dir, "host.xplane.pb"), "wb") as handle:
      handle.write(b"not a protobuf at all, not even close")
    self.assertEmpty(
        trace_timing_lib.read_device_execution_times_sec(directory)
    )

  def test_rejects_short_trace_rather_than_mismatching_samples(self):
    # A dropped event would silently shift every subsequent sample, so a short
    # trace must not be used at all.
    directory = self.create_tempdir().full_path
    _write_trace(directory, [1_000_000] * 3)
    self.assertEmpty(
        trace_timing_lib.read_device_execution_times_sec(
            directory, expected_samples=5
        )
    )

  def test_keeps_trailing_samples_when_trace_has_surplus(self):
    directory = self.create_tempdir().full_path
    _write_trace(directory, [9_000_000, 1_000_000, 2_000_000, 3_000_000])
    self.assertSequenceAlmostEqual(
        trace_timing_lib.read_device_execution_times_sec(
            directory, expected_samples=3
        ),
        [1e-6, 2e-6, 3e-6],
    )

  def test_exact_match_is_returned_whole(self):
    directory = self.create_tempdir().full_path
    _write_trace(directory, [1_000_000, 2_000_000])
    self.assertLen(
        trace_timing_lib.read_device_execution_times_sec(
            directory, expected_samples=2
        ),
        2,
    )

  def test_a_device_with_fewer_samples_is_left_out(self):
    # A plane with fewer module executions than the others cannot be aligned
    # by position, so it does not take part.
    directory = self.create_tempdir().full_path
    path = _write_trace(directory, [1_000_000, 2_000_000])
    space = xplane_pb2.XSpace()
    with open(path, "rb") as handle:
      space.ParseFromString(handle.read())
    _add_device_plane(
        space, [99_000_000], plane_name="/device:TPU:1", module_name="jit_o(7)"
    )
    _save(directory, space)
    self.assertSequenceAlmostEqual(
        trace_timing_lib.read_device_execution_times_sec(directory),
        [1e-6, 2e-6],
    )

  def test_ignores_non_core_planes(self):
    directory = self.create_tempdir().full_path
    space = xplane_pb2.XSpace()
    _add_device_plane(space, [1_000_000], plane_name="#Chip 0")
    _save(directory, space)
    self.assertIsNone(trace_timing_lib.read_device_timing(directory))


class MultiDeviceTimingTest(absltest.TestCase):

  def test_single_device_names_its_device(self):
    directory = self.create_tempdir().full_path
    _write_trace(directory, [1_000_000, 2_000_000])
    timing = trace_timing_lib.read_device_timing(directory, 2)
    self.assertIsNotNone(timing)
    self.assertEqual(timing.timing_device, "0")
    self.assertEqual(list(timing.per_device_sec), ["0"])
    self.assertSequenceAlmostEqual(timing.samples_sec, [1e-6, 2e-6])

  def test_reports_the_slowest_device_per_sample(self):
    directory = self.create_tempdir().full_path
    _write_multi_device_trace(
        directory,
        {
            "0": [1_000_000, 5_000_000, 3_000_000],
            "1": [2_000_000, 4_000_000, 6_000_000],
        },
    )
    timing = trace_timing_lib.read_device_timing(directory, 3)
    self.assertIsNotNone(timing)
    self.assertSequenceAlmostEqual(timing.samples_sec, [2e-6, 5e-6, 6e-6])
    # Device 1 is slowest in two samples of three.
    self.assertEqual(timing.timing_device, "1")
    self.assertSequenceAlmostEqual(
        timing.per_device_sec["0"], [1e-6, 5e-6, 3e-6]
    )
    self.assertSequenceAlmostEqual(
        timing.per_device_sec["1"], [2e-6, 4e-6, 6e-6]
    )

  def test_equal_devices_go_to_the_first_device(self):
    directory = self.create_tempdir().full_path
    _write_multi_device_trace(
        directory, {"1": [1_000_000, 1_000_000], "0": [1_000_000, 1_000_000]}
    )
    timing = trace_timing_lib.read_device_timing(directory, 2)
    self.assertIsNotNone(timing)
    self.assertEqual(timing.timing_device, "0")

  def test_devices_sort_by_number(self):
    directory = self.create_tempdir().full_path
    _write_multi_device_trace(directory, {"10": [1_000_000], "2": [1_000_000]})
    timing = trace_timing_lib.read_device_timing(directory, 1)
    self.assertIsNotNone(timing)
    self.assertEqual(list(timing.per_device_sec), ["2", "10"])
    self.assertEqual(timing.timing_device, "2")

  def test_a_device_with_missing_samples_is_left_out(self):
    directory = self.create_tempdir().full_path
    _write_multi_device_trace(
        directory,
        {"0": [1_000_000, 2_000_000, 3_000_000], "1": [9_000_000, 9_000_000]},
    )
    timing = trace_timing_lib.read_device_timing(directory, 3)
    self.assertIsNotNone(timing)
    self.assertEqual(list(timing.per_device_sec), ["0"])
    self.assertEqual(timing.timing_device, "0")
    self.assertSequenceAlmostEqual(timing.samples_sec, [1e-6, 2e-6, 3e-6])

  def test_surplus_is_trimmed_per_device(self):
    directory = self.create_tempdir().full_path
    _write_multi_device_trace(
        directory,
        {"0": [7_000_000, 1_000_000, 1_000_000], "1": [2_000_000, 2_000_000]},
    )
    timing = trace_timing_lib.read_device_timing(directory, 2)
    self.assertIsNotNone(timing)
    self.assertSequenceAlmostEqual(timing.per_device_sec["0"], [1e-6, 1e-6])
    self.assertSequenceAlmostEqual(timing.samples_sec, [2e-6, 2e-6])

  def test_no_device_can_be_aligned(self):
    directory = self.create_tempdir().full_path
    _write_multi_device_trace(directory, {"0": [1_000_000], "1": [1_000_000]})
    self.assertIsNone(trace_timing_lib.read_device_timing(directory, 3))


class OtherModulesTest(absltest.TestCase):
  """Only the benchmarked module counts as a sample.

  Reproduces a sharded matmul on a TPU7x host with 8 devices: device 0 also
  runs a `jit__multi_slice` input move in every sample, with the same count as
  the matmul. Counting every module made device 0 hold twice the samples, so
  the other 7 devices were left out and the first sample read the input move.
  """

  def _write(self, num_devices: int = 2, num_samples: int = 3) -> str:
    directory = self.create_tempdir().full_path
    space = xplane_pb2.XSpace()
    for device in range(num_devices):
      durations = [
          (1_000 + 100 * device + sample) * 1_000_000
          for sample in range(num_samples)
      ]
      _add_plane_events(
          space,
          str(device),
          _sharded_sample_events(durations, with_slice=device == 0),
      )
    _save(directory, space)
    return directory

  def test_named_module_uses_every_device(self):
    directory = self._write()
    timing = trace_timing_lib.read_device_timing(directory, 3, "jit__matmul")
    self.assertIsNotNone(timing)
    self.assertEqual(list(timing.per_device_sec), ["0", "1"])
    self.assertSequenceAlmostEqual(
        timing.samples_sec, [1.1e-3, 1.101e-3, 1.102e-3]
    )
    self.assertEqual(timing.timing_device, "1")
    self.assertEqual(timing.module_name, "jit__matmul")
    self.assertEmpty(timing.notes)

  def test_unnamed_module_is_the_one_on_the_most_devices(self):
    directory = self._write()
    for expected in (3, 0):
      timing = trace_timing_lib.read_device_timing(directory, expected)
      self.assertIsNotNone(timing)
      self.assertEqual(timing.module_name, "jit__matmul")
      self.assertEqual(list(timing.per_device_sec), ["0", "1"])

  def test_a_named_module_that_is_absent_falls_back_with_a_note(self):
    directory = self._write()
    timing = trace_timing_lib.read_device_timing(directory, 3, "jit_other")
    self.assertIsNotNone(timing)
    self.assertEqual(timing.module_name, "jit__matmul")
    self.assertLen(timing.notes, 1)
    self.assertIn("jit_other", timing.notes[0])

  def test_eight_devices_with_equal_counts_all_take_part(self):
    directory = self._write(num_devices=8, num_samples=10)
    timing = trace_timing_lib.read_device_timing(directory, 10, "jit__matmul")
    self.assertIsNotNone(timing)
    self.assertEqual(list(timing.per_device_sec), [str(d) for d in range(8)])
    # Device 7 is slowest in every sample.
    self.assertEqual(timing.timing_device, "7")
    self.assertAlmostEqual(timing.samples_sec[0], 1.7e-3)
    self.assertLen(timing.samples_sec, 10)

  def test_a_module_on_more_devices_wins_over_a_longer_one(self):
    directory = self.create_tempdir().full_path
    space = xplane_pb2.XSpace()
    _add_plane_events(space, "0", [("jit_big(1)", 9_000_000), (_MATMUL, 1)])
    _add_plane_events(space, "1", [(_MATMUL, 1_000_000)])
    _save(directory, space)
    timing = trace_timing_lib.read_device_timing(directory, 1)
    self.assertIsNotNone(timing)
    self.assertEqual(timing.module_name, "jit__matmul")


class AnnotateResultsTest(absltest.TestCase):

  def _surplus_trace(self) -> str:
    """Device 0 holds 2 extra executions of the module from residual warmup."""
    directory = self.create_tempdir().full_path
    space = xplane_pb2.XSpace()
    _add_plane_events(
        space, "0", [(_MATMUL, d) for d in (9_000_000, 9_000_000, 1, 2, 3)]
    )
    _add_plane_events(space, "1", [(_MATMUL, d) for d in (4, 5, 6)])
    _save(directory, space)
    return directory

  def test_a_known_sample_count_keeps_every_device(self):
    # Regression: the worker called this before it set `num_executions`, so
    # the busiest device set the count and the others were left out.
    results: dict[str, Any] = {"reference_trace_dir": self._surplus_trace()}
    trace_timing_lib.annotate_results(
        results, ["reference"], expected_samples=3
    )
    self.assertCountEqual(
        results["reference_device_execution_times_sec"], ["0", "1"]
    )
    self.assertNotIn("reference_timing_notes", results)
    self.assertEqual(results["reference_timing_module"], "jit__matmul")

  def test_an_explicit_count_wins_over_the_payload(self):
    results: dict[str, Any] = {
        "num_executions": 0,
        "reference_trace_dir": self._surplus_trace(),
    }
    trace_timing_lib.annotate_results(
        results, ["reference"], expected_samples=3
    )
    self.assertLen(results["reference_trace_matched_execution_times_sec"], 3)

  def test_an_unknown_sample_count_records_the_left_out_device(self):
    results: dict[str, Any] = {"reference_trace_dir": self._surplus_trace()}
    trace_timing_lib.annotate_results(results, ["reference"])
    self.assertLen(results["reference_trace_matched_execution_times_sec"], 5)
    self.assertNotIn("reference_device_execution_times_sec", results)
    [note] = results["reference_timing_notes"]
    self.assertIn("device 1", note)
    self.assertIn("left out", note)

  def test_the_recorded_module_name_is_used(self):
    directory = self.create_tempdir().full_path
    space = xplane_pb2.XSpace()
    _add_plane_events(
        space,
        "0",
        _sharded_sample_events([1_000_000, 2_000_000], with_slice=True),
    )
    _save(directory, space)
    results: dict[str, Any] = {
        "reference_trace_dir": directory,
        "reference_module_name": "jit__matmul",
    }
    trace_timing_lib.annotate_results(
        results, ["reference"], expected_samples=2
    )
    self.assertSequenceAlmostEqual(
        results["reference_trace_matched_execution_times_sec"], [1e-6, 2e-6]
    )

  def test_a_missing_trace_records_a_note(self):
    results: dict[str, Any] = {"reference_trace_dir": "/nonexistent"}
    trace_timing_lib.annotate_results(
        results, ["reference"], expected_samples=2
    )
    self.assertNotIn("reference_trace_matched_execution_times_sec", results)
    self.assertNotEmpty(results["reference_timing_notes"])

  def test_populates_only_roles_with_readable_traces(self):
    good = self.create_tempdir().full_path
    _write_trace(good, [1_000_000, 2_000_000])
    results = {
        "num_executions": 2,
        "reference_trace_dir": good,
        "custom_trace_dir": "/nonexistent",
    }
    trace_timing_lib.annotate_results(results, ["reference", "custom"])
    self.assertIn("reference_trace_matched_execution_times_sec", results)
    self.assertNotIn("custom_trace_matched_execution_times_sec", results)

  def test_tolerates_absent_trace_dir_entry(self):
    results = {"num_executions": 2}
    trace_timing_lib.annotate_results(results, ["reference"])
    self.assertNotIn("reference_trace_matched_execution_times_sec", results)

  def test_records_devices_for_a_multi_device_run(self):
    directory = self.create_tempdir().full_path
    _write_multi_device_trace(
        directory, {"0": [1_000_000, 4_000_000], "1": [3_000_000, 2_000_000]}
    )
    results: dict[str, Any] = {
        "num_executions": 2,
        "reference_trace_dir": directory,
    }
    trace_timing_lib.annotate_results(results, ["reference"])
    self.assertSequenceAlmostEqual(
        results["reference_trace_matched_execution_times_sec"], [3e-6, 4e-6]
    )
    self.assertEqual(results["reference_timing_device"], "0")
    self.assertCountEqual(
        results["reference_device_execution_times_sec"], ["0", "1"]
    )

  def test_one_device_records_no_per_device_times(self):
    directory = self.create_tempdir().full_path
    _write_trace(directory, [1_000_000, 2_000_000])
    results = {"num_executions": 2, "reference_trace_dir": directory}
    trace_timing_lib.annotate_results(results, ["reference"])
    self.assertEqual(results["reference_timing_device"], "0")
    self.assertNotIn("reference_device_execution_times_sec", results)


class BlockTimingTest(absltest.TestCase):
  """A role that ran in blocks has one trace per block."""

  def _block(self, durations_by_device) -> str:
    """Writes the multi-device trace of one block and returns its dir."""
    directory = self.create_tempdir().full_path
    _write_multi_device_trace(directory, durations_by_device)
    return directory

  def test_blocks_are_aligned_on_their_own_and_joined_in_order(self):
    # Block 1 holds a residual warmup execution on device 0. Only that block
    # drops it: a single count over both blocks would drop a real sample.
    block0 = self._block(
        {"0": [1_000_000, 4_000_000], "1": [3_000_000, 2_000_000]}
    )
    block1 = self._block(
        {"0": [9_000_000, 5_000_000, 6_000_000], "1": [7_000_000, 1_000_000]}
    )
    results: dict[str, Any] = {
        "reference_trace_dir": block0,
        "reference_trace_dirs": [block0, block1],
        "reference_block_sample_counts": [2, 2],
    }
    trace_timing_lib.annotate_results(
        results, ["reference"], expected_samples=4
    )
    self.assertSequenceAlmostEqual(
        results["reference_trace_matched_execution_times_sec"],
        [3e-6, 4e-6, 7e-6, 6e-6],
    )
    per_device = results["reference_device_execution_times_sec"]
    self.assertSequenceAlmostEqual(per_device["0"], [1e-6, 4e-6, 5e-6, 6e-6])
    self.assertSequenceAlmostEqual(per_device["1"], [3e-6, 2e-6, 7e-6, 1e-6])
    # Device 1 is the slowest in 2 samples and device 0 in 2: the tie goes to
    # the device that sorts first.
    self.assertEqual(results["reference_timing_device"], "0")
    self.assertEqual(results["reference_timing_module"], "jit__softmax")
    self.assertNotIn("reference_timing_notes", results)

  def test_a_block_with_missing_samples_drops_the_whole_role(self):
    # Joining the rest would move every later sample out of its pair.
    block0 = self._block({"0": [1_000_000, 2_000_000]})
    block1 = self._block({"0": [3_000_000]})
    results: dict[str, Any] = {
        "reference_trace_dir": block0,
        "reference_trace_dirs": [block0, block1],
        "reference_block_sample_counts": [2, 2],
    }
    trace_timing_lib.annotate_results(
        results, ["reference"], expected_samples=4
    )
    self.assertNotIn("reference_trace_matched_execution_times_sec", results)
    notes = results["reference_timing_notes"]
    self.assertTrue(any(note.startswith("block 1: device 0") for note in notes))
    self.assertTrue(any("block 1 cannot be aligned" in note for note in notes))

  def test_a_device_missing_from_one_block_is_left_out_with_a_note(self):
    block0 = self._block(
        {"0": [1_000_000, 2_000_000], "1": [5_000_000, 5_000_000]}
    )
    block1 = self._block({"0": [3_000_000, 4_000_000]})
    results: dict[str, Any] = {
        "reference_trace_dir": block0,
        "reference_trace_dirs": [block0, block1],
        "reference_block_sample_counts": [2, 2],
    }
    trace_timing_lib.annotate_results(
        results, ["reference"], expected_samples=4
    )
    self.assertSequenceAlmostEqual(
        results["reference_trace_matched_execution_times_sec"],
        [1e-6, 2e-6, 3e-6, 4e-6],
    )
    self.assertNotIn("reference_device_execution_times_sec", results)
    [note] = results["reference_timing_notes"]
    self.assertIn("device 1", note)
    self.assertIn("block 0", note)

  def test_counts_that_do_not_add_up_drop_the_role(self):
    block0 = self._block({"0": [1_000_000]})
    block1 = self._block({"0": [2_000_000]})
    results: dict[str, Any] = {
        "reference_trace_dir": block0,
        "reference_trace_dirs": [block0, block1],
        "reference_block_sample_counts": [1, 1],
    }
    trace_timing_lib.annotate_results(
        results, ["reference"], expected_samples=3
    )
    self.assertNotIn("reference_trace_matched_execution_times_sec", results)
    [note] = results["reference_timing_notes"]
    self.assertIn("2 samples but 3 ran", note)

  def test_each_role_is_read_from_its_own_blocks(self):
    reference = [
        self._block({"0": [1_000_000]}),
        self._block({"0": [2_000_000]}),
    ]
    custom = [
        self._block({"0": [3_000_000]}),
        self._block({"0": [4_000_000]}),
    ]
    results: dict[str, Any] = {}
    for role, directories in (("reference", reference), ("custom", custom)):
      results[f"{role}_trace_dir"] = directories[0]
      results[f"{role}_trace_dirs"] = directories
      results[f"{role}_block_sample_counts"] = [1, 1]
    trace_timing_lib.annotate_results(
        results, ["reference", "custom"], expected_samples=2
    )
    self.assertSequenceAlmostEqual(
        results["reference_trace_matched_execution_times_sec"], [1e-6, 2e-6]
    )
    self.assertSequenceAlmostEqual(
        results["custom_trace_matched_execution_times_sec"], [3e-6, 4e-6]
    )

  def test_one_block_is_read_as_before(self):
    directory = self._block({"0": [1_000_000, 2_000_000]})
    results: dict[str, Any] = {
        "reference_trace_dir": directory,
        "reference_trace_dirs": [directory],
        "reference_block_sample_counts": [2],
    }
    trace_timing_lib.annotate_results(
        results, ["reference"], expected_samples=2
    )
    self.assertSequenceAlmostEqual(
        results["reference_trace_matched_execution_times_sec"], [1e-6, 2e-6]
    )
    self.assertNotIn("reference_timing_notes", results)


if __name__ == "__main__":
  absltest.main()
