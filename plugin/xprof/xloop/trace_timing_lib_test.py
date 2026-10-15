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
"""Tests for trace_timing_lib.

The fixtures here reproduce the structure observed on a real Cloud TPU v6e-8:
a `/device:TPU:0` plane carrying an `XLA Modules` line, on which samples of the
benchmarked computation alternate with `jit_dummy_op` separators.
"""

import os

from absl.testing import absltest

from tensorflow.tsl.profiler.protobuf import xplane_pb2  # pylint: disable=g-direct-tensorflow-import
from xprof.xloop import trace_timing_lib


def _write_trace(
    directory: str,
    module_durations_ps,
    *,
    plane_name: str = "/device:TPU:0",
    line_name: str = "XLA Modules",
    module_name: str = "jit__softmax(123)",
    separators: bool = True,
    line_timestamp_ns: int = 1_000,
) -> str:
  """Writes a synthetic trace in the layout `trace_context` produces.

  Args:
    directory: Role profile directory; the run subdirectory is created here.
    module_durations_ps: Duration of each benchmarked sample, in picoseconds.
    plane_name: Name of the plane to place the events on.
    line_name: Name of the line to place the events on.
    module_name: Event name used for the benchmarked samples.
    separators: Whether to interpose `jit_dummy_op` events between samples, as
      the harness does.
    line_timestamp_ns: Line base timestamp, which offsets are relative to.

  Returns:
    The path of the trace file written.
  """
  space = xplane_pb2.XSpace()
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

  run_dir = os.path.join(directory, "plugins", "profile", "run0")
  os.makedirs(run_dir, exist_ok=True)
  path = os.path.join(run_dir, "host.xplane.pb")
  with open(path, "wb") as handle:
    handle.write(space.SerializeToString())
  return path


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

  def test_picks_the_busiest_device_plane(self):
    # Defensive: a worker is pinned to one chip, but the reading should stay
    # well-defined if several planes appear.
    directory = self.create_tempdir().full_path
    path = _write_trace(directory, [1_000_000, 2_000_000])
    space = xplane_pb2.XSpace()
    with open(path, "rb") as handle:
      space.ParseFromString(handle.read())
    quiet = space.planes.add()
    quiet.name = "/device:TPU:1"
    quiet.event_metadata[1].id = 1
    quiet.event_metadata[1].name = "jit__other(7)"
    line = quiet.lines.add()
    line.name = "XLA Modules"
    event = line.events.add()
    event.metadata_id = 1
    event.duration_ps = 99_000_000
    with open(path, "wb") as handle:
      handle.write(space.SerializeToString())
    self.assertSequenceAlmostEqual(
        trace_timing_lib.read_device_execution_times_sec(directory),
        [1e-6, 2e-6],
    )


class AnnotateResultsTest(absltest.TestCase):

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


if __name__ == "__main__":
  absltest.main()
