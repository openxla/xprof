"""Unit tests for OSS hermetic kernel_stats_tools."""

import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

# pylint: disable=g-import-not-at-top
from xprof.cli.internal.oss import kernel_stats_tools
from xprof.cli.internal.oss import xplane_tools


class OssKernelStatsToolsTest(unittest.TestCase):

  def test_get_kernel_stats_tpu_filter(self):
    # Mock TPU device XPlane containing both XLA Ops and ignored auxiliary lines
    mock_event1 = mock.MagicMock(
        name="matmul_fwd", duration_ns=1400000, stats=[]
    )
    mock_event1.name = "matmul_fwd"
    mock_line_ops = mock.MagicMock(name="XLA Ops", events=[mock_event1])
    mock_line_ops.name = "XLA Ops"

    # Auxiliary line (should be ignored to avoid bounding-box timing inflation)
    mock_event_aux = mock.MagicMock(
        name="sync_barrier", duration_ns=15000000, stats=[]
    )
    mock_event_aux.name = "sync_barrier"
    mock_line_aux = mock.MagicMock(
        name="Auxiliary Core Sync Flag", events=[mock_event_aux]
    )
    mock_line_aux.name = "Auxiliary Core Sync Flag"

    mock_plane = mock.MagicMock(
        name="/device:TPU:0", lines=[mock_line_ops, mock_line_aux]
    )
    mock_plane.name = "/device:TPU:0"

    with mock.patch.object(
        xplane_tools, "iter_planes", return_value=[mock_plane]
    ):
      res_str = kernel_stats_tools.get_kernel_stats("local_logdir")
      records = json.loads(res_str)

      self.assertEqual(len(records), 1)
      self.assertEqual(records[0]["kernel_name"], "matmul_fwd")
      self.assertEqual(records[0]["total_duration_us"], 1400.0)
      self.assertEqual(records[0]["execution_count"], 1)

  def test_get_avg_step_time(self):
    mock_step1 = mock.MagicMock(duration_ns=15000000)
    mock_step1.name = "jit_train_step"
    mock_step2 = mock.MagicMock(duration_ns=17000000)
    mock_step2.name = "jit_train_step"
    mock_line_mod = mock.MagicMock(events=[mock_step1, mock_step2])
    mock_line_mod.name = "XLA Modules"

    mock_plane = mock.MagicMock(lines=[mock_line_mod])
    mock_plane.name = "/device:TPU:0"

    with mock.patch.object(
        xplane_tools, "iter_planes", return_value=[mock_plane]
    ):
      res_str = kernel_stats_tools.get_avg_step_time(
          "local_logdir", func_name="train_step"
      )
      res = json.loads(res_str)

      self.assertEqual(res["step_count"], 2)
      self.assertAlmostEqual(res["avg_step_time_ms"], 16.0)

  def test_compute_disjoint_interval_union_ns(self):
    """Tests that overlapping intervals are correctly merged."""
    # Two overlapping intervals: [0, 100] and [50, 150] -> merged [0, 150] = 150
    intervals = [(0, 100), (50, 150)]
    result = kernel_stats_tools.compute_disjoint_interval_union_ns(intervals)
    self.assertEqual(result, 150)

    # Non-overlapping: [0, 100] and [200, 300] -> 100 + 100 = 200
    intervals = [(0, 100), (200, 300)]
    result = kernel_stats_tools.compute_disjoint_interval_union_ns(intervals)
    self.assertEqual(result, 200)

    # Nested: [0, 200] and [50, 100] -> merged [0, 200] = 200
    intervals = [(0, 200), (50, 100)]
    result = kernel_stats_tools.compute_disjoint_interval_union_ns(intervals)
    self.assertEqual(result, 200)

    # Empty list
    result = kernel_stats_tools.compute_disjoint_interval_union_ns([])
    self.assertEqual(result, 0)

  def test_get_kernel_stats_with_include_summary(self):
    """Tests that include_summary returns enriched dict with ground-truth timing."""
    # Create two overlapping events on the same XLA Ops line
    mock_event1 = mock.MagicMock(
        name="matmul_fwd", duration_ns=1400000, start_ns=0, stats=[]
    )
    mock_event1.name = "matmul_fwd"
    mock_event2 = mock.MagicMock(
        name="dot", duration_ns=800000, start_ns=700000, stats=[]
    )
    mock_event2.name = "dot"
    mock_line_ops = mock.MagicMock(
        name="XLA Ops", events=[mock_event1, mock_event2]
    )
    mock_line_ops.name = "XLA Ops"

    mock_plane = mock.MagicMock(
        name="/device:TPU:0", lines=[mock_line_ops]
    )
    mock_plane.name = "/device:TPU:0"

    with mock.patch.object(
        xplane_tools, "iter_planes", return_value=[mock_plane]
    ):
      result = kernel_stats_tools.get_kernel_stats(
          "local_logdir", output_format="dict", include_summary=True
      )

      # Verify enriched schema keys
      self.assertIn("total_device_duration_ns", result)
      self.assertIn("total_device_duration_us", result)
      self.assertIn("total_device_duration_ms", result)
      self.assertIn("kernel_records", result)
      self.assertIn("step_durations_us", result)
      self.assertIn("stats", result)

      # Verify Disjoint Interval Union: [0, 1500000] = 1500000 ns
      self.assertEqual(result["total_device_duration_ns"], 1500000)
      self.assertAlmostEqual(result["total_device_duration_us"], 1500.0)

      # Verify kernel records present
      self.assertEqual(len(result["kernel_records"]), 2)

  def test_get_kernel_stats_in_memory_profile_data(self):
    """Tests that in-memory ProfileData objects are accepted as polymorphic input."""
    mock_event = mock.MagicMock(
        name="matmul_fwd", duration_ns=1400000, start_ns=0, stats=[]
    )
    mock_event.name = "matmul_fwd"
    mock_line_ops = mock.MagicMock(name="XLA Ops", events=[mock_event])
    mock_line_ops.name = "XLA Ops"
    mock_plane = mock.MagicMock(
        name="/device:TPU:0", lines=[mock_line_ops]
    )
    mock_plane.name = "/device:TPU:0"

    # Create a mock in-memory ProfileData object with .planes attribute
    mock_profile_data = mock.MagicMock()
    mock_profile_data.planes = [mock_plane]

    # iter_planes should yield from .planes directly without server calls
    with mock.patch.object(
        xplane_tools, "iter_planes", return_value=[mock_plane]
    ) as mock_iter:
      result = kernel_stats_tools.get_kernel_stats(
          mock_profile_data, output_format="dict"
      )
      mock_iter.assert_called_once_with(mock_profile_data)
      self.assertEqual(len(result), 1)
      self.assertEqual(result[0]["kernel_name"], "matmul_fwd")

  def test_get_kernel_stats_trace_matchers(self):
    """Tests that trace_matchers filter events by name."""
    mock_event1 = mock.MagicMock(
        name="matmul_fwd", duration_ns=1400000, start_ns=0, stats=[]
    )
    mock_event1.name = "matmul_fwd"
    mock_event2 = mock.MagicMock(
        name="conv2d", duration_ns=800000, start_ns=1400000, stats=[]
    )
    mock_event2.name = "conv2d"
    mock_line_ops = mock.MagicMock(
        name="XLA Ops", events=[mock_event1, mock_event2]
    )
    mock_line_ops.name = "XLA Ops"

    mock_plane = mock.MagicMock(
        name="/device:TPU:0", lines=[mock_line_ops]
    )
    mock_plane.name = "/device:TPU:0"

    with mock.patch.object(
        xplane_tools, "iter_planes", return_value=[mock_plane]
    ):
      result = kernel_stats_tools.get_kernel_stats(
          "local_logdir",
          output_format="dict",
          trace_matchers=("matmul",),
      )
      self.assertEqual(len(result), 1)
      self.assertEqual(result[0]["kernel_name"], "matmul_fwd")

  def test_get_kernel_stats_precomputed_records(self):
    records = [{
        "kernel_name": "fusion_1",
        "total_duration_us": 100.0,
        "execution_count": 1,
        "avg_duration_us": 100.0,
    }]
    with mock.patch.object(
        xplane_tools,
        "iter_planes",
        side_effect=AssertionError("Should not parse trace"),
    ):
      json_res = kernel_stats_tools.get_kernel_stats(
          records, output_format="json"
      )
      self.assertIn("fusion_1", json_res)
      md_res = kernel_stats_tools.get_kernel_stats(
          records, output_format="markdown"
      )
      self.assertIn("fusion_1", md_res)

  def test_classify_tpu_line(self):
    self.assertEqual(kernel_stats_tools.classify_tpu_line("XLA Ops"), "kernel")
    self.assertEqual(kernel_stats_tools.classify_tpu_line("Pallas"), "kernel")
    self.assertEqual(
        kernel_stats_tools.classify_tpu_line("Pallas Primitives"),
        "intra_kernel",
    )
    self.assertEqual(
        kernel_stats_tools.classify_tpu_line("LLO Ops"), "intra_kernel"
    )
    self.assertEqual(
        kernel_stats_tools.classify_tpu_line("VPU Instructions"),
        "intra_kernel",
    )
    self.assertEqual(
        kernel_stats_tools.classify_tpu_line("XLA Modules"), "other"
    )

  def _pallas_plane_with_regions(self):
    """Builds a TPU plane where a region outlives the kernel containing it."""
    kernel_event = mock.MagicMock(duration_ns=1_556_600, start_ns=0, stats=[])
    kernel_event.name = "_at_pallas_rowblock.1"
    kernel_line = mock.MagicMock(events=[kernel_event])
    kernel_line.name = "Pallas"

    # PallasTracker merges consecutive spans of the same primitive across
    # invocations, so a region can report a longer duration than its container.
    region_event = mock.MagicMock(duration_ns=4_661_300, start_ns=0, stats=[])
    region_event.name = "reduce_sum.1"
    region_line = mock.MagicMock(events=[region_event])
    region_line.name = "Pallas Primitives"

    plane = mock.MagicMock(lines=[kernel_line, region_line])
    plane.name = "/device:TPU:0"
    return plane

  def test_intra_kernel_regions_excluded_by_default(self):
    plane = self._pallas_plane_with_regions()
    with mock.patch.object(
        xplane_tools, "iter_planes", return_value=[plane]
    ):
      records = kernel_stats_tools.get_kernel_stats(
          "local_logdir", output_format="dict"
      )

    self.assertEqual(len(records), 1)
    self.assertEqual(records[0]["kernel_name"], "_at_pallas_rowblock.1")
    self.assertFalse(records[0]["is_intra_kernel_region"])

  def test_intra_kernel_regions_opt_in_are_tagged(self):
    plane = self._pallas_plane_with_regions()
    with mock.patch.object(
        xplane_tools, "iter_planes", return_value=[plane]
    ):
      records = kernel_stats_tools.get_kernel_stats(
          "local_logdir",
          output_format="dict",
          include_intra_kernel_regions=True,
      )

    by_name = {r["kernel_name"]: r for r in records}
    self.assertCountEqual(
        by_name, ["_at_pallas_rowblock.1", "reduce_sum.1"]
    )
    self.assertTrue(by_name["reduce_sum.1"]["is_intra_kernel_region"])
    self.assertFalse(by_name["_at_pallas_rowblock.1"]["is_intra_kernel_region"])

  def test_summary_excludes_region_intervals_and_explains_why(self):
    plane = self._pallas_plane_with_regions()
    with mock.patch.object(
        xplane_tools, "iter_planes", return_value=[plane]
    ):
      summary = kernel_stats_tools.get_kernel_stats(
          "local_logdir", output_format="dict", include_summary=True
      )

    # Only the top-level kernel interval feeds the disjoint interval union.
    self.assertEqual(summary["total_device_duration_ns"], 1_556_600)
    self.assertEqual(
        summary["excluded_intra_kernel_region_lines"], ["Pallas Primitives"]
    )
    self.assertIn("include_intra_kernel_regions", summary["note"])

  def test_get_kernel_stats_logdir_root_equals_latest_run(self):
    """get_kernel_stats on a logdir root matches its latest run directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
      logdir = pathlib.Path(tmpdir) / "logs"
      old_run = logdir / "plugins" / "profile" / "2026_09_16_10_00_00"
      latest_run = logdir / "plugins" / "profile" / "2026_09_16_12_00_00"
      old_run.mkdir(parents=True)
      latest_run.mkdir(parents=True)
      (old_run / "host.xplane.pb").write_bytes(b"old_bytes")
      (latest_run / "host.xplane.pb").write_bytes(b"latest_bytes")

      def make_plane(kernel_name, duration_ns):
        ev = mock.MagicMock(duration_ns=duration_ns, start_ns=0, stats=[])
        ev.name = kernel_name
        line = mock.MagicMock(events=[ev])
        line.name = "XLA Ops"
        plane = mock.MagicMock(lines=[line])
        plane.name = "/device:TPU:0"
        return plane

      def fake_from_serialized(raw_bytes):
        pd = mock.MagicMock()
        if raw_bytes == b"old_bytes":
          pd.planes = [make_plane("old_kernel", 9_000_000)]
        else:
          pd.planes = [make_plane("latest_kernel", 2_000_000)]
        return pd

      with mock.patch.object(
          xplane_tools.profiler.ProfileData,
          "from_serialized_xspace",
          side_effect=fake_from_serialized,
      ):
        res_logdir = kernel_stats_tools.get_kernel_stats(
            str(logdir), output_format="dict"
        )
        res_latest = kernel_stats_tools.get_kernel_stats(
            str(latest_run), output_format="dict"
        )

      self.assertEqual(res_logdir, res_latest)
      self.assertEqual(len(res_logdir), 1)
      self.assertEqual(res_logdir[0]["kernel_name"], "latest_kernel")


if __name__ == "__main__":
  unittest.main()
