"""Unit tests for OSS hermetic kernel_stats_tools."""

import json
import sys
import unittest
from unittest import mock

# Mock xprof and dependencies so it can be imported in google3 test environment
sys.modules["xprof"] = mock.MagicMock()
sys.modules["xprof.convert"] = mock.MagicMock()
sys.modules["xprof.profile_data"] = mock.MagicMock()

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


if __name__ == "__main__":
  unittest.main()
