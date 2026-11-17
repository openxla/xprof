"""Unit tests for OSS hermetic kernel_stats_tools."""

import io
import json
import pathlib
import shutil
import sys
import tempfile
import unittest
from unittest import mock

# pylint: disable=g-import-not-at-top
from xprof.cli.internal import decorators
from xprof.cli.internal.oss import kernel_stats_tools
from xprof.cli.internal.oss import xplane_tools
from xprof.cli.tools import get_kernel_stats_tool


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
    """Tests that include_summary returns ground-truth timing as JSON."""
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
      result = json.loads(
          kernel_stats_tools.get_kernel_stats(
              "local_logdir", include_summary=True
          )
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

      # Verify per-line and custom-call attribution keys are always present.
      self.assertIn("by_line_duration_ns", result)
      self.assertIn("custom_call_duration_us", result)
      self.assertIn("custom_call_share_pct", result)
      self.assertEqual(result["by_line_duration_ns"]["XLA Ops"], 1500000)
      self.assertEqual(result["custom_call_share_pct"], 0.0)

      # Verify kernel records present
      self.assertEqual(len(result["kernel_records"]), 2)

  def test_get_kernel_stats_per_line_and_custom_call_attribution(self):
    """Per-line union and custom-call share are computed without overcounting."""
    mock_event1 = mock.MagicMock(
        name="matmul_fwd", duration_ns=1400000, start_ns=0, stats=[]
    )
    mock_event1.name = "matmul_fwd"
    mock_event2 = mock.MagicMock(
        name="custom-call.1", duration_ns=500000, start_ns=1400000, stats=[]
    )
    mock_event2.name = "custom-call.1"
    mock_line_ops = mock.MagicMock(
        name="XLA Ops", events=[mock_event1, mock_event2]
    )
    mock_line_ops.name = "XLA Ops"

    mock_plane = mock.MagicMock(name="/device:TPU:0", lines=[mock_line_ops])
    mock_plane.name = "/device:TPU:0"

    with mock.patch.object(
        xplane_tools, "iter_planes", return_value=[mock_plane]
    ):
      result = json.loads(
          kernel_stats_tools.get_kernel_stats(
              "local_logdir", include_summary=True
          )
      )

    # Union of [0, 1400000) and [1400000, 1900000) is contiguous: 1900000 ns.
    self.assertEqual(result["total_device_duration_ns"], 1900000)
    self.assertEqual(result["by_line_duration_ns"]["XLA Ops"], 1900000)
    self.assertAlmostEqual(result["custom_call_duration_us"], 500.0)
    self.assertAlmostEqual(result["custom_call_share_pct"], 26.32)

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
      result = json.loads(
          kernel_stats_tools.get_kernel_stats(mock_profile_data)
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
      result = json.loads(
          kernel_stats_tools.get_kernel_stats(
              "local_logdir", trace_matchers=("matmul",)
          )
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
      records = json.loads(kernel_stats_tools.get_kernel_stats("local_logdir"))

    self.assertEqual(len(records), 1)
    self.assertEqual(records[0]["kernel_name"], "_at_pallas_rowblock.1")
    self.assertFalse(records[0]["is_intra_kernel_region"])

  def test_intra_kernel_regions_opt_in_are_tagged(self):
    plane = self._pallas_plane_with_regions()
    with mock.patch.object(
        xplane_tools, "iter_planes", return_value=[plane]
    ):
      records = json.loads(
          kernel_stats_tools.get_kernel_stats(
              "local_logdir", include_intra_kernel_regions=True
          )
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
      summary = json.loads(
          kernel_stats_tools.get_kernel_stats(
              "local_logdir", include_summary=True
          )
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
        res_logdir = json.loads(
            kernel_stats_tools.get_kernel_stats(str(logdir))
        )
        res_latest = json.loads(
            kernel_stats_tools.get_kernel_stats(str(latest_run))
        )

      self.assertEqual(res_logdir, res_latest)
      self.assertEqual(len(res_logdir), 1)
      self.assertEqual(res_logdir[0]["kernel_name"], "latest_kernel")

  def _three_kernel_plane(self):
    """Returns a TPU plane with non-overlapping kernels of 3000, 2000, 1000 us."""
    events = []
    start_ns = 0
    for name, duration_ns in (
        ("k1", 3_000_000),
        ("k2", 2_000_000),
        ("k3", 1_000_000),
    ):
      ev = mock.MagicMock(duration_ns=duration_ns, start_ns=start_ns, stats=[])
      ev.name = name
      events.append(ev)
      start_ns += duration_ns
    line = mock.MagicMock(events=events)
    line.name = "XLA Ops"
    plane = mock.MagicMock(lines=[line])
    plane.name = "/device:TPU:0"
    return plane

  def test_while_loop_not_double_counted_in_total(self):
    """A `%while` op and the ops inside its body must not be summed twice."""
    # `%while.1` spans [0, 100] us and contains `fusion.1` [10, 40] us and
    # `fusion.2` [50, 90] us on the same `XLA Ops` line.
    ev_while = mock.MagicMock(duration_ns=100_000, start_ns=0, stats=[])
    ev_while.name = "%while.1"
    ev_f1 = mock.MagicMock(duration_ns=30_000, start_ns=10_000, stats=[])
    ev_f1.name = "fusion.1"
    ev_f2 = mock.MagicMock(duration_ns=40_000, start_ns=50_000, stats=[])
    ev_f2.name = "fusion.2"
    # Deliberately pass events out of start_ns order to verify sorting.
    line = mock.MagicMock(events=[ev_f1, ev_while, ev_f2])
    line.name = "XLA Ops"
    plane = mock.MagicMock(lines=[line])
    plane.name = "/device:TPU:0"

    with (
        mock.patch.object(xplane_tools, "iter_planes", return_value=[plane]),
        mock.patch.object(sys, "stderr", new_callable=io.StringIO) as stderr,
    ):
      summary = json.loads(
          kernel_stats_tools.get_kernel_stats(
              "local_logdir", include_summary=True, limit=1
          )
      )
      records = json.loads(
          kernel_stats_tools.get_kernel_stats("local_logdir", limit=1)
      )
      md = kernel_stats_tools.get_kernel_stats(
          "local_logdir", limit=1, output_format="markdown"
      )

    self.assertTrue(summary["truncated"])
    self.assertEqual(summary["total_records"], 3)
    self.assertEqual(summary["top_level_records"], 3)
    self.assertEqual(summary["returned_records"], 1)
    # Outermost total is 100 us (`%while.1` only), not 170 us (100 + 30 + 40).
    self.assertEqual(summary["total_duration_us_all_records"], 100.0)
    self.assertEqual(len(records), 1)
    self.assertIn("total_duration_us_all_records=100.0", stderr.getvalue())
    self.assertIn("100.00 us", md)

  def test_summary_limit_marks_truncation_and_keeps_totals(self):
    with mock.patch.object(
        xplane_tools, "iter_planes", return_value=[self._three_kernel_plane()]
    ):
      result = json.loads(
          kernel_stats_tools.get_kernel_stats(
              "local_logdir", include_summary=True, limit=2
          )
      )

    self.assertEqual(len(result["kernel_records"]), 2)
    self.assertTrue(result["truncated"])
    self.assertEqual(result["total_records"], 3)
    self.assertEqual(result["returned_records"], 2)
    # All three kernels, not the 5000 us of the two returned rows.
    self.assertEqual(result["total_duration_us_all_records"], 6000.0)

  def test_summary_limit_not_truncated(self):
    for limit in (10, 3, 0):
      with (
          self.subTest(limit=limit),
          mock.patch.object(
              xplane_tools,
              "iter_planes",
              return_value=[self._three_kernel_plane()],
          ),
      ):
        result = json.loads(
            kernel_stats_tools.get_kernel_stats(
                "local_logdir", include_summary=True, limit=limit
            )
        )
        self.assertEqual(len(result["kernel_records"]), 3)
        self.assertFalse(result["truncated"])
        self.assertEqual(result["total_records"], 3)
        self.assertEqual(result["returned_records"], 3)
        self.assertEqual(result["total_duration_us_all_records"], 6000.0)

  def test_list_limit_writes_truncation_note_to_stderr(self):
    with (
        mock.patch.object(
            xplane_tools,
            "iter_planes",
            return_value=[self._three_kernel_plane()],
        ),
        mock.patch.object(sys, "stderr", new_callable=io.StringIO) as stderr,
    ):
      records = json.loads(
          kernel_stats_tools.get_kernel_stats("local_logdir", limit=2)
      )

    self.assertEqual(len(records), 2)
    self.assertIn("xprof-note: truncated: returned 2 of 3", stderr.getvalue())
    self.assertIn("total_duration_us_all_records=6000.0", stderr.getvalue())

  def test_list_without_truncation_writes_no_note(self):
    with (
        mock.patch.object(
            xplane_tools,
            "iter_planes",
            return_value=[self._three_kernel_plane()],
        ),
        mock.patch.object(sys, "stderr", new_callable=io.StringIO) as stderr,
    ):
      records = json.loads(
          kernel_stats_tools.get_kernel_stats("local_logdir", limit=3)
      )

    self.assertEqual(len(records), 3)
    self.assertNotIn("xprof-note", stderr.getvalue())

  def test_markdown_limit_appends_truncation_line(self):
    with mock.patch.object(
        xplane_tools, "iter_planes", return_value=[self._three_kernel_plane()]
    ):
      truncated = kernel_stats_tools.get_kernel_stats(
          "local_logdir", limit=2, output_format="markdown"
      )
      full = kernel_stats_tools.get_kernel_stats(
          "local_logdir", limit=0, output_format="markdown"
      )

    self.assertIn("Showing 2 of 3 kernels", truncated)
    self.assertIn("6000.00 us", truncated)
    self.assertNotIn("Showing", full)

  def test_precomputed_records_limit_marks_truncation(self):
    records = [
        {
            "kernel_name": f"k{i}",
            "total_duration_us": float((3 - i) * 1000),
            "execution_count": 1,
            "avg_duration_us": float((3 - i) * 1000),
        }
        for i in range(3)
    ]
    with mock.patch.object(sys, "stderr", new_callable=io.StringIO) as stderr:
      limited = json.loads(
          kernel_stats_tools.get_kernel_stats(records, limit=2)
      )
    self.assertEqual(len(limited), 2)
    self.assertIn("xprof-note: truncated: returned 2 of 3", stderr.getvalue())
    self.assertIn("total_duration_us_all_records=null", stderr.getvalue())

    summary = json.loads(
        kernel_stats_tools.get_kernel_stats(
            {"kernel_records": records, "total_device_duration_us": 5000.0},
            include_summary=True,
            limit=2,
        )
    )
    self.assertTrue(summary["truncated"])
    self.assertEqual(summary["total_records"], 3)
    self.assertEqual(summary["top_level_records"], 3)
    self.assertEqual(summary["returned_records"], 2)
    self.assertIsNone(summary["total_duration_us_all_records"])
    self.assertEqual(summary["total_device_duration_us"], 5000.0)

    md = kernel_stats_tools.get_kernel_stats(
        records, limit=2, output_format="markdown"
    )
    self.assertIn("Showing 2 of 3 kernels", md)
    self.assertIn(
        "unavailable for pre-computed records without event timestamps", md
    )

    # Pre-computed inputs with intra-kernel region rows keep those rows by
    # default while excluding them from top_level_records.
    records_with_region = records + [{
        "kernel_name": "dma_region",
        "total_duration_us": 500.0,
        "execution_count": 1,
        "avg_duration_us": 500.0,
        "is_intra_kernel_region": True,
    }]
    kept = json.loads(
        kernel_stats_tools.get_kernel_stats(records_with_region, limit=0)
    )
    self.assertEqual(len(kept), 4)
    region_summary = json.loads(
        kernel_stats_tools.get_kernel_stats(
            records_with_region, include_summary=True, limit=0
        )
    )
    self.assertEqual(region_summary["total_records"], 4)
    self.assertEqual(region_summary["top_level_records"], 3)
    self.assertEqual(region_summary["returned_records"], 4)
    self.assertIsNone(region_summary["total_duration_us_all_records"])

  def test_summary_intra_kernel_regions_separates_top_level_count(self):
    plane = self._pallas_plane_with_regions()
    with mock.patch.object(xplane_tools, "iter_planes", return_value=[plane]):
      summary = json.loads(
          kernel_stats_tools.get_kernel_stats(
              "local_logdir",
              include_summary=True,
              include_intra_kernel_regions=True,
              limit=1,
          )
      )
    self.assertTrue(summary["truncated"])
    self.assertEqual(summary["total_records"], 2)
    self.assertEqual(summary["top_level_records"], 1)
    self.assertEqual(summary["returned_records"], 1)
    self.assertEqual(summary["total_duration_us_all_records"], 1556.6)

  def test_cached_get_kernel_stats_replays_truncation_stderr_note(self):
    temp_dir = pathlib.Path(tempfile.mkdtemp())
    self.addCleanup(lambda: shutil.rmtree(temp_dir, ignore_errors=True))
    cache = decorators.Cache(temp_dir)
    with (
        mock.patch.object(decorators, "get_cache", return_value=cache),
        mock.patch.object(
            xplane_tools,
            "iter_planes",
            return_value=[self._three_kernel_plane()],
        ) as mock_iter,
    ):
      with mock.patch.object(sys, "stderr", new_callable=io.StringIO) as err1:
        res1 = json.loads(
            get_kernel_stats_tool.get_kernel_stats("local_logdir", limit=2)
        )
      with mock.patch.object(sys, "stderr", new_callable=io.StringIO) as err2:
        res2 = json.loads(
            get_kernel_stats_tool.get_kernel_stats("local_logdir", limit=2)
        )

    self.assertEqual(mock_iter.call_count, 1)
    self.assertEqual(len(res1), 2)
    self.assertEqual(res1, res2)
    self.assertIn("xprof-note: truncated: returned 2 of 3", err1.getvalue())
    self.assertIn("xprof-note: truncated: returned 2 of 3", err2.getvalue())

  def _one_kernel_plane(self, plane_name, line_name, kernel, duration_ns):
    """Returns a device plane with one kernel event on one line."""
    ev = mock.MagicMock(duration_ns=duration_ns, start_ns=0, stats=[])
    ev.name = kernel
    line = mock.MagicMock(events=[ev])
    line.name = line_name
    plane = mock.MagicMock(lines=[line])
    plane.name = plane_name
    return plane

  def _gpu_planes(self):
    """Returns host, GPU:1 and GPU:0 planes with distinct kernels."""
    host = mock.MagicMock(lines=[])
    host.name = "/host:CPU"
    return [
        host,
        self._one_kernel_plane(
            "/device:GPU:1", "Stream #13(Compute)", "gemm_on_gpu1", 2_000_000
        ),
        self._one_kernel_plane(
            "/device:GPU:0", "Stream #13(Compute)", "gemm_on_gpu0", 1_000_000
        ),
    ]

  def test_resolve_device_planes(self):
    names = [
        "/host:CPU",
        "/device:GPU:0",
        "/device:TPU:10",
        "/device:TPU:2",
        "/device:TPU:2 SparseCore 0",
        "/device:TPU:2",
    ]
    self.assertEqual(
        kernel_stats_tools.resolve_device_planes(names, None),
        ("/device:TPU:2", ["/device:TPU:2", "/device:TPU:2 SparseCore 0"]),
    )
    self.assertEqual(
        kernel_stats_tools.resolve_device_planes(names, "gpu:0"),
        ("/device:GPU:0", ["/device:GPU:0"]),
    )
    self.assertEqual(
        kernel_stats_tools.resolve_device_planes(names, "/device:TPU:10"),
        ("/device:TPU:10", ["/device:TPU:10"]),
    )
    self.assertEqual(
        kernel_stats_tools.resolve_device_planes(names, "ALL"),
        (
            "all",
            [
                "/device:GPU:0",
                "/device:TPU:10",
                "/device:TPU:2",
                "/device:TPU:2 SparseCore 0",
            ],
        ),
    )
    self.assertEqual(
        kernel_stats_tools.resolve_device_planes(["/host:CPU"], "TPU:0"),
        ("all", []),
    )
    with self.assertRaisesRegex(ValueError, "matches no device plane"):
      kernel_stats_tools.resolve_device_planes(names, "GPU:1")

  def test_default_device_picks_first_gpu_and_notes_it(self):
    with (
        mock.patch.object(
            xplane_tools, "iter_planes", return_value=self._gpu_planes()
        ),
        mock.patch.object(sys, "stderr", new_callable=io.StringIO) as stderr,
    ):
      records = json.loads(kernel_stats_tools.get_kernel_stats("local_logdir"))
      summary = json.loads(
          kernel_stats_tools.get_kernel_stats(
              "local_logdir", include_summary=True
          )
      )
      md = kernel_stats_tools.get_kernel_stats(
          "local_logdir", output_format="markdown"
      )

    self.assertEqual([r["kernel_name"] for r in records], ["gemm_on_gpu0"])
    self.assertEqual(records[0]["total_duration_us"], 1000.0)
    self.assertEqual(summary["device"], "/device:GPU:0")
    self.assertEqual(summary["device_planes"], ["/device:GPU:0"])
    self.assertEqual(summary["total_device_duration_us"], 1000.0)
    self.assertIn("Device: `/device:GPU:0`", md)
    self.assertIn(
        "device_to_use=None picked /device:GPU:0 of 2", stderr.getvalue()
    )

  def test_explicit_device_and_all(self):
    with (
        mock.patch.object(
            xplane_tools, "iter_planes", return_value=self._gpu_planes()
        ),
        mock.patch.object(sys, "stderr", new_callable=io.StringIO) as stderr,
    ):
      gpu1 = json.loads(
          kernel_stats_tools.get_kernel_stats(
              "local_logdir", device_to_use="GPU:1"
          )
      )
      every = json.loads(
          kernel_stats_tools.get_kernel_stats(
              "local_logdir", device_to_use="all"
          )
      )

    self.assertEqual([r["kernel_name"] for r in gpu1], ["gemm_on_gpu1"])
    self.assertEqual(
        [r["kernel_name"] for r in every], ["gemm_on_gpu1", "gemm_on_gpu0"]
    )
    self.assertEqual(stderr.getvalue(), "")

  def test_unknown_device_raises_value_error(self):
    with mock.patch.object(
        xplane_tools, "iter_planes", return_value=self._gpu_planes()
    ):
      with self.assertRaisesRegex(ValueError, "TPU:0.*matches no device"):
        kernel_stats_tools.get_kernel_stats(
            "local_logdir", device_to_use="TPU:0"
        )

  def test_tool_default_device_picks_lowest_tpu_index(self):
    planes = [
        self._one_kernel_plane(
            "/device:TPU:1", "XLA Ops", "fusion_on_tpu1", 1_000_000
        ),
        self._one_kernel_plane(
            "/device:TPU:0", "XLA Ops", "fusion_on_tpu0", 1_000_000
        ),
    ]
    with (
        mock.patch.object(xplane_tools, "iter_planes", return_value=planes),
        mock.patch.object(sys, "stderr", new_callable=io.StringIO),
    ):
      records = json.loads(
          get_kernel_stats_tool.compute_kernel_stats("local_logdir")
      )

    self.assertEqual([r["kernel_name"] for r in records], ["fusion_on_tpu0"])


if __name__ == "__main__":
  unittest.main()
