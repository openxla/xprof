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
from xprof.cli.internal.oss import xprof_client
from xprof.cli.tools import get_kernel_stats_tool

# Source for tests that patch `xplane_tools.iter_planes`. A bytes source is
# read as one XSpace, so `get_kernel_stats` does not look for trace files.
_FAKE_XSPACE = b"fake-serialized-xspace"


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
      res_str = kernel_stats_tools.get_kernel_stats(_FAKE_XSPACE)
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

    # An inverted interval adds nothing instead of a negative duration.
    result = kernel_stats_tools.compute_disjoint_interval_union_ns([(100, 50)])
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
              _FAKE_XSPACE, include_summary=True
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
              _FAKE_XSPACE, include_summary=True
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
              _FAKE_XSPACE, trace_matchers=("matmul",)
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
      records = json.loads(kernel_stats_tools.get_kernel_stats(_FAKE_XSPACE))

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
              _FAKE_XSPACE, include_intra_kernel_regions=True
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
              _FAKE_XSPACE, include_summary=True
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
              _FAKE_XSPACE, include_summary=True, limit=1
          )
      )
      records = json.loads(
          kernel_stats_tools.get_kernel_stats(_FAKE_XSPACE, limit=1)
      )
      md = kernel_stats_tools.get_kernel_stats(
          _FAKE_XSPACE, limit=1, output_format="markdown"
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
              _FAKE_XSPACE, include_summary=True, limit=2
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
                _FAKE_XSPACE, include_summary=True, limit=limit
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
          kernel_stats_tools.get_kernel_stats(_FAKE_XSPACE, limit=2)
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
          kernel_stats_tools.get_kernel_stats(_FAKE_XSPACE, limit=3)
      )

    self.assertEqual(len(records), 3)
    self.assertNotIn("xprof-note", stderr.getvalue())

  def test_markdown_limit_appends_truncation_line(self):
    with mock.patch.object(
        xplane_tools, "iter_planes", return_value=[self._three_kernel_plane()]
    ):
      truncated = kernel_stats_tools.get_kernel_stats(
          _FAKE_XSPACE, limit=2, output_format="markdown"
      )
      full = kernel_stats_tools.get_kernel_stats(
          _FAKE_XSPACE, limit=0, output_format="markdown"
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
              _FAKE_XSPACE,
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
    # The cache key needs a JSON-serializable source, so use a local session
    # name (the 1P backend forwards `local_` names to this engine) that
    # resolves to one trace file.
    fake_client = mock.create_autospec(
        xprof_client.LocalXprofClient, instance=True
    )
    fake_client.get_xspace_paths.return_value = ["/run/host.xplane.pb"]
    with (
        mock.patch.object(decorators, "get_cache", return_value=cache),
        mock.patch.object(xprof_client, "get_client", return_value=fake_client),
        mock.patch.object(
            xplane_tools,
            "iter_planes",
            return_value=[self._three_kernel_plane()],
        ) as mock_iter,
    ):
      with mock.patch.object(sys, "stderr", new_callable=io.StringIO) as err1:
        res1 = json.loads(
            get_kernel_stats_tool.get_kernel_stats("local_session_1", limit=2)
        )
      with mock.patch.object(sys, "stderr", new_callable=io.StringIO) as err2:
        res2 = json.loads(
            get_kernel_stats_tool.get_kernel_stats("local_session_1", limit=2)
        )

    self.assertEqual(mock_iter.call_count, 1)
    self.assertEqual(len(res1), 2)
    self.assertEqual(res1, res2)
    self.assertIn("xprof-note: truncated: returned 2 of 3", err1.getvalue())
    self.assertIn("xprof-note: truncated: returned 2 of 3", err2.getvalue())

  def _one_kernel_plane(self, plane_name, line_name, kernel, duration_ns):
    """Returns a device plane with one kernel event on one line."""
    stats = []
    if "GPU" in plane_name:
      # GPU kernel launches carry launch parameters; memcpy events do not.
      stats = [("kernel_details", "regs:32 grid:1,1,1 block:128,1,1")]
    ev = mock.MagicMock(duration_ns=duration_ns, start_ns=0, stats=stats)
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
      records = json.loads(kernel_stats_tools.get_kernel_stats(_FAKE_XSPACE))
      summary = json.loads(
          kernel_stats_tools.get_kernel_stats(
              _FAKE_XSPACE, include_summary=True
          )
      )
      md = kernel_stats_tools.get_kernel_stats(
          _FAKE_XSPACE, output_format="markdown"
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
              _FAKE_XSPACE, device_to_use="GPU:1"
          )
      )
      every = json.loads(
          kernel_stats_tools.get_kernel_stats(_FAKE_XSPACE, device_to_use="all")
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
        kernel_stats_tools.get_kernel_stats(_FAKE_XSPACE, device_to_use="TPU:0")

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
          get_kernel_stats_tool.compute_kernel_stats(_FAKE_XSPACE)
      )

    self.assertEqual([r["kernel_name"] for r in records], ["fusion_on_tpu0"])

  def _two_host_run_dir(self):
    """Writes a run dir with 2 host traces of 2 GPUs each.

    Each trace has `gemm_<host>_gpu<N>` on `/device:GPU:<N>` and a shared
    `shared_kernel` on `/device:GPU:0`, so sums across hosts are visible.

    Returns:
      The path of the run dir.
    """
    tmpdir = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, tmpdir)
    run_dir = pathlib.Path(tmpdir)
    for host in ("host_a", "host_b"):
      (run_dir / f"{host}.xplane.pb").write_bytes(host.encode())
    return run_dir

  def _fake_host_xspace(self, raw_bytes):
    """Builds the ProfileData that `_two_host_run_dir` traces decode to."""
    host = raw_bytes.decode()
    planes = []
    for gpu in (0, 1):
      plane = self._one_kernel_plane(
          f"/device:GPU:{gpu}",
          "Stream #13(Compute)",
          f"gemm_{host}_gpu{gpu}",
          1_000_000 * (gpu + 1),
      )
      if gpu == 0:
        shared = mock.MagicMock(
            duration_ns=500_000,
            start_ns=5_000_000,
            stats=[("kernel_details", "regs:32 grid:1,1,1 block:128,1,1")],
        )
        shared.name = "shared_kernel"
        plane.lines[0].events.append(shared)
      planes.append(plane)
    pd = mock.MagicMock()
    pd.planes = planes
    return pd

  def test_multi_host_dir_default_reads_chip0_of_first_host(self):
    run_dir = self._two_host_run_dir()
    with (
        mock.patch.object(
            xplane_tools.profiler.ProfileData,
            "from_serialized_xspace",
            side_effect=self._fake_host_xspace,
        ) as mock_decode,
        mock.patch.object(sys, "stderr", new_callable=io.StringIO) as stderr,
    ):
      summary = json.loads(
          kernel_stats_tools.get_kernel_stats(
              str(run_dir), include_summary=True
          )
      )

    self.assertEqual(
        [r["kernel_name"] for r in summary["kernel_records"]],
        ["gemm_host_a_gpu0", "shared_kernel"],
    )
    self.assertEqual(summary["kernel_records"][1]["execution_count"], 1)
    self.assertEqual(summary["device"], "/device:GPU:0")
    self.assertEqual(summary["host"], "host_a")
    self.assertEqual(summary["num_hosts"], 2)
    self.assertEqual(summary["total_device_duration_us"], 1500.0)
    # Only the first host's trace is read.
    mock_decode.assert_called_once_with(b"host_a")
    self.assertIn(
        "device_to_use=None picked /device:GPU:0 on host host_a (host 1 of 2;"
        " 2 accelerator devices on that host)",
        stderr.getvalue(),
    )
    self.assertIn("--host=<name> for another host", stderr.getvalue())

  def test_multi_host_dir_explicit_device_reads_first_host(self):
    run_dir = self._two_host_run_dir()
    with (
        mock.patch.object(
            xplane_tools.profiler.ProfileData,
            "from_serialized_xspace",
            side_effect=self._fake_host_xspace,
        ),
        mock.patch.object(sys, "stderr", new_callable=io.StringIO) as stderr,
    ):
      md = kernel_stats_tools.get_kernel_stats(
          str(run_dir), device_to_use="GPU:1", output_format="markdown"
      )

    self.assertIn("`gemm_host_a_gpu1`", md)
    self.assertNotIn("host_b", md)
    self.assertIn("Device: `/device:GPU:1 on host host_a (1 of 2)`", md)
    self.assertIn(
        "device_to_use=GPU:1 read /device:GPU:1 on host host_a (host 1 of 2",
        stderr.getvalue(),
    )

  def test_multi_host_dir_all_sums_every_device_on_every_host(self):
    run_dir = self._two_host_run_dir()
    with (
        mock.patch.object(
            xplane_tools.profiler.ProfileData,
            "from_serialized_xspace",
            side_effect=self._fake_host_xspace,
        ),
        mock.patch.object(sys, "stderr", new_callable=io.StringIO) as stderr,
    ):
      summary = json.loads(
          kernel_stats_tools.get_kernel_stats(
              str(run_dir), include_summary=True, device_to_use="all"
          )
      )

    records = {r["kernel_name"]: r for r in summary["kernel_records"]}
    self.assertCountEqual(
        records,
        [
            "gemm_host_a_gpu0",
            "gemm_host_a_gpu1",
            "gemm_host_b_gpu0",
            "gemm_host_b_gpu1",
            "shared_kernel",
        ],
    )
    self.assertEqual(records["shared_kernel"]["execution_count"], 2)
    self.assertEqual(summary["device"], "all")
    self.assertEqual(summary["host"], "all")
    self.assertEqual(summary["num_hosts"], 2)
    self.assertEqual(stderr.getvalue(), "")

  def test_markdown_for_all_devices_has_no_device_line(self):
    run_dir = self._two_host_run_dir()
    with mock.patch.object(
        xplane_tools.profiler.ProfileData,
        "from_serialized_xspace",
        side_effect=self._fake_host_xspace,
    ):
      md = kernel_stats_tools.get_kernel_stats(
          str(run_dir),
          output_format="markdown",
          device_to_use=get_kernel_stats_tool.ALL_DEVICES,
      )

    self.assertIn("shared_kernel", md)
    self.assertNotIn("Device:", md)

  def test_source_without_traces_raises_file_not_found(self):
    fake_client = mock.create_autospec(
        xprof_client.LocalXprofClient, instance=True
    )
    fake_client.get_xspace_paths.return_value = []
    with mock.patch.object(
        xprof_client, "get_client", return_value=fake_client
    ):
      with self.assertRaisesRegex(FileNotFoundError, "local_empty"):
        kernel_stats_tools.load_host_planes("local_empty", all_hosts=False)

  def test_all_devices_constant_is_exported(self):
    self.assertEqual(get_kernel_stats_tool.ALL_DEVICES, "all")
    self.assertEqual(
        get_kernel_stats_tool.ALL_DEVICES, kernel_stats_tools.ALL_DEVICES
    )

  def test_single_file_reports_its_host(self):
    run_dir = self._two_host_run_dir()
    with (
        mock.patch.object(
            xplane_tools.profiler.ProfileData,
            "from_serialized_xspace",
            side_effect=self._fake_host_xspace,
        ),
        mock.patch.object(sys, "stderr", new_callable=io.StringIO),
    ):
      summary = json.loads(
          kernel_stats_tools.get_kernel_stats(
              str(run_dir / "host_b.xplane.pb"), include_summary=True
          )
      )

    self.assertEqual(summary["host"], "host_b")
    self.assertEqual(summary["num_hosts"], 1)
    self.assertEqual(
        summary["kernel_records"][0]["kernel_name"], "gemm_host_b_gpu0"
    )

  def test_merged_xspace_default_keeps_first_copy_of_each_plane(self):
    planes = [
        self._one_kernel_plane(
            "/device:GPU:0", "Stream #13(Compute)", "gemm_first", 1_000_000
        ),
        self._one_kernel_plane(
            "/device:GPU:0", "Stream #13(Compute)", "gemm_second", 2_000_000
        ),
    ]
    with (
        mock.patch.object(xplane_tools, "iter_planes", return_value=planes),
        mock.patch.object(sys, "stderr", new_callable=io.StringIO) as stderr,
    ):
      summary = json.loads(
          kernel_stats_tools.get_kernel_stats(
              _FAKE_XSPACE, include_summary=True
          )
      )
      every = json.loads(
          kernel_stats_tools.get_kernel_stats(
              _FAKE_XSPACE, include_summary=True, device_to_use="all"
          )
      )

    self.assertEqual(
        [r["kernel_name"] for r in summary["kernel_records"]], ["gemm_first"]
    )
    self.assertIsNone(summary["host"])
    self.assertEqual(summary["num_hosts"], 2)
    self.assertIn(
        "picked /device:GPU:0 on the first host (host 1 of 2; 1 accelerator"
        " devices on that host)",
        stderr.getvalue(),
    )
    self.assertIn("only the first copy of each device plane", stderr.getvalue())
    self.assertCountEqual(
        [r["kernel_name"] for r in every["kernel_records"]],
        ["gemm_first", "gemm_second"],
    )
    self.assertEqual(every["host"], "all")

  def test_parse_kernel_details(self):
    self.assertEqual(
        kernel_stats_tools.parse_kernel_details(
            "regs:128 static_shared:16 dynamic_shared:12288 grid:2176,16,1"
            " block:64,1,1 occ_pct:25 junk bad:token:x grid:1,2"
        ),
        {
            "registers_per_thread": 128,
            "static_shmem_bytes": 16,
            "dynamic_shmem_bytes": 12288,
            "block_dim": (64, 1, 1),
            "grid_dim": (2176, 16, 1),
            "occupancy_pct": 25.0,
        },
    )
    self.assertEqual(
        kernel_stats_tools.parse_kernel_details("")["block_dim"], (1, 1, 1)
    )

  def test_tensor_core_heuristics_mirror_kernel_stats_converter(self):
    self.assertTrue(
        kernel_stats_tools.is_kernel_using_tensor_core("volta_h884gemm_64x64")
    )
    self.assertTrue(
        kernel_stats_tools.is_kernel_using_tensor_core(
            "sm90_xmma_gemm_bf16bf16_bf16f32"
        )
    )
    self.assertFalse(
        kernel_stats_tools.is_kernel_using_tensor_core("loop_add_fusion")
    )
    self.assertTrue(kernel_stats_tools.is_op_tensor_core_eligible("a/MatMul"))
    self.assertTrue(
        kernel_stats_tools.is_op_tensor_core_eligible("jit(f)/dot_general")
    )
    self.assertFalse(kernel_stats_tools.is_op_tensor_core_eligible("XlaModule"))

  def test_tensor_core_flag_kept_when_later_launch_does_not_use_it(self):
    aggregate = kernel_stats_tools._GpuKernelAggregate()
    details = {"kernel_details": "regs:32 grid:1,1,1 block:128,1,1"}
    aggregate.add("sm90_xmma_gemm_bf16", 1_000, details, is_nvidia=True)
    aggregate.add("loop_add_fusion", 1_000, details, is_nvidia=True)
    self.assertTrue(aggregate.to_fields()["is_kernel_using_tensor_core"])

  def _gpu_event(self, name, start_ns, duration_ns, stats):
    ev = mock.MagicMock(start_ns=start_ns, duration_ns=duration_ns, stats=stats)
    ev.name = name
    return ev

  def _gpu_launch_plane(self):
    """Returns an Nvidia GPU plane with two kernels and one memcpy."""
    gemm_stats = [
        (
            "kernel_details",
            (
                "regs:168 static_shared:0 dynamic_shared:65536 grid:132,1,1"
                " block:384,1,1 occ_pct:12.5"
            ),
        ),
        ("hlo_op", "dot.3"),
        ("tf_op", "jit(f)/dot_general:dot_general"),
    ]
    events = [
        # 3 launches of one GEMM: two share the dominant launch config.
        self._gpu_event("sm90_xmma_gemm_bf16", 0, 40_000, gemm_stats),
        self._gpu_event("sm90_xmma_gemm_bf16", 50_000, 40_000, gemm_stats),
        self._gpu_event(
            "sm90_xmma_gemm_bf16",
            100_000,
            20_000,
            [(
                "kernel_details",
                (
                    "regs:168 dynamic_shared:65536 grid:66,1,1 block:384,1,1"
                    " occ_pct:12.5"
                ),
            )],
        ),
        # occ_pct:0 means the tracer did not compute occupancy.
        self._gpu_event(
            "loop_add_fusion",
            130_000,
            10_000,
            [
                (
                    "kernel_details",
                    "regs:16 grid:4,1,1 block:256,1,1 occ_pct:0",
                ),
                ("tf_op", "XlaModule:"),
            ],
        ),
        self._gpu_event(
            "MemcpyD2H", 200_000, 500_000, [("memcpy_details", "size:4096")]
        ),
    ]
    line = mock.MagicMock(events=events)
    line.name = "Stream #13(Compute,Memset,MemcpyD2H)"
    plane = mock.MagicMock(lines=[line], stats=[("device_vendor", "Nvidia")])
    plane.name = "/device:GPU:0"
    return plane

  def test_gpu_kernel_launch_columns(self):
    with mock.patch.object(
        xplane_tools, "iter_planes", return_value=[self._gpu_launch_plane()]
    ):
      summary = json.loads(
          kernel_stats_tools.get_kernel_stats(
              _FAKE_XSPACE, include_summary=True
          )
      )

    records = {r["kernel_name"]: r for r in summary["kernel_records"]}
    self.assertEqual(set(records), {"sm90_xmma_gemm_bf16", "loop_add_fusion"})
    gemm = records["sm90_xmma_gemm_bf16"]
    self.assertEqual(gemm["execution_count"], 3)
    self.assertEqual(gemm["total_duration_us"], 100.0)
    self.assertEqual(gemm["hlo_op_name"], "dot.3")
    self.assertEqual(gemm["registers_per_thread"], 168)
    self.assertEqual(gemm["shmem_bytes"], 65536)
    self.assertEqual(gemm["block_dim"], [384, 1, 1])
    self.assertEqual(gemm["grid_dim"], [132, 1, 1])
    self.assertEqual(gemm["occupancy_pct"], 12.5)
    self.assertEqual(gemm["num_launch_configs"], 2)
    self.assertEqual(gemm["launch_config_share_pct"], 80.0)
    self.assertTrue(gemm["is_kernel_using_tensor_core"])
    self.assertTrue(gemm["is_op_tensor_core_eligible"])
    fusion = records["loop_add_fusion"]
    self.assertIsNone(fusion["occupancy_pct"])
    self.assertFalse(fusion["is_kernel_using_tensor_core"])
    self.assertFalse(fusion["is_op_tensor_core_eligible"])
    # The memcpy is neither ranked nor counted as device kernel time.
    self.assertEqual(summary["total_device_duration_us"], 110.0)
    self.assertEqual(
        summary["excluded_gpu_non_kernel_events"],
        {
            "count": 1,
            "total_duration_us": 500.0,
            "wall_duration_us": 500.0,
            # The memcpy starts after the last kernel ends.
            "kernel_span_us": 140.0,
            "wall_duration_in_kernel_span_us": 0.0,
            "by_kind": {
                "memcpy": {
                    "count": 1,
                    "total_duration_us": 500.0,
                    "wall_duration_us": 500.0,
                }
            },
            "note": mock.ANY,
        },
    )
    self.assertNotIn("excluded_gpu_derived_lines", summary)

  def _kernel_stats(self, name, start_ns, duration_ns, hlo_op=None):
    stats = [("kernel_details", "regs:32 grid:1,1,1 block:128,1,1 occ_pct:50")]
    if hlo_op:
      stats.append(("hlo_op", hlo_op))
    return self._gpu_event(name, start_ns, duration_ns, stats)

  def _gpu_plane(self, lines_by_name):
    lines = []
    for line_name, events in lines_by_name.items():
      line = mock.MagicMock(events=events)
      line.name = line_name
      lines.append(line)
    plane = mock.MagicMock(lines=lines, stats=[("device_vendor", "Nvidia")])
    plane.name = "/device:GPU:0"
    return plane

  def _processed_gpu_plane(self):
    """Returns a GPU plane as a processed session has it.

    It has 2 streams with overlapping memcpys, a memset with the capitalized
    `Memset_details` stat, an event with no stats, and derived lines whose
    events cover the stream events again.
    """
    return self._gpu_plane({
        "Stream #13(Compute,MemcpyD2H)": [
            self._kernel_stats("gemm_kernel", 0, 100_000),
            self._gpu_event(
                "MemcpyD2H", 200_000, 300_000, [("memcpy_details", "size:8")]
            ),
            self._gpu_event("cuEventRecord", 600_000, 50_000, []),
        ],
        "Stream #14(MemcpyD2H,Memset)": [
            self._kernel_stats("late_kernel", 440_000, 10_000),
            self._gpu_event(
                "MemcpyD2H", 300_000, 300_000, [("memcpy_details", "size:8")]
            ),
            self._gpu_event(
                "Memset", 700_000, 10_000, [("Memset_details", "size:8")]
            ),
        ],
        "XLA Ops": [self._gpu_event("dot.1", 0, 900_000, [])],
        "Steps - from #13": [self._gpu_event("0", 0, 900_000, [])],
        "Framework Name Scope": [self._gpu_event("jit(f)", 0, 900_000, [])],
    })

  def test_gpu_derived_lines_and_unlabeled_events_are_not_counted(self):
    with mock.patch.object(
        xplane_tools, "iter_planes", return_value=[self._processed_gpu_plane()]
    ):
      summary = json.loads(
          kernel_stats_tools.get_kernel_stats(
              _FAKE_XSPACE, include_summary=True
          )
      )

    self.assertEqual(
        [r["kernel_name"] for r in summary["kernel_records"]],
        ["gemm_kernel", "late_kernel"],
    )
    self.assertEqual(summary["total_device_duration_us"], 110.0)
    self.assertEqual(
        summary["excluded_gpu_derived_lines"],
        ["Framework Name Scope", "Steps - from #13", "XLA Ops"],
    )
    excluded = summary["excluded_gpu_non_kernel_events"]
    # The memcpys overlap from 300 us to 500 us: they sum to 600 us and
    # cover 400 us of wall time. The event with no stats is not counted.
    self.assertEqual(excluded["count"], 3)
    self.assertEqual(excluded["total_duration_us"], 610.0)
    self.assertEqual(excluded["wall_duration_us"], 410.0)
    # The kernels span 0 to 450 us. Inside it, the memcpys cover 200 to
    # 450 us, and the memset at 700 us is outside it.
    self.assertEqual(excluded["kernel_span_us"], 450.0)
    self.assertEqual(excluded["wall_duration_in_kernel_span_us"], 250.0)
    self.assertEqual(
        excluded["by_kind"],
        {
            "memcpy": {
                "count": 2,
                "total_duration_us": 600.0,
                "wall_duration_us": 400.0,
            },
            "memset": {
                "count": 1,
                "total_duration_us": 10.0,
                "wall_duration_us": 10.0,
            },
        },
    )

  def test_is_gpu_derived_line(self):
    self.assertTrue(kernel_stats_tools.is_gpu_derived_line("XLA Ops"))
    self.assertTrue(
        kernel_stats_tools.is_gpu_derived_line("Source code - from #7")
    )
    self.assertTrue(kernel_stats_tools.is_gpu_derived_line("Command Buffers"))
    self.assertFalse(
        kernel_stats_tools.is_gpu_derived_line("Stream #13(Compute)")
    )

  def test_gpu_name_filters_omit_non_kernel_totals(self):
    for kwargs in (
        {"kernel_name": "gemm_kernel"},
        {"trace_matchers": ["gemm"]},
    ):
      with self.subTest(**kwargs):
        with mock.patch.object(
            xplane_tools,
            "iter_planes",
            return_value=[self._processed_gpu_plane()],
        ):
          summary = json.loads(
              kernel_stats_tools.get_kernel_stats(
                  _FAKE_XSPACE, include_summary=True, **kwargs
              )
          )

        self.assertEqual(summary["total_device_duration_us"], 100.0)
        self.assertNotIn("excluded_gpu_non_kernel_events", summary)
        self.assertIn(
            "Run without these filters",
            summary["excluded_gpu_non_kernel_events_note"],
        )

  def test_gpu_non_kernel_kind_uses_profiler_stat_names(self):
    self.assertEqual(
        kernel_stats_tools.gpu_non_kernel_kind({"memcpy_details": "size:8"}),
        "memcpy",
    )
    self.assertEqual(
        kernel_stats_tools.gpu_non_kernel_kind({"Memset_details": "size:8"}),
        "memset",
    )
    self.assertIsNone(
        kernel_stats_tools.gpu_non_kernel_kind({"memcpy_details": ""})
    )
    self.assertIsNone(kernel_stats_tools.gpu_non_kernel_kind({}))

  def test_launch_config_tie_breaks_on_config(self):
    first, second = (
        kernel_stats_tools._GpuKernelAggregate(),
        kernel_stats_tools._GpuKernelAggregate(),
    )
    small = {"kernel_details": "regs:32 grid:1,1,1 block:128,1,1"}
    large = {"kernel_details": "regs:64 grid:1,1,1 block:128,1,1"}
    first.add("k", 1_000, small, is_nvidia=True)
    first.add("k", 1_000, large, is_nvidia=True)
    second.add("k", 1_000, large, is_nvidia=True)
    second.add("k", 1_000, small, is_nvidia=True)

    self.assertEqual(first.to_fields(), second.to_fields())
    self.assertEqual(first.to_fields()["registers_per_thread"], 64)

  def test_hlo_op_is_the_op_with_most_launch_time(self):
    plane = self._gpu_plane({
        "Stream #7(Compute)": [
            self._kernel_stats("nvjet_gemm", 0, 10_000, hlo_op="dot.1"),
            self._kernel_stats("nvjet_gemm", 20_000, 30_000, hlo_op="dot.2"),
            self._kernel_stats("nvjet_gemm", 60_000, 10_000, hlo_op="dot.1"),
            self._kernel_stats("nvjet_gemm", 80_000, 25_000, hlo_op="dot.2"),
        ],
    })
    with mock.patch.object(xplane_tools, "iter_planes", return_value=[plane]):
      records = json.loads(kernel_stats_tools.get_kernel_stats(_FAKE_XSPACE))

    self.assertEqual(records[0]["hlo_op_name"], "dot.2")
    self.assertEqual(records[0]["num_hlo_ops"], 2)

  def test_gpu_markdown_has_launch_columns(self):
    with mock.patch.object(
        xplane_tools, "iter_planes", return_value=[self._gpu_launch_plane()]
    ):
      md = kernel_stats_tools.get_kernel_stats(
          _FAKE_XSPACE, output_format="markdown"
      )

    self.assertIn("| Occupancy (%) | Regs/Thread | Shared Mem (B) |", md)
    self.assertIn(
        "| `sm90_xmma_gemm_bf16` | 100.00 | 3 | 33.33 | 12.5 | 168 | 65536 |"
        " 384x1x1 (+1 configs) | 132x1x1 | yes | yes |",
        md,
    )
    self.assertIn(
        "| `loop_add_fusion` | 10.00 | 1 | 10.00 | n/a | 16 | 0 | 256x1x1 |"
        " 4x1x1 | no | no |",
        md,
    )

  def test_tpu_markdown_has_no_gpu_columns(self):
    with mock.patch.object(
        xplane_tools, "iter_planes", return_value=[self._three_kernel_plane()]
    ):
      md = kernel_stats_tools.get_kernel_stats(
          _FAKE_XSPACE, output_format="markdown"
      )
    self.assertNotIn("Occupancy", md)


if __name__ == "__main__":
  unittest.main()
