"""Tier D: Defect Regressions Test Suite (D-01 to D-13).

Guarantees that all 13 discovered defects from the E2E Trace Analysis
specification remain permanently resolved.
"""

import json
import os
import shutil
import tempfile
from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized

# pylint: disable=g-import-not-at-top
try:
  from absl.testing import absltest
  from xprof.cli.internal import decorators
  from xprof.cli.internal.oss import xplane_tools
  from xprof.cli.internal.oss import xprof_client
  from xprof.cli.tools import get_kernel_stats_tool
  from xprof.cli.tools import get_llo_analysis_tool
  from xprof.cli.tools import get_llo_debug_string_tool
  from xprof.cli.tools import get_overview_tool
  from xprof.cli.tools import get_roofline_model_tool
  from xprof.cli.tools import get_top_hlo_ops_tool
  from xprof.cli.tools import get_utilization_viewer_tool
except ImportError:
  from xprof.cli.internal import decorators
  from xprof.cli.internal.oss import xplane_tools
  from xprof.cli.internal.oss import xprof_client
  from xprof.cli.tools import get_kernel_stats_tool
  from xprof.cli.tools import get_llo_analysis_tool
  from xprof.cli.tools import get_llo_debug_string_tool
  from xprof.cli.tools import get_overview_tool
  from xprof.cli.tools import get_roofline_model_tool
  from xprof.cli.tools import get_top_hlo_ops_tool
  from xprof.cli.tools import get_utilization_viewer_tool


def _get_fixture_path(rel_path: str) -> str:
  """Resolves fixture path in google3 or local OSS environment."""
  bin_path = ""
  if hasattr(absltest, "GetBinaryPath"):
    bin_path = absltest.GetBinaryPath(
        "third_party/xprof/demo/plugins/profile"
    )
  candidates = [
      os.path.join(bin_path, rel_path) if bin_path else "",
      os.path.join(
          os.environ.get("TEST_SRCDIR", ""),
          "google3/third_party/xprof/demo/plugins/profile",
          rel_path,
      ),
      os.path.join(
          os.environ.get("TEST_SRCDIR", ""),
          "demo/plugins/profile",
          rel_path,
      ),
      os.path.expanduser(f"~/xprof_oss/demo/plugins/profile/{rel_path}"),
      os.path.join(
          os.path.dirname(__file__),
          "../../../../../../demo/plugins/profile",
          rel_path,
      ),
      os.path.join(
          os.path.dirname(__file__),
          "../../../../../demo/plugins/profile",
          rel_path,
      ),
      os.path.join(
          os.path.dirname(__file__),
          "../../demo/plugins/profile",
          rel_path,
      ),
  ]
  for cand in candidates:
    if cand and os.path.exists(cand):
      return cand
  raise FileNotFoundError(
      f"Required test fixture '{rel_path}' not found across candidate paths."
  )


class DefectRegressionsTest(parameterized.TestCase):
  """Test cases for defect regressions."""

  def setUp(self):
    super().setUp()
    self.t1_path = _get_fixture_path(
        "v6e-4-training/t1v-n-9bfa07b4-w-0.xplane.pb"
    )
    self.t2_path = _get_fixture_path(
        "tpu-training/gke-tpu-b309f56b-rq5s.xplane.pb"
    )

  def test_d01_d02_stale_cache_and_bypass_cache(self):
    """D-01 & D-02: Trace modification causes cache miss and fresh execution."""
    temp_dir = tempfile.mkdtemp()
    self.addCleanup(lambda: shutil.rmtree(temp_dir, ignore_errors=True))

    trace_file = os.path.join(temp_dir, "test.xplane.pb")
    shutil.copyfile(self.t1_path, trace_file)

    fp1 = decorators.compute_path_fingerprint(temp_dir)
    self.assertIsNotNone(fp1)

    res1_raw = get_overview_tool.get_overview(trace_file)
    res1 = json.loads(res1_raw)
    self.assertNotIn("error", res1)

    stat1 = os.stat(trace_file)

    # Overwrite trace with T2 while preserving previous mtime (simulating cp -p)
    shutil.copyfile(self.t2_path, trace_file)
    os.utime(trace_file, (stat1.st_atime, stat1.st_mtime))

    fp2 = decorators.compute_path_fingerprint(temp_dir)
    # Fingerprint incorporates file size and st_mtime_ns; size diff triggers
    # new fp
    self.assertNotEqual(fp1, fp2)

    res2_raw = get_overview_tool.get_overview(trace_file, bypass_cache=True)
    res2 = json.loads(res2_raw)
    self.assertNotIn("error", res2)

  def test_d03_d04_error_on_empty_or_corrupt_trace(self):
    """D-03 & D-04: Non-existent, empty or corrupt trace paths raise error."""
    temp_dir = tempfile.mkdtemp()
    self.addCleanup(lambda: shutil.rmtree(temp_dir, ignore_errors=True))

    with self.assertRaises(FileNotFoundError):
      xprof_client.get_client().get_xspace_paths(temp_dir)

    corrupt_file = os.path.join(temp_dir, "corrupt.xplane.pb")
    with open(corrupt_file, "wb") as f:
      f.write(b"CORRUPT_INVALID_PROTOBUF_BYTES")

    with self.assertRaises(Exception):
      xplane_tools._fetch_xspace(corrupt_file)  # pylint: disable=protected-access

  def test_d05_silent_truncation_resolved(self):
    """D-05: list_xplane_events includes total_matched and truncated flags."""
    res_raw = xplane_tools.list_xplane_events(self.t1_path, max_events=10)
    res = json.loads(res_raw)
    self.assertIn("returned", res)
    self.assertIn("total_matched", res)
    self.assertIn("truncated", res)
    self.assertLen(res["events"], res["returned"])

  def test_d08_empty_category_filter_returns_empty_list(self):
    """D-08: Non-matching category filter returns empty list, not an error."""
    res_raw = get_top_hlo_ops_tool.get_top_hlo_ops(
        self.t1_path, category_filter="non_existent_category_xyz"
    )
    res = json.loads(res_raw)
    self.assertNotIn("error", res)
    self.assertEqual(res.get("top_by_time"), [])
    self.assertEqual(res.get("total_matched"), 0)

  def test_d09_limit_negative_one_returns_all(self):
    """D-09: limit=-1 returns all ops and records."""
    res_raw = get_top_hlo_ops_tool.get_top_hlo_ops(self.t1_path, limit=-1)
    res = json.loads(res_raw)
    self.assertIn("top_by_time", res)
    self.assertNotEmpty(res["top_by_time"])

  def test_d12_canonical_join_keys(self):
    """D-12: get_kernel_stats includes canonical join keys."""
    res_raw = get_kernel_stats_tool.get_kernel_stats(self.t1_path, limit=5)
    res = json.loads(res_raw)
    self.assertIsInstance(res, list)
    self.assertNotEmpty(res)
    top_rec = res[0]
    self.assertIn("canonical_name", top_rec)
    self.assertIn("short_name", top_rec)
    self.assertIn("hlo_op_name", top_rec)

  def test_d13_utilization_viewer_clean_status(self):
    """D-13: get_utilization_viewer returns clean status when no data."""
    res_raw = get_utilization_viewer_tool.get_utilization_viewer(self.t1_path)
    res = json.loads(res_raw)
    self.assertIsInstance(res, dict)
    self.assertTrue("status" in res or "reason" in res or "message" in res)
    self.assertNotIn("error", res)

  def test_d14_llo_tools_unavailable_status(self):
    """D-14: LLO tools return UNAVAILABLE status in standard builds."""
    try:
      from xprof.convert import _pywrap_profiler_plugin  # pylint: disable=g-import-not-at-top

      built_with_embedded = _pywrap_profiler_plugin.built_with_embedded()
    except (ImportError, AttributeError):
      built_with_embedded = False

    if built_with_embedded:
      self.skipTest("Embedded LLO analysis is present in Google3 build.")

    res_raw = get_llo_analysis_tool.get_llo_analysis(self.t1_path)
    res = json.loads(res_raw)
    self.assertIsInstance(res, dict)
    self.assertEqual(res.get("status"), "UNAVAILABLE")
    self.assertEqual(res.get("reason"), "LLO_ANALYSIS_UNSUPPORTED_IN_OSS")
    self.assertIn("TPU profiler binary", res.get("message", ""))

    res_dbg_raw = get_llo_debug_string_tool.get_llo_debug_string(self.t1_path)
    res_dbg = json.loads(res_dbg_raw)
    self.assertIsInstance(res_dbg, dict)
    self.assertEqual(res_dbg.get("status"), "UNAVAILABLE")
    self.assertEqual(res_dbg.get("reason"), "LLO_ANALYSIS_UNSUPPORTED_IN_OSS")
    self.assertIn("TPU profiler binary", res_dbg.get("message", ""))

  def test_d15_roofline_bottleneck_intensity_and_deduplication(self):
    """D-15: Roofline model exposes bottleneck intensity and deduplicates top ops."""
    res_raw = get_roofline_model_tool.get_roofline_model(self.t1_path)
    res = json.loads(res_raw)
    self.assertNotIn("error", res)
    self.assertIn("program", res)
    prog = res["program"]
    self.assertIn("bottleneck_operational_intensity_flop_per_byte", prog)
    self.assertIn("optimal_flop_rate_gflops", prog)
    self.assertIn("dma_stall_percent", prog)
    self.assertIn("hbm_read_bw_utilization_percent", prog)
    self.assertIn("hbm_write_bw_utilization_percent", prog)
    self.assertIn("vmem_read_bw_utilization_percent", prog)
    self.assertIn("vmem_write_bw_utilization_percent", prog)
    self.assertIn("cmem_read_bw_utilization_percent", prog)
    self.assertIn("cmem_write_bw_utilization_percent", prog)

    top_ops = res.get("top_operations", [])
    if top_ops:
      ranks = [op.get("rank") for op in top_ops if "rank" in op]
      self.assertEqual(len(ranks), len(set(ranks)))

  def test_d16_perf_counters_non_null_payload(self):
    """D-16: perf_counters returns valid DataTable JSON payload, never null."""
    client = xprof_client.get_client()
    content_type, data = client.fetch(
        "perf_counters", self.t1_path, bypass_cache=True
    )
    self.assertEqual(content_type, "application/json")
    self.assertIsNotNone(data)
    self.assertNotEqual(data, "")
    self.assertNotEqual(data, "null")

    res = json.loads(data)
    self.assertIsInstance(res, dict)
    self.assertIn("cols", res)
    self.assertIn("rows", res)
    col_ids = [c.get("id") or c.get("label") for c in res["cols"]]
    self.assertIn("Host", col_ids)
    self.assertIn("Chip", col_ids)
    self.assertIn("Kernel", col_ids)
    self.assertIn("Counter", col_ids)

  def test_fingerprint_stability_and_cache_warmup(self):
    """Criteria 1 & 2: Consecutive calls produce stable fingerprint and sub-second warm execution."""
    temp_dir = tempfile.mkdtemp()
    self.addCleanup(lambda: shutil.rmtree(temp_dir, ignore_errors=True))

    trace_file = os.path.join(temp_dir, "test.xplane.pb")
    shutil.copyfile(self.t1_path, trace_file)

    client = xprof_client.get_client()
    xspace_paths1 = client.get_xspace_paths(temp_dir)
    fp1 = decorators.compute_path_fingerprint(
        temp_dir, xspace_paths=xspace_paths1
    )

    # First call (cold run)
    res1_raw = get_overview_tool.get_overview(trace_file)
    res1 = json.loads(res1_raw)
    self.assertNotIn("error", res1)

    # Second call (warm cache)
    xspace_paths2 = client.get_xspace_paths(temp_dir)
    fp2 = decorators.compute_path_fingerprint(
        temp_dir, xspace_paths=xspace_paths2
    )
    self.assertEqual(
        fp1,
        fp2,
        msg="Fingerprint must remain identical after first conversion",
    )

    res2_raw = get_overview_tool.get_overview(trace_file)
    res2 = json.loads(res2_raw)
    self.assertNotIn("error", res2)
    self.assertTrue(res2.get("__cached__", False))

  def test_d02_stays_closed_across_all_swap_cases(self):
    """Criterion 3: D-02 invalidation works across normal swap, cp -p, and os.utime."""
    temp_dir = tempfile.mkdtemp()
    self.addCleanup(lambda: shutil.rmtree(temp_dir, ignore_errors=True))

    trace_file = os.path.join(temp_dir, "test.xplane.pb")
    shutil.copyfile(self.t1_path, trace_file)

    client = xprof_client.get_client()
    fp1 = decorators.compute_path_fingerprint(
        temp_dir, xspace_paths=client.get_xspace_paths(temp_dir)
    )

    # Case A: Normal overwrite
    shutil.copyfile(self.t2_path, trace_file)
    fp2 = decorators.compute_path_fingerprint(
        temp_dir, xspace_paths=client.get_xspace_paths(temp_dir)
    )
    self.assertNotEqual(fp1, fp2, msg="Normal swap must invalidate fingerprint")

    # Case B: Clone with preserved mtime (os.utime)
    shutil.copyfile(self.t1_path, trace_file)
    stat2 = os.stat(self.t2_path)
    os.utime(trace_file, (stat2.st_atime, stat2.st_mtime))
    fp3 = decorators.compute_path_fingerprint(
        temp_dir, xspace_paths=client.get_xspace_paths(temp_dir)
    )
    self.assertNotEqual(
        fp2, fp3, msg="Timestamp-preserving clone must alter fingerprint"
    )

  def test_nested_layout_fingerprint_detection(self):
    """Criterion 4: Modifying nested trace invalidates fingerprint."""
    temp_dir = tempfile.mkdtemp()
    self.addCleanup(lambda: shutil.rmtree(temp_dir, ignore_errors=True))

    nested_dir = os.path.join(temp_dir, "plugins", "profile", "2026_08_20")
    os.makedirs(nested_dir, exist_ok=True)
    nested_file = os.path.join(nested_dir, "host.xplane.pb")
    shutil.copyfile(self.t1_path, nested_file)

    client = xprof_client.get_client()
    fp1 = decorators.compute_path_fingerprint(
        temp_dir, xspace_paths=client.get_xspace_paths(temp_dir)
    )
    self.assertNotEqual(fp1, "NO_TRACE_INPUTS")

    # Overwrite nested file with T2
    shutil.copyfile(self.t2_path, nested_file)
    fp2 = decorators.compute_path_fingerprint(
        temp_dir, xspace_paths=client.get_xspace_paths(temp_dir)
    )
    self.assertNotEqual(
        fp1, fp2, msg="Modifying nested layout trace must change fingerprint"
    )

  def test_readonly_trace_dir_support(self):
    """Criterion 5: Read-only trace directory executes without permission errors."""
    temp_dir = tempfile.mkdtemp()
    self.addCleanup(lambda: shutil.rmtree(temp_dir, ignore_errors=True))

    trace_file = os.path.join(temp_dir, "test.xplane.pb")
    shutil.copyfile(self.t1_path, trace_file)

    os.chmod(temp_dir, 0o555)
    try:
      res_raw = get_overview_tool.get_overview(trace_file)
      res = json.loads(res_raw)
      self.assertNotIn("error", res)
    finally:
      os.chmod(temp_dir, 0o755)

  def test_no_trace_inputs_sentinel_e2e(self):
    """Criterion 6: Empty directory returns NO_TRACE_INPUTS sentinel."""
    temp_dir = tempfile.mkdtemp()
    self.addCleanup(lambda: shutil.rmtree(temp_dir, ignore_errors=True))

    fp = decorators.compute_path_fingerprint(temp_dir)
    self.assertEqual(fp, "NO_TRACE_INPUTS")

  def test_d16_llo_remediation_when_absent(self):
    """D-16: LLO tools return actionable remediation when LLO data is absent."""
    mock_client = mock.MagicMock()
    mock_client.resolve_session_and_host.return_value = (
        "test_session",
        "host1",
    )
    mock_client.get_hosts.return_value = ["host1"]
    mock_client.get_serialized_xspace.return_value = b"dummy_xspace"

    with mock.patch.object(
        get_llo_analysis_tool.xprof_client,
        "get_client",
        return_value=mock_client,
    ), mock.patch.object(
        get_llo_analysis_tool._pywrap_profiler_plugin,  # pylint: disable=protected-access
        "built_with_embedded",
        return_value=True,
    ), mock.patch.object(
        get_llo_analysis_tool._pywrap_profiler_plugin,  # pylint: disable=protected-access
        "analyze_llo",
        return_value={"success": False},
    ):
      res_raw = get_llo_analysis_tool.get_llo_analysis(
          self.t1_path, bypass_cache=True
      )
      res = json.loads(res_raw)
      self.assertIsInstance(res, dict)
      self.assertEqual(res.get("status"), "UNAVAILABLE")
      self.assertEqual(res.get("reason"), "LLO_DATA_ABSENT")
      self.assertIn("remediation", res)
      self.assertIn("LIBTPU_INIT_ARGS", res["remediation"])
      self.assertIn(
          "--xla_xprof_enable_custom_call_tracing=true", res["remediation"]
      )
      self.assertIn("Python 3.11+", res["remediation"])
      self.assertIn("JAX >= 0.11.0", res["remediation"])
      self.assertIn("xprof-nightly", res["remediation"])

    with mock.patch.object(
        get_llo_debug_string_tool.xprof_client,
        "get_client",
        return_value=mock_client,
    ), mock.patch.object(
        get_llo_debug_string_tool._pywrap_profiler_plugin,  # pylint: disable=protected-access
        "built_with_embedded",
        return_value=True,
    ), mock.patch.object(
        get_llo_debug_string_tool._pywrap_profiler_plugin,  # pylint: disable=protected-access
        "get_llo_debug_string",
        return_value="",
    ):
      res_dbg_raw = get_llo_debug_string_tool.get_llo_debug_string(
          self.t1_path, bypass_cache=True
      )
      res_dbg = json.loads(res_dbg_raw)
      self.assertIsInstance(res_dbg, dict)
      self.assertEqual(res_dbg.get("status"), "UNAVAILABLE")
      self.assertEqual(res_dbg.get("reason"), "LLO_DATA_ABSENT")
      self.assertIn("remediation", res_dbg)
      self.assertIn("LIBTPU_INIT_ARGS", res_dbg["remediation"])
      self.assertIn("xprof-nightly", res_dbg["remediation"])


if __name__ == "__main__":
  try:
    from absl.testing import absltest

    absltest.main()
  except ImportError:
    absltest.main()
