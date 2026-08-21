"""Tier N: Numerical Fidelity & Ground-Truth Parity Test Suite.

Verifies end-to-end trace analysis numerical metrics (N-1 to N-15) against
independent ground-truth oracles on standard fixtures.
"""

import json
import os
from absl.testing import absltest
from absl.testing import parameterized

# pylint: disable=g-import-not-at-top
try:
  from absl.testing import absltest
  from xprof.cli.tests.e2e import oracles
  from xprof.cli.tools import get_kpi_metrics_tool
  from xprof.cli.tools import get_memory_profile_tool
  from xprof.cli.tools import get_overview_tool
  from xprof.cli.tools import get_roofline_model_tool
  from xprof.cli.tools import get_top_hlo_ops_tool
except ImportError:
  try:
    from xprof.cli.tests.e2e import oracles
  except ImportError:
    try:
      from tests.e2e import oracles
    except ImportError:
      import oracles
  from xprof.cli.tools import get_kpi_metrics_tool
  from xprof.cli.tools import get_memory_profile_tool
  from xprof.cli.tools import get_overview_tool
  from xprof.cli.tools import get_roofline_model_tool
  from xprof.cli.tools import get_top_hlo_ops_tool


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


class NumericalFidelityTest(parameterized.TestCase):
  """Test cases for numerical fidelity."""

  def setUp(self):
    super().setUp()
    self.t1_path = _get_fixture_path(
        "v6e-4-training/t1v-n-9bfa07b4-w-0.xplane.pb"
    )
    self.t2_path = _get_fixture_path(
        "tpu-training/gke-tpu-b309f56b-rq5s.xplane.pb"
    )

  def test_n01_t1_steptime_fidelity(self):
    """N-1: T1 step time matches Oracle 1 ground truth."""
    o1 = oracles.XSpaceOracle(self.t1_path)
    oracle_step_time = o1.compute_step_time_ms()
    self.assertGreater(oracle_step_time, 0.0)

    res_raw = get_overview_tool.get_overview(self.t1_path)
    res = json.loads(res_raw)
    overview_step_time_raw = res.get("performance_summary", {}).get(
        "steptime_ms_average"
    )
    if isinstance(overview_step_time_raw, (int, float)):
      overview_step_time = float(overview_step_time_raw)
    elif isinstance(overview_step_time_raw, str):
      overview_step_time = float(
          overview_step_time_raw.replace("ms", "").strip()
      )
    elif isinstance(overview_step_time_raw, dict):
      overview_step_time = float(overview_step_time_raw.get("value", 0))
    else:
      overview_step_time = 0.0

    self.assertGreater(overview_step_time, 0.0)
    self.assertAlmostEqual(
        overview_step_time, oracle_step_time, delta=0.01 * oracle_step_time
    )

  def test_n02_t1_duty_cycle_fidelity(self):
    """N-2: T1 duty cycle matches Oracle 1 Disjoint Interval Union."""
    o1 = oracles.XSpaceOracle(self.t1_path)
    oracle_duty_cycle = o1.compute_device_active_duty_cycle()
    self.assertGreater(oracle_duty_cycle, 0.0)
    self.assertLessEqual(oracle_duty_cycle, 1.0)

    res_raw = get_overview_tool.get_overview(self.t1_path)
    res = json.loads(res_raw)
    tool_duty_cycle_raw = res.get("performance_summary", {}).get(
        "device_compute_duty_cycle_percent"
    ) or res.get("performance_summary", {}).get("duty_cycle")
    if tool_duty_cycle_raw is not None:
      if isinstance(tool_duty_cycle_raw, str):
        tool_duty_cycle = float(tool_duty_cycle_raw.replace("%", "").strip())
        if tool_duty_cycle > 1.0:
          tool_duty_cycle /= 100.0
      else:
        tool_duty_cycle = float(tool_duty_cycle_raw)
        if tool_duty_cycle > 1.0:
          tool_duty_cycle /= 100.0
      self.assertAlmostEqual(tool_duty_cycle, oracle_duty_cycle, delta=0.01)

  def test_n03_t2_steptime_fidelity(self):
    """N-3: T2 step time matches Oracle 1 ground truth."""
    o1 = oracles.XSpaceOracle(self.t2_path)
    oracle_step_time = o1.compute_step_time_ms()
    self.assertGreater(oracle_step_time, 0.0)

    res_raw = get_overview_tool.get_overview(self.t2_path)
    res = json.loads(res_raw)
    overview_step_time_raw = res.get("performance_summary", {}).get(
        "steptime_ms_average"
    )
    if isinstance(overview_step_time_raw, (int, float)):
      overview_step_time = float(overview_step_time_raw)
    elif isinstance(overview_step_time_raw, str):
      overview_step_time = float(
          overview_step_time_raw.replace("ms", "").strip()
      )
    elif isinstance(overview_step_time_raw, dict):
      overview_step_time = float(overview_step_time_raw.get("value", 0))
    else:
      overview_step_time = 0.0

    self.assertGreater(overview_step_time, 0.0)
    self.assertAlmostEqual(
        overview_step_time, oracle_step_time, delta=0.01 * oracle_step_time
    )

  def test_n05_kpi_metrics_contract(self):
    """N-5: get_kpi_metrics returns structured schema with physical bounds."""
    res_raw = get_kpi_metrics_tool.get_kpi_metrics(self.t1_path)
    res = json.loads(res_raw)
    self.assertIsInstance(res, dict)
    self.assertNotIn("error", res)
    if "step_time_ms" in res and res["step_time_ms"] != "N/A":
      step_ms = float(str(res["step_time_ms"]).replace("ms", "").strip())
      self.assertGreater(step_ms, 0.0)
    if (
        "compute_duty_cycle_percent" in res
        and res["compute_duty_cycle_percent"] != "N/A"
    ):
      duty_pct = float(
          str(res["compute_duty_cycle_percent"]).replace("%", "").strip()
      )
      self.assertGreaterEqual(duty_pct, 0.0)
      self.assertLessEqual(duty_pct, 100.0)
    if "infeed_percent" in res and res["infeed_percent"] != "N/A":
      infeed_pct = float(str(res["infeed_percent"]).replace("%", "").strip())
      self.assertGreaterEqual(infeed_pct, 0.0)
      self.assertLessEqual(infeed_pct, 100.0)

  def test_n06_roofline_model_ridge_point(self):
    """N-6: get_roofline_model ridge point calculation."""
    res_raw = get_roofline_model_tool.get_roofline_model(self.t1_path)
    res = json.loads(res_raw)
    self.assertIsInstance(res, dict)
    self.assertNotIn("error", res)
    dev_info = res.get("device_info", {})
    peak_flops = dev_info.get("peak_flops") or dev_info.get("peak_flop_rate")
    peak_bw = dev_info.get("peak_hbm_bw") or dev_info.get("peak_memory_bw")
    ridge = dev_info.get("ridge_point") or dev_info.get("hbm_ridge_point")
    if peak_flops and peak_bw and ridge:
      expected_ridge = oracles.HloRooflineOracle.compute_ridge_point(
          peak_flops, peak_bw
      )
      self.assertAlmostEqual(ridge, expected_ridge, delta=0.01 * expected_ridge)

  def test_n08_top_hlo_ops_sorting(self):
    """N-8: get_top_hlo_ops returns top operations sorted by self time."""
    res_raw = get_top_hlo_ops_tool.get_top_hlo_ops(self.t1_path, limit=5)
    res = json.loads(res_raw)
    self.assertIn("top_by_time", res)
    top_ops = res["top_by_time"]
    self.assertGreaterEqual(len(top_ops), 2)
    for i in range(len(top_ops) - 1):
      t1 = top_ops[i].get("total_self_time_ms") or top_ops[i].get(
          "self_time_ps", 0
      )
      t2 = top_ops[i + 1].get("total_self_time_ms") or top_ops[i + 1].get(
          "self_time_ps", 0
      )
      self.assertGreaterEqual(t1, t2)

  def test_n11_memory_profile_peak_bounds(self):
    """N-11: get_memory_profile reports peak memory within physical limits."""
    res_raw = get_memory_profile_tool.get_memory_profile(self.t1_path)
    res = json.loads(res_raw)
    self.assertIsInstance(res, dict)
    self.assertNotIn("error", res)
    cap = res.get("memory_capacity_gib", 0.0)
    peak = res.get("peak_memory_usage_gib", 0.0)
    details = res.get("peak_usage_details", {})
    stack = details.get("stack_reservation_gib", 0.0)
    heap = details.get("heap_allocation_gib", 0.0)
    free = details.get("free_memory_gib", 0.0)

    if cap > 0:
      self.assertLessEqual(peak, cap)
      if stack >= 0 and heap >= 0:
        self.assertAlmostEqual(stack + heap, peak, delta=0.1)
      if free >= 0:
        self.assertAlmostEqual(peak + free, cap, delta=0.1)

  def test_n15_disjoint_interval_union_invariants(self):
    """N-15: Mathematical guarantees for Disjoint Interval Union."""
    o1 = oracles.XSpaceOracle(self.t1_path)
    step_time_ms = o1.compute_step_time_ms()
    duty_cycle = o1.compute_device_active_duty_cycle()
    valid, violations = oracles.DatasheetInvariantOracle.validate_metrics(
        step_time_ms=step_time_ms,
        duty_cycle=duty_cycle,
        chip_type="TPU v6e",
    )
    self.assertTrue(valid, f"Violations: {violations}")


if __name__ == "__main__":
  try:
    from absl.testing import absltest

    absltest.main()
  except ImportError:
    absltest.main()
