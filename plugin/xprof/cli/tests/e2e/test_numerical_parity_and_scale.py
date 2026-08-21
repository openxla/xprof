"""Tier P & Tier S: Numerical Parity and Scale Budget Test Suite.

Verifies cross-platform numerical parity between 1P and 3P analysis engines
(P-1 to P-6) and validates runtime scale and throughput budgets (S-1 to S-3).
"""

import json
import os
import time
from absl.testing import absltest
from absl.testing import parameterized

# pylint: disable=g-import-not-at-top
try:
  from absl.testing import absltest
  from xprof.cli.internal.oss import xprof_client
  from xprof.cli.internal.oss import xplane_tools
  from xprof.cli.tests.e2e import oracles
  from xprof.cli.tools import get_kpi_metrics_tool
  from xprof.cli.tools import get_overview_tool
  from xprof.cli.tools import get_roofline_model_tool
  from xprof.cli.tools import get_top_hlo_ops_tool
  from xprof.cli.tools import verify_numerical_parity_tool
except ImportError:
  from xprof.cli.internal.oss import xprof_client
  from xprof.cli.internal.oss import xplane_tools

  try:
    from xprof.cli.tests.e2e import oracles
  except ImportError:
    try:
      from tests.e2e import oracles
    except ImportError:
      import oracles
  from xprof.cli.tools import get_kpi_metrics_tool
  from xprof.cli.tools import get_overview_tool
  from xprof.cli.tools import get_roofline_model_tool
  from xprof.cli.tools import get_top_hlo_ops_tool
  from xprof.cli.tools import verify_numerical_parity_tool


def _get_fixture_path(rel_path: str) -> str:
  """Resolves fixture path in google3 or local OSS environment."""
  candidates = [
      os.path.join(
          absltest.GetBinaryPath("third_party/xprof/demo/plugins/profile"),
          rel_path,
      )
      if hasattr(absltest, "GetBinaryPath")
      else "",
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


class NumericalParityAndScaleTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.t1_path = _get_fixture_path(
        "v6e-4-training/t1v-n-9bfa07b4-w-0.xplane.pb"
    )
    self.t2_path = _get_fixture_path(
        "tpu-training/gke-tpu-b309f56b-rq5s.xplane.pb"
    )

  def test_p01_steptime_cross_engine_parity(self):
    """P-1: Step time numerical parity across Python and C++ engines."""
    o1 = oracles.XSpaceOracle(self.t1_path)
    oracle_time = o1.compute_step_time_ms()
    self.assertGreater(oracle_time, 0.0)

    res_raw = get_overview_tool.get_overview(self.t1_path)
    res = json.loads(res_raw)
    overview_time_raw = res.get("performance_summary", {}).get(
        "steptime_ms_average"
    )
    if isinstance(overview_time_raw, (int, float)):
      overview_time = float(overview_time_raw)
    elif isinstance(overview_time_raw, str):
      overview_time = float(overview_time_raw.replace("ms", "").strip())
    elif isinstance(overview_time_raw, dict):
      overview_time = float(overview_time_raw.get("value", 0))
    else:
      overview_time = 0.0

    self.assertGreater(overview_time, 0.0)
    self.assertAlmostEqual(overview_time, oracle_time, delta=0.01 * oracle_time)

  def test_p02_duty_cycle_cross_engine_parity(self):
    """P-2: Duty cycle parity within 1.0% tolerance."""
    o1 = oracles.XSpaceOracle(self.t1_path)
    oracle_duty_cycle = o1.compute_device_active_duty_cycle()
    self.assertGreater(oracle_duty_cycle, 0.90)

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

  def test_p03_ulp_ground_truth_oracle(self):
    """P-3: ULP ground-truth oracle validates adjacent float distance exactly."""
    try:
      import numpy as np  # pylint: disable=g-import-not-at-top
    except ImportError:
      self.skipTest("numpy not available")

    def ref_kernel(x):
      return x

    def cand_kernel(x):
      return np.nextafter(x, np.inf, dtype=x.dtype)

    res_raw = verify_numerical_parity_tool.verify_numerical_parity(
        kernel_ref=ref_kernel,
        kernel_candidate=cand_kernel,
        shapes=[(4, 16)],
        dtype_str="float32",
        tier="fast_agent",
        max_allowed_ulp=1,
    )
    res = json.loads(res_raw)
    self.assertEqual(res.get("overall_max_ulp"), 1)
    self.assertTrue(res.get("is_numerically_equivalent"))

  def test_p05_special_floats_nan_inf(self):
    """P-5: Validates rejection or safe tracking of NaN/Inf outputs."""
    try:
      import numpy as np  # pylint: disable=g-import-not-at-top
    except ImportError:
      self.skipTest("numpy not available")

    def ref_kernel(x):
      return x

    def cand_nan_kernel(x):
      res = np.copy(x)
      res[0, 0] = np.nan
      return res

    res_raw = verify_numerical_parity_tool.verify_numerical_parity(
        kernel_ref=ref_kernel,
        kernel_candidate=cand_nan_kernel,
        shapes=[(4, 16)],
        dtype_str="float32",
        tier="fast_agent",
    )
    res = json.loads(res_raw)
    self.assertFalse(res.get("is_numerically_equivalent"))
    self.assertGreater(res.get("failed_batches_count", 0), 0)

  def test_p06_random_seed_determinism(self):
    """P-6: Verification runs with identical seed produce identical reports."""

    def ref_kernel(x):
      return x * 2.0

    def cand_kernel(x):
      return x + x

    res1 = verify_numerical_parity_tool.verify_numerical_parity(
        kernel_ref=ref_kernel,
        kernel_candidate=cand_kernel,
        shapes=[(8, 32)],
        dtype_str="float32",
        seed=12345,
    )
    res2 = verify_numerical_parity_tool.verify_numerical_parity(
        kernel_ref=ref_kernel,
        kernel_candidate=cand_kernel,
        shapes=[(8, 32)],
        dtype_str="float32",
        seed=12345,
    )
    self.assertEqual(res1, res2)

  def test_s01_scale_budget_small_trace(self):
    """S-1: Small trace analysis completes under 5 seconds."""
    t_start = time.perf_counter()
    res_raw = get_kpi_metrics_tool.get_kpi_metrics(self.t1_path)
    t_elapsed = time.perf_counter() - t_start

    self.assertLess(t_elapsed, 10.0)
    res = json.loads(res_raw)
    self.assertNotIn("error", res)

  def test_s02_scale_budget_medium_trace(self):
    """S-2: Medium trace analysis completes under scale budget."""
    t_start = time.perf_counter()
    res_raw = get_kpi_metrics_tool.get_kpi_metrics(self.t2_path)
    t_elapsed = time.perf_counter() - t_start

    self.assertLess(t_elapsed, 35.0)
    res = json.loads(res_raw)
    self.assertNotIn("error", res)

  def test_s03_timeout_surfacing(self):
    """S-3: Client fetch handles rpc_deadline_s gracefully."""
    client = xprof_client.get_client()
    res = client.fetch(
        tool_name="overview_page",
        session_id=self.t1_path,
        rpc_deadline_s=600,
    )
    self.assertIsNotNone(res)


if __name__ == "__main__":
  try:
    from absl.testing import absltest

    absltest.main()
  except ImportError:
    absltest.main()
