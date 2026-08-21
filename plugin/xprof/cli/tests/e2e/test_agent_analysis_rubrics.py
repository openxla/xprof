"""Tier A: Agent Analysis Rubrics & Evaluation Scenarios (A-1 to A-10).

Validates autonomous agent analysis queries against standardized rubrics
and ground-truth evaluation expectations.
"""

import json
import os
from absl.testing import absltest
from absl.testing import parameterized

# pylint: disable=g-import-not-at-top
try:
  from absl.testing import absltest
  from xprof.cli.tools import get_kpi_metrics_tool
  from xprof.cli.tools import get_memory_profile_tool
  from xprof.cli.tools import get_overview_tool
  from xprof.cli.tools import get_roofline_model_tool
  from xprof.cli.tools import get_top_hlo_ops_tool
except ImportError:
  from xprof.cli.tools import get_kpi_metrics_tool
  from xprof.cli.tools import get_memory_profile_tool
  from xprof.cli.tools import get_overview_tool
  from xprof.cli.tools import get_roofline_model_tool
  from xprof.cli.tools import get_top_hlo_ops_tool


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


class AgentAnalysisRubricsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.t1_path = _get_fixture_path(
        "v6e-4-training/t1v-n-9bfa07b4-w-0.xplane.pb"
    )
    self.t2_path = _get_fixture_path(
        "tpu-training/gke-tpu-b309f56b-rq5s.xplane.pb"
    )

  def test_a01_steptime_and_duty_cycle_rubric(self):
    """A-1: Agent rubric for step time and duty cycle."""
    res_raw = get_overview_tool.get_overview(self.t1_path)
    res = json.loads(res_raw)
    self.assertIn("performance_summary", res)

  def test_a02_roofline_bound_rubric(self):
    """A-2: Agent rubric for compute vs memory bound classification."""
    res_raw = get_roofline_model_tool.get_roofline_model(self.t1_path)
    res = json.loads(res_raw)
    self.assertIsInstance(res, dict)

  def test_a03_hottest_op_rubric(self):
    """A-3: Agent rubric for single hottest op extraction."""
    res_raw = get_top_hlo_ops_tool.get_top_hlo_ops(self.t1_path, limit=1)
    res = json.loads(res_raw)
    self.assertIn("top_by_time", res)
    if res["top_by_time"]:
      top_op = res["top_by_time"][0]
      self.assertIn("name", top_op)
      self.assertIn("total_self_time_ms", top_op)

  def test_a04_source_provenance_attribution_rubric(self):
    """A-4: Agent rubric for source code provenance."""
    res_raw = get_top_hlo_ops_tool.get_top_hlo_ops(self.t1_path, limit=5)
    res = json.loads(res_raw)
    self.assertIn("top_by_time", res)
    for op in res.get("top_by_time", []):
      if "source_file" in op:
        self.assertIsInstance(op["source_file"], str)

  def test_a05_hbm_headroom_rubric(self):
    """A-5: Agent rubric for HBM memory headroom."""
    res_raw = get_memory_profile_tool.get_memory_profile(self.t1_path)
    res = json.loads(res_raw)
    self.assertIsInstance(res, dict)


if __name__ == "__main__":
  try:
    from absl.testing import absltest

    absltest.main()
  except ImportError:
    absltest.main()
