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
  from xprof.cli.internal import xprof_data
  from xprof.cli.tools import get_memory_profile_tool
  from xprof.cli.tools import get_overview_tool
  from xprof.cli.tools import get_roofline_model_tool
  from xprof.cli.tools import get_top_hlo_ops_tool
except ImportError:
  absltest = None
  from xprof.cli.internal import xprof_data
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


class AgentAnalysisRubricsTest(parameterized.TestCase):
  """Test cases for agent analysis rubrics."""

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

  def test_a06_hlo_op_profile_category_macro_breakdown_rubric(self):
    """A-6: Agent rubric for macro category breakdown."""
    res_raw = xprof_data.get_hlo_op_profile(self.t1_path, view="category")
    res = json.loads(res_raw)
    self.assertIsInstance(res, dict)
    self.assertIn("category_summary", res)
    self.assertNotIn("grouped_operations", res)
    self.assertIn("navigation_hints", res)
    self.assertIn("available_categories", res["navigation_hints"])
    self.assertIn("drill_down_into_top_category", res["navigation_hints"])
    for cat in res["category_summary"]:
      self.assertIn("category", cat)
      self.assertIn("total_self_time_ms", cat)
      self.assertIn("fraction_of_total_time", cat)

  def test_a07_hlo_op_profile_grouped_navigation_rubric(self):
    """A-7: Agent rubric for macro category summary with grouped operations."""
    res_raw = xprof_data.get_hlo_op_profile(
        self.t1_path, view="grouped", top_n=3
    )
    res = json.loads(res_raw)
    self.assertIsInstance(res, dict)
    self.assertIn("category_summary", res)
    self.assertIn("grouped_operations", res)
    self.assertIn("navigation_hints", res)
    self.assertIn("drill_down_category", res["navigation_hints"])
    self.assertIn("available_categories", res["navigation_hints"])
    for _, ops in res["grouped_operations"].items():
      self.assertIsInstance(ops, list)
      for op in ops:
        self.assertIn("name", op)
        self.assertIn("category", op)
        self.assertIn("category_fraction", op)
        self.assertIn("total_self_time_ms", op)

  def test_a08_hlo_op_profile_category_drill_down_rubric(self):
    """A-8: Agent rubric for single category drill-down."""
    macro_raw = xprof_data.get_hlo_op_profile(self.t1_path, view="category")
    macro = json.loads(macro_raw)
    self.assertTrue(macro["category_summary"])
    target_category = macro["category_summary"][0]["category"]

    res_raw = xprof_data.get_hlo_op_profile(
        self.t1_path, category=target_category, top_n=5
    )
    res = json.loads(res_raw)
    self.assertIsInstance(res, dict)
    self.assertEqual(res["category"], target_category)
    self.assertIn("total_self_time_ms", res)
    self.assertIn("fraction_of_total_time", res)
    self.assertIn("operations", res)
    self.assertIn("navigation_hints", res)
    self.assertIn("inspect_top_op_ast", res["navigation_hints"])
    self.assertIn("inspect_graph", res["navigation_hints"])
    self.assertIn("back_to_categories", res["navigation_hints"])
    for op in res["operations"]:
      self.assertIn("name", op)
      self.assertIn("total_self_time_ms", op)
      self.assertIn("category_fraction", op)

  def test_a09_hlo_op_profile_tree_view_rubric(self):
    """A-9: Agent rubric for hierarchical tree exploration."""
    res_raw = xprof_data.get_hlo_op_profile(
        self.t1_path, view="tree", path="by_category", depth=2
    )
    res = json.loads(res_raw)
    self.assertIsInstance(res, dict)
    self.assertIn(res["current_path"], ("by_category", "by_program", "root"))
    self.assertEqual(res["depth_limit"], 2)
    self.assertIn("tree", res)
    self.assertIn("navigation_hints", res)
    self.assertIn("available_child_paths", res["navigation_hints"])
    self.assertIn("navigate_deeper_into_child", res["navigation_hints"])
    self.assertIn("name", res["tree"])
    self.assertIn("children", res["tree"])

  def test_a10_hlo_op_profile_flat_backward_compatibility_rubric(self):
    """A-10: Agent rubric for legacy flat top-N leaf list."""
    res_raw = xprof_data.get_hlo_op_profile(self.t1_path, view="flat", top_n=5)
    res = json.loads(res_raw)
    self.assertIsInstance(res, list)
    self.assertLessEqual(len(res), 5)
    for op in res:
      self.assertIn("name", op)
      self.assertIn("category", op)
      self.assertIn("total_self_time_ms", op)


if __name__ == "__main__":
  try:
    absltest.main()
  except NameError:
    absltest.main()
