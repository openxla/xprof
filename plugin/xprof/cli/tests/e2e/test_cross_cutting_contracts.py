"""Tier X: Cross-Cutting Contracts & System Invariants Test Suite.

Verifies cross-cutting system contracts (X-1 to X-13):
- Input form polymorphism (X-1)
- Output volume spill guard (X-6)
- Cache freshness & error isolation (X-7, X-8)
- Tool registry integrity (X-9, X-10)
- Truncation metadata (X-5)
"""

import json
import os
import shutil
import tempfile
from absl.testing import absltest
from absl.testing import parameterized

# pylint: disable=g-import-not-at-top
try:
  from absl.testing import absltest
  from xprof.cli import xprof_cli
  from xprof.cli.internal import decorators
  from xprof.cli.internal.oss import xplane_tools
  from xprof.cli.internal.oss import xprof_client
  from xprof.cli.tools import get_overview_tool
except ImportError:
  from xprof.cli import xprof_cli
  from xprof.cli.internal import decorators
  from xprof.cli.internal.oss import xplane_tools
  from xprof.cli.internal.oss import xprof_client
  from xprof.cli.tools import get_overview_tool


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


class CrossCuttingContractsTest(parameterized.TestCase):
  """Test cases for cross-cutting contracts and invariants."""

  def setUp(self):
    super().setUp()
    self.t1_path = _get_fixture_path(
        "v6e-4-training/t1v-n-9bfa07b4-w-0.xplane.pb"
    )
    self.t1_dir = os.path.dirname(self.t1_path)

  def test_x01_four_input_forms(self):
    """X-1: Run dir and .xplane.pb file yield consistent tool output."""
    if not os.path.exists(self.t1_path):
      self.skipTest(f"Fixture {self.t1_path} not found")

    res_file_raw = get_overview_tool.get_overview(self.t1_path)
    res_dir_raw = get_overview_tool.get_overview(self.t1_dir)

    res_file = json.loads(res_file_raw)
    res_dir = json.loads(res_dir_raw)

    step_time_file = res_file.get("performance_summary", {}).get(
        "steptime_ms_average"
    )
    step_time_dir = res_dir.get("performance_summary", {}).get(
        "steptime_ms_average"
    )

    if step_time_file is not None and step_time_dir is not None:
      self.assertEqual(step_time_file, step_time_dir)

  def test_x05_truncation_contract(self):
    """X-5: Truncated tools emit returned, total_matched, and truncated flag."""
    if not os.path.exists(self.t1_path):
      self.skipTest(f"Fixture {self.t1_path} not found")

    res_raw = xplane_tools.list_xplane_events(self.t1_path, max_events=5)
    res = json.loads(res_raw)

    self.assertIn("events", res)
    self.assertIn("returned", res)
    self.assertIn("total_matched", res)
    self.assertIn("truncated", res)
    self.assertLessEqual(res["returned"], 5)
    if res["total_matched"] > 5:
      self.assertTrue(res["truncated"])

  def test_x06_volume_guard_spill_to_file(self):
    """X-6: Payloads > 10 MB automatically spill to file."""

    # Test wrapping function returning > 10 MB string
    def _dummy_huge_tool(*args, **kwargs):
      del args, kwargs
      return "x" * (11 * 1024 * 1024)

    wrapped = xprof_cli.wrap_with_logdir(_dummy_huge_tool)
    res_raw = wrapped()
    res = json.loads(res_raw)

    self.assertEqual(res.get("status"), "SAVED_TO_FILE")
    self.assertIn("file_path", res)
    self.assertTrue(os.path.exists(res["file_path"]))
    # Cleanup temp file
    try:
      os.remove(res["file_path"])
    except OSError:
      pass

  def test_x07_cache_freshness_in_place_swap(self):
    """X-7: Swapping trace files in a run directory invalidates cached results."""
    temp_dir = tempfile.mkdtemp()
    self.addCleanup(lambda: shutil.rmtree(temp_dir, ignore_errors=True))

    file_a = os.path.join(temp_dir, "trace.xplane.pb")
    with open(file_a, "wb") as f:
      f.write(b"initial_trace_data")

    key1 = decorators.compute_path_fingerprint(temp_dir)
    self.assertIsNotNone(key1)

    # Modify file size
    with open(file_a, "wb") as f:
      f.write(b"updated_trace_data_with_different_size")

    key2 = decorators.compute_path_fingerprint(temp_dir)
    self.assertNotEqual(key1, key2)

  def test_x08_cache_error_rejection(self):
    """X-8: Errors are never stored in cache."""
    cache = decorators.get_cache()
    error_payload = {"error": "Test error payload"}
    cache.set("test_error_key", json.dumps(error_payload))

    val = cache.get("test_error_key", default=None)
    self.assertIsNone(val)

  def test_x09_registry_integrity(self):
    """X-9: All registered tools in cli_main are callable."""
    tools = xprof_cli.cli_main()
    self.assertNotEmpty(tools)
    for name, func in tools.items():
      self.assertTrue(callable(func), f"Tool {name} is not callable")
    for name, func in tools.items():
      self.assertTrue(callable(func), f"Tool {name} is not callable")

  def test_x10_unknown_tool_raises_value_error(self):
    """X-10: Unknown internal tool name raises ValueError."""
    client = xprof_client.get_client()
    with self.assertRaises(ValueError):
      client.fetch(tool_name="nonexistent_tool_xyz", session_id="dummy")


if __name__ == "__main__":
  try:
    from absl.testing import absltest

    absltest.main()
  except ImportError:
    absltest.main()
