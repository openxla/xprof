"""Unit tests for get_top_hlo_ops_tool.

These tests verify that the tool correctly flattens the HLO OpProfile tree
and sorts operations by different criteria (Time, FLOPs, Memory).
"""

import json
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from xprof.cli.internal import decorators
from xprof.cli.internal.oss import xprof_client
from xprof.cli.tools import get_top_hlo_ops_tool
from xprof.protobuf import op_profile_pb2


class GetTopHloOpsToolTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    mock_cache = mock.create_autospec(
        decorators.Cache, instance=True, spec_set=True
    )
    mock_cache.get.return_value = decorators.Cache.UNKNOWN
    self.enter_context(
        mock.patch.object(
            decorators,
            "get_cache",
            return_value=mock_cache,
            autospec=True,
        )
    )
    self.mock_client = mock.create_autospec(xprof_client.CachedXprofClient)
    self.enter_context(
        mock.patch.object(
            xprof_client,
            "get_client",
            return_value=self.mock_client,
            autospec=True,
        )
    )

  def _create_fake_profile(self) -> op_profile_pb2.Profile:
    real_profile = op_profile_pb2.Profile()
    root = real_profile.by_category
    root.name = "root"
    root.metrics.raw_time = 1

    # Op 1: High Time, Low FLOPs, Med Memory
    op1 = root.children.add()
    op1.name = "Op1"
    op1.xla.SetInParent()
    op1.xla.category = "Fusion"
    op1.metrics.raw_time = 60_000_000_000  # 60ms
    op1.metrics.occurrences = 1
    op1.metrics.raw_flops = 100
    op1.metrics.raw_bytes_accessed_array.append(200)
    op1.xla.source_info.file_name = "/path/to/model.py"
    op1.xla.source_info.line_number = 42
    op1.xla.source_info.stack_frame = "main()\n"

    # Op 2: Med Time, High FLOPs, Low Memory
    op2 = root.children.add()
    op2.name = "Op2"
    op2.xla.SetInParent()
    op2.xla.category = "Convolution"
    op2.metrics.raw_time = 30_000_000_000  # 30ms
    op2.metrics.occurrences = 1
    op2.metrics.raw_flops = 1_000
    op2.metrics.raw_bytes_accessed_array.append(50)

    # Op 3: Low Time, Low FLOPs, High Memory
    op3 = root.children.add()
    op3.name = "Op3"
    op3.xla.SetInParent()
    op3.xla.category = "Tuple"
    op3.metrics.raw_time = 10_000_000_000  # 10ms
    op3.metrics.occurrences = 1
    op3.metrics.raw_flops = 10
    op3.metrics.raw_bytes_accessed_array.append(1_000)

    return real_profile

  def test_get_top_hlo_ops_by_time(self):
    real_profile = self._create_fake_profile()
    fake_bytes = real_profile.SerializeToString()
    self.mock_client.fetch.return_value = (None, fake_bytes)

    result_json = get_top_hlo_ops_tool.get_top_hlo_ops("test_session", limit=2)
    result = json.loads(result_json)

    self.assertNotIn("error", result)
    self.assertLen(result["top_by_time"], 2)
    self.assertEqual(result["top_by_time"][0]["name"], "root/Op1")
    self.assertEqual(result["top_by_time"][1]["name"], "root/Op2")
    self.mock_client.fetch.assert_called_with(
        tool_name="op_profile", session_id="test_session", format="pb"
    )

  def test_get_top_hlo_ops_by_flops(self):
    real_profile = self._create_fake_profile()
    fake_bytes = real_profile.SerializeToString()
    self.mock_client.fetch.return_value = (None, fake_bytes)

    result_json = get_top_hlo_ops_tool.get_top_hlo_ops("test_session", limit=2)
    result = json.loads(result_json)

    self.assertNotIn("error", result)
    self.assertLen(result["top_by_flops"], 2)
    self.assertEqual(result["top_by_flops"][0]["name"], "root/Op2")
    self.assertEqual(result["top_by_flops"][1]["name"], "root/Op1")

  def test_get_top_hlo_ops_by_bytes(self):
    real_profile = self._create_fake_profile()
    fake_bytes = real_profile.SerializeToString()
    self.mock_client.fetch.return_value = (None, fake_bytes)

    result_json = get_top_hlo_ops_tool.get_top_hlo_ops("test_session", limit=2)
    result = json.loads(result_json)

    self.assertNotIn("error", result)
    self.assertLen(result["top_by_bytes_accessed"], 2)
    self.assertEqual(result["top_by_bytes_accessed"][0]["name"], "root/Op3")
    self.assertEqual(result["top_by_bytes_accessed"][1]["name"], "root/Op1")

  def test_get_top_hlo_ops_no_ops(self):
    real_profile = op_profile_pb2.Profile()
    real_profile.by_category.metrics.raw_time = 0

    fake_bytes = real_profile.SerializeToString()
    self.mock_client.fetch.return_value = (None, fake_bytes)

    result_json = get_top_hlo_ops_tool.get_top_hlo_ops("test_session")
    result = json.loads(result_json)

    self.assertEqual(result["top_by_time"], [])
    self.assertEqual(result["total_matched"], 0)

  def test_get_top_hlo_ops_fetch_error(self):
    self.mock_client.fetch.return_value = None

    with self.assertRaises(RuntimeError) as cm:
      get_top_hlo_ops_tool.get_top_hlo_ops("test_session")
    self.assertIn("Failed to fetch op_profile", str(cm.exception))

  def test_get_top_hlo_ops_source_info(self):
    real_profile = self._create_fake_profile()
    fake_bytes = real_profile.SerializeToString()
    self.mock_client.fetch.return_value = (None, fake_bytes)

    result = json.loads(
        get_top_hlo_ops_tool.get_top_hlo_ops("test_session", limit=1)
    )
    op = result["top_by_time"][0]
    self.assertEqual(op["source_file"], "/path/to/model.py")
    self.assertEqual(op["source_line"], 42)
    self.assertEqual(op["stack_frame"], "main()\n")

  def test_get_top_hlo_ops_category_filter(self):
    real_profile = self._create_fake_profile()
    fake_bytes = real_profile.SerializeToString()
    self.mock_client.fetch.return_value = (None, fake_bytes)

    result = json.loads(
        get_top_hlo_ops_tool.get_top_hlo_ops(
            "test_session", limit=10, category_filter="Convolution"
        )
    )
    self.assertLen(result["top_by_time"], 1)
    self.assertEqual(result["top_by_time"][0]["name"], "root/Op2")

  def test_get_top_hlo_ops_custom_call_provenance(self):
    profile = op_profile_pb2.Profile()
    root = profile.by_category
    root.metrics.raw_time = 1000000000

    child = root.children.add()
    child.name = "IDLE"
    child.metrics.raw_time = 1000000000
    child.metrics.raw_flops = 0
    child.xla.category = "custom-call"
    child.xla.provenance = "pallas_call"

    self.mock_client.fetch.return_value = (None, profile.SerializeToString())

    result = json.loads(
        get_top_hlo_ops_tool.get_top_hlo_ops("test_session", limit=1)
    )
    self.assertLen(result["top_by_time"], 1)
    self.assertEqual(result["top_by_time"][0]["name"], "IDLE [pallas_call]")
    self.assertIn("guidance", result)
    self.assertIn("Pallas kernels", result["guidance"])

  def test_missing_hlo_returns_clean_diagnostic(self):
    """Verifies trace without HLO raises FileNotFoundError."""
    self.mock_client.fetch.return_value = (None, None)
    with self.assertRaises(FileNotFoundError) as cm:
      get_top_hlo_ops_tool.get_top_hlo_ops("session_without_hlo")
    self.assertIn("No HLO op_profile found in trace", str(cm.exception))

  def test_get_top_hlo_ops_op_name_filter_before_limit(self):
    """Verifies op_name filters before limit truncation so low-rank ops return."""
    real_profile = self._create_fake_profile()
    fake_bytes = real_profile.SerializeToString()
    self.mock_client.fetch.return_value = (None, fake_bytes)

    result = json.loads(
        get_top_hlo_ops_tool.get_top_hlo_ops(
            "test_session", limit=1, op_name="Op3"
        )
    )
    self.assertEqual(result["total_matched"], 1)
    self.assertLen(result["top_by_time"], 1)
    self.assertEqual(result["top_by_time"][0]["name"], "root/Op3")

  def test_get_top_hlo_ops_op_name_base_prefix_and_percent_strip(self):
    """Verifies op_name matches %-prefixed and .N-suffixed HLO ops with source info."""
    profile = op_profile_pb2.Profile()
    root = profile.by_category
    root.name = "fusion"
    root.metrics.raw_time = 1000000000

    child = root.children.add()
    child.name = "%broadcast_select_fusion.4"
    child.metrics.raw_time = 100000000
    child.xla.category = "fusion"
    child.xla.source_info.file_name = "jax/experimental/pallas/ops.py"
    child.xla.source_info.line_number = 128
    child.xla.source_info.stack_frame = "run_step -> pallas_call"

    self.mock_client.fetch.return_value = (None, profile.SerializeToString())

    result = json.loads(
        get_top_hlo_ops_tool.get_top_hlo_ops(
            "test_session", op_name="broadcast_select_fusion"
        )
    )
    self.assertEqual(result["total_matched"], 1)
    op = result["top_by_time"][0]
    self.assertEqual(op["name"], "fusion/%broadcast_select_fusion.4")
    self.assertEqual(op["source_file"], "jax/experimental/pallas/ops.py")
    self.assertEqual(op["source_line"], 128)
    self.assertEqual(op["stack_frame"], "run_step -> pallas_call")


if __name__ == "__main__":
  absltest.main()
