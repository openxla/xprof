"""Unit tests for XprofData HLO operation profile extraction and summarization."""

# pylint: disable=g-redundant-test-module-docstring

import json
from unittest import mock

from absl.testing import absltest
from xprof.cli.internal import decorators
from xprof.cli.internal import xprof_data
from xprof.cli.internal.oss import xprof_client
from xprof.protobuf import op_profile_pb2


class XprofDataTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    mock_cache = mock.create_autospec(
        decorators.Cache, instance=True, spec_set=True
    )
    mock_cache.get.return_value = decorators.Cache.UNKNOWN
    self.mock_cache_patcher = mock.patch.object(
        decorators,
        "get_cache",
        return_value=mock_cache,
        autospec=True,
    )
    self.mock_cache_patcher.start()
    # Mock the client directly to avoid internal-only string patching
    self.mock_client = mock.MagicMock(spec=xprof_client.CachedXprofClient)
    xprof_client.set_client_override(self.mock_client)

  def tearDown(self):
    self.mock_cache_patcher.stop()
    xprof_client.set_client_override(None)
    super().tearDown()

  def _create_mock_profile(self) -> op_profile_pb2.Profile:
    return op_profile_pb2.Profile(
        by_category=op_profile_pb2.Node(
            name="by_category",
            metrics=op_profile_pb2.Metrics(
                raw_time=100000000000, occurrences=15, raw_flops=1500
            ),
            children=[
                op_profile_pb2.Node(
                    name="MatMul",
                    category=op_profile_pb2.Node.InstructionCategory(),
                    metrics=op_profile_pb2.Metrics(
                        raw_time=60000000000,
                        occurrences=10,
                        raw_flops=1000,
                        raw_bytes_accessed_array=[100, 200],
                    ),
                ),
                op_profile_pb2.Node(
                    name="Fusion",
                    xla=op_profile_pb2.Node.XLAInstruction(
                        category="FusionCategory"
                    ),
                    metrics=op_profile_pb2.Metrics(
                        raw_time=40000000000, occurrences=5, raw_flops=500
                    ),
                ),
            ],
        )
    )

  def test_get_hlo_op_profile_grouped_default(self):
    profile = self._create_mock_profile()
    self.mock_client.fetch.return_value = (None, profile.SerializeToString())
    result = xprof_data.get_hlo_op_profile("session_op")
    result_json = json.loads(result)

    self.assertIn("category_summary", result_json)
    self.assertIn("grouped_operations", result_json)
    self.assertIn("navigation_hints", result_json)

    categories = [c["category"] for c in result_json["category_summary"]]
    self.assertIn("Category: MatMul", categories)
    self.assertIn("FusionCategory", categories)

    self.assertIn("drill_down_category", result_json["navigation_hints"])
    self.assertIn("available_categories", result_json["navigation_hints"])

  def test_get_hlo_op_profile_category_view(self):
    profile = self._create_mock_profile()
    self.mock_client.fetch.return_value = (None, profile.SerializeToString())
    result = xprof_data.get_hlo_op_profile("session_op", view="category")
    result_json = json.loads(result)

    self.assertIn("category_summary", result_json)
    self.assertNotIn("grouped_operations", result_json)
    self.assertLen(result_json["category_summary"], 2)
    self.assertEqual(
        result_json["category_summary"][0]["category"], "Category: MatMul"
    )
    self.assertEqual(
        result_json["category_summary"][0]["total_self_time_ms"], 60.0
    )

  def test_get_hlo_op_profile_category_filter(self):
    profile = self._create_mock_profile()
    self.mock_client.fetch.return_value = (None, profile.SerializeToString())
    result = xprof_data.get_hlo_op_profile("session_op", category="Fusion")
    result_json = json.loads(result)

    self.assertEqual(result_json["category"], "FusionCategory")
    self.assertEqual(result_json["total_self_time_ms"], 40.0)
    self.assertLen(result_json["operations"], 1)
    self.assertEqual(result_json["operations"][0]["name"], "by_category/Fusion")

  def test_get_hlo_op_profile_category_not_found(self):
    profile = self._create_mock_profile()
    self.mock_client.fetch.return_value = (None, profile.SerializeToString())

    with self.assertRaises(FileNotFoundError):
      xprof_data.get_hlo_op_profile("session_op", category="NonExistent")

  def test_get_hlo_op_profile_flat_view(self):
    profile = self._create_mock_profile()
    self.mock_client.fetch.return_value = (None, profile.SerializeToString())
    result = xprof_data.get_hlo_op_profile("session_op", view="flat")
    result_json = json.loads(result)

    self.assertIsInstance(result_json, list)
    self.assertLen(result_json, 2)
    self.assertEqual(result_json[0]["name"], "by_category/MatMul")
    self.assertEqual(result_json[0]["total_self_time_ms"], 60.0)
    self.assertEqual(result_json[1]["name"], "by_category/Fusion")
    self.assertEqual(result_json[1]["total_self_time_ms"], 40.0)

  def test_get_hlo_op_profile_tree_view(self):
    profile = self._create_mock_profile()
    self.mock_client.fetch.return_value = (None, profile.SerializeToString())
    result = xprof_data.get_hlo_op_profile("session_op", view="tree", depth=2)
    result_json = json.loads(result)

    self.assertIn("tree", result_json)
    self.assertIn("current_path", result_json)
    self.assertIn("navigation_hints", result_json)
    self.assertEqual(result_json["current_path"], "by_category")
    self.assertLen(result_json["tree"]["children"], 2)

  def test_get_hlo_op_profile_tree_path_not_found(self):
    profile = self._create_mock_profile()
    self.mock_client.fetch.return_value = (None, profile.SerializeToString())

    with self.assertRaises(FileNotFoundError):
      xprof_data.get_hlo_op_profile(
          "session_op", view="tree", path="invalid/path"
      )

  def test_get_hlo_op_profile_invalid_view(self):
    profile = self._create_mock_profile()
    self.mock_client.fetch.return_value = (None, profile.SerializeToString())

    with self.assertRaises(ValueError):
      xprof_data.get_hlo_op_profile("session_op", view="unsupported_view")

  def test_get_hlo_op_profile_invalid_sort_by(self):
    profile = self._create_mock_profile()
    self.mock_client.fetch.return_value = (None, profile.SerializeToString())

    with self.assertRaises(ValueError):
      xprof_data.get_hlo_op_profile("session_op", sort_by="unsupported_sort")

  def test_get_profile_summary_success(self):
    profile = op_profile_pb2.Profile(
        by_category=op_profile_pb2.Node(
            name="root", metrics=op_profile_pb2.Metrics(raw_time=100e12)
        )
    )
    self.mock_client.fetch.return_value = (None, profile.SerializeToString())

    result = xprof_data.get_profile_summary("session_summary")

    self.mock_client.fetch.assert_called_with(
        tool_name="op_profile",
        session_id="session_summary",
        format="json",
    )
    self.assertIn("Profile Summary", result)
    self.assertIn("Total Time:", result)

  def test_get_hosts(self):
    self.mock_client.get_hosts.return_value = [
        {"hostname": "host1", "ip": "1.2.3.4"},
        {"hostname": "host2", "ip": "5.6.7.8"},
    ]

    result = xprof_data.get_hosts("session_hosts")
    result_json = json.loads(result)

    self.assertIn("hosts", result_json)
    self.assertEqual(
        result_json["hosts"],
        [
            {"hostname": "host1", "ip": "1.2.3.4"},
            {"hostname": "host2", "ip": "5.6.7.8"},
        ],
    )
    self.mock_client.get_hosts.assert_called_with(
        "session_hosts", with_metadata=True
    )

  def test_get_hosts_error(self):
    self.mock_client.get_hosts.side_effect = Exception("RPC Fail")

    with self.assertRaises(RuntimeError):
      xprof_data.get_hosts("session_hosts")

  def test_get_hosts_empty(self):
    self.mock_client.get_hosts.return_value = []

    with self.assertRaises(FileNotFoundError):
      xprof_data.get_hosts("session_hosts")

  def test_get_device_information_success(self):
    roofline_json = json.dumps([{
        "p": {
            "device_type": "TPU v5p",
            "peak_flop_rate": "1234.5",
            "peak_hbm_bw": "678",
            "ridge_point": "not_a_number",
        }
    }]).encode("utf-8")
    self.mock_client.fetch.return_value = (81, roofline_json)

    result = xprof_data.get_device_information("session_device")
    result_json = json.loads(result)

    self.assertEqual(result_json["device_type"], "TPU v5p")
    self.assertEqual(result_json["peak_flop_rate"], 1234.5)
    self.assertEqual(result_json["peak_hbm_bw"], 678.0)
    self.assertEqual(result_json["ridge_point"], "not_a_number")

  def test_get_device_information_error(self):
    self.mock_client.fetch.side_effect = Exception("RPC Fail")

    with self.assertRaises(RuntimeError):
      xprof_data.get_device_information("session_device")

  def test_get_device_information_empty(self):
    self.mock_client.fetch.return_value = (81, b"")

    with self.assertRaises(FileNotFoundError):
      xprof_data.get_device_information("session_device")

  def test_get_profile_summary_missing_data(self):
    self.mock_client.fetch.return_value = (None, None)

    with self.assertRaises(FileNotFoundError):
      xprof_data.get_profile_summary("session_missing")

  def test_get_hlo_op_profile_missing_data(self):
    self.mock_client.fetch.return_value = (None, None)

    with self.assertRaises(FileNotFoundError):
      xprof_data.get_hlo_op_profile("session_missing")


if __name__ == "__main__":
  absltest.main()
