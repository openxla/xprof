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

  def test_get_hlo_op_profile_success(self):
    profile = op_profile_pb2.Profile(
        by_category=op_profile_pb2.Node(
            name="root",
            metrics=op_profile_pb2.Metrics(
                raw_time=1, occurrences=0, raw_flops=0
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
                        raw_time=30000000000, occurrences=5, raw_flops=500
                    ),
                ),
            ],
        )
    )

    self.mock_client.fetch.return_value = (None, profile.SerializeToString())
    result = xprof_data.get_hlo_op_profile("session_op")

    with self.subTest(name="MatMul_Assertions"):
      self.assertIn('"name": "root/MatMul"', result)
      self.assertIn('"category": "Category: MatMul"', result)
      self.assertIn('"total_self_time_ms": 60.0', result)
      self.assertIn('"bytes_accessed": 300', result)  # 100+200
    with self.subTest(name="Fusion_Assertions"):
      self.assertIn('"name": "root/Fusion"', result)
      self.assertIn('"category": "FusionCategory"', result)
      self.assertIn('"total_self_time_ms": 30.0', result)
      self.assertIn('"bytes_accessed": 0', result)

  def test_get_profile_summary_success(self):
    profile = op_profile_pb2.Profile(
        by_category=op_profile_pb2.Node(
            name="root", metrics=op_profile_pb2.Metrics(raw_time=100e12)
        )
    )
    self.mock_client.fetch.return_value = (None, profile.SerializeToString())

    result = xprof_data.get_profile_summary("session_summary")

    self.mock_client.fetch.assert_called_with(
        tool_name="hlo_op_profile.json",
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

    result = xprof_data.get_hosts("session_hosts")
    result_json = json.loads(result)

    self.assertIn("error", result_json)
    self.assertIn("RPC Fail", result_json["error"])

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

    result = xprof_data.get_device_information("session_device")
    result_json = json.loads(result)

    self.assertIn("error", result_json)
    self.assertIn("RPC Fail", result_json["error"])


if __name__ == "__main__":
  absltest.main()
