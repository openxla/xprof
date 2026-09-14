import json
from unittest import mock
from absl.testing import absltest
from xprof.cli.internal import decorators
from xprof.cli.internal.oss import xprof_client
from xprof.cli.tools import get_overview_tool


class GetOverviewToolTest(absltest.TestCase):

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

  def test_get_overview_success(self):
    overview_data = [
        {
            "p": {
                "stat_memory_bw": "800 GiB/s",
                "stat_step_time": "15ms",
                "run_environment": "TPU v4",
                "idle_percent%": "15.5",
                "compute_percent%": "84.5",
                "invalid_percent%": "not_a_number",
                "device_type": "TPU v4",
            }
        },
        {
            "cols": [
                {"id": "host_id", "type": "string"},
                {"id": "bns_address", "type": "string"},
                {"id": "command_line", "type": "string"},
            ],
            "rows": [{
                "c": [
                    {"v": "test_host"},
                    {"v": "/bns/test/address"},
                    {
                        "v": (
                            "test command"
                            " --streamz_default_root_labels=xmanager:int:12345"
                        )
                    },
                ]
            }],
        },
    ]
    overview_json = json.dumps(overview_data).encode("utf-8")
    self.mock_client.fetch.return_value = (81, overview_json)

    # Test without command line
    result = get_overview_tool.get_overview(
        "session_123", include_command=False
    )
    result_json = json.loads(result)

    self.assertEqual(
        result_json["performance_summary"]["stat_memory_bw"], "800 GiB/s"
    )
    self.assertEqual(
        result_json["performance_summary"]["stat_step_time"], "15ms"
    )
    self.assertEqual(
        result_json["run_environment"]["run_environment"], "TPU v4"
    )
    self.assertEqual(result_json["run_environment"]["device_type"], "TPU v4")
    self.assertEqual(result_json["run_environment"]["hostname"], "test_host")
    self.assertEqual(result_json["run_environment"]["bns"], "/bns/test/address")
    self.assertEqual(result_json["run_environment"]["xid"], "12345")
    self.assertNotIn("run_command", result_json["run_environment"])

    # Test with command line
    result_with_cmd = get_overview_tool.get_overview(
        "session_123", include_command=True
    )
    result_with_cmd_json = json.loads(result_with_cmd)
    self.assertEqual(
        result_with_cmd_json["run_environment"]["run_command"],
        "test command --streamz_default_root_labels=xmanager:int:12345",
    )
    self.assertEqual(
        result_with_cmd_json["run_environment"]["hostname"], "test_host"
    )
    self.assertEqual(
        result_with_cmd_json["run_environment"]["bns"], "/bns/test/address"
    )
    self.assertEqual(result_with_cmd_json["run_environment"]["xid"], "12345")

  def test_get_overview_error(self):
    self.mock_client.fetch.side_effect = Exception("RPC Fail")

    with self.assertRaises(RuntimeError) as cm:
      get_overview_tool.get_overview("session_123")
    self.assertIn("RPC Fail", str(cm.exception))

  def test_get_overview_roofline_fallback(self):
    overview_data = [{
        "p": {
            "flop_rate_utilization_relative_to_roofline": "0.0%",
            "memory_bw_utilization_relative_to_hw_limit": "0.0%",
            "device_type": "TPU v6 Lite",
        }
    }]
    roofline_raw_data = [{
        "cols": [
            {"id": "category", "type": "string"},
            {"id": "roofline_efficiency", "type": "number"},
            {"id": "compute_efficiency", "type": "number"},
            {"id": "max_mem_bw_utilization", "type": "number"},
            {"id": "bound_by", "type": "string"},
            {"id": "operational_intensity", "type": "number"},
        ],
        "rows": [{
            "c": [
                {"v": "Program"},
                {"v": 0.1349},
                {"v": 0.0076},
                {"v": 0.1349},
                {"v": "HBM"},
                {"v": 25.89},
            ]
        }],
    }]

    def fetch_side_effect(tool_name, session_id, **kwargs):
      del session_id, kwargs
      if tool_name in ("overview_page", "overview_page.json"):
        return (81, json.dumps(overview_data).encode("utf-8"))
      if tool_name in ("roofline_model", "roofline_model.json"):
        return (
            "application/json",
            json.dumps(roofline_raw_data).encode("utf-8"),
        )
      return (None, None)

    self.mock_client.fetch.side_effect = fetch_side_effect

    result = get_overview_tool.get_overview("session_123")
    result_json = json.loads(result)

    perf = result_json["performance_summary"]
    self.assertEqual(
        perf["flop_rate_utilization_relative_to_roofline"], "13.49%"
    )
    self.assertEqual(perf["roofline_efficiency_percent"], "13.49%")
    self.assertEqual(perf["compute_efficiency_percent"], "0.76%")
    self.assertEqual(
        perf["memory_bw_utilization_relative_to_hw_limit"], "13.49%"
    )
    self.assertEqual(perf["bound_by"], "HBM")
    self.assertEqual(perf["operational_intensity_flop_per_byte"], 25.89)


if __name__ == "__main__":
  absltest.main()
