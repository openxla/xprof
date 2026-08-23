"""Unit tests for check_host_boundness_tool."""

import json
from unittest import mock
from absl.testing import absltest
from xprof.cli.internal import decorators
from xprof.cli.internal.oss import xprof_client
from xprof.cli.tools import check_host_boundness_tool
from xprof.cli.tools import get_utilization_viewer_tool


class CheckHostBoundnessToolTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    mock_cache = mock.create_autospec(decorators.Cache, instance=True)
    mock_cache.get.return_value = decorators.Cache.UNKNOWN
    self.enter_context(
        mock.patch.object(decorators, "get_cache", return_value=mock_cache)
    )
    self.mock_client = mock.create_autospec(
        xprof_client.CachedXprofClient, instance=True
    )
    self.enter_context(
        mock.patch.object(
            xprof_client, "get_client", return_value=self.mock_client
        )
    )

  def test_insufficient_data_when_zero_duration(self):
    zero_duration_overview = [
        {
            "p": {
                "device_duty_cycle_percent": "0.0%",
                "device_idle_time_percent": "100.0%",
            }
        },
        {"p": {"steptime_ms_average": "0.0"}, "rows": []},
        {"p": {"device_core_count": "8"}},
    ]
    self.mock_client.fetch.side_effect = [
        ("application/json", json.dumps(zero_duration_overview)),
    ]

    result = json.loads(
        check_host_boundness_tool.check_host_boundness("test-session")
    )
    self.assertEqual(result["status"], "INSUFFICIENT_DATA")
    self.assertIn(
        "lacks valid step timing duration telemetry", result["reasons"][0]
    )

  def test_unknown_by_missing_hlo_when_duty_cycle_high(self):
    high_dc_overview = [
        {"p": {"device_duty_cycle_percent": "98.3%"}},
        {"p": {"steptime_ms_average": "100.0"}, "rows": [{}] * 10},
        {"p": {"device_core_count": "8", "host_count": "1"}},
    ]
    # overview returns high dc, hlo returns empty
    self.mock_client.fetch.side_effect = [
        ("application/json", json.dumps(high_dc_overview)),  # overview
        ("application/json", b"{}"),  # hlo
        ("application/json", b"{}"),  # trace
    ]
    self.mock_client.get_hosts.return_value = []

    with mock.patch.object(
        get_utilization_viewer_tool,
        "get_utilization_viewer",
        return_value=json.dumps({
            "idleness_percent": 90.0,
            "hbm_bandwidth_utilization_percent": 10.0,
            "ici_read_utilization_percent": 5.0,
            "ici_write_utilization_percent": 5.0,
        }),
    ):
      result = json.loads(
          check_host_boundness_tool.check_host_boundness("test-session")
      )

    self.assertEqual(result["status"], "UNKNOWN")
    self.assertEqual(result["metrics"]["tpu_duty_cycle_percent"], 98.3)
    self.assertIn("TPU duty cycle is high", result["reasons"][0])

  def test_host_bound_success_all_thresholds_met(self):
    low_dc_overview = [
        {
            "p": {
                "device_duty_cycle_percent": "15.0%",
                "device_idle_time_percent": "85.0%",
            }
        },
        {"p": {"steptime_ms_average": "100.0"}, "rows": [{}] * 10},
        {"p": {"device_core_count": "8", "host_count": "1"}},
    ]
    hlo_data = {
        "byCategory": {
            "children": [{
                "name": "matmul",
                "metrics": {"occurrences": 1, "rawTime": 3000.0},  # 3000 ps
            }]
        }
    }
    dummy_trace = {
        "traceEvents": [
            {"name": "barrier-cores", "dur": 50000.0},  # 50 ms
        ]
    }

    self.mock_client.get_hosts.return_value = [
        {"hostname": "test-host", "hasDeviceTrace": True}
    ]
    self.mock_client.fetch.side_effect = [
        ("application/json", json.dumps(low_dc_overview)),  # overview
        ("application/json", json.dumps(hlo_data)),  # hlo
        ("application/json", json.dumps(dummy_trace)),  # trace
    ]

    with mock.patch.object(
        get_utilization_viewer_tool,
        "get_utilization_viewer",
        return_value=json.dumps({
            "idleness_percent": 99.52,
            "hbm_bandwidth_utilization_percent": 2.98,
            "ici_read_utilization_percent": 16.88,
            "ici_write_utilization_percent": 16.88,
        }),
    ):
      result = json.loads(
          check_host_boundness_tool.check_host_boundness("test-session")
      )

    self.assertEqual(result["status"], "HOST_BOUND")
    self.assertEqual(result["metrics"]["idle_time_ratio_percent"], 100.0)
    self.assertEqual(result["metrics"]["equivalent_idle_chips"], 4.0)
    self.assertEqual(result["metrics"]["mxu_idleness_percent"], 99.52)
    self.assertEqual(
        result["metrics"]["hbm_bandwidth_utilization_percent"], 2.98
    )
    self.assertEqual(result["metrics"]["ici_read_utilization_percent"], 16.88)
    self.assertEqual(result["metrics"]["scaled_barrier_time_ms"], 500.0)
    self.assertIn("Workload is host-bound", result["reasons"][0])
    self.assertIn(
        "Hardware waste = 4.0 idle chips", result["recommendations"][0]
    )

  def test_host_bound_despite_high_duty_cycle(self):
    high_dc_overview = [
        {"p": {"device_duty_cycle_percent": "90.0%"}},
        {"p": {"steptime_ms_average": "100.0"}, "rows": [{}] * 10},
        {"p": {"device_core_count": "8", "host_count": "1"}},
    ]
    hlo_data = {
        "byCategory": {
            "children": [{
                "name": "matmul",
                "metrics": {"occurrences": 1, "rawTime": 3000.0},
            }]
        }
    }
    dummy_trace = {
        "traceEvents": [
            {"name": "barrier-cores", "dur": 50000.0},
        ]
    }

    self.mock_client.get_hosts.return_value = [
        {"hostname": "test-host", "hasDeviceTrace": True}
    ]
    self.mock_client.fetch.side_effect = [
        ("application/json", json.dumps(high_dc_overview)),  # overview
        ("application/json", json.dumps(hlo_data)),  # hlo
        ("application/json", json.dumps(dummy_trace)),  # trace
    ]

    with mock.patch.object(
        get_utilization_viewer_tool,
        "get_utilization_viewer",
        return_value=json.dumps({
            "idleness_percent": 99.52,
            "hbm_bandwidth_utilization_percent": 2.98,
            "ici_read_utilization_percent": 16.88,
            "ici_write_utilization_percent": 16.88,
        }),
    ):
      result = json.loads(
          check_host_boundness_tool.check_host_boundness("test-session")
      )

    self.assertEqual(result["status"], "HOST_BOUND")
    self.assertEqual(result["metrics"]["tpu_duty_cycle_percent"], 90.0)
    self.assertEqual(result["metrics"]["idle_time_ratio_percent"], 100.0)
    self.assertEqual(result["metrics"]["equivalent_idle_chips"], 4.0)

  def test_not_host_bound_by_low_idle_ratio(self):
    high_dc_overview = [
        {"p": {"device_duty_cycle_percent": "90.0%"}},
        {"p": {"steptime_ms_average": "100.0"}, "rows": [{}] * 10},
        {"p": {"device_core_count": "8", "host_count": "1"}},
    ]
    hlo_low_idle = {
        "byCategory": {
            "children": [{
                "name": "matmul",
                "metrics": {
                    "occurrences": 1,
                    "rawTime": (
                        7360000000000.0
                    ),  # 7.36s / 8 cores = 920ms (8% idle)
                },
            }]
        }
    }
    self.mock_client.get_hosts.return_value = []
    self.mock_client.fetch.side_effect = [
        ("application/json", json.dumps(high_dc_overview)),
        ("application/json", json.dumps(hlo_low_idle)),
        ("application/json", b"{}"),
    ]

    with mock.patch.object(
        get_utilization_viewer_tool,
        "get_utilization_viewer",
        return_value=json.dumps({
            "idleness_percent": 10.0,
            "hbm_bandwidth_utilization_percent": 50.0,
            "ici_read_utilization_percent": 10.0,
            "ici_write_utilization_percent": 10.0,
        }),
    ):
      result = json.loads(
          check_host_boundness_tool.check_host_boundness("test-session")
      )

    self.assertEqual(result["status"], "NOT_HOST_BOUND")
    self.assertEqual(result["metrics"]["tpu_duty_cycle_percent"], 90.0)
    self.assertEqual(result["metrics"]["idle_time_ratio_percent"], 8.7)
    self.assertIn(
        "TPU duty cycle (90.0%) is high and idle ratio is low.",
        result["reasons"][0],
    )

  def test_not_host_bound_hbm_bottleneck(self):
    low_dc_overview = [
        {"p": {"device_duty_cycle_percent": "15.0%"}},
        {"p": {"steptime_ms_average": "100.0"}, "rows": [{}] * 10},
        {"p": {"device_core_count": "8", "host_count": "1"}},
    ]
    hlo_data = {
        "byCategory": {
            "children": [{
                "name": "matmul",
                "metrics": {"occurrences": 1, "rawTime": 3000.0},
            }]
        }
    }
    self.mock_client.get_hosts.return_value = []
    self.mock_client.fetch.side_effect = [
        ("application/json", json.dumps(low_dc_overview)),
        ("application/json", json.dumps(hlo_data)),
        ("application/json", b"{}"),
    ]

    with mock.patch.object(
        get_utilization_viewer_tool,
        "get_utilization_viewer",
        return_value=json.dumps({
            "idleness_percent": 90.0,
            "hbm_bandwidth_utilization_percent": 65.0,  # >= 30% HBM bottleneck
            "ici_read_utilization_percent": 10.0,
            "ici_write_utilization_percent": 10.0,
        }),
    ):
      result = json.loads(
          check_host_boundness_tool.check_host_boundness("test-session")
      )

    self.assertEqual(result["status"], "NOT_HOST_BOUND")
    self.assertIn(
        "HBM Bandwidth Utilization (65.0%) is >= 30.0%", result["reasons"][1]
    )

  def test_not_host_bound_ici_bottleneck(self):
    low_dc_overview = [
        {"p": {"device_duty_cycle_percent": "15.0%"}},
        {"p": {"steptime_ms_average": "100.0"}, "rows": [{}] * 10},
        {"p": {"device_core_count": "8", "host_count": "1"}},
    ]
    hlo_data = {
        "byCategory": {
            "children": [{
                "name": "matmul",
                "metrics": {"occurrences": 1, "rawTime": 3000.0},
            }]
        }
    }
    self.mock_client.get_hosts.return_value = []
    self.mock_client.fetch.side_effect = [
        ("application/json", json.dumps(low_dc_overview)),
        ("application/json", json.dumps(hlo_data)),
        ("application/json", b"{}"),
    ]

    with mock.patch.object(
        get_utilization_viewer_tool,
        "get_utilization_viewer",
        return_value=json.dumps({
            "idleness_percent": 90.0,
            "hbm_bandwidth_utilization_percent": 15.0,
            "ici_read_utilization_percent": 45.0,  # >= 30% ICI bottleneck
            "ici_write_utilization_percent": 10.0,
        }),
    ):
      result = json.loads(
          check_host_boundness_tool.check_host_boundness("test-session")
      )

    self.assertEqual(result["status"], "NOT_HOST_BOUND")
    self.assertIn(
        "ICI Utilization (Read: 45.0%, Write: 10.0%) is >= 30.0%",
        result["reasons"][1],
    )

  def test_multi_host_utilization_fallback(self):
    low_dc_overview = [
        {"p": {"device_duty_cycle_percent": "15.0%"}},
        {"p": {"steptime_ms_average": "100.0"}, "rows": [{}] * 10},
        {"p": {"device_core_count": "8", "host_count": "2"}},
    ]
    hlo_data = {
        "byCategory": {
            "children": [{
                "name": "matmul",
                "metrics": {"occurrences": 1, "rawTime": 3000.0},
            }]
        }
    }
    self.mock_client.get_hosts.return_value = []
    self.mock_client.fetch.side_effect = [
        ("application/json", json.dumps(low_dc_overview)),
        ("application/json", json.dumps(hlo_data)),
        ("application/json", b"{}"),
    ]

    def mock_util_side_effect(session_id, host=0, **kwargs):
      del session_id, kwargs
      if host == 0:
        return json.dumps({"message": "No data found for Host 0"})
      return json.dumps({
          "idleness_percent": 95.0,
          "hbm_bandwidth_utilization_percent": 10.0,
          "ici_read_utilization_percent": 10.0,
          "ici_write_utilization_percent": 10.0,
      })

    with mock.patch.object(
        get_utilization_viewer_tool,
        "get_utilization_viewer",
        side_effect=mock_util_side_effect,
    ):
      result = json.loads(
          check_host_boundness_tool.check_host_boundness("test-session")
      )

    self.assertEqual(result["status"], "HOST_BOUND")
    self.assertEqual(result["metrics"]["mxu_idleness_percent"], 95.0)

  def test_error_handling_when_overview_empty(self):
    self.mock_client.fetch.return_value = ("application/json", b"")
    result = json.loads(
        check_host_boundness_tool.check_host_boundness("sess_missing")
    )
    self.assertEqual(result["status"], "UNKNOWN")
    self.assertIn("error", result)

  def test_func_name_argument_support(self):
    low_dc_overview = [
        {"p": {"device_duty_cycle_percent": "15.0%"}},
        {"p": {"steptime_ms_average": "100.0"}, "rows": [{}] * 10},
        {"p": {"device_core_count": "8", "host_count": "1"}},
    ]
    hlo_data = {
        "byCategory": {
            "children": [{
                "name": "matmul",
                "metrics": {"occurrences": 1, "rawTime": 3000.0},
            }]
        }
    }
    self.mock_client.get_hosts.return_value = []
    self.mock_client.fetch.side_effect = [
        ("application/json", json.dumps(low_dc_overview)),
        ("application/json", json.dumps(hlo_data)),
        ("application/json", b"{}"),
    ]

    with mock.patch.object(
        get_utilization_viewer_tool,
        "get_utilization_viewer",
        return_value=json.dumps({
            "idleness_percent": 99.0,
            "hbm_bandwidth_utilization_percent": 5.0,
            "ici_read_utilization_percent": 5.0,
            "ici_write_utilization_percent": 5.0,
        }),
    ):
      result = json.loads(
          check_host_boundness_tool.check_host_boundness(
              "test-session", func_name="train_step"
          )
      )

    self.assertEqual(result["status"], "HOST_BOUND")
    self.assertIn(
        "Use Lumini xprof_check_host_boundness with func_name",
        result["recommendations"][2],
    )

  def test_parse_json_safely_comprehensive(self):
    # Empty inputs
    self.assertEqual(check_host_boundness_tool._parse_json_safely(""), {})
    self.assertEqual(check_host_boundness_tool._parse_json_safely(None), {})
    self.assertEqual(check_host_boundness_tool._parse_json_safely([]), {})
    self.assertEqual(
        check_host_boundness_tool._parse_json_safely([1, 2, 3]), {}
    )

    # Valid dict and list of dicts
    self.assertEqual(
        check_host_boundness_tool._parse_json_safely('{"a": 1}'), {"a": 1}
    )
    self.assertEqual(
        check_host_boundness_tool._parse_json_safely('[{"a": 1}]'), {"a": 1}
    )
    self.assertEqual(
        check_host_boundness_tool._parse_json_safely({"a": 1}), {"a": 1}
    )
    self.assertEqual(
        check_host_boundness_tool._parse_json_safely([{"a": 1}]), {"a": 1}
    )

    # Bytes input
    self.assertEqual(
        check_host_boundness_tool._parse_json_safely(b'{"a": 1}'), {"a": 1}
    )

    # Malformed JSON
    self.assertEqual(
        check_host_boundness_tool._parse_json_safely("invalid json"), {}
    )

  def test_traverse_and_sum_hlo_times_comprehensive(self):
    # Non-dict node
    self.assertEqual(
        check_host_boundness_tool._traverse_and_sum_hlo_times(None),
        (0.0, 0.0, 0.0),
    )
    self.assertEqual(
        check_host_boundness_tool._traverse_and_sum_hlo_times({}),
        (0.0, 0.0, 0.0),
    )

    # Node with compute, HBM, and ICI ops
    tree = {
        "children": [
            {
                "name": "all-reduce",
                "metrics": {"occurrences": 1, "rawTime": 500.0},
            },
            {
                "name": "copy-start",
                "metrics": {"occurrences": 1, "rawTime": 300.0},
            },
            {
                "name": "custom-call",
                "xla": {"category": "convolution"},
                "metrics": {"occurrences": 2, "rawTime": 1200.0},
            },
            {
                "name": "zero-occurrences-op",
                "metrics": {"occurrences": 0, "rawTime": 9999.0},
            },
        ]
    }
    c_time, m_time, i_time = (
        check_host_boundness_tool._traverse_and_sum_hlo_times(tree)
    )
    self.assertEqual(c_time, 1200.0)
    self.assertEqual(m_time, 300.0)
    self.assertEqual(i_time, 500.0)

  def test_dict_overview_format_success(self):
    dict_overview = {
        "performance_summary": {
            "device_duty_cycle_percent": "15.0%",
        },
        "step_time": {
            "steptime_ms_average": "100.0",
        },
        "run_environment": {
            "device_core_count": "8",
            "host_count": "1",
        },
        "rows": [{}] * 10,
    }
    hlo_data = {
        "byCategory": {
            "children": [{
                "name": "matmul",
                "metrics": {"occurrences": 1, "rawTime": 3000.0},
            }]
        }
    }
    self.mock_client.get_hosts.return_value = ["host-0"]
    self.mock_client.fetch.side_effect = [
        ("application/json", json.dumps(dict_overview)),  # overview
        ("application/json", json.dumps(hlo_data)),  # hlo
        ("application/json", b"{}"),  # trace
    ]

    with mock.patch.object(
        get_utilization_viewer_tool,
        "get_utilization_viewer",
        return_value=json.dumps({
            "idleness_percent": 99.52,
            "hbm_bandwidth_utilization_percent": 2.98,
            "ici_read_utilization_percent": 16.88,
            "ici_write_utilization_percent": 16.88,
        }),
    ):
      result = json.loads(
          check_host_boundness_tool.check_host_boundness("test-session")
      )

    self.assertEqual(result["status"], "HOST_BOUND")
    self.assertEqual(result["metrics"]["tpu_duty_cycle_percent"], 15.0)
    self.assertEqual(result["metrics"]["mxu_idleness_percent"], 99.52)

  def test_utilization_with_none_and_missing_values(self):
    low_dc_overview = [
        {"p": {"device_duty_cycle_percent": "15.0%"}},
        {"p": {"steptime_ms_average": "100.0"}, "rows": [{}] * 10},
        {"p": {"device_core_count": "0", "host_count": "0"}},  # 0 core count
    ]
    hlo_data = {
        "byCategory": {
            "children": [{
                "name": "matmul",
                "metrics": {"occurrences": 1, "rawTime": 3000.0},
            }]
        }
    }
    self.mock_client.get_hosts.return_value = []
    self.mock_client.fetch.side_effect = [
        ("application/json", json.dumps(low_dc_overview)),
        ("application/json", json.dumps(hlo_data)),
        ("application/json", b"{}"),
    ]

    with mock.patch.object(
        get_utilization_viewer_tool,
        "get_utilization_viewer",
        return_value=json.dumps({
            "idleness_percent": None,
            "hbm_bandwidth_utilization_percent": None,
            "ici_read_utilization_percent": None,
            "ici_write_utilization_percent": None,
        }),
    ):
      result = json.loads(
          check_host_boundness_tool.check_host_boundness("test-session")
      )

    # Because idleness_percent is 0.0 (< 70.0%), it is NOT_HOST_BOUND
    self.assertEqual(result["status"], "NOT_HOST_BOUND")
    self.assertEqual(result["metrics"]["core_count"], 1)  # Fallback to 1

  def test_get_utilization_metrics_short_circuits_on_error(self):
    with mock.patch.object(
        get_utilization_viewer_tool,
        "get_utilization_viewer",
        return_value=json.dumps({"error": "RPC backend error"}),
    ) as mock_get_util:
      metrics = check_host_boundness_tool._get_utilization_metrics(
          "test-session", host_count=16
      )
      self.assertEqual(mock_get_util.call_count, 1)
      self.assertEqual(metrics["idleness_percent"], 0.0)

  def test_get_utilization_metrics_short_circuits_on_session_no_data(self):
    with mock.patch.object(
        get_utilization_viewer_tool,
        "get_utilization_viewer",
        return_value=json.dumps({
            "status": "NO_DATA",
            "message": "No data returned for session test-session",
        }),
    ) as mock_get_util:
      metrics = check_host_boundness_tool._get_utilization_metrics(
          "test-session", host_count=16
      )
      self.assertEqual(mock_get_util.call_count, 1)
      self.assertEqual(metrics["idleness_percent"], 0.0)

  def test_get_utilization_metrics_falls_back_on_host_filter_miss(self):
    with mock.patch.object(
        get_utilization_viewer_tool,
        "get_utilization_viewer",
        side_effect=[
            json.dumps({
                "status": "NO_DATA",
                "message": "No data found for Host 0 Device 0 Node 0",
            }),
            json.dumps({
                "idleness_percent": 85.0,
                "hbm_bandwidth_utilization_percent": 12.0,
                "ici_read_utilization_percent": 4.0,
                "ici_write_utilization_percent": 4.0,
            }),
        ],
    ) as mock_get_util:
      metrics = check_host_boundness_tool._get_utilization_metrics(
          "test-session", host_count=16
      )
      self.assertEqual(mock_get_util.call_count, 2)
      self.assertEqual(metrics["idleness_percent"], 85.0)
      self.assertEqual(metrics["hbm_bandwidth_utilization_percent"], 12.0)


if __name__ == "__main__":
  absltest.main()
