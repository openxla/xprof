import json
import pathlib
from unittest import mock
from absl.testing import absltest
from xprof.cli.internal import decorators
from xprof.cli.internal.oss import xprof_client
from xprof.cli.tools import get_overview_tool
from xprof.protobuf import op_stats_pb2


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

  def test_get_overview_custom_call_warning_with_nonzero_flop(self):
    overview_data = [{
        "p": {
            "flop_rate_utilization_relative_to_roofline": "20.0%",
            "memory_bw_utilization_relative_to_hw_limit": "30.0%",
            "device_type": "TPU v6 Lite",
        }
    }]
    roofline_raw_data = [{
        "cols": [
            {"id": "rank", "type": "number"},
            {"id": "operation", "type": "string"},
            {"id": "category", "type": "string"},
            {"id": "total_self_time_percent", "type": "number"},
        ],
        "rows": [
            {"c": [{"v": 0}, {"v": "Program"}, {"v": "Program"}, {"v": 1.0}]},
            {
                "c": [
                    {"v": 1},
                    {"v": "custom-call.1"},
                    {"v": "custom-call"},
                    {"v": 0.45},
                ]
            },
            {
                "c": [
                    {"v": 1},
                    {"v": "custom-call.1"},
                    {"v": "custom-call"},
                    {"v": 0.45},
                ]
            },
        ],
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
    result_json = json.loads(get_overview_tool.get_overview("session_cc"))
    perf = result_json["performance_summary"]
    self.assertEqual(perf["custom_call_share_pct"], 45.0)
    self.assertIn("custom_call_warning", perf)
    self.assertEqual(
        perf["flop_rate_utilization_relative_to_roofline"], "20.0%"
    )

  def test_get_overview_multi_host_step_time_with_slower_second_host(self):
    overview_data = [{
        "p": {
            "steptime_ms_average": "22.00",
            "steptime_ms_standard_deviation": "2.83",
            "tc_idle_ms_average": "8.00",
            "flop_rate_utilization_relative_to_roofline": "50.0%",
            "memory_bw_utilization_relative_to_hw_limit": "50.0%",
            "device_type": "TPU v7x",
            "host_count": "2",
        }
    }]
    single_host_overviews = {
        "a_fast_host": [{
            "p": {
                "steptime_ms_average": "12.00",
                "steptime_ms_standard_deviation": "2.83",
                "flop_rate_utilization_relative_to_roofline": "50.0%",
                "memory_bw_utilization_relative_to_hw_limit": "50.0%",
                "device_type": "TPU v7x",
                "host_count": "1",
            }
        }],
        "z_slow_host": [{
            "p": {
                "steptime_ms_average": "22.00",
                "steptime_ms_standard_deviation": "2.83",
                "flop_rate_utilization_relative_to_roofline": "50.0%",
                "memory_bw_utilization_relative_to_hw_limit": "50.0%",
                "device_type": "TPU v7x",
                "host_count": "1",
            }
        }],
    }
    pod_viewer_data = {
        "podStatsSequence": {
            "podStatsMap": [
                {
                    "stepNum": 0,
                    "podStatsPerCore": {
                        "0": {
                            "hostName": "a_fast_host",
                            "stepNum": 0,
                            "totalDurationUs": 9500.0,
                        },
                        "1": {
                            "hostName": "a_fast_host",
                            "stepNum": 0,
                            "totalDurationUs": 10000.0,
                        },
                        "1000": {
                            "hostName": "z_slow_host",
                            "stepNum": 0,
                            "totalDurationUs": 19000.0,
                        },
                        "1001": {
                            "hostName": "z_slow_host",
                            "stepNum": 0,
                            "totalDurationUs": 20000.0,
                        },
                    },
                },
                {
                    "stepNum": 1,
                    "podStatsPerCore": {
                        "0": {
                            "hostName": "a_fast_host",
                            "stepNum": 1,
                            "totalDurationUs": 14000.0,
                        },
                        "1": {
                            "hostName": "a_fast_host",
                            "stepNum": 1,
                            "totalDurationUs": 13500.0,
                        },
                        "1000": {
                            "hostName": "z_slow_host",
                            "stepNum": 1,
                            "totalDurationUs": 24000.0,
                        },
                        "1001": {
                            "hostName": "z_slow_host",
                            "stepNum": 1,
                            "totalDurationUs": 23000.0,
                        },
                    },
                },
            ]
        }
    }

    def fetch_side_effect(tool_name, session_id, **kwargs):
      del kwargs
      if tool_name in ("overview_page", "overview_page.json"):
        if session_id in single_host_overviews:
          return (
              81,
              json.dumps(single_host_overviews[session_id]).encode("utf-8"),
          )
        return (81, json.dumps(overview_data).encode("utf-8"))
      if tool_name in ("pod_viewer", "pod_viewer.json"):
        return (
            "application/json",
            json.dumps(pod_viewer_data).encode("utf-8"),
        )
      return (None, None)

    self.mock_client.fetch.side_effect = fetch_side_effect
    result_json = json.loads(get_overview_tool.get_overview("remote_multi"))
    perf = result_json["performance_summary"]
    self.assertEqual(perf["steptime_ms_average"], "22.00")
    self.assertEqual(perf["step_time_scope"], "per_step_max_across_hosts")
    self.assertIn("steptime_ms_average", perf["step_time_warning"])
    self.assertIn("steptime_ms_standard_deviation", perf["step_time_warning"])
    self.assertIn("tc_idle_ms_average", perf["step_time_warning"])
    self.assertNotRegex(
        perf["step_time_warning"], r"(?<!per_host_)\bstep_time_ms\b"
    )
    self.assertNotIn("steptime_ms_minimum", perf["step_time_warning"])
    self.assertEqual(
        perf["per_host_step_time_ms"],
        {
            "a_fast_host": {
                "steptime_ms_average": "12.00",
                "steptime_ms_standard_deviation": "2.83",
                "steptime_ms_minimum": "10.00",
                "steptime_ms_maximum": "14.00",
            },
            "z_slow_host": {
                "steptime_ms_average": "22.00",
                "steptime_ms_standard_deviation": "2.83",
                "steptime_ms_minimum": "20.00",
                "steptime_ms_maximum": "24.00",
            },
        },
    )
    for host_name in ("a_fast_host", "z_slow_host"):
      host_perf = json.loads(get_overview_tool.get_overview(host_name))[
          "performance_summary"
      ]
      self.assertEqual(
          perf["per_host_step_time_ms"][host_name]["steptime_ms_average"],
          host_perf["steptime_ms_average"],
      )
      self.assertEqual(
          perf["per_host_step_time_ms"][host_name][
              "steptime_ms_standard_deviation"
          ],
          host_perf["steptime_ms_standard_deviation"],
      )

  def test_get_overview_local_multi_trace_uses_trace_stems(self):
    overview_data = [{
        "p": {
            "steptime_ms_average": "25.00",
            "steptime_ms_standard_deviation": "0.00",
            "flop_rate_utilization_relative_to_roofline": "50.0%",
            "memory_bw_utilization_relative_to_hw_limit": "50.0%",
            "device_type": "TPU v7x",
            "host_count": "1",
        }
    }]
    pod_viewer_data = {
        "podStatsSequence": {
            "podStatsMap": [{
                "stepNum": 0,
                "podStatsPerCore": {
                    "0": {
                        "hostName": "shared_node",
                        "stepNum": 0,
                        "totalDurationUs": 15000.0,
                    },
                    "1000": {
                        "hostName": "shared_node",
                        "stepNum": 0,
                        "totalDurationUs": 25000.0,
                    },
                },
            }]
        }
    }

    def fetch_side_effect(tool_name, session_id, **kwargs):
      del session_id, kwargs
      if tool_name in ("overview_page", "overview_page.json"):
        return (81, json.dumps(overview_data).encode("utf-8"))
      if tool_name in ("pod_viewer", "pod_viewer.json"):
        return (
            "application/json",
            json.dumps(pod_viewer_data).encode("utf-8"),
        )
      return (None, None)

    self.mock_client.is_local_session.return_value = True
    self.mock_client.select_paths.return_value = [
        "/tmp/run/a_rank0_node0.xplane.pb",
        "/tmp/run/z_rank2_node0.xplane.pb",
    ]
    self.mock_client.fetch.side_effect = fetch_side_effect
    result_json = json.loads(get_overview_tool.get_overview("/tmp/run"))
    perf = result_json["performance_summary"]
    self.assertEqual(perf["step_time_scope"], "per_step_max_across_hosts")
    self.assertEqual(
        perf["per_host_step_time_ms"]["a_rank0_node0"]["steptime_ms_average"],
        "15.00",
    )
    self.assertEqual(
        perf["per_host_step_time_ms"]["z_rank2_node0"]["steptime_ms_average"],
        "25.00",
    )

  def test_get_overview_local_multi_trace_reads_all_hosts_op_stats_pb(self):
    run_dir = pathlib.Path(self.create_tempdir().full_path)
    op_stats = op_stats_pb2.OpStats()
    op_stats.run_environment.host_count = 2
    op_stats.core_id_to_details[0].hostname = "node0"
    op_stats.core_id_to_details[1000].hostname = "node1"
    op_stats.core_id_to_details[1_000_000].hostname = "node0"
    seq0 = op_stats.step_db.step_sequence.add()
    seq0.step_num = 1
    seq0.step_info_per_core[0].duration_ps = 10_000_000_000
    seq0.step_info_per_core[1000].duration_ps = 20_000_000_000
    seq0.step_info_per_core[1_000_000].duration_ps = 99_000_000_000
    seq1 = op_stats.step_db.step_sequence.add()
    seq1.step_num = 2
    seq1.step_info_per_core[0].duration_ps = 14_000_000_000
    seq1.step_info_per_core[1000].duration_ps = 24_000_000_000
    seq1.step_info_per_core[1_000_000].duration_ps = 99_000_000_000
    op_stats_path = run_dir / "ALL_HOSTS.op_stats_v2.pb"
    op_stats_path.write_bytes(op_stats.SerializeToString())

    overview_data = [{
        "p": {
            "steptime_ms_average": "22.00",
            "steptime_ms_standard_deviation": "2.83",
            "flop_rate_utilization_relative_to_roofline": "50.0%",
            "memory_bw_utilization_relative_to_hw_limit": "50.0%",
            "device_type": "TPU v7x",
            "host_count": "2",
        }
    }]
    self.mock_client.is_local_session.return_value = True
    self.mock_client.select_paths.return_value = [
        str(run_dir / "host0.xplane.pb"),
        str(run_dir / "host1.xplane.pb"),
    ]
    self.mock_client.get_owned_op_stats_path.return_value = op_stats_path
    self.mock_client.fetch.return_value = (
        81,
        json.dumps(overview_data).encode("utf-8"),
    )

    result_json = json.loads(get_overview_tool.get_overview(str(run_dir)))
    perf = result_json["performance_summary"]
    self.assertEqual(perf["step_time_scope"], "per_step_max_across_hosts")
    self.assertEqual(
        perf["per_host_step_time_ms"]["host0"],
        {
            "steptime_ms_average": "12.00",
            "steptime_ms_standard_deviation": "2.83",
            "steptime_ms_minimum": "10.00",
            "steptime_ms_maximum": "14.00",
        },
    )
    self.assertEqual(
        perf["per_host_step_time_ms"]["host1"],
        {
            "steptime_ms_average": "22.00",
            "steptime_ms_standard_deviation": "2.83",
            "steptime_ms_minimum": "20.00",
            "steptime_ms_maximum": "24.00",
        },
    )

  def test_get_overview_rejects_unowned_or_mismatched_op_stats_file(self):
    run_dir = pathlib.Path(self.create_tempdir().full_path)
    single_host_op_stats = op_stats_pb2.OpStats()
    single_host_op_stats.run_environment.host_count = 1
    single_host_op_stats.core_id_to_details[0].hostname = "node0"
    seq = single_host_op_stats.step_db.step_sequence.add()
    seq.step_num = 1
    seq.step_info_per_core[0].duration_ps = 99_000_000_000
    op_stats_path = run_dir / "ALL_HOSTS.op_stats_v2.pb"
    op_stats_path.write_bytes(single_host_op_stats.SerializeToString())

    overview_data = [{
        "p": {
            "steptime_ms_average": "25.00",
            "steptime_ms_standard_deviation": "0.00",
            "flop_rate_utilization_relative_to_roofline": "50.0%",
            "memory_bw_utilization_relative_to_hw_limit": "50.0%",
            "device_type": "TPU v7x",
            "host_count": "2",
        }
    }]
    pod_viewer_data = {
        "podStatsSequence": {
            "podStatsMap": [{
                "stepNum": 0,
                "podStatsPerCore": {
                    "0": {
                        "hostName": "node0",
                        "stepNum": 0,
                        "totalDurationUs": 15000.0,
                    },
                    "1000": {
                        "hostName": "node1",
                        "stepNum": 0,
                        "totalDurationUs": 25000.0,
                    },
                },
            }]
        }
    }

    def fetch_side_effect(tool_name, session_id, **kwargs):
      del session_id, kwargs
      if tool_name in ("overview_page", "overview_page.json"):
        return (81, json.dumps(overview_data).encode("utf-8"))
      if tool_name in ("pod_viewer", "pod_viewer.json"):
        return (
            "application/json",
            json.dumps(pod_viewer_data).encode("utf-8"),
        )
      return (None, None)

    self.mock_client.is_local_session.return_value = True
    self.mock_client.select_paths.return_value = [
        str(run_dir / "sub0" / "host0.xplane.pb"),
        str(run_dir / "sub1" / "host1.xplane.pb"),
    ]
    self.mock_client.fetch.side_effect = fetch_side_effect

    # Case 1: unowned op_stats file (get_owned_op_stats_path returns None).
    self.mock_client.get_owned_op_stats_path.return_value = None
    result_unowned = json.loads(get_overview_tool.get_overview(str(run_dir)))
    self.assertEqual(
        result_unowned["performance_summary"]["per_host_step_time_ms"]["host0"][
            "steptime_ms_average"
        ],
        "15.00",
    )

    # Case 2: op_stats file holds 1 host while 2 local traces were selected.
    self.mock_client.get_owned_op_stats_path.return_value = op_stats_path
    result_mismatch = json.loads(
        get_overview_tool.get_overview(str(run_dir), bypass_cache=True)
    )
    self.assertEqual(
        result_mismatch["performance_summary"]["per_host_step_time_ms"][
            "host0"
        ]["steptime_ms_average"],
        "15.00",
    )
    self.assertEqual(
        result_mismatch["performance_summary"]["per_host_step_time_ms"][
            "host1"
        ]["steptime_ms_average"],
        "25.00",
    )

  def test_get_overview_straggler_step_time_vs_min_tc_idle(self):
    # Host 0 waits inside a collective on device (30 ms step time, 20 ms idle);
    # Host 1 is a CPU straggler (12 ms device step time, 2 ms device idle).
    # XProf's merged overview pairs the slowest core's step time (30.00 ms) with
    # the minimum-step-duration core's breakdown (tc_idle_ms_average = 2.00 ms).
    overview_data = [{
        "p": {
            "steptime_ms_average": "30.00",
            "steptime_ms_standard_deviation": "0.00",
            "tc_idle_ms_average": "2.00",
            "flop_rate_utilization_relative_to_roofline": "50.0%",
            "memory_bw_utilization_relative_to_hw_limit": "50.0%",
            "device_type": "TPU v7x",
            "host_count": "2",
        }
    }]
    pod_viewer_data = {
        "podStatsSequence": {
            "podStatsMap": [{
                "stepNum": 0,
                "podStatsPerCore": {
                    "0": {
                        "hostName": "host0_waiting_on_collective",
                        "stepNum": 0,
                        "totalDurationUs": 30000.0,
                    },
                    "1000": {
                        "hostName": "host1_cpu_straggler",
                        "stepNum": 0,
                        "totalDurationUs": 12000.0,
                    },
                },
            }]
        }
    }

    def fetch_side_effect(tool_name, session_id, **kwargs):
      del session_id, kwargs
      if tool_name in ("overview_page", "overview_page.json"):
        return (81, json.dumps(overview_data).encode("utf-8"))
      if tool_name in ("pod_viewer", "pod_viewer.json"):
        return (
            "application/json",
            json.dumps(pod_viewer_data).encode("utf-8"),
        )
      return (None, None)

    self.mock_client.fetch.side_effect = fetch_side_effect
    result = json.loads(get_overview_tool.get_overview("straggler_session"))
    perf = result["performance_summary"]
    self.assertEqual(perf["steptime_ms_average"], "30.00")
    self.assertEqual(perf["tc_idle_ms_average"], "2.00")
    self.assertEqual(
        perf["per_host_step_time_ms"]["host0_waiting_on_collective"][
            "steptime_ms_average"
        ],
        "30.00",
    )
    self.assertEqual(
        perf["per_host_step_time_ms"]["host1_cpu_straggler"][
            "steptime_ms_average"
        ],
        "12.00",
    )
    self.assertIn(
        "tc_idle_ms_average use the per-step minimum across all cores",
        perf["step_time_warning"],
    )

  def test_get_overview_local_single_trace_ignores_job_host_count(self):
    overview_data = [{
        "p": {
            "steptime_ms_average": "22.07",
            "steptime_ms_standard_deviation": "4.66",
            "flop_rate_utilization_relative_to_roofline": "50.0%",
            "memory_bw_utilization_relative_to_hw_limit": "50.0%",
            "device_type": "TPU v7x",
            "host_count": "2",
        }
    }]
    self.mock_client.is_local_session.return_value = True
    self.mock_client.select_paths.return_value = [
        "/tmp/run/host0.xplane.pb",
    ]
    self.mock_client.fetch.return_value = (
        81,
        json.dumps(overview_data).encode("utf-8"),
    )

    result_json = json.loads(get_overview_tool.get_overview("/tmp/run"))
    perf = result_json["performance_summary"]
    self.assertNotIn("step_time_scope", perf)
    self.assertNotIn("step_time_warning", perf)
    self.assertNotIn("per_host_step_time_ms", perf)

  def test_get_overview_active_step_time_fields(self):
    """Verifies active step time fields are extracted into performance_summary."""
    overview_data = [
        {
            "p": {
                "device_type": "TPU v5p",
            }
        },
        {
            "p": {
                "steptime_ms_average": "50.00",
                "active_steptime_ms_average": "22.50",
                "tc_active_ms_average": "20.00",
                "sc_active_ms_average": "22.50",
            }
        },
    ]
    self.mock_client.fetch.return_value = (
        81,
        json.dumps(overview_data).encode("utf-8"),
    )
    result_json = json.loads(get_overview_tool.get_overview("session_active"))
    perf = result_json["performance_summary"]
    self.assertEqual(perf["steptime_ms_average"], "50.00")
    self.assertEqual(perf["active_steptime_ms_average"], "22.50")
    self.assertEqual(perf["tc_active_ms_average"], "20.00")
    self.assertEqual(perf["sc_active_ms_average"], "22.50")


if __name__ == "__main__":
  absltest.main()
