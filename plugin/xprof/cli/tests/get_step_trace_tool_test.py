import json
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from xprof.cli.internal import decorators
from xprof.cli.internal.oss import xprof_client
from xprof.cli.tools import get_step_trace_tool


def _fetch_input_pipeline_fallback_side_effect(tool_name, *_args, **_kwargs):
  if tool_name == "pod_viewer.json":
    return (None, b"")
  elif tool_name == "input_pipeline.json":
    input_pipeline_data = [{
        "cols": [
            {"id": "stepnum", "type": "string"},
            {"id": "noninfeedTimeMs", "type": "number"},
            {"id": "infeedTimeMs", "type": "number"},
            {"id": "tooltip", "type": "string"},
            {"id": "infeedPercentAverage", "type": "number"},
        ],
        "rows": [
            {
                "c": [
                    {"v": "1"},
                    {"v": 20.0},
                    {"v": 80.0},
                    {"v": "tooltip 1"},
                    {"v": 80.0},
                ]
            },
            {
                "c": [
                    {"v": "2"},
                    {"v": 30.0},
                    {"v": 70.0},
                    {"v": "tooltip 2"},
                    {"v": 70.0},
                ]
            },
        ],
        "p": {
            "steptime_ms_average": "100.0",
            "steptime_ms_minimum": "100.0",
            "steptime_ms_maximum": "100.0",
            "steptime_ms_standard_deviation": "0.0",
            "infeed_percent_average": "75.0",
            "summary_conclusion": "Program is input-bound",
        },
    }]
    return (None, json.dumps(input_pipeline_data).encode("utf-8"))
  return (None, b"")


def _fetch_overview_page_fallback_side_effect(tool_name, *_args, **_kwargs):
  if tool_name in ("pod_viewer.json", "input_pipeline.json"):
    return (None, b"")
  elif tool_name == "overview_page.json":
    overview_data = [{
        "p": {
            "steptime_ms_average": "50.0",
            "steptime_ms_standard_deviation": "2.5",
            "tc_infeed_ms_average": "5.0",
            "tc_outfeed_ms_average": "1.0",
            "tc_idle_ms_average": "4.0",
        }
    }]
    return (None, json.dumps(overview_data).encode("utf-8"))
  return (None, b"")


class GetStepTraceToolTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    mock_cache = mock.create_autospec(decorators.Cache, instance=True)
    mock_cache.get.return_value = decorators.Cache.UNKNOWN
    self.enter_context(
        mock.patch.object(
            decorators,
            "get_cache",
            return_value=mock_cache,
            autospec=True,
            spec_set=True,
        )
    )
    self.mock_client = mock.create_autospec(
        xprof_client.CachedXprofClient, instance=True
    )
    self.enter_context(
        mock.patch.object(
            xprof_client,
            "get_client",
            return_value=self.mock_client,
            autospec=True,
            spec_set=True,
        )
    )

  def test_get_step_trace_from_pod_viewer_success(self):
    pod_viewer_mock_data = {
        "podStatsSequence": {
            "podStatsMap": [
                {
                    "stepNum": 221,
                    "podStatsPerCore": {
                        "0": {
                            "chipId": 0,
                            "nodeId": 0,
                            "hostName": "host1",
                            "stepNum": 221,
                            "totalDurationUs": 300000.0,
                            "highFlopsComputeUs": 150000.0,
                            "crsDurationUs": 50000.0,
                            "sendDurationUs": 20000.0,
                            "recvDurationUs": 30000.0,
                            "hostInfeedDurationUs": 10000.0,
                            "hostOutfeedDurationUs": 5000.0,
                            "bottleneck": "Send and Recv",
                        },
                        "1": {
                            "chipId": 0,
                            "nodeId": 1,
                            "hostName": "host1",
                            "stepNum": 221,
                            "totalDurationUs": 300000.0,
                            "highFlopsComputeUs": 150000.0,
                            "crsDurationUs": 50000.0,
                            "sendDurationUs": 20000.0,
                            "recvDurationUs": 30000.0,
                            "hostInfeedDurationUs": 10000.0,
                            "hostOutfeedDurationUs": 5000.0,
                            "bottleneck": "Send and Recv",
                        },
                    },
                },
                {
                    "stepNum": 222,
                    "podStatsPerCore": {
                        "0": {
                            "chipId": 0,
                            "nodeId": 0,
                            "hostName": "host1",
                            "stepNum": 222,
                            "totalDurationUs": 400000.0,
                            "highFlopsComputeUs": 200000.0,
                            "crsDurationUs": 60000.0,
                            "sendDurationUs": 30000.0,
                            "recvDurationUs": 40000.0,
                            "hostInfeedDurationUs": 20000.0,
                            "hostOutfeedDurationUs": 10000.0,
                            "bottleneck": "All-Reduce",
                        },
                    },
                },
            ]
        }
    }

    self.mock_client.fetch.return_value = (
        None,
        json.dumps(pod_viewer_mock_data).encode("utf-8"),
    )

    result_json = get_step_trace_tool.get_step_trace("test_session")
    result = json.loads(result_json)

    self.assertNotIn("error", result)
    self.assertIn("summary", result)
    self.assertIn("step_breakdown", result)

    summary = result["summary"]
    self.assertEqual(summary["total_steps"], 2)
    self.assertFalse(summary["is_aggregate"])
    self.assertAlmostEqual(summary["step_time_ms_average"], 350.0, places=2)
    self.assertAlmostEqual(summary["step_time_ms_min"], 300.0, places=2)
    self.assertAlmostEqual(summary["step_time_ms_max"], 400.0, places=2)
    self.assertAlmostEqual(summary["compute_time_ms_average"], 175.0, places=2)
    self.assertAlmostEqual(
        summary["communication_time_ms_average"], 115.0, places=2
    )
    self.assertAlmostEqual(summary["infeed_time_ms_average"], 15.0, places=2)
    self.assertAlmostEqual(summary["outfeed_time_ms_average"], 7.5, places=2)

    step_breakdown = result["step_breakdown"]
    self.assertLen(step_breakdown, 2)
    self.assertEqual(step_breakdown[0]["step_num"], 221)
    self.assertAlmostEqual(step_breakdown[0]["step_time_ms"], 300.0, places=2)
    self.assertAlmostEqual(
        step_breakdown[0]["compute_time_ms"], 150.0, places=2
    )
    self.assertAlmostEqual(
        step_breakdown[0]["communication_time_ms"], 100.0, places=2
    )
    self.assertEqual(
        step_breakdown[0]["communication_breakdown_ms"]["all_reduce_ms"], 50.0
    )
    self.assertEqual(
        step_breakdown[0]["communication_breakdown_ms"]["send_ms"], 20.0
    )
    self.assertEqual(
        step_breakdown[0]["communication_breakdown_ms"]["recv_ms"], 30.0
    )
    self.assertEqual(step_breakdown[0]["bottleneck"], "Send and Recv")

  def test_get_step_trace_step_num_filter(self):
    pod_viewer_mock_data = {
        "podStatsSequence": {
            "podStatsMap": [
                {
                    "stepNum": 10,
                    "podStatsPerCore": {
                        "0": {
                            "totalDurationUs": 100000.0,
                            "highFlopsComputeUs": 80000.0,
                            "crsDurationUs": 10000.0,
                            "sendDurationUs": 5000.0,
                            "recvDurationUs": 5000.0,
                            "hostInfeedDurationUs": 0.0,
                            "hostOutfeedDurationUs": 0.0,
                        }
                    },
                },
                {
                    "stepNum": 11,
                    "podStatsPerCore": {
                        "0": {
                            "totalDurationUs": 200000.0,
                            "highFlopsComputeUs": 150000.0,
                            "crsDurationUs": 20000.0,
                            "sendDurationUs": 15000.0,
                            "recvDurationUs": 15000.0,
                            "hostInfeedDurationUs": 0.0,
                            "hostOutfeedDurationUs": 0.0,
                        }
                    },
                },
            ]
        }
    }

    self.mock_client.fetch.return_value = (
        None,
        json.dumps(pod_viewer_mock_data).encode("utf-8"),
    )

    result_json = get_step_trace_tool.get_step_trace(
        "test_session", step_num=11
    )
    result = json.loads(result_json)

    self.assertNotIn("error", result)
    step_breakdown = result["step_breakdown"]
    self.assertLen(step_breakdown, 1)
    self.assertEqual(step_breakdown[0]["step_num"], 11)
    self.assertAlmostEqual(step_breakdown[0]["step_time_ms"], 200.0, places=2)

  def test_get_step_trace_limit(self):
    pod_viewer_mock_data = {
        "podStatsSequence": {
            "podStatsMap": [
                {
                    "stepNum": i,
                    "podStatsPerCore": {
                        "0": {
                            "totalDurationUs": 100000.0,
                            "highFlopsComputeUs": 80000.0,
                        }
                    },
                }
                for i in range(5)
            ]
        }
    }

    self.mock_client.fetch.return_value = (
        None,
        json.dumps(pod_viewer_mock_data).encode("utf-8"),
    )

    result_json = get_step_trace_tool.get_step_trace("test_session", limit=2)
    result = json.loads(result_json)

    self.assertEqual(result["summary"]["total_steps"], 5)
    self.assertLen(result["step_breakdown"], 2)

  def test_get_step_trace_device_core_filter(self):
    pod_viewer_mock_data = {
        "podStatsSequence": {
            "podStatsMap": [{
                "stepNum": 1,
                "podStatsPerCore": {
                    "0": {
                        "totalDurationUs": 100000.0,
                        "highFlopsComputeUs": 90000.0,
                    },
                    "1": {
                        "totalDurationUs": 200000.0,
                        "highFlopsComputeUs": 180000.0,
                    },
                },
            }]
        }
    }

    self.mock_client.fetch.return_value = (
        None,
        json.dumps(pod_viewer_mock_data).encode("utf-8"),
    )

    result_json = get_step_trace_tool.get_step_trace(
        "test_session", device_core=1
    )
    result = json.loads(result_json)

    self.assertAlmostEqual(
        result["step_breakdown"][0]["step_time_ms"], 200.0, places=2
    )

  def test_get_step_trace_device_core_not_found(self):
    pod_viewer_mock_data = {
        "podStatsSequence": {
            "podStatsMap": [{
                "stepNum": 1,
                "podStatsPerCore": {
                    "0": {
                        "totalDurationUs": 100000.0,
                        "highFlopsComputeUs": 90000.0,
                    },
                },
            }]
        }
    }
    self.mock_client.fetch.return_value = (
        None,
        json.dumps(pod_viewer_mock_data).encode("utf-8"),
    )
    result_json = get_step_trace_tool.get_step_trace(
        "test_session", device_core=999
    )
    result = json.loads(result_json)
    self.assertEqual(result.get("status"), "NO_DATA")

  def test_get_step_trace_device_core_not_found_fallback(self):
    def side_effect(tool_name, *_args, **_kwargs):
      if tool_name == "pod_viewer.json":
        pod_viewer_mock_data = {
            "podStatsSequence": {
                "podStatsMap": [{
                    "stepNum": 1,
                    "podStatsPerCore": {
                        "0": {
                            "totalDurationUs": 100000.0,
                            "highFlopsComputeUs": 90000.0,
                        },
                    },
                }]
            }
        }
        return (None, json.dumps(pod_viewer_mock_data).encode("utf-8"))
      return _fetch_input_pipeline_fallback_side_effect(tool_name)

    self.mock_client.fetch.side_effect = side_effect
    result_json = get_step_trace_tool.get_step_trace(
        "test_session", device_core=999
    )
    result = json.loads(result_json)
    self.assertNotIn("status", result)
    self.assertIn("summary", result)
    self.assertIn("step_breakdown", result)
    self.assertEqual(result["summary"]["primary_bottleneck"], "Input / Infeed")

  def test_get_step_trace_include_summary_false(self):
    pod_viewer_mock_data = {
        "podStatsSequence": {
            "podStatsMap": [{
                "stepNum": 1,
                "podStatsPerCore": {
                    "0": {
                        "totalDurationUs": 100000.0,
                        "highFlopsComputeUs": 90000.0,
                    }
                },
            }]
        }
    }

    self.mock_client.fetch.return_value = (
        None,
        json.dumps(pod_viewer_mock_data).encode("utf-8"),
    )

    result_json = get_step_trace_tool.get_step_trace(
        "test_session", include_summary=False
    )
    result = json.loads(result_json)

    self.assertNotIn("summary", result)
    self.assertIn("step_breakdown", result)

  def test_get_step_trace_from_input_pipeline_fallback(self):
    self.mock_client.fetch.side_effect = (
        _fetch_input_pipeline_fallback_side_effect
    )

    result_json = get_step_trace_tool.get_step_trace("test_session")
    result = json.loads(result_json)

    self.assertNotIn("error", result)
    self.assertIn("summary", result)
    self.assertFalse(result["summary"]["is_aggregate"])
    self.assertEqual(result["summary"]["primary_bottleneck"], "Input / Infeed")
    self.assertEqual(result["summary"]["conclusion"], "Program is input-bound")
    self.assertLen(result["step_breakdown"], 2)
    self.assertEqual(result["step_breakdown"][0]["step_num"], 1)
    self.assertAlmostEqual(
        result["step_breakdown"][0]["compute_time_ms"], 20.0, places=2
    )
    self.assertAlmostEqual(
        result["step_breakdown"][0]["infeed_time_ms"], 80.0, places=2
    )

  def test_get_step_trace_from_overview_page_fallback(self):
    self.mock_client.fetch.side_effect = (
        _fetch_overview_page_fallback_side_effect
    )

    result_json = get_step_trace_tool.get_step_trace("test_session")
    result = json.loads(result_json)

    self.assertNotIn("error", result)
    self.assertIn("summary", result)
    self.assertIsNone(result["summary"]["total_steps"])
    self.assertTrue(result["summary"]["is_aggregate"])
    self.assertIn(
        "step count not available",
        result["summary"]["note"],
    )
    self.assertAlmostEqual(
        result["summary"]["step_time_ms_average"], 50.0, places=2
    )
    self.assertAlmostEqual(
        result["summary"]["compute_time_ms_average"], 40.0, places=2
    )

  def test_get_step_trace_from_overview_page_fallback_with_step_rows(self):
    def side_effect(tool_name, *_args, **_kwargs):
      if tool_name in ("pod_viewer.json", "input_pipeline.json"):
        return (None, b"")
      elif tool_name == "overview_page.json":
        overview_data = [
            {
                "cols": [{"id": "stepnum"}, {"id": "computeTimeMs"}],
                "rows": [
                    {"c": [{"v": "1"}, {"v": 10.0}]},
                    {"c": [{"v": "2"}, {"v": 12.0}]},
                    {"c": [{"v": "3"}, {"v": 11.0}]},
                ],
            },
            {
                "p": {
                    "steptime_ms_average": "50.0",
                    "tc_infeed_ms_average": "5.0",
                }
            },
        ]
        return (None, json.dumps(overview_data).encode("utf-8"))
      return (None, b"")

    self.mock_client.fetch.side_effect = side_effect
    result_json = get_step_trace_tool.get_step_trace("test_session")
    result = json.loads(result_json)

    self.assertNotIn("error", result)
    self.assertIn("summary", result)
    self.assertEqual(result["summary"]["total_steps"], 3)
    self.assertTrue(result["summary"]["is_aggregate"])
    self.assertNotIn(
        "step count not available",
        result["summary"]["note"],
    )

  def test_get_step_trace_from_overview_page_fallback_with_total_steps_prop(
      self,
  ):
    def side_effect(tool_name, *_args, **_kwargs):
      if tool_name in ("pod_viewer.json", "input_pipeline.json"):
        return (None, b"")
      elif tool_name == "overview_page.json":
        overview_data = [
            {
                "p": {
                    "steptime_ms_average": "50.0",
                    "total_steps": "42",
                }
            }
        ]
        return (None, json.dumps(overview_data).encode("utf-8"))
      return (None, b"")

    self.mock_client.fetch.side_effect = side_effect
    result_json = get_step_trace_tool.get_step_trace("test_session")
    result = json.loads(result_json)

    self.assertEqual(result["summary"]["total_steps"], 42)
    self.assertTrue(result["summary"]["is_aggregate"])

  def test_get_step_trace_no_data(self):
    self.mock_client.fetch.return_value = (None, b"")
    res_raw = get_step_trace_tool.get_step_trace("test_session")
    res = json.loads(res_raw)
    self.assertEqual(res.get("status"), "NO_DATA")
    self.assertIn("No step trace data", res.get("message", ""))

  def test_get_step_trace_exception_handled(self):
    self.mock_client.fetch.side_effect = RuntimeError("Backend unavailable")
    res_raw = get_step_trace_tool.get_step_trace("test_session")
    res = json.loads(res_raw)
    self.assertEqual(res.get("status"), "NO_DATA")
    self.assertIn("No step trace data", res.get("message", ""))

  def test_get_step_trace_bypass_cache(self):
    self.mock_client.fetch.return_value = (None, b"")
    res_raw = get_step_trace_tool.get_step_trace(
        "test_session", bypass_cache=True
    )
    res = json.loads(res_raw)
    self.assertEqual(res.get("status"), "NO_DATA")
    self.mock_client.fetch.assert_any_call(
        tool_name="pod_viewer.json",
        session_id="test_session",
        format="json",
        bypass_cache=True,
    )

  def test_get_step_trace_file_not_found(self):
    self.mock_client.fetch.side_effect = FileNotFoundError(
        "Trace path not found"
    )

    with self.assertRaises(FileNotFoundError):
      get_step_trace_tool.get_step_trace("non_existent_session")

  def test_get_step_trace_from_pod_viewer_step_breakdown_map(self):
    """Modern PodStatsRecords keep the breakdown in a stepBreakdownUs map."""
    pod_viewer_mock_data = {
        "podStatsSequence": {
            "podStatsMap": [{
                "stepNum": 7,
                "podStatsPerCore": {
                    "0": {
                        "chipId": 0,
                        "nodeId": 0,
                        "stepNum": 7,
                        "totalDurationUs": 200000.0,
                        # GenericEventType -> duration (us).
                        "stepBreakdownUs": {
                            "1": 150000.0,  # Device compute
                            "2": 5000.0,  # Device to device
                            "3": 25000.0,  # Device collectives
                            "6": 12000.0,  # Input
                            "7": 8000.0,  # Output
                        },
                    }
                },
            }]
        }
    }
    self.mock_client.fetch.return_value = (
        None,
        json.dumps(pod_viewer_mock_data).encode("utf-8"),
    )

    result = json.loads(get_step_trace_tool.get_step_trace("test_session"))

    step = result["step_breakdown"][0]
    self.assertEqual(step["step_num"], 7)
    self.assertAlmostEqual(step["step_time_ms"], 200.0, places=2)
    self.assertAlmostEqual(step["compute_time_ms"], 150.0, places=2)
    self.assertAlmostEqual(step["compute_percent"], 75.0, places=2)
    self.assertAlmostEqual(step["communication_time_ms"], 30.0, places=2)
    self.assertAlmostEqual(step["infeed_time_ms"], 12.0, places=2)
    self.assertAlmostEqual(step["outfeed_time_ms"], 8.0, places=2)
    self.assertEqual(step["bottleneck"], "Compute")

  def test_get_step_trace_pod_viewer_empty_breakdown_falls_back(self):
    """An all-zero pod_viewer breakdown must not produce a bogus bottleneck."""

    def side_effect(tool_name, *_args, **_kwargs):
      if tool_name in ("pod_viewer.json", "pod_viewer"):
        # pod_viewer times the step but reports no breakdown at all, and still
        # labels the step "Output".
        return (
            None,
            json.dumps({
                "podStatsSequence": {
                    "podStatsMap": [{
                        "stepNum": 0,
                        "podStatsPerCore": {
                            "0": {
                                "totalDurationUs": 234523.37,
                                "bottleneck": "Output",
                                "stepBreakdownUs": {
                                    str(i): 0 for i in range(1, 10)
                                },
                            }
                        },
                    }]
                }
            }).encode("utf-8"),
        )
      if tool_name == "input_pipeline_analyzer":
        return (
            None,
            json.dumps([{
                "cols": [
                    {"id": "stepnum", "type": "string"},
                    {"id": "tcComputeTimeMs", "type": "number"},
                    {"id": "tcInfeedTimeMs", "type": "number"},
                    {"id": "tcOutfeedTimeMs", "type": "number"},
                    {"id": "tcIdleTimeMs", "type": "number"},
                    {"id": "hostTransferTimeMs", "type": "number"},
                    {"id": "infeedPercentAverage", "type": "number"},
                ],
                "rows": [{
                    "c": [
                        {"v": "0"},
                        {"v": 231.9486925},
                        {"v": 0.0},
                        {"v": 0.0},
                        {"v": 2.5746775},
                        {"v": 0.0},
                        {"v": 0.0},
                    ]
                }],
            }]).encode("utf-8"),
        )
      return (None, b"")

    self.mock_client.fetch.side_effect = side_effect

    result = json.loads(get_step_trace_tool.get_step_trace("test_session"))

    summary = result["summary"]
    self.assertAlmostEqual(summary["step_time_ms_average"], 234.5234, places=2)
    self.assertAlmostEqual(
        summary["compute_time_ms_average"], 231.9487, places=2
    )
    self.assertGreater(summary["compute_percent"], 95.0)
    self.assertEqual(summary["primary_bottleneck"], "Compute")

  def test_get_step_trace_input_pipeline_unknown_tool_name_falls_back(self):
    """An unknown tool name must not abort the fallback chain."""

    def side_effect(tool_name, *_args, **_kwargs):
      if tool_name in ("pod_viewer.json", "pod_viewer"):
        return (None, b"")
      if tool_name == "input_pipeline_analyzer":
        raise ValueError(f"Unknown XProf tool name: {tool_name!r}")
      return _fetch_overview_page_fallback_side_effect(tool_name)

    self.mock_client.fetch.side_effect = side_effect

    result = json.loads(get_step_trace_tool.get_step_trace("test_session"))

    self.assertNotIn("error", result)
    self.assertTrue(result["summary"]["is_aggregate"])
    self.assertAlmostEqual(
        result["summary"]["step_time_ms_average"], 50.0, places=2
    )

  def test_get_step_trace_pod_viewer_reports_unattributed_time_as_idle(self):
    """Step time the breakdown does not cover is reported as idle."""
    pod_viewer_mock_data = {
        "podStatsSequence": {
            "podStatsMap": [{
                "stepNum": 3,
                "podStatsPerCore": {
                    "0": {
                        "chipId": 0,
                        "nodeId": 0,
                        "stepNum": 3,
                        "totalDurationUs": 100000.0,
                        # Only 90 ms of the 100 ms step is attributed to a
                        # bottleneck category; "4" is host compute.
                        "stepBreakdownUs": {
                            "1": 80000.0,  # Device compute
                            "6": 10000.0,  # Input
                            "4": 4000.0,  # Host compute
                        },
                    }
                },
            }]
        }
    }
    self.mock_client.fetch.return_value = (
        None,
        json.dumps(pod_viewer_mock_data).encode("utf-8"),
    )

    result = json.loads(get_step_trace_tool.get_step_trace("test_session"))

    step = result["step_breakdown"][0]
    self.assertAlmostEqual(step["step_time_ms"], 100.0, places=2)
    self.assertAlmostEqual(step["compute_time_ms"], 80.0, places=2)
    self.assertAlmostEqual(step["idle_time_ms"], 10.0, places=2)
    self.assertAlmostEqual(step["idle_percent"], 10.0, places=2)
    # Compute, communication, infeed, outfeed and idle cover the whole step.
    self.assertAlmostEqual(
        step["compute_time_ms"]
        + step["communication_time_ms"]
        + step["infeed_time_ms"]
        + step["outfeed_time_ms"]
        + step["idle_time_ms"],
        step["step_time_ms"],
        places=2,
    )
    summary = result["summary"]
    self.assertAlmostEqual(summary["idle_time_ms_average"], 10.0, places=2)
    self.assertAlmostEqual(summary["idle_percent"], 10.0, places=2)

  def test_get_step_trace_input_pipeline_reports_tc_idle_time(self):
    """The TPU step table's tcIdleTimeMs column surfaces as idle time."""
    input_pipeline_mock_data = [
        {},
        {
            "cols": [
                {"id": "stepnum"},
                {"id": "tcComputeTimeMs"},
                {"id": "tcInfeedTimeMs"},
                {"id": "tcOutfeedTimeMs"},
                {"id": "tcIdleTimeMs"},
            ],
            "rows": [{
                "c": [
                    {"v": "0"},
                    {"v": 231.9486925},
                    {"v": 0.0},
                    {"v": 0.0},
                    {"v": 2.5746775},
                ]
            }],
        },
    ]

    def side_effect(tool_name, *_args, **_kwargs):
      if tool_name in ("pod_viewer.json", "pod_viewer"):
        return (None, b"")
      return (None, json.dumps(input_pipeline_mock_data).encode("utf-8"))

    self.mock_client.fetch.side_effect = side_effect

    result = json.loads(get_step_trace_tool.get_step_trace("test_session"))

    step = result["step_breakdown"][0]
    self.assertAlmostEqual(step["step_time_ms"], 234.5234, places=3)
    self.assertAlmostEqual(step["compute_time_ms"], 231.9487, places=3)
    self.assertAlmostEqual(step["idle_time_ms"], 2.5747, places=3)
    self.assertAlmostEqual(step["compute_percent"], 98.9, places=1)
    self.assertAlmostEqual(step["idle_percent"], 1.1, places=1)
    self.assertEqual(step["bottleneck"], "Compute")


if __name__ == "__main__":
  absltest.main()
