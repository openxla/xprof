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

  def test_get_step_trace_populates_step_time_distribution_and_partial_steps(
      self,
  ):
    """Per-core durations populate step_time_distribution_ms and partial_steps."""
    pod_viewer_mock_data = {
        "podStatsSequence": {
            "podStatsMap": [
                {
                    "stepNum": 0,
                    "podStatsPerCore": {
                        "0": {
                            "totalDurationUs": 2000.0,
                            "highFlopsComputeUs": 1800.0,
                        },
                        "1": {
                            "totalDurationUs": 8000.0,
                            "highFlopsComputeUs": 7500.0,
                        },
                    },
                },
                {
                    "stepNum": 1,
                    "podStatsPerCore": {
                        "0": {
                            "totalDurationUs": 23500.0,
                            "highFlopsComputeUs": 23000.0,
                        },
                        "1": {
                            "totalDurationUs": 23700.0,
                            "highFlopsComputeUs": 23200.0,
                        },
                    },
                },
                {
                    "stepNum": 2,
                    "podStatsPerCore": {
                        "0": {
                            "totalDurationUs": 23600.0,
                            "highFlopsComputeUs": 23100.0,
                        },
                        "1": {
                            "totalDurationUs": 23800.0,
                            "highFlopsComputeUs": 23300.0,
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

    result = json.loads(get_step_trace_tool.get_step_trace("test_session"))
    summary = result["summary"]

    self.assertEqual(summary["total_steps"], 3)
    self.assertEqual(summary["full_steps_count"], 2)
    self.assertAlmostEqual(
        summary["full_step_time_ms_average"], 23.65, places=2
    )
    self.assertEqual(summary["step_source"], "Steps")

    dist = summary["step_time_distribution_ms"]
    self.assertEqual(dist["all_steps"]["count"], 6)
    self.assertEqual(dist["all_steps"]["steps"], [0, 1, 2])
    self.assertAlmostEqual(dist["all_steps"]["min_ms"], 2.0, places=2)
    self.assertAlmostEqual(dist["all_steps"]["max_ms"], 23.8, places=2)

    self.assertEqual(dist["full_steps"]["count"], 4)
    self.assertEqual(dist["full_steps"]["steps"], [1, 2])
    self.assertAlmostEqual(dist["full_steps"]["mean_ms"], 23.65, places=2)

    self.assertEqual(dist["partial_steps"]["count"], 2)
    self.assertEqual(dist["partial_steps"]["steps"], [0])
    self.assertAlmostEqual(dist["partial_steps"]["mean_ms"], 5.0, places=2)
    self.assertEqual(dist["full_steps"]["dispersion_assessment"], "CONSISTENT")
    self.assertEqual(summary["dispersion_assessment"], "CONSISTENT")

  def test_get_step_trace_with_func_name_extracts_module_step_records(self):
    """Passing func_name extracts step records from XLA Modules and computes dispersion."""
    self.mock_client.fetch.return_value = (None, b"")
    with mock.patch.object(
        get_step_trace_tool,
        "_extract_step_records_from_source",
        return_value=(
            [
                (0, 15.0),
                (1, 17.0),
            ],
            "XLA Modules",
        ),
        autospec=True,
    ) as mock_extract:
      result = json.loads(
          get_step_trace_tool.get_step_trace(
              "test_session", func_name="train_step"
          )
      )

    mock_extract.assert_called_once_with(
        "test_session",
        func_name="train_step",
        bypass_cache=False,
    )
    summary = result["summary"]
    self.assertEqual(summary["total_steps"], 2)
    self.assertAlmostEqual(summary["step_time_ms_average"], 16.0, places=2)
    self.assertEqual(summary["step_source"], "XLA Modules")
    self.assertEqual(
        summary["step_time_distribution_ms"]["all_steps"]["count"], 2
    )
    self.assertEqual(summary["dispersion_assessment"], "CONSISTENT")

  def test_compute_step_time_distribution_high_jitter(self):
    """High CV (>0.15) is flagged as HIGH_JITTER in compute_step_time_distribution."""
    records = [
        (0, 100.0),
        (1, 100.0),
        (2, 300.0),
        (3, 100.0),
    ]
    dist = get_step_trace_tool.compute_step_time_distribution(records)
    self.assertEqual(dist["full_steps"]["dispersion_assessment"], "HIGH_JITTER")
    self.assertGreater(dist["full_steps"]["cv"], 0.15)
    self.assertGreater(dist["full_steps"]["iqr_ms"], 0.0)

  def test_summary_from_step_records_uses_distinct_step_counts(self):
    """_summary_from_step_records counts distinct steps rather than per-core rows."""
    records = [
        (0, 2.0),
        (0, 8.0),
        (1, 23.5),
        (1, 23.7),
        (2, 23.6),
        (2, 23.8),
    ]
    summary = get_step_trace_tool._summary_from_step_records(records, "Steps")
    self.assertEqual(summary.total_steps, 3)
    self.assertEqual(summary.full_steps_count, 2)
    self.assertAlmostEqual(summary.full_step_time_ms_average, 23.65, places=2)
    self.assertEqual(summary.step_source, "Steps")
    self.assertEqual(summary.step_time_distribution_ms["all_steps"]["count"], 6)

  def test_extract_step_records_from_source_events_db_handles_null_dur_and_regex(
      self,
  ):
    """Events DB extraction handles NULL dur_ms safely and uses case-insensitive SQL."""
    mock_root_res = mock.MagicMock()
    mock_root_res.status = "success"
    mock_root_res.session_root = "/path/to/test/session"
    with (
        mock.patch.object(
            get_step_trace_tool.events_db_tools,
            "get_events_db_session_root",
            return_value=mock_root_res,
            autospec=True,
        ),
        mock.patch.object(
            get_step_trace_tool.google_kernel_stats_tools,
            "run_f1_sql",
            return_value=[
                {"category": "Steps", "kernel_name": "0", "dur_ms": None},
                {"category": "Steps", "kernel_name": "1", "dur_ms": 12.5},
            ],
            autospec=True,
        ) as mock_sql,
    ):
      rows, src = get_step_trace_tool._extract_step_records_from_source(
          "remote_session_123"
      )

    self.assertEqual(src, "Steps")
    self.assertEqual(rows, [(0, 0.0), (1, 12.5)])
    called_sql = mock_sql.call_args[0][1]
    self.assertIn(
        r"REGEXP_CONTAINS(device, r'^(?i)(?:/device:)?(?:tpu|gpu):[0-9]+$')",
        called_sql,
    )
    self.assertNotIn("LIKE @pattern", called_sql)
    self.assertIn("LIMIT 100000", called_sql)
    self.assertIsNone(mock_sql.call_args[1]["parameters"])

  def test_extract_step_records_from_source_events_db_with_func_name_and_error_log(
      self,
  ):
    """Events DB extraction filters by @pattern when func_name is set and logs root errors."""
    mock_fail_res = mock.MagicMock()
    mock_fail_res.status = "error"
    mock_fail_res.session_root = ""
    mock_fail_res.error_message = "DB unavailable"
    with (
        mock.patch.object(
            get_step_trace_tool.events_db_tools,
            "get_events_db_session_root",
            return_value=mock_fail_res,
            autospec=True,
        ),
        self.assertLogs(level="DEBUG") as log_ctx,
    ):
      rows, src = get_step_trace_tool._extract_step_records_from_source(
          "remote_session_err"
      )
    self.assertEqual(rows, [])
    self.assertEqual(src, "XLA Modules")
    self.assertTrue(
        any("DB unavailable" in msg for msg in log_ctx.output),
        f"Expected error log in {log_ctx.output}",
    )

    mock_ok_res = mock.MagicMock()
    mock_ok_res.status = "success"
    mock_ok_res.session_root = "/path/to/test/session"
    with (
        mock.patch.object(
            get_step_trace_tool.events_db_tools,
            "get_events_db_session_root",
            return_value=mock_ok_res,
            autospec=True,
        ),
        mock.patch.object(
            get_step_trace_tool.google_kernel_stats_tools,
            "run_f1_sql",
            return_value=[{
                "category": "XLA Modules",
                "kernel_name": "jit_train_step",
                "dur_ms": None,
                "duration_ms": 18.25,
            }],
            autospec=True,
        ) as mock_sql,
    ):
      rows, src = get_step_trace_tool._extract_step_records_from_source(
          "remote_session_123", func_name="jit_train_step"
      )
    self.assertEqual(src, "XLA Modules")
    self.assertEqual(rows, [(None, 18.25)])
    self.assertIn("AND kernel_name LIKE @pattern", mock_sql.call_args[0][1])
    self.assertEqual(
        mock_sql.call_args[1]["parameters"], {"pattern": "%train_step%"}
    )

  def test_overview_and_input_pipeline_populate_step_source_and_summary_fields(
      self,
  ):
    """Overview page and input pipeline fallbacks populate step_source and summary fields."""
    self.mock_client.fetch.side_effect = (
        _fetch_input_pipeline_fallback_side_effect
    )
    ip_res = json.loads(get_step_trace_tool.get_step_trace("ip_session"))
    self.assertEqual(ip_res["summary"]["step_source"], "input_pipeline")
    self.assertEqual(ip_res["summary"]["full_steps_count"], 2)

    self.mock_client.fetch.side_effect = (
        _fetch_overview_page_fallback_side_effect
    )
    ov_res = json.loads(get_step_trace_tool.get_step_trace("ov_session"))
    self.assertEqual(ov_res["summary"]["step_source"], "overview_page")
    self.assertEqual(ov_res["summary"]["dispersion_assessment"], "CONSISTENT")
    self.assertAlmostEqual(
        ov_res["summary"]["full_step_time_ms_average"], 50.0, places=2
    )


if __name__ == "__main__":
  absltest.main()
