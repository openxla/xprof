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
    self.assertAlmostEqual(
        result["summary"]["step_time_ms_average"], 50.0, places=2
    )
    self.assertAlmostEqual(
        result["summary"]["compute_time_ms_average"], 40.0, places=2
    )

  def test_get_step_trace_no_data_error(self):
    self.mock_client.fetch.return_value = (None, b"")

    result_json = get_step_trace_tool.get_step_trace("test_session")
    result = json.loads(result_json)

    self.assertIn("error", result)
    self.assertIn("No step trace data returned", result["error"])

  def test_get_step_trace_exception_handled(self):
    self.mock_client.fetch.side_effect = RuntimeError("Backend unavailable")

    result_json = get_step_trace_tool.get_step_trace("test_session")
    result = json.loads(result_json)

    self.assertIn("error", result)

  def test_get_step_trace_markdown_format_success(self):
    pod_viewer_mock_data = {
        "podStatsSequence": {
            "podStatsMap": [{
                "stepNum": 100,
                "podStatsPerCore": {
                    "0": {
                        "totalDurationUs": 300000.0,
                        "highFlopsComputeUs": 150000.0,
                        "crsDurationUs": 50000.0,
                        "sendDurationUs": 20000.0,
                        "recvDurationUs": 30000.0,
                        "hostInfeedDurationUs": 10000.0,
                        "hostOutfeedDurationUs": 5000.0,
                        "bottleneck": "Compute",
                    },
                },
            }]
        }
    }

    self.mock_client.fetch.return_value = (
        None,
        json.dumps(pod_viewer_mock_data).encode("utf-8"),
    )

    result_md = get_step_trace_tool.get_step_trace(
        "test_session", output_format="markdown"
    )

    self.assertIn("# Step Execution Trace & Breakdown", result_md)
    self.assertIn("## Session Summary", result_md)
    self.assertIn("- Total Steps: 1", result_md)
    self.assertIn("## Step Breakdown", result_md)
    self.assertIn("| Step | Total (ms) | Compute (ms) |", result_md)
    self.assertIn("| 100 | 300.0000 | 150.0000 (50.0%) |", result_md)

  def test_get_step_trace_markdown_format_no_summary(self):
    pod_viewer_mock_data = {
        "podStatsSequence": {
            "podStatsMap": [{
                "stepNum": 101,
                "podStatsPerCore": {
                    "0": {
                        "totalDurationUs": 200000.0,
                        "highFlopsComputeUs": 100000.0,
                    },
                },
            }]
        }
    }

    self.mock_client.fetch.return_value = (
        None,
        json.dumps(pod_viewer_mock_data).encode("utf-8"),
    )

    result_md = get_step_trace_tool.get_step_trace(
        "test_session", output_format="markdown", include_summary=False
    )

    self.assertIn("# Step Execution Trace & Breakdown", result_md)
    self.assertNotIn("## Session Summary", result_md)
    self.assertIn("## Step Breakdown", result_md)
    self.assertIn("| 101 | 200.0000 |", result_md)

  def test_get_step_trace_markdown_error(self):
    self.mock_client.fetch.return_value = (None, b"")

    result_md = get_step_trace_tool.get_step_trace(
        "test_session", output_format="markdown"
    )

    self.assertTrue(result_md.startswith("# Error\n"))
    self.assertIn("No step trace data returned", result_md)

  def test_dataclass_instantiation(self):
    comm = get_step_trace_tool.CommunicationBreakdown(
        all_reduce_ms=10.0,
        send_ms=5.0,
        recv_ms=5.0,
    )
    step = get_step_trace_tool.StepInfo(
        step_num=1,
        step_time_ms=100.0,
        compute_time_ms=50.0,
        compute_percent=50.0,
        communication_time_ms=20.0,
        communication_percent=20.0,
        infeed_time_ms=10.0,
        infeed_percent=10.0,
        outfeed_time_ms=2.0,
        outfeed_percent=2.0,
        bottleneck="Compute",
        communication_breakdown_ms=comm,
    )
    summary = get_step_trace_tool.SummaryData(
        total_steps=1,
        step_time_ms_average=100.0,
        step_time_ms_min=100.0,
        step_time_ms_max=100.0,
        step_time_ms_stddev=0.0,
        compute_time_ms_average=50.0,
        compute_percent=50.0,
        communication_time_ms_average=20.0,
        communication_percent=20.0,
        infeed_time_ms_average=10.0,
        infeed_percent=10.0,
        outfeed_time_ms_average=2.0,
        outfeed_percent=2.0,
        primary_bottleneck="Compute",
    )

    self.assertEqual(comm.all_reduce_ms, 10.0)
    self.assertEqual(step.step_num, 1)
    self.assertEqual(summary.total_steps, 1)


if __name__ == "__main__":
  absltest.main()
