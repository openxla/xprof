import json
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from xprof.cli.internal import decorators
from xprof.cli.internal.oss import xprof_client
from xprof.cli.tools import get_kpi_metrics_tool
from xprof.cli.tools import get_overview_tool
from xprof.cli.tools import get_roofline_model_tool


class GetKpiMetricsToolTest(parameterized.TestCase):

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
        )
    )

  def test_get_kpi_metrics_success(self):
    def fetch_side_effect(tool_name, *_args, **_kwargs):
      """Mocks successful fetches for overview and memory profile tools."""
      if tool_name == "overview_page.json":
        return (
            None,
            json.dumps([{
                "p": {
                    "steptime_ms_average": "10.5",
                    "device_duty_cycle_percent": "95.0",
                    "mxu_utilization_percent": "80.0",
                    "flop_rate_utilization_relative_to_roofline": "45.0",
                    "device_type": "TPU",
                    "device_core_count": "8",
                }
            }]).encode("utf-8"),
        )
      elif tool_name == "memory_profile.json":
        return (
            None,
            json.dumps([{
                "memoryProfilePerAllocator": {
                    "default": {
                        "profileSummary": {
                            "memoryCapacity": str(int(32 * 1024**3)),
                            "peakStats": {
                                "peakBytesInUse": str(int(12.5 * 1024**3)),
                            },
                        }
                    }
                }
            }]).encode("utf-8"),
        )
      return None

    self.mock_client.fetch.side_effect = fetch_side_effect

    result_json = get_kpi_metrics_tool.get_kpi_metrics("test_session")
    result = json.loads(result_json)

    expected = {
        "step_time_ms": "10.5",
        "duty_cycle_percent": "95.0",
        "mxu_utilization_percent": "80.0",
        "roofline_utilization": "45.0",
        "flops_provenance": "xla_cost_model",
        "derived_flops_fraction": 0.0,
        "peak_hbm_gib": 12.5,
        "accelerator_info": {
            "device_type": "TPU",
            "device_core_count": "8",
        },
    }
    self.assertEqual(result, expected)

  def test_get_kpi_metrics_roofline_override_when_derived_from_shapes(self):
    def fetch_side_effect(tool_name, *_args, **_kwargs):
      if tool_name == "overview_page.json":
        return (
            None,
            json.dumps([{
                "p": {
                    "steptime_ms_average": "0.396",
                    "device_duty_cycle_percent": "99.0",
                    "mxu_utilization_percent": "0.0",
                    "flop_rate_utilization_relative_to_roofline": "4.71%",
                    "device_type": "TPU v7x",
                    "device_core_count": "8",
                }
            }]).encode("utf-8"),
        )
      elif tool_name == "memory_profile.json":
        return (
            None,
            json.dumps([{
                "memoryProfilePerAllocator": {
                    "default": {
                        "profileSummary": {
                            "memoryCapacity": str(int(32 * 1024**3)),
                            "peakStats": {
                                "peakBytesInUse": str(int(1.0 * 1024**3)),
                            },
                        }
                    }
                }
            }]).encode("utf-8"),
        )
      elif tool_name == "roofline_model.json":
        return (
            "application/json",
            json.dumps([{
                "cols": [
                    {"id": "step_id", "type": "string"},
                    {"id": "rank", "type": "number"},
                    {"id": "category", "type": "string"},
                    {"id": "operation", "type": "string"},
                    {"id": "occurrences", "type": "number"},
                    {"id": "total_time", "type": "number"},
                    {"id": "total_self_time", "type": "number"},
                    {"id": "total_self_time_percent", "type": "number"},
                    {"id": "measured_flop_rate", "type": "number"},
                    {"id": "model_flop_rate", "type": "number"},
                    {"id": "measured_memory_bw", "type": "number"},
                    {"id": "hbm_bw", "type": "number"},
                    {"id": "operational_intensity", "type": "number"},
                    {"id": "bound_by", "type": "string"},
                    {"id": "roofline_efficiency", "type": "number"},
                    {"id": "compute_efficiency", "type": "number"},
                    {"id": "max_mem_bw_utilization", "type": "number"},
                    {"id": "hlo_module_id", "type": "string"},
                    {"id": "source_info", "type": "string"},
                ],
                "p": {
                    "device_type": "TPU v7x",
                    "peak_flop_rate": "918000",
                    "peak_hbm_bw": "1638.0",
                    "hbm_ridge_point": "560.44",
                },
                "rows": [
                    {
                        "c": [
                            {"v": "Total"},
                            {"v": 0.0},
                            {"v": "Program"},
                            {"v": "Program"},
                            {"v": 1.0},
                            {"v": 396.32},
                            {"v": 396.32},
                            {"v": 1.0},
                            {"v": 0.0},
                            {"v": 0.0},
                            {"v": 77.15},
                            {"v": 77.15},
                            {"v": 0.0},
                            {"v": "HBM"},
                            {"v": 0.0471},
                            {"v": 0.0},
                            {"v": 0.0471},
                            {"v": "0"},
                            {"v": ""},
                        ]
                    },
                    {
                        "c": [
                            {"v": "Total"},
                            {"v": 1.0},
                            {"v": "custom-call"},
                            {"v": "custom-call.1"},
                            {"v": 1.0},
                            {"v": 396.32},
                            {"v": 396.32},
                            {"v": 1.0},
                            {"v": 0.0},
                            {"v": 0.0},
                            {"v": 77.15},
                            {"v": 77.15},
                            {"v": 0.0},
                            {"v": "Unknown"},
                            {"v": 0.0471},
                            {"v": 0.0},
                            {"v": 0.0471},
                            {"v": "12345"},
                            {"v": ""},
                        ]
                    },
                ],
            }]).encode("utf-8"),
        )
      elif tool_name == "hlo_op_profile.json":
        return (
            "application/json",
            json.dumps({
                "by_program": {
                    "name": "main",
                    "children": [{
                        "name": "custom-call.1",
                        "xla": {
                            "expression": (
                                "%custom-call.1 = bf16[4,4096,4096]"
                                " custom-call(bf16[4,4096,2048] %p0,"
                                " bf16[2048,4096] %p1)"
                            )
                        },
                    }],
                }
            }).encode("utf-8"),
        )
      return None

    self.mock_client.fetch.side_effect = fetch_side_effect

    result_json = get_kpi_metrics_tool.get_kpi_metrics("test_session")
    result = json.loads(result_json)

    self.assertEqual(result["flops_provenance"], "derived_from_shapes")
    self.assertEqual(result["roofline_utilization"], "75.55%")
    self.assertIsNone(result["mxu_utilization_percent"])
    self.assertEqual(result["mxu_utilization_status"]["status"], "unavailable")

  @parameterized.named_parameters(
      dict(
          testcase_name="mixed_overrides",
          provenance="mixed",
          overview_compute_eff="1.00%",
          expected_compute_eff="12.34%",
      ),
      dict(
          testcase_name="cost_model_keeps_overview",
          provenance="xla_cost_model",
          overview_compute_eff="1.00%",
          expected_compute_eff="1.00%",
      ),
  )
  def test_get_kpi_metrics_mxu_status_and_compute_efficiency(
      self, provenance, overview_compute_eff, expected_compute_eff
  ):
    overview_json = json.dumps({
        "performance_summary": {
            "steptime_ms_average": "1.0",
            "mxu_utilization_percent": None,
            "mxu_utilization_status": {"status": "unavailable"},
            "compute_efficiency_percent": overview_compute_eff,
        },
        "run_environment": {},
    })
    self.enter_context(
        mock.patch.object(
            get_overview_tool,
            "get_overview",
            return_value=overview_json,
            autospec=True,
        )
    )
    self.enter_context(
        mock.patch.object(
            get_roofline_model_tool,
            "get_roofline_model",
            return_value=json.dumps({
                "program": {
                    "flops_provenance": provenance,
                    "roofline_efficiency_percent": "75.55%",
                    "compute_efficiency_percent": "12.34%",
                }
            }),
            autospec=True,
        )
    )

    result = json.loads(get_kpi_metrics_tool.get_kpi_metrics("test_session"))

    self.assertIsNone(result["mxu_utilization_percent"])
    self.assertEqual(
        result["mxu_utilization_status"], {"status": "unavailable"}
    )
    self.assertEqual(result["compute_efficiency_percent"], expected_compute_eff)

  @parameterized.named_parameters(
      ("derived_from_shapes", "derived_from_shapes", 1.0, "75.55%"),
      ("mixed", "mixed", 0.25, "75.55%"),
      ("xla_cost_model", "xla_cost_model", 0.0, "4.71%"),
  )
  def test_get_kpi_metrics_roofline_override_by_provenance(
      self, provenance, derived_flops_fraction, expected_roofline_util
  ):
    def fetch_side_effect(tool_name, *_args, **_kwargs):
      if tool_name == "overview_page.json":
        return (
            None,
            json.dumps([{
                "p": {
                    "steptime_ms_average": "0.396",
                    "flop_rate_utilization_relative_to_roofline": "4.71%",
                    "device_type": "TPU v7x",
                    "device_core_count": "8",
                }
            }]).encode("utf-8"),
        )
      return None

    self.mock_client.fetch.side_effect = fetch_side_effect
    self.enter_context(
        mock.patch.object(
            get_roofline_model_tool,
            "get_roofline_model",
            return_value=json.dumps({
                "program": {
                    "flops_provenance": provenance,
                    "derived_flops_fraction": derived_flops_fraction,
                    "roofline_efficiency_percent": "75.55%",
                }
            }),
            autospec=True,
        )
    )

    result = json.loads(get_kpi_metrics_tool.get_kpi_metrics("test_session"))

    self.assertEqual(result["flops_provenance"], provenance)
    self.assertEqual(result["derived_flops_fraction"], derived_flops_fraction)
    self.assertEqual(result["roofline_utilization"], expected_roofline_util)

  def test_get_kpi_metrics_overview_error(self):
    self.mock_client.fetch.return_value = (None, b"")

    with self.assertRaises(FileNotFoundError):
      get_kpi_metrics_tool.get_kpi_metrics("test_session")

  def test_get_kpi_metrics_memory_error(self):
    def fetch_side_effect(tool_name, *_args, **_kwargs):
      """Mocks overview success and memory profile tool failure."""
      if tool_name == "overview_page.json":
        return (
            None,
            json.dumps(
                [
                    {
                        "p": {
                            "steptime_ms_average": "10.5",
                        }
                    }
                ]
            ).encode("utf-8"),
        )
      elif tool_name == "memory_profile.json":
        raise RuntimeError("Fetch failed")
      return None

    self.mock_client.fetch.side_effect = fetch_side_effect

    result_json = get_kpi_metrics_tool.get_kpi_metrics("test_session")
    result = json.loads(result_json)

    self.assertNotIn("error", result)
    self.assertEqual(result["step_time_ms"], "10.5")
    self.assertEqual(result["peak_hbm_gib"], "N/A")

  def test_get_kpi_metrics_multi_host_with_slower_second_host(self):
    def fetch_side_effect(tool_name, *_args, **_kwargs):
      if tool_name in ("overview_page", "overview_page.json"):
        return (
            None,
            json.dumps([{
                "p": {
                    "steptime_ms_average": "22.00",
                    "steptime_ms_standard_deviation": "2.00",
                    "device_duty_cycle_percent": "90.0%",
                    "mxu_utilization_percent": "50.0%",
                    "flop_rate_utilization_relative_to_roofline": "45.0%",
                    "device_type": "TPU v7x",
                    "device_core_count": "16",
                    "host_count": "2",
                }
            }]).encode("utf-8"),
        )
      if tool_name in ("pod_viewer", "pod_viewer.json"):
        return (
            "application/json",
            json.dumps({
                "podStatsSequence": {
                    "podStatsMap": [{
                        "stepNum": 0,
                        "podStatsPerCore": {
                            "0": {
                                "hostName": "a_fast_host",
                                "stepNum": 0,
                                "totalDurationUs": 12000.0,
                            },
                            "1000": {
                                "hostName": "z_slow_host",
                                "stepNum": 0,
                                "totalDurationUs": 22000.0,
                            },
                        },
                    }]
                }
            }).encode("utf-8"),
        )
      return None

    self.mock_client.fetch.side_effect = fetch_side_effect
    result = json.loads(get_kpi_metrics_tool.get_kpi_metrics("multi_session"))
    self.assertEqual(result["step_time_ms"], "22.00")
    self.assertEqual(result["step_time_scope"], "per_step_max_across_hosts")
    self.assertEqual(
        result["per_host_step_time_ms"],
        {"a_fast_host": "12.00", "z_slow_host": "22.00"},
    )
    self.assertIn("step_time_ms", result["step_time_warning"])
    self.assertNotIn("steptime_ms_average", result["step_time_warning"])
    self.assertNotIn("steptime_ms_minimum", result["step_time_warning"])


if __name__ == "__main__":
  absltest.main()
