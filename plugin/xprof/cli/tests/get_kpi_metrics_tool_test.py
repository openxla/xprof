import json
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from xprof.cli.internal import decorators
from xprof.cli.internal.oss import xprof_client
from xprof.cli.tools import get_kpi_metrics_tool


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


if __name__ == "__main__":
  absltest.main()
