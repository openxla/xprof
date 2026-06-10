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
        "peak_hbm_gib": 12.5,
        "accelerator_info": {
            "device_type": "TPU",
            "device_core_count": "8",
        },
    }
    self.assertEqual(result, expected)

  def test_get_kpi_metrics_overview_error(self):
    self.mock_client.fetch.return_value = (None, b"")

    result_json = get_kpi_metrics_tool.get_kpi_metrics("test_session")
    result = json.loads(result_json)

    self.assertIn("error", result)
    self.assertStartsWith(result["error"], "Error in get_overview:")

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
