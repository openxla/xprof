"""Unit tests for check_host_boundness_tool."""

import json
from unittest import mock
from absl.testing import absltest
from xprof.cli.internal import decorators
from xprof.cli.tools import check_host_boundness_tool
from xprof.cli.tools import get_overview_tool
from xprof.cli.tools import get_utilization_viewer_tool


class CheckHostBoundnessToolTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    mock_cache = mock.create_autospec(decorators.Cache, instance=True)
    mock_cache.get.return_value = decorators.Cache.UNKNOWN
    self.enter_context(
        mock.patch.object(decorators, "get_cache", return_value=mock_cache)
    )

  @mock.patch.object(get_overview_tool, "get_overview")
  @mock.patch.object(get_utilization_viewer_tool, "get_utilization_viewer")
  def test_host_bound_high_idle(self, mock_util, mock_overview):
    mock_overview.return_value = json.dumps({
        "steptime_ms_average": "100.0",
        "tc_idle_ms_average": "25.0",  # 25% > 10%
        "device_duty_cycle_percent": "75.0",
    })
    mock_util.return_value = json.dumps({})

    result = json.loads(check_host_boundness_tool.check_host_boundness("sess1"))
    self.assertEqual(result["status"], "HOST_BOUND")
    self.assertEqual(result["metrics"]["tc_idle_percent"], 25.0)

  @mock.patch.object(get_overview_tool, "get_overview")
  @mock.patch.object(get_utilization_viewer_tool, "get_utilization_viewer")
  def test_not_host_bound(self, mock_util, mock_overview):
    mock_overview.return_value = json.dumps({
        "steptime_ms_average": "100.0",
        "tc_idle_ms_average": "2.0",
        "device_duty_cycle_percent": "98.0",
    })
    mock_util.return_value = json.dumps({})

    result = json.loads(check_host_boundness_tool.check_host_boundness("sess1"))
    self.assertEqual(result["status"], "NOT_HOST_BOUND")

  @mock.patch.object(get_overview_tool, "get_overview")
  def test_error(self, mock_overview):
    mock_overview.return_value = json.dumps({"error": "Session not found"})
    result = json.loads(check_host_boundness_tool.check_host_boundness("sess1"))
    self.assertEqual(result["status"], "UNKNOWN")
    self.assertIn("error", result)


if __name__ == "__main__":
  absltest.main()
