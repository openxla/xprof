"""Unit tests for get_kernel_stats_tool in Google3."""

import json
from unittest import mock
import pandas as pd
from absl.testing import absltest
from xprof.cli.internal import decorators
from xprof.cli.internal.oss import events_db_tools
from xprof.cli.tools import get_kernel_stats_tool
from xprof.cli.tools.oss import f1_utils


class GetKernelStatsToolTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    mock_cache = mock.create_autospec(decorators.Cache, instance=True)
    mock_cache.get.return_value = decorators.Cache.UNKNOWN
    self.enter_context(
        mock.patch.object(decorators, "get_cache", return_value=mock_cache)
    )

  @mock.patch.object(events_db_tools, "get_events_db_session_root")
  @mock.patch.object(f1_utils, "execute_f1_query")
  def test_get_kernel_stats_json(self, mock_execute, mock_root):
    mock_root.return_value = events_db_tools.EventsDbSessionRootResult(
        status="success", session_root="/blob/sess1"
    )
    mock_df = pd.DataFrame([{
        "kernel_name": "matmul",
        "total_duration_us": 1200.5,
        "execution_count": 10,
        "avg_duration_us": 120.05,
    }])
    mock_execute.return_value = mock_df

    result = json.loads(
        get_kernel_stats_tool.get_kernel_stats("sess1", limit=1)
    )
    self.assertLen(result, 1)
    self.assertEqual(result[0]["kernel_name"], "matmul")
    result_compute = json.loads(
        get_kernel_stats_tool.compute_kernel_stats("sess1", limit=1)
    )
    self.assertLen(result_compute, 1)
    self.assertEqual(result_compute[0]["kernel_name"], "matmul")

  @mock.patch.object(events_db_tools, "get_events_db_session_root")
  @mock.patch.object(f1_utils, "execute_f1_query")
  def test_get_avg_step_time(self, mock_execute, mock_root):
    mock_root.return_value = events_db_tools.EventsDbSessionRootResult(
        status="success", session_root="/blob/sess1"
    )
    mock_df = pd.DataFrame([{"avg_step_time_ms": 15.42, "step_count": 100}])
    mock_execute.return_value = mock_df

    result = json.loads(
        get_kernel_stats_tool.get_avg_step_time("sess1", func_name="train_step")
    )
    self.assertEqual(result["avg_step_time_ms"], 15.42)
    self.assertEqual(result["step_count"], 100)
    result_compute = json.loads(
        get_kernel_stats_tool.compute_avg_step_time(
            "sess1", func_name="train_step"
        )
    )
    self.assertEqual(result_compute["avg_step_time_ms"], 15.42)
    self.assertEqual(result_compute["step_count"], 100)

  @mock.patch(
      "google3.third_party.xprof.plugin.xprof.cli.internal.oss.kernel_stats_tools.get_kernel_stats"
  )
  def test_compute_kernel_stats_in_memory_object(self, mock_oss_get):
    """Tests that in-memory objects are forwarded to the OSS engine."""
    mock_oss_get.return_value = {
        "total_device_duration_ns": 1500,
        "total_device_duration_us": 1.5,
        "total_device_duration_ms": 0.0015,
        "kernel_records": [
            {"kernel_name": "test_op", "total_duration_us": 1.5}
        ],
        "step_durations_us": [1.5],
        "stats": {"mean_us": 1.5, "std_us": 0.0},
    }

    mock_profile_data = mock.MagicMock()
    mock_profile_data.planes = []

    result = get_kernel_stats_tool.compute_kernel_stats(
        mock_profile_data,
        output_format="dict",
        include_summary=True,
    )

    mock_oss_get.assert_called_once()
    call_kwargs = mock_oss_get.call_args
    self.assertIs(call_kwargs[0][0], mock_profile_data)
    self.assertTrue(call_kwargs[1]["include_summary"])
    self.assertIn("total_device_duration_ns", result)

  @mock.patch(
      "google3.third_party.xprof.plugin.xprof.cli.internal.oss.kernel_stats_tools.get_kernel_stats"
  )
  def test_get_kernel_stats_with_trace_matchers(self, mock_oss_get):
    """Tests that trace_matchers are passed through to the engine."""
    mock_oss_get.return_value = []

    get_kernel_stats_tool.compute_kernel_stats(
        "/local/path",
        output_format="dict",
        trace_matchers=("matmul", "conv"),
    )

    call_kwargs = mock_oss_get.call_args
    self.assertEqual(
        call_kwargs[1]["trace_matchers"], ("matmul", "conv")
    )

  def test_get_kernel_stats_remote_include_summary_raises(self):
    with self.assertRaisesRegex(
        ValueError, "include_summary=True is not supported for remote F1 SQL"
    ):
      get_kernel_stats_tool.get_kernel_stats("sess1", include_summary=True)

  @mock.patch.object(events_db_tools, "get_events_db_session_root")
  @mock.patch.object(f1_utils, "execute_f1_query")
  def test_get_kernel_stats_f1_sql_matchers(self, mock_execute, mock_root):
    mock_root.return_value = events_db_tools.EventsDbSessionRootResult(
        status="success", session_root="/blob/sess1"
    )
    mock_df = pd.DataFrame([{
        "kernel_name": "matmul",
        "total_duration_us": 1200.5,
        "execution_count": 10,
        "avg_duration_us": 120.05,
    }])
    mock_execute.return_value = mock_df

    get_kernel_stats_tool.compute_kernel_stats(
        "sess1", trace_matchers=("matmul", "conv")
    )
    mock_execute.assert_called_once()
    query_script = mock_execute.call_args[0][0].query()
    self.assertIn("tf_op_name LIKE '%matmul%'", query_script)
    self.assertIn("tf_op_name LIKE '%conv%'", query_script)

  @mock.patch.object(events_db_tools, "get_events_db_session_root")
  @mock.patch.object(f1_utils, "execute_f1_query")
  def test_get_kernel_stats_markdown_formatting(self, mock_execute, mock_root):
    mock_root.return_value = events_db_tools.EventsDbSessionRootResult(
        status="success", session_root="/blob/sess1"
    )
    mock_df = pd.DataFrame([{
        "kernel_name": "matmul",
        "total_duration_us": 1200.5,
        "execution_count": 10,
        "avg_duration_us": 120.05,
    }])
    mock_execute.return_value = mock_df

    result = get_kernel_stats_tool.compute_kernel_stats(
        "sess1", output_format="markdown"
    )
    self.assertIn("| Kernel | Total Duration (us) |", result)
    self.assertIn("`matmul`", result)


if __name__ == "__main__":
  absltest.main()
