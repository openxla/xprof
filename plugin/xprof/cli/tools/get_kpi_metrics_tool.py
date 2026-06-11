"""Tool to fetch consolidated KPI metrics from XProf.

This tool combines data from overview and memory profile to provide
a concise set of key performance indicators.
"""

import collections
import json
import logging
import traceback

from xprof.cli.tools import get_memory_profile_tool
from xprof.cli.tools import get_overview_tool


def get_kpi_metrics(session_id: str) -> str:
  """Creates a consolidated KPI metrics JSON for a specific session.

  Args:
      session_id: The unique XProf session ID.

  Returns:
      A JSON-formatted string containing the KPI metrics:
        - step_time_ms: The average step time in milliseconds.
        - duty_cycle_percent: The device duty cycle as a percentage.
        - mxu_utilization_percent: MXU utilization as a percentage.
        - roofline_utilization: Flop rate utilization relative to roofline.
        - peak_hbm_gib: Peak memory usage in GiB.
        - accelerator_info: Dictionary with device_type and device_core_count.
        - error: (Optional) Error message if the operation failed.
  """
  overview_data = {}
  memory_data = {}
  try:
    overview_json = get_overview_tool.get_overview(session_id)
    memory_json = get_memory_profile_tool.get_memory_profile(session_id)

    overview_data = json.loads(overview_json)
    memory_data = json.loads(memory_json)
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.exception(
        "Error fetching or parsing XProf tool outputs for session %s",
        session_id,
    )
    error_message = "".join(traceback.format_exception_only(type(e), e)).strip()
    return json.dumps(
        {
            "error": f"Error: {error_message}",
            "traceback": traceback.format_exc(),
        },
        indent=2,
    )

  if "error" in overview_data:
    return json.dumps(
        {"error": f"Error in get_overview: {overview_data['error']}"},
        indent=2,
    )

  if "error" in memory_data:
    logging.warning(
        "get_memory_profile failed for session %s: %s",
        session_id,
        memory_data["error"],
    )
    peak_hbm_data = {"peak_memory_usage_gib": "N/A"}
  else:
    peak_hbm_data = memory_data

  perf_summary_raw = overview_data.get("performance_summary", {})
  perf_summary = collections.defaultdict(lambda: "N/A", perf_summary_raw)

  run_env_raw = overview_data.get("run_environment", {})
  run_env = collections.defaultdict(lambda: "N/A", run_env_raw)

  peak_hbm = collections.defaultdict(lambda: "N/A", peak_hbm_data)

  return json.dumps(
      {
          "step_time_ms": perf_summary["steptime_ms_average"],
          "duty_cycle_percent": perf_summary["device_duty_cycle_percent"],
          "mxu_utilization_percent": perf_summary["mxu_utilization_percent"],
          "roofline_utilization": perf_summary[
              "flop_rate_utilization_relative_to_roofline"
          ],
          "peak_hbm_gib": peak_hbm["peak_memory_usage_gib"],
          "accelerator_info": {
              "device_type": run_env["device_type"],
              "device_core_count": run_env["device_core_count"],
          },
      },
      indent=2,
  )
