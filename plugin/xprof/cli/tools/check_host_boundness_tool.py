"""Tool to check if an XProf session is host-bound using numerical thresholds."""

import json
import logging
import traceback
from typing import Any

from xprof.cli.internal import decorators
from xprof.cli.tools import get_overview_tool
from xprof.cli.tools import get_utilization_viewer_tool

# Official thresholds from smart suggestion C++ constants.h
_INFEED_PERCENTAGE_THRESHOLD: float = 10.0
_TENSOR_CORE_IDLE_TIME_THRESHOLD_PERCENT: float = 10.0
_DUTY_CYCLE_LOW_THRESHOLD_PERCENT: float = 50.0
_MXU_IDLENESS_HIGH_THRESHOLD_PERCENT: float = 70.0
_HBM_BW_LOW_THRESHOLD_PERCENT: float = 30.0


def _parse_tool_response(response_str: str) -> dict[str, Any]:
  """Parses JSON string response from a tool call safely."""
  try:
    parsed = json.loads(response_str)
    if isinstance(parsed, dict):
      return parsed
    if isinstance(parsed, list) and parsed:
      if isinstance(parsed[0], dict):
        return parsed[0]
  except json.JSONDecodeError:
    pass
  return {}


@decorators.cached(expire=86400)
def check_host_boundness(session_id: str) -> str:
  """Checks whether an XProf session is host-bound or input-bound.

  Evaluates exact numerical metrics from overview and utilization profile views
  against canonical hardware diagnostic rules (e.g., duty cycle, TensorCore
  idleness, and HBM/MXU utilization ratios).

  Args:
      session_id: The unique XProf session ID.

  Returns:
      A JSON-formatted string containing 'status' ('HOST_BOUND',
      'NOT_HOST_BOUND', or 'UNKNOWN'), numerical 'metrics', 'reasons', and
      actionable 'recommendations'.
  """
  try:
    overview_str = get_overview_tool.get_overview(session_id)
    overview = _parse_tool_response(overview_str)
    if not overview or "error" in overview:
      error_msg = overview.get(
          "error", f"Failed to get overview for {session_id}"
      )
      return json.dumps(
          {"status": "UNKNOWN", "error": error_msg},
          indent=2,
      )

    util_str = get_utilization_viewer_tool.get_utilization_viewer(session_id)
    util = _parse_tool_response(util_str)

    # Extract deterministic numbers safely
    step_time_ms = float(overview.get("steptime_ms_average", 0.0) or 0.0)
    tc_idle_ms = float(overview.get("tc_idle_ms_average", 0.0) or 0.0)
    tc_infeed_ms = float(overview.get("tc_infeed_ms_average", 0.0) or 0.0)
    duty_cycle = float(
        overview.get("device_duty_cycle_percent", 100.0) or 100.0
    )
    host_idle_time = float(overview.get("host_idle_time_percent", 0.0) or 0.0)

    mxu_idleness = float(util.get("idleness_percent", 0.0) or 0.0)
    hbm_bw_util = float(
        util.get("hbm_bandwidth_utilization_percent", 100.0) or 100.0
    )
    ici_read_util = float(
        util.get("ici_read_utilization_percent", 100.0) or 100.0
    )

    tc_idle_percent = (
        (tc_idle_ms / step_time_ms * 100.0) if step_time_ms > 0 else 0.0
    )
    tc_infeed_percent = (
        (tc_infeed_ms / step_time_ms * 100.0) if step_time_ms > 0 else 0.0
    )

    reasons: list[str] = []
    recommendations: list[str] = []
    is_host_bound = False

    if tc_idle_percent > _TENSOR_CORE_IDLE_TIME_THRESHOLD_PERCENT:
      is_host_bound = True
      reasons.append(
          f"TensorCore idle time percentage ({tc_idle_percent:.1f}%) exceeds"
          f" threshold ({_TENSOR_CORE_IDLE_TIME_THRESHOLD_PERCENT:.1f}%)."
      )
      recommendations.append(
          "Reduce Python interpreter overhead or batch kernel execution calls."
      )

    if tc_infeed_percent > _INFEED_PERCENTAGE_THRESHOLD:
      is_host_bound = True
      reasons.append(
          f"Infeed waiting time percentage ({tc_infeed_percent:.1f}%) exceeds"
          f" threshold ({_INFEED_PERCENTAGE_THRESHOLD:.1f}%)."
      )
      recommendations.append(
          "Optimize input pipeline preprocessing using asynchronous prefetching"
          " or tf.data parallelism."
      )

    if (
        duty_cycle < _DUTY_CYCLE_LOW_THRESHOLD_PERCENT
        and mxu_idleness > _MXU_IDLENESS_HIGH_THRESHOLD_PERCENT
    ):
      if (
          hbm_bw_util < _HBM_BW_LOW_THRESHOLD_PERCENT
          and ici_read_util < _HBM_BW_LOW_THRESHOLD_PERCENT
      ):
        is_host_bound = True
        reasons.append(
            f"Low device duty cycle ({duty_cycle:.1f}%) combined with high MXU"
            f" idleness ({mxu_idleness:.1f}%) and low HBM bandwidth"
            f" ({hbm_bw_util:.1f}%) indicates device starvation."
        )

    if not is_host_bound and host_idle_time > 50.0:
      reasons.append(
          f"Host idle time ({host_idle_time:.1f}%) indicates potential non-host"
          " compute or kernel latency dominance."
      )

    status = "HOST_BOUND" if is_host_bound else "NOT_HOST_BOUND"

    return json.dumps(
        {
            "status": status,
            "metrics": {
                "step_time_ms_average": round(step_time_ms, 3),
                "device_duty_cycle_percent": round(duty_cycle, 2),
                "tc_idle_percent": round(tc_idle_percent, 2),
                "tc_infeed_percent": round(tc_infeed_percent, 2),
                "mxu_idleness_percent": round(mxu_idleness, 2),
                "hbm_bandwidth_utilization_percent": round(hbm_bw_util, 2),
            },
            "reasons": reasons,
            "recommendations": recommendations,
        },
        indent=2,
    )

  except Exception:  # pylint: disable=broad-exception-caught
    logging.exception(
        "Error checking host boundness for session %s", session_id
    )
    return json.dumps(
        {
            "status": "UNKNOWN",
            "error": (
                "Error during host boundness check:"
                f" {traceback.format_exc().strip()}"
            ),
        },
        indent=2,
    )
