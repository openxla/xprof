"""Tool to check if an XProf session is host-bound using multi-source telemetry."""

import json
import logging
import re
import traceback
from typing import Any

from xprof.cli.internal import decorators
from xprof.cli.internal.oss import xprof_client
from xprof.cli.tools import get_utilization_viewer_tool

# Four-condition threshold constants from canonical Lumini diagnostic pipeline
_IDLE_TIME_RATIO_HIGH_THRESHOLD_PERCENT: float = 10.0
_MXU_IDLENESS_HIGH_THRESHOLD_PERCENT: float = 70.0
_HBM_BW_LOW_THRESHOLD_PERCENT: float = 30.0
_ICI_BW_LOW_THRESHOLD_PERCENT: float = 30.0
_DUTY_CYCLE_HIGH_THRESHOLD_PERCENT: float = 50.0


def _parse_json_safely(response: Any) -> dict[str, Any]:
  """Parses JSON string or object safely with robust error handling and logging."""
  if not response:
    return {}
  if isinstance(response, dict):
    return response
  if isinstance(response, list):
    if response and isinstance(response[0], dict):
      return response[0]
    return {}
  if isinstance(response, bytes):
    try:
      response = response.decode("utf-8", errors="replace")
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.warning("Failed to decode bytes response: %s", e)
      return {}
  if not isinstance(response, str):
    return {}
  try:
    parsed = json.loads(response)
    if isinstance(parsed, dict):
      return parsed
    if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
      return parsed[0]
  except json.JSONDecodeError as e:
    logging.warning(
        "Failed to decode JSON response (error: %s): %s",
        e,
        response[:200],
    )
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.warning("Unexpected error parsing tool response: %s", e)
  return {}


def _traverse_and_sum_hlo_times(
    node: Any,
) -> tuple[float, float, float]:
  """Traverses HLO op profile AST to sum compute, HBM, and ICI times in picoseconds.

  Args:
      node: AST node from hlo_op_profile.json.

  Returns:
      A tuple of (compute_time_ps, hbm_time_ps, ici_time_ps).
  """
  compute_time = 0.0
  hbm_time = 0.0
  ici_time = 0.0

  if not isinstance(node, dict):
    return compute_time, hbm_time, ici_time

  metrics = node.get("metrics", {})
  occurrences = (
      metrics.get("occurrences", 0) if isinstance(metrics, dict) else 0
  )
  children = node.get("children", [])

  if not children and occurrences > 0:
    name = str(node.get("name", "")).lower()
    xla_info = node.get("xla", {}) if isinstance(node.get("xla"), dict) else {}
    category = str(xla_info.get("category", "")).lower()
    raw_time = (
        float(metrics.get("rawTime", 0.0) or 0.0)
        if isinstance(metrics, dict)
        else 0.0
    )

    ici_patterns = (
        "all-reduce",
        "all-gather",
        "all-to-all",
        "reduce-scatter",
        "collective-broadcast",
        "collective-permute",
    )
    hbm_patterns = ("copy-start", "copy", "copy-done")

    is_ici = any(pat in name or pat in category for pat in ici_patterns)
    is_hbm = any(pat in name or pat in category for pat in hbm_patterns)

    if is_ici:
      ici_time += raw_time
    elif is_hbm:
      hbm_time += raw_time
    else:
      compute_time += raw_time

  if isinstance(children, list):
    for child in children:
      if isinstance(child, dict):
        c_time, m_time, i_time = _traverse_and_sum_hlo_times(child)
        compute_time += c_time
        hbm_time += m_time
        ici_time += i_time

  return compute_time, hbm_time, ici_time


def _get_avg_barrier_cores_time_per_event(
    session_id: str, client: xprof_client.CachedXprofClient
) -> tuple[float, int]:
  """Calculates the average duration per barrier-cores event and total count."""
  try:
    hosts = client.get_hosts(session_id, with_metadata=True)
    target_host = None
    if isinstance(hosts, list):
      for host in hosts:
        if isinstance(host, dict) and host.get("hasDeviceTrace"):
          target_host = host.get("hostname")
          break
        elif isinstance(host, str):
          target_host = host
          break
      if not target_host and hosts:
        if isinstance(hosts[0], dict):
          target_host = hosts[0].get("hostname")
        elif isinstance(hosts[0], str):
          target_host = hosts[0]

    trace_filter_config = json.dumps({"device_regexes": ["TPU:0", "GPU:0"]})
    kwargs = {}
    if target_host:
      kwargs["hosts"] = target_host

    result = client.fetch(
        tool_name="trace_viewer.json",
        session_id=session_id,
        trace_filter_config=trace_filter_config,
        **kwargs,
    )
    if isinstance(result, tuple) and len(result) == 2:
      _, output = result
    else:
      output = result

    if not output:
      return 0.0, 0

    if isinstance(output, bytes):
      output = output.decode("utf-8", errors="replace")

    xprof_trace = json.loads(output) if isinstance(output, str) else output
    trace_events = (
        xprof_trace.get("traceEvents", [])
        if isinstance(xprof_trace, dict)
        else []
    )
    if not trace_events or not isinstance(trace_events, list):
      return 0.0, 0

    total_barrier_duration_us = 0.0
    barrier_event_count = 0
    for event in trace_events:
      if isinstance(event, dict) and event.get("name") == "barrier-cores":
        total_barrier_duration_us += float(event.get("dur", 0.0) or 0.0)
        barrier_event_count += 1

    if barrier_event_count == 0:
      return 0.0, 0

    avg_barrier_duration_ms = (
        total_barrier_duration_us / barrier_event_count
    ) / 1000.0
    return avg_barrier_duration_ms, barrier_event_count
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.warning(
        "Failed to fetch barrier cores duration for session %s: %s",
        session_id,
        e,
    )
    return 0.0, 0


def _get_utilization_metrics(
    session_id: str, host_count: int
) -> dict[str, float]:
  """Fetches utilization metrics with multi-host fallback."""
  idleness_percent = 0.0
  hbm_bw_util = 0.0
  ici_read_util = 0.0
  ici_write_util = 0.0

  max_hosts = max(1, min(host_count, 32))
  for h in range(max_hosts):
    try:
      util_str = get_utilization_viewer_tool.get_utilization_viewer(
          session_id, host=h
      )
      util_data = _parse_json_safely(util_str)
      if not util_data:
        break
      if "error" in util_data:
        logging.warning(
            "Session-level error fetching utilization viewer for %s: %s",
            session_id,
            util_data["error"],
        )
        break
      if util_data.get("status") == "NO_DATA":
        msg = util_data.get("message", "")
        if (
            "No data returned for session" in msg
            or "No hardware performance counter events found" in msg
        ):
          break
        continue

      if "message" not in util_data:
        if util_data.get("idleness_percent") is not None:
          idleness_percent = float(util_data["idleness_percent"])
        if util_data.get("hbm_bandwidth_utilization_percent") is not None:
          hbm_bw_util = float(util_data["hbm_bandwidth_utilization_percent"])
        if util_data.get("ici_read_utilization_percent") is not None:
          ici_read_util = float(util_data["ici_read_utilization_percent"])
        if util_data.get("ici_write_utilization_percent") is not None:
          ici_write_util = float(util_data["ici_write_utilization_percent"])
        if h != 0:
          logging.info("Default Host 0 had no data. Switched to Host %d.", h)
        break
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.warning(
          "Failed to fetch utilization viewer for host %d: %s", h, e
      )
      break

  return {
      "idleness_percent": idleness_percent,
      "hbm_bandwidth_utilization_percent": hbm_bw_util,
      "ici_read_utilization_percent": ici_read_util,
      "ici_write_utilization_percent": ici_write_util,
  }


@decorators.cached(expire=86400)
def check_host_boundness(session_id: str, func_name: str | None = None) -> str:
  """Diagnoses if a TPU workload is host-bound using canonical multi-source telemetry.

  Evaluates data across overview, HLO op profile, trace viewer (barrier cores),
  and utilization viewer against a 4-condition AND gate.

  Args:
      session_id: The unique XProf session ID.
      func_name: Optional name of the main step function for API compatibility.

  Returns:
      A JSON-formatted string containing 'status' ('HOST_BOUND',
      'NOT_HOST_BOUND', 'UNKNOWN', or 'INSUFFICIENT_DATA'), numerical 'metrics',
      'reasons', and actionable 'recommendations'.
  """
  session_id = str(session_id)
  client = xprof_client.get_client()

  try:
    # --- 1. Fetch & Parse Overview Data ---
    overview_raw = client.fetch(
        tool_name="overview_page.json",
        session_id=session_id,
        format="json",
    )
    if isinstance(overview_raw, tuple) and len(overview_raw) == 2:
      _, overview_output = overview_raw
    else:
      overview_output = overview_raw

    if not overview_output:
      return json.dumps(
          {
              "status": "UNKNOWN",
              "error": f"No overview data returned for session {session_id}",
          },
          indent=2,
      )

    if isinstance(overview_output, bytes):
      overview_output = overview_output.decode("utf-8", errors="replace")

    overview_json = (
        json.loads(overview_output)
        if isinstance(overview_output, str)
        else overview_output
    )

    if isinstance(overview_json, list):
      overview_p = (
          overview_json[0].get("p", {})
          if len(overview_json) > 0 and isinstance(overview_json[0], dict)
          else {}
      )
      step_time_p = (
          overview_json[1].get("p", {})
          if len(overview_json) > 1 and isinstance(overview_json[1], dict)
          else {}
      )
      run_env_p = (
          overview_json[2].get("p", {})
          if len(overview_json) > 2 and isinstance(overview_json[2], dict)
          else {}
      )
      number_of_steps = (
          len(overview_json[1].get("rows") or [])
          if len(overview_json) > 1 and isinstance(overview_json[1], dict)
          else 0
      )
    elif isinstance(overview_json, dict):
      overview_p = (
          overview_json.get("performance_summary")
          if isinstance(overview_json.get("performance_summary"), dict)
          else overview_json
      )
      run_env_p = (
          overview_json.get("run_environment")
          if isinstance(overview_json.get("run_environment"), dict)
          else overview_json
      )
      step_time_p = (
          overview_json.get("step_time")
          if isinstance(overview_json.get("step_time"), dict)
          else overview_json
      )
      rows = overview_json.get("rows") or []
      number_of_steps = int(
          overview_json.get(
              "number_of_steps", len(rows) if isinstance(rows, list) else 0
          )
          or 0
      )
    else:
      return json.dumps(
          {
              "status": "UNKNOWN",
              "error": (
                  f"Unexpected overview data format for session {session_id}"
              ),
          },
          indent=2,
      )

    duty_cycle_str = str(overview_p.get("device_duty_cycle_percent", "0.0%"))
    try:
      duty_cycle = float(duty_cycle_str.replace("%", "").strip())
    except (ValueError, TypeError):
      duty_cycle = 0.0

    core_count_str = str(run_env_p.get("device_core_count", "1"))
    match_core = re.search(r"\d+", core_count_str)
    core_count = max(1, int(match_core.group()) if match_core else 1)

    host_count_str = str(run_env_p.get("host_count", "1"))
    match_host = re.search(r"\d+", host_count_str)
    host_count = max(1, int(match_host.group()) if match_host else 1)

    step_time_val = (
        step_time_p.get("steptime_ms_average")
        or step_time_p.get("sc_step_time_ms_average")
        or "0.0"
    )
    match_step = re.search(r"[\d.]+", str(step_time_val))
    average_step_time_ms = float(match_step.group()) if match_step else 0.0

    total_duration_ms = average_step_time_ms * number_of_steps

    if total_duration_ms == 0.0 or number_of_steps == 0:
      return json.dumps(
          {
              "status": "INSUFFICIENT_DATA",
              "reasons": [
                  f"Session {session_id} lacks valid step timing duration"
                  " telemetry in overview_page.json."
              ],
              "recommendations": [
                  "Capture a new XProf trace with step profiling enabled to"
                  " measure host boundness."
              ],
          },
          indent=2,
      )

    # --- 2. HLO Op Profile Breakdown ---
    scaled_compute_time_ms = 0.0
    scaled_hbm_time_ms = 0.0
    scaled_ici_time_ms = 0.0
    hlo_data_available = False

    try:
      hlo_raw = client.fetch(
          tool_name="hlo_op_profile.json",
          session_id=session_id,
          group_by="category",
      )
      if isinstance(hlo_raw, tuple) and len(hlo_raw) == 2:
        _, hlo_output = hlo_raw
      else:
        hlo_output = hlo_raw

      if hlo_output:
        if isinstance(hlo_output, bytes):
          hlo_output = hlo_output.decode("utf-8", errors="replace")
        hlo_profile = (
            json.loads(hlo_output)
            if isinstance(hlo_output, str)
            else hlo_output
        )
        start_node = (
            hlo_profile.get("byCategory") or hlo_profile.get("by_category")
            if isinstance(hlo_profile, dict)
            else None
        )
        if start_node and isinstance(start_node, dict):
          hlo_data_available = True
          compute_time_ps, hbm_time_ps, ici_time_ps = (
              _traverse_and_sum_hlo_times(start_node)
          )
          # Convert ps to ms (divide by 1e9 ps/ms) and scale across core count
          scaled_compute_time_ms = (compute_time_ps / 1e9) / core_count
          scaled_hbm_time_ms = (hbm_time_ps / 1e9) / core_count
          scaled_ici_time_ms = (ici_time_ps / 1e9) / core_count
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.warning(
          "Failed to compute HLO ops breakdown for session %s: %s",
          session_id,
          e,
      )

    # --- 3. Barrier Cores Telemetry ---
    try:
      avg_barrier_ms_per_event, num_barrier_events = (
          _get_avg_barrier_cores_time_per_event(session_id, client)
      )
      if num_barrier_events > 0 and number_of_steps > 0:
        scaled_barrier_time_ms = avg_barrier_ms_per_event * number_of_steps
      else:
        scaled_barrier_time_ms = 0.0
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.warning(
          "Failed to fetch barrier cores duration for session %s: %s",
          session_id,
          e,
      )
      scaled_barrier_time_ms = 0.0

    # --- 4. Idle Time Calculations ---
    sum_scaled_hlo_ms = (
        scaled_compute_time_ms + scaled_hbm_time_ms + scaled_ici_time_ms
    )
    idle_time_ms = total_duration_ms - sum_scaled_hlo_ms
    pure_idle_time_ms = max(0.0, idle_time_ms - scaled_barrier_time_ms)

    active_compute_ms = total_duration_ms - pure_idle_time_ms
    idle_time_ratio = (
        pure_idle_time_ms / active_compute_ms if active_compute_ms > 0 else 0.0
    )
    idle_time_ratio_pct = idle_time_ratio * 100.0

    absolute_idle_fraction = (
        pure_idle_time_ms / total_duration_ms if total_duration_ms > 0 else 0.0
    )
    equivalent_idle_chips = core_count * absolute_idle_fraction

    # --- 5. Hardware Subsystem Utilization ---
    util_metrics = _get_utilization_metrics(session_id, host_count)
    idleness_percent = util_metrics["idleness_percent"]
    hbm_bw_util = util_metrics["hbm_bandwidth_utilization_percent"]
    ici_read_util = util_metrics["ici_read_utilization_percent"]
    ici_write_util = util_metrics["ici_write_utilization_percent"]

    # --- 6. Four-Condition AND Gate Decision ---
    is_pure_idle_high = (
        idle_time_ratio_pct > _IDLE_TIME_RATIO_HIGH_THRESHOLD_PERCENT
    )
    is_mxu_idle_high = idleness_percent > _MXU_IDLENESS_HIGH_THRESHOLD_PERCENT
    is_hbm_util_low = hbm_bw_util < _HBM_BW_LOW_THRESHOLD_PERCENT
    is_ici_util_low = (
        ici_read_util < _ICI_BW_LOW_THRESHOLD_PERCENT
        and ici_write_util < _ICI_BW_LOW_THRESHOLD_PERCENT
    )

    is_host_bound = (
        is_pure_idle_high
        and is_mxu_idle_high
        and is_hbm_util_low
        and is_ici_util_low
    )

    reasons: list[str] = []
    recommendations: list[str] = []

    if is_host_bound:
      status = "HOST_BOUND"
      reasons.append(
          "Workload is host-bound: Idle Time Ratio exceeds"
          f" {_IDLE_TIME_RATIO_HIGH_THRESHOLD_PERCENT:.1f}% of active compute."
      )
      recommendations.append(
          f"Opportunity Size: Hardware waste = {equivalent_idle_chips:.1f} idle"
          " chips."
      )
      recommendations.append(
          "Review data pipeline for inefficiencies (e.g., tf.data"
          " transformations or PyGrain)."
      )
      if func_name:
        recommendations.append(
            "Use Lumini xprof_check_host_boundness with func_name to run"
            " automated step dispersion analysis."
        )
      else:
        recommendations.append(
            "Provide `func_name` to enable automated Dispersion Analysis and"
            " check for transient stalls."
        )
    elif duty_cycle > _DUTY_CYCLE_HIGH_THRESHOLD_PERCENT and (
        not hlo_data_available or not is_pure_idle_high
    ):
      if not hlo_data_available:
        status = "UNKNOWN"
        reasons.append(
            f"TPU duty cycle is high ({duty_cycle:.1f}%), but HLO telemetry is"
            " missing. Unable to determine if workload is host-bound."
        )
      else:
        status = "NOT_HOST_BOUND"
        reasons.append(
            f"TPU duty cycle ({duty_cycle:.1f}%) is high and idle ratio is low."
        )
    else:
      status = "NOT_HOST_BOUND"
      reasons.append(
          "Workload is not host-bound. One or more host-bound condition"
          " thresholds were not met:"
      )
      if not is_pure_idle_high:
        reasons.append(
            f"- Idle Time Ratio ({idle_time_ratio_pct:.1f}%) is <="
            f" {_IDLE_TIME_RATIO_HIGH_THRESHOLD_PERCENT:.1f}%."
        )
      if not is_mxu_idle_high:
        reasons.append(
            f"- MXU Idleness ({idleness_percent:.1f}%) is <="
            f" {_MXU_IDLENESS_HIGH_THRESHOLD_PERCENT:.1f}%."
        )
      if not is_hbm_util_low:
        reasons.append(
            f"- HBM Bandwidth Utilization ({hbm_bw_util:.1f}%) is >="
            f" {_HBM_BW_LOW_THRESHOLD_PERCENT:.1f}%."
        )
      if not is_ici_util_low:
        reasons.append(
            "- ICI Utilization"
            f" (Read: {ici_read_util:.1f}%, Write: {ici_write_util:.1f}%) is >="
            f" {_ICI_BW_LOW_THRESHOLD_PERCENT:.1f}%."
        )

    return json.dumps(
        {
            "status": status,
            "metrics": {
                "tpu_duty_cycle_percent": round(duty_cycle, 2),
                "idle_time_ratio_percent": round(idle_time_ratio_pct, 2),
                "equivalent_idle_chips": round(equivalent_idle_chips, 2),
                "mxu_idleness_percent": round(idleness_percent, 2),
                "hbm_bandwidth_utilization_percent": round(hbm_bw_util, 2),
                "ici_read_utilization_percent": round(ici_read_util, 2),
                "ici_write_utilization_percent": round(ici_write_util, 2),
                "scaled_compute_time_ms": round(scaled_compute_time_ms, 2),
                "scaled_hbm_time_ms": round(scaled_hbm_time_ms, 2),
                "scaled_ici_time_ms": round(scaled_ici_time_ms, 2),
                "scaled_barrier_time_ms": round(scaled_barrier_time_ms, 2),
                "pure_idle_time_ms": round(pure_idle_time_ms, 2),
                "total_duration_ms": round(total_duration_ms, 2),
                "number_of_steps": number_of_steps,
                "core_count": core_count,
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
