"""Tool to fetch consolidated KPI metrics from XProf.

This tool combines data from overview and memory profile to provide
a concise set of key performance indicators.
"""

import collections
import json
import logging

from xprof.cli.internal import trace_selection
from xprof.cli.tools import get_memory_profile_tool
from xprof.cli.tools import get_overview_tool
from xprof.cli.tools import get_roofline_model_tool


def get_kpi_metrics(session_id: str, *, bypass_cache: bool = False) -> str:
  """Creates a consolidated KPI metrics JSON for a specific session.

  Args:
      session_id: The unique XProf session ID.
      bypass_cache: Whether to bypass cache and recompute metrics.

  Returns:
      A JSON-formatted string containing the KPI metrics:
        - step_time_ms: The average step time in milliseconds.
        - duty_cycle_percent: The device duty cycle as a percentage.
        - mxu_utilization_percent: MXU utilization from hardware performance
          counters, as a percentage. None if the counters were not captured.
          This value never comes from the XLA cost model.
        - mxu_utilization_status: (Optional) Present only when
          mxu_utilization_percent is None. Has 'status', 'reason' and
          'how_to_get'.
        - compute_efficiency_percent: (Optional) Roofline FLOP rate relative
          to the peak device FLOP rate, from the XLA cost model. Present when
          roofline data exists.
        - roofline_utilization: Flop rate utilization relative to roofline.
        - flops_provenance: Provenance of FLOP calculations ('xla_cost_model',
          'derived_from_shapes', 'mixed', or 'opaque_custom_call').
        - derived_flops_fraction: Share of program FLOPs that is derived from
          custom call shapes (0.0 to 1.0). When flops_provenance is
          'derived_from_shapes' or 'mixed', roofline_utilization is the
          corrected get_roofline_model efficiency. The correction replaces
          only custom call FLOPs and keeps cost-model FLOPs for the other ops,
          so it is exact at any derived share.
        - peak_hbm_gib: Peak memory usage in GiB.
        - accelerator_info: Dictionary with device_type and device_core_count.
        - error: (Optional) Error message if the operation failed.
  """
  overview_data = {}
  memory_data = {}
  try:
    overview_json = get_overview_tool.get_overview(
        session_id, bypass_cache=bypass_cache
    )
    overview_data = json.loads(overview_json)
  except (FileNotFoundError, ValueError):
    raise
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.exception(
        "Error fetching overview for session %s in get_kpi_metrics", session_id
    )
    raise RuntimeError(f"Error in get_overview: {e}") from e

  try:
    memory_json = get_memory_profile_tool.get_memory_profile(
        session_id, bypass_cache=bypass_cache
    )
    memory_data = json.loads(memory_json)
    peak_hbm_data = memory_data
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.warning(
        "get_memory_profile failed for session %s: %s",
        session_id,
        e,
    )
    peak_hbm_data = {"peak_memory_usage_gib": "N/A"}

  perf_summary_raw = overview_data.get("performance_summary", {})
  perf_summary = collections.defaultdict(lambda: "N/A", perf_summary_raw)

  run_env_raw = overview_data.get("run_environment", {})
  run_env = collections.defaultdict(lambda: "N/A", run_env_raw)

  peak_hbm = collections.defaultdict(lambda: "N/A", peak_hbm_data)

  roofline_util = perf_summary["flop_rate_utilization_relative_to_roofline"]
  compute_eff = perf_summary_raw.get("compute_efficiency_percent")
  flops_provenance = "xla_cost_model"
  derived_flops_fraction = 0.0
  try:
    roofline_json = get_roofline_model_tool.get_roofline_model(
        session_id, bypass_cache=bypass_cache
    )
    roofline_data = json.loads(roofline_json)
    prog_roofline = roofline_data.get("program", {})
    if prog_roofline.get("flops_provenance"):
      flops_provenance = prog_roofline["flops_provenance"]
    derived_flops_fraction = float(
        prog_roofline.get("derived_flops_fraction") or 0.0
    )
    if flops_provenance in ("derived_from_shapes", "mixed"):
      roofline_util = (
          prog_roofline.get("roofline_efficiency_percent") or roofline_util
      )
      compute_eff = (
          prog_roofline.get("compute_efficiency_percent") or compute_eff
      )
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.debug(
        "get_roofline_model fallback in get_kpi_metrics failed for %s: %s",
        session_id,
        e,
    )

  payload: dict[str, object] = {
      "step_time_ms": perf_summary["steptime_ms_average"],
      "duty_cycle_percent": perf_summary["device_duty_cycle_percent"],
      "mxu_utilization_percent": perf_summary["mxu_utilization_percent"],
      "roofline_utilization": roofline_util,
      "flops_provenance": flops_provenance,
      "derived_flops_fraction": derived_flops_fraction,
      "peak_hbm_gib": peak_hbm["peak_memory_usage_gib"],
      "accelerator_info": {
          "device_type": run_env["device_type"],
          "device_core_count": run_env["device_core_count"],
      },
  }
  if "mxu_utilization_status" in perf_summary_raw:
    payload["mxu_utilization_status"] = perf_summary_raw[
        "mxu_utilization_status"
    ]
  if compute_eff is not None:
    payload["compute_efficiency_percent"] = compute_eff
  if "custom_call_warning" in perf_summary_raw:
    payload["custom_call_share_pct"] = perf_summary_raw.get(
        "custom_call_share_pct"
    )
    payload["custom_call_warning"] = perf_summary_raw["custom_call_warning"]
  if "step_time_scope" in perf_summary_raw:
    per_host_raw = perf_summary_raw.get("per_host_step_time_ms")
    num_hosts: int | None = (
        len(per_host_raw)
        if isinstance(per_host_raw, dict) and len(per_host_raw) >= 2
        else None
    )
    if num_hosts is None:
      try:
        parsed_hosts = int(run_env_raw.get("host_count", 0))
        if parsed_hosts >= 2:
          num_hosts = parsed_hosts
      except (ValueError, TypeError):
        num_hosts = None
    payload["step_time_scope"] = perf_summary_raw["step_time_scope"]
    payload["step_time_warning"] = trace_selection.format_step_time_warning(
        trace_selection.traits_for("get_kpi_metrics").step_time_fields,
        num_hosts=num_hosts,
    )
  if isinstance(perf_summary_raw.get("per_host_step_time_ms"), dict):
    payload["per_host_step_time_ms"] = {
        host: (
            stats.get("steptime_ms_average")
            if isinstance(stats, dict)
            else stats
        )
        for host, stats in perf_summary_raw["per_host_step_time_ms"].items()
    }

  return json.dumps(
      payload,
      indent=2,
  )
