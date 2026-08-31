"""Tool to fetch and parse Roofline Model analysis from XProf."""

import json
import logging
import re
import traceback
from typing import Any

from xprof.cli.internal import decorators
from xprof.cli.internal.oss import xprof_client


def _strip_html_tags(text: str) -> str:
  """Strips HTML tags like <div ...>...</div> from text."""
  if not text or not isinstance(text, str):
    return ""
  match = re.search(r"title=['\"]([^'\"]+)['\"]", text)
  if match:
    return match.group(1).replace("\n", " -> ")
  return re.sub(r"<[^>]+>", "", text).strip()


@decorators.cached(expire=86400)
def get_roofline_model(
    session_id: str,
    *,
    top_n: int = 15,
    group_by: str = "program",
    bypass_cache: bool = False,
) -> str:
  """Fetches and summarizes Roofline Model analysis for the session.

  **Use this** to identify hardware compute vs memory bandwidth bottlenecks
  at both the program level and per-operation level. It surfaces operational
  intensity (FLOP/Byte), roofline efficiency, compute efficiency, memory
  bandwidth utilization, and the primary bottleneck bound ('HBM', 'Compute',
  'CMEM', 'VMEM').

  Args:
      session_id: The unique XProf session ID or trace path.
      top_n: Number of top bottleneck operations to return (default is 15).
      group_by: Grouping level ('program' or 'step'). Default is 'program'.

  Returns:
      A JSON-formatted string containing:
        - 'program': Overall program roofline metrics (roofline_efficiency,
          compute_efficiency, max_mem_bw_utilization, measured_flop_rate,
          measured_memory_bw, operational_intensity, bound_by).
        - 'device_info': Device hardware specs (device_type, peak_flop_rate,
          peak_hbm_bw, ridge_points).
        - 'top_operations': List of top N operations sorted by self-time with
          per-op operational intensity, efficiencies, and bottleneck bounds.
  """
  del group_by  # Standard roofline analysis processes full session op stats.
  client = xprof_client.get_client()
  try:
    result = client.fetch(
        tool_name="roofline_model.json",
        session_id=session_id,
        bypass_cache=bypass_cache,
    )
    if not result or (isinstance(result, tuple) and not result[1]):
      result = client.fetch(
          tool_name="roofline_model",
          session_id=session_id,
          bypass_cache=bypass_cache,
      )

    if isinstance(result, tuple) and len(result) == 2:
      _, data = result
    else:
      data = result

    if not data:
      return json.dumps(
          dict(
              status="NO_DATA",
              message=(
                  f"No roofline model data found for session {session_id!r}."
              ),
          ),
          indent=2,
      )

    if isinstance(data, bytes):
      data = data.decode("utf-8", errors="replace")

    roofline_data = json.loads(data)
    if not isinstance(roofline_data, list) or not roofline_data:
      raise ValueError(
          "Unexpected roofline model data format: expected non-empty list"
      )

    table_data = roofline_data[0]
    cols = [col.get("id", "") for col in table_data.get("cols", [])]
    rows = table_data.get("rows", [])
    custom_props = table_data.get("p", {})

    device_info = {}
    for k, v in custom_props.items():
      try:
        device_info[k] = float(v)
      except (ValueError, TypeError):
        device_info[k] = v

    if not rows:
      return json.dumps(
          dict(
              status="NO_DATA",
              message="Roofline model table has no rows",
              device_info=device_info,
          ),
          indent=2,
      )

    def row_to_dict(row_cells: Any) -> dict[str, Any]:
      vals = [c.get("v") if isinstance(c, dict) else c for c in row_cells]
      return dict(zip(cols, vals))

    prog_dict = row_to_dict(rows[0].get("c", []))

    def safe_float(val: Any, default: float = 0.0) -> float:
      if val is None:
        return default
      try:
        return float(val)
      except (ValueError, TypeError):
        return default

    def to_percent_str(val: Any) -> str:
      f = safe_float(val)
      return f"{f * 100.0:.2f}%"

    def calc_mem_util_str(
        util_key: str, bw_key: str, peak_keys: list[str]
    ) -> str:
      val = prog_dict.get(util_key)
      if val is not None and safe_float(val) > 0:
        return to_percent_str(val)
      bw = safe_float(prog_dict.get(bw_key))
      if bw == 0.0 and bw_key.startswith("hbm_"):
        bw = safe_float(prog_dict.get("hbm_bw"))
      peak = 0.0
      for pk in peak_keys:
        if pk in device_info and safe_float(device_info[pk]) > 0:
          peak = safe_float(device_info[pk])
          break
      if peak > 0:
        return f"{(bw / peak) * 100.0:.2f}%"
      if val is not None:
        return to_percent_str(val)
      return "N/A"

    program_metrics = {
        "bound_by": prog_dict.get("bound_by", "Unknown"),
        "operational_intensity_flop_per_byte": round(
            safe_float(prog_dict.get("operational_intensity")), 4
        ),
        "bottleneck_operational_intensity_flop_per_byte": round(
            safe_float(prog_dict.get("bottleneck_operational_intensity")), 4
        ),
        "roofline_efficiency_percent": to_percent_str(
            prog_dict.get("roofline_efficiency")
        ),
        "compute_efficiency_percent": to_percent_str(
            prog_dict.get("compute_efficiency")
        ),
        "max_mem_bw_utilization_percent": to_percent_str(
            prog_dict.get("max_mem_bw_utilization")
        ),
        "optimal_flop_rate_gflops": round(
            safe_float(prog_dict.get("optimal_flop_rate")), 2
        ),
        "dma_stall_percent": to_percent_str(prog_dict.get("dma_stall_percent")),
        "measured_flop_rate_gflops": round(
            safe_float(prog_dict.get("measured_flop_rate")), 2
        ),
        "model_flop_rate_gflops": round(
            safe_float(prog_dict.get("model_flop_rate")), 2
        ),
        "measured_memory_bw_gibs": round(
            safe_float(prog_dict.get("measured_memory_bw")), 2
        ),
        "hbm_bw_gibs": round(safe_float(prog_dict.get("hbm_bw")), 2),
        "hbm_read_bw_utilization_percent": calc_mem_util_str(
            "hbm_read_bw_utilization",
            "hbm_read_bw",
            ["peak_hbm_read_bw", "peak_hbm_bw"],
        ),
        "hbm_write_bw_utilization_percent": calc_mem_util_str(
            "hbm_write_bw_utilization",
            "hbm_write_bw",
            ["peak_hbm_write_bw", "peak_hbm_bw"],
        ),
        "cmem_read_bw_utilization_percent": calc_mem_util_str(
            "cmem_read_bw_utilization",
            "cmem_read_bw",
            ["peak_cmem_read_bw", "peak_cmem_bw"],
        ),
        "cmem_write_bw_utilization_percent": calc_mem_util_str(
            "cmem_write_bw_utilization",
            "cmem_write_bw",
            ["peak_cmem_write_bw", "peak_cmem_bw"],
        ),
        "vmem_read_bw_utilization_percent": calc_mem_util_str(
            "vmem_read_bw_utilization",
            "vmem_read_bw",
            ["peak_vmem_read_bw", "peak_vmem_bw"],
        ),
        "vmem_write_bw_utilization_percent": calc_mem_util_str(
            "vmem_write_bw_utilization",
            "vmem_write_bw",
            ["peak_vmem_write_bw", "peak_vmem_bw"],
        ),
        "total_time_ms": round(
            safe_float(prog_dict.get("total_time")) / 1000.0, 3
        ),
    }

    op_records = []
    for r in rows[1:]:
      r_dict = row_to_dict(r.get("c", []))
      self_time_us = safe_float(r_dict.get("total_self_time"))
      if self_time_us <= 0:
        continue

      source_info_raw = r_dict.get("source_info", "")
      cleaned_source = _strip_html_tags(source_info_raw)

      op_name = r_dict.get("operation") or r_dict.get("hlo_name", "")
      op_category = r_dict.get("category") or r_dict.get("hlo_category", "")
      bound_by_val = r_dict.get("bound_by") or "Unknown"
      if bound_by_val == "Unknown" and (
          op_name.startswith("custom-call")
          or op_category.lower() in ("custom-call", "custom_call")
      ):
        bound_by_val = "CustomCall (opaque)"

      op_records.append({
          "rank": int(safe_float(r_dict.get("rank"))),
          "name": op_name,
          "category": op_category,
          "total_self_time_ms": round(self_time_us / 1000.0, 3),
          "total_self_time_percent": to_percent_str(
              r_dict.get("total_self_time_percent")
          ),
          "operational_intensity_flop_per_byte": round(
              safe_float(r_dict.get("operational_intensity")), 4
          ),
          "bottleneck_operational_intensity_flop_per_byte": round(
              safe_float(r_dict.get("bottleneck_operational_intensity")), 4
          ),
          "roofline_efficiency_percent": to_percent_str(
              r_dict.get("roofline_efficiency")
          ),
          "compute_efficiency_percent": to_percent_str(
              r_dict.get("compute_efficiency")
          ),
          "max_mem_bw_utilization_percent": to_percent_str(
              r_dict.get("max_mem_bw_utilization")
          ),
          "optimal_flop_rate_gflops": round(
              safe_float(r_dict.get("optimal_flop_rate")), 2
          ),
          "dma_stall_percent": to_percent_str(r_dict.get("dma_stall_percent")),
          "bound_by": bound_by_val,
          "hlo_module_id": str(r_dict.get("hlo_module_id", "")),
          "source_info": cleaned_source,
      })

    # Deduplicate operations by (rank, name)
    seen_ops = set()
    unique_op_records = []
    for op in op_records:
      op_key = (op["rank"], op["name"])
      if op_key not in seen_ops:
        seen_ops.add(op_key)
        unique_op_records.append(op)

    unique_op_records.sort(key=lambda x: x["total_self_time_ms"], reverse=True)
    top_ops = unique_op_records[:top_n]

    has_custom_call = any(
        op.get("bound_by") == "CustomCall (opaque)" for op in top_ops
    )
    if has_custom_call:
      if program_metrics.get("bound_by") in ("Unknown", "", None):
        program_metrics["bound_by"] = "CustomCall (opaque)"

    output: dict[str, Any] = {
        "program": program_metrics,
        "device_info": device_info,
        "top_operations": top_ops,
        "total_operations_analyzed": len(unique_op_records),
    }
    if has_custom_call:
      output["guidance"] = (
          "Op-level metrics unavailable for custom calls. Use"
          " get_llo_analysis, get_llo_debug_string, and aggregate_xplane_events"
          " for Pallas kernels."
      )

    return json.dumps(output, indent=2)

  except (FileNotFoundError, ValueError):
    raise
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.exception(
        "Error fetching roofline model for session %s", session_id
    )
    error_msg = "".join(traceback.format_exception_only(type(e), e)).strip()
    raise RuntimeError(f"Error fetching roofline model: {error_msg}") from e
