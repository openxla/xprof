"""Tool to fetch and parse Roofline Model analysis from XProf."""

import json
import logging
import re
import traceback
from typing import Any

from google.protobuf import json_format
from xprof.cli.internal import decorators
from xprof.cli.internal import hlo_shape_utils
from xprof.cli.internal.oss import xprof_client
from xprof.protobuf import op_profile_pb2


def _fetch_op_profile_expressions(
    session_id: str, client: Any, bypass_cache: bool
) -> dict[str, str]:
  """Fetches HLO expression strings from op_profile if available."""
  expressions_by_name: dict[str, str] = {}
  try:
    result = client.fetch(
        tool_name="hlo_op_profile.json",
        session_id=session_id,
        bypass_cache=bypass_cache,
    )
    if not result or (isinstance(result, tuple) and not result[1]):
      result = client.fetch(
          tool_name="op_profile",
          session_id=session_id,
          bypass_cache=bypass_cache,
      )
    if isinstance(result, tuple) and len(result) == 2:
      _, raw_data = result
    else:
      raw_data = result
    if raw_data:
      if isinstance(raw_data, bytes):
        raw_data = raw_data.decode("utf-8", errors="replace")
      profile = op_profile_pb2.Profile()
      json_format.Parse(raw_data, profile, ignore_unknown_fields=True)
      if profile.HasField("by_program"):
        hlo_shape_utils.extract_expressions_from_op_profile_node(
            profile.by_program, expressions_by_name
        )
      if profile.HasField("by_category"):
        hlo_shape_utils.extract_expressions_from_op_profile_node(
            profile.by_category, expressions_by_name
        )
  except Exception:  # pylint: disable=broad-exception-caught
    logging.debug(
        "Could not fetch op_profile expressions for %s",
        session_id,
        exc_info=True,
    )
  return expressions_by_name


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

    device_info: dict[str, Any] = {}
    bw_renames = {
        "peak_hbm_bw": "peak_hbm_bw_gibs",
        "peak_hbm_read_bw": "peak_hbm_read_bw_gibs",
        "peak_hbm_write_bw": "peak_hbm_write_bw_gibs",
        "peak_vmem_bw": "peak_vmem_bw_gibs",
        "peak_vmem_read_bw": "peak_vmem_read_bw_gibs",
        "peak_vmem_write_bw": "peak_vmem_write_bw_gibs",
        "peak_cmem_bw": "peak_cmem_bw_gibs",
        "peak_cmem_read_bw": "peak_cmem_read_bw_gibs",
        "peak_cmem_write_bw": "peak_cmem_write_bw_gibs",
    }
    units_map: dict[str, str] = {}
    for k, v in custom_props.items():
      try:
        parsed_v: Any = float(v)
      except (ValueError, TypeError):
        parsed_v = v
      renamed_k = bw_renames.get(k, k)
      device_info[renamed_k] = parsed_v
      if k in bw_renames:
        units_map[renamed_k] = "GiB/s"
      elif k == "peak_flop_rate":
        units_map[k] = "GFLOP/s"
      elif k in (
          "hbm_ridge_point",
          "vmem_read_ridge_point",
          "vmem_write_ridge_point",
          "cmem_read_ridge_point",
          "cmem_write_ridge_point",
          "ridge_point",
      ):
        units_map[k] = "FLOP/byte"
    if units_map:
      device_info["units"] = units_map

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
        resolved_pk = bw_renames.get(pk, pk)
        if (
            resolved_pk in device_info
            and safe_float(device_info[resolved_pk]) > 0
        ):
          peak = safe_float(device_info[resolved_pk])
          break
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
        "flops_provenance": "xla_cost_model",
    }

    peak_flop_rate = safe_float(device_info.get("peak_flop_rate"))
    ridge_point = safe_float(
        device_info.get("hbm_ridge_point")
    ) or safe_float(device_info.get("ridge_point"))
    expressions_by_name: dict[str, str] | None = None

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
      op_compute_eff = safe_float(r_dict.get("compute_efficiency"))
      op_roofline_eff = safe_float(r_dict.get("roofline_efficiency"))
      op_max_mem_eff = safe_float(r_dict.get("max_mem_bw_utilization"))
      op_intensity = safe_float(r_dict.get("operational_intensity"))
      provenance = "xla_cost_model"
      derived_flops: float | None = None
      derived_bytes: float | None = None

      if expressions_by_name is None:
        expressions_by_name = _fetch_op_profile_expressions(
            session_id, client, bypass_cache
        )
      expr = (
          r_dict.get("expression")
          or r_dict.get("hlo_expression")
          or (expressions_by_name.get(op_name) if expressions_by_name else "")
          or (
              expressions_by_name.get(op_name.lstrip("%"))
              if expressions_by_name
              else ""
          )
          or op_name
      )
      is_custom = (
          op_name.startswith("custom-call")
          or "custom-call" in op_name.lower()
          or "custom_call" in op_name.lower()
          or op_category.lower() in ("custom-call", "custom_call")
          or "custom-call" in expr.lower()
          or "custom_call" in expr.lower()
          or "custom_call_target" in expr.lower()
      )
      orig_op_flop_rate = safe_float(
          r_dict.get("measured_flop_rate")
      ) or safe_float(r_dict.get("model_flop_rate"))
      orig_op_flops = orig_op_flop_rate * (self_time_us * 1e3)

      if is_custom:
        derived_flops, derived_bytes, provenance = (
            hlo_shape_utils.derive_custom_call_flops_and_bytes(
                expr, op_category, op_name
            )
        )
        if (
            provenance == "derived_from_shapes"
            and derived_flops
            and derived_flops > 0
        ):
          op_flop_rate_gflops = derived_flops / (self_time_us * 1e3)
          if peak_flop_rate > 0:
            op_compute_eff = op_flop_rate_gflops / peak_flop_rate
          op_roofline_eff = max(op_compute_eff, op_max_mem_eff)
          if derived_bytes and derived_bytes > 0:
            op_intensity = derived_flops / derived_bytes
          if ridge_point > 0:
            bound_by_val = "Compute" if op_intensity >= ridge_point else "HBM"
          elif op_compute_eff >= op_max_mem_eff:
            bound_by_val = "Compute"
          else:
            bound_by_val = "HBM"
        else:
          provenance = "opaque_custom_call"
          bound_by_val = "CustomCall (opaque)"
          op_compute_eff = 0.0
          op_roofline_eff = op_max_mem_eff
          op_intensity = 0.0

      orig_replaced_flops = 0.0
      if is_custom and (derived_flops or provenance == "opaque_custom_call"):
        orig_replaced_flops = orig_op_flops

      op_records.append({
          "rank": int(safe_float(r_dict.get("rank"))),
          "name": op_name,
          "category": op_category,
          "total_self_time_ms": round(self_time_us / 1000.0, 3),
          "total_self_time_percent": to_percent_str(
              r_dict.get("total_self_time_percent")
          ),
          "operational_intensity_flop_per_byte": round(op_intensity, 4),
          "bottleneck_operational_intensity_flop_per_byte": round(
              safe_float(r_dict.get("bottleneck_operational_intensity")), 4
          ),
          "roofline_efficiency_percent": to_percent_str(op_roofline_eff),
          "compute_efficiency_percent": to_percent_str(op_compute_eff),
          "max_mem_bw_utilization_percent": to_percent_str(op_max_mem_eff),
          "optimal_flop_rate_gflops": round(
              safe_float(r_dict.get("optimal_flop_rate")), 2
          ),
          "dma_stall_percent": to_percent_str(r_dict.get("dma_stall_percent")),
          "bound_by": bound_by_val,
          "flops_provenance": provenance,
          "hlo_module_id": str(r_dict.get("hlo_module_id", "")),
          "source_info": cleaned_source,
          "_derived_flops": derived_flops or 0.0,
          "_derived_bytes": derived_bytes or 0.0,
          "_orig_replaced_flops": orig_replaced_flops,
      })

    # Deduplicate operations by (rank, name)
    seen_ops = set()
    unique_op_records = []
    for op in op_records:
      op_key = (op["rank"], op["name"])
      if op_key not in seen_ops:
        seen_ops.add(op_key)
        unique_op_records.append(op)

    total_derived_flops = sum(
        op.pop("_derived_flops", 0.0) for op in unique_op_records
    )
    total_derived_bytes = sum(
        op.pop("_derived_bytes", 0.0) for op in unique_op_records
    )
    total_orig_replaced_flops = sum(
        op.pop("_orig_replaced_flops", 0.0) for op in unique_op_records
    )
    for op in op_records:
      op.pop("_derived_flops", None)
      op.pop("_derived_bytes", None)
      op.pop("_orig_replaced_flops", None)

    unique_op_records.sort(key=lambda x: x["total_self_time_ms"], reverse=True)
    top_ops = unique_op_records[:top_n]

    if total_derived_flops > 0:
      prog_total_time_us = safe_float(prog_dict.get("total_time"))
      if prog_total_time_us <= 0:
        prog_total_time_us = sum(
            op["total_self_time_ms"] * 1000.0 for op in unique_op_records
        )
      if prog_total_time_us > 0:
        orig_prog_flops = safe_float(prog_dict.get("measured_flop_rate")) * (
            prog_total_time_us * 1e3
        )
        new_prog_flops = max(
            0.0, orig_prog_flops - total_orig_replaced_flops
        ) + total_derived_flops
        new_prog_flop_rate = new_prog_flops / (prog_total_time_us * 1e3)
        prog_compute_eff = (
            new_prog_flop_rate / peak_flop_rate
            if peak_flop_rate > 0
            else safe_float(prog_dict.get("compute_efficiency"))
        )
        prog_max_mem_eff = safe_float(prog_dict.get("max_mem_bw_utilization"))
        prog_roofline_eff = max(prog_compute_eff, prog_max_mem_eff)
        prog_mem_bw = safe_float(prog_dict.get("measured_memory_bw"))
        if prog_mem_bw > 0:
          prog_op_intensity = new_prog_flop_rate / prog_mem_bw
        elif total_derived_bytes > 0:
          prog_op_intensity = new_prog_flops / total_derived_bytes
        else:
          prog_op_intensity = safe_float(prog_dict.get("operational_intensity"))

        if ridge_point > 0:
          prog_bound_by = (
              "Compute" if prog_op_intensity >= ridge_point else "HBM"
          )
        elif prog_compute_eff >= prog_max_mem_eff:
          prog_bound_by = "Compute"
        else:
          prog_bound_by = prog_dict.get("bound_by", "Unknown")

        program_metrics["measured_flop_rate_gflops"] = round(
            new_prog_flop_rate, 2
        )
        program_metrics["compute_efficiency_percent"] = to_percent_str(
            prog_compute_eff
        )
        program_metrics["roofline_efficiency_percent"] = to_percent_str(
            prog_roofline_eff
        )
        program_metrics["operational_intensity_flop_per_byte"] = round(
            prog_op_intensity, 4
        )
        program_metrics["bound_by"] = prog_bound_by
        program_metrics["flops_provenance"] = "derived_from_shapes"

    has_opaque_custom_call = any(
        op.get("flops_provenance") == "opaque_custom_call" for op in top_ops
    )
    if has_opaque_custom_call and total_derived_flops == 0:
      if program_metrics.get("flops_provenance") == "xla_cost_model":
        program_metrics["flops_provenance"] = "opaque_custom_call"
      if program_metrics.get("bound_by") in ("Unknown", "", None):
        program_metrics["bound_by"] = "CustomCall (opaque)"

    output: dict[str, Any] = {
        "program": program_metrics,
        "device_info": device_info,
        "top_operations": top_ops,
        "total_operations_analyzed": len(unique_op_records),
    }
    if has_opaque_custom_call:
      output["guidance"] = (
          "Op-level metrics unavailable for opaque custom calls. Use"
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
