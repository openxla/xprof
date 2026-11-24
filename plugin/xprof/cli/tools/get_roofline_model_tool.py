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

# Relative tolerance, as a fraction of the original program FLOPs, below which
# leftover XLA cost-model FLOPs count as zero. It absorbs separately totalled
# backend rows (the Program row and the per-op rows are summed independently
# and rounded on export), not only float rounding.
_FLOPS_REL_TOLERANCE = 1e-3

# Per-memory-space columns that the backend uses for the bottleneck operational
# intensity of each `bound_by` value (convert/op_metrics_to_record.h). Each
# entry is (bandwidth column in GiB/s, operational intensity column). On GPU,
# `Shm/L1` uses the HBM bandwidth and HBM operational intensity.
_BOUND_BY_SPACE_COLUMNS: dict[str, tuple[str, str]] = {
    "HBM": ("hbm_bw", "hbm_operational_intensity"),
    "Shm/L1": ("hbm_bw", "hbm_operational_intensity"),
    "CMEM Read": ("cmem_read_bw", "cmem_read_operational_intensity"),
    "CMEM Write": ("cmem_write_bw", "cmem_write_operational_intensity"),
    "VMEM Read": ("vmem_read_bw", "vmem_read_operational_intensity"),
    "VMEM Write": ("vmem_write_bw", "vmem_write_operational_intensity"),
}


def _gibi_to_giga(gibi: float) -> float:
  """Converts a GiB-based quantity (2**30) to its GB-based value (1e9)."""
  return gibi * (2**30) / 1e9


def _safe_float(val: Any, default: float = 0.0) -> float:
  """Converts `val` to float, returning `default` if that is not possible."""
  if val is None:
    return default
  try:
    return float(val)
  except (ValueError, TypeError):
    return default


def _recompute_bottleneck_intensity(
    row: dict[str, Any],
    bound_by: str,
    flop_rate_gflops: float,
    operational_intensity: float,
    flops_scale: float | None,
) -> float:
  """Recomputes the bottleneck operational intensity after a FLOP change.

  Mirrors the backend: the bottleneck intensity is the all-memory-spaces
  intensity when the row is compute bound, and the intensity of the bound
  memory space otherwise.

  Args:
    row: The roofline table row (column id to value).
    bound_by: The recomputed `bound_by` value of the row.
    flop_rate_gflops: The recomputed FLOP rate of the row in GFLOP/s.
    operational_intensity: The recomputed all-memory-spaces operational
      intensity of the row in FLOPs/byte.
    flops_scale: The ratio of recomputed to original FLOPs, or None if the
      original FLOPs are 0.

  Returns:
    The bottleneck operational intensity in FLOPs/byte. Falls back to the
    backend value when it cannot be recomputed.
  """
  backend_value = _safe_float(row.get("bottleneck_operational_intensity"))
  if bound_by == "Compute":
    return operational_intensity
  columns = _BOUND_BY_SPACE_COLUMNS.get(bound_by)
  if columns is None:
    return backend_value
  bw_key, intensity_key = columns
  space_bw_gibs = _safe_float(row.get(bw_key))
  if space_bw_gibs > 0:
    return flop_rate_gflops / _gibi_to_giga(space_bw_gibs)
  space_intensity = _safe_float(row.get(intensity_key))
  if flops_scale is not None and space_intensity > 0:
    return space_intensity * flops_scale
  return backend_value


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


def is_custom_call_op(
    op_name: str, op_category: str = "", expr: str = ""
) -> bool:
  """Returns True if an op name, category, or expression is a custom call."""
  name_lower = op_name.lower()
  expr_lower = expr.lower()
  return (
      "custom-call" in name_lower
      or "custom_call" in name_lower
      or op_category.lower() in ("custom-call", "custom_call")
      or "custom-call" in expr_lower
      or "custom_call" in expr_lower
  )


def _row_expression(
    row: dict[str, Any],
    op_name: str,
    expressions_by_name: dict[str, str] | None,
) -> str:
  """Returns the HLO expression of a roofline row, or an empty string."""
  expressions = expressions_by_name or {}
  return str(
      row.get("expression")
      or row.get("hlo_expression")
      or expressions.get(op_name)
      or expressions.get(op_name.lstrip("%"))
      or ""
  )


def format_custom_call_warning(share_pct: float) -> str:
  """Returns the standard warning string for significant CustomCall self-time."""
  return (
      f"CustomCalls account for {share_pct:.2f}% of device self-time. XLA"
      " assigns 0 FLOPs to opaque custom calls (FLOPs are invisible to the XLA"
      " roofline model, not zero). Do NOT interpret 0 FLOP/s as idle compute;"
      " use `get_kernel_utilization` or `get_llo_static_analysis` to inspect"
      " Pallas/CustomCall hardware utilization and static schedule efficiency."
  )


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
    # Only the keys the backend actually emits in the DataTable's custom
    # properties. There is no HBM read/write split (hbm_bw is combined R+W),
    # and there is no aggregate peak_vmem_bw / peak_cmem_bw.
    bw_renames = {
        "peak_hbm_bw": "peak_hbm_bw_gibs",
        "peak_vmem_read_bw": "peak_vmem_read_bw_gibs",
        "peak_vmem_write_bw": "peak_vmem_write_bw_gibs",
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

    def to_percent_str(val: Any) -> str:
      f = _safe_float(val)
      return f"{f * 100.0:.2f}%"

    def calc_mem_util_str(
        util_key: str, bw_key: str, peak_keys: list[str]
    ) -> str:
      val = prog_dict.get(util_key)
      if val is not None and _safe_float(val) > 0:
        return to_percent_str(val)
      bw = _safe_float(prog_dict.get(bw_key))
      peak = 0.0
      for pk in peak_keys:
        resolved_pk = bw_renames.get(pk, pk)
        if (
            resolved_pk in device_info
            and _safe_float(device_info[resolved_pk]) > 0
        ):
          peak = _safe_float(device_info[resolved_pk])
          break
        if pk in device_info and _safe_float(device_info[pk]) > 0:
          peak = _safe_float(device_info[pk])
          break
      if peak > 0:
        return f"{(bw / peak) * 100.0:.2f}%"
      if val is not None:
        return to_percent_str(val)
      return "N/A"

    # The ridge point that `bound_by` is actually decided against. The backend
    # picks the bottleneck as the argmax over per-memory-space utilizations,
    # which is algebraically a ridge test on the *bottleneck* operational
    # intensity for that space -- not on the all-spaces `operational_intensity`.
    bound_by_ridge_keys = {
        "HBM": "hbm_ridge_point",
        "CMEM Read": "cmem_read_ridge_point",
        "CMEM Write": "cmem_write_ridge_point",
        "VMEM Read": "vmem_read_ridge_point",
        "VMEM Write": "vmem_write_ridge_point",
    }

    def ridge_point_for(bound_by_val: str) -> float | None:
      key = bound_by_ridge_keys.get(str(bound_by_val))
      if key is None:
        return None
      val = device_info.get(key)
      return None if val is None else round(_safe_float(val), 4)

    prog_bound_by = prog_dict.get("bound_by", "Unknown")

    program_metrics = {
        "bound_by": prog_bound_by,
        "bound_by_ridge_point_flop_per_byte": ridge_point_for(prog_bound_by),
        "operational_intensity_flop_per_byte": round(
            _safe_float(prog_dict.get("operational_intensity")), 4
        ),
        "bottleneck_operational_intensity_flop_per_byte": round(
            _safe_float(prog_dict.get("bottleneck_operational_intensity")), 4
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
            _safe_float(prog_dict.get("optimal_flop_rate")), 2
        ),
        "dma_stall_percent": to_percent_str(prog_dict.get("dma_stall_percent")),
        "measured_flop_rate_gflops": round(
            _safe_float(prog_dict.get("measured_flop_rate")), 2
        ),
        "model_flop_rate_gflops": round(
            _safe_float(prog_dict.get("model_flop_rate")), 2
        ),
        "measured_memory_bw_gibs": round(
            _safe_float(prog_dict.get("measured_memory_bw")), 2
        ),
        "hbm_bw_gibs": round(_safe_float(prog_dict.get("hbm_bw")), 2),
        "hbm_bw_utilization_percent": calc_mem_util_str(
            "hbm_bw_utilization", "hbm_bw", ["peak_hbm_bw"]
        ),
        "cmem_read_bw_utilization_percent": calc_mem_util_str(
            "cmem_read_bw_utilization", "cmem_read_bw", ["peak_cmem_read_bw"]
        ),
        "cmem_write_bw_utilization_percent": calc_mem_util_str(
            "cmem_write_bw_utilization", "cmem_write_bw", ["peak_cmem_write_bw"]
        ),
        "vmem_read_bw_utilization_percent": calc_mem_util_str(
            "vmem_read_bw_utilization", "vmem_read_bw", ["peak_vmem_read_bw"]
        ),
        "vmem_write_bw_utilization_percent": calc_mem_util_str(
            "vmem_write_bw_utilization", "vmem_write_bw", ["peak_vmem_write_bw"]
        ),
        "total_time_ms": round(
            _safe_float(prog_dict.get("total_time")) / 1000.0, 3
        ),
        "flops_provenance": "xla_cost_model",
        "metric_semantics": {
            "measured_memory_bw_gibs": (
                "Total bytes across ALL memory spaces (HBM + CMEM + VMEM +"
                " bytes with no memory space attributed, such as SparseCore)"
                " divided by total op time. It is NOT an HBM figure and may"
                " legitimately exceed peak_hbm_bw_gibs. Compare hbm_bw_gibs"
                " against peak_hbm_bw_gibs instead."
            ),
            "max_mem_bw_utilization_percent": (
                "Maximum over the per-memory-space utilizations"
                " (hbm_bw/peak_hbm_bw, cmem_read, cmem_write, vmem_read,"
                " vmem_write). It is NOT measured_memory_bw_gibs divided by"
                " peak_hbm_bw_gibs."
            ),
            "operational_intensity_flop_per_byte": (
                "FLOPs (bf16-normalized) divided by bytes across ALL memory"
                " spaces. Because the denominator includes non-HBM bytes it is"
                " always <= bottleneck_operational_intensity_flop_per_byte,"
                " and comparing it against a single-space ridge point"
                " overstates memory boundness. Recomputed values convert"
                " GiB/s to GB/s first, so they are FLOPs per byte like the"
                " backend value."
            ),
            "bound_by": (
                "Argmax over resource utilizations (Compute wins ties)."
                " Verify it with"
                " bottleneck_operational_intensity_flop_per_byte against"
                " bound_by_ridge_point_flop_per_byte, not with"
                " operational_intensity_flop_per_byte."
            ),
            "flops_provenance": (
                "Source of the program FLOPs: 'xla_cost_model' (XLA static"
                " cost model), 'derived_from_shapes' (no cost-model FLOPs"
                " remain; all FLOPs come from shape-derived custom calls),"
                " 'mixed' (shape-derived custom call FLOPs plus cost-model"
                " FLOPs for the other ops, or plus opaque custom calls that"
                " count as 0; see derived_flops_fraction and"
                " has_opaque_custom_calls), or 'opaque_custom_call'."
            ),
            "derived_flops_fraction": (
                "Share of program FLOPs that is shape-derived (0.0 to 1.0)."
                " 0.0 when no custom call FLOPs were derived from shapes."
            ),
            "has_opaque_custom_calls": (
                "True if any analyzed op (not only top_operations) is a"
                " custom call whose FLOPs could not be derived."
            ),
        },
    }

    peak_flop_rate = _safe_float(device_info.get("peak_flop_rate"))
    expressions_by_name: dict[str, str] | None = None
    # Pick the FLOP rate column once from the program row. The backend sums
    # flops_v2 into measured_flop_rate and model_flops_v2 into
    # model_flop_rate, so mixing the two columns between the program and its
    # ops makes the subtraction below meaningless.
    flop_rate_key = (
        "measured_flop_rate"
        if _safe_float(prog_dict.get("measured_flop_rate")) > 0
        else "model_flop_rate"
    )

    op_records = []
    for r in rows[1:]:
      r_dict = row_to_dict(r.get("c", []))
      self_time_us = _safe_float(r_dict.get("total_self_time"))
      if self_time_us <= 0:
        continue
      raw_total_time_us = _safe_float(r_dict.get("total_time"))
      total_time_us = (
          raw_total_time_us if raw_total_time_us > 0 else self_time_us
      )

      source_info_raw = r_dict.get("source_info", "")
      cleaned_source = _strip_html_tags(source_info_raw)

      op_name = r_dict.get("operation") or r_dict.get("hlo_name", "")
      op_category = r_dict.get("category") or r_dict.get("hlo_category", "")
      bound_by_val = r_dict.get("bound_by") or "Unknown"
      op_compute_eff = _safe_float(r_dict.get("compute_efficiency"))
      op_roofline_eff = _safe_float(r_dict.get("roofline_efficiency"))
      op_max_mem_eff = _safe_float(r_dict.get("max_mem_bw_utilization"))
      op_intensity = _safe_float(r_dict.get("operational_intensity"))
      provenance = "xla_cost_model"
      derived_flops: float | None = None
      derived_bytes: float | None = None

      if expressions_by_name is None:
        expressions_by_name = _fetch_op_profile_expressions(
            session_id, client, bypass_cache
        )
      expr = _row_expression(r_dict, op_name, expressions_by_name) or op_name
      is_custom = is_custom_call_op(op_name, op_category, expr)
      # Use the same rate column as the program row so that the replaced
      # FLOPs and the program FLOPs they are subtracted from are comparable.
      # Backend measured/model FLOP rates divide by total_time, not
      # total_self_time.
      orig_op_flops = _safe_float(r_dict.get(flop_rate_key)) * (
          total_time_us * 1e3
      )
      op_bottleneck_intensity = _safe_float(
          r_dict.get("bottleneck_operational_intensity")
      )
      op_optimal_flop_rate = _safe_float(r_dict.get("optimal_flop_rate"))

      backend_bound_by = str(r_dict.get("bound_by") or "Unknown")

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
          op_flop_rate_gflops = derived_flops / (total_time_us * 1e3)
          if peak_flop_rate > 0:
            op_compute_eff = op_flop_rate_gflops / peak_flop_rate
          op_roofline_eff = max(op_compute_eff, op_max_mem_eff)
          if derived_bytes and derived_bytes > 0:
            op_intensity = derived_flops / derived_bytes
          # Mirror the backend rule: the bottleneck is the argmax over
          # resource utilizations, and Compute wins ties. Comparing
          # `op_intensity` (FLOPs over ALL memory spaces) against the HBM-only
          # ridge point mixes denominators and over-reports memory boundness.
          if op_compute_eff >= op_max_mem_eff:
            bound_by_val = "Compute"
          elif backend_bound_by not in ("Compute", "Unknown", ""):
            # Keep whichever memory space the backend identified.
            bound_by_val = backend_bound_by
          else:
            bound_by_val = "HBM"
          op_bottleneck_intensity = _recompute_bottleneck_intensity(
              r_dict,
              bound_by_val,
              op_flop_rate_gflops,
              op_intensity,
              derived_flops / orig_op_flops if orig_op_flops > 0 else None,
          )
          if op_roofline_eff > 0:
            op_optimal_flop_rate = op_flop_rate_gflops / op_roofline_eff
        else:
          provenance = "opaque_custom_call"
          bound_by_val = "CustomCall (opaque)"
          op_compute_eff = 0.0
          op_roofline_eff = op_max_mem_eff
          op_intensity = 0.0
          # The op counts 0 FLOPs, so no FLOP-based intensity or rate holds.
          op_bottleneck_intensity = 0.0
          op_optimal_flop_rate = 0.0

      orig_replaced_flops = 0.0
      if is_custom and (derived_flops or provenance == "opaque_custom_call"):
        orig_replaced_flops = orig_op_flops

      op_records.append({
          "rank": int(_safe_float(r_dict.get("rank"))),
          "name": op_name,
          "category": op_category,
          "total_self_time_ms": round(self_time_us / 1000.0, 3),
          "total_self_time_percent": to_percent_str(
              r_dict.get("total_self_time_percent")
          ),
          "operational_intensity_flop_per_byte": round(op_intensity, 4),
          "bottleneck_operational_intensity_flop_per_byte": round(
              op_bottleneck_intensity, 4
          ),
          "roofline_efficiency_percent": to_percent_str(op_roofline_eff),
          "compute_efficiency_percent": to_percent_str(op_compute_eff),
          "max_mem_bw_utilization_percent": to_percent_str(op_max_mem_eff),
          "optimal_flop_rate_gflops": round(op_optimal_flop_rate, 2),
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

    # Scan every analyzed op, not only the returned top_ops, so that --top_n
    # cannot change any program-level field.
    has_opaque_custom_call = any(
        op.get("flops_provenance") == "opaque_custom_call"
        or op.get("bound_by") == "CustomCall (opaque)"
        for op in unique_op_records
    )
    program_metrics["has_opaque_custom_calls"] = has_opaque_custom_call
    program_metrics["derived_flops_fraction"] = 0.0

    if total_derived_flops > 0:
      prog_total_time_us = _safe_float(prog_dict.get("total_time"))
      if prog_total_time_us <= 0:
        prog_total_time_us = sum(
            op["total_self_time_ms"] * 1000.0 for op in unique_op_records
        )
      if prog_total_time_us > 0:
        prog_time_ns = prog_total_time_us * 1e3
        orig_prog_flops = _safe_float(prog_dict.get(flop_rate_key)) * (
            prog_time_ns
        )
        flops_tolerance = _FLOPS_REL_TOLERANCE * orig_prog_flops
        # The replaced FLOPs come from the same rate column as the program,
        # so they cannot exceed the program total unless the backend rows
        # disagree. Do not claim a clean provenance in that case.
        rates_inconsistent = (
            total_orig_replaced_flops - orig_prog_flops > flops_tolerance
        )
        if rates_inconsistent:
          logging.warning(
              "Replaced custom call FLOPs (%g) exceed program FLOPs (%g) from"
              " column %s for session %s; labelling provenance as mixed.",
              total_orig_replaced_flops,
              orig_prog_flops,
              flop_rate_key,
              session_id,
          )
        # Shape-derived FLOPs replace only the custom calls; every other op
        # keeps its XLA cost-model FLOPs. Clamp a remainder within tolerance
        # to 0 before it enters the totals, so "derived_from_shapes" always
        # pairs with a derived_flops_fraction of 1.0.
        cost_model_flops = max(0.0, orig_prog_flops - total_orig_replaced_flops)
        if cost_model_flops <= flops_tolerance:
          cost_model_flops = 0.0
        new_prog_flops = cost_model_flops + total_derived_flops
        new_prog_flop_rate = new_prog_flops / prog_time_ns
        derived_flops_fraction = (
            total_derived_flops / new_prog_flops if new_prog_flops > 0 else 0.0
        )
        if (
            cost_model_flops > 0
            or rates_inconsistent
            or has_opaque_custom_call
        ):
          prog_flops_provenance = "mixed"
        else:
          prog_flops_provenance = "derived_from_shapes"
        flops_scale = (
            new_prog_flops / orig_prog_flops if orig_prog_flops > 0 else None
        )

        prog_compute_eff = (
            new_prog_flop_rate / peak_flop_rate
            if peak_flop_rate > 0
            else _safe_float(prog_dict.get("compute_efficiency"))
        )
        prog_max_mem_eff = _safe_float(prog_dict.get("max_mem_bw_utilization"))
        prog_roofline_eff = max(prog_compute_eff, prog_max_mem_eff)
        prog_mem_bw = _safe_float(prog_dict.get("measured_memory_bw"))
        backend_prog_intensity = _safe_float(
            prog_dict.get("operational_intensity")
        )
        if prog_mem_bw > 0:
          # `new_prog_flop_rate` is in GFLOP/s (1e9) while
          # `measured_memory_bw` is in GiB/s (2**30). Convert to GB/s so the
          # result is FLOPs per byte, matching the backend's value.
          prog_op_intensity = new_prog_flop_rate / _gibi_to_giga(prog_mem_bw)
        elif backend_prog_intensity > 0 and flops_scale is not None:
          prog_op_intensity = backend_prog_intensity * flops_scale
        elif (
            prog_flops_provenance == "derived_from_shapes"
            and total_derived_bytes > 0
        ):
          # Only shape-derived ops remain, so their bytes are the program
          # bytes (for example a pure Pallas program).
          prog_op_intensity = new_prog_flops / total_derived_bytes
        else:
          prog_op_intensity = backend_prog_intensity

        # Mirror the backend rule (argmax over resource utilizations, Compute
        # wins ties) rather than testing the all-memory-spaces intensity
        # against the HBM-only ridge point.
        backend_prog_bound_by = str(prog_dict.get("bound_by") or "Unknown")
        if prog_compute_eff >= prog_max_mem_eff:
          prog_bound_by = "Compute"
        elif backend_prog_bound_by not in ("Compute", "Unknown", ""):
          prog_bound_by = backend_prog_bound_by
        else:
          prog_bound_by = "HBM"
        prog_bottleneck_intensity = _recompute_bottleneck_intensity(
            prog_dict,
            prog_bound_by,
            new_prog_flop_rate,
            prog_op_intensity,
            flops_scale,
        )

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
        program_metrics["bottleneck_operational_intensity_flop_per_byte"] = (
            round(prog_bottleneck_intensity, 4)
        )
        if prog_roofline_eff > 0:
          program_metrics["optimal_flop_rate_gflops"] = round(
              new_prog_flop_rate / prog_roofline_eff, 2
          )
        program_metrics["bound_by"] = prog_bound_by
        program_metrics["bound_by_ridge_point_flop_per_byte"] = (
            ridge_point_for(prog_bound_by)
        )
        program_metrics["bound_by_provenance"] = (
            "recomputed_from_derived_custom_call_flops"
        )
        program_metrics["flops_provenance"] = prog_flops_provenance
        program_metrics["derived_flops_fraction"] = round(
            derived_flops_fraction, 4
        )

    prog_total = _safe_float(prog_dict.get("total_self_time_percent"), 1.0)
    percent_scale = 100.0 if prog_total <= 1.0 + 1e-6 else 1.0
    custom_call_share_pct = 0.0
    seen_custom_call_keys = set()
    for row in rows[1:]:
      row_dict = row_to_dict(row.get("c", []))
      op_name = str(row_dict.get("operation") or row_dict.get("hlo_name", ""))
      op_category = str(
          row_dict.get("category") or row_dict.get("hlo_category", "")
      )
      op_key = (int(_safe_float(row_dict.get("rank"))), op_name)
      if op_key in seen_custom_call_keys:
        continue
      row_expr = _row_expression(row_dict, op_name, expressions_by_name)
      if is_custom_call_op(op_name, op_category, row_expr):
        seen_custom_call_keys.add(op_key)
        val = _safe_float(row_dict.get("total_self_time_percent"))
        custom_call_share_pct += val * percent_scale
    custom_call_share_pct = min(round(custom_call_share_pct, 2), 100.0)

    has_custom_call = has_opaque_custom_call or custom_call_share_pct >= 10.0
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
    if has_custom_call:
      output["custom_call_share_pct"] = custom_call_share_pct
      output["custom_call_warning"] = format_custom_call_warning(
          custom_call_share_pct
      )
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
