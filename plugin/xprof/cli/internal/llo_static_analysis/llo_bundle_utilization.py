"""Per-bundle static utilization JSON renderer + structured extraction.

Reads `LloModuleProto.static_utilization` (a `StaticPerBundleUtilizationProto`)
and reports, per bundle, how many issue slots each functional unit used out of
the total available, plus spill / fill counts. Each `RationalVectorProto` field
carries `numerator[i]` (slots used at bundle `i`) and a single `denominator`
(slots available per bundle for that unit).

The compiler populates `static_utilization` only when it has a schedule profile
(`if (profile.has_value())`), so the empty case is handled explicitly.
"""

import json
from typing import Any

from xprof.embedded.llo_analysis import llo_lite_pb2

# Keys match the proto field names.
FIELDS = (
    "mxu",
    "xlu",
    "vector_alu",
    "vector_eup",
    "vector_load",
    "vector_store",
    "scalar_alu",
)


def _unit_vectors(
    util: llo_lite_pb2.StaticPerBundleUtilizationProto,
) -> tuple[tuple[str, Any], ...]:
  return (
      ("mxu", util.mxu),
      ("xlu", util.xlu),
      ("vector_alu", util.vector_alu),
      ("vector_eup", util.vector_eup),
      ("vector_load", util.vector_load),
      ("vector_store", util.vector_store),
      ("scalar_alu", util.scalar_alu),
  )


def num_bundles(util: llo_lite_pb2.StaticPerBundleUtilizationProto) -> int:
  """Returns the highest bundle index with any data, +1 (0 if empty)."""
  n = 0
  for _, rv in _unit_vectors(util):
    if rv.numerator:
      n = max(n, len(rv.numerator))
  if util.vector_spill:
    n = max(n, len(util.vector_spill))
  if util.vector_fill:
    n = max(n, len(util.vector_fill))
  return n


def has_static_utilization(module: llo_lite_pb2.LloModuleProto) -> bool:
  """True iff the compiler emitted per-bundle static utilization."""
  return num_bundles(module.static_utilization) > 0


def extract_bundle_utilization(
    module: llo_lite_pb2.LloModuleProto,
) -> list[dict[str, Any]]:
  """Returns a per-bundle list of structured utilization records.

  Each record is `{bundle, semantics, <unit>_used, <unit>_avail, spill, fill,
  util_pct}`. Returns [] when the module has no static utilization data.

  Args:
    module: The LLO module proto with optional static_utilization.
  """
  util = module.static_utilization
  total = num_bundles(util)
  unit_vecs = _unit_vectors(util)
  out: list[dict[str, Any]] = []
  for i in range(total):
    rec: dict[str, Any] = {
        "bundle": i,
        "semantics": "static_modelled_schedule",
    }
    used_total = 0
    avail_total = 0
    for f, rv in unit_vecs:
      denom = rv.denominator
      num = rv.numerator[i] if i < len(rv.numerator) else 0
      rec[f"{f}_used"] = num
      rec[f"{f}_avail"] = denom
      if denom:
        used_total += num
        avail_total += denom
    rec["spill"] = util.vector_spill[i] if i < len(util.vector_spill) else 0
    rec["fill"] = util.vector_fill[i] if i < len(util.vector_fill) else 0
    rec["util_pct"] = (
        0 if not avail_total else round(100 * used_total / avail_total)
    )
    out.append(rec)
  return out


def render_bundle_util_json(
    module: llo_lite_pb2.LloModuleProto,
    bundle_range: tuple[int, int] | None = None,
    region_path_map: dict[int, str] | None = None,
    max_rows: int = 30,
    read_all: bool = False,
) -> str:
  """Renders a per-bundle utilization report as JSON."""
  util = module.static_utilization
  total = num_bundles(util)
  start, end = 0, total
  if bundle_range is not None and total > 0:
    start, end = bundle_range
    end = min(end, total)
    start = max(0, start)
  if not read_all and (end - start) > max_rows:
    end = start + max_rows

  records = extract_bundle_utilization(module)
  filtered: list[dict[str, Any]] = []
  for rec in records:
    i = rec["bundle"]
    if not start <= i < end:
      continue
    row = dict(rec)
    if region_path_map is not None:
      row["region_path"] = region_path_map.get(i, "-")
    filtered.append(row)

  payload = {
      "hlo_instruction_name": module.hlo_instruction_name,
      "hlo_module_name": module.hlo_module_name,
      "hlo_module_id": int(module.hlo_module_id),
      "semantics": "static_modelled_schedule",
      "total_bundles": total,
      "start_bundle": start,
      "end_bundle": end,
      "bundles": filtered,
  }
  return json.dumps(payload, indent=2)
