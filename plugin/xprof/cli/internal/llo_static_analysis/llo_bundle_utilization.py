"""Per-bundle static utilization renderer + structured extraction.

Reads `LloModuleProto.static_utilization` (a `StaticPerBundleUtilizationProto`)
and reports, per bundle, how many issue slots each functional unit used out of
the total available, plus spill / fill counts. Each `RationalVectorProto` field
carries `numerator[i]` (slots used at bundle `i`) and a single `denominator`
(slots available per bundle for that unit).

The compiler populates `static_utilization` only when it has a schedule profile
(`if (profile.has_value())`), so the empty case is handled explicitly.
"""

import io
from typing import Any

from xprof.embedded.llo_analysis import llo_lite_pb2 as llo_module_pb2

# Rendered in this order; keys match the proto field names.
FIELDS = (
    "mxu",
    "xlu",
    "vector_alu",
    "vector_eup",
    "vector_load",
    "vector_store",
    "scalar_alu",
)
_COL_LABELS = {
    "mxu": "MXU",
    "xlu": "XLU",
    "vector_alu": "V_ALU",
    "vector_eup": "V_EUP",
    "vector_load": "V_LOAD",
    "vector_store": "V_STORE",
    "scalar_alu": "S_ALU",
}


def num_bundles(util: llo_module_pb2.StaticPerBundleUtilizationProto) -> int:
  """Returns the highest bundle index with any data, +1 (0 if empty)."""
  n = 0
  for f in FIELDS:
    rv = getattr(util, f)
    if rv.numerator:
      n = max(n, len(rv.numerator))
  if util.vector_spill:
    n = max(n, len(util.vector_spill))
  if util.vector_fill:
    n = max(n, len(util.vector_fill))
  return n


def has_static_utilization(module: llo_module_pb2.LloModuleProto) -> bool:
  """True iff the compiler emitted per-bundle static utilization."""
  return num_bundles(module.static_utilization) > 0


def extract_bundle_utilization(
    module: llo_module_pb2.LloModuleProto,
) -> list[dict[str, Any]]:
  """Returns a per-bundle list of structured utilization records.

  Each record is `{bundle, <unit>_used, <unit>_avail, spill, fill, util_pct}`.
  Returns [] when the module has no static utilization data.

  Args:
    module: The LLO module proto with optional static_utilization.
  """
  util = module.static_utilization
  total = num_bundles(util)
  out: list[dict[str, Any]] = []
  for i in range(total):
    rec: dict[str, Any] = {"bundle": i}
    used_total = 0
    avail_total = 0
    for f in FIELDS:
      rv = getattr(util, f)
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


def render_bundle_util_markdown(
    module: llo_module_pb2.LloModuleProto,
    bundle_range: tuple[int, int] | None = None,
    region_path_map: dict[int, str] | None = None,
    max_rows: int = 30,
    read_all: bool = False,
) -> str:
  """Renders a per-bundle utilization table as markdown."""
  util = module.static_utilization
  total = num_bundles(util)
  if total == 0:
    return (
        f"# Bundle util: {module.hlo_instruction_name}\n\n"
        "_No static utilization data in this LLO module."
        " (The compiler skipped the static bundle profiler for this op.)_\n"
    )
  start, end = 0, total
  if bundle_range is not None:
    start, end = bundle_range
    end = min(end, total)
    start = max(0, start)
  if not read_all and (end - start) > max_rows:
    end = start + max_rows

  buf = io.StringIO()
  buf.write(f"# Bundle util: {module.hlo_instruction_name}\n\n")
  buf.write(
      f"_Module: `{module.hlo_module_name}` (id={module.hlo_module_id}); "
      f"showing bundles {start}-{end} of {total} total_\n\n"
  )
  cols = (
      ["Bundle"] + [_COL_LABELS[f] for f in FIELDS] + ["spill", "fill", "util%"]
  )
  if region_path_map is not None:
    cols.append("region_path")
  buf.write("| " + " | ".join(cols) + " |\n")
  buf.write("|" + "|".join(["---:"] * len(cols)) + "|\n")

  records = extract_bundle_utilization(module)
  for rec in records:
    i = rec["bundle"]
    if not start <= i < end:
      continue
    row = [str(i)]
    for f in FIELDS:
      denom = rec[f"{f}_avail"]
      if not denom:
        row.append("·")
      else:
        row.append(f"{rec[f'{f}_used']}/{denom}")
    row.append(str(rec["spill"]))
    row.append(str(rec["fill"]))
    row.append(f"{rec['util_pct']}%")
    if region_path_map is not None:
      row.append(region_path_map.get(i, "-"))
    buf.write("| " + " | ".join(row) + " |\n")
  return buf.getvalue()
