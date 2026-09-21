"""BDI (bundle dependency interlock) stall analysis for an LLO module.

There are two sources of BDI information in an LLO module, and this module
reads both because they answer different questions.

The precise source is `LloInstructionProto.bundle_packer_info`. For each
instruction it records the earliest bundle permitted by each class of
constraint -- operand latency (O), FIFO (F), register (R), arch register (A),
source bus (B), and safe hoisting (H) -- alongside `point_of_no_return_index`
(P, the earliest bundle the instruction could occupy ignoring operands) and
`final_bundle_index` (E, where it was actually scheduled). From those the
*binding* constraint and the number of bundles it cost are recoverable:

  binding_index = max over the six constraint indices
  binding_code  = highest-priority code attaining binding_index
  stall_bundles = binding_index - P      (delay attributable to dependencies)
  packing_slack = E - binding_index      (further delay from slot pressure)

Ties are broken by `BDI_PRIORITY_ORDER` (F > B > A > H > R > O), matching the
precedence the compiler's own tooling uses, so that when several constraints
are simultaneously binding the most fundamental one is attributed.

The approximate source is the `bdi:<code>:<detail>,...` fragment the compiler
encodes into the interned annotation string referenced by `annotation_handle`.
It preserves only the category letters, not the indices, so it supports a
histogram but not a cost. It remains the fallback for modules captured before
`bundle_packer_info` was mirrored into the lite proto, and the two are
cross-checked by `extract_bdi_stalls`, which reports `coverage` -- the fraction
of annotated instructions for which the precise data is also present.

Callers must distinguish "no stall data" from "no stalls". When no instruction
carries `bundle_packer_info`, `stall_analysis` is None rather than a zeroed
summary.
"""

import json
from typing import Any

from xprof.cli.internal.llo_static_analysis import (
    llo_region_tree,
)
from xprof.protobuf import llo_lite_pb2

# Recognized single-letter BDI category keys ("ORAHFB") matching the BDI
# annotation grammar documented above.
_BDI_CODES = frozenset("ORAHFB")

# Precedence used to attribute a stall when several constraints are binding at
# the same bundle. Ordered most to least fundamental: a FIFO or source-bus
# limit is a hardware structural hazard, whereas an operand-latency wait is a
# consequence of scheduling choices upstream.
BDI_PRIORITY_ORDER = ("F", "B", "A", "H", "R", "O")

# Maps each category letter to the `BundlePackerInfoProto` field holding the
# earliest bundle that constraint permits.
_BDI_INDEX_FIELDS = {
    "F": "fifo_dep_index",
    "B": "source_bus_dep_index",
    "A": "arch_register_dep_index",
    "H": "min_safe_hoist_index",
    "R": "register_dep_index",
    "O": "operand_latency_dep_index",
}


def annotation_for(
    module: llo_lite_pb2.LloModuleProto,
    inst: llo_lite_pb2.LloInstructionProto,
) -> str | None:
  """Returns the interned annotation string for `inst`, or None.

  `annotation_handle` is an index into `module.interned_strings` (matching
  the compiler-side annotation emitter, which stores
  `annotations[i] = interned_strings(i)` and looks up by handle). Returns
  None when the instruction has no handle or the handle is out of range.

  Args:
    module: The LLO module proto containing `interned_strings`.
    inst: The LLO instruction proto to look up.
  """
  if not inst.HasField("annotation_handle"):
    return None
  handle = inst.annotation_handle
  if handle < 0 or handle >= len(module.interned_strings):
    return None
  return module.interned_strings[handle]


def parse_bdi_codes(annotation: str | None) -> list[str]:
  """Extracts the ordered, de-duplicated BDI codes from an annotation string.

  Parses BDI stall codes: finds the `bdi:` marker, splits the rest on commas,
  and for each non-empty part takes the key before its first colon; keeps it
  when it is a single recognized category letter, preserving first-seen order.

  Args:
    annotation: Raw annotation string attached to an LLO instruction.

  Returns:
    Ordered list of recognized single-character BDI stall codes.
  """
  if not annotation or "bdi:" not in annotation:
    return []
  sub = annotation[annotation.find("bdi:") + 4 :]
  codes: list[str] = []
  for part in sub.split(","):
    part = part.strip()
    if not part or ":" not in part:
      continue
    key = part.split(":", 1)[0].strip()
    if len(key) == 1 and key in _BDI_CODES and key not in codes:
      codes.append(key)
  return codes


def iter_bdi_instructions(
    module: llo_lite_pb2.LloModuleProto,
) -> list[tuple[int, int, list[str]]]:
  """Returns [(ordinal, scheduled_bundleno, codes)] for annotated instructions.

  Only instructions whose annotation yields at least one BDI code are included.

  Args:
    module: The LLO module proto whose instructions are inspected.
  """
  out: list[tuple[int, int, list[str]]] = []
  for _, inst in llo_region_tree.iter_instructions(module):
    codes = parse_bdi_codes(annotation_for(module, inst))
    if codes:
      out.append((inst.ordinal, inst.scheduled_bundleno, codes))
  return out


def bdi_code_histogram(
    module: llo_lite_pb2.LloModuleProto,
) -> dict[str, int]:
  """Returns {code: number of instructions exhibiting that code}."""
  hist: dict[str, int] = {}
  for _, _, codes in iter_bdi_instructions(module):
    for code in codes:
      hist[code] = hist.get(code, 0) + 1
  return hist


def binding_constraint(
    info: llo_lite_pb2.BundlePackerInfoProto,
) -> tuple[str, int]:
  """Returns the (code, index) of the constraint that bound this instruction.

  The binding constraint is the one permitting the latest bundle, since an
  instruction cannot issue until every constraint is satisfied. When several
  attain that same latest bundle they are all simultaneously binding, and
  `BDI_PRIORITY_ORDER` decides which one is attributed.

  Args:
    info: The bundle-packer record for a single instruction.

  Returns:
    A (code letter, bundle index) pair.
  """
  indices = {
      code: getattr(info, field) for code, field in _BDI_INDEX_FIELDS.items()
  }
  latest = max(indices.values())
  for code in BDI_PRIORITY_ORDER:
    if indices[code] == latest:
      return code, latest
  # Unreachable: BDI_PRIORITY_ORDER covers every key of _BDI_INDEX_FIELDS.
  raise AssertionError("no BDI code attained the maximum index")


def iter_packer_records(
    module: llo_lite_pb2.LloModuleProto,
) -> list[dict[str, Any]]:
  """Returns one derived stall record per instruction carrying packer info.

  Instructions without `bundle_packer_info` are skipped rather than reported
  with zeroed fields, so that an empty result means "no data" and not "no
  stalls".

  Args:
    module: The LLO module proto whose instructions are inspected.
  """
  out: list[dict[str, Any]] = []
  for _, inst in llo_region_tree.iter_instructions(module):
    if not inst.HasField("bundle_packer_info"):
      continue
    info = inst.bundle_packer_info
    code, binding_index = binding_constraint(info)
    earliest = info.point_of_no_return_index
    out.append({
        "ordinal": inst.ordinal,
        "bundle": inst.scheduled_bundleno,
        "binding_code": code,
        # Clamped at zero: a constraint permitting an earlier bundle than the
        # point of no return did not delay anything.
        "stall_bundles": max(0, binding_index - earliest),
        "packing_slack": max(0, info.final_bundle_index - binding_index),
        "hoist_distance_cur": info.hoist_distance_cur,
        "hoist_distance_prev": info.hoist_distance_prev,
    })
  return out


def summarize_stalls(
    module: llo_lite_pb2.LloModuleProto,
) -> dict[str, Any] | None:
  """Aggregates per-constraint stall cost, or None when no packer info exists.

  Args:
    module: The LLO module proto whose instructions are inspected.

  Returns:
    A summary dict, or None when not a single instruction carries
    `bundle_packer_info` -- which means the capture predates the field being
    mirrored, not that the schedule was stall-free.
  """
  records = iter_packer_records(module)
  if not records:
    return None

  # Values are int counts plus, once computed below, a float share.
  by_code: dict[str, dict[str, float]] = {}

  for rec in records:
    entry = by_code.setdefault(
        rec["binding_code"], {"instructions": 0, "stall_bundles": 0}
    )
    entry["instructions"] += 1
    entry["stall_bundles"] += rec["stall_bundles"]

  total_stall = sum(e["stall_bundles"] for e in by_code.values())
  for entry in by_code.values():
    entry["share_of_stall_bundles"] = (
        round(entry["stall_bundles"] / total_stall, 4) if total_stall else 0.0
    )

  hoisted = [r for r in records if r["hoist_distance_cur"] > 0]
  hoist_distances = [r["hoist_distance_cur"] for r in hoisted]
  avg_hoist = 0.0
  if hoist_distances:
    avg_hoist = round(sum(hoist_distances) / len(hoist_distances), 2)

  annotated = len(iter_bdi_instructions(module))
  return {
      "instructions_with_packer_info": len(records),
      "total_stall_bundles": total_stall,
      "total_packing_slack_bundles": sum(r["packing_slack"] for r in records),
      "delayed_ratio": round(
          sum(1 for r in records if r["stall_bundles"] > 0) / len(records), 4
      ),
      # How much of the annotation-derived view the precise view also covers.
      # Below 1.0 means some annotated instructions lack `bundle_packer_info`.
      "coverage": round(len(records) / annotated, 4) if annotated else None,
      "by_binding_code": {
          code: by_code[code]
          for code in sorted(
              by_code, key=lambda c: (-by_code[c]["stall_bundles"], c)
          )
      },
      "hoist": {
          "instructions_hoisted": len(hoisted),
          "max_hoist_distance": max(hoist_distances, default=0),
          "avg_hoist_distance": avg_hoist,
      },
  }


def extract_bdi_stalls(
    module: llo_lite_pb2.LloModuleProto, top_n: int = 50
) -> dict[str, Any]:
  """Returns structured BDI stall-code histogram and per-instruction records."""
  records = iter_bdi_instructions(module)
  hist = bdi_code_histogram(module)
  sorted_hist = {
      code: hist[code] for code in sorted(hist, key=lambda c: (-hist[c], c))
  }
  sorted_records = [
      {"ordinal": ordinal, "bundle": bundle, "codes": codes}
      for ordinal, bundle, codes in sorted(records, key=lambda r: (r[1], r[0]))[
          :top_n
      ]
  ]
  return {
      "hlo_instruction_name": module.hlo_instruction_name,
      "total_annotated_instructions": len(records),
      "code_histogram": sorted_hist,
      "instructions": sorted_records,
      # None when the capture carries no `bundle_packer_info` at all. Absent
      # cost is not zero cost, so callers must not sum over a missing summary.
      "stall_analysis": summarize_stalls(module),
      # Ranked by cost rather than by bundle order: the question this answers
      # is "what should I look at first", which the ordinal listing above does
      # not.
      "top_stalls": sorted(
          iter_packer_records(module),
          key=lambda r: (-r["stall_bundles"], r["bundle"], r["ordinal"]),
      )[:top_n],
  }


def render_bdi_stalls_json(
    module: llo_lite_pb2.LloModuleProto, top_n: int = 50
) -> str:
  """Renders the BDI stall-code histogram and per-instruction listing as JSON."""
  return json.dumps(extract_bdi_stalls(module, top_n=top_n), indent=2)
