"""Opcode distribution analysis for an LLO module.

Every `LloInstructionProto` carries an `opcode` (`LloOpcodeProto` enum). The
enum
name is a stable, authoritative identifier that is available directly from the
proto, so an opcode histogram (overall and per enclosing region) needs no
external ISA catalog.

A *coarse* functional category is also inferred from the opcode name for a quick
at-a-glance breakdown. This is a heuristic grouping only: the authoritative
per-opcode functional-unit / cycle-class classification lives in the TPU
ISA opcode catalog, not in this proto, so the category field must not be
treated as ground truth.
"""

import json

from xprof.embedded.llo_analysis import llo_lite_pb2
from xprof.cli.internal.llo_static_analysis import (
    llo_region_tree,
)

# Enum value -> name, resolved from the opcode field descriptor.
_OPCODE_VALUES = llo_lite_pb2.LloInstructionProto.DESCRIPTOR.fields_by_name[
    "opcode"
].enum_type.values_by_number


def opcode_name(value: int) -> str:
  """Returns the enum name for an opcode value (or a numeric fallback)."""
  desc = _OPCODE_VALUES.get(value)
  return desc.name if desc is not None else f"OPCODE_{value}"


def coarse_category(name: str) -> str:
  """Returns a heuristic functional category for an opcode enum name.

  The order of the checks matters: matrix and cross-lane ops are named with a
  `OPCODE_VECTOR_` prefix, so they must be matched before the generic vector
  bucket. This is an approximation, not the authoritative ISA classification.

  Args:
    name: The LLO opcode enum name string.
  """
  if any(
      token in name
      for token in ("MATMUL", "MATPREP", "MATRES", "DONE_WITH_GAINS", "LATCH")
  ) or name.endswith(("LOAD_GMR", "LOAD_LMR")):
    return "matrix"
  if "TRANSPOSE" in name:
    return "crosslane"
  if name.startswith("OPCODE_DMA"):
    return "dma"
  if "PREDICATE" in name:
    return "predicate"
  if any(
      token in name
      for token in ("SYNC", "FENCE", "BARRIER", "EVENT", "FLAG", "HALT")
  ):
    return "control"
  if name.startswith("OPCODE_SCALAR"):
    return "scalar"
  if name.startswith("OPCODE_VECTOR"):
    return "vector"
  return "other"


def opcode_histogram(module) -> dict[str, int]:
  """Returns {opcode_name: count} across all instructions in the module."""
  hist: dict[str, int] = {}
  for _, inst in llo_region_tree.iter_instructions(module):
    name = opcode_name(inst.opcode)
    hist[name] = hist.get(name, 0) + 1
  return hist


def category_histogram(module) -> dict[str, int]:
  """Returns {coarse_category: count} across all instructions."""
  hist: dict[str, int] = {}
  for _, inst in llo_region_tree.iter_instructions(module):
    cat = coarse_category(opcode_name(inst.opcode))
    hist[cat] = hist.get(cat, 0) + 1
  return hist


def render_opcode_stats_json(module, top_n: int = 30) -> str:
  """Renders the opcode histogram + coarse category breakdown as JSON."""
  hist = opcode_histogram(module)
  total = sum(hist.values())
  cats = category_histogram(module)
  categories = [
      {
          "category": cat,
          "count": cats[cat],
          "share_pct": round(100 * cats[cat] / total) if total else 0,
      }
      for cat in sorted(cats, key=lambda c: -cats[c])
  ]
  top_opcodes = [
      {
          "opcode": name,
          "count": hist[name],
          "share_pct": round(100 * hist[name] / total) if total else 0,
      }
      for name in sorted(hist, key=lambda n: (-hist[n], n))[:top_n]
  ]
  payload = {
      "hlo_instruction_name": module.hlo_instruction_name,
      "total_instructions": total,
      "categories": categories,
      "top_opcodes": top_opcodes,
  }
  return json.dumps(payload, indent=2)
