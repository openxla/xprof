"""Spill / allocation extraction from an LLO module.

Three complementary proto-resident signals describe spilling and memory
allocation, all extractable from the serialized LloModuleProto:

  1. `interned_constants[].allocation` (`LloAllocationProto`) with `is_spill`,
     `is_scoped`, `is_remote`, `is_virtual`, `size`, `space`.
  2. Instructions whose `pseudo_kind` metadata is a spill/fill pseudo
     (SPILL_TO_MEMORY, FILL_FROM_MEMORY, SPILL_TO_HBM, ...).
  3. `static_utilization.vector_spill` / `vector_fill`, the per-bundle spill and
     fill counts emitted by the compiler's static bundle profiler.
"""

import io

from xprof.embedded.llo_analysis import llo_lite_pb2 as llo_instruction_pb2
from xprof.embedded.llo_analysis import llo_lite_pb2 as llo_module_pb2
from xprof.cli.internal.llo_static_analysis import (
    llo_region_tree,
)

_SPILL_FILL_PSEUDO_KINDS = {
    llo_instruction_pb2.PSEUDO_KIND_SPILL_TO_MEMORY: "SPILL_TO_MEMORY",
    llo_instruction_pb2.PSEUDO_KIND_FILL_FROM_MEMORY: "FILL_FROM_MEMORY",
    llo_instruction_pb2.PSEUDO_KIND_SPILL_TO_REGISTER: "SPILL_TO_REGISTER",
    llo_instruction_pb2.PSEUDO_KIND_SPILL_TO_HBM: "SPILL_TO_HBM",
    llo_instruction_pb2.PSEUDO_KIND_FILL_FROM_HBM: "FILL_FROM_HBM",
    llo_instruction_pb2.PSEUDO_KIND_SPILL_TO_VIRTUAL: "SPILL_TO_VIRTUAL",
    llo_instruction_pb2.PSEUDO_KIND_FILL_FROM_VIRTUAL: "FILL_FROM_VIRTUAL",
}


def extract_allocations(
    module: llo_module_pb2.LloModuleProto,
) -> list[dict[str, int]]:
  """Returns structured records for every interned allocation constant.

  Values are ints; boolean flags are stored as 0/1.

  Args:
    module: The LLO module proto containing interned_constants.
  """
  out: list[dict[str, int]] = []
  for const in module.interned_constants:
    if const.WhichOneof("value") != "allocation":
      continue
    alloc = const.allocation
    out.append({
        "ordinal": int(alloc.ordinal),
        "size": int(alloc.size),
        "space": int(alloc.space),
        "is_spill": int(alloc.is_spill),
        "is_scoped": int(alloc.is_scoped),
        "is_remote": int(alloc.is_remote),
        "is_virtual": int(alloc.is_virtual),
    })
  return out


def extract_spill_fill_instructions(
    module: llo_module_pb2.LloModuleProto,
) -> list[dict[str, object]]:
  """Returns records for instructions whose pseudo_kind is a spill/fill."""
  out: list[dict[str, object]] = []
  for _, inst in llo_region_tree.iter_instructions(module):
    if inst.WhichOneof("metadata") != "pseudo_kind":
      continue
    if inst.pseudo_kind not in _SPILL_FILL_PSEUDO_KINDS:
      continue
    out.append({
        "ordinal": inst.ordinal,
        "bundle": inst.scheduled_bundleno,
        "kind": _SPILL_FILL_PSEUDO_KINDS[inst.pseudo_kind],
    })
  return out


def static_spill_fill_totals(
    module: llo_module_pb2.LloModuleProto,
) -> dict[str, int]:
  """Returns {'spill': sum, 'fill': sum} from static_utilization."""
  util = module.static_utilization
  return {
      "spill": sum(util.vector_spill),
      "fill": sum(util.vector_fill),
  }


def summarize(module: llo_module_pb2.LloModuleProto) -> dict[str, object]:
  """Returns an aggregate spill/allocation summary."""
  allocs = extract_allocations(module)
  spill_insts = extract_spill_fill_instructions(module)
  return {
      "num_allocations": len(allocs),
      "num_spill_allocations": sum(1 for a in allocs if a["is_spill"]),
      "num_scoped_allocations": sum(1 for a in allocs if a["is_scoped"]),
      "num_remote_allocations": sum(1 for a in allocs if a["is_remote"]),
      "total_allocation_bytes": sum(int(a["size"]) for a in allocs),
      "num_spill_fill_instructions": len(spill_insts),
      "static_spill_fill": static_spill_fill_totals(module),
  }


def render_allocations_markdown(
    module: llo_module_pb2.LloModuleProto, max_rows: int = 20
) -> str:
  """Renders a spill / allocation report as markdown."""
  buf = io.StringIO()
  buf.write(f"# Spills & allocations: {module.hlo_instruction_name}\n\n")
  summary = summarize(module)
  buf.write("## Summary\n\n")
  buf.write("| Metric | Value |\n|---|---:|\n")
  buf.write(f"| Allocations | {summary['num_allocations']} |\n")
  buf.write(f"| Spill allocations | {summary['num_spill_allocations']} |\n")
  buf.write(f"| Scoped allocations | {summary['num_scoped_allocations']} |\n")
  buf.write(f"| Remote allocations | {summary['num_remote_allocations']} |\n")
  buf.write(
      f"| Total allocation bytes | {summary['total_allocation_bytes']} |\n"
  )
  buf.write(
      "| Spill/fill pseudo-instructions |"
      f" {summary['num_spill_fill_instructions']} |\n"
  )
  sff = static_spill_fill_totals(module)
  buf.write(f"| Static spill count | {sff['spill']} |\n")
  buf.write(f"| Static fill count | {sff['fill']} |\n")

  spill_insts = extract_spill_fill_instructions(module)
  if spill_insts:
    buf.write("\n## Spill / fill instructions\n\n")
    buf.write("| Ordinal | Bundle | Kind |\n|---:|---:|---|\n")
    for rec in spill_insts[:max_rows]:
      buf.write(f"| {rec['ordinal']} | {rec['bundle']} | {rec['kind']} |\n")
  return buf.getvalue()
