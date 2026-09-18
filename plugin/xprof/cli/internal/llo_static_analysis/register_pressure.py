"""Register-pressure estimation from LLO instruction schedules.

Each `LloInstructionProto` that produces a value carries a `register_id`, a
`register_type` (PREG/SREG/VMREG/VREG), and its `scheduled_bundleno`. An
instruction's `operands` reference their producers by `instruction_ordinal`.
Joining these yields, for every produced value, a live range
`[def_bundle, last_use_bundle]`; sweeping those ranges gives the number of live
registers per bundle, stratified by register type. The per-type peaks are a
portable proxy for register pressure computed entirely from proto-resident data
(no target register-file catalog required).
"""

import io

from xprof.embedded.llo_analysis import llo_lite_pb2 as llo_instruction_pb2
from xprof.cli.internal.llo_static_analysis import llo_region_tree

_REGISTER_TYPE_NAMES: dict[int, str] = {
    llo_instruction_pb2.REGISTER_TYPE_NONE: "NONE",
    llo_instruction_pb2.REGISTER_TYPE_PREG: "PREG",
    llo_instruction_pb2.REGISTER_TYPE_SREG: "SREG",
    llo_instruction_pb2.REGISTER_TYPE_VMREG: "VMREG",
    llo_instruction_pb2.REGISTER_TYPE_VREG: "VREG",
}


class LiveRange:
  """A produced value's live range, in bundle indices."""

  __slots__ = ("ordinal", "register_type", "start", "end")

  def __init__(self, ordinal: int, register_type: int, start: int, end: int):
    self.ordinal = ordinal
    self.register_type = register_type
    self.start = start
    self.end = end


def compute_live_ranges(module) -> list[LiveRange]:
  """Returns the live range of every scheduled, register-producing instruction.

  The last-use bundle is the max `scheduled_bundleno` among consuming
  instructions; a value with no consumers is live only at its def bundle.

  Args:
    module: The LLO module proto whose instructions are analyzed.
  """
  def_bundle: dict[int, int] = {}
  def_type: dict[int, int] = {}
  last_use: dict[int, int] = {}

  insts = [inst for _, inst in llo_region_tree.iter_instructions(module)]
  for inst in insts:
    if (
        inst.register_id >= 0
        and inst.register_type != llo_instruction_pb2.REGISTER_TYPE_NONE
        and inst.scheduled_bundleno >= 0
    ):
      def_bundle[inst.ordinal] = inst.scheduled_bundleno
      def_type[inst.ordinal] = inst.register_type

  for inst in insts:
    if inst.scheduled_bundleno < 0:
      continue
    for operand in inst.operands:
      if operand.WhichOneof("value") != "instruction_ordinal":
        continue
      producer = operand.instruction_ordinal
      if producer in def_bundle:
        prev = last_use.get(producer, def_bundle[producer])
        last_use[producer] = max(prev, inst.scheduled_bundleno)

  ranges: list[LiveRange] = []
  for ordinal, start in def_bundle.items():
    end = last_use.get(ordinal, start)
    ranges.append(LiveRange(ordinal, def_type[ordinal], start, end))
  return ranges


MAX_BUNDLE_RANGE_SPAN = 100_000


def _compute_pressure_series_from_ranges(
    ranges: list[LiveRange],
) -> dict[int, dict[int, int]]:
  """Returns {bundle: {register_type: live_count}} with bounded range spans."""
  series: dict[int, dict[int, int]] = {}
  for r in ranges:
    if r.start < 0 or r.end < r.start:
      continue
    clamped_end = min(r.end, r.start + MAX_BUNDLE_RANGE_SPAN)
    for b in range(r.start, clamped_end + 1):
      per_type = series.setdefault(b, {})
      per_type[r.register_type] = per_type.get(r.register_type, 0) + 1
  return series


def _compute_peak_pressure_from_ranges(
    ranges: list[LiveRange],
) -> dict[int, int]:
  """Computes per-type peak pressure in O(N log N) via sweep-line transitions."""
  deltas_by_type: dict[int, dict[int, int]] = {}
  for r in ranges:
    if r.start < 0 or r.end < r.start:
      continue
    deltas = deltas_by_type.setdefault(r.register_type, {})
    deltas[r.start] = deltas.get(r.start, 0) + 1
    deltas[r.end + 1] = deltas.get(r.end + 1, 0) - 1

  peaks: dict[int, int] = {}
  for rtype, deltas in deltas_by_type.items():
    current = 0
    peak = 0
    for b in sorted(deltas):
      current += deltas[b]
      if current > peak:
        peak = current
    if peak > 0:
      peaks[rtype] = peak
  return peaks


def compute_pressure_series(module) -> dict[int, dict[int, int]]:
  """Returns {bundle: {register_type: live_count}}."""
  return _compute_pressure_series_from_ranges(compute_live_ranges(module))


def compute_peak_pressure(module) -> dict[int, int]:
  """Returns {register_type: peak live count across all bundles}."""
  return _compute_peak_pressure_from_ranges(compute_live_ranges(module))


def render_register_pressure_markdown(module, top_n: int = 10) -> str:
  """Renders per-type peak register pressure + the hottest bundles."""
  top_n = max(0, top_n)
  buf = io.StringIO()
  buf.write(f"# Register pressure: {module.hlo_instruction_name}\n\n")
  ranges = compute_live_ranges(module)
  peaks = _compute_peak_pressure_from_ranges(ranges)
  if not peaks:
    buf.write(
        "_No scheduled register-producing instructions found in this LLO"
        " module._\n"
    )
    return buf.getvalue()

  buf.write("## Peak pressure by register type\n\n")
  buf.write("| Register type | Peak live |\n|---|---:|\n")
  for rtype in sorted(peaks, key=lambda t: -peaks[t]):
    name = _REGISTER_TYPE_NAMES.get(rtype, f"TYPE_{rtype}")
    buf.write(f"| {name} | {peaks[rtype]} |\n")

  series = _compute_pressure_series_from_ranges(ranges)
  totals = {b: sum(pt.values()) for b, pt in series.items()}
  hottest = sorted(totals, key=lambda b: -totals[b])[:top_n]
  buf.write("\n## Hottest bundles (total live registers)\n\n")
  buf.write("| Bundle | Total live |\n|---:|---:|\n")
  for b in hottest:
    buf.write(f"| {b} | {totals[b]} |\n")
  return buf.getvalue()
