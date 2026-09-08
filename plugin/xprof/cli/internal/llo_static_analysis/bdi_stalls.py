"""BDI (bundle dependency interlock) stall-code analysis for an LLO module.

An `LloInstructionProto` may carry an `annotation_handle` (field 9) that indexes
into the module's `interned_strings`. The compiler encodes bundle dependency
interlock (BDI) stall reasons into that annotation string as a
`bdi:<code>:<detail>,...` fragment. This module recovers those codes directly
from the proto's metadata plane, so it needs no collector support and no device
stat stream.

The parser scans for the literal `bdi:` prefix, splits the remainder on commas,
and keeps the single-letter key before the first colon of each part when that
key is one of the recognized BDI categories. The recognized categories are the
six letters in `_BDI_CODES` (O, R, A, H, F, B). These are the compiler's raw
interlock category letters; this module reports them verbatim and does not
fabricate human-readable descriptions for them.
"""

import json
from typing import Any

from xprof.embedded.llo_analysis import llo_lite_pb2
from xprof.cli.internal.llo_static_analysis import (
    llo_region_tree,
)

# Recognized single-letter BDI category keys ("ORAHFB") matching the BDI
# annotation grammar documented above.
_BDI_CODES = frozenset("ORAHFB")


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
  }


def render_bdi_stalls_json(
    module: llo_lite_pb2.LloModuleProto, top_n: int = 50
) -> str:
  """Renders the BDI stall-code histogram and per-instruction listing as JSON."""
  return json.dumps(extract_bdi_stalls(module, top_n=top_n), indent=2)
