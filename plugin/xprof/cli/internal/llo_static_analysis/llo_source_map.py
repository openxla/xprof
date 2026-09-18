"""Source <-> bundle mapping using an exact `scheduled_bundleno` join.

`LloModuleProto.source_map` (`SourceMapProto`) associates source frames with LLO
instruction *ordinals*. Each `LloInstructionProto` carries both an `ordinal` and
its `scheduled_bundleno` (the exact bundle it issues in). Joining these gives an
exact source-frame <-> bundle mapping, which is more precise than approximating
via the enclosing region's bundle range.

For robustness, when a source-map ordinal does not correspond to a scheduled
instruction (e.g. an unscheduled pseudo, or a proto that only records region
ordinals), we fall back to the enclosing region's bundle range.
"""

import io

from xprof.embedded.llo_analysis import llo_lite_pb2 as llo_module_pb2
from xprof.cli.internal.llo_static_analysis import llo_region_tree


MAX_BUNDLE_RANGE_SPAN = 100_000


def build_ordinal_to_bundle(
    module: llo_module_pb2.LloModuleProto,
) -> dict[int, int]:
  """Returns {instruction_ordinal: scheduled_bundleno} for scheduled insts."""
  out: dict[int, int] = {}
  for _, inst in llo_region_tree.iter_instructions(module):
    if inst.scheduled_bundleno >= 0:
      out[inst.ordinal] = inst.scheduled_bundleno
  return out


def _build_region_ordinal_ranges(
    module: llo_module_pb2.LloModuleProto,
) -> dict[int, tuple[int, int]]:
  """Returns {region_ordinal: (start_bundleno, clamped_limit_bundleno)}."""
  out: dict[int, tuple[int, int]] = {}
  for region in llo_region_tree.iter_regions(module):
    if (
        region.ordinal >= 0
        and region.start_bundleno >= 0
        and region.limit_bundleno > region.start_bundleno
    ):
      start = region.start_bundleno
      limit = min(region.limit_bundleno, start + MAX_BUNDLE_RANGE_SPAN)
      out[region.ordinal] = (start, limit)
  return out


def _frame_str(source_map, frame) -> str:
  path = "?"
  if 0 <= frame.path < len(source_map.strings):
    path = source_map.strings[frame.path]
  return f"{path}:{frame.line_start}"


def _innermost_frame_str(source_map, loc) -> str:
  if not loc.frames:
    return "(no frame)"
  # Frames are ordered outer -> inner; the inner-most is the last.
  return _frame_str(source_map, loc.frames[-1])


def build_source_by_bundle(
    module: llo_module_pb2.LloModuleProto,
) -> dict[int, list[str]]:
  """Returns {bundle: [innermost source frame strings]} (deduped, ordered).

  Uses the exact instruction-ordinal -> scheduled_bundleno join, falling back to
  region ranges for ordinals that are not scheduled instructions.

  Args:
    module: The LLO module proto containing source_map and scheduled bundles.
  """
  source_map = module.source_map
  ord_to_bundle = build_ordinal_to_bundle(module)
  region_ranges = _build_region_ordinal_ranges(module)
  by_bundle: dict[int, list[str]] = {}

  def _add(bundle: int, label: str) -> None:
    lst = by_bundle.setdefault(bundle, [])
    if label not in lst:
      lst.append(label)

  for loc in source_map.locations:
    label = _innermost_frame_str(source_map, loc)
    for ordinal in loc.ordinals:
      if ordinal in ord_to_bundle:
        _add(ord_to_bundle[ordinal], label)
      elif ordinal in region_ranges:
        start, limit = region_ranges[ordinal]
        if 0 <= start < limit:
          clamped_limit = min(limit, start + MAX_BUNDLE_RANGE_SPAN)
          for b in range(start, clamped_limit):
            _add(b, label)
  return by_bundle


def render_source_for_bundle(
    module: llo_module_pb2.LloModuleProto, bundle: int
) -> str:
  """Renders the source frame(s) associated with `bundle`."""
  by_bundle = build_source_by_bundle(module)
  buf = io.StringIO()
  buf.write(f"# Source for bundle {bundle}: {module.hlo_instruction_name}\n\n")
  labels = by_bundle.get(bundle)
  if not labels:
    buf.write(f"_No source mapping for bundle {bundle}._\n")
    return buf.getvalue()
  for label in labels:
    buf.write(f"- `{label}`\n")
  return buf.getvalue()


def render_bundles_for_source(
    module: llo_module_pb2.LloModuleProto, source_query: str
) -> str:
  """Renders the bundles whose source frame matches `source_query` substring."""
  by_bundle = build_source_by_bundle(module)
  matched: dict[str, list[int]] = {}
  for bundle, labels in by_bundle.items():
    for label in labels:
      if source_query in label:
        matched.setdefault(label, []).append(bundle)
  buf = io.StringIO()
  buf.write(
      f"# Bundles for source '{source_query}':"
      f" {module.hlo_instruction_name}\n\n"
  )
  if not matched:
    buf.write(f"_No bundles map to source matching '{source_query}'._\n")
    return buf.getvalue()
  for label, bundles in sorted(matched.items()):
    bundles.sort()
    lo, hi = bundles[0], bundles[-1]
    buf.write(
        f"- `{label}`: {len(bundles)} bundle(s), range [{lo}, {hi + 1})\n"
    )
  return buf.getvalue()
