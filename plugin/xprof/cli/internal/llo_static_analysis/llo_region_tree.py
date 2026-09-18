"""Recursive walk of `LloRegionProto` -> markdown tree + bundle attribution.

`LloRegionProto` is recursive: a region holds members, each of which is one of
(instruction, loop, predicated_region, sub_region). Loops carry seven
sub-regions (pre_header, header, header_branch, body, footer, footer_branch,
exit) plus an `index_space` describing the trip count.

This module renders that structure as a markdown tree and builds a
bundle-index -> deepest-region-path map so other analyses can attribute a bundle
to its enclosing region without re-walking the proto.
"""

from collections.abc import Iterator
import io
from typing import Any

from xprof.embedded.llo_analysis import llo_lite_pb2 as llo_module_pb2

_LOOP_SUBREGIONS = (
    "pre_header",
    "header",
    "header_branch",
    "body",
    "footer",
    "footer_branch",
    "exit",
)
_PREDICATED_SUBREGIONS = (
    "check_region",
    "check_branch_region",
    "predicated_region",
    "fallthru_region",
)


class TreeNode:
  """A rendered node of the region tree."""

  __slots__ = ("name", "kind", "start", "limit", "trip", "step", "children")

  def __init__(
      self,
      name: str,
      kind: str,
      start: int = -1,
      limit: int = -1,
      trip: int | None = None,
      step: int | None = None,
  ):
    self.name = name
    self.kind = kind
    self.start = start
    self.limit = limit
    self.trip = trip
    self.step = step
    self.children: list["TreeNode"] = []


def _trip_count(idx: llo_module_pb2.LloLoopProto.LoopIndexSpaceProto) -> int:
  if not idx.step:
    return 0
  return max(0, (idx.limit - idx.start + idx.step - 1) // idx.step)


def _loop_bundle_range(loop: llo_module_pb2.LloLoopProto) -> tuple[int, int]:
  starts: list[int] = []
  limits: list[int] = []
  for sub_field in _LOOP_SUBREGIONS:
    if loop.HasField(sub_field):
      sub = getattr(loop, sub_field)
      starts.append(sub.start_bundleno)
      limits.append(sub.limit_bundleno)
  if not starts:
    return (-1, -1)
  return (min(starts), max(limits))


def _walk_region(region: llo_module_pb2.LloRegionProto) -> TreeNode:
  """Recursively constructs a `TreeNode` hierarchy for `region`."""
  node = TreeNode(
      name=region.name or f"#region_ord_{region.ordinal}",
      kind="region",
      start=region.start_bundleno,
      limit=region.limit_bundleno,
  )
  for member in region.members:
    which = member.WhichOneof("value")
    if which == "loop":
      node.children.append(_walk_loop(member.loop))
    elif which == "predicated_region":
      node.children.append(_walk_predicated(member.predicated_region))
    elif which == "sub_region":
      node.children.append(_walk_region(member.sub_region))
    # Per-instruction members are collapsed here; they are inspected via the
    # bundle-utilization and register-pressure analyses.
  return node


def _walk_loop(loop: llo_module_pb2.LloLoopProto) -> TreeNode:
  """Constructs a `TreeNode` for an `LloLoopProto` and its subregions."""
  if loop.kind == llo_module_pb2.LloLoopProto.LOOP_KIND_WHILE:
    kind_str = "loop_while"
  elif loop.kind == llo_module_pb2.LloLoopProto.LOOP_KIND_DOWHILE:
    kind_str = "loop_dowhile"
  else:
    kind_str = "loop"
  trip = _trip_count(loop.index_space) if loop.HasField("index_space") else None
  step = loop.index_space.step if loop.HasField("index_space") else None
  body_name = (
      loop.body.name if loop.HasField("body") and loop.body.name else "(loop)"
  )
  start, limit = _loop_bundle_range(loop)
  node = TreeNode(
      name=body_name,
      kind=kind_str,
      start=start,
      limit=limit,
      trip=trip,
      step=step,
  )
  for sub_field in _LOOP_SUBREGIONS:
    if loop.HasField(sub_field):
      child = _walk_region(getattr(loop, sub_field))
      child.kind = f"region/{sub_field}"
      node.children.append(child)
  return node


def _walk_predicated(
    pred: llo_module_pb2.LloPredicatedRegionProto,
) -> TreeNode:
  """Constructs a `TreeNode` for an `LloPredicatedRegionProto`."""
  starts, limits, children = [], [], []
  for sub_field in _PREDICATED_SUBREGIONS:
    if pred.HasField(sub_field):
      sub = getattr(pred, sub_field)
      starts.append(sub.start_bundleno)
      limits.append(sub.limit_bundleno)
      child = _walk_region(sub)
      child.kind = f"region/{sub_field}"
      children.append(child)
  node = TreeNode(
      name="(predicated)",
      kind="predicated",
      start=min(starts) if starts else -1,
      limit=max(limits) if limits else -1,
  )
  node.children.extend(children)
  return node


def build_tree(module: llo_module_pb2.LloModuleProto) -> TreeNode:
  """Returns the root `TreeNode` for `module.top_region`."""
  if not module.HasField("top_region"):
    return TreeNode(name="(empty)", kind="region")
  return _walk_region(module.top_region)


def render_tree_markdown(
    module: llo_module_pb2.LloModuleProto,
    max_depth: int | None = None,
) -> str:
  """Renders the LLO region tree as a markdown nested list."""
  buf = io.StringIO()
  buf.write(
      f"# LLO region tree: {module.hlo_instruction_name or '(unnamed)'}\n\n"
  )
  if module.hlo_module_name:
    buf.write(
        f"_HLO module: `{module.hlo_module_name}`"
        f" (id={module.hlo_module_id})_\n\n"
    )
  total = 0
  if module.HasField("top_region"):
    total = max(
        0, module.top_region.limit_bundleno - module.top_region.start_bundleno
    )
  buf.write(f"_Total bundles: {total}_\n\n")
  root = build_tree(module)
  _render_node(buf, root, prefix="", is_last=True, depth=0, max_depth=max_depth)
  return buf.getvalue()


def _render_node(
    buf,
    node: TreeNode,
    prefix: str,
    is_last: bool,
    depth: int,
    max_depth: int | None,
) -> None:
  """Writes `node` and its descendants to `buf` as an ASCII tree."""
  span = (
      f" [bundle {node.start}-{node.limit})"
      if node.start >= 0 and node.limit >= 0
      else ""
  )
  count = max(0, node.limit - node.start) if node.start >= 0 else 0
  count_str = f" ({count} bundles)" if count > 0 else ""
  trip_str = (
      f" trip={node.trip} step={node.step}" if node.trip is not None else ""
  )
  marker = "└── " if is_last else "├── "
  if depth == 0:
    buf.write(f"{node.name}{span}{count_str}{trip_str}\n")
  else:
    buf.write(f"{prefix}{marker}{node.name}{span}{count_str}{trip_str}\n")
  if max_depth is not None and depth >= max_depth:
    if node.children:
      sub_prefix = prefix + ("    " if is_last else "│   ")
      buf.write(f"{sub_prefix}... ({len(node.children)} children elided)\n")
    return
  sub_prefix = prefix + ("    " if is_last else "│   ") if depth > 0 else ""
  for i, child in enumerate(node.children):
    _render_node(
        buf,
        child,
        prefix=sub_prefix,
        is_last=(i == len(node.children) - 1),
        depth=depth + 1,
        max_depth=max_depth,
    )


def build_bundle_to_region_path(
    module: llo_module_pb2.LloModuleProto,
) -> dict[int, str]:
  """Returns {bundle_idx: deepest-region-path}.

  Depth-first walk; inner regions overwrite outer ones so the deepest
  containing region wins for each bundle.

  Args:
    module: The LLO module proto whose top_region hierarchy is traversed.
  """
  out: dict[int, str] = {}
  if not module.HasField("top_region"):
    return out
  _attribute_region(module.top_region, out, [])
  return out


MAX_BUNDLE_RANGE_SPAN = 100_000


def _attribute_region(
    region: llo_module_pb2.LloRegionProto,
    out: dict[int, str],
    parents: list[str],
) -> None:
  """Populates `out` with bundle-to-region-path mappings for `region`."""
  name = region.name or f"#region_ord_{region.ordinal}"
  path = parents + [name]
  if (
      region.start_bundleno >= 0
      and region.limit_bundleno > region.start_bundleno
  ):
    start = region.start_bundleno
    limit = min(region.limit_bundleno, start + MAX_BUNDLE_RANGE_SPAN)
    for b in range(start, limit):
      out[b] = "/".join(path)
  for member in region.members:
    which = member.WhichOneof("value")
    if which == "loop":
      for sub_field in _LOOP_SUBREGIONS:
        if member.loop.HasField(sub_field):
          _attribute_region(getattr(member.loop, sub_field), out, path)
    elif which == "predicated_region":
      for sub_field in _PREDICATED_SUBREGIONS:
        if member.predicated_region.HasField(sub_field):
          _attribute_region(
              getattr(member.predicated_region, sub_field), out, path
          )
    elif which == "sub_region":
      _attribute_region(member.sub_region, out, path)


def iter_regions(
    module: llo_module_pb2.LloModuleProto,
) -> Iterator[llo_module_pb2.LloRegionProto]:
  """Yields every `LloRegionProto` in `module` depth-first."""
  if not module.HasField("top_region"):
    return
  yield from _iter_region(module.top_region)


def _iter_region(
    region: llo_module_pb2.LloRegionProto,
) -> Iterator[llo_module_pb2.LloRegionProto]:
  """Yields `region` and its nested subregions depth-first."""
  yield region
  for member in region.members:
    which = member.WhichOneof("value")
    if which == "loop":
      for sub_field in _LOOP_SUBREGIONS:
        if member.loop.HasField(sub_field):
          yield from _iter_region(getattr(member.loop, sub_field))
    elif which == "predicated_region":
      for sub_field in _PREDICATED_SUBREGIONS:
        if member.predicated_region.HasField(sub_field):
          yield from _iter_region(getattr(member.predicated_region, sub_field))
    elif which == "sub_region":
      yield from _iter_region(member.sub_region)


def iter_instructions(
    module: llo_module_pb2.LloModuleProto,
) -> Iterator[tuple[llo_module_pb2.LloRegionProto, Any]]:
  """Yields (enclosing_region, instruction) for every instruction member."""
  for region in iter_regions(module):
    for member in region.members:
      if member.WhichOneof("value") == "instruction":
        yield (region, member.instruction)
