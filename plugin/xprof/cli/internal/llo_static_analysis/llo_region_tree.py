"""Recursive walk of `LloRegionProto` -> JSON tree + bundle attribution.

`LloRegionProto` is recursive: a region holds members, each of which is one of
(instruction, loop, predicated_region, sub_region). Loops carry seven
sub-regions (pre_header, header, header_branch, body, footer, footer_branch,
exit) plus an `index_space` describing the trip count.

This module renders that structure as JSON and builds a
bundle-index -> deepest-region-path map so other analyses can attribute a bundle
to its enclosing region without re-walking the proto.
"""

from collections.abc import Iterator
import json
from typing import Any

from xprof.embedded.llo_analysis import llo_lite_pb2


class TreeNode:
  """A node of the LLO region tree."""

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


def _trip_count(idx: llo_lite_pb2.LloLoopProto.LoopIndexSpaceProto) -> int:
  if not idx.step:
    return 0
  return max(0, (idx.limit - idx.start + idx.step - 1) // idx.step)


def _iter_present_loop_subregions(
    loop: llo_lite_pb2.LloLoopProto,
) -> Iterator[tuple[str, llo_lite_pb2.LloRegionProto]]:
  """Yields (field_name, subregion) for each subregion set on `loop`.

  Args:
    loop: The loop whose seven optional subregions are inspected, in program
      order (pre_header, header, header_branch, body, footer, footer_branch,
      exit).

  Yields:
    A (field_name, region) pair for every subregion that is present.
  """
  for sub_field, region in (
      ("pre_header", loop.pre_header),
      ("header", loop.header),
      ("header_branch", loop.header_branch),
      ("body", loop.body),
      ("footer", loop.footer),
      ("footer_branch", loop.footer_branch),
      ("exit", loop.exit),
  ):
    if loop.HasField(sub_field):
      yield (sub_field, region)


def _iter_present_predicated_subregions(
    pred: llo_lite_pb2.LloPredicatedRegionProto,
) -> Iterator[tuple[str, llo_lite_pb2.LloRegionProto]]:
  for sub_field, region in (
      ("check_region", pred.check_region),
      ("check_branch_region", pred.check_branch_region),
      ("predicated_region", pred.predicated_region),
      ("fallthru_region", pred.fallthru_region),
  ):
    if pred.HasField(sub_field):
      yield (sub_field, region)


def _loop_bundle_range(loop: llo_lite_pb2.LloLoopProto) -> tuple[int, int]:
  starts: list[int] = []
  limits: list[int] = []
  for _, sub in _iter_present_loop_subregions(loop):
    starts.append(sub.start_bundleno)
    limits.append(sub.limit_bundleno)
  if not starts:
    return (-1, -1)
  return (min(starts), max(limits))


def _walk_region(region: llo_lite_pb2.LloRegionProto) -> TreeNode:
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


def _walk_loop(loop: llo_lite_pb2.LloLoopProto) -> TreeNode:
  """Constructs a `TreeNode` for an `LloLoopProto` and its subregions."""
  if loop.kind == llo_lite_pb2.LloLoopProto.LOOP_KIND_WHILE:
    kind_str = "loop_while"
  elif loop.kind == llo_lite_pb2.LloLoopProto.LOOP_KIND_DOWHILE:
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
  for sub_field, sub_region in _iter_present_loop_subregions(loop):
    child = _walk_region(sub_region)
    child.kind = f"region/{sub_field}"
    node.children.append(child)
  return node


def _walk_predicated(
    pred: llo_lite_pb2.LloPredicatedRegionProto,
) -> TreeNode:
  """Constructs a `TreeNode` for an `LloPredicatedRegionProto`."""
  starts, limits, children = [], [], []
  for sub_field, sub in _iter_present_predicated_subregions(pred):
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


def build_tree(module: llo_lite_pb2.LloModuleProto) -> TreeNode:
  """Returns the root `TreeNode` for `module.top_region`."""
  if not module.HasField("top_region"):
    return TreeNode(name="(empty)", kind="region")
  return _walk_region(module.top_region)


def tree_to_dict(
    node: TreeNode,
    depth: int = 0,
    max_depth: int | None = None,
) -> dict[str, Any]:
  """Converts `node` and its descendants to a JSON-serializable dict."""
  count = max(0, node.limit - node.start) if node.start >= 0 else 0
  out: dict[str, Any] = {
      "name": node.name,
      "kind": node.kind,
      "start_bundle": node.start,
      "limit_bundle": node.limit,
      "bundle_count": count,
  }
  if node.trip is not None:
    out["trip_count"] = node.trip
  if node.step is not None:
    out["step"] = node.step
  if max_depth is not None and depth >= max_depth:
    out["elided_children"] = len(node.children)
    out["children"] = []
  else:
    out["children"] = [
        tree_to_dict(c, depth=depth + 1, max_depth=max_depth)
        for c in node.children
    ]
  return out


def render_tree_json(
    module: llo_lite_pb2.LloModuleProto,
    max_depth: int | None = None,
) -> str:
  """Renders the LLO region tree as JSON."""
  total = 0
  if module.HasField("top_region"):
    total = max(
        0, module.top_region.limit_bundleno - module.top_region.start_bundleno
    )
  root = build_tree(module)
  payload = {
      "hlo_instruction_name": module.hlo_instruction_name or "(unnamed)",
      "hlo_module_name": module.hlo_module_name,
      "hlo_module_id": int(module.hlo_module_id),
      "total_bundles": total,
      "root": tree_to_dict(root, depth=0, max_depth=max_depth),
  }
  return json.dumps(payload, indent=2)


def build_bundle_to_region_path(
    module: llo_lite_pb2.LloModuleProto,
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
    region: llo_lite_pb2.LloRegionProto,
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
      for _, sub in _iter_present_loop_subregions(member.loop):
        _attribute_region(sub, out, path)
    elif which == "predicated_region":
      for _, sub in _iter_present_predicated_subregions(
          member.predicated_region
      ):
        _attribute_region(sub, out, path)
    elif which == "sub_region":
      _attribute_region(member.sub_region, out, path)


def iter_regions(
    module: llo_lite_pb2.LloModuleProto,
) -> Iterator[llo_lite_pb2.LloRegionProto]:
  """Yields every `LloRegionProto` in `module` depth-first."""
  if not module.HasField("top_region"):
    return
  yield from _iter_region(module.top_region)


def _iter_region(
    region: llo_lite_pb2.LloRegionProto,
) -> Iterator[llo_lite_pb2.LloRegionProto]:
  """Yields `region` and its nested subregions depth-first."""
  yield region
  for member in region.members:
    which = member.WhichOneof("value")
    if which == "loop":
      for _, sub in _iter_present_loop_subregions(member.loop):
        yield from _iter_region(sub)
    elif which == "predicated_region":
      for _, sub in _iter_present_predicated_subregions(
          member.predicated_region
      ):
        yield from _iter_region(sub)
    elif which == "sub_region":
      yield from _iter_region(member.sub_region)


def iter_instructions(
    module: llo_lite_pb2.LloModuleProto,
) -> Iterator[tuple[llo_lite_pb2.LloRegionProto, Any]]:
  """Yields (enclosing_region, instruction) for every instruction member."""
  for region in iter_regions(module):
    for member in region.members:
      if member.WhichOneof("value") == "instruction":
        yield (region, member.instruction)
