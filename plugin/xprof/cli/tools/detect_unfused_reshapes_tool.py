"""MCP tool to detect unfused reshape operations causing HBM overhead."""

import collections
from collections.abc import Callable
import json
import logging
import os
import re
import time
from typing import Any

from xprof.cli.internal.oss import hlo_tools
from xprof.cli.tools import get_top_hlo_ops_tool

MIN_BYTES_ACCESSED: int = 0
DEFAULT_MIN_SELF_TIME_MS: float = 0.0

_FORMATTING_CATEGORIES = frozenset(
    {"data formatting", "copy", "reshape", "transpose"}
)
_FORMATTING_KEYWORDS = (
    "reshape",
    "transpose",
    "copy",
    "broadcast",
    "slice",
    "pad",
    "convert",
)
_FORMATTING_OPCODES = frozenset(_FORMATTING_KEYWORDS)

_COMPUTE_OPCODES = (
    "dot",
    "convolution",
    "einsum",
    "custom-call",
)
_COMPUTE_OPCODES_SET = frozenset(_COMPUTE_OPCODES)

# Zero-cost "view" opcodes. A real relayout may reach compute through a chain of
# these (e.g. a physical copy followed by a free bitcast-reshape). The compute
# consumer gate sees through them so the originating relayout is still flagged.
_VIEW_OPCODES = frozenset({"reshape", "bitcast", "copy", "transpose"})
_MAX_VIEW_CHAIN_DEPTH = 4

_COLLECTIVE_KEYWORDS = (
    "all-reduce",
    "all-to-all",
    "all-gather",
    "collective-permute",
    "reduce-scatter",
    "ragged-all-to-all",
    "megablox",
    "collective",
    "send",
    "recv",
)

_FRAMEWORK_PATH_PATTERNS = (
    r"/flax/",
    r"/jax/",
    r"/jaxlib/",
    r"/tensorflow/",
    r"/tf_nightly/",
    r"/optax/",
    r"/orbax/",
    r"/xla/",
    r"python[0-9.]*/lib/",
    r"site-packages/",
    r"axes_scan\.py",
)
_FRAMEWORK_PATH_RE = re.compile("|".join(_FRAMEWORK_PATH_PATTERNS))

_FORMATTING_CODE_PATTERNS = (
    r"\.reshape\(",
    r"\.transpose\(",
    r"\.swapaxes\(",
    r"\.squeeze\(",
    r"\.expand_dims\(",
    r"jnp\.reshape\(",
    r"jnp\.transpose\(",
    r"jnp\.swapaxes\(",
    r"np\.reshape\(",
    r"np\.transpose\(",
    r"einops",
    r"einsum",
)
_FORMATTING_CODE_RE = re.compile("|".join(_FORMATTING_CODE_PATTERNS))

_PASSTHROUGH_PATTERNS = (
    r"^\s*return\s+\w+\s*$",
    r"^\s*def\s+",
    r"^\s*class\s+",
    r"^\s*\"\"\"",
    r"^\s*'''",
    r"^\s*#",
    r"^\s*pass\s*$",
)
_PASSTHROUGH_RE = re.compile("|".join(_PASSTHROUGH_PATTERNS))


def _get_shape_dimensions(shape: Any) -> list[int]:
  if shape is None:
    return []
  if hasattr(shape, "dimensions"):
    return list(shape.dimensions)
  return []


def _get_shape_layout(shape: Any) -> list[int] | None:
  if shape is None:
    return None
  layout = getattr(shape, "layout", None)
  if layout is not None and hasattr(layout, "minor_to_major"):
    return list(layout.minor_to_major)
  return None


def _is_real_relayout(
    instr: Any, operand_instr: Any | None
) -> tuple[bool, str]:
  """Determines if an instruction performs a real physical HBM relayout.

  Args:
    instr: The formatting instruction.
    operand_instr: The operand instruction producing the input tensor.

  Returns:
    (is_real, reason) tuple.
  """
  if operand_instr is None:
    return True, "no_operand_instr"

  instr_shape = getattr(instr, "shape", None)
  op_shape = getattr(operand_instr, "shape", None)
  if instr_shape is None or op_shape is None:
    return True, "unverified_layout"

  instr_layout = _get_shape_layout(instr_shape)
  op_layout = _get_shape_layout(op_shape)
  instr_dims = _get_shape_dimensions(instr_shape)
  op_dims = _get_shape_dimensions(op_shape)

  if instr_layout is None or op_layout is None:
    return True, "unverified_layout"

  opcode = instr.opcode.lower()

  if opcode == "copy":
    if instr_dims == op_dims and instr_layout == op_layout:
      return False, "identical_copy"
    return True, "copy_relayout"

  if opcode == "transpose":
    perm = list(getattr(instr, "dimensions", []))
    if perm and perm == list(range(len(perm))):
      return False, "identity_transpose"
    if instr_dims == op_dims and instr_layout == op_layout:
      return False, "no_layout_change_transpose"
    return True, "transpose_relayout"

  if opcode == "reshape":
    if instr_dims == op_dims and instr_layout == op_layout:
      return False, "identity_reshape"
    is_op_default = op_layout == list(reversed(range(len(op_dims))))
    is_instr_default = instr_layout == list(reversed(range(len(instr_dims))))
    if is_op_default and is_instr_default:
      return False, "free_bitcast_reshape"
    return True, "reshape_relayout"

  return True, "other_formatting_relayout"


def _classify_source_line(
    source_file: str, source_line: int
) -> tuple[bool, str]:
  """Verifies whether source line corresponds to a real formatting operation.

  Args:
    source_file: Source code file path.
    source_line: 1-indexed source line number.

  Returns:
    (is_valid, reason) tuple.
  """
  if not source_file or source_line <= 0:
    return True, "no_source_info"

  resolved_path = source_file
  if not os.path.isabs(resolved_path):
    for root in [os.getcwd(), "."]:
      candidate = os.path.join(root, resolved_path)
      if os.path.isfile(candidate):
        resolved_path = candidate
        break

  try:
    with open(resolved_path, "r", encoding="utf-8", errors="replace") as f:
      lines = f.readlines()
      if 1 <= source_line <= len(lines):
        line_text = lines[source_line - 1].strip()
        start = max(0, source_line - 3)
        end = min(len(lines), source_line + 2)
        window_text = " ".join(l.strip() for l in lines[start:end])

        if _PASSTHROUGH_RE.search(line_text) and not _FORMATTING_CODE_RE.search(
            window_text
        ):
          return False, f"passthrough_line: {line_text[:50]}"
        return True, "source_verified"
      return True, "line_out_of_range"
  except (OSError, IOError):
    return True, "file_unreadable"


class _HloModuleIndexer:
  """Helper class to build HLO module indices directly from HloModuleProto."""

  def __init__(self, module_proto: Any):
    self._module_proto = module_proto
    self._computations: dict[int, Any] = {}
    self._instructions: dict[int, Any] = {}
    self._instruction_to_computation: dict[int, int] = {}
    self._fusion_callers: dict[int, tuple[int, Any]] = {}
    self._consumers: dict[int, list[Any]] = collections.defaultdict(list)
    self._instructions_by_name: dict[str, Any] = {}
    self._stack_frames: dict[int, Any] = {}

    if hasattr(module_proto, "stack_frame"):
      for frame in module_proto.stack_frame:
        self._stack_frames[frame.id] = frame

    if not hasattr(module_proto, "computations"):
      return

    for comp in module_proto.computations:
      self._computations[comp.id] = comp
      for instr in comp.instructions:
        self._instructions[instr.id] = instr
        self._instruction_to_computation[instr.id] = comp.id
        self._instructions_by_name[instr.name] = instr

        for op_id in instr.operand_ids:
          self._consumers[op_id].append(instr)

        if instr.opcode.lower() == "fusion":
          for called_comp_id in instr.called_computation_ids:
            self._fusion_callers[called_comp_id] = (comp.id, instr)

  def is_fused_instruction(self, instr_id: int) -> bool:
    """Returns True if the instruction is inside a non-root fusion computation."""
    instr = self._instructions.get(instr_id)
    if not instr:
      return False
    comp_id = self._instruction_to_computation.get(instr_id)
    if comp_id is not None:
      if comp_id in self._fusion_callers:
        return True
      comp = self._computations.get(comp_id)
      if comp and any(
          k in comp.name.lower() for k in ["fused_computation", "fusion"]
      ):
        return True
    return False

  def get_consumer_instructions(self, instr_id: int) -> list[Any]:
    """Returns the list of consumer instructions for the given instruction ID."""
    return self._consumers.get(instr_id, [])

  def get_instruction_by_name(self, name: str) -> Any | None:
    """Returns the instruction with the given name, or None."""
    return self._instructions_by_name.get(name)

  def get_computation(self, comp_id: int) -> Any | None:
    """Returns the computation with the given ID, or None."""
    return self._computations.get(comp_id)

  def get_operand_instruction(self, instr: Any, index: int = 0) -> Any | None:
    """Returns the operand instruction at the specified index, or None."""
    operand_ids = getattr(instr, "operand_ids", [])
    if operand_ids and index < len(operand_ids):
      return self._instructions.get(operand_ids[index])
    return None

  def resolve_source(self, instr: Any) -> tuple[str, int, str]:
    """Resolves the first-party source file and line for an instruction.

    Args:
      instr: The HLO instruction to resolve source for.

    Returns:
      (source_file, source_line, attribution) tuple.
    """
    metadata = getattr(instr, "metadata", None)
    if metadata is None:
      return "", 0, "no_metadata"

    frame_id = getattr(metadata, "stack_frame_id", 0)
    current_frame = self._stack_frames.get(frame_id)

    first_party_file: str = ""
    first_party_line: int = 0
    visited: set[int] = set()

    while current_frame is not None and current_frame.id not in visited:
      visited.add(current_frame.id)
      fname = str(getattr(current_frame, "file_name", "") or "")
      line = int(getattr(current_frame, "line", 0) or 0)
      if fname and not _FRAMEWORK_PATH_RE.search(fname):
        first_party_file = fname
        first_party_line = line
        break
      parent_id = getattr(current_frame, "parent_frame_id", 0)
      current_frame = self._stack_frames.get(parent_id)

    if first_party_file:
      return first_party_file, first_party_line, "first_party_resolved"

    raw_file = str(getattr(metadata, "source_file", "") or "")
    raw_line = int(getattr(metadata, "source_line", 0) or 0)
    if raw_file:
      if _FRAMEWORK_PATH_RE.search(raw_file):
        return raw_file, raw_line, "framework_scapegoat"
      return raw_file, raw_line, "first_party_direct"

    return "", 0, "no_source"


def _dedup_and_sort_ops(
    ops: list[dict[str, Any]],
) -> list[dict[str, Any]]:
  """Groups duplicate ops at the same source location and sorts by self-time."""
  grouped: dict[tuple[str, int, str], dict[str, Any]] = {}
  ungrouped: list[dict[str, Any]] = []

  for op in ops:
    evidence = op.get("evidence", {})
    src_file = evidence.get("source_file") or op.get("source_file", "")
    src_line = evidence.get("source_line") or op.get("source_line", 0)
    res_shape = str(evidence.get("result_shape", ""))

    if src_file and src_line > 0:
      key = (src_file, src_line, res_shape)
      if key in grouped:
        existing = grouped[key]
        existing["occurrences"] = existing.get("occurrences", 1) + op.get(
            "occurrences", 1
        )
        existing["total_self_time_ms"] = existing.get(
            "total_self_time_ms", 0.0
        ) + op.get("total_self_time_ms", 0.0)
        existing["bytes_accessed"] = existing.get("bytes_accessed", 0) + op.get(
            "bytes_accessed", 0
        )
        existing_names = existing.setdefault(
            "grouped_op_names", [existing.get("name")]
        )
        existing_names.append(op.get("name"))
      else:
        op_copy = dict(op)
        op_copy["grouped_op_names"] = [op.get("name")]
        grouped[key] = op_copy
    else:
      ungrouped.append(op)

  combined = list(grouped.values()) + ungrouped
  combined.sort(
      key=lambda x: (
          x.get("total_self_time_ms", 0.0),
          x.get("bytes_accessed", 0),
      ),
      reverse=True,
  )
  return combined


def _fusion_has_foldable_compute(indexer: Any, fusion_instr: Any) -> bool:
  """Returns True if a fusion body contains a non-collective compute op."""
  for comp_id in getattr(fusion_instr, "called_computation_ids", []):
    comp = indexer.get_computation(comp_id)
    if not comp:
      continue
    for fused_instr in comp.instructions:
      f_op = fused_instr.opcode.lower()
      if f_op in ("dot", "convolution", "einsum"):
        return True
      if f_op == "custom-call":
        f_name = (
            getattr(fused_instr, "custom_call_target", "")
            or getattr(fused_instr, "name", "")
        ).lower()
        if not any(k in f_name for k in _COLLECTIVE_KEYWORDS):
          return True
  return False


def _is_free_view(indexer: Any, instr: Any) -> bool:
  """Returns True if instr is a zero-cost view (bitcast or free reshape/copy)."""
  opcode = instr.opcode.lower()
  if opcode == "bitcast":
    return True
  if opcode in _VIEW_OPCODES:
    operand = indexer.get_operand_instruction(instr, 0)
    return not _is_real_relayout(instr, operand)[0]
  return False


def _find_compute_consumer(
    indexer: Any,
    instr: Any,
    depth: int = 0,
    visited: set[int] | None = None,
) -> tuple[str | None, str | None, str | None]:
  """Finds a foldable-compute consumer, seeing through zero-cost view ops.

  Direct ``dot``/``convolution``/``einsum``/non-collective ``custom-call``/
  ``fusion(compute)`` consumers are matched exactly as before. Additionally, the
  search transitively follows zero-cost view ops (``_is_free_view``) so that a
  real relayout reaching compute via a bitcast-reshape chain is still detected
  (attributed to the originating relayout op).

  Args:
    indexer: The HLO module indexer.
    instr: The instruction whose (transitive) consumers to search.
    depth: Current recursion depth through view ops.
    visited: Set of already-visited instruction ids (cycle guard).

  Returns:
    (compute_target, custom_call_target, confidence); compute_target is None
    when no foldable-compute consumer is reachable through zero-cost views.
  """
  if visited is None:
    visited = set()
  if depth > _MAX_VIEW_CHAIN_DEPTH:
    return None, None, None

  for consumer in indexer.get_consumer_instructions(instr.id):
    if consumer.id in visited:
      continue
    visited.add(consumer.id)
    consumer_opcode = consumer.opcode.lower()

    if consumer_opcode in ("dot", "convolution", "einsum"):
      return consumer_opcode, None, "high"

    if consumer_opcode == "custom-call":
      target_name = (
          getattr(consumer, "custom_call_target", "")
          or getattr(consumer, "name", "")
      ).lower()
      if any(k in target_name for k in _COLLECTIVE_KEYWORDS):
        continue
      return (
          "custom-call",
          getattr(consumer, "custom_call_target", "") or consumer.name,
          "high",
      )

    if consumer_opcode == "fusion":
      if _fusion_has_foldable_compute(indexer, consumer):
        return "fusion(compute)", None, "high"
      continue

    # See through a zero-cost view op to its own consumers.
    if _is_free_view(indexer, consumer):
      result = _find_compute_consumer(indexer, consumer, depth + 1, visited)
      if result[0] is not None:
        return result

  return None, None, None


def _analyze_candidates_from_protos(
    candidates: list[dict[str, Any]],
    debug_info: Any,
) -> list[dict[str, Any]]:
  """Analyzes candidates directly using HloModuleProto indices."""
  inefficient_ops = []
  if not hasattr(debug_info, "hlo_proto") or not debug_info.hlo_proto:
    return inefficient_ops

  module_indexers = []
  for i, proto in enumerate(debug_info.hlo_proto):
    if not proto or not hasattr(proto, "hlo_module"):
      continue
    module_proto = proto.hlo_module
    if not module_proto:
      continue
    name = getattr(module_proto, "name", "")
    program_id = (
        debug_info.program_id[i]
        if hasattr(debug_info, "program_id") and i < len(debug_info.program_id)
        else None
    )
    full_name = f"{name}({program_id})" if program_id else name
    indexer = _HloModuleIndexer(module_proto)
    module_indexers.append((full_name, name, indexer))

  if not module_indexers:
    return inefficient_ops

  for candidate in candidates:
    if candidate.get("flops", 0) > 0:
      continue

    raw_name = candidate.get("name", "")
    instr_name_part = raw_name.split("/")[-1].split(" and its ")[0]
    instr_name = instr_name_part.replace("%", "").strip()

    parts = raw_name.split("/")
    mod_name_str = parts[1] if len(parts) > 1 else None

    target_indexer = None
    target_instr = None

    # 1. Try to find in preferred module
    if mod_name_str:
      for full_name, name, indexer in module_indexers:
        if (
            (full_name and mod_name_str == full_name)
            or (name and mod_name_str == name)
            or (full_name and mod_name_str in full_name)
            or (name and name in mod_name_str)
        ):
          instr = indexer.get_instruction_by_name(instr_name)
          if instr is not None:
            target_indexer = indexer
            target_instr = instr
            break

    # 2. Fallback: search across all modules
    if target_instr is None:
      for _, _, indexer in module_indexers:
        instr = indexer.get_instruction_by_name(instr_name)
        if instr is not None:
          target_indexer = indexer
          target_instr = instr
          break

    if target_instr is None or target_indexer is None:
      continue

    # Verify opcode
    opcode = target_instr.opcode.lower()
    if opcode not in _FORMATTING_OPCODES:
      continue

    # Check if standalone (not inside a fusion)
    if target_indexer.is_fused_instruction(target_instr.id):
      continue

    # P6: Physical Relayout Gate
    operand_instr = target_indexer.get_operand_instruction(target_instr, 0)
    is_real, relayout_reason = _is_real_relayout(target_instr, operand_instr)
    if not is_real:
      continue

    # P7: Compute Consumer Gate (dot, conv, einsum, non-collective custom-call,
    # fusion peek). Sees through intervening zero-cost "view" ops (bitcast /
    # free reshape / identity copy) so that a real relayout reaching compute via
    # a bitcast-reshape chain is still flagged (attributed to the relayout op).
    compute_target, custom_call_target, confidence = _find_compute_consumer(
        target_indexer, target_instr
    )
    feeds_compute = compute_target is not None
    if not feeds_compute:
      continue
    if confidence is None:
      confidence = "medium"

    # P9: First-Party Source Attribution
    src_file, src_line, attribution = target_indexer.resolve_source(
        target_instr
    )
    if not src_file:
      src_file = candidate.get("source_file", "")
      src_line = candidate.get("source_line", 0)
      if src_file:
        attribution = (
            "framework_scapegoat"
            if _FRAMEWORK_PATH_RE.search(src_file)
            else "candidate_direct"
        )
      else:
        attribution = "unknown"

    if attribution == "framework_scapegoat":
      continue

    # P8: Source Line Verification Gate
    source_verified, verify_reason = _classify_source_line(src_file, src_line)
    if not source_verified:
      continue

    candidate["hbm_materialization_overhead"] = True
    candidate["downstream_compute"] = compute_target
    candidate["recommendation"] = (
        f"Standalone formatting op '{instr_name}' feeds into compute op"
        f" '{compute_target}'. This forces materialization of an explicit"
        " intermediate tensor in HBM. Consider folding it directly into the"
        " compute op (e.g., using einsum)."
    )

    instr_shape = getattr(target_instr, "shape", None)
    op_shape = getattr(operand_instr, "shape", None) if operand_instr else None

    candidate["evidence"] = {
        "opcode": opcode,
        "operand_shape": _get_shape_dimensions(op_shape),
        "operand_layout": _get_shape_layout(op_shape),
        "result_shape": _get_shape_dimensions(instr_shape),
        "result_layout": _get_shape_layout(instr_shape),
        "relayout_verified": relayout_reason != "unverified_layout",
        "downstream_compute": compute_target,
        "custom_call_target": custom_call_target,
        "source_file": src_file,
        "source_line": src_line,
        "source_verified": (
            source_verified and verify_reason == "source_verified"
        ),
        "attribution": attribution,
        "confidence": confidence,
        "total_self_time_ms": candidate.get(
            "total_self_time_ms", candidate.get("self_time_ms", 0.0)
        ),
        "bytes_accessed": candidate.get("bytes_accessed", 0),
    }
    inefficient_ops.append(candidate)

  return _dedup_and_sort_ops(inefficient_ops)


def _analyze_candidates_from_text_neighborhood(
    session_id: str,
    candidates: list[dict[str, Any]],
    get_hlo_neighborhood_fn: Callable[..., str],
) -> list[dict[str, Any]]:
  """Fallback analyzer using text-based get_hlo_neighborhood_fn."""
  inefficient_ops = []
  modules_str_cache = None
  module_names_cache = []

  for candidate in candidates:
    if candidate.get("flops", 0) > 0:
      continue

    raw_name = candidate.get("name", "")
    instr_name_part = raw_name.split("/")[-1].split(" and its ")[0]
    instr_name = instr_name_part.replace("%", "").strip()

    parts = raw_name.split("/")
    mod_name_str = parts[1] if len(parts) > 1 else None

    neighborhood_str = ""
    found_neighborhood = False
    if mod_name_str:
      neighborhood_str = get_hlo_neighborhood_fn(
          session_id,
          instruction_name=instr_name,
          radius=2,
          module_name=mod_name_str,
      )
      if "not found" not in neighborhood_str.lower():
        found_neighborhood = True

    if not found_neighborhood:
      if modules_str_cache is None:
        modules_str_cache = hlo_tools.list_hlo_modules(session_id)
        module_names_cache = re.findall(
            r"^\d+\.\s+([a-zA-Z0-9._-]+)\(", modules_str_cache, re.MULTILINE
        )

      for mod_name in module_names_cache:
        candidate_neighborhood = get_hlo_neighborhood_fn(
            session_id,
            instruction_name=instr_name,
            radius=2,
            module_name=mod_name,
        )
        if "not found" not in candidate_neighborhood.lower():
          neighborhood_str = candidate_neighborhood
          found_neighborhood = True
          break

      if not found_neighborhood:
        neighborhood_str = (
            f"Instruction '{instr_name}' not found in any module."
        )

    if "not found" in neighborhood_str.lower():
      continue

    is_standalone = False
    feeds_compute = False
    compute_target = None
    custom_call_target = None
    confidence = "medium"

    for line in neighborhood_str.splitlines():
      line_lower = line.lower()
      if f"%{instr_name} = " in line_lower:
        try:
          rhs = line_lower.split(f"%{instr_name} = ")[1]
          opcode = rhs.split("(")[0].split()[-1]
          if opcode not in _FORMATTING_KEYWORDS:
            break
        except IndexError:
          pass

        comp_context = line_lower.split(f"%{instr_name} = ")[0]
        if "[" in comp_context and "]" in comp_context:
          comp_name = comp_context.split("[")[-1].split("]")[0].strip()
          if not any(k in comp_name for k in ["fused_computation", "fusion"]):
            is_standalone = True
        else:
          if not any(k in line_lower for k in ["fused_computation", "fusion"]):
            is_standalone = True
      elif "[dist=1]" in line_lower and f"%{instr_name}" in line_lower:
        if any(c_kw in line_lower for c_kw in _COLLECTIVE_KEYWORDS):
          continue

        found_op = next(
            (
                op
                for op in _COMPUTE_OPCODES
                if re.search(rf"\b{re.escape(op)}\(", line_lower)
            ),
            None,
        )
        if found_op:
          feeds_compute = True
          compute_target = found_op
          confidence = "high"
          break

    if is_standalone and feeds_compute:
      src_file = candidate.get("source_file", "")
      src_line = candidate.get("source_line", 0)
      if src_file and _FRAMEWORK_PATH_RE.search(src_file):
        continue

      source_verified, verify_reason = _classify_source_line(src_file, src_line)
      if not source_verified:
        continue

      candidate["hbm_materialization_overhead"] = True
      candidate["downstream_compute"] = compute_target
      candidate["recommendation"] = (
          f"Standalone formatting op '{instr_name}' feeds into compute op"
          f" '{compute_target}'. This forces materialization of an explicit"
          " intermediate tensor in HBM. Consider folding it directly into the"
          " compute op (e.g., using einsum)."
      )
      candidate["evidence"] = {
          "opcode": candidate.get("category", "formatting"),
          "operand_shape": [],
          "operand_layout": None,
          "result_shape": [],
          "result_layout": None,
          "relayout_verified": False,
          "downstream_compute": compute_target,
          "custom_call_target": custom_call_target,
          "source_file": src_file,
          "source_line": src_line,
          "source_verified": (
              source_verified and verify_reason == "source_verified"
          ),
          "attribution": "candidate_direct" if src_file else "unknown",
          "confidence": confidence,
          "total_self_time_ms": candidate.get(
              "total_self_time_ms", candidate.get("self_time_ms", 0.0)
          ),
          "bytes_accessed": candidate.get("bytes_accessed", 0),
      }
      inefficient_ops.append(candidate)

  return _dedup_and_sort_ops(inefficient_ops)


def detect_unfused_reshapes(
    session_id: str,
    get_top_hlo_ops_fn: Callable[..., str] = (
        get_top_hlo_ops_tool.get_top_hlo_ops
    ),
    get_hlo_neighborhood_fn: Callable[..., str] | None = (
        hlo_tools.get_hlo_neighborhood
    ),
    fetch_debug_info_fn: Callable[..., Any] | None = None,
    limit: int = 75,
    min_bytes_accessed: int = MIN_BYTES_ACCESSED,
    min_self_time_ms: float = DEFAULT_MIN_SELF_TIME_MS,
    bypass_cache: bool = False,
) -> str:
  """Detects unfused reshape/transpose/copy HLO ops causing an HBM materialization overhead.

  Args:
      session_id: The unique XProf session ID.
      get_top_hlo_ops_fn: Function to retrieve top HLO operations.
      get_hlo_neighborhood_fn: Optional fallback function to retrieve HLO
        neighborhood text.
      fetch_debug_info_fn: Optional function to fetch HLO debug info proto.
      limit: How many top operations to analyze.
      min_bytes_accessed: Minimum bytes accessed to consider an operation.
      min_self_time_ms: Minimum self time in milliseconds (denoise threshold).
      bypass_cache: Whether to bypass cache and recompute metrics.

  Returns:
      A JSON string summarizing the findings.

  Raises:
      FileNotFoundError: If top HLO ops cannot be fetched for the session.
      ValueError: If backend data is malformed.
      RuntimeError: If detection fails.
  """
  try:
    total_start_time = time.perf_counter()
    fetch_time_start = time.perf_counter()
    try:
      top_ops_json = get_top_hlo_ops_fn(
          session_id, limit=limit, bypass_cache=bypass_cache
      )
    except TypeError:
      top_ops_json = get_top_hlo_ops_fn(session_id, limit=limit)
    fetch_time_end = time.perf_counter()
    if not top_ops_json:
      logging.info(
          "Unfused reshapes detection metrics - "
          "Session ID: %s, "
          "Fetch time: %.2fs, "
          "Load time: 0.00s, "
          "Total wall clock time: %.2fs, "
          "Core logic processing time: 0.00s, "
          "Dict build time: 0.00s for 0 candidates ",
          session_id,
          fetch_time_end - fetch_time_start,
          time.perf_counter() - total_start_time,
      )
      raise FileNotFoundError(
          f"Could not fetch top HLO ops for session {session_id}."
      )

    load_start = time.perf_counter()
    ops_data = json.loads(top_ops_json)
    top_by_bytes = ops_data.get("top_by_bytes_accessed", [])
    load_time_s = time.perf_counter() - load_start

    dict_build_start = time.perf_counter()
    candidates = []

    for op in top_by_bytes:
      if op.get("flops", 0) > 0:
        continue

      category = (op.get("category") or "").lower()
      name = (op.get("name") or "").lower()
      is_formatting_op = any(
          cat in category for cat in _FORMATTING_CATEGORIES
      ) or any(k in name for k in _FORMATTING_KEYWORDS)

      bytes_accessed = op.get("bytes_accessed") or 0
      if not (is_formatting_op and bytes_accessed >= min_bytes_accessed):
        continue

      self_time = op.get("total_self_time_ms", op.get("self_time_ms", 0.0))
      has_self_time = "total_self_time_ms" in op or "self_time_ms" in op
      if has_self_time and self_time < min_self_time_ms:
        continue

      candidates.append(op)

    dict_build_time_total = time.perf_counter() - dict_build_start

    if not candidates:
      return json.dumps(
          {
              "bottlenecks_found": False,
              "message": "No formatting operations found.",
              "inefficient_ops": [],
          },
          indent=2,
      )

    core_logic_start_time = time.perf_counter()
    debug_info = None
    if fetch_debug_info_fn is not None:
      try:
        debug_info = fetch_debug_info_fn(session_id)
      except Exception as e:  # pylint: disable=broad-exception-caught
        logging.warning(
            "Error fetching debug info via fetch_debug_info_fn for Session ID:"
            " %s. %s",
            session_id,
            e,
        )
    else:
      try:
        debug_info = hlo_tools._fetch_debug_info(session_id)  # pylint: disable=protected-access
      except Exception as e:  # pylint: disable=broad-exception-caught
        logging.warning(
            "Error fetching debug info for Session ID: %s. Falling back to text"
            " neighborhood analysis. %s",
            session_id,
            e,
        )

    if debug_info and hasattr(debug_info, "hlo_proto") and debug_info.hlo_proto:
      inefficient_ops = _analyze_candidates_from_protos(candidates, debug_info)
    elif get_hlo_neighborhood_fn is not None:
      inefficient_ops = _analyze_candidates_from_text_neighborhood(
          session_id, candidates, get_hlo_neighborhood_fn
      )
    else:
      inefficient_ops = []

    bottleneck_msg = (
        f"Detected {len(inefficient_ops)} standalone formatting operations"
        " causing HBM materialization overhead. See individual op"
        " recommendations for details."
    )
    safe_msg = "No unfused reshape bottlenecks detected."
    message = bottleneck_msg if inefficient_ops else safe_msg

    core_logic_end_time = time.perf_counter()
    core_logic_time_s = core_logic_end_time - core_logic_start_time
    total_end_time = time.perf_counter()
    total_time_s = total_end_time - total_start_time

    logging.info(
        "Unfused reshapes detection metrics - Session ID: %s, "
        "Total wall clock time: %.2fs, "
        "Fetch time: %.2fs, "
        "Load time: %.2fs, "
        "Core logic processing time: %.2fs, "
        "Dict build time: %.2fs for %d candidates ",
        session_id,
        total_time_s,
        fetch_time_end - fetch_time_start,
        load_time_s,
        core_logic_time_s,
        dict_build_time_total,
        len(candidates),
    )

    return json.dumps(
        {
            "bottlenecks_found": len(inefficient_ops) > 0,
            "inefficient_ops": inefficient_ops,
            "message": message,
        },
        indent=2,
    )

  except (FileNotFoundError, ValueError):
    raise
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.exception(
        "Error detecting unfused reshapes for Session ID: %s", session_id
    )
    raise RuntimeError(
        f"Error detecting unfused reshapes for session {session_id}: {e}"
    ) from e
