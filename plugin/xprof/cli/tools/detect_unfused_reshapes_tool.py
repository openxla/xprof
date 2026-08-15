"""MCP tool to detect unfused reshape operations causing HBM overhead."""

import collections
from collections.abc import Callable
import json
import logging
import re
import time
from typing import Any

from xprof.cli.internal.oss import hlo_tools
from xprof.cli.tools import get_top_hlo_ops_tool

MIN_BYTES_ACCESSED: int = 10 * 1024 * 1024  # 10 Megabytes

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
    "einsum",
    "custom-call",
    "fusion",
    "convolution",
    "reduce",
    "reduce-window",
    "fft",
    "cholesky",
    "triangular-solve",
    "sort",
    "topk",
    "batch-norm-training",
    "batch-norm-inference",
    "batch-norm-grad",
)
_COMPUTE_OPCODES_SET = frozenset(_COMPUTE_OPCODES)


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

    # Check downstream consumers (dist=1)
    feeds_compute = False
    compute_target = None
    for consumer in target_indexer.get_consumer_instructions(target_instr.id):
      consumer_opcode = consumer.opcode.lower()
      if consumer_opcode in _COMPUTE_OPCODES_SET:
        feeds_compute = True
        compute_target = consumer_opcode
        break

    if feeds_compute:
      candidate["hbm_materialization_overhead"] = True
      candidate["downstream_compute"] = compute_target
      candidate["recommendation"] = (
          f"Standalone formatting op '{instr_name}' feeds into compute op"
          f" '{compute_target}'. This forces materialization of an explicit"
          " intermediate tensor in HBM. Consider folding it directly into the"
          " compute op (e.g., using einsum)."
      )
      inefficient_ops.append(candidate)

  return inefficient_ops


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
          break

    if is_standalone and feeds_compute:
      candidate["hbm_materialization_overhead"] = True
      candidate["downstream_compute"] = compute_target
      candidate["recommendation"] = (
          f"Standalone formatting op '{instr_name}' feeds into compute op"
          f" '{compute_target}'. This forces materialization of an explicit"
          " intermediate tensor in HBM. Consider folding it directly into the"
          " compute op (e.g., using einsum)."
      )
      inefficient_ops.append(candidate)

  return inefficient_ops


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

  Returns:
      A JSON string summarizing the findings.
  """
  try:
    total_start_time = time.perf_counter()
    # 1. Get candidate operations based on bytes_accessed
    fetch_time_start = time.perf_counter()
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
      return json.dumps({"error": "Could not fetch top HLO ops."})

    load_start = time.perf_counter()
    ops_data = json.loads(top_ops_json)
    top_by_bytes = ops_data.get("top_by_bytes_accessed", [])
    load_time_s = time.perf_counter() - load_start

    dict_build_start = time.perf_counter()
    candidates = []

    for op in top_by_bytes:
      category = (op.get("category") or "").lower()
      name = (op.get("name") or "").lower()
      # Determine if it qualifies as a formatting candidate
      is_formatting_op = any(
          cat in category for cat in _FORMATTING_CATEGORIES
      ) or any(k in name for k in _FORMATTING_KEYWORDS)

      bytes_accessed = op.get("bytes_accessed") or 0
      if is_formatting_op and bytes_accessed >= min_bytes_accessed:
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

    # 2. Analyze Graph Context
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
        "Fetch time: %.2fs, "
        "Load time: %.2fs, "
        "Total wall clock time: %.2fs, "
        "Core logic processing time: %.2fs, "
        "Dict build time: %.2fs for %d candidates ",
        session_id,
        fetch_time_end - fetch_time_start,
        load_time_s,
        total_time_s,
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

  except json.JSONDecodeError as e:
    logging.error(
        "Malformed JSON from top HLO ops for Session ID: %s. %s", session_id, e
    )
    return json.dumps({"error": f"Malformed JSON data from backend: {e}"})
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.error(
        "Error detecting unfused reshapes for Session ID: %s. %s", session_id, e
    )
    return json.dumps({"error": f"Internal error during detection: {e}"})
