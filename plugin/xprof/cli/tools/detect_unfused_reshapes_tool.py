"""MCP tool to detect unfused reshape operations causing HBM overhead."""

from collections.abc import Callable
import json
import logging
import re
import time

from xprof.cli.internal.oss import hlo_tools
from xprof.cli.tools import get_top_hlo_ops_tool

MIN_BYTES_ACCESSED: int = 10 * 1024 * 1024  # 10 Megabytes


def detect_unfused_reshapes(
    session_id: str,
    get_top_hlo_ops_fn: Callable[..., str] = (
        get_top_hlo_ops_tool.get_top_hlo_ops
    ),
    get_hlo_neighborhood_fn: Callable[..., str] = (
        hlo_tools.get_hlo_neighborhood
    ),
    limit: int = 75,
    min_bytes_accessed: int = MIN_BYTES_ACCESSED,
) -> str:
  """Detects unfused reshape/transpose/copy HLO ops causing an HBM materialization overhead.

  Args:
      session_id: The unique XProf session ID.
      get_top_hlo_ops_fn: Function to retrieve top HLO operations.
      get_hlo_neighborhood_fn: Function to retrieve HLO neighborhood.
      limit: How many top operations to analyze.
      min_bytes_accessed: Minimum bytes accessed to consider an operation.

  Returns:
      A JSON string summarizing the findings.
  """
  try:
    total_start_time = time.perf_counter()
    # 1. Get candidate operations based on bytes_accessed
    # get_top_hlo_ops returns a JSON-formatted string.
    fetch_time_start = time.perf_counter()
    top_ops_json = get_top_hlo_ops_fn(session_id, limit=limit)
    fetch_time_end = time.perf_counter()
    if not top_ops_json:
      logging.info(
          "Unfused reshapes detection metrics - "
          "Session ID: %s, "
          "Total wall clock time: %.3fs, "
          "Fetch top ops time: %.3fs, "
          "Core logic processing time: N/A",
          session_id,
          time.perf_counter() - total_start_time,
          fetch_time_end - fetch_time_start,
      )

      return json.dumps({"error": "Could not fetch top HLO ops."})

    load_start = time.perf_counter()
    ops_data = json.loads(top_ops_json)
    top_by_bytes = ops_data.get("top_by_bytes_accessed", [])
    load_time_s = time.perf_counter() - load_start

    dict_build_start = time.perf_counter()
    formatting_categories = {"data formatting", "copy", "reshape", "transpose"}
    candidates = []

    for op in top_by_bytes:
      category = (op.get("category") or "").lower()
      name = (op.get("name") or "").lower()
      # Determine if it qualifies as a formatting candidate
      is_formatting_op = any(
          cat in category for cat in formatting_categories
      ) or any(
          k in name
          for k in [
              "reshape",
              "transpose",
              "copy",
              "broadcast",
              "slice",
              "pad",
              "convert",
          ]
      )

      bytes_accessed = op.get("bytes_accessed") or 0
      if is_formatting_op and bytes_accessed >= min_bytes_accessed:
        candidates.append(op)

    dict_build_time_total = time.perf_counter() - dict_build_start
    copy_count = len(candidates)

    if not candidates:
      return json.dumps(
          {
              "bottlenecks_found": False,
              "message": "No formatting operations found.",
              "inefficient_ops": [],
          },
          indent=2,
      )

    # 3. Analyze Graph Context
    core_logic_start_time = time.perf_counter()
    bfs_time_total = 0.0
    inefficient_ops = []

    # Cache modules list to avoid redundant RPC calls
    modules_str_cache = None
    module_names_cache = []

    for candidate in candidates:
      raw_name = candidate.get("name", "")
      # Extract instruction name (e.g., 'copy.27' from '.../copy.27')
      # Also clean up duplicate suffixes like ' and its duplicate(s)'
      instr_name_part = raw_name.split("/")[-1].split(" and its ")[0]
      # XProf sometimes prepends '%', just in case
      instr_name = instr_name_part.replace("%", "").strip()

      # Extract potential module name from the raw string
      # (e.g., 'by_program/jit_pallas_mla_rpa_v3(1247801404187289782)')
      parts = raw_name.split("/")
      mod_name_str = None
      if len(parts) > 1:
        mod_name_str = parts[1]

      neighborhood_str = ""
      found_neighborhood = False
      if mod_name_str:
        bfs_start = time.perf_counter()
        neighborhood_str = get_hlo_neighborhood_fn(
            session_id,
            instruction_name=instr_name,
            radius=2,
            module_name=mod_name_str,
        )
        bfs_time_total += time.perf_counter() - bfs_start
        if "not found" not in neighborhood_str.lower():
          found_neighborhood = True

      if not found_neighborhood:
        # Try to find the module that contains the instruction
        if modules_str_cache is None:
          modules_str_cache = hlo_tools.list_hlo_modules(session_id)
          module_names_cache = re.findall(
              r"^\d+\.\s+([a-zA-Z0-9._-]+)\(", modules_str_cache, re.MULTILINE
          )

        for mod_name in module_names_cache:
          bfs_start = time.perf_counter()
          candidate_neighborhood = get_hlo_neighborhood_fn(
              session_id,
              instruction_name=instr_name,
              radius=2,
              module_name=mod_name,
          )
          bfs_time_total += time.perf_counter() - bfs_start
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
          # Verify that the actual opcode is a formatting operation
          try:
            rhs = line_lower.split(f"%{instr_name} = ")[1]
            opcode = rhs.split("(")[0].split()[-1]
            if opcode not in [
                "reshape",
                "transpose",
                "copy",
                "broadcast",
                "slice",
                "pad",
                "convert",
            ]:
              break  # Not a formatting op; ignore this candidate
          except IndexError:
            pass

          # E.g., [dist=0] [main.16] %reshape.5 = ...
          # Extract the parent computation name from the bracketed context
          comp_context = line_lower.split(f"%{instr_name} = ")[0]
          if "[" in comp_context and "]" in comp_context:
            comp_name = comp_context.split("[")[-1].split("]")[0].strip()
            if not any(k in comp_name for k in ["fused_computation", "fusion"]):
              is_standalone = True
          else:
            if not any(
                k in line_lower for k in ["fused_computation", "fusion"]
            ):
              is_standalone = True
        elif "[dist=1]" in line_lower and f"%{instr_name}" in line_lower:
          # Check downstream consumer
          compute_ops = [
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
          ]
          found_op = next(
              (
                  op
                  for op in compute_ops
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
        "Dict build time: %.2fs, "
        "BFS time for %d copies: %.2fs",
        session_id,
        fetch_time_end - fetch_time_start,
        load_time_s,
        total_time_s,
        core_logic_time_s,
        dict_build_time_total,
        copy_count,
        bfs_time_total,
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
