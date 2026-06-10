"""MCP tool to detect unfused reshape operations causing HBM overhead."""

from collections.abc import Callable
import json
import logging

from xprof.cli.internal.oss import hlo_tools
from xprof.cli.tools import get_top_hlo_ops_tool


def detect_unfused_reshapes(
    session_id: str,
    get_top_hlo_ops_fn: Callable[
        ..., str
    ] = get_top_hlo_ops_tool.get_top_hlo_ops,
    get_hlo_neighborhood_fn: Callable[
        ..., str
    ] = hlo_tools.get_hlo_neighborhood,
    limit: int = 50,
) -> str:
  """Detects unfused reshape/transpose/copy HLO ops causing an HBM materialization overhead.

  Args:
      session_id: The unique XProf session ID.
      get_top_hlo_ops_fn: Function to retrieve top HLO operations.
      get_hlo_neighborhood_fn: Function to retrieve HLO neighborhood.
      limit: How many top operations to analyze.

  Returns:
      A JSON string summarizing the findings.
  """
  try:
    # 1. Get candidate operations based on bytes_accessed
    # get_top_hlo_ops returns a JSON-formatted string.
    top_ops_json = get_top_hlo_ops_fn(session_id, limit=limit)
    if not top_ops_json:
      return json.dumps({"error": "Could not fetch top HLO ops."})

    ops_data = json.loads(top_ops_json)
    top_by_bytes = ops_data.get("top_by_bytes_accessed", [])

    # 2. Filter candidates
    formatting_categories = {"data formatting", "copy", "reshape", "transpose"}
    candidates = []

    for op in top_by_bytes:
      category = op.get("category", "").lower()
      name = op.get("name", "").lower()
      # Determine if it qualifies as a formatting candidate
      is_formatting_op = any(
          cat in category for cat in formatting_categories
      ) or any(k in name for k in ["reshape", "transpose", "copy"])

      if is_formatting_op:
        candidates.append(op)

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
    inefficient_ops = []

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
      if len(parts) > 1 and "jit_" in parts[1]:
        mod_name_str = parts[1]

      # Fetch neighborhood via string parsing to simplify traversing
      neighborhood_str = get_hlo_neighborhood_fn(
          session_id,
          instruction_name=instr_name,
          radius=2,
          module_name=mod_name_str,
      )

      if "not found" in neighborhood_str.lower():
        continue

      is_standalone = False
      feeds_compute = False
      compute_target = None

      for line in neighborhood_str.splitlines():
        line_lower = line.lower()
        if f"%{instr_name} = " in line_lower:
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
          compute_ops = ["dot", "einsum", "custom-call", "fusion"]
          found_op = next((op for op in compute_ops if op in line_lower), None)
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

    return json.dumps(
        {
            "bottlenecks_found": len(inefficient_ops) > 0,
            "inefficient_ops": inefficient_ops,
            "message": message,
        },
        indent=2,
    )

  except json.JSONDecodeError as e:
    logging.exception("Malformed JSON from top HLO ops")
    return json.dumps({"error": f"Malformed JSON data from backend: {e}"})
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.exception("Error detecting unfused reshapes")
    return json.dumps({"error": f"Internal error during detection: {e}"})
