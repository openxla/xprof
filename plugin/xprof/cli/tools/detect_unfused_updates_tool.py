"""MCP tool to detect unfused sequences of small HLO operations."""

import collections
from collections.abc import Callable
import json
import logging
import time
from typing import Any, Optional

from xprof.cli.internal.oss import hlo_tools
from xprof.cli.tools import get_top_hlo_ops_tool

_ELEMENTWISE_UNARY = {
    "abs",
    "ceil",
    "convert",
    "cosine",
    "cbrt",
    "exponential",
    "floor",
    "is-finite",
    "log",
    "negate",
    "not",
    "round",
    "rsqrt",
    "sign",
    "sine",
    "sqrt",
    "tanh",
}

_ELEMENTWISE_BINARY = {
    "add",
    "subtract",
    "multiply",
    "divide",
    "power",
    "minimum",
    "maximum",
    "compare",
    "and",
    "or",
    "xor",
    "clamp",
    "select",
    "shift-left",
    "shift-right-arithmetic",
    "shift-right-logical",
}

_SHAPE_INDEX_OPS = {
    "slice",
    "dynamic-slice",
    "dynamic-update-slice",
    "concatenate",
    "pad",
    "reverse",
    "transpose",
    "copy",
    "reduce",
    "fusion",
}

SMALL_OPCODES = _ELEMENTWISE_UNARY | _ELEMENTWISE_BINARY | _SHAPE_INDEX_OPS

ZERO_COST_TIME_THRESHOLD_MS = 0.001


_PRIMITIVE_TYPE_NAMES = {
    1: "pred",
    2: "s8",
    3: "s16",
    4: "s32",
    5: "s64",
    6: "u8",
    7: "u16",
    8: "u32",
    9: "u64",
    10: "f16",
    11: "f32",
    12: "f64",
    13: "tuple",
    14: "opaque",
    15: "token",
    16: "bf16",
    17: "c64",
    18: "c128",
    19: "f8e5m2",
    20: "f8e4m3fn",
    21: "s4",
    22: "u4",
    23: "f8e4m3b11fnuz",
    24: "f8e5m2fnuz",
    25: "f8e4m3fnuz",
    26: "s2",
    27: "u2",
    28: "f8e4m3",
    29: "f8e3m4",
    30: "s1",
    31: "u1",
    32: "bf8",
    33: "f4e2m1fn",
    34: "f8e8m0fnu",
}


def format_shape_proto(shape: Any) -> Optional[str]:
  """Formats an xla.ShapeProto into a concise string representation."""
  if not shape or not hasattr(shape, "element_type"):
    return None
  element_type = getattr(shape, "element_type", 0)
  if element_type == 13 or (
      hasattr(shape, "tuple_shapes") and shape.tuple_shapes
  ):
    elements = [format_shape_proto(s) or "unknown" for s in shape.tuple_shapes]
    return f"({', '.join(elements)})"
  elif element_type == 15:
    return "token[]"
  elif element_type == 14:
    return "opaque[]"
  elif element_type == 0 and not getattr(shape, "dimensions", None):
    return None
  else:
    if isinstance(element_type, int):
      dtype_str = _PRIMITIVE_TYPE_NAMES.get(
          element_type, f"primitive_{element_type}"
      )
    else:
      dtype_str = str(element_type).lower()
    dims = [str(d) for d in getattr(shape, "dimensions", [])]
    return f"{dtype_str}[{','.join(dims)}]"


class _HloModuleIndexer:
  """Helper class to build HLO module indices directly from HloModuleProto."""

  def __init__(self, module_proto: Any):
    self.module_proto = module_proto
    self.computations: dict[int, Any] = {}
    self.instructions: dict[int, Any] = {}
    self.instruction_to_computation: dict[int, int] = {}
    self.fusion_callers: dict[int, tuple[int, Any]] = {}
    self.consumers: dict[int, list[Any]] = collections.defaultdict(list)
    self.computation_parameters: dict[int, dict[int, Any]] = (
        collections.defaultdict(dict)
    )
    self.instructions_by_name: dict[str, Any] = {}

    if not hasattr(module_proto, "computations"):
      return

    for comp in module_proto.computations:
      self.computations[comp.id] = comp
      for instr in comp.instructions:
        self.instructions[instr.id] = instr
        self.instruction_to_computation[instr.id] = comp.id
        self.instructions_by_name[instr.name] = instr

        if instr.opcode.lower() == "parameter":
          self.computation_parameters[comp.id][instr.parameter_number] = instr

        for op_id in instr.operand_ids:
          self.consumers[op_id].append(instr)

        if instr.opcode.lower() == "fusion":
          for called_comp_id in instr.called_computation_ids:
            self.fusion_callers[called_comp_id] = (comp.id, instr)

  def is_fused_instruction(self, instr_id: int) -> bool:
    """Returns True if the instruction is inside a non-root fusion computation."""
    instr = self.instructions.get(instr_id)
    if not instr:
      return False
    comp_id = self.instruction_to_computation.get(instr_id)
    if comp_id is not None and comp_id in self.fusion_callers:
      return True
    return False

  def get_operand_instructions(self, instr_id: int) -> list[Any]:
    instr = self.instructions.get(instr_id)
    if not instr:
      return []
    return [
        self.instructions[op_id]
        for op_id in instr.operand_ids
        if op_id in self.instructions
    ]

  def get_consumer_instructions(self, instr_id: int) -> list[Any]:
    return self.consumers.get(instr_id, [])


def extract_dag_subgraphs(
    nodes: list[Any], directed_edges: set[tuple[Any, Any]]
) -> list[dict[str, Any]]:
  """Extracts topological chains and unfused clusters from a DAG."""
  node_set = set(nodes)
  undirected_adj = collections.defaultdict(set)
  directed_adj = collections.defaultdict(list)
  in_degree = collections.Counter()
  out_degree = collections.Counter()

  for u, v in directed_edges:
    if u in node_set and v in node_set:
      undirected_adj[u].add(v)
      undirected_adj[v].add(u)
      directed_adj[u].append(v)
      in_degree[v] += 1
      out_degree[u] += 1

  visited = set()
  subgraphs = []

  for start_node in nodes:
    if start_node in visited or start_node not in undirected_adj:
      continue

    component = []
    queue = collections.deque([start_node])
    visited.add(start_node)

    while queue:
      curr = queue.popleft()
      component.append(curr)
      for neighbor in undirected_adj[curr]:
        if neighbor not in visited:
          visited.add(neighbor)
          queue.append(neighbor)

    if len(component) < 2:
      continue

    comp_set = set(component)
    local_in_degree = {node: in_degree[node] for node in component}

    topo_queue = collections.deque(
        [node for node in component if local_in_degree[node] == 0]
    )
    topo_order = []

    while topo_queue:
      curr = topo_queue.popleft()
      topo_order.append(curr)
      for neighbor in directed_adj[curr]:
        if neighbor in comp_set:
          local_in_degree[neighbor] -= 1
          if local_in_degree[neighbor] == 0:
            topo_queue.append(neighbor)

    if len(topo_order) < len(component):
      remaining = [n for n in component if n not in topo_order]
      topo_order.extend(remaining)

    is_linear_chain = all(
        in_degree[node] <= 1 and out_degree[node] <= 1 for node in component
    )

    subgraph_type = (
        "topological_chain" if is_linear_chain else "unfused_cluster"
    )
    subgraphs.append({
        "type": subgraph_type,
        "nodes": topo_order,
    })

  return subgraphs


def _get_profile_info(
    profile_map: dict[tuple[str, str], Any], mod_name: str, instr_name: str
) -> dict[str, Any]:
  """Looks up profile info for an HLO instruction by module and instruction name.

  Args:
    profile_map: A dictionary mapping (module_name, instruction_name) tuples to
      profile metadata dictionaries.
    mod_name: The name of the HLO module containing the instruction.
    instr_name: The name of the HLO instruction to look up.

  Returns:
    A dictionary of profile metadata for the instruction, or an empty dictionary
    if no match is found.
  """
  res = profile_map.get((mod_name, instr_name))
  if res:
    return res
  mod_name_clean = mod_name.split("(")[0] if mod_name else ""
  for (m_name, i_name), info in profile_map.items():
    if i_name == instr_name and (
        m_name in mod_name or mod_name_clean in m_name or not m_name
    ):
      return info
  return {}


def analyze_hlo_module_proto(
    module_proto: Any,
    full_mod_name: str,
    profile_map: dict[Any, Any],
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
  """Analyzes an HloModuleProto using static proto traversal to detect unfused small op bottlenecks.

  Args:
      module_proto: The HLO module proto to analyze.
      full_mod_name: The complete name of the module.
      profile_map: Dictionary mapping op names to their profile metadata.

  Returns:
      A tuple of (bottlenecks, unflagged_candidates) where unflagged_candidates
      contains metadata for ops that didn't form a chain but are still isolated
      small ops.
  """
  indexer = _HloModuleIndexer(module_proto)
  candidate_instr_ids = set()
  op_metadata = {}

  # 1. Collect all unfused candidate instructions matching SMALL_OPCODES
  for instr_id, instr in indexer.instructions.items():
    opcode = instr.opcode.lower()
    if opcode not in SMALL_OPCODES:
      continue
    # Exclude metadata aliases and zero-cost ops
    if opcode in {
        "tuple",
        "get-tuple-element",
        "bitcast",
        "constant",
        "parameter",
    }:
      continue

    # Filter out cold ops that have no profile data
    prof_info = _get_profile_info(profile_map, full_mod_name, instr.name)
    if not prof_info:
      continue

    # Skip instructions inside fusion computations
    if indexer.is_fused_instruction(instr_id):
      continue

    if opcode == "fusion":
      # Only keep micro fusions (<= 4 instructions in its computation)
      is_micro = True
      for called_id in instr.called_computation_ids:
        called_comp = indexer.computations.get(called_id)
        # instruction count includes parameters, etc.
        if called_comp and len(called_comp.instructions) > 4:
          is_micro = False
          break
      if not is_micro:
        continue

    candidate_instr_ids.add(instr_id)
    if hasattr(instr, "shape") and (
        not hasattr(instr, "HasField") or instr.HasField("shape")
    ):
      shape_str = format_shape_proto(instr.shape)
    else:
      shape_str = None
    op_metadata[instr_id] = {
        "name": instr.name,
        "shape": shape_str,
        "opcode": opcode,
        "comp_id": indexer.instruction_to_computation.get(instr_id),
    }

  if not candidate_instr_ids:
    return [], {}

  # 2. Build directed dataflow edges directly from proto operand_ids
  directed_edges = set()
  zero_cost_ops = {
      "tuple",
      "get-tuple-element",
      "bitcast",
      "constant",
      "parameter",
  }

  for instr_id in candidate_instr_ids:
    queue = collections.deque(indexer.get_consumer_instructions(instr_id))
    visited = set()
    while queue:
      consumer = queue.popleft()
      if consumer.id in visited:
        continue
      visited.add(consumer.id)

      if consumer.id in candidate_instr_ids:
        directed_edges.add((instr_id, consumer.id))
      elif consumer.opcode.lower() in zero_cost_ops:
        queue.extend(indexer.get_consumer_instructions(consumer.id))

  # 3. Extract topological chains and DAG clusters
  subgraphs = extract_dag_subgraphs(list(candidate_instr_ids), directed_edges)
  bottlenecks = []
  flagged_instr_ids = set()

  for sg in subgraphs:
    sg_type = sg["type"]
    nodes = sg["nodes"]
    flagged_instr_ids.update(nodes)

    member_names = [op_metadata[nid]["name"] for nid in nodes]
    member_shapes = [
        f"{op_metadata[nid]['name']}: {op_metadata[nid]['shape']}"
        if op_metadata[nid]["shape"]
        else op_metadata[nid]["name"]
        for nid in nodes
    ]

    if sg_type == "topological_chain":
      chain_str = " -> ".join(member_names)
      rec = (
          f"Instruction '{member_names[0]}' is part of an unfused sequence of"
          f" small operations: '{chain_str}' in module '{full_mod_name}'. These"
          " operations are executed as separate, short-duration kernels,"
          " causing unnecessary memory roundtrips to HBM. Consider merging"
          " them within a single JAX function (e.g., using jax.jit or"
          " combining JAX operations) to allow XLA to fuse them into a single"
          " loop."
      )
    else:
      cluster_str = "{" + ", ".join(member_names) + "}"
      rec = (
          f"Instruction '{member_names[0]}' is part of an unfused connected"
          f" cluster (DAG) of small operations: {cluster_str} in module"
          f" '{full_mod_name}'. These operations are executed as separate,"
          " short-duration kernels, causing unnecessary memory roundtrips to"
          " HBM. Consider merging them within a single JAX function to allow"
          " XLA to fuse them."
      )

    for nid in nodes:
      instr_name = op_metadata[nid]["name"]
      prof_info = _get_profile_info(profile_map, full_mod_name, instr_name)
      bottleneck = {
          "name": f"by_program/{full_mod_name}/{instr_name}",
          "instruction": instr_name,
          "module_name": full_mod_name,
          "unfused_sequence": True,
          "unfused_type": sg_type,
          "group_members": member_names,
          "member_shapes": member_shapes,
          "recommendation": rec,
          "total_self_time_ms": prof_info.get("total_self_time_ms", 0.0),
      }
      if sg_type == "topological_chain":
        bottleneck["unfused_chain"] = chain_str
      else:
        bottleneck["unfused_cluster"] = cluster_str
      bottlenecks.append(bottleneck)

  # 4. Collect unflagged candidates for Type 2 cross-module heuristic
  unflagged_candidates = {}
  for nid in candidate_instr_ids:
    if nid not in flagged_instr_ids:
      instr_name = op_metadata[nid]["name"]
      unflagged_candidates[(instr_name, full_mod_name)] = {
          "shape": op_metadata[nid]["shape"],
          "opcode": op_metadata[nid]["opcode"],
          "comp_id": op_metadata[nid]["comp_id"],
      }

  return bottlenecks, unflagged_candidates


def find_parallel_eager_groups(
    unflagged_candidates: dict[tuple[str, str], dict[str, Any]],
    profile_map: dict[tuple[str, str], Any],
) -> list[dict[str, Any]]:
  """Groups loosely associated small operations across or within modules."""
  bottlenecks = []
  already_flagged = set()
  candidate_nodes = list(unflagged_candidates.keys())

  for i, c_node_a in enumerate(candidate_nodes):
    if c_node_a in already_flagged:
      continue

    name_a, mod_a = c_node_a
    info_a = unflagged_candidates[c_node_a]
    shape_a = info_a["shape"]
    comp_a = info_a["comp_id"]
    occ_a = _get_profile_info(profile_map, mod_a, name_a).get("occurrences", 0)

    related_nodes = [c_node_a]

    for j, c_node_b in enumerate(candidate_nodes):
      if i == j:
        continue

      name_b, mod_b = c_node_b
      info_b = unflagged_candidates[c_node_b]
      shape_b = info_b["shape"]
      comp_b = info_b["comp_id"]
      occ_b = _get_profile_info(profile_map, mod_b, name_b).get(
          "occurrences", 0
      )

      is_same_module = mod_a == mod_b

      # 20% diff in occurrences
      occ_diff = abs(occ_a - occ_b) / max(occ_a, occ_b, 1)
      is_sim_occ = occ_diff <= 0.20

      is_same_comp = is_same_module and comp_a == comp_b and comp_a is not None
      is_same_shape = shape_a == shape_b and shape_a is not None

      base_a = name_a.split(".")[0].rstrip("0123456789")
      base_b = name_b.split(".")[0].rstrip("0123456789")
      is_same_base = base_a == base_b and len(base_a) > 1

      is_structural_match = is_same_shape or is_same_base
      is_context_related = is_same_comp or not is_same_module

      if is_sim_occ and is_structural_match and is_context_related:
        related_nodes.append(c_node_b)

    if len(related_nodes) >= 2:
      already_flagged.update(related_nodes)
      group_names = [n[0] for n in related_nodes]
      group_names_str = ", ".join(group_names)

      member_shapes = [
          (n[0] + ": " + unflagged_candidates[n]["shape"])
          if unflagged_candidates[n]["shape"]
          else n[0]
          for n in related_nodes
      ]
      distinct_mods = len(set(n[1] for n in related_nodes)) > 1

      unfused_type = (
          "eager_unfused_group" if distinct_mods else "parallel_group"
      )

      for node in related_nodes:
        instr_name, mod_name = node
        prof_info = _get_profile_info(profile_map, mod_name, instr_name)

        op_info = {
            "name": f"by_program/{mod_name}/{instr_name}",
            "instruction": instr_name,
            "module_name": mod_name,
            "unfused_sequence": True,
            "unfused_type": unfused_type,
            "group_members": group_names,
            "member_shapes": member_shapes,
            "total_self_time_ms": prof_info.get("total_self_time_ms", 0.0),
        }

        if unfused_type == "eager_unfused_group":
          mods_str = ", ".join(sorted(set(str(n[1]) for n in related_nodes)))
          op_info["unfused_group_members"] = group_names
          op_info["recommendation"] = (
              f"Instruction '{instr_name}' is part of a group of heuristically"
              " associated small operations dispatched across distinct eager"
              f" modules: [{group_names_str}] in modules [{mods_str}]. Because"
              " these operations are dispatched across separate eager modules"
              " without an enclosing @jax.jit, each operation incurs roundtrip"
              " HBM memory traffic and kernel launch overhead. Wrapping the"
              " operations inside @jax.jit will allow XLA to compile them"
              " into a single loop fusion."
          )
        else:
          op_info["recommendation"] = (
              f"Instruction '{instr_name}' is part of a loosely associated"
              " group of parallel/sibling small operations:"
              f" [{group_names_str}] in module '{mod_name}'. These operations"
              " execute with similar frequencies and share computation"
              " context, but are currently unfused, causing kernel scheduling"
              " and HBM overhead. Consider grouping them within a single JAX"
              " function or using vmap/tree_map to allow XLA to fuse them."
          )
        bottlenecks.append(op_info)

  return bottlenecks


def detect_unfused_updates(
    session_id: str,
    get_top_hlo_ops_fn: Callable[
        ..., str
    ] = get_top_hlo_ops_tool.get_top_hlo_ops,
    fetch_debug_info_fn: Callable[..., Any] | None = None,
    limit: int = 50,
) -> str:
  """Detects small, unfused HLO operations across topological chains and parallel groups.

  Args:
      session_id: The unique XProf session ID.
      get_top_hlo_ops_fn: Function to retrieve top HLO operations.
      fetch_debug_info_fn: Optional function to fetch HLO debug info proto.
      limit: How many top operations to analyze.

  Returns:
      A JSON string summarizing the findings.
  """
  total_start_time = time.time()
  if fetch_debug_info_fn is None:
    fetch_debug_info_fn = hlo_tools._fetch_debug_info  # pylint: disable=protected-access

  try:
    # 1. Get candidate operations based on profile data
    fetch_top_ops_start_time = time.time()
    top_ops_json = get_top_hlo_ops_fn(session_id, limit=limit)
    fetch_top_ops_end_time = time.time()

    if not top_ops_json:
      return json.dumps({"error": "Could not fetch top HLO ops."})

    ops_data = json.loads(top_ops_json)
    if not isinstance(ops_data, dict):
      return json.dumps({
          "error": (
              "Unexpected JSON data format from backend. Expected a dictionary."
          )
      })
    if "error" in ops_data:
      err_msg = ops_data.get("error", "")
      if "Unexpected data type" in err_msg or "NoneType" in err_msg:
        return json.dumps(
            {
                "error": (
                    "Could not analyze session: the profiler backend returned"
                    " no HLO op profile data (the underlying get_top_hlo_ops"
                    " request failed or timed out). Verify the session ID and"
                    " that an op profile is available for this session."
                ),
                "original_error": err_msg,
            },
            indent=2,
        )
      return top_ops_json

    core_logic_part1_start_time = time.time()
    # Merge all top lists to get a broad set of candidates
    all_top_ops = []
    seen_names = set()
    for list_name in ["top_by_time", "top_by_flops", "top_by_bytes_accessed"]:
      for op in ops_data.get(list_name, []):
        name = op.get("name")
        if name and name not in seen_names:
          seen_names.add(name)
          all_top_ops.append(op)

    profile_map = {}
    for op in all_top_ops:
      self_time = op.get("total_self_time_ms", 0.0)
      if self_time < ZERO_COST_TIME_THRESHOLD_MS:
        continue

      raw_name = op.get("name", "")
      parts = raw_name.split("/")
      mod_name_part = parts[-2] if len(parts) >= 2 else ""
      # Parse clean name
      instr_name_part = parts[-1].split(" and its ")[0]
      instr_name = instr_name_part.replace("%", "").strip()
      profile_map[(mod_name_part, instr_name)] = op

    core_logic_part1_end_time = time.time()
    # 2. Direct HloProto Analysis over full session modules
    fetch_hlo_proto_start_time = time.time()
    debug_info = fetch_debug_info_fn(session_id)
    fetch_hlo_proto_end_time = time.time()

    if debug_info and hasattr(debug_info, "hlo_proto") and debug_info.hlo_proto:
      core_logic_part2_start_time = time.time()
      proto_bottlenecks = []
      global_unflagged_candidates = {}

      for i, proto in enumerate(debug_info.hlo_proto):
        module_proto = proto.hlo_module
        if not module_proto:
          continue
        name = module_proto.name
        program_id = (
            debug_info.program_id[i] if i < len(debug_info.program_id) else None
        )
        mod_name = f"{name}({program_id})" if program_id else name
        mod_results, mod_unflagged = analyze_hlo_module_proto(
            module_proto, mod_name, profile_map
        )
        proto_bottlenecks.extend(mod_results)
        global_unflagged_candidates.update(mod_unflagged)

      proto_bottlenecks.extend(
          find_parallel_eager_groups(global_unflagged_candidates, profile_map)
      )

      if proto_bottlenecks:
        # Deduplicate bottlenecks by name
        dedup_bottlenecks = list(
            {op["name"]: op for op in proto_bottlenecks}.values()
        )
        result_json = json.dumps(
            {
                "bottlenecks_found": True,
                "inefficient_ops": dedup_bottlenecks,
                "message": (
                    f"Detected {len(dedup_bottlenecks)} unfused small"
                    " operations across topological chains and parallel"
                    " groups causing HBM overhead."
                ),
            },
            indent=2,
        )
      else:
        result_json = json.dumps(
            {
                "bottlenecks_found": False,
                "message": "No unfused small operations detected.",
                "inefficient_ops": [],
            },
            indent=2,
        )

      core_logic_part2_end_time = time.time()
      total_end_time = time.time()

      fetch_top_ops_time_s = fetch_top_ops_end_time - fetch_top_ops_start_time
      fetch_hlo_proto_time_s = (
          fetch_hlo_proto_end_time - fetch_hlo_proto_start_time
      )
      core_logic_time_s = (
          core_logic_part1_end_time - core_logic_part1_start_time
      ) + (core_logic_part2_end_time - core_logic_part2_start_time)
      total_time_s = total_end_time - total_start_time

      logging.info(
          "Unfused updates detection metrics - "
          "Session ID: %s, "
          "Total wall clock time: %.3fs, "
          "Fetch top ops time: %.3fs, "
          "Fetch HLO proto time: %.3fs, "
          "Core logic processing time: %.3fs",
          session_id,
          total_time_s,
          fetch_top_ops_time_s,
          fetch_hlo_proto_time_s,
          core_logic_time_s,
      )
      return result_json
    else:
      return json.dumps({
          "error": (
              "HloProto debug info unavailable for this session. "
              "Debug info is required for analysis."
          )
      })

  except json.JSONDecodeError as e:
    logging.exception("Malformed JSON from top HLO ops")
    return json.dumps({"error": f"Malformed JSON data from backend: {e}"})
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.exception("Error detecting unfused updates")
    return json.dumps({"error": f"Internal error during detection: {e}"})
