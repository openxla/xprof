"""MCP tool to detect unnecessary f32 upcasts during dynamic scale calculation and quantization."""

import collections
from collections.abc import Callable, Iterator
import json
import logging
import time
from typing import Any

from xprof.cli.internal.oss import hlo_tools
from xprof.cli.tools import get_top_hlo_ops_tool

# These values correspond to the XLA PrimitiveType enum values.
_TUPLE_TYPE = 13
_F16_TYPE = 10
_F32_TYPE = 11
_BF16_TYPE = 16
_F8E5M2_TYPE = 19
_F8E4M3FN_TYPE = 20
_F8E4M3B11FNUZ_TYPE = 23
_F8E5M2FNUZ_TYPE = 24
_F8E4M3FNUZ_TYPE = 25
_F8E4M3_TYPE = 28
_F8E3M4_TYPE = 29

_LOW_PRECISION_TYPES = {_BF16_TYPE, _F16_TYPE}
_FP8_TARGET_TYPES = {
    _F8E5M2_TYPE,
    _F8E4M3FN_TYPE,
    _F8E4M3B11FNUZ_TYPE,
    _F8E5M2FNUZ_TYPE,
    _F8E4M3FNUZ_TYPE,
    _F8E4M3_TYPE,
    _F8E3M4_TYPE,
}

# Value-preserving shape-only ops, shared by the upstream and downstream walks.
_SHAPE_OPCODES = frozenset({
    "reshape",
    "transpose",
    "slice",
    "bitcast",
    "broadcast",
    "copy",
})

# Shape ops plus the structural boundary ops needed to cross fusion boundaries
# while tracing upstream to the activation source.
_TRANSPARENT_SHAPE_OPCODES = _SHAPE_OPCODES | frozenset({
    "dynamic-slice",
    "concatenate",
    "get-tuple-element",
    "tuple",
    "parameter",
    "fusion",
})

# Reduction ops that can serve as Scale Factor Node S.
_REDUCTION_OPCODES = frozenset({"reduce", "all-reduce", "reduce-scatter"})

# Shape ops the quantization branch crosses to reach the scaling op / FP8 cast.
_QUANT_PASSTHROUGH_OPCODES = _SHAPE_OPCODES | frozenset({
    "dynamic-slice",
    "concatenate",
})


# --- HLO graph indexing and precision tracing ---
class _HloModuleTracer:
  """Helper class to build HLO module indices and run precision tracing."""

  def __init__(self, module_proto: Any):
    self.module_proto = module_proto
    self.entry_computation_id = getattr(
        module_proto, "entry_computation_id", None
    )
    self.computations: dict[int, Any] = {}
    self.instructions: dict[int, Any] = {}
    self.instruction_to_computation: dict[int, int] = {}
    self.fusion_callers: dict[int, tuple[int, Any]] = {}
    self.consumers: dict[int, list[Any]] = collections.defaultdict(list)
    self.computation_parameters: dict[int, dict[int, Any]] = (
        collections.defaultdict(dict)
    )
    self.instructions_by_name: dict[str, Any] = {}

    # Per-module memoization caches for the two pure upstream/branch-2 walks.
    self._upstream_reach_cache: dict[tuple[int, int], bool] = {}
    self._branch2_cache: dict[tuple[int, int], tuple[int, int] | None] = {}

    for computation in module_proto.computations:
      self.computations[computation.id] = computation
      for instruction in computation.instructions:
        self.instructions[instruction.id] = instruction
        self.instruction_to_computation[instruction.id] = computation.id
        self.instructions_by_name[instruction.name] = instruction

        if instruction.opcode.lower() == "parameter":
          self.computation_parameters[computation.id][
              instruction.parameter_number
          ] = instruction

        for operand_id in instruction.operand_ids:
          self.consumers[operand_id].append(instruction)

        for called_comp_id in instruction.called_computation_ids:
          self.fusion_callers[called_comp_id] = (computation.id, instruction)

  def trace_upstream_activation(
      self, start_instr_id: int
  ) -> tuple[int | None, tuple[int, ...]]:
    """Traces upstream from a convert operand to the low-precision source X."""
    stack: list[tuple[int, int | None, tuple[int, ...]]] = [
        (start_instr_id, None, ())
    ]
    visited: set[tuple[int, int | None, tuple[int, ...]]] = set()

    while stack:
      curr_id, tuple_index, call_stack = stack.pop()

      state_key = (curr_id, tuple_index, call_stack)
      if state_key in visited:
        continue
      visited.add(state_key)

      instr = self.instructions.get(curr_id)
      if not instr:
        continue

      opcode = instr.opcode.lower()
      if opcode == "constant":
        continue

      comp_id = self.instruction_to_computation.get(instr.id)
      is_root_parameter = (
          opcode == "parameter"
          and not call_stack
          and (
              comp_id == self.entry_computation_id
              or comp_id not in self.fusion_callers
          )
      )
      is_non_transparent = (
          opcode not in _TRANSPARENT_SHAPE_OPCODES and opcode != "parameter"
      )

      if is_root_parameter or is_non_transparent:
        if instr.shape.element_type in _LOW_PRECISION_TYPES:
          return (instr.id, call_stack)
        continue

      handled = False

      if opcode == "get-tuple-element":
        if instr.operand_ids:
          stack.append((instr.operand_ids[0], instr.tuple_index, call_stack))
          handled = True
      elif opcode == "tuple":
        if tuple_index is not None:
          if tuple_index < len(instr.operand_ids):
            stack.append((instr.operand_ids[tuple_index], None, call_stack))
          handled = True
      elif opcode == "parameter":
        if call_stack:
          caller_id = call_stack[-1]
          caller_instr = self.instructions.get(caller_id)
          if caller_instr and instr.parameter_number < len(
              caller_instr.operand_ids
          ):
            stack.append((
                caller_instr.operand_ids[instr.parameter_number],
                tuple_index,
                call_stack[:-1],
            ))
            handled = True
        else:
          if comp_id is not None and comp_id in self.fusion_callers:
            _, caller_instr = self.fusion_callers[comp_id]
            if instr.parameter_number < len(caller_instr.operand_ids):
              stack.append((
                  caller_instr.operand_ids[instr.parameter_number],
                  tuple_index,
                  (),
              ))
              handled = True
      elif opcode == "fusion":
        if instr.called_computation_ids:
          comp = self.computations.get(instr.called_computation_ids[0])
          if comp:
            stack.append((comp.root_id, tuple_index, call_stack + (instr.id,)))
            handled = True

      if not handled:
        for op_id in reversed(instr.operand_ids):
          op_instr = self.instructions.get(op_id)
          if op_instr and op_instr.opcode.lower() == "constant":
            continue
          stack.append((op_id, tuple_index, call_stack))

    return (None, ())

  def _iter_downstream(
      self,
      start_instr_id: int,
      passthrough_opcodes: frozenset[str],
  ) -> Iterator[tuple[Any, int, int | None]]:
    """Yields downstream consumers, crossing fusion and tuple boundaries."""
    # Stack holds: (instr_id, tuple_index, prev_instr_id, call_stack)
    stack: list[tuple[int, int | None, int | None, tuple[int, ...]]] = [
        (start_instr_id, None, None, ())
    ]
    # (instruction_id, tuple_index) keeps multi-output paths distinct.
    visited: set[tuple[int, int | None]] = set()

    while stack:
      curr_id, tuple_index, prev_id, call_stack = stack.pop()

      state_key = (curr_id, tuple_index)
      if state_key in visited:
        continue
      visited.add(state_key)

      instr = self.instructions.get(curr_id)
      if not instr:
        continue

      opcode = instr.opcode.lower()
      comp_id = self.instruction_to_computation.get(curr_id)

      # Exiting a fusion: route the computation root out to its caller.
      comp = self.computations.get(comp_id) if comp_id is not None else None
      if comp and curr_id == comp.root_id:
        caller_instr = None
        new_call_stack = call_stack
        if call_stack:
          caller_instr = self.instructions.get(call_stack[-1])
          new_call_stack = call_stack[:-1]
        else:
          fusion_caller = self.fusion_callers.get(comp_id)
          if fusion_caller:
            _, caller_instr = fusion_caller
        if caller_instr is not None:
          if opcode == "tuple" and prev_id is not None:
            indices = [
                i
                for i, op_id in enumerate(instr.operand_ids)
                if op_id == prev_id
            ]
            for idx in reversed(indices):
              stack.append(
                  (caller_instr.id, idx, caller_instr.id, new_call_stack)
              )
          else:
            stack.append(
                (caller_instr.id, tuple_index, caller_instr.id, new_call_stack)
            )
          continue

      for consumer in self.consumers.get(curr_id, []):
        yield (consumer, curr_id, tuple_index)

        consumer_op = consumer.opcode.lower()
        if consumer_op == "get-tuple-element":
          # Unwrap only the tuple element we are tracking.
          if tuple_index is None or consumer.tuple_index == tuple_index:
            stack.append((consumer.id, None, curr_id, call_stack))
        elif consumer_op == "tuple":
          # Packing into a tuple: remember which element(s) we occupy.
          indices = [
              i
              for i, op_id in enumerate(consumer.operand_ids)
              if op_id == curr_id
          ]
          for idx in reversed(indices):
            stack.append((consumer.id, idx, curr_id, call_stack))
        elif consumer_op == "fusion":
          # Entering a fusion: map operand position(s) to inner parameter(s).
          if consumer.called_computation_ids:
            called_comp_id = consumer.called_computation_ids[0]
            if called_comp_id in self.computations:
              indices = [
                  i
                  for i, op_id in enumerate(consumer.operand_ids)
                  if op_id == curr_id
              ]
              for idx in indices:
                param_instr = self.computation_parameters[called_comp_id].get(
                    idx
                )
                if param_instr is not None:
                  stack.append((
                      param_instr.id,
                      tuple_index,
                      curr_id,
                      call_stack + (consumer.id,),
                  ))
        elif consumer_op in passthrough_opcodes:
          stack.append((consumer.id, tuple_index, curr_id, call_stack))

  def trace_branch1_scale(self, convert_instr_id: int) -> int | None:
    """Traces downstream from convert_1 to the reduce that is Scale Node S."""
    # Shared shape ops plus arithmetic that can precede the reduce.
    passthrough = _SHAPE_OPCODES | frozenset({
        "abs",
        "negate",
        "subtract",
    })
    for consumer, _, tuple_index in self._iter_downstream(
        convert_instr_id, passthrough
    ):
      # Any reduce is the structural signal for Node S.
      if tuple_index is None and consumer.opcode.lower() in _REDUCTION_OPCODES:
        return consumer.id
    return None

  def trace_upstream_to_node(
      self, start_instr_id: int, target_node_id: int
  ) -> bool:
    """Checks start_instr_id reaches target_node_id in f32 (G2 f32 purity)."""
    cache_key = (start_instr_id, target_node_id)
    if cache_key in self._upstream_reach_cache:
      return self._upstream_reach_cache[cache_key]

    stack: list[int] = [start_instr_id]
    visited: set[int] = set()

    while stack:
      curr_id = stack.pop()
      if curr_id == target_node_id:
        self._upstream_reach_cache[cache_key] = True
        return True

      if curr_id in visited:
        continue
      visited.add(curr_id)

      instr = self.instructions.get(curr_id)
      if not instr:
        continue

      opcode = instr.opcode.lower()

      # Constants are never node S; stop here.
      if opcode == "constant":
        continue

      # G2: stop if the path drops below f32; tuple routing nodes are exempt.
      element_type = instr.shape.element_type
      if element_type != _F32_TYPE and element_type != _TUPLE_TYPE:
        continue

      if opcode == "fusion":
        if instr.called_computation_ids:
          comp = self.computations.get(instr.called_computation_ids[0])
          if comp:
            stack.append(comp.root_id)
      elif opcode == "parameter":
        comp_id = self.instruction_to_computation.get(curr_id)
        if comp_id is not None and comp_id in self.fusion_callers:
          _, caller_instr = self.fusion_callers[comp_id]
          if instr.parameter_number < len(caller_instr.operand_ids):
            stack.append(caller_instr.operand_ids[instr.parameter_number])
      else:
        for op_id in instr.operand_ids:
          stack.append(op_id)

    self._upstream_reach_cache[cache_key] = False
    return False

  def trace_branch2_quantization(
      self, convert_instr_id: int, scale_node_id: int
  ) -> tuple[int, int] | None:
    """Traces convert_2 to its scaling divide/multiply and FP8 downcast."""
    cache_key = (convert_instr_id, scale_node_id)
    if cache_key in self._branch2_cache:
      return self._branch2_cache[cache_key]

    for consumer, producer_id, tuple_index in self._iter_downstream(
        convert_instr_id, _QUANT_PASSTHROUGH_OPCODES
    ):
      if tuple_index is not None:
        continue
      opcode = consumer.opcode.lower()
      if opcode not in {"divide", "multiply"}:
        continue

      scale_candidate_id = None
      if opcode == "divide":
        # Division is not commutative: activation must be the dividend.
        if (
            len(consumer.operand_ids) >= 2
            and consumer.operand_ids[0] == producer_id
        ):
          scale_candidate_id = consumer.operand_ids[1]
      else:  # multiply is commutative.
        if len(consumer.operand_ids) >= 2:
          if consumer.operand_ids[0] == producer_id:
            scale_candidate_id = consumer.operand_ids[1]
          elif consumer.operand_ids[1] == producer_id:
            scale_candidate_id = consumer.operand_ids[0]

      if scale_candidate_id is not None:
        if self.trace_upstream_to_node(scale_candidate_id, scale_node_id):
          downcast_id = self._find_downcast_convert(consumer.id)
          if downcast_id is not None:
            result = (consumer.id, downcast_id)
            self._branch2_cache[cache_key] = result
            return result
    self._branch2_cache[cache_key] = None
    return None

  def _find_downcast_convert(self, scaling_instr_id: int) -> int | None:
    """Finds a downstream convert to an FP8 target type (crossing fusions)."""
    passthrough = _QUANT_PASSTHROUGH_OPCODES | frozenset({
        "clamp",
        "maximum",
        "minimum",
    })
    for consumer, _, tuple_index in self._iter_downstream(
        scaling_instr_id, passthrough
    ):
      if (
          tuple_index is None
          and consumer.opcode.lower() == "convert"
          and consumer.shape.element_type in _FP8_TARGET_TYPES
      ):
        return consumer.id
    return None


def _is_f32_upcast(tracer: _HloModuleTracer, instr: Any) -> bool:
  """Returns True if instr is an f32 convert whose operand is bf16/f16."""
  if instr.opcode.lower() != "convert" or instr.shape.element_type != _F32_TYPE:
    return False
  if not instr.operand_ids:
    return False
  operand = tracer.instructions.get(instr.operand_ids[0])
  return (
      operand is not None and operand.shape.element_type in _LOW_PRECISION_TYPES
  )


# --- Module-level analysis ---
def analyze_hlo_module(
    module_proto: Any,
    seed_convert_names: set[str] | None = None,
    tracer: _HloModuleTracer | None = None,
) -> list[dict[str, Any]]:
  """Analyzes an HLO module proto for dual-upcast dynamic scale patterns."""
  mod_tracer: _HloModuleTracer = (
      tracer if tracer is not None else _HloModuleTracer(module_proto)
  )
  module_name = getattr(module_proto, "name", "unknown_module")

  # Group upcast converts by activation source id (not call_stack) so a
  # convert_1 and convert_2 in different fusions sharing source X still pair.
  activation_to_converts: dict[int, list[Any]] = collections.defaultdict(list)

  for instr in mod_tracer.instructions.values():
    if _is_f32_upcast(mod_tracer, instr):
      src_id, _ = mod_tracer.trace_upstream_activation(instr.operand_ids[0])
      if src_id is not None:
        activation_to_converts[src_id].append(instr)

  bottlenecks: list[dict[str, Any]] = []

  for src_id, converts in activation_to_converts.items():
    if len(converts) < 2:
      continue

    # Branch 1 depends only on convert_1, so resolve Node S once per convert_1
    # (outer loop) rather than per pair: O(C^2) branch-1 walks become O(C).
    for convert_1 in converts:
      scale_node_id = mod_tracer.trace_branch1_scale(convert_1.id)
      if scale_node_id is None:
        continue

      for convert_2 in converts:
        if convert_2.id == convert_1.id:
          continue

        if seed_convert_names is not None and (
            convert_1.name not in seed_convert_names
            and convert_2.name not in seed_convert_names
        ):
          continue

        scaling_res = mod_tracer.trace_branch2_quantization(
            convert_2.id, scale_node_id
        )
        if scaling_res is None:
          continue

        scaling_instr_id, downcast_id = scaling_res

        c1_comp_id = mod_tracer.instruction_to_computation.get(convert_1.id)
        fusion_caller = (
            mod_tracer.fusion_callers.get(c1_comp_id)
            if c1_comp_id is not None
            else None
        )
        fusion_name = fusion_caller[1].name if fusion_caller else ""

        c2_comp_id = mod_tracer.instruction_to_computation.get(convert_2.id)
        c2_fusion_caller = (
            mod_tracer.fusion_callers.get(c2_comp_id)
            if c2_comp_id is not None
            else None
        )
        quant_fusion_name = c2_fusion_caller[1].name if c2_fusion_caller else ""

        finding_name = f"by_program/{module_name}/{convert_1.name}"
        recommendation = (
            "Detected unnecessary double f32 upcast pattern (bf16/f16 -> f32"
            f" in scale calculation '{convert_1.name}' and quantization branch"
            f" '{convert_2.name}') in module '{module_name}'. Recommendation:"
            " Keep scale calculation and quantization scaling in low precision"
            " (bf16/f16)."
        )
        explanation = (
            f"Activation Source ID: {src_id}\nScale Convert (convert_1):"
            f" {convert_1.name}\nQuantization Scale Convert (convert_2):"
            f" {convert_2.name}\nScale Factor Node ID: {scale_node_id}\nScaling"
            f" Op ID: {scaling_instr_id}\nDowncast Op ID: {downcast_id}"
        )

        bottleneck = {
            "name": finding_name,
            "instruction": convert_1.name,
            "fusion_name": fusion_name,
            # Second unnecessary upcast of same pattern, surfaced explicitly.
            "quant_instruction": convert_2.name,
            "quant_fusion_name": quant_fusion_name,
            "recommendation": recommendation,
            "explanation": explanation,
            "is_low_priority": False,
            "total_self_time_ms": 0.0,
        }
        bottlenecks.append(bottleneck)
        # Findings share a name and are de-duped later; first partner suffices.
        break

  unique_bottlenecks = list({b["name"]: b for b in bottlenecks}.values())
  return unique_bottlenecks


def _get_or_create_tracer(
    tracers: dict[int, _HloModuleTracer], module_proto: Any
) -> _HloModuleTracer:
  """Returns the cached tracer for a module, building it once on first use."""
  # Note: dict.setdefault would build a (discarded) tracer on every call.
  tracer = tracers.get(module_proto.id)
  if tracer is None:
    tracer = _HloModuleTracer(module_proto)
    tracers[module_proto.id] = tracer
  return tracer


# --- Public entry point ---
def detect_unnecessary_convert_dynamic_scale(
    session_id: str,
    get_top_hlo_ops_fn: Callable[
        ..., str
    ] = get_top_hlo_ops_tool.get_top_hlo_ops,
    limit: int = 50,
) -> str:
  """Detects unnecessary f32 upcasts in dynamic scale calculation and quantization.

  Args:
      session_id: The unique XProf session ID.
      get_top_hlo_ops_fn: Function to retrieve top HLO operations for timing.
      limit: How many top (highest-cost) operations to seed detection from.

  Returns:
      A JSON string summarizing the findings.
  """
  total_start_time = time.time()
  try:
    fetch_top_ops_start_time = time.time()
    top_ops = []
    try:
      top_ops_json = get_top_hlo_ops_fn(session_id, limit=limit)
      if top_ops_json:
        ops_data = json.loads(top_ops_json)
        top_ops = ops_data.get("top_by_time", [])
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.exception("Failed to fetch top ops for candidate lookup")
      return json.dumps({
          "bottlenecks_found": False,
          "inefficient_ops": [],
          "message": f"Failed to fetch top ops: {e}",
      })
    fetch_top_ops_end_time = time.time()

    fetch_hlo_proto_start_time = time.time()
    try:
      # pylint: disable=protected-access
      debug_info = hlo_tools._fetch_debug_info(session_id)
      # pylint: enable=protected-access
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.exception("Failed to fetch debug info for session %s", session_id)
      return json.dumps({
          "bottlenecks_found": False,
          "inefficient_ops": [],
          "message": f"Failed to fetch debug info: {e}",
      })
    fetch_hlo_proto_end_time = time.time()

    core_logic_start_time = time.time()

    modules: dict[str, Any] = {}
    for i, proto in enumerate(debug_info.hlo_proto):
      module_proto = proto.hlo_module
      if not module_proto:
        continue
      name = module_proto.name
      program_id = (
          debug_info.program_id[i] if i < len(debug_info.program_id) else None
      )
      mod_name = f"{name}({program_id})" if program_id else name
      modules[mod_name] = module_proto
      modules[name] = module_proto

    # Detection is gated on high-cost ops: only converts among the most
    # expensive ops are candidates. If none qualify, there is nothing to report.
    seed_convert_names: set[str] = set()
    top_ops_time_map: dict[str, float] = {}
    # Cache one tracer per module id, reused across seeding and analysis.
    tracers: dict[int, _HloModuleTracer] = {}

    if top_ops:
      for op in top_ops:
        raw_name = op.get("name", "")
        total_self_time_ms = op.get("total_self_time_ms", 0.0)
        if not raw_name or raw_name == "by_program/IDLE":
          continue

        name_clean = raw_name.replace(" and its duplicate(s)", "")
        parts = name_clean.split("/")
        if len(parts) <= 1 or parts[0] != "by_program":
          continue

        mod_name_key = parts[1]
        target_instr_name = parts[-1]

        module_proto = modules.get(mod_name_key)
        if module_proto is None:
          mod_base = mod_name_key.split("(")[0]
          module_proto = modules.get(mod_base)

        if module_proto is None:
          continue

        tracer = _get_or_create_tracer(tracers, module_proto)
        target_instr = tracer.instructions_by_name.get(target_instr_name)
        if target_instr is None:
          continue

        # A top op may be the convert itself, or a fusion wrapping the convert.
        opcode_lower = target_instr.opcode.lower()
        if opcode_lower == "convert":
          candidate_converts = [target_instr]
        elif opcode_lower == "fusion" and target_instr.called_computation_ids:
          comp = tracer.computations.get(target_instr.called_computation_ids[0])
          candidate_converts = list(comp.instructions) if comp else []
        else:
          candidate_converts = []

        for cand in candidate_converts:
          if _is_f32_upcast(tracer, cand):
            seed_convert_names.add(cand.name)
            top_ops_time_map[cand.name] = max(
                top_ops_time_map.get(cand.name, 0.0),
                total_self_time_ms,
            )

    all_bottlenecks = []
    seen_module_ids = set()
    for proto in debug_info.hlo_proto:
      module_proto = proto.hlo_module
      if not module_proto or module_proto.id in seen_module_ids:
        continue
      seen_module_ids.add(module_proto.id)
      tracer = _get_or_create_tracer(tracers, module_proto)
      bottlenecks = analyze_hlo_module(
          module_proto, seed_convert_names=seed_convert_names, tracer=tracer
      )
      for b in bottlenecks:
        # The pattern may be seeded via convert_1 or convert_2; attribute the
        # larger known self-time of the two.
        self_time = max(
            top_ops_time_map.get(b["instruction"], 0.0),
            top_ops_time_map.get(b["quant_instruction"], 0.0),
        )
        if self_time:
          b["total_self_time_ms"] = self_time
        all_bottlenecks.append(b)

    inefficient_ops = list({op["name"]: op for op in all_bottlenecks}.values())

    if inefficient_ops:
      message = (
          f"Detected {len(inefficient_ops)} dual-upcast dynamic scale"
          " operations with potential precision overhead."
      )
    else:
      message = "No unnecessary dynamic scale converts detected."

    core_logic_end_time = time.time()
    total_end_time = time.time()

    fetch_top_ops_time_s = fetch_top_ops_end_time - fetch_top_ops_start_time
    fetch_hlo_proto_time_s = (
        fetch_hlo_proto_end_time - fetch_hlo_proto_start_time
    )
    core_logic_time_s = core_logic_end_time - core_logic_start_time
    total_time_s = total_end_time - total_start_time

    logging.info(
        "Dynamic scale convert detection metrics - Session ID: %s, Total wall"
        " clock time: %.3fs, Fetch top ops time: %.3fs, Fetch HLO proto time:"
        " %.3fs, Core logic processing time: %.3fs",
        session_id,
        total_time_s,
        fetch_top_ops_time_s,
        fetch_hlo_proto_time_s,
        core_logic_time_s,
    )

    return json.dumps(
        {
            "bottlenecks_found": len(inefficient_ops) > 0,
            "inefficient_ops": inefficient_ops,
            "message": message,
        },
        indent=2,
    )

  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.exception("Error detecting dynamic scale convert overhead")
    return json.dumps({"error": f"Internal error during detection: {e}"})
