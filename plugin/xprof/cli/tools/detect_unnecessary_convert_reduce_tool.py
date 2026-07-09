"""MCP tool to detect unnecessary f32 promotions in reduction operations."""

import collections
from collections.abc import Callable
import enum
import json
import logging
import time
from typing import Any

from xprof.cli.internal.oss import hlo_tools
from xprof.cli.tools import get_top_hlo_ops_tool

# These ops are neutral to precision changes. Complex arithmetic ops
# are excluded to prevent tracing through precision-altering bounds.
_ALLOWED_TRACING_OPCODES = frozenset({
    # Element-wise operations
    "abs",
    "negate",
    "add",
    "subtract",
    "multiply",
    "divide",
    "exponential",
    "log",
    "maximum",
    "minimum",
    "select",
    "compare",
    "power",
    "sine",
    "cosine",
    "round",
    "floor",
    "ceil",
    "sign",
    "sqrt",
    "rsqrt",
    "tanh",
    # Shape & data rearrangement operations
    "reshape",
    "broadcast",
    "transpose",
    "slice",
    "concatenate",
    "bitcast",
    "dynamic-slice",
    "dynamic-update-slice",
    "copy",
    # Graph structure, boundaries & data routing
    "parameter",
    "get-tuple-element",
    "fusion",
    "convert",
    "tuple",
    "reduce",
    "all-reduce",
    "reduce-scatter",
})


_REDUCTION_OPCODES = frozenset({"reduce", "all-reduce", "reduce-scatter"})

# These values correspond to the XLA PrimitiveType enum values.
_F16_TYPE = 10
_F32_TYPE = 11
_BF16_TYPE = 16


class _ReductionContext(str, enum.Enum):
  LOSS = "LOSS"
  NORM = "NORM"
  SOFTMAX = "SOFTMAX"
  GENERAL = "GENERAL"


class _HloModuleTracer:
  """Helper class to build HLO module indices and run precision tracing."""

  def __init__(self, module_proto: Any):
    self.computations = {}
    self.instructions = {}
    self.instruction_to_computation = {}
    self.fusion_callers = {}
    self.consumers = collections.defaultdict(list)
    self.computation_parameters = collections.defaultdict(dict)
    self.instructions_by_name = {}

    for computation in module_proto.computations:
      self.computations[computation.id] = computation
      for instruction in computation.instructions:
        self.instructions[instruction.id] = instruction
        self.instruction_to_computation[instruction.id] = computation.id
        self.instructions_by_name[instruction.name] = instruction

        # Track parameters by number
        if instruction.opcode.lower() == "parameter":
          self.computation_parameters[computation.id][
              instruction.parameter_number
          ] = instruction

        # Track consumers for each operand
        for operand_id in instruction.operand_ids:
          self.consumers[operand_id].append(instruction)

        # Track fusion callers
        for called_comp_id in instruction.called_computation_ids:
          self.fusion_callers[called_comp_id] = (computation.id, instruction)

    self.phase = _classify_execution_phase(module_proto)

  def verify_reducer(self, instr: Any) -> bool:
    """Verifies reducer root opcode is add/multiply (case-insensitive)."""
    if not instr.called_computation_ids:
      return False

    comp = self.computations.get(instr.called_computation_ids[0])
    if not comp:
      return False

    root_instr = self.instructions.get(comp.root_id)
    if root_instr and root_instr.opcode.lower() in {"add", "multiply"}:
      return True

    return any(
        kw in comp.name.lower() for kw in ("add", "sum", "mul", "multiply")
    )

  def classify_context(self, instr: Any) -> _ReductionContext:
    """Classifies context as NORM, SOFTMAX, LOSS, or GENERAL."""
    names_to_check = []

    comp_id = self.instruction_to_computation.get(instr.id)
    comp = self.computations.get(comp_id) if comp_id is not None else None
    if comp:
      names_to_check.append(comp.name)
      fusion_caller = self.fusion_callers.get(comp_id)
      if fusion_caller:
        _, caller_instr = fusion_caller
        names_to_check.append(caller_instr.name)
        if caller_instr.HasField("metadata") and caller_instr.metadata.op_name:
          names_to_check.append(caller_instr.metadata.op_name)

    if instr.HasField("metadata") and instr.metadata.op_name:
      names_to_check.append(instr.metadata.op_name)

    has_loss = False
    has_norm = False
    has_softmax = False

    loss_keywords = {"loss", "entropy", "criterion", "nll"}
    norm_keywords = {"norm", "variance", "mean"}

    for name in names_to_check:
      name_lower = name.lower()
      if any(kw in name_lower for kw in loss_keywords):
        has_loss = True
      if any(kw in name_lower for kw in norm_keywords):
        has_norm = True
      if "softmax" in name_lower:
        has_softmax = True

    # Prioritize LOSS > NORM > SOFTMAX to resolve overlapping metadata
    # scopes (e.g., loss-internal softmax) to the most specific
    # precision-safety domain.
    if has_loss:
      return _ReductionContext.LOSS
    if has_norm:
      return _ReductionContext.NORM
    if has_softmax:
      return _ReductionContext.SOFTMAX

    if comp and any(
        i.opcode.lower() == "exponential" for i in comp.instructions
    ):
      return _ReductionContext.SOFTMAX

    return _ReductionContext.GENERAL

  def trace_upcast(self, start_instr_id: int) -> Any | None:
    """Traces upstream from an instruction to locate an F32 upcast."""
    # Stack holds: (instruction_id, tuple_index, call_stack)
    # call_stack tracks active fusion callers to prevent context-sensitivity
    # collision.
    stack = [(start_instr_id, None, ())]

    # FIX: Track (instruction_id, tuple_index) to prevent cycle detection
    # collision across different tuple elements of a shared tuple.
    visited = set()

    while stack:
      curr_id, tuple_index, call_stack = stack.pop()

      state_key = (curr_id, tuple_index)
      if state_key in visited:
        continue
      visited.add(state_key)

      instr = self.instructions.get(curr_id)
      if not instr:
        continue

      opcode = instr.opcode.lower()
      if opcode not in _ALLOWED_TRACING_OPCODES:
        continue

      handled = False

      # 1. Match Convert Pattern
      if opcode == "convert":
        if tuple_index is None and instr.shape.element_type == _F32_TYPE:
          if instr.operand_ids:
            operand_instr = self.instructions.get(instr.operand_ids[0])
            if operand_instr and operand_instr.shape.element_type in {
                _BF16_TYPE,
                _F16_TYPE,
            }:
              return instr

      # 2. Get-Tuple-Element Unpacking
      elif opcode == "get-tuple-element":
        if instr.operand_ids:
          stack.append((instr.operand_ids[0], instr.tuple_index, call_stack))
          handled = True

      # 3. Tuple Packing
      elif opcode == "tuple":
        if tuple_index is not None:
          if tuple_index < len(instr.operand_ids):
            stack.append((instr.operand_ids[tuple_index], None, call_stack))
          # FIX: If tuple_index is out-of-bounds, mark handled to stop
          # incorrect fallthrough.
          handled = True

      # 4. Exiting Fusion (Parameter mapped to Caller Operands)
      elif opcode == "parameter":
        # FIX: Use dynamic call stack for context-sensitive mapping if available
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
          # Fall back to static map if call_stack is empty (e.g. trace started
          # inside fusion).
          comp_id = self.instruction_to_computation.get(instr.id)
          if comp_id is not None:
            fusion_caller = self.fusion_callers.get(comp_id)
            if fusion_caller:
              _, caller_instr = fusion_caller
              if instr.parameter_number < len(caller_instr.operand_ids):
                stack.append((
                    caller_instr.operand_ids[instr.parameter_number],
                    tuple_index,
                    (),
                ))
                handled = True

      # 5. Entering Fusion (Fusion mapped to called Computation Root)
      elif opcode == "fusion":
        if instr.called_computation_ids:
          comp = self.computations.get(instr.called_computation_ids[0])
          if comp:
            stack.append((comp.root_id, tuple_index, call_stack + (instr.id,)))
            handled = True

      # 6. Fallback General Operations (Element-wise and Shapes)
      if not handled:
        # Push operands in reversed order to preserve left-to-right DFS
        # traversal order.
        for op_id in reversed(instr.operand_ids):
          op_instr = self.instructions.get(op_id)
          if op_instr and op_instr.opcode.lower() == "constant":
            continue
          stack.append((op_id, tuple_index, call_stack))

    return None

  def trace_downcast(self, start_instr_id: int) -> bool:
    """Traces downstream from an instruction to locate a downcast to BF16/F16."""
    # Stack holds: (instruction_id, tuple_index, prev_instruction_id,
    # call_stack)
    stack = [(start_instr_id, None, None, ())]
    visited = set()  # holds (instruction_id, tuple_index)

    while stack:
      curr_id, tuple_index, prev_instr_id, call_stack = stack.pop()

      state_key = (curr_id, tuple_index)
      if state_key in visited:
        continue
      visited.add(state_key)

      instr = self.instructions.get(curr_id)
      if not instr:
        continue

      opcode = instr.opcode.lower()
      comp_id = self.instruction_to_computation.get(curr_id)

      # 1. Exiting Fusion (Root Node of a Computation reached)
      comp = self.computations.get(comp_id) if comp_id is not None else None
      if comp and curr_id == comp.root_id:
        caller_instr = None
        new_call_stack = call_stack

        # FIX: Resolve context using active call stack if available
        if call_stack:
          caller_id = call_stack[-1]
          caller_instr = self.instructions.get(caller_id)
          new_call_stack = call_stack[:-1]
        else:
          # Fall back to static map
          fusion_caller = self.fusion_callers.get(comp_id)
          if fusion_caller:
            _, caller_instr = fusion_caller

        if caller_instr:
          if opcode == "tuple" and prev_instr_id is not None:
            indices = [
                i
                for i, op_id in enumerate(instr.operand_ids)
                if op_id == prev_instr_id
            ]
            for idx in reversed(indices):
              stack.append(
                  (caller_instr.id, idx, caller_instr.id, new_call_stack)
              )
          else:
            stack.append(
                (caller_instr.id, tuple_index, caller_instr.id, new_call_stack)
            )
          continue  # Handoff complete

      if opcode not in _ALLOWED_TRACING_OPCODES:
        continue

      # 2. General Consumer Traversal
      for consumer in reversed(self.consumers.get(curr_id, [])):
        consumer_op = consumer.opcode.lower()
        if consumer_op not in _ALLOWED_TRACING_OPCODES:
          continue

        # A. Match Downcast Pattern
        if consumer_op == "convert":
          if tuple_index is None:
            if consumer.shape.element_type in {_BF16_TYPE, _F16_TYPE}:
              return True
          stack.append((consumer.id, tuple_index, curr_id, call_stack))

        # B. Get-Tuple-Element
        elif consumer_op == "get-tuple-element":
          if tuple_index is None or consumer.tuple_index == tuple_index:
            stack.append((consumer.id, None, curr_id, call_stack))

        # C. Tuple Packing
        elif consumer_op == "tuple":
          indices = [
              i
              for i, op_id in enumerate(consumer.operand_ids)
              if op_id == curr_id
          ]
          for idx in reversed(indices):
            stack.append((consumer.id, idx, curr_id, call_stack))

        # D. Entering Fusion
        elif consumer_op == "fusion":
          if consumer.called_computation_ids:
            called_comp_id = consumer.called_computation_ids[0]
            if called_comp_id in self.computations:
              indices = [
                  i
                  for i, op_id in enumerate(consumer.operand_ids)
                  if op_id == curr_id
              ]
              for idx in reversed(indices):
                param_instr = self.computation_parameters[called_comp_id].get(
                    idx
                )
                if param_instr is not None:
                  # Add consumer.id (fusion instruction) to call_stack
                  stack.append((
                      param_instr.id,
                      tuple_index,
                      curr_id,
                      call_stack + (consumer.id,),
                  ))

        # E. General Fallback
        else:
          stack.append((consumer.id, tuple_index, curr_id, call_stack))

    return False


def _classify_execution_phase(module_proto: Any) -> str:
  """Classifies execution phase (TRAINING vs INFERENCE) using case-insensitive computation name matches."""
  training_keywords = {
      "grad",
      "backward",
      "loss",
      "optimizer",
      "adam",
      "sgd",
      "update",
  }
  for computation in module_proto.computations:
    comp_name_lower = computation.name.lower()
    if any(keyword in comp_name_lower for keyword in training_keywords):
      return "TRAINING"
  return "INFERENCE"


def _calculate_reduction_size(shape_proto: Any, dimensions: list[int]) -> int:
  """Calculates the product of collapsed dimensions."""
  if not shape_proto.dimensions:
    return 0
  product = 1
  for dim in dimensions:
    if dim < 0 or dim >= len(shape_proto.dimensions):
      continue
    dim_size = shape_proto.dimensions[dim]
    if dim_size <= 0:
      return 0
    product *= dim_size
  return product


def _evaluate_optimization(
    phase: str,
    context: _ReductionContext,
    reduction_size: int,
) -> tuple[bool, str, str, bool]:
  """Applies rules matrix and returns (is_inefficient, recommendation, warning, is_low_priority)."""
  rec = (
      "Keep intermediate reduction calculation in BF16/F16 precision to match"
      " inputs."
  )

  if phase == "TRAINING":
    if context == _ReductionContext.GENERAL:
      return (
          True,
          "",
          "Low priority: Training upcast detected on a general reduction.",
          True,
      )
    return False, "", "", False

  if context in {_ReductionContext.NORM, _ReductionContext.SOFTMAX}:
    warning = ""
    if reduction_size > 1024:
      warning = (
          "Warning: The reduction size is large (>1024). Verify model accuracy"
          " after downcasting to avoid precision loss issues."
      )
    return True, rec, warning, False

  return True, rec, "", False


def detect_unnecessary_convert_reduce(
    session_id: str,
    get_top_hlo_ops_fn: Callable[
        ..., str
    ] = get_top_hlo_ops_tool.get_top_hlo_ops,
    limit: int = 50,
) -> str:
  """Detects reduce ops that unnecessarily promote bf16 to f32.

  Args:
      session_id: The unique XProf session ID.
      get_top_hlo_ops_fn: Function to retrieve top HLO operations for timing.
      limit: How many top operations to analyze.

  Returns:
      A JSON string summarizing the findings.
  """
  total_start_time = time.time()
  try:
    # 1. Fetch Top Ops to drive candidate scanning
    fetch_top_ops_start_time = time.time()
    try:
      top_ops_json = get_top_hlo_ops_fn(session_id, limit=limit)
      if not top_ops_json:
        return json.dumps({
            "bottlenecks_found": False,
            "inefficient_ops": [],
            "message": "No top ops found to analyze.",
        })
      ops_data = json.loads(top_ops_json)
      top_ops = ops_data.get("top_by_time", []) + ops_data.get(
          "top_by_bytes_accessed", []
      )
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.exception("Failed to fetch top ops for candidate lookup")
      return json.dumps({
          "bottlenecks_found": False,
          "inefficient_ops": [],
          "message": f"Failed to fetch top ops: {e}",
      })
    fetch_top_ops_end_time = time.time()

    # 2. Fetch HLO proto for all modules
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

    # Map module names and clean keys for lookup
    modules = {}
    for i, proto in enumerate(debug_info.hlo_proto):
      module_proto = proto.hlo_module
      if not module_proto:
        continue
      name = module_proto.name
      program_id = (
          debug_info.program_id[i] if i < len(debug_info.program_id) else None
      )
      mod_name = f"{name}({program_id})" if program_id else name
      modules[mod_name] = (module_proto, mod_name)
      modules[name] = (module_proto, mod_name)

    inefficient_ops = []
    tracers = {}

    core_logic_start_time = time.time()
    # 3. Scan and match only the top ops candidates
    for op in top_ops:
      raw_name = op.get("name", "")
      if not raw_name or raw_name == "by_program/IDLE":
        continue

      # Parse module name and target instruction name
      name_clean = raw_name.replace(" and its duplicate(s)", "")
      parts = name_clean.split("/")
      if len(parts) <= 1 or parts[0] != "by_program":
        continue

      mod_name_key = parts[1]
      target_instr_name = parts[-1]

      if mod_name_key not in modules:
        mod_base = mod_name_key.split("(")[0]
        if mod_base in modules:
          module_proto, full_mod_name = modules[mod_base]
        else:
          continue
      else:
        module_proto, full_mod_name = modules[mod_name_key]

      if full_mod_name not in tracers:
        tracers[full_mod_name] = _HloModuleTracer(module_proto)
      tracer = tracers[full_mod_name]

      # Locate the target instruction by name
      target_instr = tracer.instructions_by_name.get(target_instr_name)
      if target_instr is None:
        continue

      # Retrieve reduce operations to trace
      reduces_to_trace: list[Any] = []
      opcode_lower = target_instr.opcode.lower()
      if opcode_lower in _REDUCTION_OPCODES:
        reduces_to_trace.append(target_instr)
      elif opcode_lower == "fusion":
        if target_instr.called_computation_ids:
          comp_id = target_instr.called_computation_ids[0]
          if comp_id in tracer.computations:
            comp = tracer.computations[comp_id]
            for inner_instr in comp.instructions:
              if inner_instr.opcode.lower() in _REDUCTION_OPCODES:
                reduces_to_trace.append(inner_instr)

      # Run tracing on discovered reduces
      for reduce_instr in reduces_to_trace:
        if reduce_instr.shape.element_type != _F32_TYPE:
          continue

        if not tracer.verify_reducer(reduce_instr):
          continue

        # Trace upstream for upcast
        upcast_found = False
        upcast_instr = None
        for op_id in reduce_instr.operand_ids:
          upcast_instr = tracer.trace_upcast(op_id)
          if upcast_instr is not None:
            upcast_found = True
            break

        if not upcast_found or upcast_instr is None:
          continue

        # Trace downstream for downcast
        found_downcast = tracer.trace_downcast(reduce_instr.id)
        if not found_downcast:
          continue

        # Calculate reduction size
        reduction_size = 0
        if reduce_instr.operand_ids:
          operand_shape = tracer.instructions[reduce_instr.operand_ids[0]].shape
          reduction_size = _calculate_reduction_size(
              operand_shape, list(reduce_instr.dimensions)
          )

        context = tracer.classify_context(reduce_instr)

        # Classify execution phase
        phase = tracer.phase

        # Evaluate rules matrix
        is_inefficient, recommendation, warning, is_low_priority = (
            _evaluate_optimization(phase, context, reduction_size)
        )

        if is_inefficient:
          upcast_comp_id = tracer.instruction_to_computation.get(
              upcast_instr.id
          )
          fusion_caller = (
              tracer.fusion_callers.get(upcast_comp_id)
              if upcast_comp_id is not None
              else None
          )
          fusion_name = fusion_caller[1].name if fusion_caller else ""

          # Format output recommendation
          formatted_rec = (
              "Detected unnecessary promotion pattern (bf16/f16 -> f32 ->"
              f" reduce -> bf16/f16) involving upcast '{upcast_instr.name}'"
              f" before reduce '{reduce_instr.name}' in module"
              f" '{full_mod_name}'. Recommendation: {recommendation}"
          )
          explanation = (
              f"Phase: {phase}\nContext: {context}\nReduction Size:"
              f" {reduction_size}\nWarning: {warning}"
          )

          bottleneck = {
              "name": f"by_program/{full_mod_name}/{upcast_instr.name}",
              "instruction": upcast_instr.name,
              "fusion_name": fusion_name,
              "recommendation": formatted_rec,
              "explanation": explanation,
              "is_low_priority": is_low_priority,
              "total_self_time_ms": op.get("total_self_time_ms", 0.0),
          }
          inefficient_ops.append(bottleneck)

    # De-duplicate bottlenecks by name
    inefficient_ops = list({op["name"]: op for op in inefficient_ops}.values())

    if inefficient_ops:
      message = (
          f"Detected {len(inefficient_ops)} reduction operations with potential"
          " default type promotion overhead."
      )
    else:
      message = "No inefficient reduction promotions detected."

    core_logic_end_time = time.time()
    total_end_time = time.time()

    fetch_top_ops_time_s = fetch_top_ops_end_time - fetch_top_ops_start_time
    fetch_hlo_proto_time_s = (
        fetch_hlo_proto_end_time - fetch_hlo_proto_start_time
    )
    core_logic_time_s = core_logic_end_time - core_logic_start_time
    total_time_s = total_end_time - total_start_time

    logging.info(
        "Convert-reduce type promotion detection metrics - "
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

    return json.dumps(
        {
            "bottlenecks_found": len(inefficient_ops) > 0,
            "inefficient_ops": inefficient_ops,
            "message": message,
        },
        indent=2,
    )

  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.exception("Error detecting reduce convert overhead")
    return json.dumps({"error": f"Internal error during detection: {e}"})
