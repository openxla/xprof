"""MCP tool to detect unnecessary f32 promotions in reduction operations."""

import collections
from collections.abc import Callable
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

# Numerical classes of a reduction's summand. Only non-negative,
# cancellation-free sums (squares / non-negatives) are safe to keep in bf16.
_REDUCER_SUM_OF_SQUARES = "SUM_OF_SQUARES"
_REDUCER_SUM_NONNEG = "SUM_NONNEG"
_REDUCER_SUM_SIGNED = "SUM_SIGNED"
_REDUCER_PRODUCT = "PRODUCT"
_REDUCER_OTHER = "OTHER"
_NONNEG_REDUCER_CLASSES = frozenset(
    {_REDUCER_SUM_OF_SQUARES, _REDUCER_SUM_NONNEG}
)

# Precision-transparent, single-operand ops we can look through when reasoning
# about the sign/structure of a reduction's summand.
_PASSTHROUGH_OPCODES = frozenset({
    "reshape",
    "broadcast",
    "transpose",
    "copy",
    "convert",
    "bitcast",
    "slice",
})


class _HloModuleTracer:
  """Helper class to build HLO module indices and run precision tracing."""

  def __init__(self, module_proto: Any):
    self.computations = {}
    self.instructions = {}
    self.instruction_to_computation = {}
    self.fusion_callers = {}
    # Records EVERY call site of a computation (fusion_callers keeps only the
    # last), so the escape analysis traces all exit paths of shared comps.
    self.computation_callers = collections.defaultdict(list)
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
          self.computation_callers[called_comp_id].append(
              (computation.id, instruction)
          )

    self.phase = _classify_execution_phase(module_proto)
    self.entry_computation_id = module_proto.entry_computation_id

  def trace_upcast(
      self, start_instr_id: int, max_nodes: int = 10000
  ) -> Any | None:
    """Traces upstream from an instruction to locate an F32 upcast."""
    # Stack holds: (instruction_id, tuple_index, call_stack)
    # call_stack tracks active fusion callers to prevent context-sensitivity
    # collision.
    stack = [(start_instr_id, None, ())]

    # Track (instruction_id, tuple_index, call_stack) so different tuple
    # elements and different fusion-caller contexts are explored independently.
    visited = set()
    count = 0

    while stack:
      curr_id, tuple_index, call_stack = stack.pop()

      state_key = (curr_id, tuple_index, call_stack)
      if state_key in visited:
        continue
      visited.add(state_key)

      count += 1
      if count > max_nodes:
        # Give up conservatively on a pathological graph: report no upcast
        # found so the caller skips this op rather than emitting a
        # potentially unsafe recommendation.
        return None

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

  def _reaches_f32_sink(
      self,
      start_id: int,
      stop_ids: frozenset[int] = frozenset(),
      max_nodes: int = 10000,
  ) -> bool:
    """Returns True if the f32 value from ``start_id`` reaches an f32 sink.

    A sink consumes the value while still f32 (dot/conv/custom-call or module
    output); a downcast to bf16/f16 or a ``stop_ids`` node caps the path.

    Args:
      start_id: Instruction id whose f32 result is traced forward.
      stop_ids: Ids that legitimately consume the value (not escapes).
      max_nodes: Traversal cap; exceeding it conservatively returns True.
    """
    # Stack: (id, tuple_index, prev_instruction_id, call_stack).
    stack = [(start_id, None, None, ())]
    visited = set()  # holds (id, tuple_index, prev_instruction_id, call_stack)
    count = 0

    while stack:
      curr_id, tuple_index, prev_instr_id, call_stack = stack.pop()

      state_key = (curr_id, tuple_index, prev_instr_id, call_stack)
      if state_key in visited:
        continue
      visited.add(state_key)

      count += 1
      if count > max_nodes:
        # Give up conservatively: assume the value escapes so we do not emit a
        # potentially unsafe recommendation on a pathological graph.
        return True

      # A stop node marks a legitimate consumption of the f32 value; the path
      # through it is not an escape.
      if curr_id in stop_ids:
        continue

      instr = self.instructions.get(curr_id)
      if not instr:
        continue

      opcode = instr.opcode.lower()
      comp_id = self.instruction_to_computation.get(curr_id)
      comp = self.computations.get(comp_id) if comp_id is not None else None

      # 1. Exiting a fusion: hand off from the comp root to its caller(s).
      if comp and curr_id == comp.root_id:
        # Known call site -> return to that caller; otherwise fan out to ALL
        # call sites so shared computations don't hide an escape path.
        callers = []  # list of (caller_instr, new_call_stack)
        if call_stack:
          caller_instr = self.instructions.get(call_stack[-1])
          if caller_instr:
            callers.append((caller_instr, call_stack[:-1]))
        else:
          for _, caller_instr in self.computation_callers.get(comp_id, []):
            callers.append((caller_instr, ()))

        if callers:
          for caller_instr, new_call_stack in callers:
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
              stack.append((
                  caller_instr.id,
                  tuple_index,
                  caller_instr.id,
                  new_call_stack,
              ))
          continue

        # No caller: this is a top-level computation root. If it is the module
        # entry root, the f32 value is a module output -> escape.
        if comp_id == self.entry_computation_id:
          return True
        continue

      if opcode not in _ALLOWED_TRACING_OPCODES:
        continue

      # 2. Consumer traversal.
      for consumer in reversed(self.consumers.get(curr_id, [])):
        consumer_op = consumer.opcode.lower()

        # A. Downcast to bf16/f16 caps the path (f32 precision discarded here).
        if consumer_op == "convert":
          if tuple_index is None and consumer.shape.element_type in {
              _BF16_TYPE,
              _F16_TYPE,
          }:
            continue
          stack.append((consumer.id, tuple_index, curr_id, call_stack))
          continue

        # Any consumer we cannot trace through consumes the value while still
        # f32 -> f32 sink.
        if consumer_op not in _ALLOWED_TRACING_OPCODES:
          return True

        # B. Get-Tuple-Element
        if consumer_op == "get-tuple-element":
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
                  stack.append((
                      param_instr.id,
                      tuple_index,
                      curr_id,
                      call_stack + (consumer.id,),
                  ))

        # E. General precision-transparent op
        else:
          stack.append((consumer.id, tuple_index, curr_id, call_stack))

    return False

  def result_escapes_as_f32(self, reduce_id: int) -> bool:
    """True if the reduce's f32 result is used as f32 anywhere (not fully discarded)."""
    return self._reaches_f32_sink(reduce_id)

  def upcast_serves_only_reduce(self, upcast_id: int, reduce_id: int) -> bool:
    """True if the upcast's f32 result only feeds ``reduce_id``."""
    return not self._reaches_f32_sink(
        upcast_id, stop_ids=frozenset({reduce_id})
    )

  def classify_reducer(self, reduce_instr: Any) -> str:
    """Classifies the reduction's numerical behavior (see _REDUCER_* constants)."""
    if not reduce_instr.called_computation_ids:
      return _REDUCER_OTHER
    comp = self.computations.get(reduce_instr.called_computation_ids[0])
    if not comp:
      return _REDUCER_OTHER

    root_instr = self.instructions.get(comp.root_id)
    root_op = root_instr.opcode.lower() if root_instr else ""

    if root_op == "multiply":
      return _REDUCER_PRODUCT

    if root_op != "add":
      return _REDUCER_OTHER

    if not reduce_instr.operand_ids:
      return _REDUCER_SUM_SIGNED
    summand = self.instructions.get(reduce_instr.operand_ids[0])
    return self._classify_summand(summand)

  def _classify_summand(self, instr: Any, depth: int = 0) -> str:
    """Classifies a summand as sum-of-squares / non-negative / signed."""
    if instr is None or depth > 8:
      return _REDUCER_SUM_SIGNED

    opcode = instr.opcode.lower()

    if opcode == "multiply":
      # x * x is a square (non-negative) regardless of the sign of x.
      if (
          len(instr.operand_ids) == 2
          and instr.operand_ids[0] == instr.operand_ids[1]
      ):
        return _REDUCER_SUM_OF_SQUARES
      # Product of two provably non-negative operands stays non-negative.
      if (
          len(instr.operand_ids) == 2
          and self._is_nonneg(self.instructions.get(instr.operand_ids[0]))
          and self._is_nonneg(self.instructions.get(instr.operand_ids[1]))
      ):
        return _REDUCER_SUM_NONNEG
      return _REDUCER_SUM_SIGNED

    if opcode in {"abs", "exponential"}:
      return _REDUCER_SUM_NONNEG

    if opcode in _PASSTHROUGH_OPCODES and instr.operand_ids:
      return self._classify_summand(
          self.instructions.get(instr.operand_ids[0]), depth + 1
      )

    return _REDUCER_SUM_SIGNED

  def _is_nonneg(self, instr: Any, depth: int = 0) -> bool:
    """Best-effort static check that ``instr`` produces non-negative values."""
    if instr is None or depth > 6:
      return False
    opcode = instr.opcode.lower()
    if opcode in {"abs", "exponential"}:
      return True
    if (
        opcode == "multiply"
        and len(instr.operand_ids) == 2
        and instr.operand_ids[0] == instr.operand_ids[1]
    ):
      return True
    if opcode in _PASSTHROUGH_OPCODES and instr.operand_ids:
      return self._is_nonneg(
          self.instructions.get(instr.operand_ids[0]), depth + 1
      )
    return False


def _classify_execution_phase(module_proto: Any) -> str:
  """Classifies execution phase as TRAINING or INFERENCE by computation name."""
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

        # Context gate 1: skip training (must stay high precision).
        if tracer.phase == "TRAINING":
          continue

        # Context gate 2: only non-negative, cancellation-free sums
        # (sum-of-squares / sum-of-non-negatives) are safe to keep in bf16.
        reducer_class = tracer.classify_reducer(reduce_instr)
        if reducer_class not in _NONNEG_REDUCER_CLASSES:
          continue

        # Reduction size N, reported for context: bf16 accumulation error grows
        # with N, so large N warrants numerical validation before downgrading.
        reduction_size = 0
        if reduce_instr.operand_ids:
          operand_shape = tracer.instructions[reduce_instr.operand_ids[0]].shape
          reduction_size = _calculate_reduction_size(
              operand_shape, list(reduce_instr.dimensions)
          )

        # Locate the bf16/f16 -> f32 upcast feeding the reduction.
        upcast_instr = None
        for op_id in reduce_instr.operand_ids:
          upcast_instr = tracer.trace_upcast(op_id)
          if upcast_instr is not None:
            break

        if upcast_instr is None:
          continue

        # Structural gate (WS1): reject if the f32 result reaches any f32 sink
        # (i.e. it is used as f32, not fully downcast back to bf16/f16).
        if tracer.result_escapes_as_f32(reduce_instr.id):
          continue

        # Structural gate (WS1): the upcast must feed only this reduction.
        if not tracer.upcast_serves_only_reduce(
            upcast_instr.id, reduce_instr.id
        ):
          continue

        # Match: the f32 convert before this reduction is unnecessary.
        phase = tracer.phase

        upcast_comp_id = tracer.instruction_to_computation.get(upcast_instr.id)
        fusion_caller = (
            tracer.fusion_callers.get(upcast_comp_id)
            if upcast_comp_id is not None
            else None
        )
        fusion_name = fusion_caller[1].name if fusion_caller else ""

        formatted_rec = (
            "Detected candidate unnecessary promotion (bf16/f16 -> f32 ->"
            f" reduce -> bf16/f16) involving upcast '{upcast_instr.name}'"
            f" before reduce '{reduce_instr.name}' in module '{full_mod_name}'."
            " Recommendation: consider keeping the reduction accumulation in"
            f" bf16/f16 to match inputs. Reduction size N={reduction_size}; for"
            " large N validate numerically first, since bf16 accumulation"
            " error grows with N (tree reduction stays accurate, sequential"
            " does not)."
            f" [reducer_class={reducer_class}] [fusion_name={fusion_name}]"
        )
        explanation = (
            f"Phase: {phase}\nReducer: {reducer_class}\n"
            f"Reduction Size: {reduction_size}"
        )

        bottleneck = {
            "name": f"by_program/{full_mod_name}/{upcast_instr.name}",
            "category": op.get("category", ""),
            "total_self_time_ms": op.get("total_self_time_ms", 0.0),
            "occurrences": op.get("occurrences", 1),
            "flops": op.get("flops", 0),
            "bytes_accessed": op.get("bytes_accessed", 0),
            "instruction": upcast_instr.name,
            "fusion_name": fusion_name,
            "recommendation": formatted_rec,
            "explanation": explanation,
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
