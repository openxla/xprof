"""MCP tool to detect layout mismatch copy operations causing HBM overhead."""

import collections
from collections.abc import Mapping, Sequence
import itertools
import json
import logging
import time
import types
from typing import Any, Callable, TypedDict

from xprof.cli.internal.oss import (
    hlo_tools,
)
from xprof.cli.tools import (
    get_top_hlo_ops_tool,
)

_PRIMITIVE_TYPE_NAMES = {
    0: "PRIMITIVE_TYPE_INVALID",
    1: "PRED",
    30: "S1",
    26: "S2",
    21: "S4",
    2: "S8",
    3: "S16",
    4: "S32",
    5: "S64",
    31: "U1",
    27: "U2",
    22: "U4",
    6: "U8",
    7: "U16",
    8: "U32",
    9: "U64",
    10: "F16",
    11: "F32",
    16: "BF16",
    12: "F64",
    19: "F8E5M2",
    28: "F8E4M3",
    20: "F8E4M3FN",
    23: "F8E4M3B11FNUZ",
    29: "F8E3M4",
    24: "F8E5M2FNUZ",
    25: "F8E4M3FNUZ",
    32: "F4E2M1FN",
    33: "F8E8M0FNU",
    35: "F6E3M2FN",
    36: "F6E2M3FN",
    15: "C64",
    18: "C128",
    13: "TUPLE",
    14: "OPAQUE_TYPE",
    17: "TOKEN",
    34: "BUFFER",
}
_PRIMITIVE_TYPE_VALUES = {v: k for k, v in _PRIMITIVE_TYPE_NAMES.items()}


class UpstreamProducer(TypedDict):
  """Represents an upstream HLO producer (compute op, parameter, or constant).

  Attributes:
    name: The name of the HLO instruction.
    opcode: The opcode of the HLO instruction.
    distance: The topological distance (in hops) from the copy instruction.
  """

  name: str
  opcode: str
  distance: int


class DownstreamStage(TypedDict):
  """Represents a downstream compute-intensive HLO instruction.

  Attributes:
    name: The name of the HLO instruction.
    opcode: The opcode of the HLO instruction.
    distance: The topological distance (in hops) from the copy instruction.
  """

  name: str
  opcode: str
  distance: int


class BottleneckEntry(TypedDict):
  """Represents a detected layout mismatch copy bottleneck.

  Attributes:
    instruction_name: The name of the copy HLO instruction.
    source_shape: The formatted shape of the copy's operand.
    target_shape: The formatted shape of the copy's result.
    layout_mismatch: True if a physical layout mismatch was detected.
    source_minor_dim_optimal: True if the source shape's minor dimension is
      optimally aligned with TPU vector lane boundaries.
    target_minor_dim_optimal: True if the target shape's minor dimension is
      optimally aligned with TPU vector lane boundaries.
    upstream_stages: A list of upstream producers feeding into this copy.
    downstream_stages: A list of downstream compute stages consuming the copy's
      output.
    total_self_time_ms: The self-time spent executing this copy (in ms).
    bytes_accessed: The number of HBM bytes accessed by this copy operation.
    recommendation: A detailed, actionable recommendation for JAX refactoring.
  """

  instruction_name: str
  source_shape: str
  target_shape: str
  layout_mismatch: bool
  source_minor_dim_optimal: bool
  target_minor_dim_optimal: bool
  upstream_stages: list[UpstreamProducer]
  downstream_stages: list[DownstreamStage]
  total_self_time_ms: float
  bytes_accessed: int
  recommendation: str


def format_shape(shape_proto: Any) -> str:
  """Formats a ShapeProto into a human-readable string with layout.

  Args:
    shape_proto: The ShapeProto message to format.

  Returns:
    A formatted human-readable string.
  """
  if not shape_proto:
    return "unknown"

  if shape_proto.tuple_shapes:
    parts = [format_shape(s) for s in shape_proto.tuple_shapes]
    return f"({', '.join(parts)})"

  try:
    element_type_str = _PRIMITIVE_TYPE_NAMES.get(
        shape_proto.element_type, str(shape_proto.element_type)
    ).lower()
  except Exception:  # pylint: disable=broad-exception-caught
    element_type_str = str(shape_proto.element_type)

  dims_str = ", ".join(str(d) for d in shape_proto.dimensions)

  layout_str = ""
  if shape_proto.layout and shape_proto.layout.minor_to_major:
    minor_to_major_str = ",".join(
        str(m) for m in shape_proto.layout.minor_to_major
    )
    layout_str = f"{{{minor_to_major_str}}}"

  return f"{element_type_str}[{dims_str}]{layout_str}"


# 512-byte vector register represents exactly 4096 bits.
_PRIMITIVE_TYPE_SPECS = (
    ("F64", 64),
    ("S64", 64),
    ("U64", 64),
    ("C64", 64),  # Complex float (paired F32): 64 bits
    ("C128", 128),  # Complex double (paired F64): 128 bits
    ("F32", 32),
    ("S32", 32),
    ("U32", 32),
    ("BF16", 16),
    ("F16", 16),
    ("S16", 16),
    ("U16", 16),
    ("S8", 8),
    ("U8", 8),
    ("PRED", 8),  # Predicate storage width is 8 bits (1 byte)
    ("F8E5M2", 8),
    ("F8E4M3", 8),
    ("F8E4M3FN", 8),
    ("F8E4M3B11FNUZ", 8),
    ("F8E5M2FNUZ", 8),
    ("F8E4M3FNUZ", 8),
    ("F8E3M4", 8),
    ("F8E8M0FNU", 8),
    ("S4", 4),
    ("U4", 4),
    ("F4E2M1FN", 4),
    ("S2", 2),
    ("U2", 2),
    ("S1", 1),
    ("U1", 1),
)


def _parse_bit_width_from_name(name: str) -> int | None:
  """Dynamically parses the bit width from a type name string.

  Args:
    name: String representation of the type name (e.g. 'F32', 'BF16').

  Returns:
    The bit width as an integer, or None if the type is un-parseable.
  """
  name = name.upper()
  if not name:
    return None
  if name == "PRED":
    return 8
  if name.startswith("BF16"):
    return 16
  if name.startswith("C"):
    digits = "".join(c for c in name[1:] if c.isdigit())
    if digits:
      return int(digits)
  if name.startswith(("S", "U", "F")):
    prefix_len = 1
    digits = []
    for c in name[prefix_len:]:
      if c.isdigit():
        digits.append(c)
      else:
        if digits:
          break
    if digits:
      return int("".join(digits))
  return None


def _initialize_primitive_type_mapping() -> dict[int, int]:
  """Initializes the primitive type to lane size mapping dynamically and safely.

  Returns:
    A mapping dictionary from PrimitiveType enum integers to hardware lane
    sizes.
  """
  mapping = {}
  for spec_name, bit_width in _PRIMITIVE_TYPE_SPECS:
    val = None
    try:
      val = _PRIMITIVE_TYPE_VALUES.get(spec_name)
    except TypeError:
      pass

    if val is not None:
      mapping[val] = 4096 // bit_width
  return mapping


LANE_SIZE_BY_PRIMITIVE_TYPE = types.MappingProxyType(
    _initialize_primitive_type_mapping()
)

# Known non-compute custom-call targets.
_KNOWN_NON_COMPUTE_CUSTOM_CALLS = frozenset({
    "allocatebuffer",
    "pin",
    "unpin",
    "createbuffer",
    "movetodevice",
    "movetohost",
    "x64splitlow",
    "x64splithigh",
    "x64combine",
    "control_dep",
    "nopcustomcalltarget",
    "barrierstart",
    "barrierend",
    "barrier",
    "trace",
    "windowprefetch",
    "getrngseed",
    "padtostatic",
    "slicetodynamic",
    "setbound",
    "assumegatherindicesinbound",
    "assumescatterindicesinbound",
    "hostexecute",
    "hostcallback",
    "xla_ffi_python_cpu_callback",
    "sendtohost",
    "recvfromhost",
    "sharding",
    "xla.sdy.funcresultsharding",
    "xla.sdy.globaltolocalshape",
    "xla.sdy.localtoglobalshape",
    "spmdfulltoshardshape",
    "spmdshardtofullshape",
})

# Keywords indicating a compute-intensive HLO operation.
_COMPUTE_KEYWORDS = frozenset({
    "convolution",
    "dot",
    "reduce",
    "scatter",
    "gather",
    "fft",
    "triangular-solve",
    "cholesky",
    "sort",
    "topk",
    "batch-norm",
    "all-to-all",
    "collective-permute",
    "collective-broadcast",
})


def get_tpu_lane_size(
    element_type: int, max_packing_factor: int = 32
) -> int | None:
  """Returns the number of elements required for a 512-byte TPU alignment.

  Args:
    element_type: The integer representation of the PrimitiveType.
    max_packing_factor: The maximum packing factor on TPU TensorCore.

  Returns:
    The alignment lane size (number of elements), or None if the bit width
    cannot be determined.
  """
  bit_width = None
  if element_type in LANE_SIZE_BY_PRIMITIVE_TYPE:
    bit_width = 4096 // LANE_SIZE_BY_PRIMITIVE_TYPE[element_type]
  else:
    try:
      name = _PRIMITIVE_TYPE_NAMES.get(element_type, str(element_type))
      bit_width = _parse_bit_width_from_name(name)
    except Exception:  # pylint: disable=broad-exception-caught
      pass

  if not bit_width:
    return None

  effective_bit_width = max(bit_width, 32 // max_packing_factor)
  return 4096 // effective_bit_width


def has_layout_mismatch(source_shape: Any, target_shape: Any) -> bool:
  """Recursively checks for layout mismatches down to leaf shapes.

  Args:
    source_shape: The starting ShapeProto message.
    target_shape: The target/result ShapeProto message.

  Returns:
    True if layout changes physically between structural equivalents, False
    otherwise.
  """
  if not source_shape or not target_shape:
    return False

  if source_shape.tuple_shapes or target_shape.tuple_shapes:
    if len(source_shape.tuple_shapes) != len(target_shape.tuple_shapes):
      return True
    return any(
        has_layout_mismatch(src, tgt)
        for src, tgt in zip(
            source_shape.tuple_shapes, target_shape.tuple_shapes
        )
    )

  if source_shape.layout and source_shape.layout.minor_to_major:
    src_layout = list(source_shape.layout.minor_to_major)
  else:
    src_layout = None

  if target_shape.layout and target_shape.layout.minor_to_major:
    tgt_layout = list(target_shape.layout.minor_to_major)
  else:
    tgt_layout = None

  if src_layout is None and tgt_layout is not None:
    src_layout = list(reversed(range(len(source_shape.dimensions))))
  if tgt_layout is None and src_layout is not None:
    tgt_layout = list(reversed(range(len(target_shape.dimensions))))

  return src_layout != tgt_layout


def check_minor_dimension_optimality(
    shape_proto: Any,
    max_packing_factor: int = 32,
) -> tuple[bool, int | None, int | None]:
  """Recursively checks TPU lane alignment for all leaf minor dimensions.

  Args:
    shape_proto: The XLA shape proto to check recursively. Can be a tuple shape
      or a primitive leaf shape.
    max_packing_factor: The maximum packing factor for TPU dimension alignment
      considerations.

  Returns:
    A tuple containing:
      - bool: True if all leaf minor dimensions are optimal multiples of the
        calculated TPU lane sizes, False otherwise.
      - int | None: The size of the first discovered non-optimal minor-most
        dimension, otherwise None.
      - int | None: The calculated optimal TPU lane size for the element type of
        the non-optimal dimension, otherwise None.
  """
  if not shape_proto:
    return True, None, None

  if shape_proto.tuple_shapes:
    for sub in shape_proto.tuple_shapes:
      optimal, size, lane = check_minor_dimension_optimality(
          sub, max_packing_factor
      )
      if not optimal:
        return False, size, lane
    return True, None, None

  if not shape_proto.dimensions:
    return True, None, None

  minor_idx = len(shape_proto.dimensions) - 1
  if shape_proto.layout and shape_proto.layout.minor_to_major:
    minor_idx = shape_proto.layout.minor_to_major[0]

  if 0 <= minor_idx < len(shape_proto.dimensions):
    minor_size = shape_proto.dimensions[minor_idx]
    lane_size = get_tpu_lane_size(shape_proto.element_type, max_packing_factor)
    if lane_size is not None and minor_size % lane_size != 0:
      return False, minor_size, lane_size

  return True, None, None


def is_compute_custom_call(instr: Any) -> bool:
  """Determines if a custom-call HLO instruction is compute-intensive.

  Args:
    instr: The CustomCall instruction proto to check.

  Returns:
    True if the custom call matches a heavy compute kernel, False otherwise.
  """
  target = getattr(instr, "custom_call_target", "")
  if not target:
    return False

  target_lower = target.lower()

  if (
      target_lower == "tpu_custom_call"
      or target_lower in ("mosaic_gpu", "mosaic_gpu_v2")
      or target_lower.startswith((
          "__gpu$xla.gpu.triton",
          "__gpu$xla.gpu.ptx",
          "__cublas",
          "__cudnn",
          "__onednn",
          "sparsedense",
          "__op$",
      ))
      or target_lower == "edge_tpu_pallas_kernel"
  ):
    return True

  if target_lower in _KNOWN_NON_COMPUTE_CUSTOM_CALLS:
    return False

  if (
      "sharding" in target_lower
      or "placement" in target_lower
      or "metadata" in target_lower
      or "control_dep" in target_lower
      or target_lower.startswith(("annotate", "_spmdinternalop_"))
  ):
    return False

  return True


def is_compute_stage(
    instr: Any,
    comp_by_id: Mapping[int, Any],
    visited_fusions: set[int] | None = None,
) -> bool:
  """Checks if an HLO instruction is a compute-intensive stage.

  Args:
    instr: The HLO instruction proto to check.
    comp_by_id: A mapping from computation IDs to computation protos.
    visited_fusions: A set of fusion instruction IDs already visited to prevent
      infinite recursion.

  Returns:
    True if the instruction is a compute-intensive stage, False otherwise.
  """
  if visited_fusions is None:
    visited_fusions = set()

  opcode_lower = instr.opcode.lower()

  if any(keyword in opcode_lower for keyword in _COMPUTE_KEYWORDS):
    return True

  if opcode_lower == "custom-call":
    return is_compute_custom_call(instr)

  if opcode_lower == "fusion":
    if instr.id in visited_fusions:
      return False
    visited_fusions.add(instr.id)

    for comp_id in instr.called_computation_ids:
      comp = comp_by_id.get(comp_id)
      if comp:
        for inner_instr in comp.instructions:
          inner_op = inner_instr.opcode.lower()
          if any(keyword in inner_op for keyword in _COMPUTE_KEYWORDS):
            return True
          if inner_op == "custom-call" and is_compute_custom_call(inner_instr):
            return True
          if inner_op == "fusion":
            if is_compute_stage(inner_instr, comp_by_id, visited_fusions):
              return True
  return False


def find_upstream_compute_stages(
    copy_instr_id: int,
    instr_by_id: Mapping[int, Any],
    comp_by_id: Mapping[int, Any],
    comp_id_by_instr_id: Mapping[int, int],
    callers_by_comp_id: Mapping[int, Sequence[int]],
    max_depth: int = 5,
) -> Sequence[tuple[Any, int]]:
  """Finds compute-intensive producers upstream from a copy instruction.

  Traverses backward, tracking data and control flow dependency boundaries.

  Args:
    copy_instr_id: The instruction ID of the starting HLO Copy operation.
    instr_by_id: A mapping from HLO instruction IDs to instruction protos.
    comp_by_id: A mapping from computation IDs to computation protos.
    comp_id_by_instr_id: A mapping from instruction IDs to their computation ID.
    callers_by_comp_id: A mapping from computation IDs to their caller
      instruction IDs.
    max_depth: The maximum depth of the dataflow graph traversal (in number of
      hops).

  Returns:
    A sequence of tuples, where each tuple contains:
      - Any: The upstream HLO instruction proto, which can be a compute
        operation, a Constant, or a module-level Parameter.
      - int: The topological distance (hops) from the copy instruction.
  """
  visited = {(copy_instr_id, ())}
  queue = collections.deque([(copy_instr_id, (), 0)])
  upstream_producers = []

  while queue:
    curr_id, shape_idx, dist = queue.popleft()
    curr_instr = instr_by_id[curr_id]

    if dist > 0:
      if is_compute_stage(curr_instr, comp_by_id):
        upstream_producers.append((curr_instr, dist))
        continue
      if curr_instr.opcode.lower() == "constant":
        upstream_producers.append((curr_instr, dist))
        continue

      curr_comp_id = comp_id_by_instr_id.get(curr_id)
      if curr_instr.opcode.lower() == "parameter" and (
          curr_comp_id not in callers_by_comp_id
      ):
        upstream_producers.append((curr_instr, dist))
        continue

    if dist < max_depth:
      opcode = curr_instr.opcode.lower()

      if opcode == "parameter":
        curr_comp_id = comp_id_by_instr_id.get(curr_id)
        callers = callers_by_comp_id.get(curr_comp_id, [])
        param_num = getattr(curr_instr, "parameter_number", 0)
        for caller_id in callers:
          caller = instr_by_id.get(caller_id)
          if not caller:
            continue

          caller_opcode = caller.opcode.lower()
          if caller_opcode == "conditional":
            for branch_idx, comp_id in enumerate(caller.called_computation_ids):
              if comp_id == curr_comp_id:
                if branch_idx + 1 < len(caller.operand_ids):
                  target_id = caller.operand_ids[branch_idx + 1]
                  if (target_id, shape_idx) not in visited:
                    visited.add((target_id, shape_idx))
                    queue.append((target_id, shape_idx, dist + 1))
          elif caller_opcode in ("fusion", "call"):
            if param_num < len(caller.operand_ids):
              target_id = caller.operand_ids[param_num]
              if (target_id, shape_idx) not in visited:
                visited.add((target_id, shape_idx))
                queue.append((target_id, shape_idx, dist + 1))
          elif caller_opcode == "while":
            if param_num < len(caller.operand_ids):
              init_id = caller.operand_ids[param_num]
              if (init_id, shape_idx) not in visited:
                visited.add((init_id, shape_idx))
                queue.append((init_id, shape_idx, dist + 1))
            body_comp_id = caller.called_computation_ids[1]
            body_comp = comp_by_id.get(body_comp_id)
            if body_comp and body_comp.root_id:
              if (body_comp.root_id, shape_idx) not in visited:
                visited.add((body_comp.root_id, shape_idx))
                queue.append((body_comp.root_id, shape_idx, dist + 1))

      elif opcode == "get-tuple-element":
        idx = getattr(curr_instr, "tuple_index", 0)
        parent_idx = (idx,) + shape_idx
        if curr_instr.operand_ids:
          target_id = curr_instr.operand_ids[0]
          if (target_id, parent_idx) not in visited:
            visited.add((target_id, parent_idx))
            queue.append((target_id, parent_idx, dist + 1))

      elif opcode == "tuple":
        if shape_idx:
          idx = shape_idx[0]
          remaining_idx = shape_idx[1:]
          if idx < len(curr_instr.operand_ids):
            target_id = curr_instr.operand_ids[idx]
            if (target_id, remaining_idx) not in visited:
              visited.add((target_id, remaining_idx))
              queue.append((target_id, remaining_idx, dist + 1))
        else:
          for target_id in curr_instr.operand_ids:
            if (target_id, ()) not in visited:
              visited.add((target_id, ()))
              queue.append((target_id, (), dist + 1))

      elif opcode == "while":
        if curr_instr.operand_ids:
          init_id = curr_instr.operand_ids[0]
          if (init_id, shape_idx) not in visited:
            visited.add((init_id, shape_idx))
            queue.append((init_id, shape_idx, dist + 1))
        if len(curr_instr.called_computation_ids) > 1:
          body_comp_id = curr_instr.called_computation_ids[1]
          body_comp = comp_by_id.get(body_comp_id)
          if body_comp and body_comp.root_id:
            if (body_comp.root_id, shape_idx) not in visited:
              visited.add((body_comp.root_id, shape_idx))
              queue.append((body_comp.root_id, shape_idx, dist + 1))

      elif opcode == "call":
        if curr_instr.called_computation_ids:
          called_comp_id = curr_instr.called_computation_ids[0]
          called_comp = comp_by_id.get(called_comp_id)
          if called_comp and called_comp.root_id:
            if (called_comp.root_id, shape_idx) not in visited:
              visited.add((called_comp.root_id, shape_idx))
              queue.append((called_comp.root_id, shape_idx, dist + 1))

      else:
        for operand_id in curr_instr.operand_ids:
          if (
              operand_id,
              shape_idx,
          ) not in visited and operand_id in instr_by_id:
            visited.add((operand_id, shape_idx))
            queue.append((operand_id, shape_idx, dist + 1))

  return upstream_producers


def find_downstream_compute_stages(
    copy_instr_id: int,
    instr_by_id: Mapping[int, Any],
    users_by_id: Mapping[int, Sequence[int]],
    comp_by_id: Mapping[int, Any],
    comp_id_by_instr_id: Mapping[int, int],
    callers_by_comp_id: Mapping[int, Sequence[int]],
    root_id_by_comp_id: Mapping[int, int],
    max_depth: int = 5,
) -> Sequence[tuple[Any, int]]:
  """Finds compute-intensive consumers downstream from a copy instruction.

  Traverses forward, tracking data and control flow dependency boundaries.

  Args:
    copy_instr_id: The instruction ID of the starting HLO Copy operation.
    instr_by_id: A mapping from HLO instruction IDs to instruction protos.
    users_by_id: A mapping from instruction IDs to their user instruction IDs.
    comp_by_id: A mapping from computation IDs to computation protos.
    comp_id_by_instr_id: A mapping from instruction IDs to their computation ID.
    callers_by_comp_id: A mapping from computation IDs to their caller
      instruction IDs.
    root_id_by_comp_id: A mapping from computation IDs to their root
      instruction ID.
    max_depth: The maximum depth of the dataflow graph traversal (in number of
      hops).

  Returns:
    A sequence of tuples, where each tuple contains:
      - Any: The downstream compute-intensive HLO instruction proto.
      - int: The topological distance (hops) from the copy instruction.
  """
  visited = {(copy_instr_id, ())}
  queue = collections.deque([(copy_instr_id, (), 0)])
  compute_consumers = []

  while queue:
    curr_id, shape_idx, dist = queue.popleft()
    curr_instr = instr_by_id[curr_id]

    if dist > 0:
      if is_compute_stage(curr_instr, comp_by_id):
        compute_consumers.append((curr_instr, dist))
        continue

    if dist < max_depth:
      opcode = curr_instr.opcode.lower()
      curr_comp_id = comp_id_by_instr_id.get(curr_id)
      root_id = root_id_by_comp_id.get(curr_comp_id)

      if curr_id == root_id:
        callers = callers_by_comp_id.get(curr_comp_id, [])
        for caller_id in callers:
          caller = instr_by_id.get(caller_id)
          if not caller:
            continue
          caller_opcode = caller.opcode.lower()

          if caller_opcode == "while":
            if caller.called_computation_ids:
              if len(caller.called_computation_ids) > 1:
                body_comp_id = caller.called_computation_ids[1]
              else:
                body_comp_id = caller.called_computation_ids[0]
              body_comp = comp_by_id.get(body_comp_id)
              if body_comp:
                for inner_i in body_comp.instructions:
                  if (
                      inner_i.opcode.lower() == "parameter"
                      and inner_i.parameter_number == 0
                  ):
                    if (inner_i.id, shape_idx) not in visited:
                      visited.add((inner_i.id, shape_idx))
                      queue.append((inner_i.id, shape_idx, dist + 1))

            for user_id in users_by_id.get(caller_id, []):
              if (user_id, shape_idx) not in visited and user_id in instr_by_id:
                visited.add((user_id, shape_idx))
                queue.append((user_id, shape_idx, dist + 1))
          else:
            for user_id in users_by_id.get(caller_id, []):
              if (user_id, shape_idx) not in visited and user_id in instr_by_id:
                visited.add((user_id, shape_idx))
                queue.append((user_id, shape_idx, dist + 1))

      elif opcode == "tuple":
        for user_id in users_by_id.get(curr_id, []):
          user_instr = instr_by_id.get(user_id)
          if not user_instr:
            continue
          if user_instr.opcode.lower() == "get-tuple-element":
            idx = getattr(user_instr, "tuple_index", 0)
            if shape_idx and shape_idx[0] == idx:
              remaining_idx = shape_idx[1:]
              if (user_id, remaining_idx) not in visited:
                visited.add((user_id, remaining_idx))
                queue.append((user_id, remaining_idx, dist + 1))
            elif not shape_idx:
              if (user_id, ()) not in visited:
                visited.add((user_id, ()))
                queue.append((user_id, (), dist + 1))
          else:
            if (user_id, shape_idx) not in visited:
              visited.add((user_id, shape_idx))
              queue.append((user_id, shape_idx, dist + 1))

      elif opcode == "get-tuple-element":
        idx = getattr(curr_instr, "tuple_index", 0)
        new_idx = shape_idx + (idx,)
        for user_id in users_by_id.get(curr_id, []):
          if (user_id, new_idx) not in visited:
            visited.add((user_id, new_idx))
            queue.append((user_id, new_idx, dist + 1))

      elif opcode == "while":
        if len(curr_instr.called_computation_ids) > 1:
          body_comp_id = curr_instr.called_computation_ids[1]
          body_comp = comp_by_id.get(body_comp_id)
          if body_comp:
            for inner_i in body_comp.instructions:
              if (
                  inner_i.opcode.lower() == "parameter"
                  and inner_i.parameter_number == 0
              ):
                if (inner_i.id, shape_idx) not in visited:
                  visited.add((inner_i.id, shape_idx))
                  queue.append((inner_i.id, shape_idx, dist + 1))

        for user_id in users_by_id.get(curr_id, []):
          if (user_id, shape_idx) not in visited and user_id in instr_by_id:
            visited.add((user_id, shape_idx))
            queue.append((user_id, shape_idx, dist + 1))

      elif opcode == "call":
        if curr_instr.called_computation_ids:
          called_comp_id = curr_instr.called_computation_ids[0]
          called_comp = comp_by_id.get(called_comp_id)
          if called_comp:
            for inner_i in called_comp.instructions:
              if (
                  inner_i.opcode.lower() == "parameter"
                  and inner_i.parameter_number == 0
              ):
                if (inner_i.id, shape_idx) not in visited:
                  visited.add((inner_i.id, shape_idx))
                  queue.append((inner_i.id, shape_idx, dist + 1))

      else:
        for user_id in users_by_id.get(curr_id, []):
          user_instr = instr_by_id.get(user_id)
          if not user_instr:
            continue
          if user_instr.opcode.lower() == "tuple":
            for operand_idx, operand_id in enumerate(user_instr.operand_ids):
              if operand_id == curr_id:
                new_shape_idx = shape_idx + (operand_idx,)
                if (user_id, new_shape_idx) not in visited:
                  visited.add((user_id, new_shape_idx))
                  queue.append((user_id, new_shape_idx, dist + 1))
          else:
            if (user_id, shape_idx) not in visited and user_id in instr_by_id:
              visited.add((user_id, shape_idx))
              queue.append((user_id, shape_idx, dist + 1))

  return compute_consumers


def detect_layout_mismatch_copies(
    session_id: str,
    get_top_hlo_ops_fn: Callable[
        ..., str
    ] = get_top_hlo_ops_tool.get_top_hlo_ops,
    limit: int = 100,
) -> str:
  """Detects layout mismatch copy ops sandwiched between compute stages.

  Args:
    session_id: The unique XProf session ID.
    get_top_hlo_ops_fn: Function to retrieve profiled top HLO operations.
    limit: Number of top operations to fetch for profile data enrichment.

  Returns:
    A JSON-formatted string detailing the detected bottlenecks and JAX-level
    refactoring recommendations.
  """
  try:
    total_start_time = time.time()

    debug_info = hlo_tools._fetch_debug_info(session_id)  # pylint: disable=protected-access
    if not debug_info.hlo_proto:
      return json.dumps({"error": "No HLO proto found in the session."})

    op_metrics = {}
    try:
      top_ops_json = get_top_hlo_ops_fn(session_id, limit=limit)
      if top_ops_json:
        ops_data = json.loads(top_ops_json)
        all_profiled_ops = itertools.chain(
            ops_data.get("top_by_time", []),
            ops_data.get("top_by_flops", []),
            ops_data.get("top_by_bytes_accessed", []),
        )
        for op in all_profiled_ops:
          raw_name = op.get("name", "")
          parts = raw_name.split("/")
          if len(parts) > 1:
            comp_name = parts[0]
            instr_name_part = parts[-1].split(" and its ")[0]
            instr_name = instr_name_part.replace("%", "").strip()
            op_metrics[(comp_name, instr_name)] = op
          else:
            instr_name_part = raw_name.split(" and its ")[0]
            instr_name = instr_name_part.replace("%", "").strip()
            op_metrics[("", instr_name)] = op
    except (json.JSONDecodeError, TypeError) as e:
      logging.warning(
          "Failed to fetch or parse top HLO ops: %r", e, exc_info=True
      )

    core_logic_start_time = time.time()

    inefficient_ops = []

    for hlo_proto in debug_info.hlo_proto:
      module_proto = hlo_proto.hlo_module

      instr_by_id = {}
      comp_by_id = {}
      users_by_id = collections.defaultdict(list)

      comp_name_by_id = {
          comp.id: comp.name for comp in module_proto.computations
      }
      instr_id_to_comp_id = {}
      callers_by_comp_id = collections.defaultdict(list)
      comp_id_by_instr_id = {}
      root_id_by_comp_id = {}

      for comp in module_proto.computations:
        comp_by_id[comp.id] = comp
        root_id_by_comp_id[comp.id] = comp.root_id
        for instr in comp.instructions:
          instr_id_to_comp_id[instr.id] = comp.id
          comp_id_by_instr_id[instr.id] = comp.id
          instr_by_id[instr.id] = instr
          for comp_id in instr.called_computation_ids:
            callers_by_comp_id[comp_id].append(instr.id)
          for operand_id in instr.operand_ids:
            users_by_id[operand_id].append(instr.id)

      for instr in instr_by_id.values():
        if instr.opcode.lower() != "copy":
          continue

        if not instr.operand_ids:
          continue

        operand_id = instr.operand_ids[0]
        operand_instr = instr_by_id.get(operand_id)
        if not operand_instr:
          continue

        upstream_producers = find_upstream_compute_stages(
            instr.id,
            instr_by_id,
            comp_by_id,
            comp_id_by_instr_id,
            callers_by_comp_id,
            max_depth=5,
        )
        downstream_stages = find_downstream_compute_stages(
            instr.id,
            instr_by_id,
            users_by_id,
            comp_by_id,
            comp_id_by_instr_id,
            callers_by_comp_id,
            root_id_by_comp_id,
            max_depth=5,
        )

        if upstream_producers and downstream_stages:
          source_shape = operand_instr.shape
          target_shape = instr.shape

          source_shape_str = format_shape(source_shape)
          target_shape_str = format_shape(target_shape)

          layout_mismatch = has_layout_mismatch(source_shape, target_shape)

          source_optimal, source_minor_size, source_lane_size = (
              check_minor_dimension_optimality(source_shape)
          )
          target_optimal, target_minor_size, target_lane_size = (
              check_minor_dimension_optimality(target_shape)
          )

          upstream_names_str = ", ".join(
              f"'{u.name}' ({u.opcode}, dist={d})"
              for u, d in upstream_producers
          )
          downstream_names_str = ", ".join(
              f"'{d.name}' ({d.opcode}, dist={dist})"
              for d, dist in downstream_stages
          )

          recommendation_parts = []
          recommendation_parts.append(
              f"Copy op '{instr.name}' is sandwiched between upstream producers"
              f" (compute, parameters, or constants) [{upstream_names_str}]"
              f" and downstream compute stages [{downstream_names_str}]."
          )

          if layout_mismatch:
            recommendation_parts.append(
                f" Layout mismatch detected! Input: {source_shape_str} ->"
                f" Output: {target_shape_str}. This forces physical tensor"
                " reordering in HBM."
            )
          else:
            recommendation_parts.append(
                " While layouts match structurally, consider if this copy can"
                " be eliminated by layout propagation."
            )

          non_optimal_dims = []
          if not source_optimal and source_minor_size is not None:
            non_optimal_dims.append(
                f"input minor-most dimension ({source_minor_size}, expected"
                f" multiple of {source_lane_size})"
            )
          if not target_optimal and target_minor_size is not None:
            non_optimal_dims.append(
                f"output minor-most dimension ({target_minor_size}, expected"
                f" multiple of {target_lane_size})"
            )

          if non_optimal_dims:
            recommendation_parts.append(
                " Non-optimal dimension lane sizes found for TPU:"
                f" {', '.join(non_optimal_dims)}. This causes padded alignment"
                " overhead."
            )

          recommendation_parts.append(
              " Refactoring Recommendation: Restructure your JAX computation by"
              " moving transposes or dimension reorderings earlier in your"
              " model. Where multiple updates are applied to the same data"
              " (e.g. updates to K, V, and their scale parameters), consolidate"
              " these into a single JAX function call to enable XLA layout"
              " propagation and copy fusion."
          )
          recommendation = "".join(recommendation_parts)

          metrics = {}
          instr_comp_id = instr_id_to_comp_id.get(instr.id)
          if instr_comp_id:
            instr_comp_name = comp_name_by_id.get(instr_comp_id, "")
            metrics = op_metrics.get((instr_comp_name, instr.name), {})
          if not metrics:
            metrics = op_metrics.get(("", instr.name), {})

          self_time_ms = metrics.get("total_self_time_ms", 0.0)
          bytes_accessed = metrics.get("bytes_accessed", 0)

          bottleneck_entry = BottleneckEntry(
              instruction_name=instr.name,
              source_shape=source_shape_str,
              target_shape=target_shape_str,
              layout_mismatch=layout_mismatch,
              source_minor_dim_optimal=source_optimal,
              target_minor_dim_optimal=target_optimal,
              upstream_stages=[
                  UpstreamProducer(name=u.name, opcode=u.opcode, distance=d)
                  for u, d in upstream_producers
              ],
              downstream_stages=[
                  DownstreamStage(name=d.name, opcode=d.opcode, distance=dist)
                  for d, dist in downstream_stages
              ],
              total_self_time_ms=self_time_ms,
              bytes_accessed=bytes_accessed,
              recommendation=recommendation,
          )
          inefficient_ops.append(bottleneck_entry)

    inefficient_ops.sort(
        key=lambda x: (x["total_self_time_ms"], x["bytes_accessed"]),
        reverse=True,
    )

    bottlenecks_found = bool(inefficient_ops)
    if bottlenecks_found:
      message = (
          f"Detected {len(inefficient_ops)} layout mismatch copy operations"
          " causing HBM materialization and layout mismatch overhead between"
          " compute stages."
      )
    else:
      message = "No layout mismatch copy bottlenecks detected."

    core_logic_end_time = time.time()
    core_logic_time_ms = (core_logic_end_time - core_logic_start_time) * 1000.0
    core_logic_time_s = core_logic_end_time - core_logic_start_time
    total_end_time = time.time()
    total_time_ms = (total_end_time - total_start_time) * 1000.0
    total_time_s = total_end_time - total_start_time

    logging.info(
        "Layout mismatch copy detection metrics - "
        "Total wall clock time: %.2fs (%.2fms), "
        "Core logic processing time: %.2fs (%.2fms)",
        total_time_s,
        total_time_ms,
        core_logic_time_s,
        core_logic_time_ms,
    )

    return json.dumps(
        {
            "bottlenecks_found": bottlenecks_found,
            "inefficient_ops": inefficient_ops,
            "message": message,
        },
        indent=2,
    )

  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.exception("Error detecting layout mismatch copy operations")
    return json.dumps({"error": f"Internal error during detection: {repr(e)}"})
