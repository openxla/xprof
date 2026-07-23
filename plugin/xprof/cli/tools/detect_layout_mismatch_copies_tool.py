"""MCP tool to detect layout mismatch copy operations causing HBM overhead."""

import collections
from collections.abc import Mapping, Sequence
import functools
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

try:
  from tensorflow.compiler.xla.python import xla_client  # pylint: disable=g-import-not-at-top
except ImportError:
  xla_client = None

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


def is_compute_custom_call(
    instr: Any, proto_instr_by_id: Mapping[int, Any] | None = None
) -> bool:
  """Determines if a custom-call HLO instruction is compute-intensive.

  Args:
    instr: The CustomCall instruction proto to check.
    proto_instr_by_id: Optional fallback mapping to protobuf objects.

  Returns:
    True if the custom call matches a heavy compute kernel, False otherwise.
  """
  target = getattr(instr, "custom_call_target", "")
  if callable(target):
    target = target()
  if not target and proto_instr_by_id is not None:
    proto_instr = proto_instr_by_id.get(_get_id(instr))
    if proto_instr:
      target = getattr(proto_instr, "custom_call_target", "")
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


def _get_id(obj):
  if isinstance(obj, (int, str)):
    return obj
  if hasattr(obj, "id"):
    return obj.id
  name = getattr(obj, "name", "")
  return name() if callable(name) else name


def _get_opcode_lower(instr: Any) -> str:
  """Gets the lowercased opcode string for an HLO instruction."""
  op = getattr(instr, "opcode_name", instr.opcode)
  if callable(op):
    op = op()
  if isinstance(op, str):
    res = op.lower()
  elif hasattr(op, "name"):
    name_val = op.name
    res = (name_val() if callable(name_val) else name_val).lower()
  else:
    res = str(op).lower()
  if res.startswith("k"):
    # C++ enum values often start with 'k' like kCopy, kFusion, etc.
    res = res[1:]

  # C++ enums don't have hyphens, so kCustomCall -> customcall
  # But protobuf has 'custom-call'
  # Actually, we can just replace hyphens and compare without hyphens,
  # or map them.
  # Let's map a few common ones
  mapping = {
      "customcall": "custom-call",
      "gettupleelement": "get-tuple-element",
  }
  return mapping.get(res, res)


def _get_called_comp_ids(
    instr: Any, comp_id_by_name: Mapping[str, int]
) -> list[int]:
  """Gets the IDs of computations called by this instruction."""
  if hasattr(instr, "called_computation_ids"):
    called_ids = instr.called_computation_ids
    return list(called_ids() if callable(called_ids) else called_ids)

  called_names = getattr(instr, "called_computation_names", None)
  if called_names is not None:
    names = called_names() if callable(called_names) else called_names
    return [comp_id_by_name[n] for n in names if n in comp_id_by_name]

  called_comps = getattr(instr, "called_computations", None)
  if called_comps is not None:
    comps = called_comps() if callable(called_comps) else called_comps
    res = []
    for c in comps:
      cid = _get_id(c)
      if isinstance(cid, int):
        res.append(cid)
      elif cid in comp_id_by_name:
        res.append(comp_id_by_name[cid])
    return res

  return []


def is_compute_stage(
    instr: Any,
    comp_by_id: Mapping[int, Any],
    comp_id_by_name: Mapping[str, int],
    visited_fusions: set[int] | None = None,
    memo: dict[int, bool] | None = None,
    proto_instr_by_id: Mapping[int, Any] | None = None,
) -> bool:
  """Checks if an HLO instruction is a compute-intensive stage.

  Args:
    instr: The HLO instruction proto to check.
    comp_by_id: A mapping from computation IDs to computation protos.
    comp_id_by_name: A mapping from computation names to IDs.
    visited_fusions: A set of fusion instruction IDs already visited.
    memo: Optional dictionary to cache whether an instruction is a compute
      stage.
    proto_instr_by_id: Optional fallback mapping to protobuf objects.

  Returns:
    True if the instruction is a compute-intensive stage, False otherwise.
  """
  if memo is not None and _get_id(instr) in memo:
    return memo[_get_id(instr)]

  if visited_fusions is None:
    visited_fusions = set()
  opcode_lower = _get_opcode_lower(instr)
  result = False

  if any(keyword in opcode_lower for keyword in _COMPUTE_KEYWORDS):
    result = True
  elif opcode_lower == "custom-call":
    result = is_compute_custom_call(instr, proto_instr_by_id)
  elif opcode_lower == "fusion":
    if _get_id(instr) in visited_fusions:
      return False
    visited_fusions.add(_get_id(instr))

    called_names = getattr(instr, "called_computation_names", None)
    if called_names is not None:
      called_comp_names = (
          called_names() if callable(called_names) else called_names
      )
    else:
      called_comp_names = [
          comp_by_id[cid].name
          for cid in getattr(instr, "called_computation_ids", [])
          if cid in comp_by_id
      ]

    for comp_name in called_comp_names:
      comp_id = comp_id_by_name.get(comp_name)
      if comp_id is None:
        continue
      comp = comp_by_id.get(comp_id)
      if comp:
        inner_instrs = (
            comp.instructions()
            if callable(getattr(comp, "instructions", None))
            else comp.instructions
        )
        for inner_instr in inner_instrs:
          inner_op = _get_opcode_lower(inner_instr)
          if any(keyword in inner_op for keyword in _COMPUTE_KEYWORDS):
            result = True
            break
          if inner_op == "custom-call" and is_compute_custom_call(
              inner_instr, proto_instr_by_id
          ):
            result = True
            break
          if inner_op == "fusion":
            if is_compute_stage(
                inner_instr,
                comp_by_id,
                comp_id_by_name,
                visited_fusions,
                memo,
                proto_instr_by_id,
            ):
              result = True
              break
        if result:
          break

  if memo is not None:
    memo[_get_id(instr)] = result
  return result


def find_upstream_compute_stages(
    copy_instr_id: int,
    instr_by_id: Mapping[int, Any],
    comp_by_id: Mapping[int, Any],
    comp_id_by_name: Mapping[str, int],
    comp_id_by_instr_id: Mapping[int, int],
    callers_by_comp_id: Mapping[int, Sequence[int]],
    max_depth: int = 5,
    memo: dict[int, bool] | None = None,
    proto_instr_by_id: Mapping[int, Any] | None = None,
) -> Sequence[tuple[Any, int]]:
  """Finds compute-intensive producers upstream from a copy instruction.

  Traverses backward, tracking data and control flow dependency boundaries.

  Args:
    copy_instr_id: The instruction ID of the starting HLO Copy operation.
    instr_by_id: A mapping from HLO instruction IDs to instruction protos.
    comp_by_id: A mapping from computation IDs to computation protos.
    comp_id_by_name: A mapping from computation names to IDs.
    comp_id_by_instr_id: A mapping from instruction IDs to their computation ID.
    callers_by_comp_id: A mapping from computation IDs to their caller
      instruction IDs.
    max_depth: The maximum depth of the dataflow graph traversal (in number of
      hops).
    memo: Optional dictionary to cache whether an instruction is a compute
      stage.
    proto_instr_by_id: Optional fallback mapping to protobuf objects.

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
      if is_compute_stage(
          curr_instr,
          comp_by_id,
          comp_id_by_name,
          memo=memo,
          proto_instr_by_id=proto_instr_by_id,
      ):
        upstream_producers.append((curr_instr, dist))
        continue
      opcode_lower = _get_opcode_lower(curr_instr)
      if opcode_lower == "constant":
        upstream_producers.append((curr_instr, dist))
        continue

      curr_comp_id = comp_id_by_instr_id.get(curr_id)
      if opcode_lower == "parameter" and (
          curr_comp_id not in callers_by_comp_id
      ):
        upstream_producers.append((curr_instr, dist))
        continue

    if dist < max_depth:
      opcode = _get_opcode_lower(curr_instr)

      if opcode == "parameter":
        curr_comp_id = comp_id_by_instr_id.get(curr_id)
        callers = callers_by_comp_id.get(curr_comp_id, [])
        param_num = getattr(curr_instr, "parameter_number", 0)
        for caller_id in callers:
          caller = instr_by_id.get(caller_id)
          if not caller:
            continue
          caller_opcode = _get_opcode_lower(caller)
          caller_operands = (
              caller.operands()
              if callable(getattr(caller, "operands", None))
              else getattr(caller, "operand_ids", [])
          )
          if caller_opcode == "conditional":
            for branch_idx, comp_id in enumerate(
                _get_called_comp_ids(caller, comp_id_by_name)
            ):
              if comp_id == curr_comp_id:
                if branch_idx + 1 < len(caller_operands):
                  target_id = _get_id(caller_operands[branch_idx + 1])
                  if (target_id, shape_idx) not in visited:
                    visited.add((target_id, shape_idx))
                    queue.append((target_id, shape_idx, dist + 1))
          elif caller_opcode in ("fusion", "call"):
            if param_num < len(caller_operands):
              target_id = _get_id(caller_operands[param_num])
              if (target_id, shape_idx) not in visited:
                visited.add((target_id, shape_idx))
                queue.append((target_id, shape_idx, dist + 1))
          elif caller_opcode == "while":
            if param_num < len(caller_operands):
              init_id = _get_id(caller_operands[param_num])
              if (init_id, shape_idx) not in visited:
                visited.add((init_id, shape_idx))
                queue.append((init_id, shape_idx, dist + 1))
            body_comp_id = _get_called_comp_ids(caller, comp_id_by_name)[1]
            body_comp = comp_by_id.get(body_comp_id)
            if body_comp and body_comp.root_id:
              if (body_comp.root_id, shape_idx) not in visited:
                visited.add((body_comp.root_id, shape_idx))
                queue.append((body_comp.root_id, shape_idx, dist + 1))

      elif opcode == "get-tuple-element":
        idx = getattr(curr_instr, "tuple_index", 0)
        parent_idx = (idx,) + shape_idx
        operands = (
            curr_instr.operands()
            if callable(getattr(curr_instr, "operands", None))
            else getattr(curr_instr, "operand_ids", [])
        )
        if operands:
          target_id = _get_id(operands[0])
          if (target_id, parent_idx) not in visited:
            visited.add((target_id, parent_idx))
            queue.append((target_id, parent_idx, dist + 1))

      elif opcode == "tuple":
        operands = (
            curr_instr.operands()
            if callable(getattr(curr_instr, "operands", None))
            else getattr(curr_instr, "operand_ids", [])
        )
        if shape_idx:
          idx = shape_idx[0]
          remaining_idx = shape_idx[1:]
          if idx < len(operands):
            target_id = _get_id(operands[idx])
            if (target_id, remaining_idx) not in visited:
              visited.add((target_id, remaining_idx))
              queue.append((target_id, remaining_idx, dist + 1))
        else:
          for op in operands:
            target_id = _get_id(op)
            if (target_id, ()) not in visited:
              visited.add((target_id, ()))
              queue.append((target_id, (), dist + 1))

      elif opcode == "while":
        operands = (
            curr_instr.operands()
            if callable(getattr(curr_instr, "operands", None))
            else getattr(curr_instr, "operand_ids", [])
        )
        if operands:
          init_id = _get_id(operands[0])
          if (init_id, shape_idx) not in visited:
            visited.add((init_id, shape_idx))
            queue.append((init_id, shape_idx, dist + 1))
        if len(_get_called_comp_ids(curr_instr, comp_id_by_name)) > 1:
          body_comp_id = _get_called_comp_ids(curr_instr, comp_id_by_name)[1]
          body_comp = comp_by_id.get(body_comp_id)
          if body_comp and body_comp.root_id:
            if (body_comp.root_id, shape_idx) not in visited:
              visited.add((body_comp.root_id, shape_idx))
              queue.append((body_comp.root_id, shape_idx, dist + 1))

      elif opcode == "call":
        if _get_called_comp_ids(curr_instr, comp_id_by_name):
          called_comp_id = _get_called_comp_ids(curr_instr, comp_id_by_name)[0]
          called_comp = comp_by_id.get(called_comp_id)
          if called_comp and called_comp.root_id:
            if (called_comp.root_id, shape_idx) not in visited:
              visited.add((called_comp.root_id, shape_idx))
              queue.append((called_comp.root_id, shape_idx, dist + 1))

      else:
        operands = (
            curr_instr.operands()
            if callable(getattr(curr_instr, "operands", None))
            else getattr(curr_instr, "operand_ids", [])
        )
        for op in operands:
          op_id = _get_id(op)
          if (op_id, shape_idx) not in visited and op_id in instr_by_id:
            visited.add((op_id, shape_idx))
            queue.append((op_id, shape_idx, dist + 1))

  return upstream_producers


def find_downstream_compute_stages(
    copy_instr_id: int,
    instr_by_id: Mapping[int, Any],
    users_by_id: Mapping[int, Sequence[int]],
    comp_by_id: Mapping[int, Any],
    comp_id_by_name: Mapping[str, int],
    comp_id_by_instr_id: Mapping[int, int],
    callers_by_comp_id: Mapping[int, Sequence[int]],
    root_id_by_comp_id: Mapping[int, int],
    max_depth: int = 5,
    memo: dict[int, bool] | None = None,
    proto_instr_by_id: Mapping[int, Any] | None = None,
) -> Sequence[tuple[Any, int]]:
  """Finds compute-intensive consumers downstream from a copy instruction.

  Traverses forward, tracking data and control flow dependency boundaries.

  Args:
    copy_instr_id: The instruction ID of the starting HLO Copy operation.
    instr_by_id: A mapping from HLO instruction IDs to instruction protos.
    users_by_id: A mapping from instruction IDs to their user instruction IDs.
    comp_by_id: A mapping from computation IDs to computation protos.
    comp_id_by_name: A mapping from computation names to computation IDs.
    comp_id_by_instr_id: A mapping from instruction IDs to their computation ID.
    callers_by_comp_id: A mapping from computation IDs to their caller
      instruction IDs.
    root_id_by_comp_id: A mapping from computation IDs to their root instruction
      ID.
    max_depth: The maximum depth of the dataflow graph traversal (in number of
      hops).
    memo: Optional dictionary to cache whether an instruction is a compute
      stage.
    proto_instr_by_id: Optional fallback mapping to protobuf objects.

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
      if is_compute_stage(
          curr_instr,
          comp_by_id,
          comp_id_by_name,
          memo=memo,
          proto_instr_by_id=proto_instr_by_id,
      ):
        compute_consumers.append((curr_instr, dist))
        continue

    if dist < max_depth:
      opcode = _get_opcode_lower(curr_instr)
      curr_comp_id = comp_id_by_instr_id.get(curr_id)
      root_id = root_id_by_comp_id.get(curr_comp_id)

      if curr_id == root_id:
        callers = callers_by_comp_id.get(curr_comp_id, [])
        for caller_id in callers:
          caller = instr_by_id.get(caller_id)
          if not caller:
            continue
          caller_opcode = _get_opcode_lower(caller)

          if caller_opcode == "while":
            if _get_called_comp_ids(caller, comp_id_by_name):
              if len(_get_called_comp_ids(caller, comp_id_by_name)) > 1:
                body_comp_id = _get_called_comp_ids(caller, comp_id_by_name)[1]
              else:
                body_comp_id = _get_called_comp_ids(caller, comp_id_by_name)[0]
              body_comp = comp_by_id.get(body_comp_id)
              if body_comp:
                for inner_i in body_comp.instructions():
                  if (
                      getattr(inner_i, "opcode_name", inner_i.opcode).lower()
                      == "parameter"
                      and inner_i.parameter_number == 0
                  ):
                    if (_get_id(inner_i), shape_idx) not in visited:
                      visited.add((_get_id(inner_i), shape_idx))
                      queue.append((_get_id(inner_i), shape_idx, dist + 1))

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
          if _get_opcode_lower(user_instr) == "get-tuple-element":
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
        if len(_get_called_comp_ids(curr_instr, comp_id_by_name)) > 1:
          body_comp_id = _get_called_comp_ids(curr_instr, comp_id_by_name)[1]
          body_comp = comp_by_id.get(body_comp_id)
          if body_comp:
            for inner_i in body_comp.instructions():
              if (
                  _get_opcode_lower(inner_i) == "parameter"
                  and inner_i.parameter_number == 0
              ):
                if (_get_id(inner_i), shape_idx) not in visited:
                  visited.add((_get_id(inner_i), shape_idx))
                  queue.append((_get_id(inner_i), shape_idx, dist + 1))

        for user_id in users_by_id.get(curr_id, []):
          if (user_id, shape_idx) not in visited and user_id in instr_by_id:
            visited.add((user_id, shape_idx))
            queue.append((user_id, shape_idx, dist + 1))

      elif opcode == "call":
        if _get_called_comp_ids(curr_instr, comp_id_by_name):
          called_comp_id = _get_called_comp_ids(curr_instr, comp_id_by_name)[0]
          called_comp = comp_by_id.get(called_comp_id)
          if called_comp:
            for inner_i in called_comp.instructions():
              if (
                  _get_opcode_lower(inner_i) == "parameter"
                  and inner_i.parameter_number == 0
              ):
                if (_get_id(inner_i), shape_idx) not in visited:
                  visited.add((_get_id(inner_i), shape_idx))
                  queue.append((_get_id(inner_i), shape_idx, dist + 1))

      else:
        for user_id in users_by_id.get(curr_id, []):
          user_instr = instr_by_id.get(user_id)
          if not user_instr:
            continue
          if _get_opcode_lower(user_instr) == "tuple":
            if callable(getattr(user_instr, "operands", None)):
              operands = [
                  _get_id(op) for op in getattr(user_instr, "operands")()
              ]
            else:
              operands = getattr(user_instr, "operand_ids", [])
            for operand_idx, operand_id in enumerate(operands):
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


@functools.lru_cache(maxsize=16)
def _get_c_modules(session_id: str) -> list[Any]:
  """Fetches and caches the C++ HloModules to avoid repeated serialization."""
  debug_info = hlo_tools._fetch_debug_info(session_id)  # pylint: disable=protected-access
  c_modules = []
  for hlo_proto in debug_info.hlo_proto:
    if xla_client is None:
      c_modules.append(None)
      continue
    try:
      c_modules.append(
          xla_client.hlo.HloModule.from_serialized_hlo_module_proto(  # type: ignore
              hlo_proto.hlo_module.SerializeToString()
          )
      )
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.warning(
          "Failed to parse C++ HloModule, falling back to Python: %s", e
      )
      c_modules.append(None)
  return c_modules


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

    fetch_time_start = time.time()
    debug_info = hlo_tools._fetch_debug_info(session_id)  # pylint: disable=protected-access
    fetch_time_end = time.time()
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

    dict_build_time_total = 0.0
    bfs_time_total = 0.0
    copy_count = 0

    load_start = time.time()
    c_modules = _get_c_modules(session_id)
    load_time_s = time.time() - load_start

    for idx, hlo_proto in enumerate(debug_info.hlo_proto):
      dict_build_start = time.time()
      module_proto = hlo_proto.hlo_module

      instr_by_id = {}
      comp_by_id = {}
      users_by_id = collections.defaultdict(list)

      comp_name_by_id = {}
      comp_id_by_name = {}
      root_id_by_comp_id = {}

      for comp in module_proto.computations:
        comp_by_id[comp.id] = comp
        comp_name_by_id[comp.id] = comp.name
        comp_id_by_name[comp.name] = comp.id
        root_id_by_comp_id[comp.id] = comp.root_id

      instr_id_to_comp_id = {}
      callers_by_comp_id = collections.defaultdict(list)
      comp_id_by_instr_id = {}
      compute_stage_memo = {}

      # Use C++ xla_client to bypass lazy Python Protobuf instantiation
      # over massive graphs
      c_module = c_modules[idx] if idx < len(c_modules) else None

      proto_instr_by_id = {}
      for comp in module_proto.computations:
        for instr in comp.instructions:
          proto_instr_by_id[_get_id(instr)] = instr
          if instr.name:
            proto_instr_by_id[instr.name] = instr
      if c_module:
        c_computations = (
            c_module.computations()
            if callable(getattr(c_module, "computations", None))
            else c_module.computations
        )
        for c_comp in c_computations:
          comp_id = comp_id_by_name.get(
              c_comp.name, getattr(c_comp, "id", c_comp.name)
          )
          comp_by_id[comp_id] = c_comp  # Overwrite with fast C++ object
          c_instructions = (
              c_comp.instructions()
              if callable(getattr(c_comp, "instructions", None))
              else c_comp.instructions
          )
          for c_instr in c_instructions:
            c_instr_name = (
                c_instr.name()
                if callable(getattr(c_instr, "name", None))
                else c_instr.name
            )
            instr_id = getattr(c_instr, "id", c_instr_name)
            instr_id_to_comp_id[instr_id] = comp_id
            comp_id_by_instr_id[instr_id] = comp_id
            instr_by_id[instr_id] = c_instr
            c_called_names = (
                c_instr.called_computation_names()
                if callable(getattr(c_instr, "called_computation_names", None))
                else getattr(c_instr, "called_computation_names", [])
            )
            for comp_name in c_called_names:
              called_comp_id = comp_id_by_name.get(comp_name)
              if called_comp_id is not None:
                callers_by_comp_id[called_comp_id].append(instr_id)
            c_users = (
                c_instr.users()
                if callable(getattr(c_instr, "users", None))
                else getattr(c_instr, "users", [])
            )
            for u in c_users:
              u_name = (
                  u.name() if callable(getattr(u, "name", None)) else u.name
              )
              users_by_id[instr_id].append(getattr(u, "id", u_name))
      else:
        for comp in module_proto.computations:
          comp_by_id[comp.id] = comp
          for instr in comp.instructions:
            instr_id = _get_id(instr)
            instr_id_to_comp_id[instr_id] = comp.id
            comp_id_by_instr_id[instr_id] = comp.id
            instr_by_id[instr_id] = instr
            for called_id in getattr(instr, "called_computation_ids", []):
              callers_by_comp_id[called_id].append(instr_id)
            for op_id in getattr(instr, "operand_ids", []):
              users_by_id[op_id].append(instr_id)

      dict_build_time_total += time.time() - dict_build_start

      for instr in instr_by_id.values():
        opcode = _get_opcode_lower(instr)
        if opcode != "copy":
          continue

        if callable(getattr(instr, "operands", None)):
          operands = getattr(instr, "operands")()
        else:
          operands = getattr(instr, "operand_ids", [])

        if not operands:
          continue

        operand_id = (
            _get_id(operands[0])
            if hasattr(operands[0], "id") or hasattr(operands[0], "name")
            else operands[0]
        )
        operand_instr = instr_by_id.get(operand_id)
        if not operand_instr:
          continue

        copy_count += 1
        bfs_start = time.time()
        upstream_producers = find_upstream_compute_stages(
            _get_id(instr),
            instr_by_id,
            comp_by_id,
            comp_id_by_name,
            comp_id_by_instr_id,
            callers_by_comp_id,
            max_depth=5,
            memo=compute_stage_memo,
            proto_instr_by_id=proto_instr_by_id,
        )
        downstream_stages = find_downstream_compute_stages(
            _get_id(instr),
            instr_by_id,
            users_by_id,
            comp_by_id,
            comp_id_by_name,
            comp_id_by_instr_id,
            callers_by_comp_id,
            root_id_by_comp_id,
            max_depth=5,
            memo=compute_stage_memo,
            proto_instr_by_id=proto_instr_by_id,
        )
        bfs_time_total += time.time() - bfs_start

        if upstream_producers and downstream_stages:
          proto_instr = proto_instr_by_id.get(_get_id(instr))
          proto_operand = proto_instr_by_id.get(operand_id)

          if not proto_instr or not proto_operand:
            continue

          source_shape = proto_operand.shape
          target_shape = proto_instr.shape

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
              f"'{u.name}' ({_get_opcode_lower(u)}, dist={d})"
              for u, d in upstream_producers
          )
          downstream_names_str = ", ".join(
              f"'{d.name}' ({_get_opcode_lower(d)}, dist={dist})"
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
          instr_comp_id = instr_id_to_comp_id.get(_get_id(instr))
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
                  UpstreamProducer(
                      name=u.name,
                      opcode=_get_opcode_lower(u),
                      distance=d,
                  )
                  for u, d in upstream_producers
              ],
              downstream_stages=[
                  DownstreamStage(
                      name=d.name,
                      opcode=_get_opcode_lower(d),
                      distance=dist,
                  )
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
    core_logic_time_s = core_logic_end_time - core_logic_start_time
    total_end_time = time.time()
    total_time_s = total_end_time - total_start_time

    logging.info(
        "Layout mismatch copy detection metrics - Session ID: %s, "
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
            "bottlenecks_found": bottlenecks_found,
            "inefficient_ops": inefficient_ops,
            "message": message,
        },
        indent=2,
    )

  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.exception("Error detecting layout mismatch copy operations")
    return json.dumps({"error": f"Internal error during detection: {repr(e)}"})
