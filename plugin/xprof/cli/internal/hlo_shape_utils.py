"""Utilities for deriving FLOPs and bytes accessed from HLO shape expressions."""

import math
import re
from typing import Any

# Element byte widths for XLA primitive types.
DTYPE_BYTES: dict[str, int] = {
    "pred": 1,
    "s4": 1,
    "u4": 1,
    "s8": 1,
    "u8": 1,
    "f8e4m3fn": 1,
    "f8e4m3b11fnuz": 1,
    "f8e4m3fnuz": 1,
    "f8e5m2": 1,
    "f8e5m2fnuz": 1,
    "bf16": 2,
    "f16": 2,
    "s16": 2,
    "u16": 2,
    "f32": 4,
    "s32": 4,
    "u32": 4,
    "f64": 8,
    "c64": 8,
    "s64": 8,
    "u64": 8,
    "c128": 16,
}

# Matches shapes like "bf16[4,4096,2048]" or "f32[1024,1024]{1,0}".
_TENSOR_SHAPE_RE = re.compile(
    r"\b([a-z][a-z0-9_]*)\[([0-9,\s]+)\]", re.IGNORECASE
)


def parse_hlo_tensor_shapes(
    expression: str,
) -> tuple[tuple[str, list[int]] | None, list[tuple[str, list[int]]]]:
  """Extracts output shape and operand shapes from an HLO instruction string.

  Args:
    expression: The HLO instruction text (e.g., '%custom-call.1 =
      bf16[4,4096,4096] custom-call(bf16[4,4096,2048] %p0, bf16[2048,4096]
      %p1)').

  Returns:
    A tuple of (output_shape, operand_shapes), where each shape is a tuple of
    (dtype_str, dims_list). Returns (None, []) if no shapes can be parsed.
  """
  if not expression or "=" not in expression:
    return None, []

  lhs_rhs = expression.split("=", 1)
  rhs = lhs_rhs[1].strip()

  # Split at first '(' to separate output shape/opcode from operands.
  paren_idx = rhs.find("(")
  if paren_idx == -1:
    return None, []

  header_part = rhs[:paren_idx]
  operands_part = rhs[paren_idx:]

  out_matches = _TENSOR_SHAPE_RE.findall(header_part)
  if not out_matches:
    return None, []

  out_dtype, out_dims_str = out_matches[-1]
  out_dims = [
      int(d.strip()) for d in out_dims_str.split(",") if d.strip().isdigit()
  ]
  output_shape = (out_dtype.lower(), out_dims)

  operand_shapes: list[tuple[str, list[int]]] = []
  for dt, dims_str in _TENSOR_SHAPE_RE.findall(operands_part):
    dims = [int(d.strip()) for d in dims_str.split(",") if d.strip().isdigit()]
    if dims:
      operand_shapes.append((dt.lower(), dims))

  return output_shape, operand_shapes


def _try_match_contraction(
    out_dims: list[int],
    dims_a: list[int],
    dims_b: list[int],
) -> int | None:
  """Returns contraction dimension K if dims_a and dims_b form a matmul."""
  if len(out_dims) < 2 or len(dims_a) < 2 or len(dims_b) < 2:
    return None

  m_out, n_out = out_dims[-2], out_dims[-1]
  a_last2 = dims_a[-2:]
  b_last2 = dims_b[-2:]

  # Standard ordering: A has m_out, B has n_out
  if m_out in a_last2 and n_out in b_last2:
    k_a = a_last2[1] if a_last2[0] == m_out else a_last2[0]
    k_b = b_last2[0] if b_last2[1] == n_out else b_last2[1]
    if k_a == k_b and k_a > 0:
      return k_a

  # Transposed ordering: A has n_out, B has m_out
  if n_out in a_last2 and m_out in b_last2:
    k_a = a_last2[1] if a_last2[0] == n_out else a_last2[0]
    k_b = b_last2[0] if b_last2[1] == m_out else b_last2[1]
    if k_a == k_b and k_a > 0:
      return k_a

  return None


def derive_custom_call_flops_and_bytes(
    expression: str,
    category: str = "",
    name: str = "",
) -> tuple[float | None, float | None, str]:
  """Derives FLOPs and bytes accessed from tensor shapes for custom calls.

  For custom-call contraction/matmul operations (e.g. Pallas or Mosaic kernels),
  XLA's static cost model emits 0 FLOPs. This function parses the instruction's
  output and operand tensor shapes and derives 2 * B * M * N * K FLOPs and total
  tensor bytes accessed when the shapes match a matrix contraction.

  Args:
    expression: The HLO expression string containing tensor shapes.
    category: The HLO category string (e.g. 'custom-call').
    name: The HLO operation name (e.g. '%custom-call.1').

  Returns:
    A tuple of (flops, bytes_accessed, provenance), where provenance is one of:
      - 'derived_from_shapes': Successfully derived contraction FLOPs/bytes.
      - 'opaque_custom_call': Operation is a custom call with unresolvable
        FLOPs.
      - 'xla_cost_model': Standard XLA operation (not a custom call).
  """
  cat_lower = (category or "").lower()
  name_lower = (name or "").lower()
  expr_lower = (expression or "").lower()
  is_custom = (
      "custom-call" in cat_lower
      or "custom_call" in cat_lower
      or "custom-call" in name_lower
      or "custom_call" in name_lower
      or "custom-call" in expr_lower
      or "custom_call_target" in expr_lower
  )

  if not is_custom:
    return None, None, "xla_cost_model"

  output_shape, operand_shapes = parse_hlo_tensor_shapes(expression)
  if output_shape is not None and len(operand_shapes) >= 2:
    _, out_dims = output_shape
    if len(out_dims) >= 2:
      # Check pairs of operands (typically the first two tensor inputs)
      for i in range(min(len(operand_shapes), 3)):
        for j in range(i + 1, min(len(operand_shapes), 4)):
          _, dims_a = operand_shapes[i]
          _, dims_b = operand_shapes[j]
          k_dim = _try_match_contraction(out_dims, dims_a, dims_b)
          if k_dim is not None:
            batch_elems = math.prod(out_dims[:-2]) if len(out_dims) > 2 else 1
            m_out, n_out = out_dims[-2], out_dims[-1]
            flops = float(2 * batch_elems * m_out * n_out * k_dim)

            all_shapes = [output_shape, operand_shapes[i], operand_shapes[j]]
            total_bytes = 0.0
            for dt, dims in all_shapes:
              elem_bytes = DTYPE_BYTES.get(dt, 2)
              total_bytes += float(elem_bytes * math.prod(dims))
            return flops, total_bytes, "derived_from_shapes"

  if is_custom:
    return None, None, "opaque_custom_call"
  return None, None, "xla_cost_model"


def extract_expressions_from_op_profile_node(
    node: Any, expressions_by_name: dict[str, str]
) -> None:
  """Recursively populates op name -> HLO expression map from an op_profile."""
  if hasattr(node, "name") and node.name:
    if hasattr(node, "xla") and getattr(node.xla, "expression", ""):
      expr = node.xla.expression
      expressions_by_name[node.name] = expr
      expressions_by_name[node.name.lstrip("%")] = expr
  if hasattr(node, "children"):
    for child in node.children:
      extract_expressions_from_op_profile_node(child, expressions_by_name)
