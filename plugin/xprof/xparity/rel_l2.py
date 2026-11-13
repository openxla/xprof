"""Whole-output and per-row relative L2 error checks.

A ULP gate asks whether every element is within a few representable steps of
the reference. That is the right question for a kernel that should reproduce
its reference up to rounding, and the wrong one for a kernel whose arithmetic
is deliberately lossy: dequantizing FP8 or FP4 weights, accumulating in a
different order across split-K tiles, or rounding activations to a narrow
format. Those kernels differ from the reference by many ULP in individual
elements while being correct in aggregate, so the usual acceptance test is a
relative L2 error, |candidate - reference| / |reference|.

A single whole-output norm can hide a localized defect: one wrong row of a
large output barely moves it. `compare_row_rel_l2` therefore also bounds every
row (the last dimension) by `row_factor * tol`. Each row's error is taken
against that row's own norm, floored at `row_norm_floor` times the mean row
norm so that rows that are legitimately near zero cannot inflate the ratio.

Like `ulp`, this module imports only the standard library and NumPy, so that
standalone benchmark harnesses can carry its source inline. Inputs may be
NumPy arrays (including ml_dtypes types such as bfloat16, the float8 formats
and float4_e2m1fn), JAX arrays on any device, or PyTorch tensors on any
device.

Public API:
  compare_row_rel_l2: The combined whole-output and per-row check.
  as_host_array: Converts a NumPy, JAX or PyTorch array to a host NumPy array.
  check_parameters: Validates tol, row_factor and row_norm_floor.
  RowRelL2Comparison: The structured result.
  ROW_FACTOR, ROW_NORM_FLOOR: Default row bound and row-norm floor.
"""

import dataclasses
import math
from typing import Any

import numpy as np

# Every row may be off by at most this multiple of the whole-output tolerance.
ROW_FACTOR = 5.0
# A row's error is measured against its own norm, but not below this fraction
# of the mean row norm.
ROW_NORM_FLOOR = 0.1
# Guards the division when the reference (or every row of it) is all zeros.
_NORM_EPS = 1e-12

REASON_SHAPE_MISMATCH = "shape_mismatch"
REASON_DTYPE_MISMATCH = "dtype_mismatch"
REASON_NON_FINITE_CANDIDATE = "non_finite_candidate"
REASON_NON_FINITE_REFERENCE = "non_finite_reference"
REASON_WHOLE_OUTPUT = "whole_output_rel_l2"
REASON_ROW = "row_rel_l2"


@dataclasses.dataclass(frozen=True)
class RowRelL2Comparison:
  """Result of `compare_row_rel_l2`.

  Attributes:
    passed: True when the whole-output error is at most `tol` and every row's
      error is at most `row_factor * tol`.
    rel_l2: Whole-output relative L2 error. NaN when not computed.
    worst_row_rel_l2: Largest per-row relative L2 error. NaN when not computed.
    worst_row_index: Index of that row over the leading dimensions; `()` for 0-d
      and 1-d outputs, which are a single row.
    rows_over_bound: Number of rows whose error exceeds `row_factor * tol`.
    row_count: Number of rows.
    tol: Whole-output tolerance.
    row_factor: Row bound as a multiple of `tol`.
    row_norm_floor: Row-norm floor as a fraction of the mean row norm.
    reason: None when `passed`; otherwise one of the `REASON_*` constants.
    message: One-line human-readable verdict.
    candidate_dtype: Dtype name of the candidate before widening.
    reference_dtype: Dtype name of the reference before widening.
    first_non_finite_index: First NaN or Inf element of the failing operand.
  """

  passed: bool
  rel_l2: float
  worst_row_rel_l2: float
  worst_row_index: tuple[int, ...]
  rows_over_bound: int
  row_count: int
  tol: float
  row_factor: float
  row_norm_floor: float
  reason: str | None
  message: str
  candidate_dtype: str
  reference_dtype: str
  first_non_finite_index: tuple[int, ...] | None = None


def _is_torch_tensor(value: Any) -> bool:
  return type(value).__module__.startswith("torch") and hasattr(value, "detach")


def as_host_array(value: Any) -> tuple[np.ndarray, str]:
  """Returns `value` as a host NumPy array and its original dtype name.

  PyTorch tensors are detached and copied to the host. Tensor dtypes that
  NumPy cannot hold (bfloat16, the float8 formats) are widened to float32,
  which is exact for all of them; the returned name is the original dtype,
  without the `torch.` prefix, so it compares equal to the ml_dtypes name.
  Everything else, including JAX arrays on an accelerator, goes through
  `np.asarray`.

  Args:
    value: A NumPy array, JAX array, PyTorch tensor or array-like.

  Raises:
    TypeError: For PyTorch dtypes that cannot be widened element-wise, such as
      the packed `float4_e2m1fn_x2`.
  """
  if _is_torch_tensor(value):
    tensor = value.detach().cpu()
    name = str(tensor.dtype).removeprefix("torch.")
    try:
      return tensor.numpy(), name
    except TypeError:
      pass
    try:
      return tensor.float().numpy(), name
    except (RuntimeError, TypeError) as e:
      raise TypeError(
          f"Cannot convert a torch.{name} tensor to NumPy. Packed formats"
          " (for example float4_e2m1fn_x2) hold two values per element:"
          " unpack or dequantize the output before comparing it."
      ) from e
  arr = np.asarray(value)
  return arr, str(arr.dtype)


def _check_positive(name: str, value: float, allow_zero: bool = False) -> None:
  ok = math.isfinite(value) and (value >= 0 if allow_zero else value > 0)
  if not ok:
    bound = "non-negative" if allow_zero else "positive"
    raise ValueError(f"{name} must be finite and {bound}, got {value}.")


def check_parameters(
    tol: float,
    row_factor: float = ROW_FACTOR,
    row_norm_floor: float = ROW_NORM_FLOOR,
) -> None:
  """Raises ValueError unless the `compare_row_rel_l2` parameters are valid.

  Args:
    tol: Must be finite and positive.
    row_factor: Must be finite and positive.
    row_norm_floor: Must be finite and non-negative.
  """
  _check_positive("tol", tol)
  _check_positive("row_factor", row_factor)
  _check_positive("row_norm_floor", row_norm_floor, allow_zero=True)


def _first_non_finite(arr: np.ndarray) -> tuple[int, ...] | None:
  mask = ~np.isfinite(arr)
  if not mask.any():
    return None
  flat = int(np.argmax(mask))
  return tuple(int(i) for i in np.unravel_index(flat, arr.shape))


def compare_row_rel_l2(
    actual: Any,
    expected: Any,
    tol: float,
    row_factor: float = ROW_FACTOR,
    row_norm_floor: float = ROW_NORM_FLOOR,
    require_same_dtype: bool = True,
) -> RowRelL2Comparison:
  """Checks whole-output and per-row relative L2 error against `tol`.

  The candidate passes when all of these hold:
  - shapes match, and dtypes match unless `require_same_dtype` is False;
  - both operands are finite;
  - |actual - expected| / |expected| <= tol over the whole output;
  - for every row r (the last dimension), |actual_r - expected_r| /
    max(|expected_r|, row_norm_floor * mean_r |expected_r|) <= row_factor *
    tol.

  Norms are computed in float64 after widening each operand.

  Args:
    actual: Candidate output (NumPy, JAX or PyTorch).
    expected: Reference output with the same shape.
    tol: Whole-output relative L2 tolerance. Must be positive.
    row_factor: Row bound as a multiple of `tol`. Must be positive.
    row_norm_floor: Floor on each row's norm as a fraction of the mean row norm.
      Must be non-negative; 0 measures every row against its own norm.
    require_same_dtype: If True (default), a dtype mismatch fails the check.

  Returns:
    A RowRelL2Comparison. Shape mismatches are reported as a failure rather
    than raised.

  Raises:
    ValueError: If `tol`, `row_factor` or `row_norm_floor` is out of range.
  """
  check_parameters(tol, row_factor, row_norm_floor)
  act, act_dtype = as_host_array(actual)
  exp, exp_dtype = as_host_array(expected)
  nan = float("nan")
  params = dict(
      tol=float(tol),
      row_factor=float(row_factor),
      row_norm_floor=float(row_norm_floor),
      candidate_dtype=act_dtype,
      reference_dtype=exp_dtype,
  )

  def fail_early(reason: str, message: str, **extra: Any) -> RowRelL2Comparison:
    return RowRelL2Comparison(
        passed=False,
        rel_l2=nan,
        worst_row_rel_l2=nan,
        worst_row_index=(),
        rows_over_bound=0,
        row_count=0,
        reason=reason,
        message=f"FAILED: {message}",
        **params,
        **extra,
    )

  if act.shape != exp.shape:
    return fail_early(
        REASON_SHAPE_MISMATCH,
        f"shape mismatch: candidate {act.shape} vs reference {exp.shape}.",
    )
  if require_same_dtype and act_dtype != exp_dtype:
    return fail_early(
        REASON_DTYPE_MISMATCH,
        f"dtype mismatch: candidate {act_dtype} vs reference {exp_dtype}.",
    )

  act64 = act.astype(np.float64)
  exp64 = exp.astype(np.float64)
  for operand, arr, reason in (
      ("candidate", act64, REASON_NON_FINITE_CANDIDATE),
      ("reference", exp64, REASON_NON_FINITE_REFERENCE),
  ):
    index = _first_non_finite(arr)
    if index is not None:
      return fail_early(
          reason,
          f"{operand} contains NaN or Inf (first at index {index}).",
          first_non_finite_index=index,
      )

  if exp64.size == 0:
    return RowRelL2Comparison(
        passed=True,
        rel_l2=0.0,
        worst_row_rel_l2=0.0,
        worst_row_index=(),
        rows_over_bound=0,
        row_count=0,
        reason=None,
        message="PASSED: empty output.",
        **params,
    )

  diff = act64 - exp64
  rel_l2 = float(
      np.linalg.norm(diff) / max(float(np.linalg.norm(exp64)), _NORM_EPS)
  )

  row_len = exp64.shape[-1] if exp64.ndim else 1
  rows_diff = np.linalg.norm(diff.reshape(-1, row_len), axis=-1)
  rows_ref = np.linalg.norm(exp64.reshape(-1, row_len), axis=-1)
  floor = max(row_norm_floor * float(rows_ref.mean()), _NORM_EPS)
  row_err = rows_diff / np.maximum(rows_ref, floor)
  worst = int(np.argmax(row_err))
  worst_row_rel_l2 = float(row_err[worst])
  worst_row_index = (
      tuple(int(i) for i in np.unravel_index(worst, exp64.shape[:-1]))
      if exp64.ndim > 1
      else ()
  )
  row_bound = row_factor * tol
  rows_over_bound = int(np.count_nonzero(row_err > row_bound))

  if rel_l2 > tol:
    reason = REASON_WHOLE_OUTPUT
    message = (
        f"FAILED: relative L2 error {rel_l2:.3e} > tol {tol:g} (worst row"
        f" {worst_row_index} {worst_row_rel_l2:.3e})."
    )
  elif rows_over_bound:
    reason = REASON_ROW
    message = (
        f"FAILED: relative L2 error {rel_l2:.3e} <= tol {tol:g}, but row"
        f" {worst_row_index} is off by {worst_row_rel_l2:.3e} >"
        f" {row_factor:g} x {tol:g}; {rows_over_bound} of {row_err.size}"
        " rows over the row bound."
    )
  else:
    reason = None
    message = (
        f"PASSED: relative L2 error {rel_l2:.3e} <= tol {tol:g}; worst row"
        f" {worst_row_index} {worst_row_rel_l2:.3e} <= {row_factor:g} x"
        f" {tol:g}."
    )
  return RowRelL2Comparison(
      passed=reason is None,
      rel_l2=rel_l2,
      worst_row_rel_l2=worst_row_rel_l2,
      worst_row_index=worst_row_index,
      rows_over_bound=rows_over_bound,
      row_count=int(row_err.size),
      reason=reason,
      message=message,
      **params,
  )
