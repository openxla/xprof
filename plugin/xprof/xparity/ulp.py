"""Bitwise ULP distance and per-dtype ULP contracts.

This module is the dependency-light core of xparity: it imports only the
standard library, NumPy and ml_dtypes, and no other xparity or google3 module.
Benchmark harnesses that run where google3 cannot be imported (for example the
standalone scripts KernelBench renders for Minibench and JaxFu) can therefore
carry this module's source inline and measure outputs in exactly the units and
against exactly the contracts that `numerical_validator` uses.

Public API:
  compute_ulp_distance: Elementwise integer ULP distance between two arrays.
  bitwise_mismatch_mask: Elements whose raw bit patterns differ.
  tolerance_ratio: Worst error as a multiple of an atol/rtol allowance.
  get_contract: (recommended, hard ceiling) ULP contract for a dtype.
  resolve_canonical_dtype: Normalizes any accepted dtype spelling.
  is_discrete_dtype: True for integer and boolean dtypes.
  get_finfo: finfo for NumPy and ml_dtypes floating types.
  RECOMMENDED_CONTRACT_ULP, MAX_HARD_CEILING_ULP, INTEGER_DTYPES,
  DTYPE_TOLERANCES: The tables behind the functions above.
"""

import functools
import logging
import types
from typing import Any

import ml_dtypes
import numpy as np


def get_finfo(dtype: Any) -> Any:
  """Returns finfo object supporting both NumPy and ml_dtypes floating types."""
  np_dtype = np.dtype(dtype)
  try:
    return np.finfo(np_dtype)
  except (ValueError, TypeError) as e:
    logging.debug("np.finfo failed for dtype %s: %s", np_dtype, e)
    return ml_dtypes.finfo(np_dtype)


def _narrow_bits(arr: np.ndarray, target: Any, view_dtype: Any) -> np.ndarray:
  """Reinterprets `arr` as `target`, avoiding a cast when already that type."""
  if arr.dtype == target:
    return arr.view(view_dtype).astype(np.int64)
  return arr.astype(target).view(view_dtype).astype(np.int64)


# Canonical dtype -> (ml_dtypes/numpy scalar type, unsigned view type,
# sign bit mask, magnitude mask). Keyed on canonical names only; callers must
# route through `resolve_canonical_dtype` first.
_ULP_BIT_LAYOUT: types.MappingProxyType[str, tuple[Any, Any, int, int]] = (
    types.MappingProxyType({
        "float32": (np.float32, np.uint32, 0x80000000, 0x7FFFFFFF),
        "float16": (np.float16, np.uint16, 0x8000, 0x7FFF),
        "bfloat16": (ml_dtypes.bfloat16, np.uint16, 0x8000, 0x7FFF),
        "float8_e4m3fn": (ml_dtypes.float8_e4m3fn, np.uint8, 0x80, 0x7F),
        "float8_e5m2": (ml_dtypes.float8_e5m2, np.uint8, 0x80, 0x7F),
    })
)


def _sign_magnitude_to_continuous_int(
    arr: np.ndarray, dtype_str: str
) -> np.ndarray:
  """Converts floating-point sign-magnitude bits to continuous int64 index.

  Args:
    arr: The tensor to reinterpret.
    dtype_str: A canonical or aliased dtype name. Aliases (``fp8_e4m3``,
      ``bf16``, ``ml_dtypes.float8_e4m3fn``, ...) are resolved before lookup, so
      every spelling accepted by `resolve_canonical_dtype` works here.

  Returns:
    An int64 array whose values are ordered consistently with the real line,
    so that a difference of 1 corresponds to one representable step.
  """
  canonical = resolve_canonical_dtype(dtype_str)

  if canonical == "float64":
    raw = arr.astype(np.float64).view(np.uint64)
    sign_mask_u64 = np.uint64(0x8000000000000000)
    mag_mask_u64 = np.uint64(0x7FFFFFFFFFFFFFFF)
    is_negative = (raw & sign_mask_u64) != 0
    magnitude = (raw & mag_mask_u64).astype(np.int64)
    return np.where(is_negative, -magnitude, magnitude)

  layout = _ULP_BIT_LAYOUT.get(canonical)
  if layout is None:
    raise ValueError(
        f"Unsupported dtype for ULP conversion: '{dtype_str}' (resolved to"
        f" '{canonical}'). Supported floating-point dtypes:"
        f" {sorted(list(_ULP_BIT_LAYOUT) + ['float64'])}."
    )

  target, view_dtype, sign_mask, mag_mask = layout
  raw = _narrow_bits(arr, target, view_dtype)
  is_negative = (raw & sign_mask) != 0
  magnitude = raw & mag_mask
  return np.where(is_negative, -magnitude, magnitude)


INTEGER_DTYPES: frozenset[str] = frozenset({
    "int32",
    "int64",
    "int16",
    "int8",
    "uint32",
    "uint64",
    "uint16",
    "uint8",
})

RECOMMENDED_CONTRACT_ULP: types.MappingProxyType[str, int] = (
    types.MappingProxyType({
        "bool": 0,
        "int8": 0,
        "int16": 0,
        "int32": 0,
        "int64": 0,
        "uint8": 0,
        "uint16": 0,
        "uint32": 0,
        "uint64": 0,
        "float4_e2m1fn": 0,
        "float8_e4m3fn": 1,
        "float8_e5m2": 1,
        "bfloat16": 2,
        "float16": 2,
        "float32": 2,
        "float64": 1,
    })
)

MAX_HARD_CEILING_ULP: types.MappingProxyType[str, int] = (
    types.MappingProxyType({
        "bool": 0,
        "int8": 0,
        "int16": 0,
        "int32": 0,
        "int64": 0,
        "uint8": 0,
        "uint16": 0,
        "uint32": 0,
        "uint64": 0,
        "float4_e2m1fn": 0,
        "float8_e4m3fn": 2,
        "float8_e5m2": 2,
        "bfloat16": 8,
        "float16": 8,
        "float32": 4,
        "float64": 4,
    })
)

DTYPE_TOLERANCES: types.MappingProxyType[str, tuple[float, float]] = (
    types.MappingProxyType({
        "float32": (float(np.finfo(np.float32).eps), 1e-6),
        "bfloat16": (float(ml_dtypes.finfo(ml_dtypes.bfloat16).eps), 1e-3),
        "float16": (float(np.finfo(np.float16).eps), 1e-4),
        "float8_e4m3fn": (
            float(ml_dtypes.finfo(ml_dtypes.float8_e4m3fn).eps),
            0.05,
        ),
        "float8_e5m2": (
            float(ml_dtypes.finfo(ml_dtypes.float8_e5m2).eps),
            0.05,
        ),
        "float64": (float(np.finfo(np.float64).eps), 1e-12),
    })
)


_DTYPE_ALIASES: types.MappingProxyType[str, str] = types.MappingProxyType({
    "fp8_e4m3": "float8_e4m3fn",
    "float8_e4m3": "float8_e4m3fn",
    "float8_e4m3fn": "float8_e4m3fn",
    "fp8_e5m2": "float8_e5m2",
    "float8_e5m2": "float8_e5m2",
    "fp4_e2m1": "float4_e2m1fn",
    "float4_e2m1": "float4_e2m1fn",
    "bf16": "bfloat16",
    "fp16": "float16",
    "half": "float16",
    "fp32": "float32",
    "single": "float32",
    "fp64": "float64",
    "double": "float64",
})


# Prefixes stripped before alias lookup so that `str(some_array.dtype)` and
# module-qualified names resolve identically. KernelBench and Rosetta both
# pass `str(arr.dtype)`, which for ml_dtypes renders as e.g.
# "float8_e4m3fn" and for module-qualified references as
# "ml_dtypes.float8_e4m3fn".
_DTYPE_PREFIXES: tuple[str, ...] = ("ml_dtypes.", "np.", "numpy.", "jnp.")


def resolve_canonical_dtype(dtype_str: Any) -> str:
  """Normalizes any accepted dtype spelling to its canonical name.

  Accepts canonical names ("float8_e4m3fn"), short aliases ("fp8_e4m3",
  "bf16"), module-qualified names ("ml_dtypes.float8_e4m3fn"), and NumPy or
  ml_dtypes dtype objects. Every public entry point routes through this so
  that the contract tables, the ULP bit layouts, and the generator all agree
  on a single vocabulary.

  Args:
    dtype_str: A dtype name or a dtype-like object.

  Returns:
    The canonical dtype name. Unrecognized inputs are returned unchanged so
    that callers can raise a domain-specific error with the original text.
  """
  if not isinstance(dtype_str, str):
    dtype_str = getattr(dtype_str, "name", None) or str(dtype_str)
  name = dtype_str.strip()
  for prefix in _DTYPE_PREFIXES:
    if name.startswith(prefix):
      name = name[len(prefix) :]
      break
  return _DTYPE_ALIASES.get(name, name)


def get_contract(dtype_str: Any) -> tuple[int, int]:
  """Returns the (recommended_ulp, hard_ceiling_ulp) contract for a dtype.

  Args:
    dtype_str: Any spelling accepted by `resolve_canonical_dtype`.

  Returns:
    A tuple of the recommended golden contract and the immutable hard safety
    ceiling. Unknown dtypes fall back to the conservative float default.
  """
  canonical = resolve_canonical_dtype(dtype_str)
  return (
      RECOMMENDED_CONTRACT_ULP.get(canonical, 2),
      MAX_HARD_CEILING_ULP.get(canonical, 8),
  )


def is_discrete_dtype(dtype_str: str) -> bool:
  """True if dtype is discrete (bool, integer, unsigned integer)."""
  canonical = resolve_canonical_dtype(dtype_str)
  return canonical == "bool" or canonical in INTEGER_DTYPES


@functools.lru_cache(maxsize=None)
def _magnitude_index(threshold: float, dtype_str: str) -> int:
  """Returns the sign-magnitude integer index of a positive scalar magnitude.

  Lets magnitude comparisons run in the integer domain against the same
  continuous index space `_sign_magnitude_to_continuous_int` produces, so the
  caller never has to materialize float64 copies of its tensors.

  Args:
    threshold: A positive magnitude in real units.
    dtype_str: Any spelling accepted by `resolve_canonical_dtype`.

  Returns:
    The integer index corresponding to `threshold`. Values at or above this
    index have magnitude at or above `threshold`.
  """
  scalar = np.asarray([abs(threshold)])
  return int(_sign_magnitude_to_continuous_int(scalar, dtype_str)[0])


def compute_ulp_distance(
    actual: np.ndarray,
    expected: np.ndarray,
    dtype_str: str = "bfloat16",
    zero_threshold: float = 0.05,
) -> np.ndarray:
  """Computes exact integer bitwise ULP distance with zero-crossing mitigation.

  Args:
    actual: The candidate tensor.
    expected: The reference tensor.
    dtype_str: The data type string.
    zero_threshold: Magnitude threshold below which opposite-sign values are
      evaluated using scaled absolute difference to avoid sign-crossing ULP
      singularities.

  Returns:
    An ndarray of bitwise ULP distances.
  """
  if (
      dtype_str == "bool"
      or actual.dtype == np.bool_
      or expected.dtype == np.bool_
  ):
    return (actual != expected).astype(np.int64)
  if dtype_str in INTEGER_DTYPES or np.issubdtype(actual.dtype, np.integer):
    a = np.asarray(actual)
    e = np.asarray(expected)
    if np.issubdtype(a.dtype, np.unsignedinteger):
      diff = np.where(a >= e, a - e, e - a)
      return np.minimum(diff, np.iinfo(np.int64).max).astype(np.int64)
    if a.dtype != np.int64:
      return np.abs(a.astype(np.int64) - e.astype(np.int64))
    hi = np.maximum(a, e)
    lo = np.minimum(a, e)
    diff = np.where(
        (lo < 0) & (hi >= 0),
        np.minimum(
            hi.astype(np.float64) - lo.astype(np.float64),
            float(np.iinfo(np.int64).max),
        ).astype(np.int64),
        hi - lo,
    )
    return diff

  int_act = _sign_magnitude_to_continuous_int(actual, dtype_str)
  int_exp = _sign_magnitude_to_continuous_int(expected, dtype_str)
  raw_ulp = np.abs(int_act - int_exp)

  # Zero-crossing mitigation. Opposite-sign values whose magnitudes are both
  # tiny sit on either side of the sign-magnitude discontinuity, where the raw
  # bit distance jumps by ~2^(mantissa+exponent bits) despite the values being
  # numerically adjacent.
  #
  # The predicate is ordered cheapest-first. A sign disagreement is necessary
  # for mitigation and is a single pass over booleans, so testing it first
  # lets the common case -- no element straddles zero -- skip the magnitude
  # comparisons entirely.
  sign_differs = (int_act < 0) != (int_exp < 0)
  if not sign_differs.any():
    return raw_ulp

  # Compare against the threshold's own bit pattern rather than calling
  # np.abs, which would allocate a full int64 temporary per operand.
  threshold_idx = _magnitude_index(zero_threshold, dtype_str)
  mitigate_mask = (
      sign_differs
      & (int_act < threshold_idx)
      & (int_act > -threshold_idx)
      & (int_exp < threshold_idx)
      & (int_exp > -threshold_idx)
      # Only the sign-magnitude jump itself needs rescaling; a genuine small
      # distance across zero is already correct.
      & (raw_ulp > 10)
  )

  if not mitigate_mask.any():
    return raw_ulp

  canonical_d = resolve_canonical_dtype(dtype_str)
  if canonical_d in DTYPE_TOLERANCES:
    eps = DTYPE_TOLERANCES[canonical_d][0]
  elif canonical_d.startswith("float8"):
    eps = 0.125
  else:
    eps = 1e-3

  # Restrict the float64 work to the affected elements.
  idx = np.nonzero(mitigate_mask)
  act_sel = np.asarray(actual, dtype=np.float64)[idx]
  exp_sel = np.asarray(expected, dtype=np.float64)[idx]
  scaled = np.ceil(np.abs(act_sel - exp_sel) / eps).astype(np.int64)
  out = raw_ulp.copy()
  out[idx] = scaled
  return out


def bitwise_mismatch_mask(actual: Any, expected: Any) -> np.ndarray:
  """Returns a boolean mask of the elements whose raw bit patterns differ.

  A zero ULP distance is not bitwise equality: `compute_ulp_distance` maps
  `-0.0` and `+0.0` to the same index and says nothing useful about NaN. This
  compares raw bytes instead, so `-0.0` differs from `+0.0` and two NaNs are
  equal only when their payloads match.

  Args:
    actual: The candidate tensor.
    expected: The reference tensor, with the same shape and dtype.

  Returns:
    A boolean array shaped like the inputs.

  Raises:
    ValueError: If the shapes or dtypes differ.
  """
  act = np.asarray(actual)
  exp = np.asarray(expected)
  if act.shape != exp.shape or act.dtype != exp.dtype:
    raise ValueError(
        "bitwise_mismatch_mask needs equal shapes and dtypes, got"
        f" {act.shape} {act.dtype} vs {exp.shape} {exp.dtype}."
    )
  itemsize = act.dtype.itemsize
  if act.size == 0 or itemsize == 0:
    return np.zeros(act.shape, dtype=bool)

  def _as_byte_rows(arr: np.ndarray) -> np.ndarray:
    flat = np.ascontiguousarray(arr).reshape(-1)
    return flat.view(np.uint8).reshape(-1, itemsize)

  return np.any(_as_byte_rows(act) != _as_byte_rows(exp), axis=1).reshape(
      act.shape
  )


def tolerance_ratio(
    candidate: Any, reference: Any, atol: float, rtol: float
) -> float:
  """Returns max(|candidate - reference| / (atol + rtol * |reference|)).

  A result at or below 1.0 means every element passes an allclose check with
  these tolerances scaled by the reference; how far below 1.0 is the headroom.
  Equal elements, including matching Inf and NaN positions, count as 0. Any
  other non-finite difference counts as Inf.

  Args:
    candidate: Candidate output.
    reference: Reference output with the same shape.
    atol: Absolute tolerance.
    rtol: Relative tolerance.
  """
  cand = np.asarray(candidate, dtype=np.float64)
  ref = np.asarray(reference, dtype=np.float64)
  if cand.size == 0:
    return 0.0
  same = (cand == ref) | (np.isnan(cand) & np.isnan(ref))
  with np.errstate(invalid="ignore", over="ignore"):
    diff = np.where(same, 0.0, np.abs(cand - ref))
  diff = np.where(np.isnan(diff), np.inf, diff)
  bound = atol + rtol * np.abs(np.where(np.isfinite(ref), ref, 0.0))
  ratio = np.divide(
      diff, bound, out=np.full_like(diff, np.inf), where=bound > 0
  )
  ratio = np.where(diff == 0.0, 0.0, ratio)
  return float(np.max(ratio))
