"""Reusable library for comparing two kernel implementations on test suites."""

import collections.abc
import dataclasses
import types
from typing import Any
import ml_dtypes
import numpy as np
from xprof.cli.internal import numerical_generator


@dataclasses.dataclass(frozen=True)
class ToleranceAudit:
  recommended_contract_ulp: int
  configured_max_ulp: int
  hard_safety_ceiling: int
  is_relaxed_override: bool
  caution_banner: str | None = None


@dataclasses.dataclass(frozen=True)
class BatchValidationResult:
  batch_name: str
  regime: str
  max_ulp_distance: int
  p99_9_ulp_distance: float
  mean_ulp_distance: float
  ulp_histogram: dict[str, int]
  has_nan_or_inf: bool
  passed: bool


@dataclasses.dataclass(frozen=True)
class KernelValidationReport:
  is_numerically_equivalent: bool
  overall_max_ulp: int
  failed_batches_count: int
  total_batches_count: int
  batch_results: list[BatchValidationResult]
  summary_message: str
  tolerance_audit: ToleranceAudit | None = None


def _sign_magnitude_to_continuous_int(
    arr: np.ndarray, dtype_str: str
) -> np.ndarray:
  """Converts floating-point sign-magnitude bits to continuous int64 index."""
  if dtype_str == "float32":
    raw = arr.astype(np.float32).view(np.uint32).astype(np.int64)
    sign_mask = 0x80000000
    mag_mask = 0x7FFFFFFF
  elif dtype_str == "bfloat16":
    if arr.dtype == ml_dtypes.bfloat16:
      raw = arr.view(np.uint16).astype(np.int64)
    else:
      raw = arr.astype(ml_dtypes.bfloat16).view(np.uint16).astype(np.int64)
    sign_mask = 0x8000
    mag_mask = 0x7FFF
  elif dtype_str == "float16":
    raw = arr.astype(np.float16).view(np.uint16).astype(np.int64)
    sign_mask = 0x8000
    mag_mask = 0x7FFF
  elif dtype_str == "float64":
    raw = arr.astype(np.float64).view(np.uint64)
    sign_mask_u64 = np.uint64(0x8000000000000000)
    mag_mask_u64 = np.uint64(0x7FFFFFFFFFFFFFFF)
    is_negative = (raw & sign_mask_u64) != 0
    magnitude = (raw & mag_mask_u64).astype(np.int64)
    return np.where(is_negative, -magnitude, magnitude)
  elif dtype_str == "fp8_e4m3":
    if arr.dtype == ml_dtypes.float8_e4m3fn:
      raw = arr.view(np.uint8).astype(np.int64)
    else:
      raw = arr.astype(ml_dtypes.float8_e4m3fn).view(np.uint8).astype(np.int64)
    sign_mask = 0x80
    mag_mask = 0x7F
  elif dtype_str == "fp8_e5m2":
    if arr.dtype == ml_dtypes.float8_e5m2:
      raw = arr.view(np.uint8).astype(np.int64)
    else:
      raw = arr.astype(ml_dtypes.float8_e5m2).view(np.uint8).astype(np.int64)
    sign_mask = 0x80
    mag_mask = 0x7F
  else:
    raise ValueError(f"Unsupported dtype for ULP conversion: {dtype_str}")

  is_negative = (raw & sign_mask) != 0
  magnitude = raw & mag_mask
  return np.where(is_negative, -magnitude, magnitude)


_INTEGER_DTYPES: frozenset[str] = frozenset({
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


def _resolve_canonical_dtype(dtype_str: str) -> str:
  """Normalizes dtype strings (e.g. fp8_e4m3, bfloat16, int32)."""
  dtype_map = {
      "fp8_e4m3": "float8_e4m3fn",
      "float8_e4m3": "float8_e4m3fn",
      "fp8_e5m2": "float8_e5m2",
      "float8_e5m2": "float8_e5m2",
      "bf16": "bfloat16",
      "fp16": "float16",
      "fp32": "float32",
      "fp64": "float64",
  }
  return dtype_map.get(dtype_str, dtype_str)


def compute_ulp_distance(
    actual: np.ndarray,
    expected: np.ndarray,
    dtype_str: str = "bfloat16",
) -> np.ndarray:
  """Computes exact integer bitwise ULP distance or exact integer/boolean delta."""
  if (
      dtype_str == "bool"
      or actual.dtype == np.bool_
      or expected.dtype == np.bool_
  ):
    return (actual != expected).astype(np.int64)
  if dtype_str in _INTEGER_DTYPES or np.issubdtype(actual.dtype, np.integer):
    if np.issubdtype(actual.dtype, np.unsignedinteger):
      diff = np.where(actual >= expected, actual - expected, expected - actual)
      return np.minimum(diff, np.iinfo(np.int64).max).astype(np.int64)
    # Cast to uint64 so differences between extreme signed values do not
    # overflow.
    diff = np.where(
        actual >= expected,
        actual.astype(np.uint64) - expected.astype(np.uint64),
        expected.astype(np.uint64) - actual.astype(np.uint64),
    )
    return np.minimum(diff, np.iinfo(np.int64).max).astype(np.int64)
  int_act = _sign_magnitude_to_continuous_int(actual, dtype_str)
  int_exp = _sign_magnitude_to_continuous_int(expected, dtype_str)
  return np.abs(int_act - int_exp)


def validate_kernels(
    kernel_ref: collections.abc.Callable[..., Any],
    kernel_candidate: collections.abc.Callable[..., Any],
    shapes: (
        collections.abc.Sequence[int]
        | collections.abc.Sequence[collections.abc.Sequence[int]]
    ),
    dtype_str: str = "bfloat16",
    test_suite: list[dict[str, Any]] | None = None,
    tier: str = "presubmit",
    max_allowed_ulp: int = 2,
    p99_9_allowed_ulp: int = 1,
    seed: int = 42,
) -> KernelValidationReport:
  """Validates candidate kernel against reference implementation."""
  canonical_dtype = _resolve_canonical_dtype(dtype_str)
  recommended_ulp = RECOMMENDED_CONTRACT_ULP.get(canonical_dtype, 2)
  hard_ceiling = MAX_HARD_CEILING_ULP.get(canonical_dtype, 8)

  # If using the default floating-point tolerance (max_allowed_ulp=2) on a
  # dtype whose hard ceiling is lower (e.g. discrete integers/bool where
  # ceiling is 0), automatically adapt to the recommended contract.
  if max_allowed_ulp == 2 and hard_ceiling < 2:
    actual_max_allowed_ulp = recommended_ulp
  else:
    actual_max_allowed_ulp = max_allowed_ulp

  if actual_max_allowed_ulp > hard_ceiling:
    raise ValueError(
        f"Requested max_allowed_ulp={actual_max_allowed_ulp} exceeds immutable"
        f" safety ceiling ({hard_ceiling}) for dtype '{canonical_dtype}'."
    )

  is_relaxed = actual_max_allowed_ulp > recommended_ulp
  caution_msg = None
  if is_relaxed:
    caution_msg = (
        f"⚠️ CAUTION: A relaxed tolerance threshold"
        f" (max_allowed_ulp={actual_max_allowed_ulp}) was configured. The"
        f" recommended contract is <= {recommended_ulp} ULP for"
        f" '{canonical_dtype}' to guarantee numerical correctness. Ensure this"
        " elevation is analytically justified (e.g. Split-K tree reordering)."
    )

  tolerance_audit = ToleranceAudit(
      recommended_contract_ulp=recommended_ulp,
      configured_max_ulp=actual_max_allowed_ulp,
      hard_safety_ceiling=hard_ceiling,
      is_relaxed_override=is_relaxed,
      caution_banner=caution_msg,
  )

  if test_suite is None:
    test_suite = numerical_generator.generate_test_suite(
        shapes, dtype_str=dtype_str, tier=tier, seed=seed
    )

  batch_results = []
  overall_max_ulp = 0
  failed_batches = 0

  for batch in test_suite:
    args = batch.get("args", (batch.get("tensor"),))
    kwargs = batch.get("kwargs", {})

    out_ref = np.asarray(kernel_ref(*args, **kwargs))
    out_cand = np.asarray(kernel_candidate(*args, **kwargs))

    if out_cand.shape != out_ref.shape:
      raise ValueError(
          f"Shape mismatch in batch '{batch.get('name')}': candidate shape "
          f"{out_cand.shape} != reference shape {out_ref.shape}"
      )

    is_discrete = (
        dtype_str == "bool"
        or dtype_str in _INTEGER_DTYPES
        or np.issubdtype(out_cand.dtype, np.integer)
        or out_cand.dtype == np.bool_
    )

    if not is_discrete:
      out_cand_f32 = out_cand.astype(np.float32)
      out_ref_f32 = out_ref.astype(np.float32)
      has_nan_or_inf = bool(
          np.isnan(out_cand_f32).any()
          or np.isinf(out_cand_f32).any()
          or np.isnan(out_ref_f32).any()
          or np.isinf(out_ref_f32).any()
      )
    else:
      has_nan_or_inf = False

    if has_nan_or_inf:
      max_ulp = 999999
      p99_9 = 999999.0
      mean_ulp = 999999.0
      hist = {"<=1_ulp": 0, "<=2_ulp": 0, ">2_ulp": out_cand.size}
      passed = False
    else:
      ulp_arr = compute_ulp_distance(out_cand, out_ref, dtype_str)
      max_ulp = int(np.max(ulp_arr))
      p99_9 = float(np.percentile(ulp_arr, 99.9))
      mean_ulp = float(np.mean(ulp_arr))
      hist = {
          "<=1_ulp": int(np.sum(ulp_arr <= 1)),
          "<=2_ulp": int(np.sum(ulp_arr <= 2)),
          ">2_ulp": int(np.sum(ulp_arr > 2)),
      }
      effective_max_ulp = actual_max_allowed_ulp
      effective_p99_9 = (
          0.0 if is_discrete and p99_9_allowed_ulp == 1 else p99_9_allowed_ulp
      )
      passed = bool(max_ulp <= effective_max_ulp and p99_9 <= effective_p99_9)

    overall_max_ulp = max(overall_max_ulp, max_ulp)
    if not passed:
      failed_batches += 1

    batch_results.append(
        BatchValidationResult(
            batch_name=batch["name"],
            regime=batch["regime"],
            max_ulp_distance=max_ulp,
            p99_9_ulp_distance=p99_9,
            mean_ulp_distance=mean_ulp,
            ulp_histogram=hist,
            has_nan_or_inf=has_nan_or_inf,
            passed=passed,
        )
    )

  is_equivalent = failed_batches == 0
  if is_equivalent:
    summary = (
        f"PASSED: Kernels are numerically equivalent across {len(test_suite)}"
        f" batches (Max ULP: {overall_max_ulp}, Configured Limit:"
        f" {actual_max_allowed_ulp}, Recommended: <= {recommended_ulp})."
    )
  else:
    summary = (
        f"FAILED: Numerical divergence detected ({failed_batches}/"
        f"{len(test_suite)} batches failed ULP criteria)."
    )

  if caution_msg:
    summary = f"{caution_msg}\n{summary}"

  return KernelValidationReport(
      is_numerically_equivalent=is_equivalent,
      overall_max_ulp=overall_max_ulp,
      failed_batches_count=failed_batches,
      total_batches_count=len(test_suite),
      batch_results=batch_results,
      summary_message=summary,
      tolerance_audit=tolerance_audit,
  )
