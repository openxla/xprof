"""Reusable library for comparing two kernel implementations on test suites."""

import collections.abc
import dataclasses
from typing import Any
import ml_dtypes
import numpy as np
from xprof.cli.internal import numerical_generator


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


def compute_ulp_distance(
    actual: np.ndarray,
    expected: np.ndarray,
    dtype_str: str = "bfloat16",
) -> np.ndarray:
  """Computes exact integer bitwise ULP distance without integer overflow."""
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

    out_cand_f32 = out_cand.astype(np.float32)
    out_ref_f32 = out_ref.astype(np.float32)
    has_nan_or_inf = bool(
        np.isnan(out_cand_f32).any()
        or np.isinf(out_cand_f32).any()
        or np.isnan(out_ref_f32).any()
        or np.isinf(out_ref_f32).any()
    )

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
      passed = (max_ulp <= max_allowed_ulp) and (p99_9 <= p99_9_allowed_ulp)

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
        f" batches (Max ULP: {overall_max_ulp}, Limit: {max_allowed_ulp})."
    )
  else:
    summary = (
        f"FAILED: Numerical divergence detected ({failed_batches}/"
        f"{len(test_suite)} batches failed ULP criteria)."
    )

  return KernelValidationReport(
      is_numerically_equivalent=is_equivalent,
      overall_max_ulp=overall_max_ulp,
      failed_batches_count=failed_batches,
      total_batches_count=len(test_suite),
      batch_results=batch_results,
      summary_message=summary,
  )
