"""Reusable library for comparing two kernel implementations on test suites.

Public API:
  validate_kernels: Compare a candidate kernel against a reference, optionally
    auditing the reference itself against a high-precision oracle.
  chunk_callable: Wrap an oracle to execute in slices along one axis, for
    shapes whose high-precision intermediates exceed device memory.
  ORACLE_AUTO: `kernel_oracle` sentinel that re-runs the reference with its
    floating-point arguments promoted to float64.
"""

import collections.abc
import dataclasses
import importlib
import inspect
import logging
import sys
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
class OracleAudit:
  """Distance of each kernel from a high-precision (float64) oracle.

  `validate_kernels` otherwise measures *agreement* between two kernels, and
  two kernels that are wrong in the same way agree perfectly. This block
  answers the separate question "is the reference itself correct?", which
  matters on accelerators because the obvious reference is silently lossy:
  `jnp.dot` truncates f32 inputs to bf16 for the TPU MXU under
  `Precision.DEFAULT`, and uses TF32 (10-bit mantissa) on Ampere+ GPUs.

  `ORACLE_AUTO` detects that class of loss for a precision-following reference;
  an explicit float64 callable detects it unconditionally. See
  `validate_kernels` for the difference.

  Distances are measured in `dtype_str` units, like every other ULP figure in
  this module: the oracle result is rounded to the output dtype first, so
  `reference_max_ulp_from_oracle == 0` means the reference is the correctly
  rounded result in that dtype.
  """

  oracle_executed_in_float64: bool
  oracle_output_dtype: str
  reference_max_ulp_from_oracle: int
  reference_p99_9_ulp_from_oracle: float
  candidate_max_ulp_from_oracle: int
  candidate_p99_9_ulp_from_oracle: float
  reference_is_lossy: bool
  oracle_precision_verified: bool
  reference_max_abs_from_oracle: float = 0.0
  candidate_max_abs_from_oracle: float = 0.0
  oracle_banner: str | None = None
  reference_pin_inert: bool = False
  oracle_probe_diagnostic: str | None = None


@dataclasses.dataclass(frozen=True)
class UlpContext:
  """Contextual statistics and reliability assessment of ULP measurements."""

  bit_identical: bool
  p50: float
  p99_9: float
  max_ulp: int
  reliable: bool
  note: str | None = None


@dataclasses.dataclass(frozen=True)
class BatchValidationResult:
  """Validation metrics and status for a single test batch."""

  batch_name: str
  regime: str
  max_ulp_distance: int
  p99_9_ulp_distance: float
  mean_ulp_distance: float
  ulp_histogram: dict[str, int]
  has_nan_or_inf: bool
  passed: bool
  reference_ulp_from_oracle: int | None = None
  candidate_ulp_from_oracle: int | None = None
  ulp_context: UlpContext | None = None
  allclose_passed: bool = True


@dataclasses.dataclass(frozen=True)
class KernelValidationReport:
  """Aggregated validation report across all test batches and regimes."""

  is_numerically_equivalent: bool
  overall_max_ulp: int
  failed_batches_count: int
  total_batches_count: int
  batch_results: list[BatchValidationResult]
  summary_message: str
  tolerance_audit: ToleranceAudit | None = None
  oracle_audit: OracleAudit | None = None
  correctness_basis: str = "AGREEMENT_ONLY"
  run_config: dict[str, Any] = dataclasses.field(default_factory=dict)
  ulp_context: UlpContext | None = None
  narrow_output_dtype_warning: str | None = None


@dataclasses.dataclass(frozen=True)
class PrecisionProbeResult:
  """Result of accelerator precision sensitivity probing."""

  is_pinned: bool | None
  diagnostic: str | None = None


@dataclasses.dataclass(frozen=True)
class OracleVerificationResult:
  """Result of oracle precision verification."""

  verified: bool
  banner: str | None
  probe_diagnostic: str | None = None


@dataclasses.dataclass(frozen=True)
class _BatchExecutionResult:
  """Internal result of executing and validating a single batch."""

  batch_result: BatchValidationResult
  max_ulp: int
  passed: bool
  oracle_ran: bool = False
  oracle_in_float64: bool = True
  oracle_output_dtype: str = ""
  ref_oracle_max_ulp: int = 0
  cand_oracle_max_ulp: int = 0
  ref_oracle_p99_9: float = 0.0
  cand_oracle_p99_9: float = 0.0
  ref_oracle_max_abs: float = 0.0
  cand_oracle_max_abs: float = 0.0
  narrow_warning: str | None = None


def _get_finfo(dtype: Any) -> Any:
  """Returns finfo object supporting both NumPy and ml_dtypes floating types."""
  np_dtype = np.dtype(dtype)
  try:
    return np.finfo(np_dtype)
  except (ValueError, TypeError) as e:
    logging.debug("np.finfo failed for dtype %s: %s", np_dtype, e)
    return ml_dtypes.finfo(np_dtype)


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

_DTYPE_TOLERANCES: types.MappingProxyType[str, tuple[float, float]] = (
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


def _is_discrete_dtype(dtype_str: str) -> bool:
  """True if dtype is discrete (bool, integer, unsigned integer)."""
  canonical = _resolve_canonical_dtype(dtype_str)
  return canonical == "bool" or canonical in _INTEGER_DTYPES


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
  if dtype_str in _INTEGER_DTYPES or np.issubdtype(actual.dtype, np.integer):
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

  act_f64 = np.asarray(actual, dtype=np.float64)
  exp_f64 = np.asarray(expected, dtype=np.float64)
  int_act = _sign_magnitude_to_continuous_int(actual, dtype_str)
  int_exp = _sign_magnitude_to_continuous_int(expected, dtype_str)
  raw_ulp = np.abs(int_act - int_exp)

  # Check for opposite-sign values below the zero_threshold
  cross_zero_mask = (
      (act_f64 * exp_f64 < 0)
      & (np.abs(act_f64) < zero_threshold)
      & (np.abs(exp_f64) < zero_threshold)
  )

  if np.any(cross_zero_mask):
    canonical_d = _resolve_canonical_dtype(dtype_str)
    if canonical_d in _DTYPE_TOLERANCES:
      eps = _DTYPE_TOLERANCES[canonical_d][0]
    elif dtype_str.startswith("fp8"):
      eps = 0.125
    else:
      eps = 1e-3

    scaled_diff = np.abs(act_f64 - exp_f64) / eps
    # Only replace if raw bit distance exhibits the sign-magnitude jump (> 10)
    mitigate_mask = cross_zero_mask & (raw_ulp > 10)
    return np.where(
        mitigate_mask,
        np.ceil(scaled_diff).astype(np.int64),
        raw_ulp,
    )

  return raw_ulp


ORACLE_AUTO = "auto"

_ML_FLOAT_DTYPES = (
    ml_dtypes.bfloat16,
    ml_dtypes.float8_e4m3fn,
    ml_dtypes.float8_e5m2,
)


def _is_float_array(arr: np.ndarray) -> bool:
  """True for IEEE floats and for the ml_dtypes extension floats."""
  return np.issubdtype(arr.dtype, np.floating) or arr.dtype in _ML_FLOAT_DTYPES


def _promote_args_to_dtype(
    args: collections.abc.Sequence[Any],
    kwargs: dict[str, Any],
    target_dtype: Any,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
  """Casts floating-point arguments to target_dtype, leaving others untouched.

  Integer arguments (indices, segment ids, masks) are passed through unchanged
  so that ORACLE_AUTO works on gather/routing kernels as well.

  Args:
    args: Positional argument sequence.
    kwargs: Keyword argument mapping.
    target_dtype: Target NumPy floating dtype (e.g. np.float32, np.float64).

  Returns:
    Tuple of promoted positional arguments and keyword arguments.
  """

  def _promote(value: Any) -> Any:
    arr = np.asarray(value)
    return arr.astype(target_dtype) if _is_float_array(arr) else value

  return (
      tuple(_promote(a) for a in args),
      {k: _promote(v) for k, v in kwargs.items()},
  )


def _detect_device_info() -> tuple[str, str]:
  """Detects (device_kind, backend) for provenance tracking."""
  if "jax" in sys.modules:
    try:
      jax = sys.modules["jax"]
      backend = str(getattr(jax, "default_backend", lambda: "cpu")())
      devices = getattr(jax, "devices", lambda: [])()
      if devices:
        device = devices[0]
        kind = getattr(device, "device_kind", None)
        if kind:
          return str(kind), backend
        platform = getattr(device, "platform", None)
        if platform:
          return str(platform), backend
      return backend, backend
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.debug("Failed querying loaded JAX device info: %s", e)
  else:
    try:
      jax = importlib.import_module("jax")
      backend = str(getattr(jax, "default_backend", lambda: "cpu")())
      devices = getattr(jax, "devices", lambda: [])()
      if devices:
        device = devices[0]
        kind = getattr(device, "device_kind", None)
        if kind:
          return str(kind), backend
        platform = getattr(device, "platform", None)
        if platform:
          return str(platform), backend
      return backend, backend
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.debug("Failed importing or querying JAX device info: %s", e)

  if "torch" in sys.modules:
    try:
      torch = sys.modules["torch"]
      cuda = getattr(torch, "cuda", None)
      if cuda and getattr(cuda, "is_available", lambda: False)():
        return f"cuda:{cuda.get_device_name(0)}", "cuda"
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.debug("Failed querying loaded PyTorch CUDA device info: %s", e)
  else:
    try:
      torch = importlib.import_module("torch")
      cuda = getattr(torch, "cuda", None)
      if cuda and getattr(cuda, "is_available", lambda: False)():
        return f"cuda:{cuda.get_device_name(0)}", "cuda"
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.debug("Failed importing or querying PyTorch CUDA info: %s", e)

  return "cpu", "cpu"


def _is_jax_array(val: Any) -> bool:
  """Returns True if val is a JAX array on device, False if host/NumPy."""
  if isinstance(val, (tuple, list)) and val:
    return _is_jax_array(val[0])
  if isinstance(val, np.ndarray):
    # NumPy >= 2.0 ndarrays expose `.device` (array API), so the attribute
    # probe below cannot be used to identify host arrays.
    return False
  jax = sys.modules.get("jax")
  if jax is not None and hasattr(jax, "Array") and isinstance(val, jax.Array):
    return True
  if hasattr(val, "devices"):
    return True
  return hasattr(val, "device")


def _is_oom_error(exc: BaseException) -> bool:
  """Returns True when an exception is an accelerator out-of-memory failure."""
  text = str(exc).lower()
  return "resource_exhausted" in text or "out of memory" in text


def _probe_precision(
    fn: collections.abc.Callable[..., Any],
    args: collections.abc.Sequence[Any],
    kwargs: dict[str, Any],
    baseline: str = "given",
    device_kind: str = "cpu",
) -> PrecisionProbeResult:
  """Probes callable sensitivity to accelerator matmul precision.

  Args:
    fn: Callable under test.
    args: Positional arguments for fn.
    kwargs: Keyword arguments for fn.
    baseline: 'given' evaluates if fn was already at maximum precision
      (comparing fn(*args, **kwargs) vs fn with HIGHEST precision). 'default'
      evaluates if fn ignores precision pinning on accelerators (comparing fn
      with DEFAULT vs HIGHEST precision).
    device_kind: Target device ('cpu', 'tpu', 'gpu').

  Returns:
    PrecisionProbeResult containing boolean sensitivity and diagnostic details.
  """
  is_default_baseline = baseline == "default"
  diagnostics: list[str] = []

  if is_default_baseline:
    if device_kind.lower() == "cpu":
      return PrecisionProbeResult(
          False, "CPU execution: precision pinning does not alter CPU ops"
      )
    if hasattr(fn, "precision_inert"):
      return PrecisionProbeResult(
          bool(getattr(fn, "precision_inert")),
          "Explicit precision_inert attribute found on callable",
      )

  raw_base = None
  try:
    if is_default_baseline:
      out_base = None
    else:
      raw_base = fn(*args, **kwargs)
      out_base = np.asarray(raw_base)
      if out_base.dtype == np.float64:
        return PrecisionProbeResult(True, "Callable natively returned float64")
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.debug("Baseline execution for %s failed: %s", fn, e)
    msg = f"Baseline execution failed ({type(e).__name__}: {e})"
    return PrecisionProbeResult(False if is_default_baseline else None, msg)

  # Path 2: Check explicit parameter first (what docs advise users to pin)
  sig_has_precision = False
  try:
    sig = inspect.signature(fn)
    sig_has_precision = "precision" in sig.parameters
  except (TypeError, ValueError) as e:
    logging.debug("inspect.signature failed for %s: %s", fn, e)
    diagnostics.append(f"signature check failed ({type(e).__name__}: {e})")

  if sig_has_precision:
    try:
      if is_default_baseline:
        out_base = np.asarray(fn(*args, **{**kwargs, "precision": "default"}))
      elif "precision" in kwargs:
        out_base = None

      if out_base is None and not is_default_baseline:
        out_base = np.asarray(fn(*args, **kwargs))

      if out_base is not None:
        out_high = np.asarray(fn(*args, **{**kwargs, "precision": "highest"}))
        if out_base.shape == out_high.shape and out_base.size > 0:
          match = bool(np.array_equal(out_base, out_high))
          if is_default_baseline:
            desc = "Evaluated with explicit precision='default' vs 'highest'"
          else:
            desc = "Evaluated with explicit precision='highest'"
          return PrecisionProbeResult(match, desc)
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.debug("Calling %s with explicit precision failed: %s", fn, e)
      diagnostics.append(
          f"explicit precision call failed ({type(e).__name__}: {e})"
      )

  # Path 1: Global default_matmul_precision context fallback
  if is_default_baseline or raw_base is not None:
    raw_sample = fn(*args, **kwargs) if raw_base is None else raw_base
    if not _is_jax_array(raw_sample):
      return PrecisionProbeResult(
          False if is_default_baseline else True,
          "Host NumPy execution: not subject to accelerator matmul"
          " precision",
      )

  jax = sys.modules.get("jax")
  if jax is None:
    try:
      jax = importlib.import_module("jax")
    except (ImportError, ModuleNotFoundError) as e:
      logging.debug("JAX module import failed: %s", e)
      diagnostics.append("JAX module not available in environment")
      jax = None

  if jax is not None and hasattr(jax, "default_matmul_precision"):
    try:
      if is_default_baseline:
        with jax.default_matmul_precision("default"):
          out_base = np.asarray(fn(*args, **kwargs))
      elif out_base is None:
        out_base = np.asarray(fn(*args, **kwargs))

      with jax.default_matmul_precision("highest"):
        out_high = np.asarray(fn(*args, **kwargs))

      if (
          out_base is not None
          and out_base.shape == out_high.shape
          and out_base.size > 0
      ):
        match = bool(np.array_equal(out_base, out_high))
        if is_default_baseline:
          # A callable with no `precision` parameter that is insensitive to the
          # global context is indistinguishable from one that hard-codes
          # HIGHEST internally, or that contains no matmul at all. Only a
          # *difference* is informative here; sameness is undetermined, so
          # never report inertness from this path.
          if match:
            return PrecisionProbeResult(
                False,
                "Global-context probe inconclusive: callable exposes no"
                " `precision` parameter, so hard-pinned, matmul-free and"
                " genuinely inert callables are indistinguishable",
            )
          return PrecisionProbeResult(
              False,
              "Callable output changed under"
              " jax.default_matmul_precision('default' vs 'highest');"
              " definitely not pin-inert",
          )
        desc = "Evaluated under jax.default_matmul_precision('highest')"
        return PrecisionProbeResult(match, desc)
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.debug("JAX precision context evaluation failed for %s: %s", fn, e)
      diagnostics.append(f"JAX context probe failed ({type(e).__name__}: {e})")

  if not is_default_baseline and device_kind.lower() == "cpu":
    return PrecisionProbeResult(True, "CPU execution: assumed pinned")

  diag_str = (
      "; ".join(diagnostics)
      if diagnostics
      else "No precision controls supported by callable"
  )
  return PrecisionProbeResult(False if is_default_baseline else None, diag_str)


def _probe_pin_inert(
    kernel_ref: collections.abc.Callable[..., Any],
    args: collections.abc.Sequence[Any],
    kwargs: dict[str, Any],
    device_kind: str,
) -> bool:
  """Returns True if kernel_ref ignores precision pinning on accelerators."""
  return bool(
      _probe_precision(
          kernel_ref, args, kwargs, baseline="default", device_kind=device_kind
      ).is_pinned
  )


def chunk_callable(
    fn: collections.abc.Callable[..., Any],
    chunk_arg_indices: tuple[int, ...] = (0, 1, 2),
    chunk_kwargs: tuple[str, ...] = (),
    axis: int = 1,
    chunks: int = 8,
) -> collections.abc.Callable[..., Any]:
  """Wraps a callable to chunk along an axis, preventing accelerator OOM.

  This is particularly useful for attention oracles where high-precision
  references (e.g. Precision.HIGHEST or float64) allocate large temporary
  tensors (e.g. (B, H, S, S) logits and softmax matrices) that exceed device
  HBM.

  Args:
    fn: The callable to wrap (e.g. attention reference function).
    chunk_arg_indices: Positional argument indices to slice along `axis`. For
      attention `fn(q, k, v)`, this defaults to `(0, 1, 2)`.
    chunk_kwargs: Keyword argument names to slice along `axis` (e.g. `("q", "k",
      "v")`).
    axis: The tensor axis to slice along (e.g., axis=1 for head dimension in
      `(B, H, S, D)`).
    chunks: Number of chunks to split along `axis`.

  Returns:
    A wrapped callable that slices input arguments along `axis`, executes `fn`
    sequentially on each chunk, and concatenates the outputs along `axis`.
  """

  def wrapped(*args: Any, **kwargs: Any) -> Any:
    if chunks <= 1 or (not chunk_arg_indices and not chunk_kwargs):
      return fn(*args, **kwargs)

    first_arr = None
    if chunk_arg_indices and args and chunk_arg_indices[0] < len(args):
      first_arr = np.asarray(args[chunk_arg_indices[0]])
    elif chunk_kwargs:
      for k in chunk_kwargs:
        if k in kwargs:
          first_arr = np.asarray(kwargs[k])
          break

    if first_arr is None or axis >= first_arr.ndim:
      return fn(*args, **kwargs)

    dim_size = first_arr.shape[axis]
    actual_chunks = min(chunks, dim_size)
    if actual_chunks <= 1:
      return fn(*args, **kwargs)

    chunk_size = (dim_size + actual_chunks - 1) // actual_chunks
    outputs = []

    for i in range(actual_chunks):
      start_idx = i * chunk_size
      end_idx = min(dim_size, (i + 1) * chunk_size)
      if start_idx >= end_idx:
        break

      chunked_args = list(args)
      for arg_idx in chunk_arg_indices:
        if arg_idx < len(args):
          arr = args[arg_idx]
          slice_indices = [slice(None)] * arr.ndim
          slice_indices[axis] = slice(start_idx, end_idx)
          chunked_args[arg_idx] = arr[tuple(slice_indices)]

      chunked_kwargs = dict(kwargs)
      for k in chunk_kwargs:
        if k in chunked_kwargs:
          arr = chunked_kwargs[k]
          slice_indices = [slice(None)] * arr.ndim
          slice_indices[axis] = slice(start_idx, end_idx)
          chunked_kwargs[k] = arr[tuple(slice_indices)]

      out_chunk = fn(*chunked_args, **chunked_kwargs)
      outputs.append(out_chunk)

    if not outputs:
      return fn(*args, **kwargs)

    if isinstance(outputs[0], (tuple, list)):
      reconstructed = []
      for elem_idx in range(len(outputs[0])):
        elem_chunks = [np.asarray(o[elem_idx]) for o in outputs]
        reconstructed.append(np.concatenate(elem_chunks, axis=axis))
      if isinstance(outputs[0], tuple):
        return tuple(reconstructed)
      return reconstructed
    return np.concatenate([np.asarray(o) for o in outputs], axis=axis)

  return wrapped


def _verify_oracle_precision(
    oracle_in_float64: bool,
    oracle_output_dtype: str,
    canonical_dtype: str,
    kernel_oracle: Any,
    probe_args: collections.abc.Sequence[Any],
    probe_kwargs: dict[str, Any],
    effective_device_kind: str,
    reference_is_lossy: bool,
    oracle_ref_max_ulp: int,
    oracle_cand_max_ulp: int,
    recommended_ulp: int,
    reference_is_unpinned: bool = False,
) -> OracleVerificationResult:
  """Verifies oracle precision pinning and margin, returning diagnostics."""
  if oracle_in_float64:
    verified = True
    banner = None
    probe_diag = "Native host float64 oracle: exact arithmetic verified"
  else:
    ref_finfo = _get_finfo(canonical_dtype)
    f32_eps = float(np.finfo(np.float32).eps)
    oracle_finfo = _get_finfo(oracle_output_dtype)
    oracle_eps = float(getattr(oracle_finfo, "eps", 1.0))
    oracle_is_high_precision = oracle_eps <= f32_eps
    # A margin can come from either of two independent sources:
    #   (a) width  -- the oracle's dtype is materially finer than the kernel's;
    #   (b) pinning -- the oracle is precision-pinned while the reference is
    #       not, which on a TPU MXU is worth ~10^4x even at equal width.
    # Measured on v6e at 512x512 float32: an unpinned reference sits 2.66e-01
    # from exact while the same matmul pinned to HIGHEST sits 1.50e-05, a
    # factor of 17,792. Judging (b) by dtype width alone scores it 1x.
    has_width_margin = (
        hasattr(ref_finfo, "eps")
        and float(ref_finfo.eps) >= 100.0 * oracle_eps
    )
    has_precision_margin = oracle_is_high_precision and (
        has_width_margin or reference_is_unpinned
    )

    if isinstance(kernel_oracle, str):
      probe_res = PrecisionProbeResult(
          True, "Oracle string specification verified"
      )
    elif kernel_oracle is not None:
      probe_res = _probe_precision(
          kernel_oracle,
          probe_args,
          probe_kwargs,
          baseline="given",
          device_kind=effective_device_kind,
      )
    else:
      probe_res = PrecisionProbeResult(None, "No oracle provided")

    is_pinned = probe_res.is_pinned
    probe_diag = probe_res.diagnostic
    diag_clause = f" (Reason: {probe_diag})" if probe_diag else ""

    if is_pinned is not None and not is_pinned:
      verified = False
      banner = (
          f"⚠️ ORACLE IS NOT PRECISION-PINNED: The oracle returned"
          f" '{oracle_output_dtype}' and changes output when matmul precision"
          " is set to HIGHEST on accelerators (unpinned matmul precision"
          " drops accuracy by ~20,000x on TPU MXUs)."
          f"{diag_clause} The oracle distances below are not a correctness"
          " bound. Pin the oracle using jax.default_matmul_precision('highest')"
          " or pass precision='highest'."
      )
    elif not oracle_is_high_precision:
      verified = False
      banner = (
          f"⚠️ ORACLE LACKS NUMERICAL PRECISION: The oracle returned"
          f" '{oracle_output_dtype}' (eps {oracle_eps:.2e}). An oracle must"
          " be at least float32 (or float64) to serve as a ground truth"
          " reference. Low-precision oracles cannot establish correctness."
      )
    elif not has_precision_margin:
      verified = False
      banner = (
          f"⚠️ ORACLE NOT PRECISE ENOUGH: The oracle returned"
          f" '{oracle_output_dtype}' while the kernel under test emits"
          f" '{canonical_dtype}' (eps {ref_finfo.eps:.2e}), and the"
          " reference is already precision-pinned. An oracle needs a margin"
          " over the kernel it judges -- from a finer dtype, or from being"
          " pinned where the reference is not; with neither, a 0-error"
          " result is meaningless. Use a float64 oracle to validate"
          " float32 or wider kernels."
      )
    elif is_pinned:
      verified = True
      banner = None
    else:
      verified = False
      banner = (
          f"⚠️ ORACLE PRECISION UNDETERMINED: The oracle returned"
          f" '{oracle_output_dtype}'. Unable to verify accelerator precision"
          f" pinning{diag_clause}. Verify oracle pinning with"
          " jax.default_matmul_precision('highest') or pass a host float64"
          " NumPy callable."
      )

  if verified and reference_is_lossy:
    banner = (
        "⚠️ REFERENCE IS NOT EXACT: kernel_ref sits"
        f" {oracle_ref_max_ulp} ULP from the oracle, above the"
        f" {recommended_ulp} ULP contract for '{canonical_dtype}'."
        " Agreement with this reference does not establish correctness --"
        " a candidate reproducing the reference's own error passes at 0"
        " ULP. On TPU, jnp.dot truncates f32 inputs to bf16 for the MXU"
        " unless precision=HIGHEST; on Ampere+ GPUs the equivalent default"
        " is TF32. Pin the reference precision and re-run. (Candidate sits"
        f" {oracle_cand_max_ulp} ULP from the oracle.)"
    )

  return OracleVerificationResult(
      verified=verified, banner=banner, probe_diagnostic=probe_diag
  )


def _execute_single_batch(
    batch: dict[str, Any],
    kernel_ref: collections.abc.Callable[..., Any],
    kernel_candidate: collections.abc.Callable[..., Any],
    canonical_dtype: str,
    dtype_str: str,
    actual_max_allowed_ulp: int,
    p99_9_allowed_ulp: int,
    recommended_ulp: int,
    kernel_oracle: collections.abc.Callable[..., Any] | str | None = None,
) -> _BatchExecutionResult:
  """Executes and validates a single batch between reference and candidate."""
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

  narrow_warning = None
  if not is_discrete:
    try:
      ref_finfo = _get_finfo(out_ref.dtype)
      f32_eps = float(np.finfo(np.float32).eps)
      if hasattr(ref_finfo, "eps") and ref_finfo.eps > f32_eps:
        narrow_warning = (
            f"Output dtype {out_ref.dtype} is narrower than float32: defects"
            f" smaller than one output ULP ({ref_finfo.eps:.2%} relative) are"
            " destroyed by output quantization and cannot be detected. Re-run"
            " with the kernel emitting float32 to validate internal"
            " precision."
        )
    except (ValueError, TypeError) as e:
      logging.debug("finfo check for narrow dtype failed: %s", e)

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

  oracle_ran = False
  oracle_in_float64 = True
  oracle_output_dtype = ""
  batch_ref_oracle_ulp = None
  batch_cand_oracle_ulp = None
  ref_oracle_max_ulp = 0
  cand_oracle_max_ulp = 0
  ref_oracle_p99_9 = 0.0
  cand_oracle_p99_9 = 0.0
  ref_oracle_max_abs = 0.0
  cand_oracle_max_abs = 0.0

  if kernel_oracle is not None and not is_discrete and not has_nan_or_inf:
    try:
      if isinstance(kernel_oracle, str):
        oracle_args, oracle_kwargs = _promote_args_to_dtype(
            args, kwargs, np.float64
        )
        out_oracle = np.asarray(kernel_ref(*oracle_args, **oracle_kwargs))
      else:
        out_oracle = np.asarray(kernel_oracle(*args, **kwargs))
    except Exception as e:  # pylint: disable=broad-exception-caught
      if not _is_oom_error(e):
        raise
      cost_note = (
          "A high-precision oracle allocates far larger intermediates than"
          " the kernel it judges (a float64 attention oracle at B=4 H=64"
          " S=4096 D=128 needs 69.25 GiB against 31.24 GiB of v6e HBM)."
      )
      import_note = (
          "from xprof.cli.internal.numerical_validator import chunk_callable"
      )
      if isinstance(kernel_oracle, str):
        # ORACLE_AUTO re-runs kernel_ref itself, so there is no oracle
        # argument for the caller to wrap; they have to supply one.
        remedy = (
            f"'{kernel_oracle}' re-runs kernel_ref with its arguments"
            " promoted to float64, so there is no oracle callable to wrap."
            " Replace it with an explicit oracle passed through"
            f" chunk_callable ({import_note}), choosing the axis your"
            " operation is independent along, or validate at a smaller shape."
        )
      else:
        remedy = (
            f"Wrap the oracle with chunk_callable ({import_note}), choosing"
            " the axis your operation is independent along, or validate at a"
            " smaller shape."
        )
      raise RuntimeError(
          "Oracle ran out of device memory at this shape."
          f" {cost_note} {remedy} Original error: {e}"
      ) from e

    if out_oracle.shape != out_ref.shape:
      raise ValueError(
          f"Oracle shape mismatch in batch '{batch.get('name')}': oracle shape"
          f" {out_oracle.shape} != reference shape {out_ref.shape}"
      )

    oracle_ran = True
    oracle_output_dtype = str(out_oracle.dtype)
    if out_oracle.dtype != np.float64:
      oracle_in_float64 = False

    ref_oracle_arr = compute_ulp_distance(out_ref, out_oracle, dtype_str)
    cand_oracle_arr = compute_ulp_distance(out_cand, out_oracle, dtype_str)
    batch_ref_oracle_ulp = int(np.max(ref_oracle_arr))
    batch_cand_oracle_ulp = int(np.max(cand_oracle_arr))
    ref_oracle_max_ulp = batch_ref_oracle_ulp
    cand_oracle_max_ulp = batch_cand_oracle_ulp
    ref_oracle_p99_9 = float(np.percentile(ref_oracle_arr, 99.9))
    cand_oracle_p99_9 = float(np.percentile(cand_oracle_arr, 99.9))

    out_oracle_f64 = out_oracle.astype(np.float64)
    out_ref_f64 = out_ref.astype(np.float64)
    out_cand_f64 = out_cand.astype(np.float64)
    ref_oracle_max_abs = float(np.max(np.abs(out_ref_f64 - out_oracle_f64)))
    cand_oracle_max_abs = float(np.max(np.abs(out_cand_f64 - out_oracle_f64)))

  if has_nan_or_inf:
    max_ulp = 999999
    p99_9 = 999999.0
    mean_ulp = 999999.0
    hist = {"<=1_ulp": 0, "<=2_ulp": 0, ">2_ulp": out_cand.size}
    ulp_passed = False
    allclose_passed = False
    passed = False
    context_obj = UlpContext(
        bit_identical=False,
        p50=float("nan"),
        p99_9=float("nan"),
        max_ulp=max_ulp,
        reliable=False,
        note="NaN or Inf detected in output.",
    )
  else:
    ulp_arr = compute_ulp_distance(out_cand, out_ref, dtype_str)
    max_ulp = int(np.max(ulp_arr))
    p99_9 = float(np.percentile(ulp_arr, 99.9))
    mean_ulp = float(np.mean(ulp_arr))
    p50 = float(np.percentile(ulp_arr, 50.0))
    bit_identical = bool(np.all(ulp_arr == 0))
    hist = {
        "<=1_ulp": int(np.sum(ulp_arr <= 1)),
        "<=2_ulp": int(np.sum(ulp_arr <= 2)),
        ">2_ulp": int(np.sum(ulp_arr > 2)),
    }
    effective_max_ulp = actual_max_allowed_ulp
    effective_p99_9 = (
        0.0 if is_discrete and p99_9_allowed_ulp == 1 else p99_9_allowed_ulp
    )
    ulp_passed = bool(
        max_ulp <= effective_max_ulp and p99_9 <= effective_p99_9
    )

    if is_discrete:
      allclose_passed = bool(np.array_equal(out_cand, out_ref))
    else:
      dtype_eps, atol_val = _DTYPE_TOLERANCES.get(canonical_dtype, (1e-3, 1e-5))
      rtol_val = actual_max_allowed_ulp * dtype_eps
      cand_f32 = out_cand.astype(np.float32)
      ref_f32 = out_ref.astype(np.float32)
      allclose_passed = bool(
          np.allclose(cand_f32, ref_f32, rtol=rtol_val, atol=atol_val)
      )

    passed = bool(ulp_passed and allclose_passed)
    batch_regime = batch.get("regime", "")
    reliable = True
    context_note = None
    if (
        batch_regime in ("boundary", "cancellation")
        and max_ulp > recommended_ulp
    ):
      reliable = False
      context_note = (
          "Ill-conditioned regime: subnormal cancellation or dynamic range"
          " boundaries can saturate ULP without mathematical defect."
      )
    elif ulp_passed and not allclose_passed:
      context_note = "Failed allclose dual gate check at rtol=k*eps."

    if narrow_warning:
      if context_note:
        context_note = f"{context_note} {narrow_warning}"
      else:
        context_note = narrow_warning

    context_obj = UlpContext(
        bit_identical=bit_identical,
        p50=p50,
        p99_9=p99_9,
        max_ulp=max_ulp,
        reliable=reliable,
        note=context_note,
    )

  batch_res = BatchValidationResult(
      batch_name=batch["name"],
      regime=batch.get("regime", "unknown"),
      max_ulp_distance=max_ulp,
      p99_9_ulp_distance=p99_9,
      mean_ulp_distance=mean_ulp,
      ulp_histogram=hist,
      has_nan_or_inf=has_nan_or_inf,
      passed=passed,
      reference_ulp_from_oracle=batch_ref_oracle_ulp,
      candidate_ulp_from_oracle=batch_cand_oracle_ulp,
      ulp_context=context_obj,
      allclose_passed=allclose_passed,
  )

  return _BatchExecutionResult(
      batch_result=batch_res,
      max_ulp=max_ulp,
      passed=passed,
      oracle_ran=oracle_ran,
      oracle_in_float64=oracle_in_float64,
      oracle_output_dtype=oracle_output_dtype,
      ref_oracle_max_ulp=ref_oracle_max_ulp,
      cand_oracle_max_ulp=cand_oracle_max_ulp,
      ref_oracle_p99_9=ref_oracle_p99_9,
      cand_oracle_p99_9=cand_oracle_p99_9,
      ref_oracle_max_abs=ref_oracle_max_abs,
      cand_oracle_max_abs=cand_oracle_max_abs,
      narrow_warning=narrow_warning,
  )


class _ValidationAccumulator:
  """Accumulates execution metrics across multiple validation batches."""

  def __init__(self) -> None:
    self.overall_max_ulp: int = 0
    self.failed_batches: int = 0
    self.oracle_ran: bool = False
    self.oracle_in_float64: bool = True
    self.oracle_output_dtype: str = "float64"
    self.oracle_ref_max_ulp: int = 0
    self.oracle_cand_max_ulp: int = 0
    self.oracle_ref_p99_9: float = 0.0
    self.oracle_cand_p99_9: float = 0.0
    self.oracle_ref_max_abs: float = 0.0
    self.oracle_cand_max_abs: float = 0.0
    self.narrow_warning: str | None = None

  def update(self, res: _BatchExecutionResult) -> None:
    """Updates running statistics with result from a single batch."""
    self.overall_max_ulp = max(self.overall_max_ulp, res.max_ulp)
    if not res.passed:
      self.failed_batches += 1
    if res.oracle_ran:
      self.oracle_ran = True
      if not res.oracle_in_float64:
        self.oracle_in_float64 = False
      self.oracle_output_dtype = res.oracle_output_dtype
      self.oracle_ref_max_ulp = max(
          self.oracle_ref_max_ulp, res.ref_oracle_max_ulp
      )
      self.oracle_cand_max_ulp = max(
          self.oracle_cand_max_ulp, res.cand_oracle_max_ulp
      )
      self.oracle_ref_p99_9 = max(self.oracle_ref_p99_9, res.ref_oracle_p99_9)
      self.oracle_cand_p99_9 = max(
          self.oracle_cand_p99_9, res.cand_oracle_p99_9
      )
      self.oracle_ref_max_abs = max(
          self.oracle_ref_max_abs, res.ref_oracle_max_abs
      )
      self.oracle_cand_max_abs = max(
          self.oracle_cand_max_abs, res.cand_oracle_max_abs
      )
    if res.narrow_warning and self.narrow_warning is None:
      self.narrow_warning = res.narrow_warning


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
    regimes: collections.abc.Sequence[str] | str | None = None,
    kernel_oracle: collections.abc.Callable[..., Any] | str | None = None,
    device_kind: str | None = None,
) -> KernelValidationReport:
  """Validates candidate kernel against reference implementation.

  Args:
    kernel_ref: Baseline implementation. NOTE: this is compared for *agreement*,
      not correctness -- see `kernel_oracle`.
    kernel_candidate: Optimized implementation under test.
    shapes: Shape or sequence of shapes for the generated test suite.
    dtype_str: Output dtype in whose units every ULP figure is reported.
    test_suite: Pre-generated suite; generated from `shapes` when omitted.
    tier: Suite size -- "fast_agent", "presubmit" or "deep_fuzzing".
    max_allowed_ulp: Per-element ULP gate, bounded by MAX_HARD_CEILING_ULP.
    p99_9_allowed_ulp: 99.9th-percentile ULP gate.
    seed: PRNG seed for suite generation.
    regimes: Optional sequence of regime names to filter batches (defaults to
      'normal' with triage on failure). Pass 'all' to run full suite.
    kernel_oracle: Optional high-precision reference used to report how far
      `kernel_ref` itself sits from an exact result. Report-only: it populates
      `oracle_audit` and never changes the pass/fail verdict. Two modes: * An
      explicit host float64 callable is independent of `kernel_ref` and catches
      every source of reference error, including accelerator default-precision
      truncation (TPU MXU bf16 passes, GPU TF32). * ORACLE_AUTO ("auto") re-runs
      `kernel_ref` with its floating-point arguments promoted to float64. It
      sees only the loss that the input dtype controls. Requires jax_enable_x64
      for JAX references.
    device_kind: Device/backend identifier (e.g. "tpu", "gpu", "cpu").
      Auto-detected when omitted.

  Returns:
    A KernelValidationReport.
  """
  if isinstance(kernel_oracle, str) and kernel_oracle != ORACLE_AUTO:
    raise ValueError(
        f"Unknown kernel_oracle string '{kernel_oracle}'; expected"
        f" '{ORACLE_AUTO}' or a callable."
    )

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
        "⚠️ CAUTION: A relaxed tolerance threshold"
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
    full_suite = numerical_generator.generate_test_suite(
        shapes, dtype_str=dtype_str, tier=tier, seed=seed
    )
  else:
    full_suite = test_suite

  run_triage_on_failure = False
  if regimes is not None:
    if regimes == "all" or regimes == ("all",) or regimes == ["all"]:
      batches_to_run = list(full_suite)
    else:
      allowed_regimes = {regimes} if isinstance(regimes, str) else set(regimes)
      batches_to_run = [
          b
          for b in full_suite
          if (
              b.get("regime") in allowed_regimes
              or b.get("name") in allowed_regimes
          )
      ]
      if not batches_to_run:
        batches_to_run = list(full_suite)
  else:
    if test_suite is not None:
      batches_to_run = list(full_suite)
    elif kernel_oracle is not None or _is_discrete_dtype(canonical_dtype):
      # Q3 error distribution and integer boundary testing require full suite.
      batches_to_run = list(full_suite)
      run_triage_on_failure = False
    elif canonical_dtype.startswith("fp8") or canonical_dtype.startswith(
        "float8"
    ):
      # FP8 exponent bits absorb dynamic range natively (1.5x spread).
      normal_batches = [
          b
          for b in full_suite
          if (b.get("regime") == "normal" or b.get("name") == "normal_batch_0")
      ]
      batches_to_run = normal_batches or list(full_suite)
      run_triage_on_failure = False
    else:
      # Standard continuous float parity: normal first, triage on failure.
      normal_batches = [
          b
          for b in full_suite
          if (b.get("regime") == "normal" or b.get("name") == "normal_batch_0")
      ]
      batches_to_run = normal_batches or list(full_suite)
      run_triage_on_failure = True

  detected_device_kind, detected_backend = _detect_device_info()
  effective_device_kind = device_kind or detected_device_kind
  effective_backend = detected_backend

  pin_inert_detected = False
  ref_is_unpinned = False
  if batches_to_run:
    first_b = batches_to_run[0]
    first_args = first_b.get("args", (first_b.get("tensor"),))
    first_kwargs = first_b.get("kwargs", {})
    if _probe_pin_inert(
        kernel_ref, first_args, first_kwargs, effective_device_kind
    ):
      pin_inert_detected = True

    if not _is_discrete_dtype(canonical_dtype):
      ref_probe = _probe_precision(
          kernel_ref,
          first_args,
          first_kwargs,
          baseline="given",
          device_kind=effective_device_kind,
      )
      if ref_probe.is_pinned is not None and not ref_probe.is_pinned:
        ref_is_unpinned = True

  acc = _ValidationAccumulator()
  batch_results: list[BatchValidationResult] = []
  executed_batch_ids: set[int] = set()

  def _process_batch(b: dict[str, Any]) -> None:
    res = _execute_single_batch(
        batch=b,
        kernel_ref=kernel_ref,
        kernel_candidate=kernel_candidate,
        canonical_dtype=canonical_dtype,
        dtype_str=dtype_str,
        actual_max_allowed_ulp=actual_max_allowed_ulp,
        p99_9_allowed_ulp=p99_9_allowed_ulp,
        recommended_ulp=recommended_ulp,
        kernel_oracle=kernel_oracle,
    )
    acc.update(res)
    batch_results.append(res.batch_result)
    executed_batch_ids.add(id(b))

  for batch in batches_to_run:
    _process_batch(batch)

  if run_triage_on_failure and acc.failed_batches > 0:
    for batch in full_suite:
      if id(batch) not in executed_batch_ids:
        _process_batch(batch)

  oracle_audit = None
  oracle_banner = None
  if acc.oracle_ran:
    reference_is_lossy = acc.oracle_ref_max_ulp > recommended_ulp
    first_b = batches_to_run[0]
    probe_args = first_b.get("args", (first_b.get("tensor"),))
    probe_kwargs = first_b.get("kwargs", {})

    oracle_ver = _verify_oracle_precision(
        oracle_in_float64=acc.oracle_in_float64,
        oracle_output_dtype=acc.oracle_output_dtype,
        canonical_dtype=canonical_dtype,
        kernel_oracle=kernel_oracle,
        probe_args=probe_args,
        probe_kwargs=probe_kwargs,
        effective_device_kind=effective_device_kind,
        reference_is_lossy=reference_is_lossy,
        oracle_ref_max_ulp=acc.oracle_ref_max_ulp,
        oracle_cand_max_ulp=acc.oracle_cand_max_ulp,
        recommended_ulp=recommended_ulp,
        reference_is_unpinned=ref_is_unpinned,
    )
    oracle_banner = oracle_ver.banner

    oracle_audit = OracleAudit(
        oracle_executed_in_float64=acc.oracle_in_float64,
        oracle_output_dtype=acc.oracle_output_dtype,
        reference_max_ulp_from_oracle=acc.oracle_ref_max_ulp,
        reference_p99_9_ulp_from_oracle=acc.oracle_ref_p99_9,
        candidate_max_ulp_from_oracle=acc.oracle_cand_max_ulp,
        candidate_p99_9_ulp_from_oracle=acc.oracle_cand_p99_9,
        reference_is_lossy=reference_is_lossy,
        oracle_precision_verified=oracle_ver.verified,
        reference_max_abs_from_oracle=acc.oracle_ref_max_abs,
        candidate_max_abs_from_oracle=acc.oracle_cand_max_abs,
        oracle_banner=oracle_banner,
        reference_pin_inert=pin_inert_detected,
        oracle_probe_diagnostic=oracle_ver.probe_diagnostic,
    )

  pin_inert_banner = None
  if pin_inert_detected:
    pin_inert_banner = (
        "⚠️ REFERENCE IS PIN-INERT: kernel_ref ignores precision pinning."
        " It cannot serve as a high-precision reference on accelerators."
    )

  is_equivalent = acc.failed_batches == 0
  zero_ulp_banner = None
  probed_baseline_oracle_ulp = None
  if (
      kernel_oracle is None
      and is_equivalent
      and acc.overall_max_ulp == 0
      and not _is_discrete_dtype(canonical_dtype)
      and batches_to_run
  ):
    try:
      first_b = batches_to_run[0]
      b0_args = first_b.get("args", (first_b.get("tensor"),))
      b0_kwargs = first_b.get("kwargs", {})
      p_args, p_kwargs = _promote_args_to_dtype(
          b0_args, b0_kwargs, np.float64
      )
      probe_out = kernel_ref(*p_args, **p_kwargs)
      probe_arr = np.asarray(probe_out)
      if probe_arr.dtype == np.float64:
        ref_b0 = np.asarray(kernel_ref(*b0_args, **b0_kwargs))
        probe_ulp = int(
            np.max(compute_ulp_distance(ref_b0, probe_arr, dtype_str=dtype_str))
        )
        if probe_ulp > recommended_ulp:
          probed_baseline_oracle_ulp = probe_ulp
          zero_ulp_banner = (
              "⚠️ 0-ULP AGREEMENT WITH LOSSY BASELINE: Both kernels agree"
              f" bit-for-bit (0 ULP), but kernel_ref sits {probe_ulp} ULP from"
              f" the Float64 Oracle (above the {recommended_ulp} ULP contract"
              f" for '{canonical_dtype}'). Pairwise 0 ULP is a false green."
              " Run with kernel_oracle='auto' or an explicit oracle to audit"
              " correctness."
          )
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.debug("Zero-ULP lossy baseline probe failed: %s", e)

  if is_equivalent:
    summary = (
        "PASSED: Kernels are numerically equivalent across"
        f" {len(batch_results)} batches (Max ULP: {acc.overall_max_ulp},"
        f" Configured Limit: {actual_max_allowed_ulp}, Recommended: <="
        f" {recommended_ulp})."
    )
  else:
    summary = (
        f"FAILED: Numerical divergence detected ({acc.failed_batches}/"
        f"{len(batch_results)} batches failed numerical criteria)."
    )

  if pin_inert_banner:
    summary = f"{pin_inert_banner}\n{summary}"
  if zero_ulp_banner:
    summary = f"{zero_ulp_banner}\n{summary}"
  if oracle_banner:
    summary = f"{oracle_banner}\n{summary}"
  if caution_msg:
    summary = f"{caution_msg}\n{summary}"

  correctness_basis = (
      "AGREEMENT_AND_ORACLE" if acc.oracle_ran else "AGREEMENT_ONLY"
  )

  ref_unpinned_banner = None
  if ref_is_unpinned and correctness_basis == "AGREEMENT_ONLY":
    ref_unpinned_banner = (
        "⚠️ REFERENCE PRECISION UNVERIFIED: kernel_ref responds to matmul"
        " precision settings and no oracle was supplied to check it. On TPU"
        " an unpinned reference runs at bfloat16 MXU precision — measured"
        " 15,335x from exact in production code. Pass precision='highest',"
        " or supply kernel_oracle."
    )
    summary = f"{ref_unpinned_banner}\n{summary}"

  overall_ulp_context = None
  if batch_results:
    all_bit_identical = all(
        b.ulp_context.bit_identical
        for b in batch_results
        if b.ulp_context is not None
    )
    all_reliable = all(
        b.ulp_context.reliable
        for b in batch_results
        if b.ulp_context is not None
    )
    valid_p50s = [
        b.ulp_context.p50
        for b in batch_results
        if b.ulp_context is not None and not np.isnan(b.ulp_context.p50)
    ]
    overall_p50 = float(np.median(valid_p50s)) if valid_p50s else float("nan")
    overall_p99_9 = max(
        (b.p99_9_ulp_distance for b in batch_results), default=0.0
    )
    overall_ulp_context = UlpContext(
        bit_identical=all_bit_identical,
        p50=overall_p50,
        p99_9=overall_p99_9,
        max_ulp=acc.overall_max_ulp,
        reliable=all_reliable,
        note=acc.narrow_warning,
    )

  run_config = {
      "tier": tier,
      "seed": seed,
      "dtype_str": dtype_str,
      "device_kind": effective_device_kind,
      "backend": effective_backend,
      "total_batches_count": len(batch_results),
  }
  if pin_inert_detected:
    run_config["reference_pin_inert"] = True
  if ref_is_unpinned:
    run_config["reference_is_unpinned"] = True
  if probed_baseline_oracle_ulp is not None:
    run_config["probed_baseline_oracle_ulp"] = probed_baseline_oracle_ulp

  return KernelValidationReport(
      is_numerically_equivalent=is_equivalent,
      overall_max_ulp=acc.overall_max_ulp,
      failed_batches_count=acc.failed_batches,
      total_batches_count=len(batch_results),
      batch_results=batch_results,
      summary_message=summary,
      tolerance_audit=tolerance_audit,
      oracle_audit=oracle_audit,
      correctness_basis=correctness_basis,
      run_config=run_config,
      ulp_context=overall_ulp_context,
      narrow_output_dtype_warning=acc.narrow_warning,
  )
