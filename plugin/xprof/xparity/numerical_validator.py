"""Reusable library for comparing two kernel implementations on test suites.

Public API:
  validate_kernels: Compare a candidate kernel against a reference, optionally
    auditing the reference itself against a high-precision oracle. Pass
    `contract=CONTRACT_BITWISE` to require exact bit-pattern equality. Kernels
    may return pytrees; each leaf is validated, with optional per-leaf
    contracts via `contract_by_leaf`.
  compare_bitwise: Exact bit-pattern comparison of two arrays.
  make_fwd_bwd: Wrap a JAX function so its output also carries its VJP, which
    puts the backward pass under validation.
  chunk_callable: Wrap an oracle to execute in slices along one axis, for
    shapes whose high-precision intermediates exceed device memory.
  ORACLE_AUTO: `kernel_oracle` sentinel that re-runs the reference with its
    floating-point arguments promoted to float64.
"""

import collections.abc
import dataclasses
import functools
import importlib
import inspect
import logging
import sys
from typing import Any

import ml_dtypes
import numpy as np

from xprof.xparity import numerical_generator
from xprof.xparity import ulp


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

  @property
  def is_downcasting(self) -> bool:
    return self.reference_is_lossy

  @property
  def max_ulp_vs_fp64(self) -> int:
    return self.reference_max_ulp_from_oracle


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
class WorstOffender:
  """Spatial coordinate and value attribution of the element with largest ULP divergence."""

  max_ulp_index: tuple[int, ...]
  ref_value: float
  cand_value: float
  abs_diff: float
  rel_diff: float
  mismatch_count: int
  mismatch_ratio: float


@dataclasses.dataclass(frozen=True)
class BitwiseComparison:
  """Exact bit-pattern comparison between a candidate and a reference.

  Unlike a 0-ULP gate, this distinguishes `-0.0` from `+0.0` and treats two
  NaNs with the same payload as equal.

  Attributes:
    equal: True when every element has the same bit pattern and dtype.
    diff_count: Number of elements whose bit patterns differ.
    diff_ratio: `diff_count` divided by the element count.
    first_diff_index: Index of the first differing element in C order.
    reference_bits: Hex bit pattern of the reference at `first_diff_index`.
    candidate_bits: Hex bit pattern of the candidate at `first_diff_index`.
    note: Explanation when the comparison could not be made element-wise, for
      example on a dtype mismatch.
  """

  equal: bool
  diff_count: int
  diff_ratio: float
  first_diff_index: tuple[int, ...] | None = None
  reference_bits: str | None = None
  candidate_bits: str | None = None
  note: str | None = None


@dataclasses.dataclass(frozen=True)
class BatchValidationResult:
  """Validation metrics and status for a single test batch.

  For kernels that return a pytree (tuple, list, dict or registered JAX
  container), each output leaf is validated on its own and stored in
  `leaf_results`, keyed by its JAX key path (for example `[0]` or `['lse']`).
  The batch-level metrics then come from the leaf named in `leaf_path`: the
  first failing leaf, or the leaf with the largest ULP distance when all pass.
  `passed` is True only when every leaf passes.
  """

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
  nan_count: int = 0
  inf_count: int = 0
  first_non_finite_index: tuple[int, ...] | None = None
  finite_max_ulp: int | None = None
  worst_offender: WorstOffender | None = None
  bitwise: BitwiseComparison | None = None
  leaf_path: str | None = None
  leaf_results: dict[str, "BatchValidationResult"] | None = None

  @property
  def max_ulp(self) -> int:
    return self.max_ulp_distance

  @property
  def mean_ulp(self) -> float:
    return self.mean_ulp_distance

  @property
  def p99_9_ulp(self) -> float:
    return self.p99_9_ulp_distance

  @property
  def has_nan_inf(self) -> bool:
    return self.has_nan_or_inf


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
  shape_mismatch: dict[str, Any] | None = None


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


# The dtype tables and ULP arithmetic live in `ulp`, which has no google3
# dependencies so that standalone benchmark scripts can carry it inline. These
# names are re-exported so that callers of this module keep working.
RECOMMENDED_CONTRACT_ULP = ulp.RECOMMENDED_CONTRACT_ULP
MAX_HARD_CEILING_ULP = ulp.MAX_HARD_CEILING_ULP
compute_ulp_distance = ulp.compute_ulp_distance
get_contract = ulp.get_contract
resolve_canonical_dtype = ulp.resolve_canonical_dtype
# Deprecated private aliases retained for in-flight callers.
_resolve_canonical_dtype = ulp.resolve_canonical_dtype
_get_finfo = ulp.get_finfo
_is_discrete_dtype = ulp.is_discrete_dtype
_INTEGER_DTYPES = ulp.INTEGER_DTYPES
_DTYPE_TOLERANCES = ulp.DTYPE_TOLERANCES


def _as_compare_float(arr: np.ndarray) -> np.ndarray:
  """Returns `arr` in a dtype NumPy's allclose accepts, avoiding copies.

  float32 and wider are already comparable, so they are returned as-is.
  Narrow types (bfloat16, the float8 formats, float16) are widened to
  float32, which is lossless for all of them.

  Args:
    arr: Input array to convert for comparison.

  Returns:
    An array with dtype float32 or wider suitable for NumPy's allclose.
  """
  if arr.dtype in (np.float32, np.float64):
    return arr
  return arr.astype(np.float32)


@functools.lru_cache(maxsize=None)
def _rel_diff_floor(canonical: str) -> float:
  """Smallest normal magnitude for a dtype, used to guard relative division."""
  try:
    return float(_get_finfo(np.dtype(canonical)).smallest_normal)
  except (TypeError, ValueError, AttributeError) as e:
    logging.debug("No finfo for dtype %s: %s", canonical, e)
    return float(np.finfo(np.float32).smallest_normal)


def _relative_diff(abs_diff: float, ref_value: float, canonical: str) -> float:
  """Computes a relative difference guarded by the dtype's own scale.

  The previous guard added a fixed 1e-12 to the denominator. That constant is
  many orders of magnitude larger than the smallest representable value of
  every narrow dtype -- bfloat16's smallest normal is ~1.18e-38 -- so for any
  reference value near the bottom of the range the guard dominated the
  denominator and drove the reported ratio toward zero. A kernel that was
  100% wrong at a small magnitude was reported as having ~1e-26 relative
  error, which reads as a pass.

  Flooring the denominator at the dtype's own smallest normal keeps the ratio
  meaningful across the entire representable range while still preventing
  division by zero.

  Args:
    abs_diff: The absolute difference at the element of interest.
    ref_value: The reference value at that element.
    canonical: The canonical dtype name.

  Returns:
    The guarded relative difference.
  """
  denom = max(abs(ref_value), _rel_diff_floor(canonical))
  return abs_diff / denom


# Verdict contracts accepted by `validate_kernels` and `validate_arrays`.
# CONTRACT_ULP gates on ULP distance and an allclose check. CONTRACT_BITWISE
# requires every output element to have the same bit pattern as the reference.
CONTRACT_ULP = "ulp"
CONTRACT_BITWISE = "bitwise"
_CONTRACTS = frozenset({CONTRACT_ULP, CONTRACT_BITWISE})


def _check_contract(contract: str) -> None:
  if contract not in _CONTRACTS:
    raise ValueError(
        f"Unknown contract '{contract}'; expected one of {sorted(_CONTRACTS)}."
    )


def _hex_bits(element_bytes: np.ndarray) -> str:
  """Formats the raw bytes of one element as a fixed-width hex integer."""
  value = int.from_bytes(element_bytes.tobytes(), sys.byteorder)
  return f"0x{value:0{2 * element_bytes.size}x}"


def compare_bitwise(actual: Any, expected: Any) -> BitwiseComparison:
  """Compares two arrays for exact bit-pattern equality.

  A 0-ULP gate is not a bitwise gate: `compute_ulp_distance` maps `-0.0` and
  `+0.0` to the same index, and ULP statistics are undefined for NaN. This
  function compares raw bytes instead, so it is the check to use when a
  candidate must reproduce a reference exactly.

  Args:
    actual: The candidate tensor.
    expected: The reference tensor.

  Returns:
    A BitwiseComparison. Arrays with different dtypes are reported as unequal
    in every element, with an explanatory `note`.

  Raises:
    ValueError: If the shapes differ.
  """
  act = np.asarray(actual)
  exp = np.asarray(expected)
  if act.shape != exp.shape:
    raise ValueError(
        f"Shape mismatch in compare_bitwise: {act.shape} vs {exp.shape}"
    )
  size = int(act.size)
  if act.dtype != exp.dtype:
    return BitwiseComparison(
        equal=False,
        diff_count=size,
        diff_ratio=1.0 if size else 0.0,
        note=(
            f"dtype mismatch: candidate {act.dtype} vs reference {exp.dtype}."
            " Bitwise equality requires identical output dtypes."
        ),
    )
  if size == 0:
    return BitwiseComparison(equal=True, diff_count=0, diff_ratio=0.0)

  diff_mask = ulp.bitwise_mismatch_mask(act, exp).reshape(-1)
  diff_count = int(np.count_nonzero(diff_mask))
  if diff_count == 0:
    return BitwiseComparison(equal=True, diff_count=0, diff_ratio=0.0)

  flat_first = int(np.argmax(diff_mask))

  def _element_bytes(arr: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(arr).reshape(-1)[flat_first : flat_first + 1]

  return BitwiseComparison(
      equal=False,
      diff_count=diff_count,
      diff_ratio=float(diff_count) / float(size),
      first_diff_index=tuple(
          int(x) for x in np.unravel_index(flat_first, act.shape)
      ),
      reference_bits=_hex_bits(_element_bytes(exp).view(np.uint8)),
      candidate_bits=_hex_bits(_element_bytes(act).view(np.uint8)),
  )


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
      torch: Any = sys.modules["torch"]
      cuda: Any = getattr(torch, "cuda", None)
      if cuda and getattr(cuda, "is_available", lambda: False)():
        get_device_name = getattr(cuda, "get_device_name", None)
        device_name = get_device_name(0) if callable(get_device_name) else "0"
        return f"cuda:{device_name}", "cuda"
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.debug("Failed querying loaded PyTorch CUDA device info: %s", e)
  else:
    try:
      torch: Any = importlib.import_module("torch")
      cuda: Any = getattr(torch, "cuda", None)
      if cuda and getattr(cuda, "is_available", lambda: False)():
        get_device_name = getattr(cuda, "get_device_name", None)
        device_name = get_device_name(0) if callable(get_device_name) else "0"
        return f"cuda:{device_name}", "cuda"
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
          "Host NumPy execution: not subject to accelerator matmul precision",
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


def make_fwd_bwd(
    fn: collections.abc.Callable[..., Any],
    argnums: int | collections.abc.Sequence[int] | None = None,
    cotangent_seed: int = 0,
) -> collections.abc.Callable[..., dict[str, Any]]:
  """Wraps a JAX function so that validation covers its backward pass.

  The wrapper returns `{"out": fn(*args), "vjp": grads}`, where `grads` holds
  one vector-Jacobian product per differentiated argument. The output
  cotangent is drawn from N(0, 1) with a fixed seed and the shape and dtype of
  each output leaf, so a reference and a candidate with the same output
  structure receive the same cotangent. Wrap both kernels and pass them to
  `validate_kernels`. Each result is then validated as its own leaf (`['out']`,
  `['vjp'][0]`, ...), and `contract_by_leaf` can set a contract per leaf.

  Args:
    fn: A function that `jax.vjp` can differentiate.
    argnums: Positional arguments to differentiate. Defaults to every positional
      argument with a floating-point or complex dtype.
    cotangent_seed: Seed for the output cotangent.

  Returns:
    A callable that takes the same arguments as `fn`.
  """
  jax = importlib.import_module("jax")
  jnp = jax.numpy

  def _is_inexact(value: Any) -> bool:
    try:
      return bool(jnp.issubdtype(jnp.asarray(value).dtype, jnp.inexact))
    except TypeError:
      return False

  def _cotangent(index: int, leaf: Any) -> Any:
    if not jnp.issubdtype(leaf.dtype, jnp.inexact):
      return np.zeros(leaf.shape, dtype=jax.dtypes.float0)
    rng = np.random.default_rng(cotangent_seed + index)
    return jnp.asarray(
        rng.standard_normal(leaf.shape).astype(np.float32), dtype=leaf.dtype
    )

  @functools.wraps(fn)
  def fwd_bwd(*args: Any, **kwargs: Any) -> dict[str, Any]:
    if argnums is None:
      diff = tuple(i for i, a in enumerate(args) if _is_inexact(a))
    elif isinstance(argnums, int):
      diff = (argnums,)
    else:
      diff = tuple(argnums)
    if not diff:
      raise ValueError(
          "make_fwd_bwd found no floating-point positional argument to"
          " differentiate. Pass argnums explicitly."
      )

    def partial_fn(*diff_args: Any) -> Any:
      full = list(args)
      for i, value in zip(diff, diff_args):
        full[i] = value
      return fn(*full, **kwargs)

    out, vjp_fn = jax.vjp(partial_fn, *(jnp.asarray(args[i]) for i in diff))
    leaves, treedef = jax.tree_util.tree_flatten(out)
    cotangent = jax.tree_util.tree_unflatten(
        treedef, [_cotangent(i, leaf) for i, leaf in enumerate(leaves)]
    )
    return {"out": out, "vjp": tuple(vjp_fn(cotangent))}

  return fwd_bwd


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
    has_width_margin = (
        hasattr(ref_finfo, "eps") and float(ref_finfo.eps) >= 100.0 * oracle_eps
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
          "⚠️ ORACLE IS NOT PRECISION-PINNED: The oracle returned"
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
          "⚠️ ORACLE LACKS NUMERICAL PRECISION: The oracle returned"
          f" '{oracle_output_dtype}' (eps {oracle_eps:.2e}). An oracle must"
          " be at least float32 (or float64) to serve as a ground truth"
          " reference. Low-precision oracles cannot establish correctness."
      )
    elif not has_precision_margin:
      verified = False
      banner = (
          "⚠️ ORACLE NOT PRECISE ENOUGH: The oracle returned"
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
          "⚠️ ORACLE PRECISION UNDETERMINED: The oracle returned"
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


def _flatten_output_fallback(value: Any, path: str) -> list[tuple[str, Any]]:
  """Flattens tuples, lists, namedtuples and dicts without JAX."""
  if isinstance(value, dict):
    pairs = []
    for key in sorted(value):
      pairs.extend(_flatten_output_fallback(value[key], f"{path}[{key!r}]"))
    return pairs
  if isinstance(value, tuple) and hasattr(value, "_fields"):
    pairs = []
    for field in getattr(value, "_fields"):
      pairs.extend(
          _flatten_output_fallback(getattr(value, field), f"{path}.{field}")
      )
    return pairs
  if isinstance(value, (tuple, list)):
    pairs = []
    for index, item in enumerate(value):
      pairs.extend(_flatten_output_fallback(item, f"{path}[{index}]"))
    return pairs
  if value is None:
    return []
  return [(path, value)]


def _flatten_output(value: Any) -> list[tuple[str, Any]]:
  """Flattens a kernel output into `(key_path, leaf)` pairs.

  A bare array yields one pair with an empty path. Containers use JAX key-path
  notation (`[0]`, `['lse']`, `.field`), so `contract_by_leaf` keys are the
  same whether or not JAX is loaded. When JAX is loaded, registered custom
  pytree nodes are flattened as well.

  Args:
    value: The kernel output.

  Returns:
    The leaves in deterministic order with their key paths.
  """
  jax = sys.modules.get("jax")
  if jax is not None:
    tree_util = jax.tree_util
    pairs = tree_util.tree_flatten_with_path(value)[0]
    return [(tree_util.keystr(path), leaf) for path, leaf in pairs]
  return _flatten_output_fallback(value, "")


def _is_pytree_output(leaves: list[tuple[str, Any]]) -> bool:
  return len(leaves) != 1 or bool(leaves[0][0])


def _leaf_selector(
    fn: collections.abc.Callable[..., Any],
    index: int,
    cached_args: tuple[Any, ...] | None = None,
    cached_kwargs: dict[str, Any] | None = None,
    cached_leaf: Any = None,
) -> collections.abc.Callable[..., Any]:
  """Returns a callable that yields leaf `index` of `fn`'s output.

  The call that produced `cached_leaf` is not repeated: when the arguments are
  the same objects as `cached_args`/`cached_kwargs`, the cached leaf is
  returned. Any other call (for example the float64-promoted oracle re-run)
  executes `fn` and selects the leaf.

  Args:
    fn: The kernel whose output is a pytree.
    index: Position of the leaf in `_flatten_output` order.
    cached_args: Positional arguments of the cached call, if any.
    cached_kwargs: Keyword arguments of the cached call.
    cached_leaf: The leaf returned by the cached call.
  """
  cached_kwargs = cached_kwargs or {}

  def _is_cached_call(args: tuple[Any, ...], kwargs: dict[str, Any]) -> bool:
    if cached_args is None or len(args) != len(cached_args):
      return False
    if kwargs.keys() != cached_kwargs.keys():
      return False
    return all(a is b for a, b in zip(args, cached_args)) and all(
        kwargs[k] is cached_kwargs[k] for k in kwargs
    )

  @functools.wraps(fn)
  def select(*args: Any, **kwargs: Any) -> Any:
    if _is_cached_call(args, kwargs):
      return cached_leaf
    return _flatten_output(fn(*args, **kwargs))[index][1]

  return select


def _primary_leaf_view(
    fn: collections.abc.Callable[..., Any],
) -> collections.abc.Callable[..., Any]:
  """Returns `fn` with pytree outputs reduced to their first leaf.

  Precision probes compare one array before and after pinning. For kernels
  with several outputs the first leaf is the representative. The wrapper keeps
  the signature and attributes of `fn` so `precision` detection still works.

  Args:
    fn: A kernel that may return a pytree.
  """

  @functools.wraps(fn)
  def view(*args: Any, **kwargs: Any) -> Any:
    out = fn(*args, **kwargs)
    leaves = _flatten_output(out)
    if not _is_pytree_output(leaves):
      return out
    return leaves[0][1] if leaves else out

  return view


def _execute_pytree_batch(
    batch: dict[str, Any],
    ref_leaves: list[tuple[str, Any]],
    cand_leaves: list[tuple[str, Any]],
    kernel_ref: collections.abc.Callable[..., Any],
    kernel_candidate: collections.abc.Callable[..., Any],
    kernel_oracle: collections.abc.Callable[..., Any] | str | None,
    contract: str,
    contract_by_leaf: collections.abc.Mapping[str, str],
    **gate_kwargs: Any,
) -> _BatchExecutionResult:
  """Validates every leaf of a pytree output and merges the results."""
  name = batch.get("name")
  ref_paths = [path for path, _ in ref_leaves]
  cand_paths = [path for path, _ in cand_leaves]
  if ref_paths != cand_paths:
    raise ValueError(
        f"Output structure mismatch in batch '{name}': candidate leaves"
        f" {cand_paths} != reference leaves {ref_paths}"
    )
  if not ref_paths:
    raise ValueError(f"Kernel returned no array leaves in batch '{name}'.")
  unknown = sorted(set(contract_by_leaf) - set(ref_paths))
  if unknown:
    raise ValueError(
        f"contract_by_leaf names unknown output leaves {unknown}. Available"
        f" leaves: {ref_paths}."
    )

  args = batch.get("args", (batch.get("tensor"),))
  kwargs = batch.get("kwargs", {})
  subs: list[_BatchExecutionResult] = []
  for index, (path, ref_leaf) in enumerate(ref_leaves):
    leaf_oracle = kernel_oracle
    if callable(kernel_oracle):
      leaf_oracle = _leaf_selector(kernel_oracle, index)
    sub = _execute_single_batch(
        batch=batch,
        kernel_ref=_leaf_selector(kernel_ref, index, args, kwargs, ref_leaf),
        kernel_candidate=_leaf_selector(
            kernel_candidate, index, args, kwargs, cand_leaves[index][1]
        ),
        kernel_oracle=leaf_oracle,
        contract=contract_by_leaf.get(path, contract),
        **gate_kwargs,
    )
    subs.append(sub)

  leaf_results = {
      path: dataclasses.replace(sub.batch_result, leaf_path=path)
      for path, sub in zip(ref_paths, subs)
  }
  worst_index = next(
      (i for i, sub in enumerate(subs) if not sub.passed),
      max(range(len(subs)), key=lambda i: subs[i].max_ulp),
  )
  passed = all(sub.passed for sub in subs)
  batch_res = dataclasses.replace(
      leaf_results[ref_paths[worst_index]],
      passed=passed,
      leaf_results=leaf_results,
  )
  oracle_subs = [sub for sub in subs if sub.oracle_ran]
  return _BatchExecutionResult(
      batch_result=batch_res,
      max_ulp=max(sub.max_ulp for sub in subs),
      passed=passed,
      oracle_ran=bool(oracle_subs),
      oracle_in_float64=all(sub.oracle_in_float64 for sub in oracle_subs),
      oracle_output_dtype=(
          oracle_subs[0].oracle_output_dtype if oracle_subs else ""
      ),
      ref_oracle_max_ulp=max(
          (sub.ref_oracle_max_ulp for sub in oracle_subs), default=0
      ),
      cand_oracle_max_ulp=max(
          (sub.cand_oracle_max_ulp for sub in oracle_subs), default=0
      ),
      ref_oracle_p99_9=max(
          (sub.ref_oracle_p99_9 for sub in oracle_subs), default=0.0
      ),
      cand_oracle_p99_9=max(
          (sub.cand_oracle_p99_9 for sub in oracle_subs), default=0.0
      ),
      ref_oracle_max_abs=max(
          (sub.ref_oracle_max_abs for sub in oracle_subs), default=0.0
      ),
      cand_oracle_max_abs=max(
          (sub.cand_oracle_max_abs for sub in oracle_subs), default=0.0
      ),
      narrow_warning=next(
          (sub.narrow_warning for sub in subs if sub.narrow_warning), None
      ),
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
    contract: str = CONTRACT_ULP,
    contract_by_leaf: collections.abc.Mapping[str, str] | None = None,
) -> _BatchExecutionResult:
  """Executes and validates a single batch between reference and candidate."""
  args = batch.get("args", (batch.get("tensor"),))
  kwargs = batch.get("kwargs", {})

  raw_ref = kernel_ref(*args, **kwargs)
  raw_cand = kernel_candidate(*args, **kwargs)
  ref_leaves = _flatten_output(raw_ref)
  cand_leaves = _flatten_output(raw_cand)
  if _is_pytree_output(ref_leaves) or _is_pytree_output(cand_leaves):
    return _execute_pytree_batch(
        batch=batch,
        ref_leaves=ref_leaves,
        cand_leaves=cand_leaves,
        kernel_ref=kernel_ref,
        kernel_candidate=kernel_candidate,
        kernel_oracle=kernel_oracle,
        contract=contract,
        contract_by_leaf=contract_by_leaf or {},
        canonical_dtype=canonical_dtype,
        dtype_str=dtype_str,
        actual_max_allowed_ulp=actual_max_allowed_ulp,
        p99_9_allowed_ulp=p99_9_allowed_ulp,
        recommended_ulp=recommended_ulp,
    )

  out_ref = np.asarray(raw_ref)
  out_cand = np.asarray(raw_cand)

  if out_cand.shape != out_ref.shape:
    raise ValueError(
        f"Shape mismatch in batch '{batch.get('name')}': candidate shape "
        f"{out_cand.shape} != reference shape {out_ref.shape}"
    )

  bitwise_cmp = compare_bitwise(out_cand, out_ref)

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

  nan_count = 0
  inf_count = 0
  first_non_finite_index: tuple[int, ...] | None = None
  finite_max_ulp: int | None = None

  if not is_discrete:
    out_cand_f32 = out_cand.astype(np.float32)
    out_ref_f32 = out_ref.astype(np.float32)
    cand_nan = np.isnan(out_cand_f32)
    ref_nan = np.isnan(out_ref_f32)
    cand_inf = np.isinf(out_cand_f32)
    ref_inf = np.isinf(out_ref_f32)
    nan_count = int(np.sum(cand_nan | ref_nan))
    inf_count = int(np.sum(cand_inf | ref_inf))
    has_nan_or_inf = bool(nan_count > 0 or inf_count > 0)
    if has_nan_or_inf:
      non_finite_mask = cand_nan | ref_nan | cand_inf | ref_inf
      non_finite_coords = np.argwhere(non_finite_mask)
      if non_finite_coords.size > 0:
        first_non_finite_index = tuple(int(x) for x in non_finite_coords[0])
      finite_mask = ~non_finite_mask
      if np.any(finite_mask):
        finite_ulp_arr = compute_ulp_distance(
            out_cand[finite_mask], out_ref[finite_mask], dtype_str
        )
        finite_max_ulp = int(np.max(finite_ulp_arr))
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
          "from"
          " google3.third_party.xprof.plugin.xprof.xparity"
          " import chunk_callable"
      )
      if isinstance(kernel_oracle, str):
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

  worst_offender: WorstOffender | None = None
  if has_nan_or_inf:
    max_ulp = 999999
    p99_9 = 999999.0
    mean_ulp = 999999.0
    hist = {"<=1_ulp": 0, "<=2_ulp": 0, ">2_ulp": out_cand.size}
    allclose_passed = False
    passed = False
    context_obj = UlpContext(
        bit_identical=bitwise_cmp.equal,
        p50=float("nan"),
        p99_9=float("nan"),
        max_ulp=max_ulp,
        reliable=False,
        note="NaN or Inf detected in output.",
    )
    if first_non_finite_index is not None:
      ref_v = float(
          np.asarray(out_ref, dtype=np.float64)[first_non_finite_index]
      )
      cand_v = float(
          np.asarray(out_cand, dtype=np.float64)[first_non_finite_index]
      )
      non_finite_total = nan_count + inf_count
      worst_offender = WorstOffender(
          max_ulp_index=first_non_finite_index,
          ref_value=ref_v,
          cand_value=cand_v,
          abs_diff=float("nan"),
          rel_diff=float("nan"),
          mismatch_count=non_finite_total,
          mismatch_ratio=(
              float(non_finite_total) / float(out_cand.size)
              if out_cand.size > 0
              else 0.0
          ),
      )
  else:
    ulp_arr = compute_ulp_distance(out_cand, out_ref, dtype_str)
    max_ulp = int(np.max(ulp_arr))
    finite_max_ulp = max_ulp
    p99_9 = float(np.percentile(ulp_arr, 99.9))
    mean_ulp = float(np.mean(ulp_arr))
    p50 = float(np.percentile(ulp_arr, 50.0))
    bit_identical = bitwise_cmp.equal
    hist = {
        "<=1_ulp": int(np.sum(ulp_arr <= 1)),
        "<=2_ulp": int(np.sum(ulp_arr <= 2)),
        ">2_ulp": int(np.sum(ulp_arr > 2)),
    }
    effective_max_ulp = actual_max_allowed_ulp
    effective_p99_9 = (
        0.0 if is_discrete and p99_9_allowed_ulp == 1 else p99_9_allowed_ulp
    )
    ulp_passed = bool(max_ulp <= effective_max_ulp and p99_9 <= effective_p99_9)

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

    if out_cand.size > 0:
      flat_max_idx = int(np.argmax(ulp_arr))
      max_idx = tuple(
          int(x) for x in np.unravel_index(flat_max_idx, ulp_arr.shape)
      )
      ref_v = float(np.asarray(out_ref, dtype=np.float64)[max_idx])
      cand_v = float(np.asarray(out_cand, dtype=np.float64)[max_idx])
      abs_d = abs(cand_v - ref_v)
      rel_d = abs_d / (abs(ref_v) + 1e-12)
      mismatch_cnt = int(np.sum(ulp_arr > effective_max_ulp))
      worst_offender = WorstOffender(
          max_ulp_index=max_idx,
          ref_value=ref_v,
          cand_value=cand_v,
          abs_diff=abs_d,
          rel_diff=rel_d,
          mismatch_count=mismatch_cnt,
          mismatch_ratio=float(mismatch_cnt) / float(out_cand.size),
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
    elif not ulp_passed and p99_9 <= effective_p99_9 and allclose_passed:
      context_note = (
          "NEAR_ZERO_MAX_ULP_OUTLIER: max_ulp exceeded threshold on"
          " tail/near-zero elements while p99.9 ULP and relative error"
          " (allclose) passed. Inspect mean_ulp_distance and"
          " p99_9_ulp_distance for reduction reorderings."
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

  if contract == CONTRACT_BITWISE:
    passed = bitwise_cmp.equal
    if bitwise_cmp.equal and has_nan_or_inf:
      # Identical bits are zero distance by definition, including NaN and Inf
      # positions that ULP statistics cannot measure.
      max_ulp = 0
      p99_9 = 0.0
      mean_ulp = 0.0
      hist = {"<=1_ulp": out_cand.size, "<=2_ulp": out_cand.size, ">2_ulp": 0}

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
      nan_count=nan_count,
      inf_count=inf_count,
      first_non_finite_index=first_non_finite_index,
      finite_max_ulp=finite_max_ulp,
      worst_offender=worst_offender,
      bitwise=bitwise_cmp,
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
    contract: str = CONTRACT_ULP,
    contract_by_leaf: collections.abc.Mapping[str, str] | None = None,
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
      Ignored for the verdict under CONTRACT_BITWISE.
    p99_9_allowed_ulp: 99.9th-percentile ULP gate. Ignored for the verdict under
      CONTRACT_BITWISE.
    seed: PRNG seed for suite generation.
    regimes: Optional sequence of regime names to filter batches. Under
      CONTRACT_ULP this defaults to 'normal' with triage on failure; under
      CONTRACT_BITWISE it defaults to the full suite, because bit equality must
      hold for every input. Pass 'all' to run the full suite.
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
    contract: CONTRACT_ULP (default) gates on ULP distance and allclose.
      CONTRACT_BITWISE passes only when every output element of the candidate
      has the same bit pattern and dtype as the reference. Use it for refactors
      and reduction-order-controlled rewrites that must be exact.
    contract_by_leaf: For kernels that return a pytree, overrides `contract` for
      individual output leaves, keyed by JAX key path (for example `{"[1]":
      CONTRACT_ULP}` or `{"['lse']": CONTRACT_BITWISE}`). Every leaf is
      validated; leaves not named here use `contract`. Unknown paths raise
      ValueError listing the available leaves.

  Returns:
    A KernelValidationReport. For pytree outputs each batch result carries
    per-leaf results in `leaf_results`.
  """
  _check_contract(contract)
  contract_by_leaf = dict(contract_by_leaf or {})
  for leaf_contract in contract_by_leaf.values():
    _check_contract(leaf_contract)
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
        available = sorted({
            str(b.get("regime")) for b in full_suite if b.get("regime")
        })
        raise ValueError(
            f"No test batches matched regimes {sorted(allowed_regimes)}."
            f" Available regimes for dtype '{canonical_dtype}': {available}."
            " Pass regimes='all' to run the full suite."
        )
  else:
    any_bitwise = contract == CONTRACT_BITWISE or any(
        c == CONTRACT_BITWISE for c in contract_by_leaf.values()
    )
    if test_suite is not None or any_bitwise:
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

  # Precision probes compare a single array; for pytree outputs they observe
  # the first leaf.
  probe_ref = _primary_leaf_view(kernel_ref)
  pin_inert_detected = False
  ref_is_unpinned = False
  if batches_to_run:
    first_b = batches_to_run[0]
    first_args = first_b.get("args", (first_b.get("tensor"),))
    first_kwargs = first_b.get("kwargs", {})
    if _probe_pin_inert(
        probe_ref, first_args, first_kwargs, effective_device_kind
    ):
      pin_inert_detected = True

    if not _is_discrete_dtype(canonical_dtype):
      ref_probe = _probe_precision(
          probe_ref,
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
        contract=contract,
        contract_by_leaf=contract_by_leaf,
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
        kernel_oracle=(
            _primary_leaf_view(kernel_oracle)
            if callable(kernel_oracle)
            else kernel_oracle
        ),
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
      p_args, p_kwargs = _promote_args_to_dtype(b0_args, b0_kwargs, np.float64)
      probe_out = probe_ref(*p_args, **p_kwargs)
      probe_arr = np.asarray(probe_out)
      if probe_arr.dtype == np.float64:
        ref_b0 = np.asarray(probe_ref(*b0_args, **b0_kwargs))
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

  if contract == CONTRACT_BITWISE:
    caution_msg = None
    if is_equivalent:
      summary = (
          "PASSED: Candidate is bit-identical to the reference across"
          f" {len(batch_results)} batches (contract: bitwise)."
      )
    else:
      first_bad = next(b for b in batch_results if not b.passed)
      bw = first_bad.bitwise
      leaf = (
          f" output leaf {first_bad.leaf_path}," if first_bad.leaf_path else ""
      )
      if bw is not None and bw.first_diff_index is not None:
        where = (
            f" First difference: batch '{first_bad.batch_name}' (regime"
            f" '{first_bad.regime}'),{leaf} at index {bw.first_diff_index},"
            f" reference bits {bw.reference_bits}, candidate bits"
            f" {bw.candidate_bits}; {bw.diff_count} elements differ, max ULP"
            f" {first_bad.max_ulp_distance}."
        )
      else:
        note = bw.note if bw is not None else "no bitwise detail."
        where = f" Batch '{first_bad.batch_name}',{leaf} {note}"
      summary = (
          "FAILED: Candidate differs bitwise from the reference in"
          f" {acc.failed_batches}/{len(batch_results)} batches"
          f" (contract: bitwise).{where}"
      )
  elif is_equivalent:
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
    overall_note = acc.narrow_warning
    for b in batch_results:
      if (
          b.ulp_context is not None
          and b.ulp_context.note
          and "NEAR_ZERO_MAX_ULP_OUTLIER" in b.ulp_context.note
      ):
        if overall_note:
          overall_note = f"{overall_note} {b.ulp_context.note}"
        else:
          overall_note = b.ulp_context.note
        if "NEAR_ZERO_MAX_ULP_OUTLIER" not in summary:
          summary = f"{summary}\n{b.ulp_context.note}"
        break
    overall_ulp_context = UlpContext(
        bit_identical=all_bit_identical,
        p50=overall_p50,
        p99_9=overall_p99_9,
        max_ulp=acc.overall_max_ulp,
        reliable=all_reliable,
        note=overall_note,
    )

  run_config: dict[str, Any] = {
      "tier": tier,
      "seed": seed,
      "dtype_str": dtype_str,
      "device_kind": effective_device_kind,
      "backend": effective_backend,
      "total_batches_count": len(batch_results),
      "contract": contract,
  }
  if contract_by_leaf:
    run_config["contract_by_leaf"] = contract_by_leaf
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


def validate_arrays(
    actual: Any,
    expected: Any,
    dtype_str: str = "bfloat16",
    max_allowed_ulp: int | None = None,
    p99_9_allowed_ulp: float = 1.0,
    contract: str = CONTRACT_ULP,
) -> BatchValidationResult:
  """Validates bitwise ULP parity between two pre-computed arrays.

  Args:
    actual: The candidate tensor.
    expected: The reference tensor.
    dtype_str: Any spelling accepted by `resolve_canonical_dtype`.
    max_allowed_ulp: Per-element ULP gate. Defaults to the dtype's recommended
      contract; may not exceed its immutable hard safety ceiling.
    p99_9_allowed_ulp: Gate on the 99.9th percentile of the ULP distribution.
    contract: CONTRACT_ULP (default) or CONTRACT_BITWISE. Under CONTRACT_BITWISE
      `passed` is True only when both arrays have the same dtype and every
      element has the same bit pattern.

  Returns:
    A BatchValidationResult. When either tensor contains non-finite values the
    result carries the structural taxonomy (`nan_count`, `inf_count`,
    `first_non_finite_index`) alongside `finite_max_ulp` computed over the
    finite subset, rather than a sentinel magnitude.
  """
  _check_contract(contract)
  act_np = np.asarray(actual)
  exp_np = np.asarray(expected)
  canonical = resolve_canonical_dtype(dtype_str)
  recommended_ulp, hard_ceiling = get_contract(canonical)

  if max_allowed_ulp is None:
    limit_ulp = recommended_ulp
  else:
    limit_ulp = int(max_allowed_ulp)

  if limit_ulp > hard_ceiling:
    raise ValueError(
        f"Requested max_allowed_ulp={limit_ulp} exceeds immutable safety"
        f" ceiling ({hard_ceiling}) for dtype '{canonical}'."
    )

  if act_np.shape != exp_np.shape:
    raise ValueError(
        f"Shape mismatch in validate_arrays: {act_np.shape} vs {exp_np.shape}"
    )

  is_discrete = _is_discrete_dtype(canonical)

  # Non-finite detection runs in the native dtype. ml_dtypes registers isnan
  # and isinf ufuncs for bfloat16 and the float8 formats, so upcasting to
  # float64 first -- which costs 8 bytes per element for each of the two
  # tensors -- buys nothing.
  nan_count = 0
  inf_count = 0
  first_non_finite_index = None
  non_finite_mask = None
  if not is_discrete:
    nan_mask = np.isnan(act_np) | np.isnan(exp_np)
    inf_mask = np.isinf(act_np) | np.isinf(exp_np)
    nan_count = int(np.count_nonzero(nan_mask))
    inf_count = int(np.count_nonzero(inf_mask))
    if nan_count or inf_count:
      non_finite_mask = nan_mask | inf_mask
      flat_first = int(np.argmax(non_finite_mask))
      first_non_finite_index = tuple(
          int(x) for x in np.unravel_index(flat_first, act_np.shape)
      )

  has_nan_inf = bool(nan_count or inf_count)

  ulp_dist = compute_ulp_distance(act_np, exp_np, dtype_str=canonical)

  # Metrics over the finite subset so that a single NaN does not erase all
  # diagnostic signal from the rest of the tensor.
  if non_finite_mask is not None:
    finite_ulp = ulp_dist[~non_finite_mask]
  else:
    finite_ulp = ulp_dist

  if finite_ulp.size > 0:
    max_ulp = int(np.max(finite_ulp))
    mean_ulp = float(np.mean(finite_ulp))
    # A single sort serves both percentiles; two np.percentile calls sort the
    # array twice.
    p50_ulp, p99_9_ulp = (
        float(v) for v in np.percentile(finite_ulp, [50.0, 99.9])
    )
  else:
    max_ulp, mean_ulp, p50_ulp, p99_9_ulp = 0, 0.0, 0.0, 0.0

  bitwise_cmp = compare_bitwise(act_np, exp_np)
  bit_identical = bitwise_cmp.equal
  total = int(ulp_dist.size)
  le_1 = int(np.count_nonzero(ulp_dist <= 1))
  le_2 = int(np.count_nonzero(ulp_dist <= 2))
  hist = {
      "<=1_ulp": le_1,
      "<=2_ulp": le_2,
      ">2_ulp": total - le_2,
  }

  effective_p99_9 = (
      0.0
      if is_discrete and p99_9_allowed_ulp == 1.0
      else max(float(p99_9_allowed_ulp), float(limit_ulp))
  )
  ulp_passed = bool(
      not has_nan_inf
      and max_ulp <= limit_ulp
      and p99_9_ulp <= effective_p99_9
  )

  if is_discrete:
    allclose_passed = bool(np.array_equal(act_np, exp_np))
  else:
    dtype_eps, atol_val = _DTYPE_TOLERANCES.get(canonical, (1e-3, 1e-5))
    rtol_val = max(1, limit_ulp) * dtype_eps
    allclose_passed = bool(
        np.allclose(
            _as_compare_float(act_np),
            _as_compare_float(exp_np),
            rtol=rtol_val,
            atol=atol_val,
            equal_nan=False,
        )
    )

  worst_offender = None
  if act_np.size > 0:
    flat_max_idx = int(np.argmax(ulp_dist))
    max_idx = tuple(
        int(x) for x in np.unravel_index(flat_max_idx, ulp_dist.shape)
    )
    # Index first, convert second. Converting the whole tensor to float64 to
    # read a single element allocates 8 bytes per element for each operand.
    ref_v = float(exp_np[max_idx])
    cand_v = float(act_np[max_idx])
    abs_d = abs(cand_v - ref_v)
    rel_d = _relative_diff(abs_d, ref_v, canonical)
    mismatch_cnt = int(np.count_nonzero(ulp_dist > limit_ulp))
    worst_offender = WorstOffender(
        max_ulp_index=max_idx,
        ref_value=ref_v,
        cand_value=cand_v,
        abs_diff=abs_d,
        rel_diff=rel_d,
        mismatch_count=mismatch_cnt,
        mismatch_ratio=float(mismatch_cnt) / float(act_np.size),
    )

  passed = bool(ulp_passed and allclose_passed)
  if contract == CONTRACT_BITWISE:
    passed = bitwise_cmp.equal
  note = None
  if has_nan_inf:
    note = (
        f"Non-finite values present: {nan_count} NaN, {inf_count} Inf."
        " Reported ULP statistics cover the finite subset only."
    )
  elif ulp_passed and not allclose_passed:
    note = "Failed allclose dual gate check at rtol=k*eps."

  context_obj = UlpContext(
      bit_identical=bit_identical,
      p50=p50_ulp,
      p99_9=p99_9_ulp,
      max_ulp=max_ulp,
      reliable=not has_nan_inf,
      note=note,
  )

  return BatchValidationResult(
      batch_name="direct_array_comparison",
      regime="direct",
      max_ulp_distance=max_ulp,
      p99_9_ulp_distance=p99_9_ulp,
      mean_ulp_distance=mean_ulp,
      ulp_histogram=hist,
      has_nan_or_inf=has_nan_inf,
      passed=passed,
      ulp_context=context_obj,
      allclose_passed=allclose_passed,
      nan_count=nan_count,
      inf_count=inf_count,
      first_non_finite_index=first_non_finite_index,
      finite_max_ulp=max_ulp,
      worst_offender=worst_offender,
      bitwise=bitwise_cmp,
  )


def probe_reference_precision(
    kernel_ref: collections.abc.Callable[..., Any],
    shapes: Any,
    dtype_str: str = "float32",
    seed: int = 42,
    kernel_oracle: Any = ORACLE_AUTO,
    regimes: list[str] | None = None,
) -> OracleAudit:
  """Audits a reference callable against a Float64 oracle to detect downcasting."""
  report = validate_kernels(
      kernel_ref,
      kernel_ref,
      shapes=shapes,
      dtype_str=dtype_str,
      tier="fast_agent",
      seed=seed,
      kernel_oracle=kernel_oracle,
      regimes=regimes if regimes is not None else ["normal"],
  )
  if report.oracle_audit is not None:
    return report.oracle_audit
  return OracleAudit(
      oracle_executed_in_float64=True,
      oracle_output_dtype=dtype_str,
      reference_max_ulp_from_oracle=0,
      reference_p99_9_ulp_from_oracle=0.0,
      candidate_max_ulp_from_oracle=0,
      candidate_p99_9_ulp_from_oracle=0.0,
      reference_is_lossy=False,
      oracle_precision_verified=True,
  )
