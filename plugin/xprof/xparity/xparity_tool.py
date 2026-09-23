"""Tool and CLI functions for Xparity numerical accuracy and parity verification."""

import ast
import builtins
import collections.abc
import dataclasses
import importlib
import json
import math
from typing import Any

from xprof.xparity import numerical_generator
from xprof.xparity import numerical_validator

_Callable = collections.abc.Callable
_Sequence = collections.abc.Sequence


def _sanitize_for_json(obj: Any) -> Any:
  """Recursively converts non-finite floats to None for RFC 8259 compliance."""
  if isinstance(obj, float):
    return obj if math.isfinite(obj) else None
  if hasattr(obj, "dtype") and getattr(obj.dtype, "kind", None) in ("f", "V"):
    try:
      val = float(obj)
      return val if math.isfinite(val) else None
    except (TypeError, ValueError):
      pass
  if isinstance(obj, dict):
    return {k: _sanitize_for_json(v) for k, v in obj.items()}
  if isinstance(obj, (list, tuple)):
    return [_sanitize_for_json(v) for v in obj]
  return obj


def _resolve_callable(target: _Callable[..., Any] | str) -> _Callable[..., Any]:
  """Resolves a callable object or a module-qualified string path to a callable.

  Supports:
    - Direct callable objects (functions, lambdas, classes).
    - Colon format: 'package.module:function_name'
    - Dotted path: 'package.module.function_name' or 'math.sin'

  Args:
    target: Callable object or string import path.

  Returns:
    Resolved callable object.

  Raises:
    TypeError: If target is not callable or cannot be resolved to a callable.
    ImportError: If the specified module cannot be imported.
    AttributeError: If the specified attribute is missing from the module.
  """
  if callable(target):
    return target

  if not isinstance(target, str):
    raise TypeError(
        f"Expected callable or string import path, got: {type(target).__name__}"
    )

  target_str = target.strip()
  if not target_str:
    raise ValueError("Empty callable string path provided.")

  if ":" in target_str:
    mod_name, attr_name = target_str.split(":", 1)
    try:
      mod = importlib.import_module(mod_name)
    except ImportError as e:
      raise ImportError(
          f"Could not import module '{mod_name}' for target '{target_str}': {e}"
      ) from e
    current: Any = mod
    for attr in attr_name.split("."):
      if not hasattr(current, attr):
        raise AttributeError(
            f"Module '{mod_name}' has no attribute '{attr}' in '{target_str}'"
        )
      current = getattr(current, attr)
    if not callable(current):
      raise TypeError(
          f"Target '{target_str}' resolved to non-callable:"
          f" {type(current).__name__}"
      )
    return current

  # Dotted path resolution: e.g. "package.subpackage.module.function"
  parts = target_str.split(".")
  if len(parts) == 1:
    if hasattr(builtins, target_str):
      fn = getattr(builtins, target_str)
      if callable(fn):
        return fn
    raise ValueError(
        f"Invalid callable string: '{target_str}'. Expected module.attribute"
        " or module:attribute format."
    )

  # Find the module split point from right to left
  for i in range(len(parts) - 1, 0, -1):
    mod_name = ".".join(parts[:i])
    attr_parts = parts[i:]
    try:
      mod = importlib.import_module(mod_name)
    except ImportError:
      continue

    current = mod
    for attr in attr_parts:
      if not hasattr(current, attr):
        raise AttributeError(
            f"Module '{mod_name}' has no attribute '{attr}' in '{target_str}'"
        )
      current = getattr(current, attr)
    if not callable(current):
      raise TypeError(
          f"Resolved target '{target_str}' is not callable:"
          f" {type(current).__name__}"
      )
    return current

  root_pkg = parts[0]
  if root_pkg in ("jax", "torch", "tensorflow", "flax", "equinox"):
    try:
      importlib.import_module(root_pkg)
    except ModuleNotFoundError as e:
      raise ImportError(
          f"Package '{root_pkg}' is not installed in current environment. "
          f"To verify '{target_str}', please install {root_pkg} separately "
          f"(e.g. 'pip install {root_pkg}')."
      ) from e

  raise ImportError(f"Could not import module from '{target_str}'")


def _parse_shapes(
    shapes: _Sequence[int] | _Sequence[_Sequence[int]] | str,
) -> Any:
  """Parses a shape tuple, list of shapes, or string literal."""
  if isinstance(shapes, str):
    return ast.literal_eval(shapes)
  return shapes


def verify_numerical_parity(
    kernel_ref: _Callable[..., Any] | str,
    kernel_candidate: _Callable[..., Any] | str,
    shapes: _Sequence[int] | _Sequence[_Sequence[int]] | str,
    dtype_str: str = "bfloat16",
    tier: str = "presubmit",
    max_allowed_ulp: int = 2,
    p99_9_allowed_ulp: int = 1,
    seed: int = 42,
    regimes: _Sequence[str] | str | None = None,
    kernel_oracle: _Callable[..., Any] | str | None = None,
    device_kind: str | None = None,
    strict_shape_error: bool = False,
) -> str:
  """Validates numerical parity between two kernels and returns a JSON report.

  Args:
    kernel_ref: The baseline/reference implementation (callable or string path).
    kernel_candidate: The candidate/optimized implementation (callable or string
      path).
    shapes: A shape tuple (e.g. (16, 1024)), list of shapes, or literal string.
    dtype_str: The target floating-point dtype (e.g. "float32", "bfloat16").
    tier: Operational testing tier ("fast_agent", "presubmit", "deep_fuzzing").
    max_allowed_ulp: Maximum acceptable bitwise ULP distance across all
      elements.
    p99_9_allowed_ulp: Maximum acceptable 99.9th percentile ULP distance.
    seed: PRNG seed for reproducibility.
    regimes: Optional sequence of regime names (e.g. ['normal']) or
      comma-separated string. Defaults to 'normal' with automated triage
      fallback on failure.
    kernel_oracle: Optional high-precision reference used to report how far
      `kernel_ref` itself sits from an exact result. Pass a callable (or
      "module.fn" path) that computes in float64 on the host, or the literal
      string "auto" to re-run `kernel_ref` with its floating-point arguments
      promoted to float64. Report-only: it populates `oracle_audit` and never
      changes the verdict.
    device_kind: Device/backend identifier (e.g. "tpu", "gpu", "cpu").
      Auto-detected when omitted.
    strict_shape_error: If True, raises ValueError on output shape mismatch
      instead of returning a structured failure JSON report.

  Returns:
    A JSON string containing the validation report.
  """
  ref_fn = _resolve_callable(kernel_ref)
  candidate_fn = _resolve_callable(kernel_candidate)

  if kernel_oracle is None or kernel_oracle == numerical_validator.ORACLE_AUTO:
    oracle_fn = kernel_oracle
  else:
    oracle_fn = _resolve_callable(kernel_oracle)

  parsed_shapes = _parse_shapes(shapes)

  parsed_regimes = regimes
  if isinstance(regimes, str) and regimes != "all":
    if "," in regimes:
      parsed_regimes = [r.strip() for r in regimes.split(",") if r.strip()]
    elif regimes.startswith("[") or regimes.startswith("("):
      parsed_regimes = ast.literal_eval(regimes)

  try:
    report = numerical_validator.validate_kernels(
        kernel_ref=ref_fn,
        kernel_candidate=candidate_fn,
        shapes=parsed_shapes,
        dtype_str=dtype_str,
        tier=tier,
        max_allowed_ulp=max_allowed_ulp,
        p99_9_allowed_ulp=p99_9_allowed_ulp,
        seed=seed,
        regimes=parsed_regimes,
        kernel_oracle=oracle_fn,
        device_kind=device_kind,
    )
  except ValueError as e:
    msg = str(e)
    if not strict_shape_error and (
        "Shape mismatch in batch" in msg
        or "Oracle shape mismatch in batch" in msg
    ):
      mismatch_payload = {
          "is_numerically_equivalent": False,
          "correctness_basis": "SHAPE_MISMATCH",
          "run_config": {
              "tier": tier,
              "seed": seed,
              "dtype_str": dtype_str,
              "device_kind": device_kind or "auto",
              "total_batches_count": 0,
          },
          "overall_max_ulp": 999999,
          "failed_batches_count": 1,
          "total_batches_count": 1,
          "summary_message": f"FAILED: {msg}",
          "tolerance_audit": None,
          "oracle_audit": None,
          "ulp_context": None,
          "narrow_output_dtype_warning": None,
          "shape_mismatch": {"error": msg},
          "batch_results": [],
      }
      return json.dumps(mismatch_payload, indent=2, allow_nan=False)
    raise

  results_dict = {
      "is_numerically_equivalent": report.is_numerically_equivalent,
      "correctness_basis": report.correctness_basis,
      "run_config": report.run_config,
      "overall_max_ulp": report.overall_max_ulp,
      "failed_batches_count": report.failed_batches_count,
      "total_batches_count": report.total_batches_count,
      "summary_message": report.summary_message,
      "tolerance_audit": (
          dataclasses.asdict(report.tolerance_audit)
          if report.tolerance_audit is not None
          else None
      ),
      "oracle_audit": (
          dataclasses.asdict(report.oracle_audit)
          if report.oracle_audit is not None
          else None
      ),
      "ulp_context": (
          dataclasses.asdict(report.ulp_context)
          if report.ulp_context is not None
          else None
      ),
      "narrow_output_dtype_warning": report.narrow_output_dtype_warning,
      "shape_mismatch": report.shape_mismatch,
      "batch_results": [dataclasses.asdict(b) for b in report.batch_results],
  }
  sanitized_results = _sanitize_for_json(results_dict)
  return json.dumps(sanitized_results, indent=2, allow_nan=False)


def generate_suite(
    shapes: _Sequence[int] | _Sequence[_Sequence[int]] | str,
    output_path: str,
    dtype_str: str = "bfloat16",
    tier: str = "presubmit",
    seed: int = 42,
) -> str:
  """Generates a multi-regime test suite and saves it to a .npz file.

  Args:
    shapes: Single shape tuple, sequence of shapes, or string literal.
    output_path: Destination .npz path to write the generated suite.
    dtype_str: Target data type string.
    tier: Operational testing tier ("fast_agent", "presubmit", "deep_fuzzing").
    seed: PRNG seed.

  Returns:
    A JSON summary of the generated test suite.
  """
  parsed_shapes = _parse_shapes(shapes)
  suite = numerical_generator.generate_test_suite(
      parsed_shapes,
      dtype_str=dtype_str,
      tier=tier,
      seed=seed,
      persisted_path=output_path,
      mode="record",
  )
  summary = {
      "output_path": output_path,
      "dtype_str": dtype_str,
      "tier": tier,
      "seed": seed,
      "num_batches": len(suite),
      "batches": [
          {
              "name": b["name"],
              "regime": b["regime"],
              "arg_shapes": [list(a.shape) for a in b.get("args", ())],
          }
          for b in suite
      ],
  }
  return json.dumps(summary, indent=2)


def inspect_suite(source_path: str) -> str:
  """Inspects a persisted .npz test suite and returns its batch metadata.

  Args:
    source_path: Path to the .npz suite file.

  Returns:
    JSON summary of the batches, shapes, dtypes, and regimes in the suite.
  """
  suite = numerical_generator.load_test_suite(source_path)
  summary = {
      "source_path": source_path,
      "num_batches": len(suite),
      "batches": [
          {
              "name": b["name"],
              "regime": b["regime"],
              "args": [
                  {"shape": list(a.shape), "dtype": str(a.dtype)}
                  for a in b.get("args", ())
              ],
          }
          for b in suite
      ],
  }
  return json.dumps(summary, indent=2)


def probe_precision(
    kernel_fn: _Callable[..., Any] | str,
    shapes: _Sequence[int] | _Sequence[_Sequence[int]] | str,
    dtype_str: str = "float32",
    device_kind: str = "cpu",
    seed: int = 42,
) -> str:
  """Probes a callable for accelerator matmul precision pinning and inertness.

  Args:
    kernel_fn: Callable or module-qualified string path.
    shapes: Shape tuple or list of shapes.
    dtype_str: Input tensor dtype string.
    device_kind: Target device ('cpu', 'tpu', 'gpu').
    seed: PRNG seed for probe inputs.

  Returns:
    JSON report with precision pinning and pin-inertness probe results.
  """
  fn = _resolve_callable(kernel_fn)
  parsed_shapes = _parse_shapes(shapes)
  suite = numerical_generator.generate_test_suite(
      parsed_shapes, dtype_str=dtype_str, tier="fast_agent", seed=seed
  )
  first_b = suite[0]
  args = first_b.get("args", ())
  kwargs = first_b.get("kwargs", {})

  given_probe = numerical_validator._probe_precision(  # pylint: disable=protected-access
      fn, args, kwargs, baseline="given", device_kind=device_kind
  )
  inert_probe = numerical_validator._probe_precision(  # pylint: disable=protected-access
      fn, args, kwargs, baseline="default", device_kind=device_kind
  )
  return json.dumps(
      {
          "is_pinned_at_highest": given_probe.is_pinned,
          "pinned_diagnostic": given_probe.diagnostic,
          "reference_pin_inert": bool(inert_probe.is_pinned),
          "inert_diagnostic": inert_probe.diagnostic,
          "device_kind": device_kind,
      },
      indent=2,
  )
