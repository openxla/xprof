"""Tool to verify numerical parity between reference and candidate kernels."""

# pylint: disable=g-import-not-at-top

import ast
import builtins
import collections.abc
import dataclasses
import importlib
import json
from typing import Any

_Callable = collections.abc.Callable
_Sequence = collections.abc.Sequence


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
          f"(e.g. 'pip install {root_pkg}'). Note: {root_pkg} is not bundled "
          "in xprof to prevent cyclic dependencies (jax <--> xprof)."
      ) from e

  raise ImportError(f"Could not import module from '{target_str}'")


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
      changes the verdict. Without it this tool validates *agreement*, not
      correctness -- two kernels wrong in the same way agree perfectly.
    device_kind: Device/backend identifier (e.g. "tpu", "gpu", "cpu").
      Auto-detected when omitted.

  Returns:
    A JSON string containing the validation report:
      - is_numerically_equivalent: Boolean verdict.
      - correctness_basis: Basis of claim ("AGREEMENT_ONLY" or
        "AGREEMENT_AND_ORACLE").
      - run_config: Execution parameters (tier, seed, dtype_str, device_kind,
        backend, total_batches_count).
      - overall_max_ulp: Peak ULP distance across all batches.
      - failed_batches_count: Number of batches failing ULP criteria.
      - total_batches_count: Total test batches executed.
      - summary_message: High-level diagnostics.
      - tolerance_audit: Contract tolerance elevation diagnostic details.
      - oracle_audit: Reference- and candidate-vs-float64 distances.
      - ulp_context: Statistical and reliability assessment of ULP.
      - batch_results: Detailed per-batch breakdown.
  """
  ref_fn = _resolve_callable(kernel_ref)
  candidate_fn = _resolve_callable(kernel_candidate)

  if kernel_oracle is None or kernel_oracle == "auto":
    oracle_fn = kernel_oracle
  else:
    oracle_fn = _resolve_callable(kernel_oracle)

  parsed_shapes: Any = shapes
  if isinstance(shapes, str):
    parsed_shapes = ast.literal_eval(shapes)

  parsed_regimes = regimes
  if isinstance(regimes, str) and regimes != "all":
    if "," in regimes:
      parsed_regimes = [r.strip() for r in regimes.split(",") if r.strip()]
    elif regimes.startswith("[") or regimes.startswith("("):
      parsed_regimes = ast.literal_eval(regimes)

  try:
    from xprof.cli.internal import numerical_validator
  except ModuleNotFoundError as e:
    raise ImportError(
        "Required numerical dependencies are not installed in current"
        f" environment: {e}. Please install numpy and ml_dtypes (e.g. 'pip"
        " install numpy ml_dtypes') to use verify_numerical_parity."
    ) from e

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
      "batch_results": [dataclasses.asdict(b) for b in report.batch_results],
  }
  return json.dumps(results_dict, indent=2)
