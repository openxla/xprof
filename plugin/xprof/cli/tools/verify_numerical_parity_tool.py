"""Tool to verify numerical parity between reference and candidate kernels."""

import ast
import builtins
import collections.abc
import dataclasses
import importlib
import json
from typing import Any
from xprof.cli.internal import numerical_validator

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

  Returns:
    A JSON string containing the validation report:
      - is_numerically_equivalent: Boolean verdict.
      - overall_max_ulp: Peak ULP distance across all batches.
      - failed_batches_count: Number of batches failing ULP criteria.
      - total_batches_count: Total test batches executed.
      - summary_message: High-level diagnostics.
      - batch_results: Detailed per-batch breakdown.
  """
  ref_fn = _resolve_callable(kernel_ref)
  candidate_fn = _resolve_callable(kernel_candidate)

  parsed_shapes: Any = shapes
  if isinstance(shapes, str):
    parsed_shapes = ast.literal_eval(shapes)

  report = numerical_validator.validate_kernels(
      kernel_ref=ref_fn,
      kernel_candidate=candidate_fn,
      shapes=parsed_shapes,
      dtype_str=dtype_str,
      tier=tier,
      max_allowed_ulp=max_allowed_ulp,
      p99_9_allowed_ulp=p99_9_allowed_ulp,
      seed=seed,
  )

  results_dict = {
      "is_numerically_equivalent": report.is_numerically_equivalent,
      "overall_max_ulp": report.overall_max_ulp,
      "failed_batches_count": report.failed_batches_count,
      "total_batches_count": report.total_batches_count,
      "summary_message": report.summary_message,
      "batch_results": [dataclasses.asdict(b) for b in report.batch_results],
  }
  return json.dumps(results_dict, indent=2)
