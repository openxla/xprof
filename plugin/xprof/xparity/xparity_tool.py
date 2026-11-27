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
from xprof.xparity import rel_l2
from xprof.xparity import sqnr

_Callable = collections.abc.Callable
_Mapping = collections.abc.Mapping
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


def _sqnr_config(
    min_sqnr_db: float | str | None,
    alpha: float,
    preset: str | None,
    strict: bool,
) -> sqnr.SqnrConfig:
  """Builds the SQNR policy from flat CLI values.

  Args:
    min_sqnr_db: A number, a numeric string, "auto", or None. None uses the
      default floor of the preset: "auto" for "derived" and 45 dB for
      "six_gate_v1". An explicit "auto" derives the floor from the dtypes under
      every preset. Command-line parsing can turn "45" into an int, so numbers
      are converted to float here.
    alpha: False-alarm rate of the derived checks.
    preset: None, "derived" or "six_gate_v1".
    strict: Whether to run the anti-cheat checks.

  Returns:
    The config.

  Raises:
    ValueError: For a value outside its domain, such as a non-positive floor.
  """
  if isinstance(min_sqnr_db, bool):
    raise ValueError(
        f"min_sqnr_db must be a number or 'auto', got {min_sqnr_db}."
    )
  if isinstance(min_sqnr_db, (int, float)):
    min_sqnr_db = float(min_sqnr_db)
  if not isinstance(strict, bool):
    raise ValueError(f"sqnr_strict must be True or False, got {strict!r}.")
  return sqnr.SqnrConfig.from_options(
      min_sqnr_db=min_sqnr_db,
      alpha=float(alpha),
      strict=strict,
      preset=preset,
  )


def _uses_sqnr_contract(
    contract: str | None, contract_by_leaf: _Mapping[str, str] | None
) -> bool:
  """True if the run or any output leaf uses the "sqnr" contract."""
  contracts = [contract, *(contract_by_leaf or {}).values()]
  return numerical_validator.CONTRACT_SQNR in contracts


def _cli_sqnr_config(
    min_sqnr_db: float | str | None,
    alpha: float | None,
    preset: str | None,
    strict: bool | None,
    *,
    gated: bool,
) -> sqnr.SqnrConfig | None:
  """Builds the SQNR policy of a run from CLI values, or None for defaults.

  None lets the validator pick the contract default: the strict checks under
  "sqnr" only, and the global and oracle metrics only elsewhere.

  Args:
    min_sqnr_db: See `_sqnr_config`. None uses the default floor of the preset.
    alpha: See `_sqnr_config`. None means the default.
    preset: See `_sqnr_config`.
    strict: See `_sqnr_config`. None means True when `gated` and else False.
    gated: True if any output uses the "sqnr" contract.

  Returns:
    The policy, or None when every value is None.
  """
  if all(value is None for value in (min_sqnr_db, alpha, preset, strict)):
    return None
  return _sqnr_config(
      min_sqnr_db,
      sqnr.DEFAULT_ALPHA if alpha is None else alpha,
      preset,
      gated if strict is None else strict,
  )


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
    contract: str = numerical_validator.CONTRACT_ULP,
    contract_by_leaf: _Mapping[str, str] | str | None = None,
    dump_failures_to: str | None = None,
    atol: float | None = None,
    rtol: float | None = None,
    verdict_only: bool = False,
    rel_l2_tol: float | None = None,
    row_factor: float = rel_l2.ROW_FACTOR,
    row_norm_floor: float = rel_l2.ROW_NORM_FLOOR,
    min_sqnr_db: float | str | None = None,
    sqnr_alpha: float | None = None,
    sqnr_preset: str | None = None,
    sqnr_strict: bool | None = None,
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
      comma-separated string. By default "presubmit" and "deep_fuzzing" run
      every regime, and "fast_agent" runs 'normal' first with the other regimes
      as triage after a failure. `coverage` in the report lists the regimes that
      ran.
    kernel_oracle: Optional high-precision reference used to report how far
      `kernel_ref` itself sits from an exact result. Pass a callable (or
      "module.fn" path) that computes in float64 on the host, or the literal
      string "auto" to re-run `kernel_ref` with its floating-point arguments
      promoted to float64. It populates `oracle_audit` for finite outputs of a
      float run. It never changes the verdict of the "ulp", "bitwise" and
      "row_rel_l2" contracts. Under "sqnr" it gates the verdict: the candidate
      is measured against the oracle, so a candidate far from the oracle, a
      reference below the floor against the oracle, or a NaN or Inf that the
      oracle does not have fails the output. Then `correctness_basis` is
      "AGREEMENT_AND_ORACLE". An oracle callable also gives it for an integer
      run with float outputs or for NaN or Inf outputs, where `oracle_audit`
      stays null. "auto" does not give it in those cases.
    device_kind: Device/backend identifier (e.g. "tpu", "gpu", "cpu").
      Auto-detected when omitted.
    strict_shape_error: If True, raises ValueError on output shape or structure
      mismatch instead of returning a structured failure JSON report.
    contract: "ulp" (default) gates on ULP distance; "bitwise" requires every
      output element to match the reference bit for bit; "row_rel_l2" requires
      whole-output relative L2 error <= `rel_l2_tol` and every row (last
      dimension) within `row_factor * rel_l2_tol`, for intentionally lossy
      kernels such as FP8/FP4 dequantization; "sqnr" requires the
      signal-to-quantization-noise ratio of every floating-point output to meet
      `min_sqnr_db` and, with `sqnr_strict`, the anti-cheat checks.
    contract_by_leaf: Per-leaf contract overrides for pytree outputs, keyed by
      JAX key path, as a mapping or a dict literal string (e.g. "{'[1]':
      'ulp'}").
    dump_failures_to: Optional local directory for the inputs of failing
      batches, one `.npz` per batch. Paths are listed in `failure_dumps`; run
      `replay` on one to re-check a fix on the exact failing input.
    atol: Optional absolute tolerance of an existing allclose test. Report
      only: `tolerance_headroom` lists the worst error as a multiple of atol +
        rtol * |reference| per regime.
    rtol: Optional relative tolerance; see `atol`.
    verdict_only: If True, returns a compact single-line JSON verdict instead of
      the full report: the verdict, correctness basis, contract, overall max
      ULP, batch counts, up to 5 failing batches (name, regime, output leaf, max
      ULP), regimes not run, regimes over `atol`/`rtol`, failure dump paths and
      the summary verdict line. Under "row_rel_l2" each failure also carries its
      relative L2 figures and worst row, and `rel_l2` holds the worst figures of
      the run.
    rel_l2_tol: Whole-output relative L2 tolerance, required by "row_rel_l2".
    row_factor: Per-row bound as a multiple of `rel_l2_tol` (default 5).
    row_norm_floor: Floor on each row's norm as a fraction of the mean row norm
      (default 0.1), so that near-zero rows do not inflate the ratio.
    min_sqnr_db: SQNR floor in dB (a positive number) or "auto", which derives
      the floor from the output dtype. When omitted: "auto" under "derived" and
      45 dB under "six_gate_v1". Every contract reports SQNR figures per batch
      (`sqnr`) and for the run (`sqnr_summary`); only "sqnr" gates on them.
    sqnr_alpha: False-alarm rate per output of the derived SQNR checks.
    sqnr_preset: "derived" (default) or "six_gate_v1" for the fixed-constant
      checks with a 45 dB default floor.
    sqnr_strict: If True, also runs the SQNR anti-cheat checks. The default is
      True when any output uses "sqnr" and False otherwise, so that the
      report-only SQNR of the other contracts stays cheap.

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

  parsed_contract_by_leaf: _Mapping[str, str] | None
  if isinstance(contract_by_leaf, str):
    literal = ast.literal_eval(contract_by_leaf)
    if not isinstance(literal, dict):
      raise ValueError(
          "contract_by_leaf must be a dict literal mapping leaf paths to"
          f" contracts, got {contract_by_leaf!r}."
      )
    parsed_contract_by_leaf = literal
  else:
    parsed_contract_by_leaf = contract_by_leaf

  gated = _uses_sqnr_contract(contract, parsed_contract_by_leaf)
  sqnr_config = _cli_sqnr_config(
      min_sqnr_db, sqnr_alpha, sqnr_preset, sqnr_strict, gated=gated
  )

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
        contract=contract,
        contract_by_leaf=parsed_contract_by_leaf,
        dump_failures_to=dump_failures_to,
        atol=atol,
        rtol=rtol,
        rel_l2_tol=rel_l2_tol,
        row_factor=row_factor,
        row_norm_floor=row_norm_floor,
        sqnr_config=sqnr_config,
    )
  except ValueError as e:
    msg = str(e)
    if not strict_shape_error and (
        "Shape mismatch in batch" in msg
        or "Oracle shape mismatch in batch" in msg
        or "Output structure mismatch in batch" in msg
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
              "contract": contract,
              "sqnr_config": dataclasses.asdict(
                  sqnr_config or sqnr.SqnrConfig(strict=gated)
              ),
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
          "failure_dumps": [],
          "coverage": {},
          "tolerance_headroom": {},
          "rel_l2_summary": {},
          "sqnr_summary": {},
      }
      return _dump_json(mismatch_payload, verdict_only)
    raise

  return _report_to_json(report, verdict_only=verdict_only)


# Failing batches listed in the compact verdict; the rest are only counted.
_VERDICT_MAX_FAILURES = 5


def _summary_verdict_line(summary: str) -> str:
  """Returns the PASSED/FAILED line of a summary, or its first line."""
  lines = summary.splitlines()
  return next(
      (line for line in lines if line.startswith(("PASSED", "FAILED"))),
      lines[0] if lines else "",
  )


def _compact_verdict(payload: _Mapping[str, Any]) -> dict[str, Any]:
  """Reduces a full JSON report payload to the fields that decide the verdict."""
  run_config = payload.get("run_config") or {}
  failures = [
      _compact_failure(batch, run_config)
      for batch in payload.get("batch_results", [])
      if not batch.get("passed")
  ]
  coverage = payload.get("coverage") or {}
  headroom = payload.get("tolerance_headroom") or {}
  verdict = {
      "is_numerically_equivalent": payload["is_numerically_equivalent"],
      "correctness_basis": payload.get("correctness_basis"),
      "contract": run_config.get("contract"),
      "overall_max_ulp": payload.get("overall_max_ulp"),
      "failed_batches_count": payload.get("failed_batches_count"),
      "total_batches_count": payload.get("total_batches_count"),
      "failures": failures[:_VERDICT_MAX_FAILURES],
      "failures_omitted": max(0, len(failures) - _VERDICT_MAX_FAILURES),
      "regimes_not_run": coverage.get("regimes_not_run", []),
      "regimes_over_tolerance": headroom.get("regimes_over_tolerance", []),
      "failure_dumps": payload.get("failure_dumps", []),
      "verdict": _summary_verdict_line(payload.get("summary_message", "")),
  }
  rel_l2_summary = payload.get("rel_l2_summary")
  if rel_l2_summary:
    verdict["rel_l2"] = rel_l2_summary
  sqnr_summary = payload.get("sqnr_summary")
  if sqnr_summary and _uses_sqnr_contract(
      run_config.get("contract"), run_config.get("contract_by_leaf")
  ):
    verdict["sqnr"] = {
        key: sqnr_summary.get(key)
        for key in _VERDICT_SQNR_KEYS
        if key in sqnr_summary
    }
    verdict["sqnr"]["failed_reasons"] = (
        sqnr_summary.get("failed_reasons") or []
    )[:_VERDICT_MAX_FAILURES]
  return verdict


# SQNR summary fields copied into the compact verdict under "sqnr".
_VERDICT_SQNR_KEYS = (
    "sqnr_status",
    "passed",
    "sqnr_db",
    "oracle_sqnr_db",
    "threshold_db",
    "threshold_source",
    "gate_math",
    "failed_reports",
    "warnings",
)


def _compact_failure(
    batch: _Mapping[str, Any], run_config: _Mapping[str, Any]
) -> dict[str, Any]:
  """Returns the verdict-relevant fields of one failing batch result.

  The SQNR fields are included only when the batch's output uses the "sqnr"
  contract. The report-only SQNR of another contract does not decide the
  verdict, so it is not shown as a failure reason.

  Args:
    batch: One batch result of the full JSON report.
    run_config: The `run_config` of the full JSON report.
  """
  failure = {
      "batch_name": batch.get("batch_name"),
      "regime": batch.get("regime"),
      "leaf_path": batch.get("leaf_path"),
      "max_ulp": batch.get("max_ulp_distance"),
      "has_nan_or_inf": batch.get("has_nan_or_inf"),
  }
  row_l2 = batch.get("row_rel_l2")
  if row_l2:
    failure.update(
        rel_l2=row_l2.get("rel_l2"),
        worst_row_rel_l2=row_l2.get("worst_row_rel_l2"),
        worst_row_index=row_l2.get("worst_row_index"),
        rows_over_bound=row_l2.get("rows_over_bound"),
        reason=row_l2.get("reason"),
    )
  report = batch.get("sqnr")
  own_contract = numerical_validator.leaf_contract(
      run_config.get("contract"),
      run_config.get("contract_by_leaf"),
      batch.get("leaf_path"),
  )
  if (
      report
      and not report.get("passed")
      and own_contract == numerical_validator.CONTRACT_SQNR
  ):
    failure.update(
        sqnr_db=report.get("sqnr_db"),
        sqnr_status=report.get("sqnr_status"),
        sqnr_failed_reasons=report.get("failed_reasons"),
    )
  return failure


def _dump_json(payload: _Mapping[str, Any], verdict_only: bool) -> str:
  """Serializes a full report payload, or its compact verdict, to JSON."""
  sanitized = _sanitize_for_json(payload)
  if verdict_only:
    return json.dumps(
        _compact_verdict(sanitized), separators=(",", ":"), allow_nan=False
    )
  return json.dumps(sanitized, indent=2, allow_nan=False)


def _report_to_json(
    report: numerical_validator.KernelValidationReport,
    verdict_only: bool = False,
) -> str:
  """Serializes a KernelValidationReport to RFC 8259 JSON."""
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
      "failure_dumps": report.failure_dumps,
      "coverage": report.coverage,
      "tolerance_headroom": report.tolerance_headroom,
      "rel_l2_summary": report.rel_l2_summary,
      "sqnr_summary": report.sqnr_summary,
  }
  return _dump_json(results_dict, verdict_only)


def replay(
    dump_path: str,
    kernel_ref: _Callable[..., Any] | str,
    kernel_candidate: _Callable[..., Any] | str,
    kernel_oracle: _Callable[..., Any] | str | None = None,
    contract: str | None = None,
    verdict_only: bool = False,
    rel_l2_tol: float | None = None,
    row_factor: float | None = None,
    row_norm_floor: float | None = None,
    min_sqnr_db: float | str | None = None,
    sqnr_alpha: float | None = None,
    sqnr_preset: str | None = None,
    sqnr_strict: bool | None = None,
) -> str:
  """Re-runs one saved failing batch and returns a JSON report.

  Args:
    dump_path: A `.npz` file from `failure_dumps` of an earlier run.
    kernel_ref: The reference implementation (callable or string path).
    kernel_candidate: The candidate implementation (callable or string path).
    kernel_oracle: Optional oracle (callable, string path, or "auto").
    contract: Overrides the contract stored in the dump ("ulp", "bitwise",
      "row_rel_l2" or "sqnr").
    verdict_only: If True, returns the compact verdict described in
      `verify_numerical_parity` instead of the full report.
    rel_l2_tol: Overrides the "row_rel_l2" tolerance stored in the dump.
    row_factor: Overrides the stored per-row factor.
    row_norm_floor: Overrides the stored row-norm floor.
    min_sqnr_db: Overrides the stored SQNR floor (a number or "auto"). When
      omitted, the stored floor stays. With `sqnr_preset`, a stored floor that
      equals the default of the stored preset ("auto" under "derived", 45 dB
      under "six_gate_v1") counts as not given, so it takes the default of the
      new preset. An explicit floor equal to that default (45 dB under
      "six_gate_v1", or "auto" under "derived") also counts as not given.
    sqnr_alpha: Overrides the stored SQNR false-alarm rate.
    sqnr_preset: Overrides the stored SQNR gate math ("derived" or
      "six_gate_v1").
    sqnr_strict: Overrides whether the SQNR anti-cheat checks run.

  Returns:
    A JSON string containing the validation report for the saved batch.
  """
  sqnr_config = None
  if any(
      value is not None
      for value in (min_sqnr_db, sqnr_alpha, sqnr_preset, sqnr_strict)
  ):
    info = numerical_generator.read_suite_metadata(dump_path)
    if info.get("sqnr_config"):
      base = numerical_validator.stored_sqnr_config(info["sqnr_config"])
    else:
      # The run used the contract default, so the overrides start from it.
      gated = _uses_sqnr_contract(
          contract or info.get("contract"), info.get("contract_by_leaf")
      )
      base = sqnr.SqnrConfig(strict=gated)
    # Only a preset that the caller passes switches the gate math. Without
    # one, the stored gate math and the stored floor stay, AUTO included.
    # With one, a stored floor counts as given unless it equals the default
    # floor of the stored preset. A floor that was not given takes the
    # default of the new preset, as in `verify_numerical_parity`. An explicit
    # floor equal to that default (45 dB under "six_gate_v1", or "auto" under
    # "derived") also counts as not given. An explicit "auto" from the caller
    # stays AUTO.
    floor = min_sqnr_db
    if floor is None:
      if base.gate_math == sqnr.GATE_MATH_SIX_GATE_V1:
        stored_default = sqnr.DEFAULT_MIN_SQNR_DB
      else:
        stored_default = sqnr.AUTO
      if sqnr_preset is None or base.min_sqnr_db != stored_default:
        floor = base.min_sqnr_db
    merged = _sqnr_config(
        floor,
        base.alpha if sqnr_alpha is None else sqnr_alpha,
        sqnr_preset,
        base.strict if sqnr_strict is None else sqnr_strict,
    )
    # Fields without a flag keep their stored values.
    sqnr_config = dataclasses.replace(
        base,
        min_sqnr_db=merged.min_sqnr_db,
        alpha=merged.alpha,
        strict=merged.strict,
        gate_math=base.gate_math if sqnr_preset is None else merged.gate_math,
    )
  if kernel_oracle is None or kernel_oracle == numerical_validator.ORACLE_AUTO:
    oracle_fn = kernel_oracle
  else:
    oracle_fn = _resolve_callable(kernel_oracle)
  report = numerical_validator.replay_failure_dump(
      dump_path,
      _resolve_callable(kernel_ref),
      _resolve_callable(kernel_candidate),
      kernel_oracle=oracle_fn,
      contract=contract,
      rel_l2_tol=rel_l2_tol,
      row_factor=row_factor,
      row_norm_floor=row_norm_floor,
      sqnr_config=sqnr_config,
  )
  return _report_to_json(report, verdict_only=verdict_only)


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
