"""Aggregation and human-readable reporting of XLoop results.

This module owns the bridge from a worker's `results.json` to the proto:
`build_execution_result` turns the flat dictionary the harness emits into an
`ExecutionResult`, and `aggregate_report` reduces per-config results to the
`ExecutionReport` an agent or a dashboard reads.
"""

from __future__ import annotations

import collections.abc
import math
import statistics
from typing import Any

from xprof.protobuf import xloop_item_pb2
from xprof.xloop import input_utils

_Sequence = collections.abc.Sequence

# A measured speedup within epsilon of 1.0 is reported as "no change" rather
# than a win or a regression. The floor absorbs run-to-run clock drift; the
# spread term widens it when the samples themselves are noisy.
MIN_SPEEDUP_EPSILON = 0.02
DEFAULT_SPEEDUP_EPSILON = 0.03
MAX_SPEEDUP_EPSILON = 0.10

_FASTER = "faster"
_SLOWER = "slower"
_NO_CHANGE = "no change"


def _relative_half_iqr(values: _Sequence[float]) -> float:
  """Returns half the relative interquartile range: (Q3 - Q1) / (2 * Q2)."""
  if len(values) < 2:
    return 0.0
  q1, q2, q3 = statistics.quantiles(values, n=4)
  return (q3 - q1) / (2.0 * q2) if q2 > 0 else 0.0


def compute_speedup_epsilon(
    ref_times: _Sequence[float],
    code_times: _Sequence[float],
) -> float:
  """Computes a bounded relative epsilon from a floor plus the relative IQR.

  Widens with the relative interquartile range of both arms rather than with
  the standard error, deliberately: dividing by sqrt(N) would let a large
  sample count shrink the threshold below real run-to-run drift, and would let
  a single outlier dominate.

  Args:
    ref_times: Per-sample reference execution times.
    code_times: Per-sample candidate execution times.

  Returns:
    Epsilon in `[MIN_SPEEDUP_EPSILON, MAX_SPEEDUP_EPSILON]`.
  """
  valid_ref = [r for r in ref_times if r > 0]
  valid_code = [c for c in code_times if c > 0]
  if len(valid_ref) < 2 or len(valid_code) < 2:
    return DEFAULT_SPEEDUP_EPSILON
  rel_spread = _relative_half_iqr(valid_ref) + _relative_half_iqr(valid_code)
  return min(max(MIN_SPEEDUP_EPSILON, rel_spread), MAX_SPEEDUP_EPSILON)


def _geomean_ratio(
    numerators: _Sequence[float], denominators: _Sequence[float]
) -> float | None:
  """Geometric mean of paired ratios, ignoring non-positive pairs.

  The geometric mean rather than the arithmetic one because these are ratios:
  a 2x speedup on one sample and a 0.5x on another should average to 1.0.

  Args:
    numerators: Numerator of each pair.
    denominators: Denominator of each pair.

  Returns:
    The geometric mean, or None if no pair was usable.
  """
  logs = [
      math.log(n / d)
      for n, d in zip(numerators, denominators)
      if n > 0 and d > 0 and math.isfinite(n) and math.isfinite(d)
  ]
  if not logs:
    return None
  return math.exp(sum(logs) / len(logs))


def sample_times(
    result: xloop_item_pb2.ExecutionResult, role: str
) -> list[float]:
  """Per-sample execution times for one role."""
  field = f"{role}_execution_time_sec"
  return [getattr(s, field) for s in result.sample_execution_results]


def _timing_source(
    results: dict[str, Any], role: str
) -> tuple[str, list[float]]:
  """Chooses the per-sample times to report for one role.

  Device-derived times are preferred whenever the trace yielded a usable set.
  Wall-clock times around a jitted call include host dispatch, which on a v6e
  is ~160us against ~2.5us of compute for a 1024-cube matmul, so they cannot
  resolve a microbenchmark. They remain the fallback because a run whose
  outputs are valid should still report something.

  Args:
    results: The worker's results payload.
    role: Role to read.

  Returns:
    A `(source, times)` pair, where source is "trace" or "wall_clock". Times
    may be empty, in which case the caller spreads the aggregate.
  """
  trace = results.get(f"{role}_trace_matched_execution_times_sec")
  if isinstance(trace, list) and trace:
    return "trace", [float(v) for v in trace]
  wall = results.get(f"{role}_per_sample_sec")
  if isinstance(wall, list) and wall:
    return "wall_clock", [float(v) for v in wall]
  return "wall_clock", []


def build_execution_result(
    results: dict[str, Any], roles: _Sequence[str] | None = None
) -> xloop_item_pb2.ExecutionResult:
  """Converts a worker's `results.json` payload into an `ExecutionResult`.

  Args:
    results: The dictionary written by `eval_jax_lib.serialize_results`,
      optionally augmented with correctness verdicts by the worker.
    roles: Roles to read. Defaults to all conventional roles; absent ones are
      skipped.

  Returns:
    The populated result. Per-sample times are device durations read back from
    the emitted trace where available, and wall-clock measurements otherwise;
    `timing_source` records which, per role.
  """
  roles = list(roles) if roles is not None else list(input_utils.ROLES)
  present = [r for r in roles if f"{r}_active_sec" in results]

  out = xloop_item_pb2.ExecutionResult()
  out.execution_status = xloop_item_pb2.CODE_STATUS_EXECUTE_SUCCESS

  sources = {role: _timing_source(results, role) for role in present}
  for role, (source, _) in sources.items():
    out.timing_source[role] = source

  num_samples = int(results.get("num_executions", 0) or 0)
  for index in range(num_samples):
    sample = out.sample_execution_results.add()
    for role in present:
      _, per_sample = sources[role]
      if index < len(per_sample):
        value = per_sample[index]
      else:
        # Neither source retained per-sample detail; spread the aggregate.
        active = float(results.get(f"{role}_active_sec", 0.0) or 0.0)
        value = active / num_samples if num_samples else 0.0
      setattr(sample, f"{role}_execution_time_sec", value)

    for role in present:
      if role == input_utils.REFERENCE_ROLE:
        continue
      verdicts = results.get(f"{role}_per_sample_match")
      field = f"{role}_outputs_match_reference"
      if not hasattr(sample, field):
        continue
      if isinstance(verdicts, list) and index < len(verdicts):
        setattr(sample, field, bool(verdicts[index]))
      else:
        setattr(
            sample,
            field,
            bool(results.get(f"{role}_outputs_match_reference", False)),
        )

  for role in present:
    metrics = results.get(f"{role}_utilization_metrics")
    if isinstance(metrics, dict):
      target = getattr(out, f"{role}_utilization_metrics")
      for key, value in metrics.items():
        target[key] = float(value)

  return out


def aggregate_report(
    item: xloop_item_pb2.XLoopItem,
) -> xloop_item_pb2.ExecutionReport:
  """Reduces an item's per-config results to a single report.

  Args:
    item: Item whose `execution_results` are populated.

  Returns:
    The aggregate report. Speedups are geometric means over every sample of
    every config, so one slow config cannot be hidden by a fast one.
  """
  report = xloop_item_pb2.ExecutionReport()
  reference: list[float] = []
  custom: list[float] = []
  optimized: list[float] = []
  custom_match = True
  optimized_match = True
  saw_custom = False
  saw_optimized = False
  status = xloop_item_pb2.CODE_STATUS_EXECUTE_SUCCESS

  for result in item.execution_results.values():
    if result.execution_status != xloop_item_pb2.CODE_STATUS_EXECUTE_SUCCESS:
      status = result.execution_status
    for sample in result.sample_execution_results:
      reference.append(sample.reference_execution_time_sec)
      custom.append(sample.custom_execution_time_sec)
      optimized.append(sample.optimized_execution_time_sec)
      if sample.custom_execution_time_sec > 0:
        saw_custom = True
        custom_match = custom_match and sample.custom_outputs_match_reference
      if sample.optimized_execution_time_sec > 0:
        saw_optimized = True
        optimized_match = (
            optimized_match and sample.optimized_outputs_match_reference
        )

  report.execution_status = status
  report.custom_outputs_match_reference = saw_custom and custom_match
  report.optimized_outputs_match_reference = saw_optimized and optimized_match

  # Speedup is reference/candidate: greater than one means the candidate is
  # faster.
  for value, field in (
      (_geomean_ratio(reference, custom), "gmean_speedup_over_reference"),
      (_geomean_ratio(optimized, custom), "gmean_speedup_over_optimized"),
      (
          _geomean_ratio(reference, optimized),
          "gmean_speedup_optimized_over_reference",
      ),
  ):
    if value is not None:
      setattr(report, field, value)

  return report


def format_utilization_value(val: float) -> str:
  """Formats a utilization metric, which may be a fraction or a percentage."""
  if not math.isfinite(val):
    return "n/a"
  if 0.0 <= val <= 1.0:
    return f"{val * 100:.1f}%"
  return f"{val:.2f}"


def _fmt_time(val: float | None) -> str:
  if val is None or not math.isfinite(val) or val <= 0:
    return "-"
  if val < 1e-3:
    return f"{val * 1e6:.1f}us"
  if val < 1.0:
    return f"{val * 1e3:.2f}ms"
  return f"{val:.3f}s"


def _fmt_speedup(val: float | None) -> str:
  if val is None or not math.isfinite(val) or val <= 0:
    return "-"
  return f"{val:.2f}x"


def classify_speedup(speedup: float, epsilon: float) -> str:
  """Labels a speedup as faster, slower, or within noise.

  Args:
    speedup: Reference time divided by candidate time.
    epsilon: Relative threshold from `compute_speedup_epsilon`.

  Returns:
    One of `faster`, `slower` or `no change`.
  """
  if not math.isfinite(speedup) or speedup <= 0:
    return _NO_CHANGE
  if speedup > 1.0 + epsilon:
    return _FASTER
  if speedup < 1.0 - epsilon:
    return _SLOWER
  return _NO_CHANGE


def _match_str(matched: bool, applicable: bool) -> str:
  if not applicable:
    return "-"
  return "pass" if matched else "FAIL"


def _row(cells: _Sequence[str], widths: _Sequence[int]) -> str:
  return "  ".join(c.ljust(w) for c, w in zip(cells, widths)).rstrip()


def format_summary(items: _Sequence[xloop_item_pb2.XLoopItem]) -> str:
  """Renders a one-line-per-item summary table.

  Args:
    items: Items whose `execution_report` is populated.

  Returns:
    A plain-text table. Empty string if there is nothing to report.
  """
  if not items:
    return ""

  header = (
      "item",
      "status",
      "correct",
      "custom vs ref",
      "verdict",
      "opt vs ref",
  )
  rows: list[tuple[str, ...]] = []
  for item in items:
    report = item.execution_report
    speedup = report.gmean_speedup_over_reference or None

    # Epsilon is derived from the actual sample spread of this item, so a
    # noisy kernel is held to a looser bar than a stable one.
    reference_times: list[float] = []
    custom_times: list[float] = []
    for result in item.execution_results.values():
      reference_times.extend(sample_times(result, "reference"))
      custom_times.extend(sample_times(result, "custom"))
    epsilon = compute_speedup_epsilon(reference_times, custom_times)

    rows.append((
        item.name or item.id or "(unnamed)",
        xloop_item_pb2.CodeStatus.Name(report.execution_status),
        _match_str(report.custom_outputs_match_reference, bool(custom_times)),
        _fmt_speedup(speedup),
        classify_speedup(speedup, epsilon) if speedup else "-",
        _fmt_speedup(report.gmean_speedup_optimized_over_reference or None),
    ))

  widths = [
      max(len(header[i]), max(len(r[i]) for r in rows))
      for i in range(len(header))
  ]
  lines = [_row(header, widths), _row(["-" * w for w in widths], widths)]
  lines.extend(_row(r, widths) for r in rows)
  return "\n".join(lines)


def format_detail(item: xloop_item_pb2.XLoopItem) -> str:
  """Renders per-config, per-sample detail for one item.

  Args:
    item: Item whose `execution_results` are populated.

  Returns:
    A plain-text report.
  """
  lines: list[str] = [f"item: {item.name or item.id or '(unnamed)'}"]

  for config_name in sorted(item.execution_results):
    result = item.execution_results[config_name]
    lines.append("")
    lines.append(f"  config: {config_name}")
    status = xloop_item_pb2.CodeStatus.Name(result.execution_status)
    lines.append(f"    status: {status}")
    if result.error_message:
      lines.append(f"    error: {result.error_message}")

    roles = [r for r in input_utils.ROLES if any(sample_times(result, r))]
    if not roles:
      continue

    header = ["sample"] + roles
    rows = []
    for index, sample in enumerate(result.sample_execution_results):
      rows.append(
          [str(index)]
          + [
              _fmt_time(getattr(sample, f"{r}_execution_time_sec"))
              for r in roles
          ]
      )
    for role in roles:
      times = [t for t in sample_times(result, role) if t > 0]
      if times:
        rows.append(
            ["median"]
            + [
                _fmt_time(statistics.median(times)) if r == role else ""
                for r in roles
            ]
        )
        break

    widths = [
        max(len(header[i]), max(len(r[i]) for r in rows))
        for i in range(len(header))
    ]
    lines.append("    " + _row(header, widths))
    for row in rows:
      lines.append("    " + _row(row, widths))

    for role in roles:
      metrics = getattr(result, f"{role}_utilization_metrics")
      if metrics:
        rendered = ", ".join(
            f"{k}={format_utilization_value(v)}"
            for k, v in sorted(metrics.items())
        )
        lines.append(f"    {role} utilization: {rendered}")

  return "\n".join(lines)
