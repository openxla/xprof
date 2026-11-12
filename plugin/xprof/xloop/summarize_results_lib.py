# Copyright 2026 The XProf Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Aggregation and human-readable reporting of XLoop results.

This module owns the bridge from a worker's `results.json` to the proto:
`build_execution_result` turns the flat dictionary the harness emits into an
`ExecutionResult`, and `aggregate_report` reduces per-config results to the
`ExecutionReport` an agent or a dashboard reads.
"""

from __future__ import annotations

import collections.abc
import dataclasses
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

# Two-sided coverage that `speedup_interval` aims for. A verdict other than
# "no change" also needs that interval to exclude 1.0.
SPEEDUP_INTERVAL_CONFIDENCE = 0.95

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


def _paired_log_ratios(
    numerators: _Sequence[float], denominators: _Sequence[float]
) -> list[float]:
  """Log of each paired ratio, skipping pairs that are not positive."""
  return [
      math.log(n / d)
      for n, d in zip(numerators, denominators)
      if n > 0 and d > 0 and math.isfinite(n) and math.isfinite(d)
  ]


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
  logs = _paired_log_ratios(numerators, denominators)
  if not logs:
    return None
  return math.exp(sum(logs) / len(logs))


def _sign_test_rank(n: int, confidence: float) -> int:
  """Returns the order-statistic rank for a sign-test median interval.

  For `n` sorted values, `[x_(k), x_(n+1-k)]` contains the population median
  with probability `1 - 2 * P(B < k)`, where `B` is binomial with `n` trials
  and probability one half. This makes no assumption about the shape of the
  distribution, so one slow sample can move the interval by one rank at most.

  Args:
    n: Number of values.
    confidence: Required two-sided coverage, e.g. 0.95.

  Returns:
    The largest 1-based rank `k` that gives at least `confidence`, or 0 if no
    rank does, because `n` is too small.
  """
  rank = 0
  lower_tail = 0.0
  for k in range(1, (n + 1) // 2 + 1):
    lower_tail += math.comb(n, k - 1) / 2.0**n
    if 2.0 * lower_tail > 1.0 - confidence:
      break
    rank = k
  return rank


def speedup_interval(
    ref_times: _Sequence[float],
    code_times: _Sequence[float],
    confidence: float = SPEEDUP_INTERVAL_CONFIDENCE,
) -> tuple[float, float] | None:
  """Computes a distribution-free interval on the median paired speedup.

  Each sample gives one ratio, reference time over candidate time. The
  interval comes from the order statistics of those ratios (a sign-test
  interval). It is not symmetric around the mean, and a few slow samples on one
  side cannot move it far. A noise band alone does not do this: a band is at
  most `MAX_SPEEDUP_EPSILON` wide, so a large spread can still produce a
  "faster" verdict between identical kernels.

  Args:
    ref_times: Per-sample reference execution times.
    code_times: Per-sample candidate execution times, paired by index.
    confidence: Required two-sided coverage.

  Returns:
    `(low, high)` speedups, or None if there are too few usable pairs for the
    coverage. At 95% coverage, at least 6 pairs are necessary.
  """
  logs = sorted(_paired_log_ratios(ref_times, code_times))
  rank = _sign_test_rank(len(logs), confidence)
  if not rank:
    return None
  return math.exp(logs[rank - 1]), math.exp(logs[len(logs) - rank])


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


def _device_times(results: dict[str, Any], role: str) -> dict[str, list[float]]:
  """Each device's per-sample times for one role, keyed by device id.

  Args:
    results: The worker's results payload.
    role: Role to read.

  Returns:
    The times, or an empty dictionary if the run used one device or the
    payload holds no usable detail. Malformed entries are skipped.
  """
  raw = results.get(f"{role}_device_execution_times_sec")
  if not isinstance(raw, dict):
    return {}
  per_device: dict[str, list[float]] = {}
  for device, times in raw.items():
    if not isinstance(times, list):
      continue
    try:
      per_device[str(device)] = [float(v) for v in times]
    except (TypeError, ValueError):
      continue
  return per_device


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
    `timing_source` records which, per role. For trace times, `timing_device`
    names the device that set them most often and `timing_module` the module
    that was timed. When the run used more than one device, the per-device
    maps of each sample hold every device's time, and the sample's time is the
    slowest of them. `timing_notes` says why a device was left out or why the
    trace gave no times.
  """
  roles = list(roles) if roles is not None else list(input_utils.ROLES)
  present = [r for r in roles if f"{r}_active_sec" in results]

  out = xloop_item_pb2.ExecutionResult()
  out.execution_status = xloop_item_pb2.CODE_STATUS_EXECUTE_SUCCESS

  sources = {role: _timing_source(results, role) for role in present}
  for role, (source, _) in sources.items():
    out.timing_source[role] = source

  # Device detail belongs to trace times only; a wall-clock fallback has none.
  device_times = {
      role: _device_times(results, role)
      for role in present
      if sources[role][0] == "trace"
  }
  for role in device_times:
    device = results.get(f"{role}_timing_device")
    if isinstance(device, str) and device:
      out.timing_device[role] = device
    module = results.get(f"{role}_timing_module")
    if isinstance(module, str) and module:
      out.timing_module[role] = module
  # Notes apply to both sources: they also say why a role fell back to wall
  # clock.
  for role in present:
    notes = results.get(f"{role}_timing_notes")
    if isinstance(notes, list) and notes:
      out.timing_notes[role] = "; ".join(str(note) for note in notes)

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

    for role, per_device in device_times.items():
      field = getattr(sample, f"{role}_device_execution_times_sec", None)
      if field is None:
        continue
      for device, times in per_device.items():
        if index < len(times):
          field[device] = times[index]

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

  environment = results.get("environment")
  if isinstance(environment, dict):
    _fill_environment(environment, out.environment)

  return out


def _is_repeated(field: Any) -> bool:
  """Whether a protobuf field descriptor is repeated.

  Newer protobuf releases remove `FieldDescriptor.label` and offer
  `is_repeated` instead. Older ones have only `label`.

  Args:
    field: A field descriptor.

  Returns:
    True if the field is repeated.
  """
  is_repeated = getattr(field, "is_repeated", None)
  if is_repeated is not None:
    return bool(is_repeated)
  return field.label == field.LABEL_REPEATED


def _fill_environment(
    environment: dict[str, Any], out: xloop_item_pb2.RunEnvironment
) -> None:
  """Copies the known keys of an environment record into `out`.

  Unknown keys and values of the wrong shape are skipped, so a worker built
  from a newer engine cannot fail the run by recording more than this one
  knows about.

  Args:
    environment: Record produced by `eval_jax_lib.describe_environment`.
    out: Message to fill.
  """
  for field in out.DESCRIPTOR.fields:
    value = environment.get(field.name)
    if value is None:
      continue
    try:
      if _is_repeated(field):
        if isinstance(value, (list, tuple)):
          getattr(out, field.name).extend(str(v) for v in value)
      elif field.type == field.TYPE_STRING:
        setattr(out, field.name, str(value))
      else:
        setattr(out, field.name, int(value))
    except (TypeError, ValueError):
      continue


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

  # The noise band comes from the spread of the pooled samples, so a noisy
  # kernel is held to a looser bar than a stable one. The paired interval
  # stops a few slow samples on one side from making a verdict alone.
  if report.gmean_speedup_over_reference > 0:
    judgement = judge_speedup(
        reference, custom, report.gmean_speedup_over_reference
    )
    report.speedup_noise_band = judgement.noise_band
    report.verdict = judgement.verdict
    report.verdict_note = judgement.note
    if judgement.interval is not None:
      report.speedup_interval_low, report.speedup_interval_high = (
          judgement.interval
      )

  return report


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


@dataclasses.dataclass(frozen=True)
class SpeedupJudgement:
  """A verdict on a speedup and the evidence it rests on.

  Attributes:
    verdict: `faster`, `slower` or `no change`.
    noise_band: Relative band from `compute_speedup_epsilon`.
    interval: `(low, high)` from `speedup_interval`, or None if there are too
      few pairs for one.
    note: Why a speedup outside the noise band is still `no change`. Empty
      otherwise.
  """

  verdict: str
  noise_band: float
  interval: tuple[float, float] | None
  note: str = ""


def judge_speedup(
    ref_times: _Sequence[float],
    code_times: _Sequence[float],
    speedup: float,
) -> SpeedupJudgement:
  """Judges a speedup against the noise band and the paired interval.

  `faster` or `slower` needs two things: the speedup is outside the noise band,
  and the interval from `speedup_interval` does not include 1.0. Each check can
  only change a verdict to `no change`. Thus stable kernels get the same
  verdict as with the band alone.

  When there are too few pairs for an interval, a verdict other than
  `no change` needs samples that are stable enough to keep the band at its
  floor.

  Args:
    ref_times: Per-sample reference execution times.
    code_times: Per-sample candidate execution times, paired by index.
    speedup: Reference time divided by candidate time.

  Returns:
    The judgement.
  """
  epsilon = compute_speedup_epsilon(ref_times, code_times)
  interval = speedup_interval(ref_times, code_times)
  verdict = classify_speedup(speedup, epsilon)
  if verdict == _NO_CHANGE:
    return SpeedupJudgement(verdict, epsilon, interval)

  coverage = f"{SPEEDUP_INTERVAL_CONFIDENCE:.0%}"
  if interval is None:
    if epsilon <= MIN_SPEEDUP_EPSILON:
      return SpeedupJudgement(verdict, epsilon, interval)
    pairs = len(_paired_log_ratios(ref_times, code_times))
    note = (
        f"{pairs} paired samples are too few for a {coverage} interval on the"
        f" speedup, and their spread sets the noise band to {epsilon:.1%}."
        " Run more samples."
    )
    return SpeedupJudgement(_NO_CHANGE, epsilon, interval, note)

  low, high = interval
  if low <= 1.0 <= high:
    note = (
        f"The {coverage} interval on the paired speedup, {low:.3f} to"
        f" {high:.3f}, includes 1.0."
    )
    return SpeedupJudgement(_NO_CHANGE, epsilon, interval, note)
  return SpeedupJudgement(verdict, epsilon, interval)


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

    reference_times: list[float] = []
    custom_times: list[float] = []
    for result in item.execution_results.values():
      reference_times.extend(sample_times(result, "reference"))
      custom_times.extend(sample_times(result, "custom"))

    # Reports written before the verdict was recorded fall back to deriving
    # it here from the sample spread.
    verdict = report.verdict
    if not verdict and speedup:
      verdict = judge_speedup(reference_times, custom_times, speedup).verdict

    rows.append((
        item.name or item.id or "(unnamed)",
        xloop_item_pb2.CodeStatus.Name(report.execution_status),
        _match_str(report.custom_outputs_match_reference, bool(custom_times)),
        _fmt_speedup(speedup),
        verdict or "-",
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
    for role in sorted(result.timing_notes):
      lines.append(f"    timing note ({role}): {result.timing_notes[role]}")

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
    medians = []
    for role in roles:
      times = [t for t in sample_times(result, role) if t > 0]
      medians.append(_fmt_time(statistics.median(times)) if times else "-")
    rows.append(["median"] + medians)

    widths = [
        max(len(header[i]), max(len(r[i]) for r in rows))
        for i in range(len(header))
    ]
    lines.append("    " + _row(header, widths))
    for row in rows:
      lines.append("    " + _row(row, widths))

  return "\n".join(lines)
