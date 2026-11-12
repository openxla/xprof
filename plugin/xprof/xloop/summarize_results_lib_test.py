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
"""Unit tests for speedup aggregation, noise thresholds, and table rendering."""

import statistics
import types

from absl.testing import absltest

from xprof.protobuf import xloop_item_pb2
from xprof.xloop import summarize_results_lib


def _result(
    pairs: list[tuple[float, float]], match: bool = True
) -> xloop_item_pb2.ExecutionResult:
  """Builds a result from (reference, custom) time pairs."""
  out = xloop_item_pb2.ExecutionResult(
      execution_status=xloop_item_pb2.CODE_STATUS_EXECUTE_SUCCESS
  )
  for reference, custom in pairs:
    sample = out.sample_execution_results.add()
    sample.reference_execution_time_sec = reference
    sample.custom_execution_time_sec = custom
    sample.custom_outputs_match_reference = match
  return out


class SpeedupEpsilonTest(absltest.TestCase):

  def test_too_few_samples_uses_default(self):
    self.assertEqual(
        summarize_results_lib.compute_speedup_epsilon([1.0], [1.0]),
        summarize_results_lib.DEFAULT_SPEEDUP_EPSILON,
    )

  def test_non_positive_times_are_discarded(self):
    # Only one usable sample per arm remains, so this is the default case.
    self.assertEqual(
        summarize_results_lib.compute_speedup_epsilon(
            [1.0, 0.0, -1.0], [1.0, 0.0, 0.0]
        ),
        summarize_results_lib.DEFAULT_SPEEDUP_EPSILON,
    )

  def test_stable_samples_hit_the_floor(self):
    steady = [1.0] * 8
    self.assertEqual(
        summarize_results_lib.compute_speedup_epsilon(steady, steady),
        summarize_results_lib.MIN_SPEEDUP_EPSILON,
    )

  def test_noisy_samples_widen_epsilon(self):
    steady = [1.0] * 8
    noisy = [0.2, 0.6, 1.0, 1.4, 1.8, 2.2, 2.6, 3.0]
    quiet = summarize_results_lib.compute_speedup_epsilon(steady, steady)
    loud = summarize_results_lib.compute_speedup_epsilon(noisy, noisy)
    self.assertGreater(loud, quiet)

  def test_epsilon_is_capped(self):
    wild = [0.001, 1.0, 100.0, 5000.0, 0.02, 900.0, 3.0, 700.0]
    self.assertLessEqual(
        summarize_results_lib.compute_speedup_epsilon(wild, wild),
        summarize_results_lib.MAX_SPEEDUP_EPSILON,
    )


class ClassifySpeedupTest(absltest.TestCase):

  def test_within_epsilon_is_no_change(self):
    self.assertEqual(
        summarize_results_lib.classify_speedup(1.01, 0.03), "no change"
    )

  def test_above_epsilon_is_faster(self):
    self.assertEqual(
        summarize_results_lib.classify_speedup(1.5, 0.03), "faster"
    )

  def test_below_epsilon_is_slower(self):
    self.assertEqual(
        summarize_results_lib.classify_speedup(0.5, 0.03), "slower"
    )

  def test_a_wide_epsilon_absorbs_a_real_looking_win(self):
    # The point of the widening: a 5% win on noisy data is not a win.
    self.assertEqual(
        summarize_results_lib.classify_speedup(1.05, 0.10), "no change"
    )

  def test_nonsense_speedup_is_no_change(self):
    self.assertEqual(
        summarize_results_lib.classify_speedup(float("nan"), 0.03),
        "no change",
    )
    self.assertEqual(
        summarize_results_lib.classify_speedup(0.0, 0.03), "no change"
    )


# Raw device-0 times (us) from a 2-device run where both arms ran the same
# matmul. The first sample of each arm is slow, so the geometric mean says
# 1.12x, which is outside the capped 10% noise band.
_IDENTICAL_NOISY_REF_US = (
    1120.0,
    530.9,
    428.1,
    418.4,
    435.8,
    420.6,
    415.0,
    414.4,
    432.4,
    390.5,
)
_IDENTICAL_NOISY_CUSTOM_US = (
    813.2,
    415.4,
    404.4,
    388.4,
    354.9,
    387.5,
    450.9,
    429.1,
    388.2,
    328.5,
)

# Raw times (us) from a single-device calibration run: a 1024-cube matmul,
# the same code on both arms.
_CALIBRATION_REF_US = (
    9.1,
    8.9,
    9.0,
    8.9,
    9.0,
    9.0,
    9.1,
    9.0,
    9.0,
    8.9,
    9.0,
    9.0,
    9.1,
    9.0,
    9.1,
    9.0,
    9.0,
    9.1,
    9.0,
    9.1,
)
_CALIBRATION_CUSTOM_US = (
    9.0,
    9.0,
    9.0,
    9.1,
    9.0,
    9.4,
    9.1,
    9.2,
    9.2,
    9.1,
    8.9,
    9.0,
    9.0,
    8.9,
    9.3,
    9.0,
    9.1,
    8.9,
    9.1,
    8.9,
)


def _seconds(times_us: tuple[float, ...]) -> list[float]:
  return [t * 1e-6 for t in times_us]


def _geomean_speedup(ref: list[float], custom: list[float]) -> float:
  return statistics.geometric_mean(r / c for r, c in zip(ref, custom))


def _item_from_times(
    ref_us: tuple[float, ...], custom_us: tuple[float, ...]
) -> xloop_item_pb2.XLoopItem:
  item = xloop_item_pb2.XLoopItem(name="k")
  item.execution_results["a"].CopyFrom(
      _result(list(zip(_seconds(ref_us), _seconds(custom_us))))
  )
  return item


class SpeedupIntervalTest(absltest.TestCase):

  def _interval(
      self, ref: list[float], custom: list[float]
  ) -> tuple[float, float]:
    interval = summarize_results_lib.speedup_interval(ref, custom)
    self.assertIsNotNone(interval)
    assert interval is not None  # Narrows the type for the type checker.
    return interval

  def test_too_few_pairs_give_no_interval(self):
    self.assertIsNone(
        summarize_results_lib.speedup_interval([2.0] * 5, [1.0] * 5)
    )

  def test_six_pairs_give_the_extreme_ratios(self):
    ref = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    low, high = self._interval(ref, [1.0] * 6)
    self.assertAlmostEqual(low, 1.0)
    self.assertAlmostEqual(high, 6.0)

  def test_one_slow_sample_moves_the_interval_one_rank(self):
    ref = [1.0] * 10
    custom = [1.0] * 9 + [100.0]
    low, high = self._interval(ref, custom)
    self.assertAlmostEqual(low, 1.0)
    self.assertAlmostEqual(high, 1.0)

  def test_non_positive_pairs_are_skipped(self):
    ref = [2.0] * 6 + [0.0]
    custom = [1.0] * 6 + [1.0]
    low, high = self._interval(ref, custom)
    self.assertAlmostEqual(low, 2.0)
    self.assertAlmostEqual(high, 2.0)

  def test_the_interval_on_identical_noisy_kernels_includes_one(self):
    low, high = self._interval(
        _seconds(_IDENTICAL_NOISY_REF_US), _seconds(_IDENTICAL_NOISY_CUSTOM_US)
    )
    self.assertLess(low, 1.0)
    self.assertGreater(high, 1.0)


class JudgeSpeedupTest(absltest.TestCase):

  def _judge(
      self, ref: list[float], custom: list[float]
  ) -> summarize_results_lib.SpeedupJudgement:
    speedup = _geomean_speedup(ref, custom)
    return summarize_results_lib.judge_speedup(ref, custom, speedup)

  def test_identical_noisy_kernels_are_no_change(self):
    # Regression: the band alone called this "faster" at 1.12x.
    ref = _seconds(_IDENTICAL_NOISY_REF_US)
    custom = _seconds(_IDENTICAL_NOISY_CUSTOM_US)
    speedup = _geomean_speedup(ref, custom)
    epsilon = summarize_results_lib.compute_speedup_epsilon(ref, custom)
    self.assertEqual(
        summarize_results_lib.classify_speedup(speedup, epsilon), "faster"
    )
    judgement = summarize_results_lib.judge_speedup(ref, custom, speedup)
    self.assertEqual(judgement.verdict, "no change")
    self.assertIn("includes 1.0", judgement.note)

  def test_calibration_samples_stay_no_change(self):
    judgement = self._judge(
        _seconds(_CALIBRATION_REF_US), _seconds(_CALIBRATION_CUSTOM_US)
    )
    self.assertEqual(judgement.verdict, "no change")
    self.assertAlmostEqual(
        judgement.noise_band, summarize_results_lib.MIN_SPEEDUP_EPSILON
    )
    self.assertEmpty(judgement.note)

  def test_a_clear_win_on_noisy_samples_is_faster(self):
    ref = [2.0, 2.6, 1.9, 2.4, 2.1, 3.0, 2.2, 2.5, 2.0, 2.7]
    custom = [1.0, 1.1, 0.9, 1.2, 1.0, 1.3, 0.95, 1.05, 1.0, 1.4]
    judgement = self._judge(ref, custom)
    self.assertEqual(judgement.verdict, "faster")
    self.assertIsNotNone(judgement.interval)
    assert judgement.interval is not None  # Narrows the type.
    self.assertGreater(judgement.interval[0], 1.0)
    self.assertEmpty(judgement.note)

  def test_a_clear_regression_is_slower(self):
    judgement = self._judge([1.0] * 10, [1.5, 1.6, 1.4] * 3 + [1.5])
    self.assertEqual(judgement.verdict, "slower")

  def test_few_stable_samples_keep_their_verdict(self):
    judgement = self._judge([2.0] * 4, [1.0] * 4)
    self.assertEqual(judgement.verdict, "faster")
    self.assertIsNone(judgement.interval)

  def test_few_noisy_samples_are_no_change(self):
    judgement = self._judge([2.0, 3.0, 2.5, 4.0], [1.0, 1.4, 1.1, 1.8])
    self.assertEqual(judgement.verdict, "no change")
    self.assertIsNone(judgement.interval)
    self.assertIn("too few", judgement.note)

  def test_within_the_band_is_no_change_without_a_note(self):
    judgement = self._judge([1.01] * 8, [1.0] * 8)
    self.assertEqual(judgement.verdict, "no change")
    self.assertEmpty(judgement.note)


class AggregateReportTest(absltest.TestCase):

  def test_speedup_is_reference_over_custom(self):
    item = xloop_item_pb2.XLoopItem()
    item.execution_results["a"].CopyFrom(_result([(2.0, 1.0), (2.0, 1.0)]))
    report = summarize_results_lib.aggregate_report(item)
    self.assertAlmostEqual(report.gmean_speedup_over_reference, 2.0)

  def test_speedups_are_geometric_not_arithmetic(self):
    # 2x and 0.5x must average to 1.0, not to 1.25.
    item = xloop_item_pb2.XLoopItem()
    item.execution_results["a"].CopyFrom(_result([(2.0, 1.0), (1.0, 2.0)]))
    report = summarize_results_lib.aggregate_report(item)
    self.assertAlmostEqual(report.gmean_speedup_over_reference, 1.0)

  def test_configs_are_pooled(self):
    item = xloop_item_pb2.XLoopItem()
    item.execution_results["fast"].CopyFrom(_result([(4.0, 1.0)]))
    item.execution_results["slow"].CopyFrom(_result([(1.0, 4.0)]))
    report = summarize_results_lib.aggregate_report(item)
    # A slow config cannot be hidden behind a fast one.
    self.assertAlmostEqual(report.gmean_speedup_over_reference, 1.0)

  def test_any_mismatch_fails_the_item(self):
    item = xloop_item_pb2.XLoopItem()
    item.execution_results["ok"].CopyFrom(_result([(1.0, 1.0)], match=True))
    item.execution_results["bad"].CopyFrom(_result([(1.0, 1.0)], match=False))
    report = summarize_results_lib.aggregate_report(item)
    self.assertFalse(report.custom_outputs_match_reference)

  def test_correctness_is_false_when_nothing_ran(self):
    # Absence of evidence must not read as a pass.
    item = xloop_item_pb2.XLoopItem()
    item.execution_results["a"].CopyFrom(
        xloop_item_pb2.ExecutionResult(
            execution_status=xloop_item_pb2.CODE_STATUS_EXECUTE_SUCCESS
        )
    )
    report = summarize_results_lib.aggregate_report(item)
    self.assertFalse(report.custom_outputs_match_reference)

  def test_a_failing_config_propagates_its_status(self):
    item = xloop_item_pb2.XLoopItem()
    item.execution_results["ok"].CopyFrom(_result([(1.0, 1.0)]))
    item.execution_results["bad"].CopyFrom(
        xloop_item_pb2.ExecutionResult(
            execution_status=xloop_item_pb2.CODE_STATUS_EXECUTE_ERROR
        )
    )
    report = summarize_results_lib.aggregate_report(item)
    self.assertEqual(
        report.execution_status, xloop_item_pb2.CODE_STATUS_EXECUTE_ERROR
    )

  def test_zero_times_do_not_produce_a_speedup(self):
    item = xloop_item_pb2.XLoopItem()
    item.execution_results["a"].CopyFrom(_result([(0.0, 0.0)]))
    report = summarize_results_lib.aggregate_report(item)
    self.assertEqual(report.gmean_speedup_over_reference, 0.0)


class BuildExecutionResultTest(absltest.TestCase):

  def test_per_sample_times_are_read(self):
    result = summarize_results_lib.build_execution_result({
        "num_executions": 2,
        "reference_active_sec": 3.0,
        "reference_per_sample_sec": [1.0, 2.0],
        "custom_active_sec": 1.0,
        "custom_per_sample_sec": [0.4, 0.6],
    })
    self.assertLen(result.sample_execution_results, 2)
    self.assertAlmostEqual(
        result.sample_execution_results[0].reference_execution_time_sec, 1.0
    )
    self.assertAlmostEqual(
        result.sample_execution_results[1].custom_execution_time_sec, 0.6
    )

  def test_aggregate_time_is_spread_when_no_detail(self):
    result = summarize_results_lib.build_execution_result({
        "num_executions": 4,
        "reference_active_sec": 8.0,
    })
    self.assertLen(result.sample_execution_results, 4)
    for sample in result.sample_execution_results:
      self.assertAlmostEqual(sample.reference_execution_time_sec, 2.0)

  def test_absent_roles_are_skipped(self):
    result = summarize_results_lib.build_execution_result({
        "num_executions": 1,
        "custom_active_sec": 1.0,
    })
    self.assertEqual(
        result.sample_execution_results[0].reference_execution_time_sec, 0.0
    )

  def test_environment_is_carried(self):
    result = summarize_results_lib.build_execution_result({
        "num_executions": 1,
        "custom_active_sec": 1.0,
        "environment": {
            "platform": "tpu",
            "device_kind": "TPU v6 lite",
            "device_count": 8,
            "xla_flags": ["--xla_foo=1"],
            "not_a_field": "ignored",
        },
    })
    self.assertEqual(result.environment.platform, "tpu")
    self.assertEqual(result.environment.device_kind, "TPU v6 lite")
    self.assertEqual(result.environment.device_count, 8)
    self.assertEqual(list(result.environment.xla_flags), ["--xla_foo=1"])

  def test_malformed_environment_values_are_skipped(self):
    result = summarize_results_lib.build_execution_result({
        "num_executions": 1,
        "custom_active_sec": 1.0,
        "environment": {"device_count": "many", "xla_flags": "--not-a-list"},
    })
    self.assertEqual(result.environment.device_count, 0)
    self.assertEmpty(result.environment.xla_flags)

  def test_repeated_fields_are_found_without_the_label_attribute(self):
    # Newer protobuf releases have no `FieldDescriptor.label`.
    field = types.SimpleNamespace(is_repeated=True)
    self.assertTrue(summarize_results_lib._is_repeated(field))
    field = types.SimpleNamespace(is_repeated=False)
    self.assertFalse(summarize_results_lib._is_repeated(field))

  def test_repeated_fields_are_found_with_only_the_label_attribute(self):
    # Older protobuf releases have no `is_repeated`.
    field = types.SimpleNamespace(label=3, LABEL_REPEATED=3)
    self.assertTrue(summarize_results_lib._is_repeated(field))
    field = types.SimpleNamespace(label=1, LABEL_REPEATED=3)
    self.assertFalse(summarize_results_lib._is_repeated(field))

  def test_multi_device_trace_fills_device_maps(self):
    result = summarize_results_lib.build_execution_result({
        "num_executions": 2,
        "reference_active_sec": 1.0,
        "reference_trace_matched_execution_times_sec": [3e-6, 4e-6],
        "reference_timing_device": "0",
        "reference_device_execution_times_sec": {
            "0": [1e-6, 4e-6],
            "1": [3e-6, 2e-6],
        },
    })
    self.assertEqual(result.timing_source["reference"], "trace")
    self.assertEqual(result.timing_device["reference"], "0")
    first, second = result.sample_execution_results
    self.assertAlmostEqual(first.reference_execution_time_sec, 3e-6)
    self.assertAlmostEqual(second.reference_execution_time_sec, 4e-6)
    self.assertCountEqual(
        first.reference_device_execution_times_sec, ["0", "1"]
    )
    self.assertAlmostEqual(
        first.reference_device_execution_times_sec["1"], 3e-6
    )
    self.assertAlmostEqual(
        second.reference_device_execution_times_sec["0"], 4e-6
    )

  def test_timing_module_and_notes_are_recorded(self):
    result = summarize_results_lib.build_execution_result({
        "num_executions": 1,
        "reference_active_sec": 1.0,
        "reference_trace_matched_execution_times_sec": [2e-6],
        "reference_timing_device": "0",
        "reference_timing_module": "jit__matmul",
        "reference_timing_notes": ["device 1 is left out", "second note"],
    })
    self.assertEqual(result.timing_module["reference"], "jit__matmul")
    self.assertEqual(
        result.timing_notes["reference"], "device 1 is left out; second note"
    )
    item = xloop_item_pb2.XLoopItem(name="k")
    item.execution_results["default"].CopyFrom(result)
    self.assertIn(
        "timing note (reference): device 1 is left out",
        summarize_results_lib.format_detail(item),
    )

  def test_notes_are_kept_for_a_wall_clock_fallback(self):
    result = summarize_results_lib.build_execution_result({
        "num_executions": 1,
        "reference_active_sec": 1.0,
        "reference_per_sample_sec": [1.0],
        "reference_timing_module": "jit__matmul",
        "reference_timing_notes": ["the trace cannot be read"],
    })
    self.assertEqual(result.timing_source["reference"], "wall_clock")
    self.assertNotIn("reference", result.timing_module)
    self.assertEqual(
        result.timing_notes["reference"], "the trace cannot be read"
    )

  def test_single_device_trace_names_the_device_only(self):
    result = summarize_results_lib.build_execution_result({
        "num_executions": 1,
        "reference_active_sec": 1.0,
        "reference_trace_matched_execution_times_sec": [2e-6],
        "reference_timing_device": "0",
    })
    self.assertEqual(result.timing_device["reference"], "0")
    self.assertEmpty(
        result.sample_execution_results[0].reference_device_execution_times_sec
    )

  def test_wall_clock_fallback_has_no_device_detail(self):
    result = summarize_results_lib.build_execution_result({
        "num_executions": 1,
        "reference_active_sec": 1.0,
        "reference_per_sample_sec": [1.0],
        "reference_timing_device": "0",
        "reference_device_execution_times_sec": {"0": [1.0], "1": [2.0]},
    })
    self.assertEqual(result.timing_source["reference"], "wall_clock")
    self.assertNotIn("reference", result.timing_device)
    self.assertEmpty(
        result.sample_execution_results[0].reference_device_execution_times_sec
    )

  def test_malformed_device_times_are_skipped(self):
    result = summarize_results_lib.build_execution_result({
        "num_executions": 1,
        "reference_active_sec": 1.0,
        "reference_trace_matched_execution_times_sec": [2e-6],
        "reference_device_execution_times_sec": {
            "0": [2e-6],
            "1": "not a list",
            "2": ["bad"],
        },
    })
    self.assertEqual(
        list(
            result.sample_execution_results[
                0
            ].reference_device_execution_times_sec
        ),
        ["0"],
    )


class NoiseBandTest(absltest.TestCase):

  def test_report_records_noise_band_and_verdict(self):
    item = xloop_item_pb2.XLoopItem()
    item.execution_results["a"].CopyFrom(_result([(2.0, 1.0)] * 4))
    report = summarize_results_lib.aggregate_report(item)
    self.assertAlmostEqual(
        report.speedup_noise_band, summarize_results_lib.MIN_SPEEDUP_EPSILON
    )
    self.assertEqual(report.verdict, "faster")

  def test_small_change_is_no_change(self):
    item = xloop_item_pb2.XLoopItem()
    item.execution_results["a"].CopyFrom(_result([(1.01, 1.0)] * 4))
    report = summarize_results_lib.aggregate_report(item)
    self.assertEqual(report.verdict, "no change")

  def test_regression_is_slower(self):
    item = xloop_item_pb2.XLoopItem()
    item.execution_results["a"].CopyFrom(_result([(1.0, 2.0)] * 4))
    self.assertEqual(
        summarize_results_lib.aggregate_report(item).verdict, "slower"
    )

  def test_no_speedup_leaves_verdict_unset(self):
    item = xloop_item_pb2.XLoopItem()
    item.execution_results["a"].CopyFrom(_result([(0.0, 0.0)]))
    report = summarize_results_lib.aggregate_report(item)
    self.assertEmpty(report.verdict)
    self.assertEqual(report.speedup_noise_band, 0.0)

  def test_report_records_the_interval_and_the_note(self):
    item = _item_from_times(_IDENTICAL_NOISY_REF_US, _IDENTICAL_NOISY_CUSTOM_US)
    report = summarize_results_lib.aggregate_report(item)
    self.assertGreater(
        report.gmean_speedup_over_reference,
        1.0 + summarize_results_lib.MAX_SPEEDUP_EPSILON,
    )
    self.assertEqual(report.verdict, "no change")
    self.assertLess(report.speedup_interval_low, 1.0)
    self.assertGreater(report.speedup_interval_high, 1.0)
    self.assertIn("includes 1.0", report.verdict_note)

  def test_summary_judges_old_reports_the_same_way(self):
    item = _item_from_times(_IDENTICAL_NOISY_REF_US, _IDENTICAL_NOISY_CUSTOM_US)
    item.execution_report.CopyFrom(summarize_results_lib.aggregate_report(item))
    item.execution_report.ClearField("verdict")
    summary = summarize_results_lib.format_summary([item])
    self.assertIn("no change", summary)
    self.assertNotIn("faster", summary)

  def test_summary_uses_the_recorded_verdict(self):
    item = xloop_item_pb2.XLoopItem(name="k")
    item.execution_results["a"].CopyFrom(_result([(2.0, 1.0)] * 4))
    item.execution_report.CopyFrom(summarize_results_lib.aggregate_report(item))
    item.execution_report.verdict = "slower"
    self.assertIn("slower", summarize_results_lib.format_summary([item]))


class FormattingTest(absltest.TestCase):

  def test_summary_of_nothing_is_empty(self):
    self.assertEqual(summarize_results_lib.format_summary([]), "")

  def test_summary_names_the_item_and_its_speedup(self):
    item = xloop_item_pb2.XLoopItem(name="flash_attention")
    item.execution_results["a"].CopyFrom(_result([(2.0, 1.0)] * 4))
    item.execution_report.CopyFrom(summarize_results_lib.aggregate_report(item))
    summary = summarize_results_lib.format_summary([item])
    self.assertIn("flash_attention", summary)
    self.assertIn("2.00x", summary)
    self.assertIn("faster", summary)

  def test_summary_reports_a_correctness_failure(self):
    item = xloop_item_pb2.XLoopItem(name="broken")
    item.execution_results["a"].CopyFrom(_result([(1.0, 1.0)], match=False))
    item.execution_report.CopyFrom(summarize_results_lib.aggregate_report(item))
    self.assertIn("FAIL", summarize_results_lib.format_summary([item]))

  def test_detail_lists_each_config_and_the_error(self):
    item = xloop_item_pb2.XLoopItem(name="k")
    item.execution_results["small"].CopyFrom(_result([(1.0, 0.5)]))
    item.execution_results["large"].CopyFrom(
        xloop_item_pb2.ExecutionResult(
            execution_status=xloop_item_pb2.CODE_STATUS_EXECUTE_ERROR,
            error_message="out of memory",
        )
    )
    detail = summarize_results_lib.format_detail(item)
    self.assertIn("small", detail)
    self.assertIn("large", detail)
    self.assertIn("out of memory", detail)

  def test_detail_uses_readable_time_units(self):
    item = xloop_item_pb2.XLoopItem(name="k")
    item.execution_results["a"].CopyFrom(_result([(1e-5, 1e-5)]))
    self.assertIn("us", summarize_results_lib.format_detail(item))

  def test_detail_gives_a_median_for_every_role(self):
    item = xloop_item_pb2.XLoopItem(name="k")
    item.execution_results["a"].CopyFrom(
        _result([(1e-5, 1e-6), (3e-5, 3e-6), (2e-5, 2e-6)])
    )
    detail = summarize_results_lib.format_detail(item)
    median_rows = [
        line.split() for line in detail.splitlines() if "median" in line
    ]
    self.assertEqual(median_rows, [["median", "20.0us", "2.0us"]])


if __name__ == "__main__":
  absltest.main()
