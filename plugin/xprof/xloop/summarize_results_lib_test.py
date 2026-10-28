"""Unit tests for speedup aggregation, noise thresholds, and table rendering."""

import math

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

  def test_utilization_metrics_are_carried(self):
    result = summarize_results_lib.build_execution_result({
        "num_executions": 1,
        "custom_active_sec": 1.0,
        "custom_utilization_metrics": {"mxu": 0.42},
    })
    self.assertAlmostEqual(result.custom_utilization_metrics["mxu"], 0.42)


class FormattingTest(absltest.TestCase):

  def test_fractions_render_as_percentages(self):
    self.assertEqual(
        summarize_results_lib.format_utilization_value(0.42), "42.0%"
    )

  def test_values_above_one_are_left_alone(self):
    self.assertEqual(
        summarize_results_lib.format_utilization_value(3.5), "3.50"
    )

  def test_non_finite_utilization_is_not_a_number(self):
    self.assertEqual(
        summarize_results_lib.format_utilization_value(math.inf), "n/a"
    )

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


if __name__ == "__main__":
  absltest.main()
