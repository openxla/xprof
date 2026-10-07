"""Tests for default regime selection by tier and the coverage report."""

import json

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from xprof.xparity import numerical_generator
from xprof.xparity import numerical_validator
from xprof.xparity import xparity_tool

_SHAPES = (8, 8)


def _ref(x):
  return np.asarray(x, dtype=np.float32) * 2.0


def _wrong_on_large_inputs(x):
  # Correct for |x| <= 8, which covers the 'normal' regime, and wrong above.
  return _ref(np.clip(np.asarray(x, dtype=np.float32), -8.0, 8.0))


def _all_regimes(tier):
  suite = numerical_generator.generate_test_suite(
      _SHAPES, dtype_str="float32", tier=tier
  )
  return sorted({b["regime"] for b in suite})


class TierSelectionTest(parameterized.TestCase):

  @parameterized.parameters("presubmit", "deep_fuzzing")
  def test_gating_tiers_run_every_regime(self, tier):
    report = numerical_validator.validate_kernels(
        _ref, _ref, shapes=_SHAPES, dtype_str="float32", tier=tier
    )
    self.assertTrue(report.is_numerically_equivalent, report.summary_message)
    coverage = report.coverage
    self.assertEqual(coverage["selection"], "full_suite")
    self.assertEqual(coverage["regimes_run"], _all_regimes(tier))
    self.assertEmpty(coverage["regimes_not_run"])
    self.assertEqual(coverage["batches_run"], coverage["batches_available"])
    self.assertNotIn("Coverage:", report.summary_message)

  def test_presubmit_catches_bug_outside_normal_regime(self):
    report = numerical_validator.validate_kernels(
        _ref,
        _wrong_on_large_inputs,
        shapes=_SHAPES,
        dtype_str="float32",
        tier="presubmit",
    )
    self.assertFalse(report.is_numerically_equivalent)
    failed = {b.regime for b in report.batch_results if not b.passed}
    self.assertIn("outliers", failed)
    self.assertNotIn("normal", failed)

  def test_fast_agent_keeps_normal_first_and_reports_the_gap(self):
    report = numerical_validator.validate_kernels(
        _ref,
        _wrong_on_large_inputs,
        shapes=_SHAPES,
        dtype_str="float32",
        tier="fast_agent",
    )
    self.assertTrue(report.is_numerically_equivalent)
    coverage = report.coverage
    self.assertEqual(coverage["selection"], "normal_first")
    self.assertEqual(coverage["regimes_run"], ["normal"])
    self.assertIn("outliers", coverage["regimes_not_run"])
    self.assertIn(
        "Coverage: passed on regimes ['normal'] only", report.summary_message
    )

  def test_fast_agent_triage_is_reported(self):
    def broken(x):
      return _ref(x) + np.float32(1e-3)

    report = numerical_validator.validate_kernels(
        _ref, broken, shapes=_SHAPES, dtype_str="float32", tier="fast_agent"
    )
    self.assertFalse(report.is_numerically_equivalent)
    self.assertEqual(report.coverage["selection"], "normal_first_with_triage")
    self.assertEmpty(report.coverage["regimes_not_run"])

  def test_requested_regimes_are_reported(self):
    report = numerical_validator.validate_kernels(
        _ref,
        _ref,
        shapes=_SHAPES,
        dtype_str="float32",
        tier="presubmit",
        regimes=["normal"],
    )
    self.assertEqual(report.coverage["selection"], "requested")
    self.assertEqual(report.coverage["regimes_run"], ["normal"])

  def test_tool_json_includes_coverage(self):
    payload = json.loads(
        xparity_tool.verify_numerical_parity(
            _ref, _ref, shapes=_SHAPES, dtype_str="float32", tier="fast_agent"
        )
    )
    self.assertEqual(payload["coverage"]["regimes_run"], ["normal"])


if __name__ == "__main__":
  absltest.main()
