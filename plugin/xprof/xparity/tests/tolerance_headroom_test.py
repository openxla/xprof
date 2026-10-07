"""Tests for the report-only atol/rtol tolerance headroom."""

import json

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from xprof.xparity import numerical_validator
from xprof.xparity import xparity_tool


def _ref(x):
  return np.asarray(x, dtype=np.float32) * np.float32(2.0)


def _clipped(x):
  """Matches _ref for |x| <= 8 and is wrong for larger inputs."""
  return np.clip(np.asarray(x, dtype=np.float32), -8.0, 8.0) * np.float32(2.0)


def _scaled_by(factor):
  def fn(x):
    return _ref(x) * np.float32(factor)

  return fn


_KWARGS = dict(shapes=(64, 64), dtype_str="float32", tier="fast_agent")


class ToleranceRatioTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("equal", [1.0, -2.0], [1.0, -2.0], 0.0, 0.1, 0.0),
      ("relative", [1.01], [1.0], 0.0, 0.02, 0.5),
      ("absolute", [0.003], [0.0], 0.006, 0.0, 0.5),
      ("matching_nan_and_inf", [np.nan, np.inf], [np.nan, np.inf], 0, 1, 0.0),
      ("nan_vs_finite", [np.nan], [1.0], 1.0, 1.0, np.inf),
      ("finite_vs_inf", [1.0], [np.inf], 1.0, 1.0, np.inf),
  )
  def test_ratio(self, cand, ref, atol, rtol, expected):
    ratio = numerical_validator._tolerance_ratio(
        np.asarray(cand), np.asarray(ref), atol, rtol
    )
    self.assertAlmostEqual(ratio, expected, places=6)


class ToleranceHeadroomTest(parameterized.TestCase):

  def test_absent_without_tolerance(self):
    report = numerical_validator.validate_kernels(_ref, _ref, **_KWARGS)
    self.assertEqual(report.tolerance_headroom, {})
    self.assertIsNone(report.batch_results[0].tolerance_ratio)

  def test_reports_regimes_that_exceed_tolerance(self):
    report = numerical_validator.validate_kernels(
        _ref, _clipped, regimes="all", atol=3e-3, rtol=3e-3, **_KWARGS
    )
    headroom = report.tolerance_headroom
    self.assertEqual(headroom["atol"], 3e-3)
    self.assertEqual(headroom["max_ratio_by_regime"]["normal"], 0.0)
    self.assertNotIn("normal", headroom["regimes_over_tolerance"])
    self.assertNotEmpty(headroom["regimes_over_tolerance"])
    self.assertIn("Tolerance headroom (report only)", report.summary_message)

  def test_verdict_is_unchanged(self):
    # 1e-6 relative error is about 8 float32 ULP: well inside rtol=1e-3, but
    # above the ULP gate.
    cand = _scaled_by(1.0 + 1e-6)
    plain = numerical_validator.validate_kernels(_ref, cand, **_KWARGS)
    with_tol = numerical_validator.validate_kernels(
        _ref, cand, rtol=1e-3, **_KWARGS
    )
    self.assertFalse(plain.is_numerically_equivalent)
    self.assertEqual(
        plain.is_numerically_equivalent, with_tol.is_numerically_equivalent
    )
    self.assertEqual(with_tol.tolerance_headroom["atol"], 0.0)
    self.assertEqual(with_tol.tolerance_headroom["regimes_over_tolerance"], [])
    ratio = with_tol.tolerance_headroom["max_ratio_by_regime"]["normal"]
    self.assertBetween(ratio, 1e-4, 1e-2)

  def test_candidate_vs_oracle_ratio(self):
    def oracle(x):
      return np.asarray(x, dtype=np.float64) * 2.0

    report = numerical_validator.validate_kernels(
        _ref,
        _ref,
        kernel_oracle=oracle,
        regimes=["normal"],
        rtol=1e-6,
        **_KWARGS,
    )
    by_regime = report.tolerance_headroom["oracle_max_ratio_by_regime"]
    self.assertIn("normal", by_regime)
    self.assertLess(by_regime["normal"], 1.0)

  def test_pytree_takes_worst_leaf(self):
    def ref(x):
      return _ref(x), _ref(x)

    def cand(x):
      return _ref(x), _scaled_by(1.01)(x)

    report = numerical_validator.validate_kernels(
        ref, cand, regimes=["normal"], rtol=0.02, **_KWARGS
    )
    batch = report.batch_results[0]
    leaves = batch.leaf_results
    assert leaves is not None and batch.tolerance_ratio is not None
    self.assertEqual(leaves["[0]"].tolerance_ratio, 0.0)
    self.assertAlmostEqual(batch.tolerance_ratio, 0.5, places=2)

  @parameterized.named_parameters(
      ("negative", dict(atol=-1.0)),
      ("non_finite", dict(rtol=float("inf"))),
      ("both_zero", dict(atol=0.0, rtol=0.0)),
  )
  def test_invalid_tolerance(self, kwargs):
    with self.assertRaises(ValueError):
      numerical_validator.validate_kernels(_ref, _ref, **kwargs, **_KWARGS)

  def test_tool_json_has_headroom(self):
    payload = json.loads(
        xparity_tool.verify_numerical_parity(
            _ref, _clipped, regimes="all", atol=3e-3, rtol=3e-3, **_KWARGS
        )
    )
    headroom = payload["tolerance_headroom"]
    self.assertEqual(headroom["rtol"], 3e-3)
    self.assertNotEmpty(headroom["regimes_over_tolerance"])


if __name__ == "__main__":
  absltest.main()
