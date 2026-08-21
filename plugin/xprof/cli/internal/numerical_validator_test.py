"""Tests for numerical_validator ULP distance measurement and kernel comparisons."""

from typing import Any
from absl.testing import absltest
from absl.testing import parameterized
import ml_dtypes
import numpy as np
from xprof.cli.internal import numerical_validator


def _ref_mult_two(x: np.ndarray) -> np.ndarray:
  return x * 2.0


def _buggy_nan_fn(x: np.ndarray) -> np.ndarray:
  out = np.array(x * 2.0)
  out[0, 0] = np.nan
  return out


def _ref_add_one(x: np.ndarray) -> np.ndarray:
  return x + 1.0


def _buggy_inf_fn(x: np.ndarray) -> np.ndarray:
  out = np.array(x + 1.0)
  out[0, 0] = np.inf
  return out


def _identity_fn(x: np.ndarray) -> np.ndarray:
  return x


def _wrong_shape_fn(x: np.ndarray) -> np.ndarray:
  return x[:, :8]


def _off_by_5_ulp_fn(x: np.ndarray) -> np.ndarray:
  raw = np.asarray(x).astype(np.float32).view(np.uint32)
  return (raw + 5).view(np.float32)


class NumericalValidatorTest(parameterized.TestCase):

  # ===========================================================================
  # 1. Bitwise Step & Consecutive Float ULP Tests
  # ===========================================================================

  @parameterized.named_parameters(
      ("float32", "float32", np.float32, np.uint32),
      ("bfloat16", "bfloat16", ml_dtypes.bfloat16, np.uint16),
      ("float16", "float16", np.float16, np.uint16),
      ("fp8_e4m3", "fp8_e4m3", ml_dtypes.float8_e4m3fn, np.uint8),
      ("fp8_e5m2", "fp8_e5m2", ml_dtypes.float8_e5m2, np.uint8),
  )
  def test_consecutive_float_is_exact_one_ulp(
      self, dtype_str: str, np_dtype: Any, uint_dtype: Any
  ):
    """Confirms adjacent representable floating-point numbers have distance 1 ULP.

    Mathematical Definition:
    ------------------------
    1 ULP is the distance between two adjacent representable floating-point
    values. Incrementing the raw integer bits of a positive float by +1 must
    yield an exact ULP distance of 1. Incrementing by +k yields exact k ULP.
    """
    base_val = np.array([1.0], dtype=np_dtype)
    raw_bits = base_val.view(uint_dtype)

    # +1 bit step
    next_bits = raw_bits + 1
    next_val = next_bits.view(np_dtype)
    ulp_dist_1 = numerical_validator.compute_ulp_distance(
        next_val, base_val, dtype_str=dtype_str
    )
    self.assertEqual(int(ulp_dist_1[0]), 1)

    # +5 bit step
    step5_bits = raw_bits + 5
    step5_val = step5_bits.view(np_dtype)
    ulp_dist_5 = numerical_validator.compute_ulp_distance(
        step5_val, base_val, dtype_str=dtype_str
    )
    self.assertEqual(int(ulp_dist_5[0]), 5)

  # ===========================================================================
  # 2. Signed Zero Equivalence Tests
  # ===========================================================================

  @parameterized.parameters(
      "float32",
      "bfloat16",
      "float16",
      "fp8_e4m3",
      "fp8_e5m2",
  )
  def test_signed_zeros_are_zero_ulp_distance(self, dtype_str: str):
    """Confirms +0.0 and -0.0 have 0 ULP distance without spurious jumps.

    Mathematical Decision:
    ----------------------
    In IEEE-754 sign-magnitude encoding, +0.0 has raw bits 0x0000 and -0.0 has
    the sign bit set (e.g. 0x8000). Since +0.0 == -0.0 in arithmetic, continuous
    sign-magnitude mapping must map both to integer index 0, guaranteeing
    distance = |0 - 0| = 0 ULP.
    """
    pos_zero = np.array([0.0])
    neg_zero = np.array([-0.0])
    ulp_dist = numerical_validator.compute_ulp_distance(
        pos_zero, neg_zero, dtype_str=dtype_str
    )
    self.assertEqual(int(ulp_dist[0]), 0)

  # ===========================================================================
  # 3. Cross-Zero & Subnormal Transition Tests across all 5 Dtypes
  # ===========================================================================

  @parameterized.parameters(
      ("float32", np.float32, 0x80000000),
      ("bfloat16", ml_dtypes.bfloat16, 0x8000),
      ("float16", np.float16, 0x8000),
      ("fp8_e4m3", ml_dtypes.float8_e4m3fn, 0x80),
      ("fp8_e5m2", ml_dtypes.float8_e5m2, 0x80),
  )
  def test_cross_zero_subnormal_transition(
      self, dtype_str: str, np_dtype: Any, sign_bit: int
  ):
    """Confirms smallest positive subnormal vs smallest negative subnormal is 2 ULP."""
    u_type = (
        np.uint32
        if dtype_str == "float32"
        else (np.uint8 if "fp8" in dtype_str else np.uint16)
    )
    min_sub_pos = np.array([1], dtype=u_type).view(np_dtype)
    min_sub_neg = np.array([sign_bit | 1], dtype=u_type).view(np_dtype)
    zero = np.array([0], dtype=u_type).view(np_dtype)

    ulp_pos_neg = numerical_validator.compute_ulp_distance(
        min_sub_pos, min_sub_neg, dtype_str=dtype_str
    )
    ulp_pos_zero = numerical_validator.compute_ulp_distance(
        min_sub_pos, zero, dtype_str=dtype_str
    )
    ulp_neg_zero = numerical_validator.compute_ulp_distance(
        min_sub_neg, zero, dtype_str=dtype_str
    )

    self.assertEqual(int(ulp_pos_neg[0]), 2, msg=f"Failed for {dtype_str}")
    self.assertEqual(int(ulp_pos_zero[0]), 1, msg=f"Failed for {dtype_str}")
    self.assertEqual(int(ulp_neg_zero[0]), 1, msg=f"Failed for {dtype_str}")

  @parameterized.parameters(
      ("float32", np.float32, 0x00800000, 0x007FFFFF),
      ("bfloat16", ml_dtypes.bfloat16, 0x0080, 0x007F),
      ("float16", np.float16, 0x0400, 0x03FF),
      ("fp8_e4m3", ml_dtypes.float8_e4m3fn, 0x08, 0x07),
      ("fp8_e5m2", ml_dtypes.float8_e5m2, 0x04, 0x03),
  )
  def test_normal_to_subnormal_boundary_is_one_ulp(
      self,
      dtype_str: str,
      np_dtype: Any,
      min_normal_raw: int,
      max_subnormal_raw: int,
  ):
    """Confirms min_normal to max_subnormal boundary step is exact 1 ULP."""
    u_type = (
        np.uint32
        if dtype_str == "float32"
        else (np.uint8 if "fp8" in dtype_str else np.uint16)
    )
    min_normal = np.array([min_normal_raw], dtype=u_type).view(np_dtype)
    max_subnormal = np.array([max_subnormal_raw], dtype=u_type).view(np_dtype)

    ulp_dist = numerical_validator.compute_ulp_distance(
        min_normal, max_subnormal, dtype_str=dtype_str
    )
    self.assertEqual(int(ulp_dist[0]), 1, msg=f"Failed for {dtype_str}")

  # ===========================================================================
  # 4. Scale Invariance & Hardware FTZ (Flush-To-Zero) Tests
  # ===========================================================================

  def test_scale_invariance_across_magnitudes(self):
    """Confirms 1 ULP step remains 1 across huge dynamic ranges."""
    for base_f32 in [1e-30, 1e-15, 1.0, 1e15, 1e30]:
      arr = np.array([base_f32], dtype=np.float32)
      raw = arr.view(np.uint32)
      arr_next = (raw + 1).view(np.float32)

      ulp = numerical_validator.compute_ulp_distance(
          arr_next, arr, dtype_str="float32"
      )
      self.assertEqual(
          int(ulp[0]), 1, msg=f"Scale invariance failed at base {base_f32}"
      )

  def test_hardware_ftz_flush_to_zero_simulation(self):
    """Simulates hardware Flush-To-Zero (FTZ) mode vs IEEE subnormal preservation."""
    # In bfloat16, max subnormal is raw 0x007F (index 127).
    # When FTZ is active on hardware, max subnormal flushes to 0.0 (index 0).
    max_subnormal = np.array([0x007F], dtype=np.uint16).view(ml_dtypes.bfloat16)
    flushed_zero = np.array([0.0], dtype=ml_dtypes.bfloat16)

    ulp_ftz_gap = numerical_validator.compute_ulp_distance(
        max_subnormal, flushed_zero, dtype_str="bfloat16"
    )
    # The ULP validator correctly detects the 127 ULP loss from FTZ flush
    self.assertEqual(int(ulp_ftz_gap[0]), 127)

  # ===========================================================================
  # 5. Negative Test Cases: NaN/Inf, Shape Mismatch, Thresholds & Dtypes
  # ===========================================================================

  def test_nan_poisoning_in_candidate_fails_validation(self):
    """Verifies candidate producing NaN is flagged has_nan_or_inf and fails."""
    report = numerical_validator.validate_kernels(
        _ref_mult_two,
        _buggy_nan_fn,
        shapes=(16, 16),
        dtype_str="float32",
        tier="fast_agent",
    )

    self.assertFalse(report.is_numerically_equivalent)
    self.assertEqual(report.overall_max_ulp, 999999)
    self.assertGreater(report.failed_batches_count, 0)
    self.assertTrue(report.batch_results[0].has_nan_or_inf)

  def test_inf_poisoning_in_candidate_fails_validation(self):
    """Verifies candidate producing Inf is flagged has_nan_or_inf and fails."""
    report = numerical_validator.validate_kernels(
        _ref_add_one,
        _buggy_inf_fn,
        shapes=(16, 16),
        dtype_str="float32",
        tier="fast_agent",
    )

    self.assertFalse(report.is_numerically_equivalent)
    self.assertEqual(report.overall_max_ulp, 999999)
    self.assertTrue(report.batch_results[0].has_nan_or_inf)

  def test_inf_poisoning_in_reference_fails_validation(self):
    """Verifies reference producing Inf is symmetrically flagged has_nan_or_inf."""
    report = numerical_validator.validate_kernels(
        _buggy_inf_fn,
        _ref_add_one,
        shapes=(16, 16),
        dtype_str="float32",
        tier="fast_agent",
    )

    self.assertFalse(report.is_numerically_equivalent)
    self.assertEqual(report.overall_max_ulp, 999999)
    self.assertTrue(report.batch_results[0].has_nan_or_inf)

  def test_shape_mismatch_between_kernels_raises_value_error(self):
    """Verifies shape mismatch between kernels raises ValueError."""
    with self.assertRaises(ValueError):
      numerical_validator.validate_kernels(
          _identity_fn,
          _wrong_shape_fn,
          shapes=(16, 16),
          dtype_str="float32",
          tier="fast_agent",
      )

  def test_ulp_threshold_violation_fails_validation(self):
    """Verifies candidate with max ULP > max_allowed_ulp fails validation."""
    report = numerical_validator.validate_kernels(
        _identity_fn,
        _off_by_5_ulp_fn,
        shapes=(16, 16),
        dtype_str="float32",
        tier="fast_agent",
        max_allowed_ulp=2,  # Strict threshold of 2 ULP
    )

    self.assertFalse(report.is_numerically_equivalent)
    self.assertEqual(report.overall_max_ulp, 5)
    self.assertGreater(report.failed_batches_count, 0)

  def test_unsupported_dtype_raises_value_error(self):
    """Verifies unsupported dtype string raises ValueError."""
    arr = np.array([1.0], dtype=np.float32)
    with self.assertRaises(ValueError):
      numerical_validator.compute_ulp_distance(
          arr, arr, dtype_str="unsupported_custom_dtype"
      )

  # ===========================================================================
  # 6. Discrete Integer & Boolean Parity Validation Tests
  # ===========================================================================

  def test_integer_exact_equality_passes(self):
    """Verifies candidate with exact integer match passes validation."""
    report = numerical_validator.validate_kernels(
        _identity_fn,
        _identity_fn,
        shapes=(16, 16),
        dtype_str="int32",
        tier="fast_agent",
    )
    self.assertTrue(report.is_numerically_equivalent)
    self.assertEqual(report.overall_max_ulp, 0)
    self.assertEqual(report.failed_batches_count, 0)

  def test_integer_mismatch_fails_with_exact_delta(self):
    """Verifies integer off-by-one is caught and delta is reported."""

    def _cand_add_one(x):
      return x + 1

    custom_suite = [{
        "name": "bounded_batch",
        "args": (np.array([[10, 20], [30, 40]], dtype=np.int32),),
        "kwargs": {},
        "regime": "small_int",
    }]
    report = numerical_validator.validate_kernels(
        _identity_fn,
        _cand_add_one,
        shapes=(2, 2),
        dtype_str="int32",
        test_suite=custom_suite,
    )
    self.assertFalse(report.is_numerically_equivalent)
    self.assertEqual(report.overall_max_ulp, 1)
    self.assertGreater(report.failed_batches_count, 0)

  def test_boolean_mask_parity_and_mismatch(self):
    """Verifies boolean mask validation and mismatch detection."""
    report_pass = numerical_validator.validate_kernels(
        _identity_fn,
        _identity_fn,
        shapes=(8, 8),
        dtype_str="bool",
        tier="fast_agent",
    )
    self.assertTrue(report_pass.is_numerically_equivalent)
    self.assertEqual(report_pass.overall_max_ulp, 0)

    def _invert_mask(x):
      return ~x

    report_fail = numerical_validator.validate_kernels(
        _identity_fn,
        _invert_mask,
        shapes=(8, 8),
        dtype_str="bool",
        tier="fast_agent",
    )
    self.assertFalse(report_fail.is_numerically_equivalent)
    self.assertEqual(report_fail.overall_max_ulp, 1)

  # ===========================================================================
  # 7. Tolerance Audit, Caution Banners & Hard Safety Ceilings
  # ===========================================================================

  def test_uint64_extreme_difference_no_overflow(self):
    """Confirms uint64 distance computation does not wrap around to 1."""
    actual = np.array([0], dtype=np.uint64)
    expected = np.array([18446744073709551615], dtype=np.uint64)  # 2**64 - 1
    dist = numerical_validator.compute_ulp_distance(
        actual, expected, dtype_str="uint64"
    )
    self.assertGreater(int(dist[0]), 1000000)

  def test_recommended_contract_and_caution_banner_emitted_when_relaxed(self):
    """Verifies caution banner is generated when max_allowed_ulp exceeds contract."""
    report = numerical_validator.validate_kernels(
        _identity_fn,
        _identity_fn,
        shapes=(16, 16),
        dtype_str="bfloat16",
        tier="fast_agent",
        max_allowed_ulp=6,
    )
    self.assertTrue(report.is_numerically_equivalent)
    self.assertIsNotNone(report.tolerance_audit)
    self.assertTrue(report.tolerance_audit.is_relaxed_override)
    self.assertEqual(report.tolerance_audit.recommended_contract_ulp, 2)
    self.assertEqual(report.tolerance_audit.configured_max_ulp, 6)
    self.assertIn("⚠️ CAUTION", report.summary_message)
    self.assertIsNotNone(report.tolerance_audit.caution_banner)
    self.assertIn(
        "recommended contract is <= 2 ULP",
        str(report.tolerance_audit.caution_banner),
    )

  def test_recommended_contract_standard_summary_when_strict(self):
    """Verifies no caution banner when max_allowed_ulp is within contract."""
    report = numerical_validator.validate_kernels(
        _identity_fn,
        _identity_fn,
        shapes=(16, 16),
        dtype_str="bfloat16",
        tier="fast_agent",
        max_allowed_ulp=2,
    )
    self.assertTrue(report.is_numerically_equivalent)
    self.assertIsNotNone(report.tolerance_audit)
    self.assertFalse(report.tolerance_audit.is_relaxed_override)
    self.assertIsNone(report.tolerance_audit.caution_banner)
    self.assertNotIn("⚠️ CAUTION", report.summary_message)
    self.assertIn("Recommended: <= 2", report.summary_message)

  @parameterized.parameters(
      ("bool", 1),
      ("int32", 1),
      ("int64", 1),
      ("fp8_e4m3", 3),
      ("fp8_e5m2", 3),
      ("bfloat16", 10),
      ("float16", 10),
      ("float32", 5),
      ("float64", 5),
  )
  def test_hard_safety_ceiling_all_dtypes_abuse_blocked(
      self, dtype_str: str, abusive_max_ulp: int
  ):
    """Verifies exceeding hard ceiling raises ValueError across all dtypes."""
    with self.assertRaises(ValueError) as ctx:
      numerical_validator.validate_kernels(
          _identity_fn,
          _identity_fn,
          shapes=(16, 16),
          dtype_str=dtype_str,
          tier="fast_agent",
          max_allowed_ulp=abusive_max_ulp,
      )
    self.assertIn("exceeds immutable safety ceiling", str(ctx.exception))

  @parameterized.parameters("bool", "int32", "int64", "uint32", "uint8")
  def test_discrete_tolerance_override_raises_value_error(
      self, dtype_str: str
  ):
    """Verifies specifying max_allowed_ulp > 0 on discrete dtypes raises ValueError."""
    with self.assertRaises(ValueError) as ctx:
      numerical_validator.validate_kernels(
          _identity_fn,
          _identity_fn,
          shapes=(8, 8),
          dtype_str=dtype_str,
          tier="fast_agent",
          max_allowed_ulp=1,
      )
    self.assertIn("exceeds immutable safety ceiling", str(ctx.exception))

  def test_multi_arg_and_kwargs_variadic_validation(self):
    """Verifies validate_kernels with multi-arguments and kwargs."""

    def _custom_fused_op(a, b, scale=1.0, bias=0.0):
      return (a + b) * scale + bias

    def _cand_fused_op(a, b, scale=1.0, bias=0.0):
      return (a + b) * scale + bias

    custom_suite = [{
        "name": "fused_batch",
        "args": (
            np.ones((4, 4), dtype=np.float32),
            np.ones((4, 4), dtype=np.float32),
        ),
        "kwargs": {"scale": 2.0, "bias": 0.5},
        "regime": "custom",
    }]
    report = numerical_validator.validate_kernels(
        _custom_fused_op,
        _cand_fused_op,
        shapes=[(4, 4), (4, 4)],
        dtype_str="float32",
        test_suite=custom_suite,
    )
    self.assertTrue(report.is_numerically_equivalent)
    self.assertEqual(report.overall_max_ulp, 0)

  def test_float64_golden_reference_bound(self):
    """Verifies float64 high-precision golden reference validation within 1 ULP."""
    report = numerical_validator.validate_kernels(
        _identity_fn,
        _identity_fn,
        shapes=(16, 16),
        dtype_str="float64",
        tier="fast_agent",
        max_allowed_ulp=1,
    )
    self.assertTrue(report.is_numerically_equivalent)
    self.assertEqual(report.overall_max_ulp, 0)
    self.assertIsNotNone(report.tolerance_audit)
    assert report.tolerance_audit is not None
    self.assertEqual(report.tolerance_audit.recommended_contract_ulp, 1)

  def test_validate_kernels_with_regimes_filtering(self):
    """Verifies validate_kernels runs only requested regimes when specified."""
    report = numerical_validator.validate_kernels(
        _identity_fn,
        _identity_fn,
        shapes=(16, 16),
        dtype_str="float32",
        tier="fast_agent",
        regimes=["student_t"],
    )
    self.assertTrue(report.is_numerically_equivalent)
    self.assertNotEmpty(report.batch_results)
    for batch in report.batch_results:
      self.assertIn("student_t", batch.batch_name)


if __name__ == "__main__":
  absltest.main()

