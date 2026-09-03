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

  # --- Oracle audit (reference-vs-float64) ---

  def test_oracle_absent_by_default(self):
    """No oracle requested -> no oracle_audit block, verdict unchanged."""
    report = numerical_validator.validate_kernels(
        _ref_mult_two,
        _ref_mult_two,
        shapes=(16, 16),
        dtype_str="bfloat16",
        tier="fast_agent",
    )
    self.assertTrue(report.is_numerically_equivalent)
    self.assertIsNone(report.oracle_audit)
    self.assertEqual(report.correctness_basis, "AGREEMENT_ONLY")
    self.assertEqual(report.run_config["tier"], "fast_agent")
    self.assertEqual(report.run_config["dtype_str"], "bfloat16")

  def test_oracle_clears_exact_reference(self):
    """An exact reference is reported as 0 ULP from the oracle, no banner."""
    report = numerical_validator.validate_kernels(
        _ref_mult_two,
        _ref_mult_two,
        shapes=(16, 16),
        dtype_str="bfloat16",
        tier="fast_agent",
        kernel_oracle=numerical_validator.ORACLE_AUTO,
    )
    self.assertIsNotNone(report.oracle_audit)
    assert report.oracle_audit is not None
    self.assertTrue(report.oracle_audit.oracle_executed_in_float64)
    self.assertEqual(report.oracle_audit.reference_max_ulp_from_oracle, 0)
    self.assertFalse(report.oracle_audit.reference_is_lossy)
    self.assertIsNone(report.oracle_audit.oracle_banner)
    self.assertEqual(report.correctness_basis, "AGREEMENT_AND_ORACLE")

  def test_oracle_flags_lossy_reference_that_candidate_agrees_with(self):
    """The false green: two kernels lossy in the same way agree at 0 ULP.

    Both sides accumulate in float16, mimicking a TPU reference that runs at
    bf16 input precision. Agreement is perfect; the oracle exposes that both
    are far from exact.
    """

    def _lossy(x: np.ndarray) -> np.ndarray:
      return (np.asarray(x, dtype=np.float16) * np.float16(3.0)).astype(
          np.float32
      )

    report = numerical_validator.validate_kernels(
        _lossy,
        _lossy,
        shapes=(16, 16),
        dtype_str="float32",
        tier="fast_agent",
        kernel_oracle=lambda x: np.asarray(x, dtype=np.float64) * 3.0,
    )
    # Agreement is perfect and the verdict is unchanged (report-only).
    self.assertTrue(report.is_numerically_equivalent)
    self.assertEqual(report.overall_max_ulp, 0)
    # But the reference is nowhere near exact.
    assert report.oracle_audit is not None
    self.assertTrue(report.oracle_audit.reference_is_lossy)
    self.assertGreater(report.oracle_audit.reference_max_ulp_from_oracle, 2)
    self.assertIn(
        "REFERENCE IS NOT EXACT", report.oracle_audit.oracle_banner or ""
    )
    self.assertIn("REFERENCE IS NOT EXACT", report.summary_message)
    self.assertEqual(report.correctness_basis, "AGREEMENT_AND_ORACLE")

  def test_oracle_not_in_float64_is_reported_not_trusted(self):
    """An oracle that silently ran at f32 must not be read as a bound."""
    report = numerical_validator.validate_kernels(
        _ref_mult_two,
        _ref_mult_two,
        shapes=(16, 16),
        dtype_str="bfloat16",
        tier="fast_agent",
        kernel_oracle=lambda x: np.asarray(x, dtype=np.float32) * 2.0,
    )
    assert report.oracle_audit is not None
    self.assertFalse(report.oracle_audit.oracle_executed_in_float64)
    self.assertEqual(report.oracle_audit.oracle_output_dtype, "float32")
    self.assertIn(
        "DID NOT EXECUTE IN FLOAT64", report.oracle_audit.oracle_banner or ""
    )

  def test_oracle_auto_passes_integer_args_through(self):
    """ORACLE_AUTO promotes floats only; indices/masks stay integral."""

    def _gather_scale(x: np.ndarray, idx: np.ndarray) -> np.ndarray:
      return x[idx] * 2.0

    suite = [{
        "name": "gather_batch",
        "regime": "boundary",
        "args": (np.arange(16, dtype=np.float32), np.arange(16)),
    }]
    report = numerical_validator.validate_kernels(
        _gather_scale,
        _gather_scale,
        shapes=(16,),
        dtype_str="float32",
        test_suite=suite,
        kernel_oracle=numerical_validator.ORACLE_AUTO,
    )
    assert report.oracle_audit is not None
    self.assertTrue(report.oracle_audit.oracle_executed_in_float64)
    self.assertEqual(report.oracle_audit.reference_max_ulp_from_oracle, 0)

  def test_oracle_auto_catches_precision_following_reference(self):
    """ORACLE_AUTO exposes precision-following references.

    A reference whose precision follows its input dtype, as a bare `jnp.dot`
    does, is tested here. Promoting the inputs to float64 re-runs the reference
    at float64 and the truncation shows up.
    """

    def _precision_following(x: np.ndarray) -> np.ndarray:
      xa = np.asarray(x)
      if xa.dtype == np.float32:  # backend default truncates to bf16
        xa = xa.astype(ml_dtypes.bfloat16).astype(np.float32)
      return xa * 3.0

    report = numerical_validator.validate_kernels(
        _precision_following,
        _precision_following,
        shapes=(32, 32),
        dtype_str="float32",
        tier="fast_agent",
        kernel_oracle=numerical_validator.ORACLE_AUTO,
    )
    self.assertTrue(report.is_numerically_equivalent)  # agreement is perfect
    assert report.oracle_audit is not None
    self.assertTrue(report.oracle_audit.oracle_executed_in_float64)
    self.assertTrue(report.oracle_audit.reference_is_lossy)  # but not correct

  def test_oracle_auto_warns_on_hardcoded_internal_precision(self):
    """ORACLE_AUTO's blind spot, and the guard that keeps it from lying.

    A reference that hardcodes `.astype(bfloat16)` in its body ignores the
    promoted inputs, so `auto` cannot measure its loss. Because the hardcode
    also pins the output dtype, the float64 guard fires and the report declines
    to certify rather than reporting a clean 0 ULP. An explicit float64 oracle
    catches it outright.
    """

    def _hardcoded(x: np.ndarray) -> np.ndarray:
      xb = np.asarray(x).astype(ml_dtypes.bfloat16).astype(np.float32)
      return xb * 3.0

    auto = numerical_validator.validate_kernels(
        _hardcoded,
        _hardcoded,
        shapes=(32, 32),
        dtype_str="float32",
        tier="fast_agent",
        kernel_oracle=numerical_validator.ORACLE_AUTO,
    )
    assert auto.oracle_audit is not None
    self.assertFalse(auto.oracle_audit.reference_is_lossy)  # blind...
    self.assertFalse(
        auto.oracle_audit.oracle_executed_in_float64
    )  # ...but loud
    self.assertIn(
        "DID NOT EXECUTE IN FLOAT64", auto.oracle_audit.oracle_banner or ""
    )

    explicit = numerical_validator.validate_kernels(
        _hardcoded,
        _hardcoded,
        shapes=(32, 32),
        dtype_str="float32",
        tier="fast_agent",
        kernel_oracle=lambda x: np.asarray(x, dtype=np.float64) * 3.0,
    )
    assert explicit.oracle_audit is not None
    self.assertTrue(explicit.oracle_audit.reference_is_lossy)

  def test_oracle_rejects_unknown_sentinel(self):
    with self.assertRaises(ValueError):
      numerical_validator.validate_kernels(
          _ref_mult_two,
          _ref_mult_two,
          shapes=(16, 16),
          dtype_str="bfloat16",
          tier="fast_agent",
          kernel_oracle="float64_please",
      )

  # ===========================================================================
  # 9. Correctness Basis, Run Config & Device Provenance Tests
  # ===========================================================================

  def test_correctness_basis_agreement_only(self):
    """Verifies correctness_basis is AGREEMENT_ONLY when oracle is absent."""
    report = numerical_validator.validate_kernels(
        _identity_fn,
        _identity_fn,
        shapes=(16, 16),
        dtype_str="bfloat16",
        tier="fast_agent",
    )
    self.assertEqual(report.correctness_basis, "AGREEMENT_ONLY")
    self.assertIsNone(report.oracle_audit)

  def test_correctness_basis_agreement_and_oracle(self):
    """Verifies correctness_basis is AGREEMENT_AND_ORACLE when oracle runs."""
    report = numerical_validator.validate_kernels(
        _identity_fn,
        _identity_fn,
        shapes=(16, 16),
        dtype_str="float32",
        tier="fast_agent",
        kernel_oracle=numerical_validator.ORACLE_AUTO,
    )
    self.assertEqual(report.correctness_basis, "AGREEMENT_AND_ORACLE")
    self.assertIsNotNone(report.oracle_audit)

  def test_run_config_payload_contract_and_device_kind(self):
    """Verifies run_config echoes tier, seed, dtype, device_kind and count."""
    report = numerical_validator.validate_kernels(
        _identity_fn,
        _identity_fn,
        shapes=(16, 16),
        dtype_str="bfloat16",
        tier="fast_agent",
        seed=1234,
        device_kind="tpu_v5p",
    )
    config = report.run_config
    self.assertIn("tier", config)
    self.assertEqual(config["tier"], "fast_agent")
    self.assertIn("seed", config)
    self.assertEqual(config["seed"], 1234)
    self.assertIn("dtype_str", config)
    self.assertEqual(config["dtype_str"], "bfloat16")
    self.assertIn("device_kind", config)
    self.assertEqual(config["device_kind"], "tpu_v5p")
    self.assertIn("total_batches_count", config)
    self.assertLen(report.batch_results, config["total_batches_count"])

  def test_run_config_auto_detects_default_device_kind(self):
    """Verifies device_kind auto-detection falls back to a non-empty string."""
    report = numerical_validator.validate_kernels(
        _identity_fn,
        _identity_fn,
        shapes=(16, 16),
        dtype_str="bfloat16",
        tier="fast_agent",
    )
    config = report.run_config
    self.assertIn("device_kind", config)
    self.assertIsInstance(config["device_kind"], str)
    self.assertTrue(config["device_kind"])

  # ===========================================================================
  # 10. UlpContext, Dual Gating, Pin-Inert & Default Regimes Tests
  # ===========================================================================

  def test_ulp_context_p50_and_reliability(self):
    """Verifies UlpContext fields (p50, p99_9, reliable, bit_identical)."""
    report = numerical_validator.validate_kernels(
        _identity_fn,
        _identity_fn,
        shapes=(16, 16),
        dtype_str="float32",
        tier="fast_agent",
    )
    self.assertIsNotNone(report.ulp_context)
    self.assertTrue(report.ulp_context.bit_identical)
    self.assertEqual(report.ulp_context.p50, 0.0)
    self.assertEqual(report.ulp_context.p99_9, 0.0)
    self.assertEqual(report.ulp_context.max_ulp, 0)
    self.assertTrue(report.ulp_context.reliable)

    # Test boundary batch unreliability marking
    boundary_batch = {
        "name": "boundary_probe",
        "regime": "boundary",
        "args": (np.array([[1e-35]], dtype=np.float32),),
        "kwargs": {},
    }
    report_boundary = numerical_validator.validate_kernels(
        lambda x: x,
        lambda x: x * 100.0,
        shapes=(1, 1),
        dtype_str="float32",
        test_suite=[boundary_batch],
        max_allowed_ulp=4,
    )
    self.assertFalse(report_boundary.is_numerically_equivalent)
    b_res = report_boundary.batch_results[0]
    self.assertIsNotNone(b_res.ulp_context)
    self.assertFalse(b_res.ulp_context.reliable)
    self.assertIn("Ill-conditioned regime", b_res.ulp_context.note or "")

  def test_dual_gate_allclose_catches_divergence(self):
    """Verifies allclose dual gate check fails even when ULP is mitigated."""

    def cand_shift_fn(x: np.ndarray) -> np.ndarray:
      out = np.array(x, dtype=np.float32)
      out[0, 0] = out[0, 0] + 0.01
      return out

    report = numerical_validator.validate_kernels(
        lambda x: np.zeros_like(x, dtype=np.float32),
        cand_shift_fn,
        shapes=(8, 8),
        dtype_str="float32",
        tier="fast_agent",
    )
    self.assertFalse(report.is_numerically_equivalent)
    self.assertFalse(report.batch_results[0].allclose_passed)

  def test_pin_inert_reference_warning_on_accelerator(self):
    """Verifies pin-inert probe detects reference that ignores precision."""

    def inert_ref_fn(x: np.ndarray, precision: str = "default") -> np.ndarray:
      del precision
      return x * 2.0

    report = numerical_validator.validate_kernels(
        inert_ref_fn,
        _identity_fn,
        shapes=(16, 16),
        dtype_str="float32",
        device_kind="tpu_v5p",
    )
    config = report.run_config
    self.assertTrue(config.get("reference_pin_inert", False))
    self.assertIn("⚠️ REFERENCE IS PIN-INERT", report.summary_message)

  def test_default_regimes_normal_and_triage_fallback(self):
    """Verifies passing kernel runs 1 batch, failing kernel triggers triage."""
    # 1. Passing run: only normal batch executes
    report_pass = numerical_validator.validate_kernels(
        _identity_fn,
        _identity_fn,
        shapes=(16, 16),
        dtype_str="bfloat16",
        tier="presubmit",
    )
    self.assertTrue(report_pass.is_numerically_equivalent)
    self.assertEqual(report_pass.total_batches_count, 1)
    self.assertEqual(report_pass.batch_results[0].regime, "normal")

    # 2. Failing run: triggers triage fallback across full procedural suite
    report_fail = numerical_validator.validate_kernels(
        _identity_fn,
        _buggy_nan_fn,
        shapes=(16, 16),
        dtype_str="bfloat16",
        tier="fast_agent",
    )
    self.assertFalse(report_fail.is_numerically_equivalent)
    self.assertGreater(report_fail.total_batches_count, 1)
    regimes_executed = {b.regime for b in report_fail.batch_results}
    self.assertIn("normal", regimes_executed)
    self.assertTrue(
        regimes_executed.intersection({"student_t", "outliers", "boundary"})
    )

  def test_chunk_callable_multi_head_attention(self):
    """Verifies chunk_callable chunks multi-head tensors and reassembles."""

    def dummy_attn(q, k, v):
      return q * 0.5 + k * 0.25 + v * 0.25

    chunked_attn = numerical_validator.chunk_callable(
        dummy_attn, chunk_arg_indices=(0, 1, 2), axis=1, chunks=4
    )

    rng = np.random.default_rng(42)
    q = rng.standard_normal((2, 8, 16, 32)).astype(np.float32)
    k = rng.standard_normal((2, 8, 16, 32)).astype(np.float32)
    v = rng.standard_normal((2, 8, 16, 32)).astype(np.float32)

    out_direct = dummy_attn(q, k, v)
    out_chunked = chunked_attn(q, k, v)

    np.testing.assert_array_equal(out_direct, out_chunked)

  def test_regime_dispatch_criterion_dependent(self):
    """Verifies criterion-dependent regime selection across modes."""
    # 1. Parity (float32): runs 1 normal batch
    report_parity = numerical_validator.validate_kernels(
        _identity_fn,
        _identity_fn,
        shapes=(16, 16),
        dtype_str="float32",
        tier="fast_agent",
    )
    self.assertEqual(report_parity.total_batches_count, 1)

    # 2. Oracle audit (Q2/Q3): runs full procedural suite
    report_oracle = numerical_validator.validate_kernels(
        _identity_fn,
        _identity_fn,
        shapes=(16, 16),
        dtype_str="float32",
        tier="fast_agent",
        kernel_oracle=numerical_validator.ORACLE_AUTO,
    )
    self.assertGreater(report_oracle.total_batches_count, 1)

    # 3. Discrete integer dtype: runs full suite
    report_int = numerical_validator.validate_kernels(
        _identity_fn,
        _identity_fn,
        shapes=(16, 16),
        dtype_str="int32",
        tier="fast_agent",
    )
    self.assertGreater(report_int.total_batches_count, 1)

    # 4. FP8 dtype: runs normal batch only
    report_fp8 = numerical_validator.validate_kernels(
        _identity_fn,
        _identity_fn,
        shapes=(16, 16),
        dtype_str="fp8_e4m3",
        tier="fast_agent",
    )
    self.assertEqual(report_fp8.total_batches_count, 1)

  def test_zero_ulp_false_green_probe_warns(self):
    """Verifies that 0-ULP agreement with a lossy reference emits a caution."""

    def lossy_fn(x: np.ndarray) -> np.ndarray:
      xa = np.asarray(x)
      if xa.dtype == np.float32:
        xa = xa.astype(ml_dtypes.bfloat16).astype(np.float32)
      return xa * 2.0

    report = numerical_validator.validate_kernels(
        lossy_fn,
        lossy_fn,
        shapes=(32, 32),
        dtype_str="float32",
        tier="fast_agent",
    )
    self.assertTrue(report.is_numerically_equivalent)
    self.assertEqual(report.overall_max_ulp, 0)
    self.assertIn(
        "⚠️ 0-ULP AGREEMENT WITH LOSSY BASELINE", report.summary_message
    )

  def test_zero_ulp_clean_reference_no_warning(self):
    """Verifies exact reference agreement does not trigger false green banner."""
    report = numerical_validator.validate_kernels(
        _identity_fn,
        _identity_fn,
        shapes=(16, 16),
        dtype_str="float32",
        tier="fast_agent",
    )
    self.assertTrue(report.is_numerically_equivalent)
    self.assertEqual(report.overall_max_ulp, 0)
    self.assertNotIn(
        "⚠️ 0-ULP AGREEMENT WITH LOSSY BASELINE", report.summary_message
    )


if __name__ == "__main__":
  absltest.main()
