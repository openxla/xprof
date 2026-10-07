"""Tests for the bitwise verdict contract and `compare_bitwise`."""

import json

from absl.testing import absltest
from absl.testing import parameterized
import ml_dtypes
import numpy as np

from xprof.xparity import numerical_generator
from xprof.xparity import numerical_validator
from xprof.xparity import xparity_tool

_BITWISE = numerical_validator.CONTRACT_BITWISE
_ULP = numerical_validator.CONTRACT_ULP


def _identity(x: np.ndarray) -> np.ndarray:
  return np.array(x, copy=True)


def _plus_one_ulp(x: np.ndarray) -> np.ndarray:
  raw = np.asarray(x, dtype=np.float32).view(np.uint32)
  return (raw + 1).view(np.float32)


def _plus_one_ulp_on_large_values(x: np.ndarray) -> np.ndarray:
  """Exact on N(0, 1) data; off by 1 ULP only where |x| > 15."""
  x = np.asarray(x, dtype=np.float32)
  bumped = (x.view(np.uint32) + 1).view(np.float32)
  return np.where(np.abs(x) > 15.0, bumped, x)


class CompareBitwiseTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("float32", np.float32),
      ("bfloat16", ml_dtypes.bfloat16),
      ("int32", np.int32),
      ("bool", np.bool_),
  )
  def test_identical_arrays_are_equal(self, dtype):
    x = (np.arange(24).reshape(4, 6) % 3).astype(dtype)
    cmp = numerical_validator.compare_bitwise(x, x.copy())
    self.assertTrue(cmp.equal)
    self.assertEqual(cmp.diff_count, 0)
    self.assertIsNone(cmp.first_diff_index)

  def test_single_ulp_change_is_located(self):
    ref = np.linspace(1.0, 2.0, 48, dtype=np.float32).reshape(6, 8)
    cand = ref.copy()
    cand[3, 5] = _plus_one_ulp(cand[3, 5])
    cmp = numerical_validator.compare_bitwise(cand, ref)
    self.assertFalse(cmp.equal)
    self.assertEqual(cmp.diff_count, 1)
    self.assertAlmostEqual(cmp.diff_ratio, 1.0 / 48.0)
    self.assertEqual(cmp.first_diff_index, (3, 5))
    assert cmp.reference_bits is not None and cmp.candidate_bits is not None
    ref_bits = int(cmp.reference_bits, 16)
    self.assertEqual(int(cmp.candidate_bits, 16), ref_bits + 1)
    self.assertLen(cmp.reference_bits, 2 + 8)

  def test_signed_zero_is_a_bitwise_difference(self):
    cmp = numerical_validator.compare_bitwise(
        np.array([-0.0], dtype=np.float32), np.array([0.0], dtype=np.float32)
    )
    self.assertFalse(cmp.equal)
    self.assertEqual(cmp.reference_bits, "0x00000000")
    self.assertEqual(cmp.candidate_bits, "0x80000000")

  def test_identical_nan_payloads_are_equal(self):
    nan = np.array([np.nan, 1.0, -np.inf], dtype=np.float32)
    self.assertTrue(numerical_validator.compare_bitwise(nan, nan.copy()).equal)

  def test_dtype_mismatch_is_unequal_with_note(self):
    ref = np.ones((2, 3), dtype=np.float32)
    cmp = numerical_validator.compare_bitwise(ref.astype(np.float16), ref)
    self.assertFalse(cmp.equal)
    self.assertEqual(cmp.diff_count, 6)
    self.assertIn("dtype mismatch", cmp.note or "")

  def test_empty_arrays_are_equal(self):
    empty = np.zeros((0, 4), dtype=np.float32)
    self.assertTrue(numerical_validator.compare_bitwise(empty, empty).equal)

  def test_zero_dim_arrays(self):
    a = np.float32(1.5)
    cmp = numerical_validator.compare_bitwise(np.asarray(a), np.asarray(-a))
    self.assertFalse(cmp.equal)
    self.assertEqual(cmp.first_diff_index, ())

  def test_shape_mismatch_raises(self):
    with self.assertRaisesRegex(ValueError, "Shape mismatch"):
      numerical_validator.compare_bitwise(np.zeros(3), np.zeros(4))


class ValidateArraysContractTest(absltest.TestCase):

  def test_signed_zero_passes_ulp_but_fails_bitwise(self):
    cand = np.array([-0.0, 1.0], dtype=np.float32)
    ref = np.array([0.0, 1.0], dtype=np.float32)
    ulp = numerical_validator.validate_arrays(cand, ref, dtype_str="float32")
    self.assertTrue(ulp.passed)
    self.assertEqual(ulp.max_ulp_distance, 0)
    # A 0-ULP result is not a claim of bit identity.
    assert ulp.ulp_context is not None
    self.assertFalse(ulp.ulp_context.bit_identical)

    bitwise = numerical_validator.validate_arrays(
        cand, ref, dtype_str="float32", contract=_BITWISE
    )
    self.assertFalse(bitwise.passed)
    assert bitwise.bitwise is not None
    self.assertEqual(bitwise.bitwise.first_diff_index, (0,))

  def test_identical_nan_fails_ulp_but_passes_bitwise(self):
    x = np.array([np.nan, 2.0, np.inf], dtype=np.float32)
    ulp = numerical_validator.validate_arrays(x, x.copy(), dtype_str="float32")
    self.assertFalse(ulp.passed)
    bitwise = numerical_validator.validate_arrays(
        x, x.copy(), dtype_str="float32", contract=_BITWISE
    )
    self.assertTrue(bitwise.passed)
    assert bitwise.ulp_context is not None
    self.assertTrue(bitwise.ulp_context.bit_identical)

  def test_unknown_contract_raises(self):
    x = np.ones(3, dtype=np.float32)
    with self.assertRaisesRegex(ValueError, "Unknown contract"):
      numerical_validator.validate_arrays(x, x, contract="exact")


class ValidateKernelsContractTest(absltest.TestCase):

  def test_identical_kernel_passes_on_full_suite(self):
    shapes = (16, 16)
    report = numerical_validator.validate_kernels(
        _identity,
        _identity,
        shapes=shapes,
        dtype_str="float32",
        tier="fast_agent",
        contract=_BITWISE,
    )
    self.assertTrue(report.is_numerically_equivalent)
    self.assertIn("bit-identical", report.summary_message)
    self.assertEqual(report.run_config["contract"], _BITWISE)
    full_suite = numerical_generator.generate_test_suite(
        shapes, dtype_str="float32", tier="fast_agent"
    )
    self.assertLen(full_suite, report.total_batches_count)
    self.assertTrue(
        all(
            b.bitwise is not None and b.bitwise.equal
            for b in report.batch_results
        )
    )

  def test_one_ulp_change_passes_ulp_but_fails_bitwise(self):
    kwargs = dict(shapes=(16, 16), dtype_str="float32", tier="fast_agent")
    ulp = numerical_validator.validate_kernels(
        _identity, _plus_one_ulp, **kwargs
    )
    self.assertTrue(ulp.is_numerically_equivalent)
    self.assertEqual(ulp.run_config["contract"], _ULP)

    bitwise = numerical_validator.validate_kernels(
        _identity, _plus_one_ulp, contract=_BITWISE, **kwargs
    )
    self.assertFalse(bitwise.is_numerically_equivalent)
    self.assertIn("First difference", bitwise.summary_message)
    self.assertIn("index (0, 0)", bitwise.summary_message)

  def test_bitwise_default_covers_regimes_beyond_normal(self):
    kwargs = dict(shapes=(16, 16), dtype_str="float32", tier="presubmit")
    # The default ULP contract runs only the normal regime unless it fails.
    ulp = numerical_validator.validate_kernels(
        _identity, _plus_one_ulp_on_large_values, **kwargs
    )
    self.assertTrue(ulp.is_numerically_equivalent)
    self.assertEqual({b.regime for b in ulp.batch_results}, {"normal"})

    bitwise = numerical_validator.validate_kernels(
        _identity, _plus_one_ulp_on_large_values, contract=_BITWISE, **kwargs
    )
    self.assertFalse(bitwise.is_numerically_equivalent)
    failing = {b.regime for b in bitwise.batch_results if not b.passed}
    self.assertNotIn("normal", failing)
    self.assertNotEmpty(failing)

  def test_unknown_contract_raises(self):
    with self.assertRaisesRegex(ValueError, "Unknown contract"):
      numerical_validator.validate_kernels(
          _identity, _identity, shapes=(4, 4), contract="exact"
      )


class ToolContractTest(absltest.TestCase):

  def test_json_report_carries_contract_and_bitwise_detail(self):
    payload = json.loads(
        xparity_tool.verify_numerical_parity(
            _identity,
            _plus_one_ulp,
            shapes=(8, 8),
            dtype_str="float32",
            tier="fast_agent",
            contract="bitwise",
        )
    )
    self.assertFalse(payload["is_numerically_equivalent"])
    self.assertEqual(payload["run_config"]["contract"], "bitwise")
    first = payload["batch_results"][0]["bitwise"]
    self.assertFalse(first["equal"])
    self.assertEqual(first["first_diff_index"], [0, 0])


if __name__ == "__main__":
  absltest.main()
