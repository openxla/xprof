"""Tests for float4_e2m1fn outputs in the ULP core and the validator."""

from absl.testing import absltest
from absl.testing import parameterized
import ml_dtypes
import numpy as np

from xprof.xparity import numerical_validator
from xprof.xparity import ulp

_FP4 = ml_dtypes.float4_e2m1fn
# Every non-negative finite float4_e2m1fn value, in order.
_FP4_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _fp4(values) -> np.ndarray:
  return np.asarray(values, dtype=np.float32).astype(_FP4)


def _quantize(x):
  return np.asarray(x, dtype=np.float32).astype(_FP4)


def _quantize_off_by_one_step(x):
  out = _quantize(x).copy()
  # _suite() pins x[0, 0] to 1.0; 1.5 is the next float4_e2m1fn value.
  out[0, 0] = _FP4(1.5)
  return out


def _suite():
  rng = np.random.default_rng(0)
  x = rng.uniform(-6.0, 6.0, size=(16, 32)).astype(np.float32)
  x[0, 0] = 1.0
  return [
      {"name": "fp4_direct_0", "args": (x,), "kwargs": {}, "regime": "normal"}
  ]


class Fp4UlpTest(absltest.TestCase):

  def test_adjacent_values_are_one_ulp_apart(self):
    values = _fp4(_FP4_VALUES)
    dist = ulp.compute_ulp_distance(
        values[1:], values[:-1], "float4_e2m1fn", near_zero_fraction=0.0
    )
    np.testing.assert_array_equal(dist, np.ones(len(_FP4_VALUES) - 1))

  def test_sign_crossing_and_signed_zero(self):
    dist = ulp.compute_ulp_distance(
        _fp4([0.5, -0.0, -6.0]),
        _fp4([-0.5, 0.0, 6.0]),
        "fp4_e2m1",
        near_zero_fraction=0.0,
    )
    np.testing.assert_array_equal(dist, [2, 0, 14])

  def test_aliases_and_tables(self):
    self.assertEqual(
        ulp.resolve_canonical_dtype(np.dtype(_FP4)), "float4_e2m1fn"
    )
    self.assertEqual(ulp.get_contract("float4_e2m1"), (0, 0))
    self.assertEqual(ulp.DTYPE_TOLERANCES["float4_e2m1fn"], (0.5, 0.25))


class Fp4ValidatorTest(parameterized.TestCase):

  @parameterized.parameters(
      (numerical_validator.CONTRACT_ULP, {}),
      (numerical_validator.CONTRACT_BITWISE, {}),
      (numerical_validator.CONTRACT_ROW_REL_L2, {"rel_l2_tol": 0.1}),
  )
  def test_validate_arrays_under_every_contract(self, contract, params):
    ref = _fp4(np.linspace(-6.0, 6.0, 64).reshape(8, 8))
    same = numerical_validator.validate_arrays(
        ref.copy(), ref, dtype_str="float4_e2m1fn", contract=contract, **params
    )
    self.assertTrue(same.passed)
    self.assertEqual(same.max_ulp_distance, 0)

    bumped = ref.copy()
    bumped[3, 3] = _FP4(-6.0) if float(ref[3, 3]) > 0 else _FP4(6.0)
    bad = numerical_validator.validate_arrays(
        bumped, ref, dtype_str="float4_e2m1fn", contract=contract, **params
    )
    self.assertFalse(bad.passed)
    self.assertGreater(bad.max_ulp_distance, 0)

  def test_one_step_fails_the_zero_ulp_contract(self):
    ref = _fp4([[1.0, 2.0], [3.0, 4.0]])
    cand = _fp4([[1.5, 2.0], [3.0, 4.0]])
    res = numerical_validator.validate_arrays(
        cand, ref, dtype_str="float4_e2m1fn"
    )
    self.assertFalse(res.passed)
    self.assertEqual(res.max_ulp_distance, 1)

  def test_signed_zero_passes_ulp_but_not_bitwise(self):
    ref = _fp4([[0.0, 1.0]])
    cand = _fp4([[-0.0, 1.0]])
    self.assertTrue(
        numerical_validator.validate_arrays(
            cand, ref, dtype_str="float4_e2m1fn"
        ).passed
    )
    self.assertFalse(
        numerical_validator.validate_arrays(
            cand,
            ref,
            dtype_str="float4_e2m1fn",
            contract=numerical_validator.CONTRACT_BITWISE,
        ).passed
    )

  @parameterized.parameters(
      (numerical_validator.CONTRACT_ULP, {}),
      (numerical_validator.CONTRACT_BITWISE, {}),
      (numerical_validator.CONTRACT_ROW_REL_L2, {"rel_l2_tol": 0.1}),
  )
  def test_validate_kernels_with_fp4_outputs(self, contract, params):
    def run(candidate):
      return numerical_validator.validate_kernels(
          _quantize,
          candidate,
          shapes=(16, 32),
          dtype_str="float4_e2m1fn",
          test_suite=_suite(),
          contract=contract,
          **params,
      )

    good = run(_quantize)
    self.assertTrue(good.is_numerically_equivalent, good.summary_message)
    bad = run(_quantize_off_by_one_step)
    if contract == numerical_validator.CONTRACT_ROW_REL_L2:
      # One step in one element of 512 is well within a 10% relative L2.
      self.assertTrue(bad.is_numerically_equivalent, bad.summary_message)
    else:
      self.assertFalse(bad.is_numerically_equivalent)
    self.assertEqual(bad.batch_results[0].max_ulp_distance, 1)


if __name__ == "__main__":
  absltest.main()
