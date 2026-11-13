"""Tests for the whole-output and per-row relative L2 check."""

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import ml_dtypes
import numpy as np

from xprof.xparity import rel_l2


def _ref(shape=(8, 64), seed=0):
  return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)


class CompareRowRelL2Test(parameterized.TestCase):

  def test_identical_outputs_pass(self):
    ref = _ref()
    res = rel_l2.compare_row_rel_l2(ref, ref.copy(), tol=1e-3)
    self.assertTrue(res.passed)
    self.assertIsNone(res.reason)
    self.assertEqual(res.rel_l2, 0.0)
    self.assertEqual(res.worst_row_rel_l2, 0.0)
    self.assertEqual(res.row_count, 8)
    self.assertTrue(res.message.startswith("PASSED"))

  def test_uniform_small_noise_passes_and_matches_definition(self):
    ref = _ref()
    noise = 1e-4 * _ref(seed=1)
    cand = ref + noise
    res = rel_l2.compare_row_rel_l2(cand, ref, tol=1e-3)
    expected = np.linalg.norm(noise.astype(np.float64)) / np.linalg.norm(
        ref.astype(np.float64)
    )
    self.assertTrue(res.passed)
    self.assertAlmostEqual(res.rel_l2, expected, delta=1e-6)

  def test_whole_output_over_tol_fails(self):
    ref = _ref()
    res = rel_l2.compare_row_rel_l2(ref * 1.01, ref, tol=1e-3)
    self.assertFalse(res.passed)
    self.assertEqual(res.reason, rel_l2.REASON_WHOLE_OUTPUT)
    self.assertAlmostEqual(res.rel_l2, 0.01, delta=1e-6)

  def test_one_bad_row_fails_although_whole_output_passes(self):
    ref = _ref(shape=(4, 256, 64))
    cand = ref.copy()
    # 3% error on one of 1024 rows: ~1e-3 over the whole output.
    cand[2, 17] *= 1.03
    res = rel_l2.compare_row_rel_l2(cand, ref, tol=2e-3)
    self.assertLessEqual(res.rel_l2, 2e-3)
    self.assertFalse(res.passed)
    self.assertEqual(res.reason, rel_l2.REASON_ROW)
    self.assertEqual(res.worst_row_index, (2, 17))
    self.assertAlmostEqual(res.worst_row_rel_l2, 0.03, delta=1e-5)
    self.assertEqual(res.rows_over_bound, 1)
    self.assertEqual(res.row_count, 1024)
    self.assertIn("(2, 17)", res.message)

  def test_row_factor_widens_the_row_bound(self):
    ref = _ref(shape=(4, 256, 64))
    cand = ref.copy()
    cand[2, 17] *= 1.03
    res = rel_l2.compare_row_rel_l2(cand, ref, tol=2e-3, row_factor=20.0)
    self.assertTrue(res.passed)

  def test_near_zero_row_is_measured_against_the_floor(self):
    ref = _ref()
    ref[3] *= 1e-6  # A legitimately near-zero row.
    cand = ref.copy()
    cand[3] += 1e-6  # Large relative to the row, tiny relative to the output.
    res = rel_l2.compare_row_rel_l2(cand, ref, tol=1e-3)
    self.assertTrue(res.passed, res.message)
    self.assertEqual(res.worst_row_index, (3,))
    # Without the floor the same row is off by about 100%.
    unfloored = rel_l2.compare_row_rel_l2(
        cand, ref, tol=1e-3, row_norm_floor=0.0
    )
    self.assertFalse(unfloored.passed)
    self.assertEqual(unfloored.reason, rel_l2.REASON_ROW)
    self.assertGreater(unfloored.worst_row_rel_l2, 0.5)

  def test_all_zero_reference(self):
    ref = np.zeros((4, 8), np.float32)
    self.assertTrue(rel_l2.compare_row_rel_l2(ref, ref, tol=1e-3).passed)
    res = rel_l2.compare_row_rel_l2(ref + 1e-3, ref, tol=1e-3)
    self.assertFalse(res.passed)
    self.assertEqual(res.reason, rel_l2.REASON_WHOLE_OUTPUT)

  @parameterized.named_parameters(
      ("nan", np.nan), ("pos_inf", np.inf), ("neg_inf", -np.inf)
  )
  def test_non_finite_candidate_fails(self, bad):
    ref = _ref()
    cand = ref.copy()
    cand[5, 7] = bad
    res = rel_l2.compare_row_rel_l2(cand, ref, tol=1.0)
    self.assertFalse(res.passed)
    self.assertEqual(res.reason, rel_l2.REASON_NON_FINITE_CANDIDATE)
    self.assertEqual(res.first_non_finite_index, (5, 7))

  def test_non_finite_reference_fails(self):
    ref = _ref()
    cand = ref.copy()
    ref[0, 0] = np.nan
    res = rel_l2.compare_row_rel_l2(cand, ref, tol=1.0)
    self.assertFalse(res.passed)
    self.assertEqual(res.reason, rel_l2.REASON_NON_FINITE_REFERENCE)

  def test_shape_mismatch_is_a_failure_not_an_exception(self):
    ref = _ref()
    res = rel_l2.compare_row_rel_l2(ref[:, :-1], ref, tol=1e-3)
    self.assertFalse(res.passed)
    self.assertEqual(res.reason, rel_l2.REASON_SHAPE_MISMATCH)
    self.assertIn("(8, 63)", res.message)

  def test_dtype_mismatch_fails_unless_allowed(self):
    ref = _ref()
    cand = ref.astype(ml_dtypes.bfloat16)
    res = rel_l2.compare_row_rel_l2(cand, ref, tol=1e-2)
    self.assertFalse(res.passed)
    self.assertEqual(res.reason, rel_l2.REASON_DTYPE_MISMATCH)
    self.assertTrue(
        rel_l2.compare_row_rel_l2(
            cand, ref, tol=1e-2, require_same_dtype=False
        ).passed
    )

  @parameterized.named_parameters(
      ("bfloat16", ml_dtypes.bfloat16, 1e-2),
      ("float16", np.float16, 1e-3),
      ("float8_e4m3fn", ml_dtypes.float8_e4m3fn, 8e-2),
      ("float4_e2m1fn", ml_dtypes.float4_e2m1fn, 0.5),
  )
  def test_narrow_dtypes(self, dtype, tol):
    ref32 = _ref()
    ref = ref32.astype(dtype)
    # One rounding step on every element of a single row.
    cand = ref.copy()
    cand[1] = (ref32[1] * 1.001).astype(dtype)
    res = rel_l2.compare_row_rel_l2(cand, ref, tol=tol)
    self.assertTrue(res.passed, res.message)
    self.assertEqual(res.candidate_dtype, np.dtype(dtype).name)

  def test_bfloat16_row_defect_is_caught(self):
    ref = _ref(shape=(256, 128)).astype(ml_dtypes.bfloat16)
    cand = ref.copy()
    cand[9] = (ref[9].astype(np.float32) * 1.2).astype(ml_dtypes.bfloat16)
    res = rel_l2.compare_row_rel_l2(cand, ref, tol=2e-2)
    # ~20% on one of 256 rows is ~1.25% over the whole output.
    self.assertLess(res.rel_l2, 2e-2)
    self.assertFalse(res.passed)
    self.assertEqual(res.reason, rel_l2.REASON_ROW)
    self.assertEqual(res.worst_row_index, (9,))

  def test_jax_arrays(self):
    ref = _ref()
    cand = ref.copy()
    cand[6] *= 1.2
    res = rel_l2.compare_row_rel_l2(
        jnp.asarray(cand, jnp.bfloat16), jnp.asarray(ref, jnp.bfloat16), 2e-2
    )
    self.assertFalse(res.passed)
    self.assertEqual(res.worst_row_index, (6,))
    self.assertEqual(res.candidate_dtype, "bfloat16")

  @parameterized.named_parameters(
      ("scalar", ()), ("vector", (64,)), ("rank4", (2, 3, 4, 16))
  )
  def test_ranks(self, shape):
    ref = np.asarray(_ref(shape=shape) + 3.0)
    res = rel_l2.compare_row_rel_l2(ref * (1 + 1e-5), ref, tol=1e-3)
    self.assertTrue(res.passed, res.message)
    self.assertLen(res.worst_row_index, max(len(shape) - 1, 0))
    self.assertEqual(res.row_count, int(np.prod(shape[:-1])) if shape else 1)

  def test_empty_output_passes(self):
    ref = np.zeros((0, 16), np.float32)
    self.assertTrue(rel_l2.compare_row_rel_l2(ref, ref, tol=1e-3).passed)

  @parameterized.named_parameters(
      ("zero_tol", dict(tol=0.0)),
      ("nan_tol", dict(tol=float("nan"))),
      ("negative_row_factor", dict(tol=1e-3, row_factor=-1.0)),
      ("negative_floor", dict(tol=1e-3, row_norm_floor=-0.1)),
  )
  def test_invalid_parameters_raise(self, kwargs):
    ref = _ref()
    with self.assertRaises(ValueError):
      rel_l2.compare_row_rel_l2(ref, ref, **kwargs)


if __name__ == "__main__":
  absltest.main()
