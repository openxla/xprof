"""Tests for the dependency-light ulp module."""

import ast
import inspect
import types

from absl.testing import absltest
from absl.testing import parameterized
import ml_dtypes
import numpy as np

from xprof.xparity import numerical_validator
from xprof.xparity import ulp

# Benchmark scripts carry ulp's source inline where only these are importable.
_ALLOWED_TOP_LEVEL_IMPORTS = frozenset(
    {"functools", "logging", "types", "typing", "ml_dtypes", "numpy"}
)


class UlpTest(parameterized.TestCase):

  def test_imports_only_stdlib_numpy_and_ml_dtypes(self):
    tree = ast.parse(inspect.getsource(ulp))
    imported = set()
    for node in ast.walk(tree):
      if isinstance(node, ast.Import):
        imported.update(alias.name.split(".")[0] for alias in node.names)
      elif isinstance(node, ast.ImportFrom):
        imported.add((node.module or "").split(".")[0])
    self.assertContainsSubset(imported, _ALLOWED_TOP_LEVEL_IMPORTS)

  def test_source_runs_as_a_standalone_module(self):
    module = types.ModuleType("inline_ulp")
    exec(compile(inspect.getsource(ulp), "<inline_ulp>", "exec"), module.__dict__)  # pylint: disable=exec-used
    one = np.array([1.0], dtype=ml_dtypes.bfloat16)
    next_up = np.array([1.0078125], dtype=ml_dtypes.bfloat16)
    self.assertEqual(
        module.compute_ulp_distance(next_up, one, "bfloat16").tolist(), [1]
    )

  def test_validator_reexports_are_the_same_objects(self):
    self.assertIs(
        numerical_validator.compute_ulp_distance, ulp.compute_ulp_distance
    )
    self.assertIs(numerical_validator.get_contract, ulp.get_contract)
    self.assertIs(
        numerical_validator.RECOMMENDED_CONTRACT_ULP,
        ulp.RECOMMENDED_CONTRACT_ULP,
    )
    self.assertIs(
        numerical_validator.MAX_HARD_CEILING_ULP, ulp.MAX_HARD_CEILING_ULP
    )

  @parameterized.named_parameters(
      ("bfloat16", ml_dtypes.bfloat16, "bfloat16"),
      ("float32", np.float32, "float32"),
      ("float16", np.float16, "float16"),
  )
  def test_adjacent_floats_are_one_ulp_apart(self, dtype, dtype_str):
    x = np.array([1.0, -3.5, 1e-3], dtype=dtype)
    up = np.nextafter(x.astype(np.float32), np.float32(np.inf)).astype(dtype)
    # bfloat16 and float16 are coarser than float32, so step in their own
    # bit space instead.
    if dtype is not np.float32:
      bits = x.view(np.uint16).astype(np.int32)
      up = np.where(x >= 0, bits + 1, bits - 1).astype(np.uint16).view(dtype)
    self.assertEqual(
        ulp.compute_ulp_distance(up, x, dtype_str).tolist(), [1, 1, 1]
    )

  def test_integer_distance_is_absolute_difference(self):
    a = np.array([3, -2, 7], dtype=np.int32)
    b = np.array([3, 1, 4], dtype=np.int32)
    self.assertEqual(
        ulp.compute_ulp_distance(a, b, "int32").tolist(), [0, 3, 3]
    )

  @parameterized.named_parameters(
      ("uint8", np.uint8, 255),
      ("uint16", np.uint16, 65535),
      ("uint32", np.uint32, 2**32 - 1),
      ("uint64", np.uint64, 2**64 - 1),
  )
  def test_unsigned_distance_does_not_overflow(self, dtype, top):
    a = np.array([0, top, 5], dtype=dtype)
    b = np.array([top, 0, 7], dtype=dtype)
    want = min(top, 2**63 - 1)
    self.assertEqual(
        ulp.compute_ulp_distance(a, b, np.dtype(dtype).name).tolist(),
        [want, want, 2],
    )

  @parameterized.named_parameters(
      ("uint64_vs_int32", np.uint64, 2**63, np.int32, 0, 2**63 - 1),
      (
          "uint64_max_vs_minus_one",
          np.uint64,
          2**64 - 1,
          np.int64,
          -1,
          2**63 - 1,
      ),
      ("above_2_pow_53", np.uint64, 2**60 + 1, np.uint64, 2**60, 1),
      ("int64_extremes", np.int64, 2**63 - 1, np.int64, -(2**63), 2**63 - 1),
      ("int64_min_vs_zero", np.int64, -(2**63), np.int64, 0, 2**63 - 1),
      ("int64_close_negatives", np.int64, -(2**62) - 3, np.int64, -(2**62), 3),
      ("uint8_vs_int8", np.uint8, 255, np.int8, -128, 383),
  )
  def test_integer_distance_is_exact(
      self, a_dtype, a_value, b_dtype, b_value, want
  ):
    a = np.array([a_value], dtype=a_dtype)
    b = np.array([b_value], dtype=b_dtype)
    self.assertEqual(
        ulp.compute_ulp_distance(a, b, np.dtype(a_dtype).name).tolist(), [want]
    )
    self.assertEqual(
        ulp.compute_ulp_distance(b, a, np.dtype(b_dtype).name).tolist(), [want]
    )

  def test_integer_candidate_against_float_reference(self):
    a = np.array([5, -3, -3, 7], dtype=np.int32)
    e = np.array([5.0, -3.0, 4.0, np.nan], dtype=np.float32)
    self.assertEqual(
        ulp.compute_ulp_distance(a, e, "int32").tolist(),
        [0, 0, 7, 2**63 - 1],
    )
    self.assertEqual(
        ulp.compute_ulp_distance(e[:3], a[:3], "int32").tolist(), [0, 0, 7]
    )

  def test_bool_distance_counts_flips(self):
    a = np.array([True, False, True])
    b = np.array([True, True, False])
    self.assertEqual(ulp.compute_ulp_distance(a, b, "bool").tolist(), [0, 1, 1])

  def test_near_zero_does_not_depend_on_sign(self):
    # float32 has 786,163,455 values between 0 and 1e-10. In a tensor whose
    # largest value is 3, all three errors are far below eps * 3.
    expected = np.array([3.0, 0.0, 0.0, 1e-10], dtype=np.float32)
    actual = np.array([3.0, -1e-10, 1e-10, 2e-10], dtype=np.float32)
    self.assertEqual(
        ulp.compute_ulp_distance(actual, expected, "float32").tolist(),
        [0, 1, 1, 1],
    )

  def test_near_zero_is_relative_to_the_reference_scale(self):
    # Alone, 1e-10 vs 2e-10 is the whole tensor, so neither is near zero.
    expected = np.array([1e-10], dtype=np.float32)
    actual = np.array([2e-10], dtype=np.float32)
    raw = int(
        actual.view(np.int32).astype(np.int64)[0]
        - expected.view(np.int32).astype(np.int64)[0]
    )
    self.assertEqual(
        ulp.compute_ulp_distance(actual, expected, "float32").tolist(), [raw]
    )
    self.assertEqual(
        ulp.compute_ulp_distance(
            actual, expected, "float32", scale=3.0
        ).tolist(),
        [1],
    )

  def test_near_zero_error_above_contract_is_still_counted(self):
    eps = float(np.finfo(np.float32).eps)
    expected = np.array([2.0, 0.0], dtype=np.float32)
    actual = np.array([2.0, 16 * eps * 2.0], dtype=np.float32)
    self.assertEqual(
        ulp.compute_ulp_distance(actual, expected, "float32").tolist(),
        [0, 16],
    )

  def test_elements_above_the_floor_keep_their_raw_distance(self):
    # A 17% error on a bfloat16 probability of 0.01 in a tensor whose largest
    # value is 1. 0.01 is above 2**-8, so the error keeps its raw distance.
    expected = np.array([1.0, 0.01], dtype=ml_dtypes.bfloat16)
    actual = np.array([1.0, 0.0117], dtype=ml_dtypes.bfloat16)
    raw = int(
        actual.view(np.uint16).astype(np.int64)[1]
        - expected.view(np.uint16).astype(np.int64)[1]
    )
    self.assertGreater(raw, ulp.get_contract("bfloat16")[1])
    self.assertEqual(
        ulp.compute_ulp_distance(actual, expected, "bfloat16").tolist(),
        [0, raw],
    )

  def test_near_zero_scaling_can_be_disabled_and_needs_a_scale(self):
    expected = np.array([3.0, 0.0], dtype=np.float32)
    actual = np.array([3.0, 1e-10], dtype=np.float32)
    raw = int(actual.view(np.int32)[1])
    self.assertEqual(
        ulp.compute_ulp_distance(
            actual, expected, "float32", near_zero_fraction=0.0
        ).tolist(),
        [0, raw],
    )
    zeros = np.zeros(2, dtype=np.float32)
    self.assertEqual(ulp.reference_scale(zeros), 0.0)
    self.assertEqual(
        ulp.compute_ulp_distance(
            np.array([0.0, 1e-10], dtype=np.float32), zeros, "float32"
        ).tolist(),
        [0, raw],
    )

  def test_reference_scale_ignores_non_finite_values(self):
    self.assertEqual(
        ulp.reference_scale(np.array([np.nan, -np.inf, -4.0, 2.0])), 4.0
    )
    self.assertEqual(ulp.reference_scale(np.zeros(0, np.float32)), 0.0)

  @parameterized.named_parameters(
      ("bf16_alias", "bf16", (2, 8)),
      ("float32", "float32", (2, 4)),
      ("int32", "int32", (0, 0)),
      ("ml_dtypes_qualified", "ml_dtypes.float8_e4m3fn", (1, 2)),
      ("dtype_object", np.dtype(np.float32), (2, 4)),
  )
  def test_get_contract(self, dtype, expected):
    self.assertEqual(ulp.get_contract(dtype), expected)

  def test_is_discrete_dtype(self):
    self.assertTrue(ulp.is_discrete_dtype("uint8"))
    self.assertTrue(ulp.is_discrete_dtype("bool"))
    self.assertFalse(ulp.is_discrete_dtype("bf16"))

  @parameterized.named_parameters(
      ("float32", np.float32),
      ("bfloat16", ml_dtypes.bfloat16),
      ("float16", np.float16),
  )
  def test_bitwise_mask_separates_signed_zero_that_ulp_merges(self, dtype):
    actual = np.array([0.0, -0.0, 1.0], dtype=dtype)
    expected = np.array([-0.0, -0.0, 1.0], dtype=dtype)
    self.assertEqual(
        ulp.compute_ulp_distance(
            actual, expected, np.dtype(dtype).name
        ).tolist(),
        [0, 0, 0],
    )
    self.assertEqual(
        ulp.bitwise_mismatch_mask(actual, expected).tolist(),
        [True, False, False],
    )

  def test_bitwise_mask_compares_nan_payloads(self):
    quiet = np.array([0x7FC00000], dtype=np.uint32).view(np.float32)
    payload = np.array([0x7FC00001], dtype=np.uint32).view(np.float32)
    self.assertEqual(
        ulp.bitwise_mismatch_mask(quiet, quiet.copy()).tolist(), [False]
    )
    self.assertEqual(ulp.bitwise_mismatch_mask(quiet, payload).tolist(), [True])

  def test_bitwise_mask_keeps_shape_and_handles_empty(self):
    a = np.zeros((2, 3), dtype=np.int8)
    b = a.copy()
    b[1, 2] = 1
    mask = ulp.bitwise_mismatch_mask(a, b)
    self.assertEqual(mask.shape, (2, 3))
    self.assertEqual(int(mask.sum()), 1)
    self.assertTrue(mask[1, 2])
    empty = np.zeros((0, 4), dtype=np.float32)
    self.assertEqual(ulp.bitwise_mismatch_mask(empty, empty).shape, (0, 4))

  def test_bitwise_mask_rejects_dtype_or_shape_mismatch(self):
    with self.assertRaisesRegex(ValueError, "equal shapes and dtypes"):
      ulp.bitwise_mismatch_mask(
          np.zeros(2, np.float32), np.zeros(2, np.float16)
      )
    with self.assertRaisesRegex(ValueError, "equal shapes and dtypes"):
      ulp.bitwise_mismatch_mask(np.zeros(2), np.zeros(3))

  def test_validator_compare_bitwise_agrees_with_mask(self):
    actual = np.array([1.0, -0.0, 2.0], dtype=np.float32)
    expected = np.array([1.0, 0.0, 2.0], dtype=np.float32)
    result = numerical_validator.compare_bitwise(actual, expected)
    self.assertEqual(
        result.diff_count,
        int(ulp.bitwise_mismatch_mask(actual, expected).sum()),
    )
    self.assertEqual(result.first_diff_index, (1,))
    self.assertEqual(result.candidate_bits, "0x80000000")
    self.assertEqual(result.reference_bits, "0x00000000")

  @parameterized.named_parameters(
      ("equal", [1.0, 2.0], [1.0, 2.0], 0.0),
      # |1.1 - 1.0| / (0.0 + 0.1 * 1.0) == 1.0, the allclose boundary.
      ("at_boundary", [1.1], [1.0], 1.0),
      ("half_headroom", [1.05], [1.0], 0.5),
      ("matching_nan_and_inf", [np.nan, np.inf], [np.nan, np.inf], 0.0),
      ("nan_vs_finite", [np.nan], [1.0], np.inf),
  )
  def test_tolerance_ratio(self, candidate, reference, expected):
    ratio = ulp.tolerance_ratio(
        np.array(candidate), np.array(reference), atol=0.0, rtol=0.1
    )
    self.assertAlmostEqual(ratio, expected, places=6)

  def test_tolerance_ratio_zero_bound_and_empty(self):
    self.assertEqual(
        ulp.tolerance_ratio(np.array([1e-9]), np.array([0.0]), 0.0, 0.0),
        np.inf,
    )
    self.assertEqual(ulp.tolerance_ratio(np.zeros(0), np.zeros(0), 0.0, 0.1), 0)

  def test_validator_tolerance_ratio_is_the_ulp_function(self):
    self.assertIs(numerical_validator._tolerance_ratio, ulp.tolerance_ratio)


if __name__ == "__main__":
  absltest.main()
