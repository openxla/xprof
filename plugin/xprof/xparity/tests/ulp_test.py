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

  def test_bool_distance_counts_flips(self):
    a = np.array([True, False, True])
    b = np.array([True, True, False])
    self.assertEqual(ulp.compute_ulp_distance(a, b, "bool").tolist(), [0, 1, 1])

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
        ulp.compute_ulp_distance(actual, expected, np.dtype(dtype).name)
        .tolist(),
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


if __name__ == "__main__":
  absltest.main()
