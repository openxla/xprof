"""Unit tests for numerical_generator in xprof_cli."""

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized
from xprof.cli.internal import numerical_generator


class NumericalGeneratorTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("float32", "float32"),
      ("bfloat16", "bfloat16"),
      ("float16", "float16"),
  )
  def test_generate_student_t_shapes(self, dtype_str: str):
    shape = (8, 64)
    t = numerical_generator.generate_student_t_tensor(
        shape, dtype_str=dtype_str
    )
    self.assertEqual(t.shape, shape)
    arr = np.array(t)
    self.assertFalse(np.isnan(arr).any())
    self.assertFalse(np.isinf(arr).any())

  def test_generate_outliers(self):
    shape = (16, 128)
    t = numerical_generator.generate_outlier_tensor(
        shape, dtype_str="float32", outlier_ratio=0.05, outlier_scale=50.0
    )
    arr = np.array(t)
    self.assertEqual(arr.shape, shape)
    # Check that at least some elements have large absolute value
    self.assertGreater(np.max(np.abs(arr)), 10.0)

  def test_generate_cancellation(self):
    shape = (4, 16)
    t = numerical_generator.generate_cancellation_tensor(
        shape, dtype_str="bfloat16"
    )
    arr = np.array(t)
    self.assertEqual(arr.shape, shape)
    # Elements at index 0 and 1 along last axis should nearly cancel
    sum_pair = float(arr[0, 0] + arr[0, 1])
    self.assertGreater(abs(sum_pair), 0.0)
    self.assertLess(abs(sum_pair), 50.0)

  def test_generate_boundary_probes(self):
    shape = (16, 128)
    t = numerical_generator.generate_boundary_probe_tensor(
        shape, dtype_str="float16"
    )
    arr = np.array(t)
    self.assertEqual(arr.shape, shape)
    # Check probe values exist in array
    self.assertIn(1000.0, arr)
    self.assertIn(0.0, arr)

  @parameterized.named_parameters(
      ("fast_agent", "fast_agent", 5),
      ("presubmit", "presubmit", 11),
  )
  def test_generate_test_suite_tiers(
      self, tier: str, expected_min_batches: int
  ):
    shapes = (8, 32)
    suite = numerical_generator.generate_test_suite(
        shapes, dtype_str="bfloat16", tier=tier
    )
    self.assertGreaterEqual(len(suite), expected_min_batches)
    for batch in suite:
      self.assertIn("name", batch)
      self.assertIn("args", batch)
      self.assertIn("regime", batch)


if __name__ == "__main__":
  absltest.main()
