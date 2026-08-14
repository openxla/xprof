"""Parameterized tests proving Traditional Gaussian failure vs Heavy-Tailed success."""

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from absl.testing import parameterized
from xprof.cli.internal import numerical_generator
from xprof.cli.internal import numerical_validator


def reference_softmax(x: jax.Array) -> jax.Array:
  """Numerically stable softmax reference."""
  x_max = jnp.max(x, axis=-1, keepdims=True)
  exp_x = jnp.exp(x - x_max)
  return exp_x / jnp.sum(exp_x, axis=-1, keepdims=True)


def buggy_softmax(x: jax.Array) -> jax.Array:
  """Buggy softmax omitting x - max(x) subtraction."""
  exp_x = jnp.exp(x)
  return exp_x / jnp.sum(exp_x, axis=-1, keepdims=True)


def reference_reduction(a: jax.Array) -> jax.Array:
  """Reference summation with high-precision float32 accumulation."""
  return jnp.sum(a.astype(jnp.float32), axis=-1).astype(a.dtype)


def buggy_bf16_reduction(a: jax.Array) -> jax.Array:
  """Buggy summation accumulating in coarse bfloat16."""
  return jnp.sum(a, axis=-1)


def reference_matmul(a: jax.Array, b: jax.Array) -> jax.Array:
  """Reference matrix multiplication."""
  return jnp.dot(a, b)


class ToleranceDilemmaTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("loose_1e_1", 1e-1, 1e-1),
      ("medium_1e_3", 1e-3, 1e-3),
      ("strict_1e_6", 1e-6, 1e-6),
  )
  def test_gaussian_falsely_passes_buggy_softmax(
      self, atol: float, rtol: float
  ):
    """Proves Gaussian inputs pass buggy softmax across all tolerances."""
    key = jax.random.PRNGKey(42)
    shape = (16, 1024)
    gaussian_input = jax.random.normal(key, shape, dtype=jnp.float32)

    y_ref = np.array(reference_softmax(gaussian_input))
    y_buggy = np.array(buggy_softmax(gaussian_input))

    self.assertTrue(
        np.allclose(y_buggy, y_ref, rtol=rtol, atol=atol),
        f"Expected Gaussian input to pass at rtol={rtol}",
    )

  def test_heavy_tailed_catches_buggy_softmax(self):
    """Proves heavy-tailed suite catches buggy softmax with Inf/NaN."""
    shape = (16, 1024)
    suite = numerical_generator.generate_test_suite(
        shape, "float32", tier="fast_agent"
    )
    # Shift one batch into the activation overflow regime (> 88.72)
    suite[0]["args"] = (suite[0]["args"][0] + 90.0,)

    report = numerical_validator.validate_kernels(
        reference_softmax,
        buggy_softmax,
        shapes=shape,
        dtype_str="float32",
        test_suite=suite,
    )

    self.assertFalse(report.is_numerically_equivalent)
    self.assertGreater(report.failed_batches_count, 0)
    self.assertGreater(report.overall_max_ulp, 1000)

  @parameterized.named_parameters(
      ("float32", "float32"),
      ("bfloat16", "bfloat16"),
      ("float16", "float16"),
  )
  def test_cancellation_caught_by_validator(self, dtype_str: str):
    """Proves cancellation generator catches low-precision reduction bugs."""
    shape = (32, 2048)
    report = numerical_validator.validate_kernels(
        reference_reduction,
        buggy_bf16_reduction,
        shapes=shape,
        dtype_str=dtype_str,
        tier="fast_agent",
    )

    if dtype_str == "bfloat16":
      self.assertFalse(report.is_numerically_equivalent)
      self.assertGreater(report.overall_max_ulp, 2)

  def test_multi_arg_matmul_validation(self):
    """Tests variadic multi-input kernel validation (a, b)."""
    shapes = [(128, 64), (64, 128)]
    report = numerical_validator.validate_kernels(
        reference_matmul,
        reference_matmul,
        shapes=shapes,
        dtype_str="bfloat16",
        tier="fast_agent",
    )
    self.assertTrue(report.is_numerically_equivalent)
    self.assertEqual(report.overall_max_ulp, 0)


if __name__ == "__main__":
  absltest.main()
