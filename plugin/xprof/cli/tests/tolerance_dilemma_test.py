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

  def test_split_k_reduction_non_associativity_pass(self):
    """Proves parallel Split-K reduction reordering passes under relaxed contract."""

    def parallel_split_k_sim(a):
      # Simulates parallel reduction with intermediate block sums
      # Produces ~2 ULP non-associative jitter relative to sequential FP32 sum
      reshaped = a.reshape(a.shape[0], 16, -1)
      block_sums = jnp.sum(reshaped.astype(jnp.float32), axis=-1)
      return jnp.sum(block_sums, axis=-1).astype(a.dtype)

    shape = (32, 2048)
    report = numerical_validator.validate_kernels(
        reference_reduction,
        parallel_split_k_sim,
        shapes=shape,
        dtype_str="bfloat16",
        tier="fast_agent",
        max_allowed_ulp=4,  # Analytically justified Split-K contract
    )
    self.assertTrue(report.is_numerically_equivalent)
    self.assertLessEqual(report.overall_max_ulp, 4)
    self.assertIsNotNone(report.tolerance_audit)
    assert report.tolerance_audit is not None
    self.assertTrue(report.tolerance_audit.is_relaxed_override)

  def test_flashattention_accumulator_downcast_detected(self):
    """Proves accumulator downcast is caught when scalar allclose falsely passes."""

    def fp32_acc_attention(q, k):
      # Golden reference with FP32 accumulator
      scores = jnp.matmul(q.astype(jnp.float32), k.astype(jnp.float32).T)
      return jnp.sum(scores, axis=-1).astype(q.dtype)

    def bf16_acc_attention(q, k):
      # Buggy candidate accumulating in BF16
      scores = jnp.matmul(q, k.T)
      return jnp.sum(scores, axis=-1)

    shapes = [(16, 128), (16, 128)]
    report = numerical_validator.validate_kernels(
        fp32_acc_attention,
        bf16_acc_attention,
        shapes=shapes,
        dtype_str="bfloat16",
        tier="fast_agent",
        max_allowed_ulp=2,  # Standard recommended contract
    )
    # The accumulator truncation error is caught by ULP validator
    self.assertFalse(report.is_numerically_equivalent)
    self.assertGreater(report.overall_max_ulp, 2)

  def test_moe_token_routing_off_by_one_boundary_catch(self):
    """Proves MoE boundary off-by-one wrap (expert 63 -> 0) is caught with Delta > 0."""

    def ref_dispatch(expert_table, expert_ids):
      return expert_table[expert_ids]

    def buggy_dispatch(expert_table, expert_ids):
      buggy_ids = np.where(expert_ids == 63, 0, expert_ids)
      return expert_table[buggy_ids]

    expert_table = np.arange(64 * 32, dtype=np.float32).reshape(64, 32)
    expert_ids = numerical_generator.generate_index_tensor(
        shape=(512,),
        upper_bound=64,
        include_boundaries=True,
        dtype_str="int32",
        seed=42,
    )

    custom_suite = [{
        "name": "moe_batch",
        "args": (expert_table, expert_ids),
        "kwargs": {},
        "regime": "discrete_index",
    }]
    report = numerical_validator.validate_kernels(
        ref_dispatch,
        buggy_dispatch,
        shapes=[(64, 32), (512,)],
        dtype_str="float32",
        test_suite=custom_suite,
        max_allowed_ulp=0,
    )
    self.assertFalse(report.is_numerically_equivalent)
    self.assertGreater(report.overall_max_ulp, 0)


if __name__ == "__main__":
  absltest.main()

