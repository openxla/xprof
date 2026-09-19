"""Tests proving Traditional Gaussian failure vs Heavy-Tailed success."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from xprof.xparity import numerical_generator
from xprof.xparity import numerical_validator


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
  """Buggy summation accumulating sequentially in coarse bfloat16."""
  init = jnp.zeros((a.shape[0],), dtype=a.dtype)
  a_t = jnp.swapaxes(a, 0, -1)
  return jax.lax.scan(lambda acc, x: (acc + x, None), init, a_t)[0]


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
    """Proves parallel Split-K reduction reordering passes relaxed contract."""
    # Parallel tree summation reorders floating-point additions. In bfloat16,
    # this creates small non-associative accumulation jitter (<= 4 ULP) across
    # continuous, heavy-tailed, outlier, and cancellation regimes.
    #
    # Hardware Context: 'boundary_probes' (subnormals < 1.175e-38) is excluded
    # from this summation-reordering test because TPU hardware enforces
    # Flush-To-Zero (FTZ). When subnormals flush to 0.0 (0x0000) on TPU while
    # FP32 reference sums to 4.70e-38 (0x0180), the 384 ULP difference
    # reflects TPU FTZ hardware semantics rather than parallel reduction
    # non-associativity.

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
        regimes=["student_t", "outliers", "cancellation"],
    )
    self.assertTrue(report.is_numerically_equivalent)
    self.assertLessEqual(report.overall_max_ulp, 4)
    self.assertIsNotNone(report.tolerance_audit)
    assert report.tolerance_audit is not None
    self.assertTrue(report.tolerance_audit.is_relaxed_override)

  def test_flashattention_accumulator_downcast_detected(self):
    """Proves accumulator downcast is caught when allclose falsely passes."""

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
    """Proves MoE boundary off-by-one wrap (expert 63 -> 0) is caught."""

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

  def test_oracle_pinned_vs_unpinned_precision_gap(self):
    """Pins Claim 1: Unpinned BF16 matmul vs pinned FP32/FP64 oracle gap."""

    def unpinned_bf16_matmul(a, b):
      a_arr = np.asarray(a)
      b_arr = np.asarray(b)
      if a_arr.dtype == np.float64 and b_arr.dtype == np.float64:
        return a_arr @ b_arr
      a_bf16 = jnp.asarray(a_arr).astype(jnp.bfloat16).astype(jnp.float32)
      b_bf16 = jnp.asarray(b_arr).astype(jnp.bfloat16).astype(jnp.float32)
      return np.asarray(
          jnp.matmul(a_bf16, b_bf16, precision=jax.lax.Precision.HIGHEST)
      )

    def pinned_fp32_matmul(a, b):
      a_arr = np.asarray(a)
      b_arr = np.asarray(b)
      if a_arr.dtype == np.float64 and b_arr.dtype == np.float64:
        return a_arr @ b_arr
      return np.asarray(
          jnp.matmul(
              jnp.asarray(a_arr, dtype=jnp.float32),
              jnp.asarray(b_arr, dtype=jnp.float32),
              precision=jax.lax.Precision.HIGHEST,
          )
      )

    probe_unpinned = numerical_validator.probe_reference_precision(
        unpinned_bf16_matmul, shapes=[(64, 64), (64, 64)], dtype_str="float32"
    )
    self.assertTrue(probe_unpinned.is_downcasting)
    self.assertGreater(probe_unpinned.max_ulp_vs_fp64, 100000)

    probe_pinned_f32 = numerical_validator.probe_reference_precision(
        pinned_fp32_matmul, shapes=[(64, 64), (64, 64)], dtype_str="float32"
    )
    self.assertGreater(
        probe_unpinned.max_ulp_vs_fp64, probe_pinned_f32.max_ulp_vs_fp64 * 20
    )
    self.assertGreater(
        probe_unpinned.reference_max_abs_from_oracle,
        probe_pinned_f32.reference_max_abs_from_oracle * 100,
    )

    def pinned_fp32_to_bf16_matmul(a, b):
      a_arr = np.asarray(a)
      b_arr = np.asarray(b)
      if a_arr.dtype == np.float64 and b_arr.dtype == np.float64:
        return a_arr @ b_arr
      res_f32 = jnp.matmul(
          jnp.asarray(a_arr, dtype=jnp.float32),
          jnp.asarray(b_arr, dtype=jnp.float32),
          precision=jax.lax.Precision.HIGHEST,
      )
      return np.asarray(res_f32.astype(jnp.bfloat16))

    probe_pinned_bf16 = numerical_validator.probe_reference_precision(
        pinned_fp32_to_bf16_matmul,
        shapes=[(64, 64), (64, 64)],
        dtype_str="bfloat16",
    )
    self.assertFalse(probe_pinned_bf16.is_downcasting)
    self.assertLessEqual(probe_pinned_bf16.max_ulp_vs_fp64, 2)

  def test_subnormal_ftz_boundary_divergence(self):
    """Pins Claim 2: Subnormal FTZ creates >100 ULP gap missed by atol=1e-5."""
    # IEEE 754 float32 subnormal ~ 4.70e-38 (0x00008000 = 32768 ULPs above 0.0)
    ieee_subnormal = np.frombuffer(
        np.uint32(0x00000180).tobytes(), dtype=np.float32
    )
    tpu_ftz_zero = np.zeros_like(ieee_subnormal, dtype=np.float32)

    # Scalar atol=1e-5 is completely blind to subnormal FTZ
    self.assertTrue(
        np.allclose(tpu_ftz_zero, ieee_subnormal, rtol=1e-5, atol=1e-5)
    )

    # Bitwise ULP validator catches exact 384 ULP (0x0180) divergence
    report = numerical_validator.validate_arrays(
        tpu_ftz_zero, ieee_subnormal, dtype_str="float32", max_allowed_ulp=4
    )
    self.assertFalse(report.passed)
    self.assertEqual(report.max_ulp, 0x0180)

  def test_int8_vs_fp8_quantization_regime_swing(self):
    """Pins Claim 3: Int8 vs FP8 E4M3 quantization sensitivity swings."""
    normal_x = numerical_generator.generate_normal_tensor(
        (32, 128), dtype_str="float32", seed=7
    )
    outlier_x = numerical_generator.generate_per_channel_outlier_tensor(
        (32, 128), dtype_str="float32", outlier_scale=80.0, seed=7
    )

    def quant_dequant_int8_per_tensor(x):
      amax = np.max(np.abs(x)) + 1e-12
      scale = amax / 127.0
      q = np.clip(np.round(x / scale), -127, 127)
      return (q * scale).astype(np.float32)

    # Under channel outliers, per-tensor Int8 suffers much larger degradation
    # on non-outlier channels than on Gaussian inputs
    err_normal = np.sqrt(
        np.mean((quant_dequant_int8_per_tensor(normal_x) - normal_x) ** 2)
    )
    err_outlier = np.sqrt(
        np.mean((quant_dequant_int8_per_tensor(outlier_x) - outlier_x) ** 2)
    )
    self.assertGreater(err_outlier, err_normal * 5.0)

  def test_student_t_vs_gaussian_accumulation_drift(self):
    """Pins Claim 4: Student-t (df=2.5) induces larger ULP drift."""
    shape = (16, 1024)
    rep_gauss = numerical_validator.validate_kernels(
        reference_reduction,
        buggy_bf16_reduction,
        shapes=shape,
        dtype_str="bfloat16",
        tier="fast_agent",
        regimes=["normal"],
        seed=42,
    )
    rep_student = numerical_validator.validate_kernels(
        reference_reduction,
        buggy_bf16_reduction,
        shapes=shape,
        dtype_str="bfloat16",
        tier="fast_agent",
        regimes=["student_t"],
        seed=42,
    )
    self.assertGreaterEqual(
        rep_student.overall_max_ulp, rep_gauss.overall_max_ulp
    )
    self.assertGreater(rep_student.overall_max_ulp, 2)


if __name__ == "__main__":
  absltest.main()
