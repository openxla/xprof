"""Tests for numerical_generator serialization, loading, and replay."""

import importlib.util
import io
import os
import pathlib
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from xprof.cli.internal import numerical_generator


class NumericalGeneratorTest(parameterized.TestCase):

  def test_generate_procedural_default(self):
    """Verifies default generation produces valid in-memory suite without I/O."""
    suite = numerical_generator.generate_test_suite(
        shapes=[(128, 1, 8192), (8192,)],
        dtype_str="bfloat16",
        tier="fast_agent",
        seed=42,
    )
    self.assertNotEmpty(suite)
    for item in suite:
      self.assertIn("name", item)
      self.assertIn("args", item)
      self.assertIn("kwargs", item)
      self.assertIn("regime", item)
      self.assertLen(item["args"], 2)
      self.assertEqual(item["args"][0].shape, (128, 1, 8192))
      self.assertEqual(item["args"][1].shape, (8192,))

  @parameterized.parameters(
      "float32",
      "bfloat16",
      "float16",
      "fp8_e4m3",
      "fp8_e5m2",
  )
  def test_roundtrip_save_and_load_lossless(self, dtype_str: str):
    """Verifies exact bitwise parity across all supported data types."""
    original_suite = numerical_generator.generate_test_suite(
        shapes=[(16, 64), (64,)],
        dtype_str=dtype_str,
        tier="fast_agent",
        seed=123,
    )
    # Add sample kwargs
    original_suite[0]["kwargs"] = {
        "mask": np.ones((16, 64), dtype=np.bool_),
        "scalar_val": 42,
    }

    temp_file = self.create_tempfile(f"test_suite_{dtype_str}.npz").full_path

    # Save to file path
    numerical_generator.save_test_suite(original_suite, temp_file)

    # Load from file path
    reloaded_suite = numerical_generator.load_test_suite(
        temp_file, as_jax_arrays=False
    )

    self.assertLen(reloaded_suite, len(original_suite))
    for orig, loaded in zip(original_suite, reloaded_suite):
      self.assertEqual(orig["name"], loaded["name"])
      self.assertEqual(orig["regime"], loaded["regime"])
      self.assertLen(orig["args"], len(loaded["args"]))
      for o_arg, l_arg in zip(orig["args"], loaded["args"]):
        self.assertEqual(o_arg.dtype, l_arg.dtype)
        self.assertEqual(o_arg.shape, l_arg.shape)
        np.testing.assert_array_equal(
            np.asarray(o_arg).view(np.uint8),
            np.asarray(l_arg).view(np.uint8),
            err_msg=f"Bitwise mismatch in {orig['name']}",
        )
      if "mask" in orig["kwargs"]:
        np.testing.assert_array_equal(
            np.asarray(orig["kwargs"]["mask"]),
            np.asarray(loaded["kwargs"]["mask"]),
        )
      if "scalar_val" in orig["kwargs"]:
        self.assertEqual(
            orig["kwargs"]["scalar_val"], loaded["kwargs"]["scalar_val"]
        )

  def test_load_from_bytesio_and_bytes(self):
    """Verifies loading from in-memory stream and raw bytes buffer."""
    suite = numerical_generator.generate_test_suite(
        shapes=[(32, 32)],
        dtype_str="bfloat16",
        tier="fast_agent",
        seed=99,
    )
    bio = io.BytesIO()
    numerical_generator.save_test_suite(suite, bio)
    bio.seek(0)
    raw_bytes = bio.getvalue()

    # Load from BytesIO
    loaded_from_bio = numerical_generator.load_test_suite(
        io.BytesIO(raw_bytes), as_jax_arrays=False
    )
    self.assertLen(loaded_from_bio, len(suite))

    # Load from raw bytes
    loaded_from_bytes = numerical_generator.load_test_suite(
        raw_bytes, as_jax_arrays=False
    )
    self.assertLen(loaded_from_bytes, len(suite))
    np.testing.assert_array_equal(
        np.asarray(suite[0]["args"][0]).view(np.uint8),
        np.asarray(loaded_from_bytes[0]["args"][0]).view(np.uint8),
    )

  def test_load_from_pathlib_path(self):
    """Verifies loading using pathlib.Path object (os.PathLike)."""
    suite = numerical_generator.generate_test_suite(
        shapes=[(8, 8)],
        dtype_str="float32",
        tier="fast_agent",
        seed=1,
    )
    temp_path = pathlib.Path(self.create_tempfile("pathlib_test.npz").full_path)
    numerical_generator.save_test_suite(suite, temp_path)

    loaded = numerical_generator.load_test_suite(temp_path, as_jax_arrays=False)
    self.assertLen(loaded, len(suite))

  def test_generate_test_suite_opt_in_persistence(self):
    """Verifies generate_test_suite can save and reload via persisted_path."""
    temp_file = self.create_tempfile("auto_persist.npz").full_path
    if os.path.exists(temp_file):
      os.remove(temp_file)

    # 1. First call creates the file in mode='auto'
    suite1 = numerical_generator.generate_test_suite(
        shapes=[(16, 16)],
        dtype_str="bfloat16",
        tier="fast_agent",
        seed=77,
        persisted_path=temp_file,
        mode="auto",
    )
    self.assertTrue(os.path.exists(temp_file))

    # 2. Second call reloads from file
    suite2 = numerical_generator.generate_test_suite(
        shapes=[(16, 16)],
        dtype_str="bfloat16",
        tier="fast_agent",
        seed=9999,  # Different seed, should still load identical saved tensors
        persisted_path=temp_file,
        mode="read_only",
    )
    np.testing.assert_array_equal(
        np.asarray(suite1[0]["args"][0]).view(np.uint8),
        np.asarray(suite2[0]["args"][0]).view(np.uint8),
    )

  def test_read_only_missing_raises(self):
    """Verifies mode='read_only' raises if fixture is missing."""
    non_existent = "/tmp/non_existent_fixture_12345.npz"
    with self.assertRaises(RuntimeError):
      numerical_generator.generate_test_suite(
          shapes=[(8, 8)],
          persisted_path=non_existent,
          mode="read_only",
      )

  def test_as_jax_arrays(self):
    """Verifies as_jax_arrays=True returns JAX arrays."""
    if importlib.util.find_spec("jax") is None:
      self.skipTest("JAX is not installed in the test environment.")

    suite = numerical_generator.generate_test_suite(
        shapes=[(8, 8)],
        dtype_str="bfloat16",
        tier="fast_agent",
        seed=1,
        as_jax_arrays=True,
    )
    self.assertNotEmpty(suite)
    arg0 = suite[0]["args"][0]
    self.assertTrue(hasattr(arg0, "device"))

  def test_save_and_load_non_contiguous_arrays(self):
    """Verifies that non-contiguous and transposed array slices persist safely."""
    temp_file = self.create_tempfile("non_contiguous.npz").full_path
    base = np.arange(16, dtype=np.float32).reshape(4, 4)
    sliced = base[::2, ::2]  # Non-contiguous strided slice
    transposed = base.T  # Transposed Fortran-order array

    suite = [{
        "name": "strided",
        "args": (sliced,),
        "kwargs": {"t": transposed},
        "regime": "custom",
    }]
    numerical_generator.save_test_suite(suite, temp_file)
    loaded = numerical_generator.load_test_suite(temp_file)
    np.testing.assert_array_equal(loaded[0]["args"][0], sliced)
    np.testing.assert_array_equal(loaded[0]["kwargs"]["t"], transposed)

  # ===========================================================================
  # Statistical Distribution Validation Tests
  # ===========================================================================

  @parameterized.named_parameters(
      ("df_5", 5.0, 0.04),
      ("df_6", 6.0, 0.03),
      ("df_8", 8.0, 0.03),
      ("df_10", 10.0, 0.025),
  )
  def test_student_t_variance_scaling_finite_moment(
      self, df: float, rtol: float
  ):
    """Confirms sample variance matches theoretical ν / (ν - 2) for ν >= 5.

    Mathematical Decision & Derivation:
    -----------------------------------
    For Student's t-distribution with degrees of freedom ν > 4, the 4th central
    moment μ4 = 3ν² / ((ν - 2)(ν - 4)) is finite. The asymptotic variance of the
    sample variance S² for sample size N is:
      Var(S²) ≈ (σ⁴ / N) * (κ + 2)
    where excess kurtosis κ = 6 / (ν - 4), and theoretical variance
    σ² = ν / (ν - 2).

    The Relative Standard Error (RSE) is:
      RSE(S²) = sqrt(Var(S²)) / σ² = sqrt((κ + 2) / N)

    For N = 200,000 samples (shape: 200 x 1000):
      - ν = 5.0:  κ = 6.0, RSE = 0.63%, 5σ bound = 3.16% -> rtol = 0.04 (4%)
      - ν = 6.0:  κ = 3.0, RSE = 0.50%, 5σ bound = 2.50% -> rtol = 0.03 (3%)
      - ν = 8.0:  κ = 1.5, RSE = 0.42%, 5σ bound = 2.10% -> rtol = 0.03 (3%)
      - ν = 10.0: κ = 1.0, RSE = 0.39%, 5σ bound = 1.95% -> rtol = 0.025 (2.5%)

    This guarantees zero test flakiness (p < 10^-6) while strictly verifying
    that the generator scales variance faithfully as a function of ν.
    """
    tensor = numerical_generator.generate_student_t_tensor(
        shape=(200, 1000), dtype_str="float32", df=df, seed=100 + int(df)
    )
    arr = np.asarray(tensor, dtype=np.float64)
    expected_var = df / (df - 2.0)
    sample_var = float(np.var(arr))
    np.testing.assert_allclose(
        sample_var,
        expected_var,
        rtol=rtol,
        err_msg=(
            f"Sample variance {sample_var:.4f} deviates from theoretical"
            f" {expected_var:.4f} for df={df}"
        ),
    )

  @parameterized.named_parameters(
      ("df_2p5", 2.5, 0.78501, 1.73025, 2.55822),
      ("df_3", 3.0, 0.76489, 1.63774, 2.35336),
      ("df_4", 4.0, 0.74070, 1.53321, 2.13185),
  )
  def test_student_t_quantiles_heavy_tail(
      self,
      df: float,
      expected_q75: float,
      expected_q90: float,
      expected_q95: float,
  ):
    """Confirms sample quantiles match theoretical t-distribution for ν <= 4.

    Mathematical Decision & Derivation:
    -----------------------------------
    For heavy-tailed regimes where ν <= 4.0 (such as ν = 2.5, 3.0, or 4.0),
    the 4th central moment μ4 is INFINITE, which implies Var(S²) = ∞.
    Therefore, evaluating sample variance S² with a fixed rtol is
    mathematically ill-posed and leads to sporadic flakiness when rare extreme
    values are drawn.

    Instead, we test non-parametric sample quantiles (Q25, Q50, Q75, Q90, Q95)
    and the Interquartile Range (IQR = Q75 - Q25 = 2 * Q75 by symmetry).
    Sample quantiles have finite asymptotic variance governed by:
      Var(Q_p) ≈ p * (1 - p) / (N * [f(Q_p)]²)
    which is bounded and well-behaved for all degrees of freedom ν > 0.

    Theoretical quantile values Q(p) = t_ν⁻¹(p) (from scipy.stats.t.ppf):
      - ν = 2.5: Q75 = 0.78501, Q90 = 1.73025, Q95 = 2.55822
      - ν = 3.0: Q75 = 0.76489, Q90 = 1.63774, Q95 = 2.35336
      - ν = 4.0: Q75 = 0.74070, Q90 = 1.53321, Q95 = 2.13185

    With N = 200,000, sample IQR and quantiles match within rtol = 0.02 (2%).
    """
    tensor = numerical_generator.generate_student_t_tensor(
        shape=(200, 1000), dtype_str="float32", df=df, seed=42 + int(df * 10)
    )
    arr = np.asarray(tensor, dtype=np.float64)
    q25, q50, q75, q90, q95 = np.percentile(arr, [25, 50, 75, 90, 95])

    # Median should be 0.0 by symmetry
    self.assertAlmostEqual(q50, 0.0, delta=0.015)
    # Interquartile Range (IQR = Q75 - Q25)
    np.testing.assert_allclose(q75 - q25, 2.0 * expected_q75, rtol=0.02)
    # 75th, 90th, and 95th percentiles
    np.testing.assert_allclose(q75, expected_q75, rtol=0.02)
    np.testing.assert_allclose(q90, expected_q90, rtol=0.02)
    np.testing.assert_allclose(q95, expected_q95, rtol=0.03)

  @parameterized.named_parameters(
      ("df_5", 5.0, 2.0),
      ("df_6", 6.0, 1.5),
  )
  def test_student_t_excess_kurtosis_heavy_tail(
      self, df: float, min_excess_kurtosis: float
  ):
    """Confirms generated tensor exhibits positive excess kurtosis.

    Mathematical Decision:
    ----------------------
    Standard normal distributions have excess kurtosis κ = 0.
    Student's t-distribution with ν > 4 has theoretical excess kurtosis:
      κ = 6 / (ν - 4) > 0
    For ν = 5.0: theoretical κ = 6.0.
    For ν = 6.0: theoretical κ = 3.0.

    Note on 8th Central Moment:
    For ν <= 6, the 8th central moment μ8 is INFINITE, which implies that the
    sampling variance of the sample kurtosis estimator is infinite
    (Var(κ_sample) = ∞). Therefore, testing sample kurtosis with a fixed
    two-sided tolerance is ill-posed. We test a one-sided lower bound to prove
    heavy tails relative to Gaussian without flakiness.
    """
    tensor = numerical_generator.generate_student_t_tensor(
        shape=(200, 1000), dtype_str="float32", df=df, seed=42
    )
    arr = np.asarray(tensor, dtype=np.float64)
    mean = np.mean(arr)
    std = np.std(arr)
    kurtosis = np.mean(((arr - mean) / std) ** 4) - 3.0
    self.assertGreater(
        kurtosis,
        min_excess_kurtosis,
        msg=f"Sample kurtosis {kurtosis:.2f} is not heavy-tailed for df={df}",
    )

  @parameterized.parameters(2.5, 3.0, 4.0, 5.0, 10.0)
  def test_student_t_symmetry_and_mean(self, df: float):
    """Confirms zero-mean, zero-skewness, and 50/50 sign balance.

    Mathematical Decision & Derivation:
    -----------------------------------
    Student's t-distribution is strictly symmetric about zero for all ν > 1.
    For ν <= 3, the theoretical 3rd moment μ3 is INFINITE, which causes sample
    moment skewness to fluctuate wildly. Therefore, we evaluate symmetry via
    Bowley's quantile skewness:
      S_Bowley = (Q75 + Q25 - 2*Q50) / (Q75 - Q25)
    which is non-parametric, strictly bounded in [-1, 1], and equals 0.0 for
    any symmetric distribution.

    For ν = 2.5, Var(X) = 5.0, yielding standard error SE(X_bar) = 0.005 for
    N = 200,000. delta = 0.025 provides a safe 5-sigma bound.

    We verify:
      1. E[X] = 0 (sample mean |x_bar| < 0.025).
      2. Symmetry: Bowley's quantile skewness |S_Bowley| < 0.02.
      3. Sign balance: fraction of positive elements is 50% ± 1%.
    """
    tensor = numerical_generator.generate_student_t_tensor(
        shape=(200, 1000), dtype_str="float32", df=df, seed=77 + int(df * 10)
    )
    arr = np.asarray(tensor, dtype=np.float64)
    mean = float(np.mean(arr))
    q25, q50, q75 = np.percentile(arr, [25, 50, 75])
    bowley_skewness = (q75 + q25 - 2.0 * q50) / (q75 - q25)
    frac_positive = float(np.mean(arr > 0))

    self.assertAlmostEqual(mean, 0.0, delta=0.025)
    self.assertAlmostEqual(bowley_skewness, 0.0, delta=0.02)
    self.assertAlmostEqual(frac_positive, 0.50, delta=0.01)

  @parameterized.parameters(
      "float32",
      "bfloat16",
      "float16",
      "fp8_e4m3",
      "fp8_e5m2",
  )
  def test_student_t_seed_determinism(self, dtype_str: str):
    """Confirms identical seeds yield identical bitwise arrays."""
    t1 = numerical_generator.generate_student_t_tensor(
        shape=(128, 128), dtype_str=dtype_str, seed=999
    )
    t2 = numerical_generator.generate_student_t_tensor(
        shape=(128, 128), dtype_str=dtype_str, seed=999
    )
    t3 = numerical_generator.generate_student_t_tensor(
        shape=(128, 128), dtype_str=dtype_str, seed=1000
    )
    np.testing.assert_array_equal(t1, t2)
    self.assertFalse(np.array_equal(t1, t3))

  @parameterized.named_parameters(
      ("ratio_1pct_scale_20x", 0.01, 20.0),
      ("ratio_2pct_scale_50x", 0.02, 50.0),
      ("ratio_5pct_scale_80x", 0.05, 80.0),
  )
  def test_outlier_injection_ratio_and_magnitude(
      self, outlier_ratio: float, outlier_scale: float
  ):
    """Confirms outlier proportion and magnitude scaling."""
    shape = (200, 1000)
    tensor = numerical_generator.generate_outlier_tensor(
        shape=shape,
        dtype_str="float32",
        outlier_ratio=outlier_ratio,
        outlier_scale=outlier_scale,
        seed=42,
    )
    arr = np.asarray(tensor, dtype=np.float32)
    median_val = np.median(np.abs(arr)) + 1e-6
    expected_spike_mag = median_val * outlier_scale

    # Elements exceeding 50% of expected spike magnitude are treated as outliers
    is_outlier = np.abs(arr) >= (expected_spike_mag * 0.5)
    empirical_ratio = float(np.mean(is_outlier))

    # Binomial proportion standard error SE = sqrt(p*(1-p)/N) ≈ 0.02%
    np.testing.assert_allclose(empirical_ratio, outlier_ratio, atol=0.005)

  @parameterized.parameters(
      ("float32", -1, 0.1),
      ("bfloat16", -1, 0.1),
      ("float16", -1, 0.1),
      ("fp8_e4m3", -1, 0.1),
      ("fp8_e5m2", -1, 0.1),
      ("float32", 0, 0.05),
      ("bfloat16", 0, 0.05),
  )
  def test_cancellation_tensor_structure_and_residual(
      self, dtype_str: str, reduction_axis: int, epsilon: float
  ):
    """Confirms alternating cancellation pairs and sum reduction residual."""
    shape = (64, 128)
    tensor = numerical_generator.generate_cancellation_tensor(
        shape=shape,
        dtype_str=dtype_str,
        reduction_axis=reduction_axis,
        epsilon=epsilon,
    )
    arr = np.asarray(tensor, dtype=np.float32)

    # Reduction sum along reduction_axis should be strictly non-zero
    reduced_sum = np.sum(arr, axis=reduction_axis)
    self.assertTrue(
        np.all(np.abs(reduced_sum) > 0.0),
        msg=(
            "Cancellation residual was wiped out by quantization in"
            f" {dtype_str}"
        ),
    )
    if dtype_str == "float32":
      dim_len = shape[reduction_axis]
      expected_sum = (dim_len / 2.0) * epsilon
      np.testing.assert_allclose(reduced_sum, expected_sum, rtol=0.005)

  @parameterized.parameters(
      "float32",
      "bfloat16",
      "float16",
      "fp8_e4m3",
      "fp8_e5m2",
  )
  def test_boundary_probe_values_and_strides(self, dtype_str: str):
    """Confirms presence of subnormals, min normals, and stride alignment."""
    shape = (16, 256)
    tile_stride = 128
    tensor = numerical_generator.generate_boundary_probe_tensor(
        shape=shape, dtype_str=dtype_str, tile_stride=tile_stride
    )
    arr = np.asarray(tensor)
    profile = numerical_generator.PROFILES[dtype_str]

    # Verify zero, min_normal, and min_subnormal are present
    self.assertIn(0.0, arr)
    arr_f64 = arr.astype(np.float64)
    self.assertTrue(
        np.any(np.isclose(arr_f64, profile.min_normal, rtol=1e-2)),
        msg=f"min_normal {profile.min_normal} missing for {dtype_str}",
    )
    self.assertTrue(
        np.any(np.isclose(arr_f64, profile.min_subnormal, rtol=1e-2)),
        msg=f"min_subnormal {profile.min_subnormal} missing for {dtype_str}",
    )

    # Verify periodic tile stride alignment
    flat = arr.flatten()
    stride_step = 5 * tile_stride  # 5 probe values
    if stride_step < len(flat):
      self.assertEqual(flat[0], flat[stride_step])

  # ===========================================================================
  # Negative Test Cases: Statistical Discriminators & Input Validation
  # ===========================================================================

  def test_statistical_negative_gaussian_fails_heavy_tail_kurtosis(self):
    """Verifies standard normal distribution fails the heavy-tail kurtosis check.

    Negative Test Rationale:
    ------------------------
    Gaussian noise has theoretical excess kurtosis κ = 0.
    Our heavy-tail check requires sample kurtosis > 2.0.
    Passing Gaussian normal samples must fail this test with sample
    kurtosis ≈ 0.
    """
    rng = np.random.default_rng(42)
    gaussian_samples = rng.standard_normal((200, 1000)).astype(np.float64)
    mean = np.mean(gaussian_samples)
    std = np.std(gaussian_samples)
    sample_kurtosis = np.mean(((gaussian_samples - mean) / std) ** 4) - 3.0

    # Gaussian excess kurtosis is approximately 0 (< 0.1) and NOT > 2.0
    self.assertLess(sample_kurtosis, 0.1)
    with self.assertRaises(AssertionError):
      self.assertGreater(sample_kurtosis, 2.0)

  def test_statistical_negative_gaussian_fails_variance_scaling(self):
    """Verifies Gaussian distribution with Var=1.0 fails variance scaling.

    Negative Test Rationale:
    ------------------------
    Student's t with df=5.0 requires theoretical variance ν / (ν - 2) = 1.667.
    A standard Gaussian distribution has Var = 1.0 (a 40% discrepancy),
    which drastically fails the rtol=0.04 tolerance.
    """
    rng = np.random.default_rng(42)
    gaussian_samples = rng.standard_normal((200, 1000)).astype(np.float64)
    expected_t_var = 5.0 / (5.0 - 2.0)  # 1.6667
    sample_var = float(np.var(gaussian_samples))  # ≈ 1.0

    relative_error = abs(sample_var - expected_t_var) / expected_t_var
    self.assertGreater(relative_error, 0.35)  # > 35% error
    with self.assertRaises(AssertionError):
      np.testing.assert_allclose(sample_var, expected_t_var, rtol=0.04)

  def test_statistical_negative_gaussian_fails_heavy_tail_quantiles(self):
    """Verifies Gaussian distribution fails Student's t heavy-tail quantiles.

    Negative Test Rationale:
    ------------------------
    For Student's t with df=3.0, theoretical Q75 = 0.76489.
    For standard Gaussian, theoretical Q75 = 0.67449 (an 11.8% discrepancy).
    Gaussian samples must fail the rtol=0.02 quantile check.
    """
    rng = np.random.default_rng(42)
    gaussian_samples = rng.standard_normal((200, 1000)).astype(np.float64)
    q75 = float(np.percentile(gaussian_samples, 75))
    expected_t_q75 = 0.76489

    with self.assertRaises(AssertionError):
      np.testing.assert_allclose(q75, expected_t_q75, rtol=0.02)

  def test_statistical_negative_exponential_fails_symmetry_and_mean(self):
    """Verifies asymmetric exponential distribution fails symmetry and mean.

    Negative Test Rationale:
    ------------------------
    Exponential distribution Exp(1) is strictly positive, has mean = 1.0 != 0,
    and Bowley quantile skewness ≈ 0.26 != 0.
    It must fail both the zero-mean and Bowley skewness checks.
    """
    rng = np.random.default_rng(42)
    exp_samples = rng.exponential(scale=1.0, size=(200, 1000)).astype(
        np.float64
    )
    mean = float(np.mean(exp_samples))
    q25, q50, q75 = np.percentile(exp_samples, [25, 50, 75])
    bowley_skewness = (q75 + q25 - 2.0 * q50) / (q75 - q25)
    frac_positive = float(np.mean(exp_samples > 0))

    # Mean is ~1.0, not 0.0
    self.assertGreater(abs(mean), 0.9)
    # Bowley skewness is ~0.26, not 0.0
    self.assertGreater(abs(bowley_skewness), 0.2)
    # Fraction positive is 100%, not 50%
    self.assertEqual(frac_positive, 1.0)

  @parameterized.parameters(
      -2.0,
      -0.5,
      0.0,
      float("nan"),
      float("inf"),
      float("-inf"),
  )
  def test_generate_student_t_invalid_df_raises(self, invalid_df: float):
    """Verifies non-positive or non-finite degrees of freedom raise ValueError."""
    with self.assertRaises(ValueError):
      numerical_generator.generate_student_t_tensor(
          shape=(16, 16), df=invalid_df
      )

  def test_generate_student_t_unsupported_dtype_raises(self):
    """Verifies unsupported dtype string raises KeyError."""
    with self.assertRaises(KeyError):
      numerical_generator.generate_student_t_tensor(
          shape=(16, 16), dtype_str="unsupported_fake_dtype"
      )

  @parameterized.parameters(-0.1, 1.5, 2.0)
  def test_generate_outlier_invalid_ratio_raises(self, invalid_ratio: float):
    """Verifies outlier ratio outside [0.0, 1.0] raises ValueError."""
    with self.assertRaises(ValueError):
      numerical_generator.generate_outlier_tensor(
          shape=(16, 16), outlier_ratio=invalid_ratio
      )

  @parameterized.parameters(
      -10.0,
      0.0,
      float("nan"),
      float("inf"),
      float("-inf"),
  )
  def test_generate_outlier_invalid_scale_raises(self, invalid_scale: float):
    """Verifies non-positive or non-finite outlier scale raises ValueError."""
    with self.assertRaises(ValueError):
      numerical_generator.generate_outlier_tensor(
          shape=(16, 16), outlier_scale=invalid_scale
      )

  @parameterized.parameters(2, -3, 10)
  def test_generate_cancellation_invalid_axis_raises(self, invalid_axis: int):
    """Verifies out-of-bounds reduction axis raises IndexError."""
    with self.assertRaises(IndexError):
      numerical_generator.generate_cancellation_tensor(
          shape=(16, 16), reduction_axis=invalid_axis
      )

  def test_generate_cancellation_odd_dimension_raises(self):
    """Verifies odd reduction dimension length raises ValueError."""
    with self.assertRaises(ValueError):
      numerical_generator.generate_cancellation_tensor(
          shape=(16, 127), reduction_axis=-1
      )

  def test_load_corrupted_bytes_raises(self):
    """Verifies loading corrupted byte payload raises error."""
    corrupted_bytes = b"NOT_A_VALID_ZIP_ARCHIVE_DATA"
    with self.assertRaises((RuntimeError, ValueError, Exception)):
      numerical_generator.load_test_suite(corrupted_bytes)

  @parameterized.parameters("fp8_e4m3", "float16", "fp8_e5m2")
  def test_student_t_no_nan_or_inf_in_narrow_dtypes(self, dtype_str: str):
    """Verifies heavy-tailed Student-t never produces NaN or Inf in narrow dtypes."""
    # 100,000 draws with heavy tail (df=2.5) to stress test extreme draws
    tensor = numerical_generator.generate_student_t_tensor(
        shape=(100, 1000), dtype_str=dtype_str, df=2.5, seed=12345
    )
    arr = np.asarray(tensor, dtype=np.float32)
    self.assertFalse(
        np.isnan(arr).any(),
        msg=f"Found NaN in Student-t tensor for {dtype_str}",
    )
    self.assertFalse(
        np.isinf(arr).any(),
        msg=f"Found Inf in Student-t tensor for {dtype_str}",
    )

  def test_scalar_shape_empty_tuple(self):
    """Verifies shapes=() produces batches with a single scalar tensor."""
    suite = numerical_generator.generate_test_suite(
        shapes=(),
        dtype_str="float32",
        tier="fast_agent",
        seed=42,
    )
    self.assertNotEmpty(suite)
    for batch in suite:
      self.assertLen(batch["args"], 1)
      self.assertEqual(batch["args"][0].shape, ())

  def test_scalar_shape_in_sequence(self):
    """Verifies shapes=[(), (16, 16)] produces scalar and matrix tensors."""
    suite = numerical_generator.generate_test_suite(
        shapes=[(), (16, 16)],
        dtype_str="float32",
        tier="fast_agent",
        seed=42,
    )
    self.assertNotEmpty(suite)
    for batch in suite:
      self.assertLen(batch["args"], 2)
      self.assertEqual(batch["args"][0].shape, ())
      self.assertEqual(batch["args"][1].shape, (16, 16))

  def test_load_test_suite_closes_file_descriptor(self):
    """Verifies load_test_suite does not leak open file handles."""
    suite = numerical_generator.generate_test_suite(
        shapes=[(8, 8)],
        dtype_str="float32",
        tier="fast_agent",
        seed=42,
    )
    temp_path = self.create_tempfile("leak_test.npz").full_path
    numerical_generator.save_test_suite(suite, temp_path)

    # Repeatedly load the fixture to verify no EMFILE file descriptor leaks
    for _ in range(50):
      loaded = numerical_generator.load_test_suite(temp_path)
      self.assertLen(loaded, len(suite))


if __name__ == "__main__":
  absltest.main()
