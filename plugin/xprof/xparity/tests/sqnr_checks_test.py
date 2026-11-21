"""Tests for the derived checks of sqnr.py (gate_math="derived")."""

import gc
import math
import tracemalloc

from absl.testing import absltest
from absl.testing import parameterized
import ml_dtypes
import numpy as np

from xprof.xparity import sqnr

_BF16 = ml_dtypes.bfloat16
_BF16_OUT = sqnr.PrecisionSpec(output_dtype="bfloat16")
_MXFP4_OUT = sqnr.PrecisionSpec(output_dtype="mxfp4")
_SEEDS = (0, 1, 2)


def _audit(ref, cand, precision=None, views=(), shards=None, **cfg):
  pairs = shards if shards is not None else [(ref, cand)]
  return sqnr.audit_leaf(
      pairs,
      config=sqnr.SqnrConfig(**cfg),
      precision=precision,
      input_views=[[v] for v in views],
      path="out",
  )


def _checks(report) -> set[str]:
  """Returns the first word of each reason after "out: "."""
  return {r.split(": ", 1)[1].split()[0] for r in report.failed_reasons}


def _mxfp4_round(x: np.ndarray) -> np.ndarray:
  """Rounds to MX FP4 (E2M1, one power-of-two scale per 32 elements)."""
  blocks = x.astype(np.float64).reshape(-1, 32)
  amax = np.max(np.abs(blocks), axis=1, keepdims=True)
  scale = np.ldexp(1.0, np.frexp(np.maximum(amax, 1e-300))[1] - 1 - 2)
  q = (blocks / scale).astype(ml_dtypes.float4_e2m1fn).astype(np.float64)
  return (q * scale).reshape(x.shape).astype(np.float32)


def _distribution(name: str, rng, shape) -> np.ndarray:
  if name == "gaussian":
    return rng.standard_normal(shape)
  if name == "student_t3":
    return np.clip(rng.standard_t(3, shape), -1e3, 1e3)
  if name == "lognormal":
    return rng.lognormal(0.0, 1.5, shape)
  if name == "outliers":
    x = rng.standard_normal(shape)
    hit = rng.random(shape) < 1e-3
    return np.where(hit, 100.0 * x, x)
  # cancellation_pairs: +-L alternates along the minor axis, plus N(0, 1).
  big = rng.uniform(1.0, 1000.0, shape[:-1] + (shape[-1] // 2,))
  pairs = np.stack([big, -big], axis=-1).reshape(shape)
  return pairs + rng.standard_normal(shape)


class HonestTest(parameterized.TestCase):

  @parameterized.product(
      fmt=("bfloat16", "float16", "float8_e4m3fn", "float8_e5m2"),
      dist=("gaussian", "student_t3", "lognormal", "outliers"),
  )
  def test_round_to_nearest_cast_passes(self, fmt, dist):
    for seed in _SEEDS:
      rng = np.random.default_rng(seed)
      ref = _distribution(dist, rng, (64, 256)).astype(np.float32)
      cand = ref.astype(np.dtype(fmt))
      report = _audit(ref, cand, sqnr.PrecisionSpec(output_dtype=fmt))
      self.assertTrue(report.passed, (seed, report.failed_reasons))

  def test_bf16_cancellation_pairs_pass(self):
    for seed in _SEEDS:
      ref = _distribution(
          "cancellation_pairs", np.random.default_rng(seed), (64, 256)
      ).astype(np.float32)
      report = _audit(ref, ref.astype(_BF16), _BF16_OUT)
      self.assertEmpty(report.failed_reasons, seed)

  @parameterized.parameters("gaussian", "outliers", "student_t3")
  def test_mxfp4_dequantized_to_f32_passes(self, dist):
    for seed in _SEEDS:
      ref = _distribution(dist, np.random.default_rng(seed), (64, 256))
      ref = ref.astype(np.float32)
      report = _audit(ref, _mxfp4_round(ref), _MXFP4_OUT)
      self.assertEmpty(_checks(report) - {"sqnr_db"}, seed)

  def test_f32_matmul_with_bf16_noise_passes(self):
    # f32 output with bf16 operand noise and no PrecisionSpec: the fitted B
    # term carries the noise that the f32 ulp does not explain.
    for seed in _SEEDS:
      rng = np.random.default_rng(seed)
      a = rng.standard_normal((128, 256)).astype(np.float32)
      b = rng.standard_normal((256, 128)).astype(np.float32)
      ref = a.astype(np.float64) @ b.astype(np.float64)
      cand = a.astype(_BF16).astype(np.float32) @ b.astype(_BF16).astype(
          np.float32
      )
      report = _audit(ref, cand, min_sqnr_db=40.0)
      self.assertTrue(report.passed, (seed, report.failed_reasons))

  @parameterized.parameters(0.25, 1.0)
  def test_quiet_row_passes(self, ratio):
    for seed in _SEEDS:
      rng = np.random.default_rng(seed)
      ref = rng.standard_normal((128, 128)).astype(np.float32)
      ref[0] = (math.sqrt(ratio) * 1e-3 * rng.standard_normal(128)).astype(
          np.float32
      )
      cand = ref + (1e-3 * rng.standard_normal((128, 128))).astype(np.float32)
      report = _audit(ref, cand)
      self.assertTrue(report.passed, (seed, report.failed_reasons))

  @parameterized.parameters(1000.0, 1002.0, 1004.0)
  def test_bf16_dc_grid_passes(self, dc):
    # At a DC of 1000 the bf16 step is 4, so most outputs round to the DC. The
    # rounding error is then a function of y, and correlated with it.
    for seed in _SEEDS:
      ref = dc + np.random.default_rng(seed).standard_normal((128, 256))
      report = _audit(ref, ref.astype(_BF16).astype(np.float32))
      self.assertTrue(report.passed, (seed, report.failed_reasons))

  def test_complex_noise_in_small_part_passes(self):
    rng = np.random.default_rng(0)
    ref = rng.standard_normal(1024) + 1j * 1e-5 * rng.standard_normal(1024)
    ref = ref.astype(np.complex64)
    cand = ref.real + 1j * 1e-5 * rng.standard_normal(1024)
    report = _audit(ref, cand.astype(np.complex64))
    self.assertTrue(report.passed, report.failed_reasons)

  def test_honest_residual_update_passes(self):
    for seed in _SEEDS:
      rng = np.random.default_rng(seed)
      x = rng.standard_normal((128, 128)).astype(np.float32)
      ref = x + np.float32(1e-2) * np.tanh(x)
      report = _audit(ref, ref.astype(_BF16), _BF16_OUT, views=(x,))
      self.assertTrue(report.passed, (seed, report.failed_reasons))

  @parameterized.parameters("bfloat16", "float32")
  def test_same_dtype_summation_order_passes(self, dtype):
    # Reference and candidate are the same f32-accumulated matmul in two
    # summation orders, both rounded to dtype. Only a few elements differ, by
    # one step each.
    for seed in _SEEDS:
      rng = np.random.default_rng(seed)
      a = rng.standard_normal((256, 1024)).astype(np.float32)
      b = rng.standard_normal((1024, 256)).astype(np.float32)
      ref = (a @ b).astype(dtype)
      parts = [a[:, i : i + 256] @ b[i : i + 256] for i in range(0, 1024, 256)]
      cand = (parts[3] + parts[2] + parts[1] + parts[0]).astype(dtype)
      report = _audit(ref, cand, sqnr.PrecisionSpec(output_dtype=dtype))
      self.assertTrue(report.passed, (seed, report.failed_reasons))

  @parameterized.product(
      dtype=("bfloat16", "float32"), fraction=(1e-4, 1e-2, 0.1)
  )
  def test_sparse_one_step_differences_pass(self, dtype, fraction):
    # Up to 10% of elements one step away. At 50% the error power is 6 times
    # one rounding (-7.8 dB), below the 6 dB margin of the global floor.
    for seed in _SEEDS:
      rng = np.random.default_rng(seed)
      ref = rng.standard_normal((256, 256)).astype(dtype)
      hit = rng.random(ref.shape) < fraction
      side = np.where(rng.random(ref.shape) < 0.5, np.inf, -np.inf)
      cand = np.where(hit, np.nextafter(ref, side.astype(dtype)), ref)
      report = _audit(ref, cand.astype(dtype))
      self.assertTrue(report.passed, (seed, report.failed_reasons))

  def test_two_sided_bf16_matmul_passes(self):
    # bf16 operands, f32 accumulation in two orders, bf16 outputs on both
    # sides.
    for seed in _SEEDS:
      rng = np.random.default_rng(seed)
      a = rng.standard_normal((256, 1024)).astype(_BF16).astype(np.float32)
      b = rng.standard_normal((1024, 256)).astype(_BF16).astype(np.float32)
      ref = (a @ b).astype(_BF16)
      parts = [a[:, i : i + 256] @ b[i : i + 256] for i in range(0, 1024, 256)]
      cand = (parts[3] + parts[2] + parts[1] + parts[0]).astype(_BF16)
      report = _audit(ref, cand, _BF16_OUT)
      self.assertTrue(report.passed, (seed, report.failed_reasons))

  def test_metrics_are_reported(self):
    ref = np.random.default_rng(0).standard_normal((64, 256)).astype(np.float32)
    report = _audit(ref, ref.astype(_BF16), _BF16_OUT)
    self.assertLessEqual(report.max_error_ulp, 0.5)
    self.assertIsNotNone(report.max_block_noise_ratio_db)
    self.assertIsNotNone(report.min_block_sqnr_db)
    self.assertIsNotNone(report.min_ac_snr_db)
    self.assertIsNotNone(report.centered_snr_db)


class CheatTest(parameterized.TestCase):
  """Each cheat passes the global SQNR floor. The named check must catch it."""

  def _assert_caught(self, report, check):
    self.assertFalse(report.passed)
    self.assertNotIn("sqnr_db", _checks(report))
    self.assertIn(check, _checks(report))

  @parameterized.named_parameters(
      ("single_block", np.s_[3, 32:64]),
      ("tail", np.s_[-1, -32:]),
  )
  def test_zero_block_is_caught_by_block_noise(self, where):
    ref = np.random.default_rng(0).standard_normal((512, 512))
    ref = ref.astype(np.float32)
    cand = ref.astype(_BF16).astype(np.float32)
    cand[where] = 0.0
    report = _audit(ref, cand, _BF16_OUT, min_sqnr_db=30.0)
    self._assert_caught(report, "max_block_noise_ratio_db")

  def test_dc_skip_is_caught_by_block_noise(self):
    ref = 100.0 + np.random.default_rng(0).standard_normal((128, 256))
    cand = ref.copy()
    cand[0, :32] = 100.0
    self._assert_caught(_audit(ref, cand), "max_block_noise_ratio_db")

  def test_sign_flip_block_is_caught_by_block_noise(self):
    ref = np.random.default_rng(0).standard_normal((512, 512))
    ref = ref.astype(np.float32)
    cand = ref.astype(_BF16).astype(np.float32)
    cand[7, 64:96] *= -1.0
    report = _audit(ref, cand, _BF16_OUT, min_sqnr_db=30.0)
    self._assert_caught(report, "max_block_noise_ratio_db")

  def test_single_spike_is_caught(self):
    ref = np.random.default_rng(0).standard_normal((128, 128))
    ref = ref.astype(np.float32)
    cand = ref.copy()
    cand[5, 9] += 0.3
    report = _audit(ref, cand)
    self._assert_caught(report, "max_error_ulp")

  def test_spike_next_to_an_exact_large_element_is_caught(self):
    # The large element is exact in bf16 (e = 0). The fit must not drop A for
    # it, or the 8 ulp spike on a small element falls inside the bound.
    ref = np.random.default_rng(0).standard_normal((512, 512))
    ref = ref.astype(np.float32)
    ref[9, 0] = 1000.0
    cand = ref.astype(_BF16).astype(np.float32)
    ulp = 2.0 ** (math.floor(math.log2(abs(float(ref[9, 5])))) - 7)
    cand[9, 5] += np.float32(8.0 * ulp)
    report = _audit(ref, cand, _BF16_OUT)
    self._assert_caught(report, "max_error_ulp")

  def test_spike_is_caught_next_to_a_worst_element_at_a_zero_reference(self):
    # ReLU output: reference zeros have v = 0 and carry small noise, so the
    # fit gets B > 0. One larger error at a reference zero has the largest
    # |e| / sigma in the tile but no step to compare with. It must not hide
    # the 4 ulp spike on a nonzero element.
    rng = np.random.default_rng(0)
    x = rng.standard_normal((64, 64)).astype(np.float32)
    ref = np.maximum(x, 0).astype(_BF16).astype(np.float32)
    zero = ref == 0
    cand = ref.copy()
    cand[zero] = 1e-4 * rng.standard_normal(np.count_nonzero(zero))
    cand[tuple(np.argwhere(zero)[3])] = 2e-3
    spot = tuple(np.argwhere(~zero)[5])
    cand[spot] += np.float32(4.0 * float(np.spacing(ref[spot])) * 2**16)
    report = sqnr.audit_leaf(
        [(ref, cand)], precision=sqnr.PrecisionSpec(output_dtype="bfloat16")
    )
    self._assert_caught(report, "max_error_ulp")

  @parameterized.parameters(1000.0, 1e15)
  def test_constant_output_is_caught_by_centered(self, dc):
    ref = dc + np.random.default_rng(0).standard_normal((64, 128))
    report = _audit(ref, np.full_like(ref, dc))
    self._assert_caught(report, "centered_corr_snr_db")

  @parameterized.named_parameters(
      ("rows", np.s_[:64, :]),
      ("cols", np.s_[:, :128]),
  )
  def test_half_constant_is_caught_by_centered(self, where):
    ref = 1000.0 + np.random.default_rng(0).standard_normal((128, 256))
    cand = ref.copy()
    cand[where] = 1000.0
    self._assert_caught(_audit(ref, cand), "centered_corr_snr_db")

  def test_quiet_expert_dropped_is_caught_by_centered(self):
    rng = np.random.default_rng(0)
    ref = (rng.standard_normal((64, 256)) / 32.0).astype(np.float32)
    ref[0] = rng.standard_normal(256).astype(np.float32) * 80.0
    cand = np.zeros_like(ref)
    cand[0] = ref[0]
    report = _audit(ref, cand, min_sqnr_db=40.0)
    self._assert_caught(report, "centered_corr_snr_db")

  def test_dropped_imaginary_part_is_caught_by_centered(self):
    rng = np.random.default_rng(0)
    ref = rng.standard_normal(128) + 1j * 1e-3 * rng.standard_normal(128)
    ref = ref.astype(np.complex64)
    report = _audit(ref, (ref.real + 0j).astype(np.complex64))
    self._assert_caught(report, "centered_corr_snr_db")

  def test_row_mean_is_caught_by_diff(self):
    rng = np.random.default_rng(0)
    ref = rng.standard_normal((32, 32)) + 5000.0 * rng.standard_normal((32, 1))
    ref = ref.astype(np.float32)
    cand = np.broadcast_to(np.mean(ref, 1, keepdims=True), ref.shape)
    cand = cand.astype(np.float32)
    shards = [(ref[:16], cand[:16]), (ref[16:], cand[16:])]
    report = _audit(None, None, shards=shards)
    self._assert_caught(report, "diff_corr_snr_db")

  def test_separable_bias_is_caught_by_plaquette(self):
    rng = np.random.default_rng(0)
    rb = rng.standard_normal((64, 1)) * 200.0
    cb = rng.standard_normal((1, 128)) * 200.0
    cand = (rb + cb) * np.ones((64, 128))
    ref = cand + rng.standard_normal((64, 128))
    self._assert_caught(_audit(ref, cand), "plaquette_corr_snr_db")

  @parameterized.named_parameters(
      ("return_x", 0.0),
      ("plus_eps", 1e-6),
  )
  def test_passthrough_is_caught_by_input_delta(self, eps):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((256, 256)).astype(np.float32)
    upd = np.tanh(rng.standard_normal((256, 256)))
    upd *= 10.0 ** (-50.0 / 20.0) / float(np.std(upd))
    ref = (x.astype(np.float64) + upd).astype(np.float32)
    report = _audit(ref, x + np.float32(eps), views=(x,))
    self._assert_caught(report, "min_delta_sqnr_db")

  def test_scalar_passthrough_is_caught_by_input_delta(self):
    report = _audit(
        np.float32(10.05), np.float32(10.0), views=(np.float32(10.0),)
    )
    self._assert_caught(report, "min_delta_sqnr_db")


class MemoryTest(absltest.TestCase):

  def test_peak_memory_is_bounded(self):
    # The derived checks reduce each tile before the next one is read, so the
    # peak is a fixed tile workspace (about 21 MiB) and does not grow with the
    # leaf. At 2**23 elements that is about 2.6 bytes per element. Keeping one
    # float64 statistic per 8-element block would add 1 byte per element, and
    # a leaf-sized float64 array would add 8.
    for shape in ((2048, 4096), (64, 131072), (32, 64, 64, 64)):
      rng = np.random.default_rng(0)
      ref = rng.standard_normal(shape).astype(np.float32)
      cand = ref.astype(_BF16).astype(np.float32)
      gc.collect()
      tracemalloc.start()
      try:
        report = _audit(ref, cand, _BF16_OUT)
        _, peak_bytes = tracemalloc.get_traced_memory()
      finally:
        tracemalloc.stop()
      self.assertTrue(report.passed, (shape, report.failed_reasons))
      self.assertLess(peak_bytes / ref.size, 3.5, msg=str(shape))


class AlphaTest(absltest.TestCase):

  def test_smaller_alpha_widens_the_bounds(self):
    ref = np.random.default_rng(0).standard_normal((512, 512))
    ref = ref.astype(np.float32)
    cand = ref.astype(_BF16).astype(np.float32)
    cand[3, 32:64] *= 1.02  # A small block error: about 2**-6 relative.
    loose = _audit(ref, cand, _BF16_OUT, alpha=1e-300)
    tight = _audit(ref, cand, _BF16_OUT, alpha=0.5)
    self.assertNotIn("max_block_noise_ratio_db", _checks(loose))
    self.assertIn("max_block_noise_ratio_db", _checks(tight))

  def test_six_gate_v1_does_not_use_the_derived_metrics(self):
    ref = np.random.default_rng(0).standard_normal((64, 256)).astype(np.float32)
    report = _audit(
        ref, ref.astype(_BF16), _BF16_OUT, gate_math=sqnr.GATE_MATH_SIX_GATE_V1
    )
    self.assertIsNone(report.max_block_noise_ratio_db)


if __name__ == "__main__":
  absltest.main()
