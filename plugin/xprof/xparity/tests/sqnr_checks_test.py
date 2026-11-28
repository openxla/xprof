"""Tests for the derived checks of sqnr.py (gate_math="derived")."""

import gc
import math
import tracemalloc
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import ml_dtypes
import numpy as np

from xprof.xparity import numerical_generator
from xprof.xparity import numerical_validator
from xprof.xparity import sqnr

_BF16 = ml_dtypes.bfloat16
_BF16_OUT = sqnr.PrecisionSpec(output_dtype="bfloat16")
_MXFP4_OUT = sqnr.PrecisionSpec(output_dtype="mxfp4")
_FP8_OUT = sqnr.PrecisionSpec(output_dtype="float8_e4m3fn")
_F32_OUT = sqnr.PrecisionSpec(output_dtype="float32")
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


def _per_channel_outlier_operands(
    seed: int, outlier_scale: float = 50.0
) -> tuple[np.ndarray, np.ndarray]:
  """Returns x (256, 1024) and w (1024, 512) with 2% outlier columns."""
  x, w = (
      numerical_generator.generate_per_channel_outlier_tensor(
          shape, "float32", outlier_scale=outlier_scale, seed=s
      )
      for shape, s in (((256, 1024), seed), ((1024, 512), seed + 1))
  )
  return np.asarray(x, np.float32), np.asarray(w, np.float32)


def _zero_pad_columns(a: np.ndarray, n_cols: int) -> np.ndarray:
  """Returns a with zero columns appended up to n_cols columns."""
  return np.pad(a, ((0, 0), (0, n_cols - a.shape[1])))


def _round_to(x: np.ndarray, fmt: str) -> np.ndarray:
  """Rounds x to fmt. float8_e4m3fn uses one scale per tensor."""
  x = np.asarray(x, np.float64)
  if fmt == "mxfp4":
    # One power-of-two scale per 32 elements along K, the contraction axis.
    return _mxfp4_round(x.T).T.astype(np.float64)
  scale = np.max(np.abs(x)) / 448.0 if fmt == "float8_e4m3fn" else 1.0
  q = (x / scale).astype(np.float32).astype(np.dtype(fmt))
  return q.astype(np.float64) * scale


def _outlier_matmul(
    kernel: str, x: np.ndarray, w: np.ndarray
) -> tuple[np.ndarray, sqnr.PrecisionSpec]:
  """Returns the candidate of kernel "<x format>_x_<w format>" and its spec."""
  x_fmt, w_fmt = kernel.split("_x_")
  cand = (_round_to(x, x_fmt) @ _round_to(w, w_fmt)).astype(np.float32)
  precision = sqnr.precision_from_dtypes(
      "float32",
      "float32",
      input_dtypes=(x_fmt, w_fmt),
      reduction_length=x.shape[1],
  )
  return cand, precision


_SUITE_SHAPES = ((512, 2048), (2048, 768))
# The AUTO floor of each kernel from its input and output formats: the floor
# that a caller sets when the PrecisionSpec has no input formats.
_BF16_X_BF16_FLOOR = 46.6
_E4M3_X_E4M3_FLOOR = 22.5
_BF16_X_E5M2_FLOOR = 19.5
# Kernels and suite batches where some elements have one dominant product:
# (name, kernel, floor, kind, seed).
_HEAVY_TAILED_CASES = (
    (
        "e4m3_student_t",
        "float8_e4m3fn_x_float8_e4m3fn",
        _E4M3_X_E4M3_FLOOR,
        "student_t",
        92,
    ),
    (
        "bf16_student_t_17",
        "bfloat16_x_bfloat16",
        _BF16_X_BF16_FLOOR,
        "student_t",
        17,
    ),
    (
        "bf16_student_t_37",
        "bfloat16_x_bfloat16",
        _BF16_X_BF16_FLOOR,
        "student_t",
        37,
    ),
    (
        "e5m2_student_t",
        "bfloat16_x_float8_e5m2",
        _BF16_X_E5M2_FLOOR,
        "student_t",
        7,
    ),
    (
        "bf16_outliers",
        "bfloat16_x_bfloat16",
        _BF16_X_BF16_FLOOR,
        "outliers",
        162,
    ),
)


def _heavy_tailed_operands(
    kind: str, seed: int
) -> tuple[np.ndarray, np.ndarray]:
  """Returns x and w of `_SUITE_SHAPES` as the test suite makes them.

  Args:
    kind: "student_t", or "outliers" for an outlier scale of 80.
    seed: Seed of x. w uses seed + 1.
  """
  if kind == "student_t":
    x, w = (
        numerical_generator.generate_student_t_tensor(
            s, "float32", seed=seed + i
        )
        for i, s in enumerate(_SUITE_SHAPES)
    )
  else:
    x, w = (
        numerical_generator.generate_outlier_tensor(
            s, "float32", outlier_scale=80.0, seed=seed + i
        )
        for i, s in enumerate(_SUITE_SHAPES)
    )
  return np.asarray(x, np.float32), np.asarray(w, np.float32)


def _boundary_operands() -> tuple[np.ndarray, np.ndarray]:
  """Returns the boundary probe x and w of `_SUITE_SHAPES`."""
  x, w = (
      numerical_generator.generate_boundary_probe_tensor(s, "float32")
      for s in _SUITE_SHAPES
  )
  return np.asarray(x, np.float32), np.asarray(w, np.float32)


def _narrow_matmul(
    kernel: str, x: np.ndarray, w: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
  """Returns (ref, cand): the float32 matmul and the kernel of `_outlier_matmul`."""
  ref = (x.astype(np.float64) @ w.astype(np.float64)).astype(np.float32)
  cand, _ = _outlier_matmul(kernel, x, w)
  return ref, cand


def _validator_audit(ref: np.ndarray, cand: np.ndarray, floor: float):
  """Audits as `numerical_validator` does: output dtypes only, explicit floor.

  The floor of each kernel below is the AUTO value of its input and output
  formats. The input formats are not in the PrecisionSpec, so the checks do
  not know the input rounding.

  Args:
    ref: The reference.
    cand: The candidate.
    floor: The global SQNR floor in dB.

  Returns:
    The SqnrLeafReport of the leaf.
  """
  precision = sqnr.precision_from_dtypes("float32", "float32")
  return _audit(ref, cand, precision, min_sqnr_db=floor)


def _dominated(ref: np.ndarray) -> np.ndarray:
  """Returns the elements with y**2 above 16 times their row-column energy."""
  y2 = np.square(ref, dtype=np.float64)
  h = np.outer(np.mean(y2, axis=1), np.mean(y2, axis=0)) / np.mean(y2)
  return y2 > 16.0 * h


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


def _band_leaf(seed: int, level: float) -> np.ndarray:
  """Sparse outputs near 1e3, a band of 8 columns near level, zeros else."""
  rng = np.random.default_rng(seed)
  ref = np.zeros((256, 512))
  hit = rng.random(ref.shape) < 0.1
  ref[hit] = 1e3 * rng.standard_normal(np.count_nonzero(hit))
  ref[:, 64:72] = level * rng.standard_normal((256, 8))
  return ref.astype(np.float32)


def _quiet_row_leaf(seed: int) -> np.ndarray:
  """N(0, 1) values with row 5 near 1e-4."""
  rng = np.random.default_rng(seed)
  ref = rng.standard_normal((64, 1024))
  ref[5] = 1e-4 * rng.standard_normal(1024)
  return ref.astype(np.float32)


def _relu_leaf(seed: int) -> np.ndarray:
  """ReLU values near 0.5 with row 0 near 1500."""
  rng = np.random.default_rng(seed)
  ref = np.maximum(0.5 * rng.standard_normal((64, 1024)), 0.0)
  ref[0] = 1500.0 * rng.standard_normal(1024)
  return ref.astype(np.float32)


def _zero_block_relu_leaf(seed: int) -> np.ndarray:
  """`_relu_leaf` with 3/4 of the 32-element blocks of rows 1-63 set to 0."""
  ref = _relu_leaf(seed)
  rng = np.random.default_rng(100 + seed)
  blocks = ref[1:].reshape(63, 32, 32)
  blocks[rng.random((63, 32)) < 0.75] = 0.0
  ref[1:] = blocks.reshape(63, 1024)
  return ref


def _one_part_update(steps: int) -> tuple[np.ndarray, np.ndarray]:
  """Returns (y, x): y.real = x.real + a 1% update, y.imag `steps` from x."""
  rng = np.random.default_rng(0)
  x = rng.standard_normal(4096) + 1j * rng.standard_normal(4096)
  x = x.astype(np.complex64)
  ref = np.empty_like(x)
  ref.real = x.real + (0.01 * rng.standard_normal(4096)).astype(np.float32)
  ref.imag = x.imag
  for _ in range(steps):
    ref.imag = np.nextafter(ref.imag, np.float32(np.inf))
  return ref, x


def _round_f32(a: np.ndarray) -> np.ndarray:
  return a.astype(np.float32).astype(np.float64)


def _quiet_part_update(
    k: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Returns (y, x, c) for a complex128 update of x.

  x.real is near 1e100 with a 10% update, and x.imag is near 1 with an
  update of up to k float32 steps. c rounds y.real to float32 and keeps
  x.imag.

  Args:
    k: Size of the imaginary update in float32 steps of x.imag.
  """
  rng = np.random.default_rng(0)
  n = 4096
  xr = _round_f32(rng.standard_normal(n)) * 2.0**332
  xi = _round_f32(rng.standard_normal(n))
  step = np.ldexp(1.0, np.frexp(xi)[1] - 24)
  ref = xr * (1.0 + 0.1 * rng.standard_normal(n)) + 1j * (
      xi + k * step * np.clip(rng.standard_normal(n), -1.0, 1.0)
  )
  cand = _round_f32(ref.real / 2.0**332) * 2.0**332 + 1j * xi
  return ref, xr + 1j * xi, cand


def _noisy_quiet_part_update(seed: int) -> tuple[np.ndarray, ...]:
  """Returns (y, x, c) for a complex128 update with noise on the quiet part.

  x.real is near 1e100 with a 10% update, and x.imag is near 1 with an
  update near 1e-3. c keeps y.real and adds independent noise 100 times the
  imaginary update to y.imag.

  Args:
    seed: The random seed.
  """
  rng = np.random.default_rng(seed)
  n = 4096
  xr = _round_f32(rng.standard_normal(n)) * 2.0**332
  xi = rng.standard_normal(n)
  ref = xr * (1.0 + 0.1 * rng.standard_normal(n)) + 1j * (
      xi + 1e-3 * rng.standard_normal(n)
  )
  cand = ref + 1j * (0.1 * rng.standard_normal(n))
  return ref, xr + 1j * xi, cand


def _bf16_view_update(
    seed: int, gain: float
) -> tuple[np.ndarray, np.ndarray, sqnr.PrecisionSpec]:
  """Returns (y, x, precision) for y = gain * x, with x on the bf16 grid."""
  rng = np.random.default_rng(seed)
  x = rng.standard_normal((256, 256)).astype(_BF16)
  ref = (gain * x.astype(np.float32)).astype(np.float32)
  spec = sqnr.PrecisionSpec(
      operand_dtypes=("bfloat16",), output_dtype="bfloat16"
  )
  return ref, x, spec


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

  def test_residual_update_with_gain_error_and_noise_passes(self):
    # An approximate update: gain 0.8 on y - x plus independent noise 10 dB
    # below the update.
    for seed in _SEEDS:
      rng = np.random.default_rng(seed)
      x = rng.standard_normal((256, 256)).astype(np.float32)
      upd = np.tanh(rng.standard_normal((256, 256)))
      upd *= 10.0 ** (-50.0 / 20.0) / float(np.std(upd))
      ref = (x.astype(np.float64) + upd).astype(np.float32)
      noise = 0.3 * float(np.std(upd)) * rng.standard_normal(x.shape)
      cand = (x + 0.8 * upd + noise).astype(np.float32)
      report = _audit(ref, cand, views=(x,), min_sqnr_db=40.0)
      self.assertTrue(report.passed, (seed, report.failed_reasons))

  @parameterized.parameters((6, 6), (12, 12))
  def test_small_honest_residual_update_passes(self, *shape):
    for seed in _SEEDS:
      rng = np.random.default_rng(seed)
      x = rng.standard_normal(shape).astype(np.float32)
      upd = np.tanh(rng.standard_normal(shape))
      upd *= 10.0 ** (-50.0 / 20.0) / float(np.std(upd))
      ref = (x.astype(np.float64) + upd).astype(np.float32)
      for cand in (ref.copy(), ref.astype(_BF16).astype(np.float32)):
        report = _audit(ref, cand, views=(x,), min_sqnr_db=40.0)
        self.assertTrue(report.passed, (seed, report.failed_reasons))

  def test_single_binade_cast_passes(self):
    for seed in _SEEDS:
      rng = np.random.default_rng(seed)
      sign = rng.choice([-1.0, 1.0], (256, 256))
      ref = ((1.0 + rng.random((256, 256))) * sign).astype(np.float32)
      report = _audit(ref, ref.astype(_BF16), _BF16_OUT)
      self.assertTrue(report.passed, (seed, report.failed_reasons))

  def test_exact_separable_broadcast_passes(self):
    # The float32 reference holds only its own rounding of row + col. The
    # differenced transforms remove the separable part, so their AC is that
    # rounding, which the reference dtype cannot resolve.
    for seed in _SEEDS:
      rng = np.random.default_rng(seed)
      row = rng.standard_normal((256, 1))
      col = rng.standard_normal((1, 256))
      ref = row.astype(np.float32) + col.astype(np.float32)
      report = _audit(ref, row + col)
      self.assertTrue(report.passed, (seed, report.failed_reasons))

  def test_exact_near_constant_sums_pass_at_float64_precision(self):
    # Each reference element is a float32 sum of weights that add up to 1.
    # The candidate is the exact sum. The AC of the reference is its own
    # rounding, so no window is resolved, also when v is at float64.
    for seed in _SEEDS:
      rng = np.random.default_rng(seed)
      p = rng.dirichlet(np.ones(64), size=(256, 256))
      ref = p.astype(np.float32).sum(-1)
      report = _audit(
          ref, np.ones((256, 256)), sqnr.PrecisionSpec(output_dtype="float64")
      )
      self.assertTrue(report.passed, (seed, report.failed_reasons))

  @parameterized.parameters("bfloat16", "float16", "mxfp4")
  def test_tiny_complex_with_non_finite_parts_does_not_raise(self, out):
    # Every nonzero element has a NaN part, so no element enters the fit. The
    # zero-block variance then comes from the subnormal step of the output
    # format, far outside the range of the 1e-200 leaf scale.
    rng = np.random.default_rng(0)
    ref = np.empty((64, 64), np.complex128)
    ref.real = 1e-200 * rng.standard_normal((64, 64))
    ref.imag = np.nan
    ref[0, :8] = 0.0
    cand = ref.copy()
    cand.real *= 1.0 + 1e-3
    report = _audit(ref, cand, sqnr.PrecisionSpec(output_dtype=out))
    self.assertNotIn("max_block_noise_ratio_db", _checks(report))

  def test_tiny_error_in_a_zero_block_gives_no_finite_block_sqnr(self):
    # A block with y = 0 and e != 0 has a block SQNR of -inf. The report
    # field is then None, not a large negative number such as -2842 dB.
    rng = np.random.default_rng(0)
    ref = rng.standard_normal((64, 256)).astype(np.float32)
    ref[:, :32] = 0.0
    cand = ref.astype(_BF16).astype(np.float32)
    cand[3, 5] = 1e-7
    report = _audit(ref, cand, _BF16_OUT)
    self.assertTrue(report.passed, report.failed_reasons)
    self.assertIsNone(report.min_block_sqnr_db)

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

  @parameterized.parameters((0,), (4, 0), (0, 4))
  def test_empty_shard_next_to_a_full_shard_passes(self, *empty_shape):
    ref = np.random.default_rng(0).standard_normal((64, 256)).astype(np.float32)
    cand = ref.astype(_BF16).astype(np.float32)
    empty = np.zeros(empty_shape, np.float32)
    for shards in (
        [(ref, cand), (empty, empty)],
        [(empty, empty), (ref, cand)],
    ):
      report = _audit(None, None, _BF16_OUT, shards=shards)
      self.assertGreater(report.max_error_ulp, 0.0)
      self.assertTrue(report.passed, (empty_shape, report.failed_reasons))

  def test_complex_with_a_non_finite_part_in_every_element_passes(self):
    # Each element has a finite real part and a NaN imaginary part, so the
    # leaf has finite components but no fully finite element.
    ref = np.full((32, 32), complex(1.0, np.nan), np.complex64)
    cand = ref.copy()
    cand.real[3, 4] = 1.0 + 2.0**-20
    report = _audit(ref, cand)
    self.assertTrue(report.passed, report.failed_reasons)

  @parameterized.parameters(*_SEEDS)
  def test_tiny_outputs_flushed_to_zero_pass(self, seed):
    # Sparse outputs near 1e3 and a band of outputs near 1e-30, as in a
    # matmul of inputs at the float32 range limits. A fixed-point candidate
    # with step 1 flushes the band to 0. The band AC is far below the
    # resolution of the leaf, so its windows must not be tested.
    ref = _band_leaf(seed, 1e-30)
    report = _audit(ref, np.round(ref), min_sqnr_db=30.0)
    self.assertTrue(report.passed, report.failed_reasons)

  @parameterized.parameters(*_SEEDS)
  def test_fp8_cast_of_a_row_below_half_a_step_passes(self, seed):
    # Row 5 is near 1e-4, below half the float8_e4m3fn subnormal step
    # (2**-9), so the cast sets it to 0. That row holds no AC that the
    # candidate must keep.
    ref = _quiet_row_leaf(seed)
    cand = ref.astype(ml_dtypes.float8_e4m3fn).astype(np.float32)
    report = _audit(ref, cand, _FP8_OUT)
    self.assertTrue(report.passed, report.failed_reasons)

  @parameterized.parameters((1,), (2,))
  def test_complex_part_one_step_from_the_view_passes(self, steps):
    # The imaginary part of y is 1 or 2 steps from x, below the resolution
    # of the reference dtype. The candidate keeps x there. Its gain on that
    # part holds no information and must not be tested.
    ref, x = _one_part_update(steps)
    cand = ref.copy()
    cand.imag = x.imag
    report = _audit(ref, cand, views=(x,))
    self.assertTrue(report.passed, report.failed_reasons)

  def test_float32_rounding_of_a_float64_complex_update_passes(self):
    # The real part is near 1e100 and the imaginary part near 1. The prescaled
    # imaginary sums are then near 1e-200, and a product of two of them
    # underflows to 0. The imaginary update is below half a float32 step, so
    # the float32 rounding keeps x there.
    ref, x, cand = _quiet_part_update(0.2)
    report = _audit(ref, cand, _F32_OUT, views=(x,))
    self.assertTrue(report.passed, report.failed_reasons)

  @parameterized.parameters(*_SEEDS)
  def test_noise_on_a_quiet_complex_part_passes_the_gain_test(self, seed):
    # The noise on the imaginary part is independent of the update, so the
    # gain bound must hold its random-sign term. The squares of the prescaled
    # imaginary products near 1e-200 underflow to 0 in float64.
    ref, x, cand = _noisy_quiet_part_update(seed)
    report = _audit(ref, cand, views=(x,))
    self.assertTrue(report.passed, report.failed_reasons)

  @parameterized.parameters(*_SEEDS)
  def test_relu_leaf_with_one_loud_row_passes(self, seed):
    # Most rows are ReLU outputs near 0.5 with many exact zeros. Row 0 is
    # near 1500, so the other rows sit far below the leaf scale.
    ref = _relu_leaf(seed)
    report = _audit(ref, ref.astype(_BF16), _BF16_OUT)
    self.assertTrue(report.passed, report.failed_reasons)

  @parameterized.parameters(*_SEEDS)
  def test_mxfp4_relu_leaf_with_zero_blocks_passes(self, seed):
    # The layout of test_relu_leaf_with_one_loud_row_passes, where most
    # 32-element blocks of the quiet rows are all zeros.
    ref = _zero_block_relu_leaf(seed)
    report = _audit(ref, _mxfp4_round(ref), _MXFP4_OUT)
    self.assertTrue(report.passed, report.failed_reasons)

  @parameterized.parameters(*_SEEDS)
  def test_rounding_of_a_small_update_to_the_input_passes(self, seed):
    # y = 1.0019 x moves each element by less than half a bf16 step of x,
    # so round(y) = x on every element. With 3 elements at 1.006, a few
    # elements move and the rest stay at x.
    ref, x, spec = _bf16_view_update(seed, 1.0019)
    cand = ref.astype(_BF16)
    self.assertTrue(np.array_equal(cand, x))
    report = _audit(ref, cand, spec, views=(x,))
    self.assertTrue(report.passed, report.failed_reasons)
    ref.flat[:3] = (1.006 * x.flat[:3].astype(np.float32)).astype(np.float32)
    cand = ref.astype(_BF16)
    self.assertEqual(np.count_nonzero(cand != x), 3)
    report = _audit(ref, cand, spec, views=(x,))
    self.assertTrue(report.passed, report.failed_reasons)

  @parameterized.product(rows=(64, 256), seed=_SEEDS)
  def test_cast_of_a_staircase_with_negative_diff_correlation_passes(
      self, rows, seed
  ):
    # Each row is 1 + u * (0.45, 0.55, 1.45, 1.55, ...) plus a whole-step
    # offset, with u the bf16 step in [1, 2). The reference differences
    # alternate 0.1 u and 0.9 u, and the rounded ones alternate u and 0, so
    # the diff transform has a correlation near -1. The cast is correctly
    # rounded, so the sign test must not apply to the diff transform.
    u = 2.0**-7
    cols = 128
    base = np.array([0.45, 0.55] * (cols // 2)) + np.repeat(
        np.arange(cols // 2), 2
    )
    off = np.random.default_rng(seed).integers(0, 8, size=(rows, 1))
    ref = (1.0 + u * (base[None, :] + off)).astype(np.float32)
    report = _audit(ref, ref.astype(_BF16), _BF16_OUT)
    self.assertTrue(report.passed, report.failed_reasons)

  def test_metrics_are_reported(self):
    ref = np.random.default_rng(0).standard_normal((64, 256)).astype(np.float32)
    report = _audit(ref, ref.astype(_BF16), _BF16_OUT)
    self.assertLessEqual(report.max_error_ulp, 0.5)
    self.assertIsNotNone(report.max_block_noise_ratio_db)
    self.assertIsNotNone(report.min_block_sqnr_db)
    self.assertIsNotNone(report.min_ac_snr_db)
    self.assertIsNotNone(report.centered_snr_db)

  @parameterized.parameters(
      "bfloat16_x_bfloat16",
      "bfloat16_x_float8_e5m2",
      "float8_e4m3fn_x_float8_e4m3fn",
      "bfloat16_x_mxfp4",
  )
  def test_per_channel_outlier_matmul_passes(self, kernel):
    # The output noise of each column scales with its energy, which is about
    # 50**2 times larger in the outlier columns of w.
    for seed in _SEEDS:
      x, w = _per_channel_outlier_operands(seed)
      ref = (x.astype(np.float64) @ w.astype(np.float64)).astype(np.float32)
      cand, precision = _outlier_matmul(kernel, x, w)
      report = _audit(ref, cand, precision)
      self.assertTrue(report.passed, (seed, report.failed_reasons))

  @parameterized.named_parameters(*_HEAVY_TAILED_CASES)
  def test_heavy_tailed_matmul_passes(self, kernel, floor, kind, seed):
    # In some elements, one product x_ik w_kj holds most of the energy
    # sum_k x_ik**2 w_kj**2. The error of such an element is the input
    # rounding of that product, about eps_in |y|, far above C h with h the
    # row-column mean energy. The checks use C max(h, y**2) for it.
    x, w = _heavy_tailed_operands(kind, seed)
    ref, cand = _narrow_matmul(kernel, x, w)
    self.assertGreater(np.count_nonzero(_dominated(ref)), 0)
    report = _validator_audit(ref, cand, floor)
    self.assertTrue(report.passed, report.failed_reasons)

  @parameterized.parameters(
      ("bfloat16_x_bfloat16", _BF16_X_BF16_FLOOR),
      ("bfloat16_x_float8_e5m2", _BF16_X_E5M2_FLOOR),
  )
  def test_boundary_probe_matmul_passes(self, kernel, floor):
    # The outputs are 3e8 and 4e8 (one probe product each, so y**2 is far
    # above h), values near 4e-41 and 4e-34 that the kernel flushes to zero
    # (e = -y), and zeros.
    ref, cand = _narrow_matmul(kernel, *_boundary_operands())
    report = _validator_audit(ref, cand, floor)
    self.assertTrue(report.passed, report.failed_reasons)

  @parameterized.parameters(
      "bfloat16_x_bfloat16",
      "bfloat16_x_float8_e5m2",
      "float8_e4m3fn_x_float8_e4m3fn",
      "bfloat16_x_mxfp4",
  )
  def test_per_channel_outlier_matmul_with_variable_length_rows_passes(
      self, kernel
  ):
    # Each row is NaN after its own length in both outputs, as with padding
    # of variable-length sequences. The column means of the energy term are
    # over the finite elements, and the row factor is over the finite columns.
    for seed in _SEEDS:
      x, w = _per_channel_outlier_operands(seed)
      ref = (x.astype(np.float64) @ w.astype(np.float64)).astype(np.float32)
      cand, precision = _outlier_matmul(kernel, x, w)
      lens = np.random.default_rng(seed).integers(8, ref.shape[1], ref.shape[0])
      pad = np.arange(ref.shape[1]) >= lens[:, None]
      ref[pad] = np.nan
      cand[pad] = np.nan
      report = _audit(ref, cand, precision)
      self.assertTrue(report.passed, (seed, report.failed_reasons))

  @parameterized.parameters(
      "bfloat16_x_bfloat16",
      "bfloat16_x_float8_e5m2",
      "float8_e4m3fn_x_float8_e4m3fn",
      "bfloat16_x_mxfp4",
  )
  def test_one_row_shard_next_to_a_per_channel_outlier_shard_passes(
      self, kernel
  ):
    # The one-row shard has no energy term. It is graded with the fit
    # without the energy term, which keeps the noise of the large columns in
    # B. With 50x outlier columns, that one B can still be too small for the
    # large columns of such a shard.
    for seed in _SEEDS:
      x, w = _per_channel_outlier_operands(seed, outlier_scale=8.0)
      ref = (x.astype(np.float64) @ w.astype(np.float64)).astype(np.float32)
      cand, precision = _outlier_matmul(kernel, x, w)
      shards = [(ref[:-1], cand[:-1]), (ref[-1:], cand[-1:])]
      report = _audit(None, None, precision, shards=shards)
      self.assertTrue(report.passed, (seed, report.failed_reasons))


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

  def test_noise_on_a_zero_reference_block_is_caught_by_block_noise(self):
    # The reference block is all zeros and the other elements are exact bf16
    # roundings, so the fit gives B = 0. The block must still be graded.
    rng = np.random.default_rng(0)
    ref = rng.standard_normal((256, 1024)).astype(np.float32)
    ref[1, 32:64] = 0.0
    cand = ref.astype(_BF16).astype(np.float32)
    cand[1, 32:64] = 1e-3 * rng.choice([-1.0, 1.0], 32)
    report = _audit(ref, cand, _BF16_OUT)
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
    # ReLU output: reference zeros have v = 0 and carry small noise. The fit
    # leaves them out, so the small noise on the nonzero elements gives
    # B > 0. One larger error at a reference zero has the largest |e| / sigma
    # in the tile but no step to compare with. It must not hide the 4 ulp
    # spike on a nonzero element.
    rng = np.random.default_rng(0)
    x = rng.standard_normal((64, 64)).astype(np.float32)
    ref = np.maximum(x, 0).astype(_BF16).astype(np.float32)
    zero = ref == 0
    cand = ref.copy()
    cand[zero] = 1e-4 * rng.standard_normal(np.count_nonzero(zero))
    cand[~zero] += 1e-4 * rng.standard_normal(np.count_nonzero(~zero))
    cand[tuple(np.argwhere(zero)[3])] = 2e-3
    # The same leaf without the 4 ulp spike passes: the error at the zero
    # must not count as a spike by itself.
    self.assertTrue(_audit(ref, cand, _BF16_OUT).passed)
    spot = tuple(np.argwhere(~zero)[5])
    cand[spot] += np.float32(4.0 * float(np.spacing(ref[spot])) * 2**16)
    report = sqnr.audit_leaf(
        [(ref, cand)], precision=sqnr.PrecisionSpec(output_dtype="bfloat16")
    )
    self._assert_caught(report, "max_error_ulp")

  @parameterized.parameters(0.07, 0.1)
  def test_noise_on_quiet_float64_windows_is_caught(self, gain):
    # One O(1) element sets the leaf scale, so the other prescaled values are
    # near 1e-100 and their window sums near 1e-197. The product syy * scc
    # of such sums underflows to 0 in float64.
    for seed in (0, 1):
      rng = np.random.default_rng(seed)
      ref = 1e-100 * rng.standard_normal((64, 1024))
      ref[0, 0] = 1.0
      cand = gain * ref + 0.5e-100 * rng.standard_normal((64, 1024))
      cand[0, 0] = 1.0
      self._assert_caught(_audit(ref, cand), "centered_corr_snr_db")

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

  @parameterized.parameters((32, 4, 128), (16, 32, 32), (2, 16, 4, 128))
  def test_separable_bias_with_leading_axes_is_caught_by_plaquette(
      self, *shape
  ):
    # The plaquette must stay inside each (rows, columns) plane. A plaquette
    # across two planes sees the difference of their row biases.
    rng = np.random.default_rng(0)
    rb = rng.standard_normal(shape[:-1] + (1,))
    cb = rng.standard_normal(shape[:-2] + (1, shape[-1]))
    ref = (rb + cb + 0.02 * rng.standard_normal(shape)).astype(np.float32)
    honest = ref * (1.0 + 2.0**-24 * rng.standard_normal(shape))
    self.assertTrue(_audit(ref, honest.astype(np.float32)).passed)
    cand = (rb + cb).astype(np.float32)
    report = _audit(ref, cand, min_sqnr_db=30.0)
    self._assert_caught(report, "plaquette_corr_snr_db")

  @parameterized.parameters((32, 4, 128), (16, 32, 32))
  def test_column_mean_with_leading_axes_is_caught_by_diff(self, *shape):
    rng = np.random.default_rng(0)
    cb = rng.standard_normal(shape[:-2] + (1, shape[-1]))
    ref = (10.0 * cb + rng.standard_normal(shape)).astype(np.float32)
    honest = ref * (1.0 + 2.0**-24 * rng.standard_normal(shape))
    self.assertTrue(_audit(ref, honest.astype(np.float32)).passed)
    cand = np.broadcast_to(np.mean(ref, -2, keepdims=True), shape)
    report = _audit(ref, cand.astype(np.float32), min_sqnr_db=15.0)
    self._assert_caught(report, "diff_corr_snr_db")

  def test_smooth_detail_dropped_from_a_separable_leaf_is_caught(self):
    # The cheat drops a smooth detail 34 dB below the leaf. Its error raises
    # the fitted B. The correlation gate must not follow B, or the windows
    # that show the loss are not tested.
    rng = np.random.default_rng(3)
    rb = rng.standard_normal((128, 1))
    cb = rng.standard_normal((1, 128))
    detail = rng.standard_normal((128, 128))
    kernel = np.exp(-0.5 * (np.arange(-12, 13) / 4.0) ** 2)
    for axis in (0, 1):
      detail = np.apply_along_axis(
          lambda r: np.convolve(r, kernel, mode="same"), axis, detail
      )
    ref = (rb + cb + 0.02 * detail / np.std(detail)).astype(np.float32)
    rms = float(np.sqrt(np.mean(np.square(ref.astype(np.float64)))))
    noise = rms * 10.0 ** (-37.0 / 20.0) * rng.standard_normal(ref.shape)
    honest = (ref + noise).astype(np.float32)
    self.assertTrue(_audit(ref, honest, min_sqnr_db=30.0).passed)
    cand = (rb + cb).astype(np.float32)
    report = _audit(ref, cand, min_sqnr_db=30.0)
    self._assert_caught(report, "plaquette_corr_snr_db")

  def test_dropped_band_above_the_resolution_is_caught_by_centered(self):
    # The layout of test_tiny_outputs_flushed_to_zero_pass, with the band
    # near 10. The band is now above the resolution and must be kept.
    ref = _band_leaf(0, 10.0)
    self.assertTrue(_audit(ref, np.round(ref), min_sqnr_db=30.0).passed)
    cand = np.round(ref)
    cand[:, 64:72] = 0.0
    report = _audit(ref, cand, min_sqnr_db=30.0)
    self._assert_caught(report, "centered_corr_snr_db")

  @parameterized.parameters(*_SEEDS)
  def test_zeroed_relu_rows_are_caught_by_centered(self, seed):
    # The twin of test_relu_leaf_with_one_loud_row_passes. The cheat keeps
    # row 0 and sets the other rows to 0. The global SQNR passes. A zero
    # reference has no rounding binade, so it must not widen the noise
    # bound of the windows.
    ref = _relu_leaf(seed)
    cand = ref.astype(_BF16).astype(np.float32)
    cand[1:] = 0.0
    report = _audit(ref, cand, _BF16_OUT)
    self._assert_caught(report, "centered_corr_snr_db")

  @parameterized.parameters(*_SEEDS)
  def test_zeroed_relu_rows_are_caught_with_a_block_scaled_output(self, seed):
    # The twin of test_mxfp4_relu_leaf_with_zero_blocks_passes. The cheat
    # keeps row 0 and sets the other rows to 0. A block of 32 zeros has no
    # scale, so it must not get a scale binade that widens the window noise
    # bound.
    ref = _zero_block_relu_leaf(seed)
    cand = ref.copy()
    cand[1:] = 0.0
    report = _audit(ref, cand, _MXFP4_OUT)
    self._assert_caught(report, "centered_corr_snr_db")

  @parameterized.named_parameters(
      ("quiet_row", np.s_[5], 2.0**-6),
      ("normal_block", np.s_[9, 64:96], None),
  )
  def test_fp8_cheat_next_to_a_quiet_row_is_caught_by_block_noise(
      self, where, value
  ):
    # The layout of test_fp8_cast_of_a_row_below_half_a_step_passes. An
    # error of 8 subnormal steps on the quiet row, or a sign flip of a block
    # of normal values.
    ref = _quiet_row_leaf(0)
    cand = ref.astype(ml_dtypes.float8_e4m3fn).astype(np.float32)
    cand[where] = -cand[where] if value is None else value
    report = _audit(ref, cand, _FP8_OUT)
    self.assertFalse(report.passed)
    self.assertIn("max_block_noise_ratio_db", _checks(report))

  def test_extra_noise_on_quiet_float64_blocks_is_caught_by_block_noise(self):
    # Blocks near 1e-100 next to one O(1) element. The block sums of s2**2
    # underflow to 0, so the effective count must not use them directly.
    rng = np.random.default_rng(0)
    ref = 1e-100 * rng.standard_normal((64, 256))
    ref[0] = 0.0
    ref[0, 0] = 1.0
    err = ref * 2.0**-53 * rng.standard_normal(ref.shape)
    self.assertTrue(_audit(ref, ref + err).passed)
    cand = ref + err
    cand[5, 32:64] += 3.0 * err[5, 32:64]
    self._assert_caught(_audit(ref, cand), "max_block_noise_ratio_db")

  def test_ignored_update_on_one_complex_part_is_caught_by_input_delta(self):
    # The twin of test_complex_part_one_step_from_the_view_passes with an
    # update of 64 steps, above the resolution.
    ref, x = _one_part_update(64)
    cand = ref.copy()
    cand.imag = x.imag
    report = _audit(ref, cand, views=(x,), min_sqnr_db=30.0)
    self._assert_caught(report, "min_delta_sqnr_db")
    # The gain test must catch it. The correlation test uses the same name.
    self.assertTrue(
        any("(gain 0 " in r for r in report.failed_reasons),
        report.failed_reasons,
    )

  def test_ignored_quiet_complex_update_is_caught_by_input_delta(self):
    # The twin of test_float32_rounding_of_a_float64_complex_update_passes
    # with an imaginary update of 8 steps.
    ref, x, cand = _quiet_part_update(8.0)
    report = _audit(ref, cand, _F32_OUT, views=(x,))
    self._assert_caught(report, "min_delta_sqnr_db")
    self.assertTrue(
        any("(gain 0 " in r for r in report.failed_reasons),
        report.failed_reasons,
    )

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

  @parameterized.parameters(*_SEEDS)
  def test_bf16_passthrough_above_half_a_step_is_caught(self, seed):
    # The twin of test_rounding_of_a_small_update_to_the_input_passes, with
    # y = 1.003 x: the update is above half a step on part of the leaf. The
    # global SQNR passes, so the input-delta test must catch return x, and
    # also return x with 30 elements updated.
    ref, x, spec = _bf16_view_update(seed, 1.003)
    report = _audit(ref, x, spec, views=(x,))
    self._assert_caught(report, "min_delta_sqnr_db")
    self.assertTrue(
        any(
            "candidate equals an input view" in r for r in report.failed_reasons
        ),
        report.failed_reasons,
    )
    cand = x.copy()
    cand.flat[:30] = ref.flat[:30].astype(_BF16)
    report = _audit(ref, cand, spec, views=(x,))
    self._assert_caught(report, "min_delta_sqnr_db")

  def test_one_step_block_in_a_single_binade_is_caught_by_block_noise(self):
    # All elements are in [1, 2), so every element has the same rounding
    # variance v. The fit must hold A at its floor and keep B for the noise
    # beyond v. Otherwise sigma**2 = 2 v and the block of one-step errors
    # falls inside the bound.
    for seed in _SEEDS:
      rng = np.random.default_rng(seed)
      sign = rng.choice([-1.0, 1.0], (256, 256))
      ref = ((1.0 + rng.random((256, 256))) * sign).astype(np.float32)
      cand = ref.astype(_BF16).astype(np.float32)
      step = np.float32(2.0**-7) * rng.choice([-1.0, 1.0], 32)
      cand[3, 32:64] = ref[3, 32:64] + step.astype(np.float32)
      report = _audit(ref, cand, _BF16_OUT)
      self._assert_caught(report, "max_block_noise_ratio_db")

  @parameterized.named_parameters(
      ("attenuated", 0.01),
      ("wrong_sign", -0.5),
      ("over_scale", 2.0),
      ("noise", None),
  )
  def test_wrong_update_gain_is_caught_by_input_delta(self, gain):
    for seed in _SEEDS:
      rng = np.random.default_rng(seed)
      x = rng.standard_normal((256, 256)).astype(np.float32)
      upd = np.tanh(rng.standard_normal((256, 256)))
      upd *= 10.0 ** (-50.0 / 20.0) / float(np.std(upd))
      ref = (x.astype(np.float64) + upd).astype(np.float32)
      d = ref.astype(np.float64) - x
      if gain is None:
        cand = x + rng.standard_normal(x.shape) * float(np.std(d))
      else:
        cand = x + gain * d
      report = _audit(
          ref, cand.astype(np.float32), views=(x,), min_sqnr_db=40.0
      )
      self._assert_caught(report, "min_delta_sqnr_db")

  @parameterized.named_parameters(
      ("attenuated", 0.01, False),
      ("wrong_sign", -0.5, False),
      ("over_scale", 2.0, False),
      ("noise_scaled_by_x", 1.0, True),
  )
  def test_wrong_multiplicative_update_is_caught_by_input_delta(
      self, gain, noise
  ):
    # y = x (1 + s). The error of the cheat scales with |x|, so it goes into
    # the fitted A. The gain bound must not use that A.
    s = 10.0 ** (-50.0 / 20.0)
    for seed in _SEEDS:
      rng = np.random.default_rng(seed)
      x = rng.standard_normal((256, 256)).astype(np.float32)
      ref = (x.astype(np.float64) * (1.0 + s)).astype(np.float32)
      if noise:
        cand = x * (1.0 + s * rng.standard_normal(x.shape))
      else:
        cand = x * (1.0 + gain * s)
      report = _audit(ref, cand.astype(np.float32), views=(x,))
      self._assert_caught(report, "min_delta_sqnr_db")

  def test_honest_multiplicative_update_passes(self):
    s = 10.0 ** (-50.0 / 20.0)
    for seed in _SEEDS:
      rng = np.random.default_rng(seed)
      x = rng.standard_normal((256, 256)).astype(np.float32)
      ref = (x.astype(np.float64) * (1.0 + s)).astype(np.float32)
      report = _audit(ref, x * np.float32(1.0 + s), views=(x,))
      self.assertTrue(report.passed, (seed, report.failed_reasons))

  @parameterized.named_parameters(
      ("constant_update", (256, 256), True),
      ("leaf_of_144", (12, 12), False),
      ("leaf_of_36", (6, 6), False),
      ("leaf_of_25", (5, 5), False),
  )
  def test_passthrough_plus_eps_is_caught_by_input_delta(self, shape, const):
    # The gain test needs no variance in y - x. On small leaves the sign
    # count catches the cheat, because every d e has the same sign.
    for seed in _SEEDS:
      rng = np.random.default_rng(seed)
      x = rng.standard_normal(shape).astype(np.float32)
      if const:
        upd = np.full(shape, 10.0 ** (-50.0 / 20.0))
      else:
        upd = np.tanh(rng.standard_normal(shape))
        upd *= 10.0 ** (-50.0 / 20.0) / float(np.std(upd))
      ref = (x.astype(np.float64) + upd).astype(np.float32)
      report = _audit(ref, x + np.float32(1e-6), views=(x,), min_sqnr_db=40.0)
      self._assert_caught(report, "min_delta_sqnr_db")

  @parameterized.named_parameters(
      ("big_column_times_1_02", "big", "times_1_02"),
      ("big_column_times_1_1", "big", "times_1_1"),
      ("big_column_zeroed", "big", "zeroed"),
      ("big_column_biased", "big", "biased"),
      ("big_column_32_rows_zeroed", "big", "rows_zeroed"),
      ("block_in_big_column_zeroed", "big", "block_zeroed"),
      ("column_times_1_1", "median", "times_1_1"),
      ("column_zeroed", "median", "zeroed"),
      ("column_biased", "median", "biased"),
  )
  def test_per_channel_outlier_column_cheat_is_caught(self, column, cheat):
    for seed in _SEEDS:
      x, w = _per_channel_outlier_operands(seed)
      ref = (x.astype(np.float64) @ w.astype(np.float64)).astype(np.float32)
      cand, precision = _outlier_matmul("bfloat16_x_bfloat16", x, w)
      col_max = np.max(np.abs(w), axis=0)
      if column == "big":
        j = int(np.argmax(col_max))
      else:
        j = int(np.argmin(np.abs(col_max - np.median(col_max))))
      if cheat == "times_1_02":
        cand[:, j] *= 1.02
      elif cheat == "times_1_1":
        cand[:, j] *= 1.1
      elif cheat == "zeroed":
        cand[:, j] = 0.0
      elif cheat == "biased":
        cand[:, j] += 0.05 * np.sqrt(np.mean(np.square(ref[:, j], dtype=float)))
      elif cheat == "rows_zeroed":
        cand[64:96, j] = 0.0
      else:
        cand[3, j // 32 * 32 : j // 32 * 32 + 32] = 0.0
      report = _audit(ref, cand, precision)
      self.assertFalse(report.passed, seed)
      if column == "big":
        # A big column holds most of the leaf energy, and the energy term of
        # the fit sets C from the largest columns. So C takes in a gain error
        # there, and only the global SQNR floor bounds it: with a 1.02 gain,
        # seed 2 fails only sqnr_db (46.2 dB against the 46.6 dB floor).
        self.assertNotEmpty(_checks(report), seed)
      else:
        self.assertNotEmpty(_checks(report) - {"sqnr_db"}, seed)

  def test_column_cheats_are_caught_next_to_zero_padded_columns(self):
    # 64 output columns, zero-padded to 2048. The padded columns have no
    # energy, so they must not change the energy term of the other columns.
    # They add zero blocks, and more blocks raise the block threshold, so the
    # padded leaf can fail fewer checks: on seed 2 the 1.1 gain is caught by
    # the block check at 64 columns, and only at 64 columns.
    fit_energy = sqnr._dc_fit_energy

    def audit_and_fits(ref, cand, precision):
      fits = []

      def record(*args):
        fits.append(fit_energy(*args))
        return fits[-1]

      with mock.patch.object(sqnr, "_dc_fit_energy", side_effect=record):
        return _audit(ref, cand, precision), fits

    for seed in _SEEDS:
      x, w = _per_channel_outlier_operands(seed)
      w = w[:, :64]
      ref = (x.astype(np.float64) @ w.astype(np.float64)).astype(np.float32)
      col_max = np.max(np.abs(w), axis=0)
      j = int(np.argmin(np.abs(col_max - np.median(col_max))))
      for kernel, cheat in (
          ("bfloat16_x_bfloat16", "times_1_1"),
          ("bfloat16_x_float8_e5m2", "rows_zeroed"),
      ):
        cand, precision = _outlier_matmul(kernel, x, w)
        if cheat == "times_1_1":
          cand[:, j] *= 1.1
        else:
          cand[64:96, j] = 0.0
        report, fits = audit_and_fits(ref, cand, precision)
        padded, padded_fits = audit_and_fits(
            _zero_pad_columns(ref, 2048),
            _zero_pad_columns(cand, 2048),
            precision,
        )
        self.assertNotEmpty(fits)
        np.testing.assert_allclose(padded_fits, fits, rtol=1e-9)
        # The zero blocks add no noise, so the worst block ratio stays the
        # same. Only the block threshold moves.
        block_db = report.max_block_noise_ratio_db
        padded_block_db = padded.max_block_noise_ratio_db
        assert block_db is not None and padded_block_db is not None
        self.assertAlmostEqual(
            padded_block_db, block_db, places=9, msg=(seed, kernel, cheat)
        )
        self.assertFalse(report.passed, (seed, kernel, cheat))
        self.assertLessEqual(
            _checks(padded), _checks(report), (seed, kernel, cheat)
        )
        if cheat == "rows_zeroed":
          self._assert_caught(padded, "max_block_noise_ratio_db")

  @parameterized.named_parameters(*_HEAVY_TAILED_CASES)
  def test_heavy_tailed_dominant_element_noise_is_caught(
      self, kernel, floor, kind, seed
  ):
    # Four times the honest error on the elements with one dominant product.
    # These elements are few, so the global SQNR stays above its floor. The
    # block-noise check grades them against C max(h, y**2) and must catch it.
    x, w = _heavy_tailed_operands(kind, seed)
    ref, cand = _narrow_matmul(kernel, x, w)
    dom = _dominated(ref)
    err = cand[dom].astype(np.float64) - ref[dom]
    cand[dom] = (ref[dom] + 4.0 * err).astype(np.float32)
    report = _validator_audit(ref, cand, floor)
    self._assert_caught(report, "max_block_noise_ratio_db")

  @parameterized.parameters(np.float64, np.float32)
  def test_inverted_ac_on_a_dc_dominated_leaf_is_caught_by_centered(
      self, dtype
  ):
    # The error 2 (y - mean) has 4 times the AC energy, so the window gate on
    # the error energy skips each window. A significantly negative
    # correlation must still admit it.
    for seed in _SEEDS:
      rng = np.random.default_rng(seed)
      ref = 1000.0 + rng.standard_normal((64, 128))
      honest = ref + 1e-6 * rng.standard_normal(ref.shape)
      report = _audit(ref.astype(dtype), honest.astype(dtype))
      self.assertTrue(report.passed, (seed, report.failed_reasons))
      report = _audit(ref.astype(dtype), (2000.0 - ref).astype(dtype))
      self._assert_caught(report, "centered_corr_snr_db")

  def test_conjugated_quiet_imaginary_part_is_caught_by_centered(self):
    for seed in _SEEDS:
      rng = np.random.default_rng(seed)
      shape = (64, 128)
      ref = rng.standard_normal(shape) + 1e-3j * rng.standard_normal(shape)
      noise = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
      honest = (ref + 1e-8 * noise).astype(np.complex64)
      ref = ref.astype(np.complex64)
      report = _audit(ref, honest)
      self.assertTrue(report.passed, (seed, report.failed_reasons))
      self._assert_caught(_audit(ref, np.conj(ref)), "centered_corr_snr_db")

  def test_update_with_gain_and_noise_is_caught_by_input_delta(self):
    # cand - x = 2 s + 0.8 s n has more error energy than the AC of y - x, so
    # a delta gate on the error energy would skip this update.
    for seed in _SEEDS:
      rng = np.random.default_rng(seed)
      x = rng.standard_normal(4096)
      ref = x + 2e-3 + 1e-3 * rng.standard_normal(4096)
      honest = ref + 1e-9 * rng.standard_normal(4096)
      report = _audit(ref, honest, views=(x,))
      self.assertTrue(report.passed, (seed, report.failed_reasons))
      cand = x + 2e-3 + 0.8e-3 * rng.standard_normal(4096)
      self._assert_caught(_audit(ref, cand, views=(x,)), "min_delta_sqnr_db")

  def test_noise_on_a_zero_block_next_to_nan_imaginary_parts_is_caught(self):
    # Only the real parts are finite, and they are exact, so the fit has no
    # data and B = 0. The zero block must still be graded.
    for seed in _SEEDS:
      rng = np.random.default_rng(seed)
      ref = rng.standard_normal((64, 64)).astype(np.complex128)
      ref.imag = np.nan
      ref[0, :32] = 0.0
      self.assertTrue(_audit(ref, ref.copy()).passed)
      cand = ref.copy()
      cand[0, :32] = 0.01
      self._assert_caught(_audit(ref, cand), "max_block_noise_ratio_db")


class NormMergeTest(absltest.TestCase):

  def _root(self, acc):
    return acc[0] * np.sqrt(acc[1])

  def test_tiny_terms_match_an_exact_sum(self):
    rng = np.random.default_rng(0)
    t = 1e-200 * rng.standard_normal((3, 512, 2))
    t[..., 1] *= 1e-5
    acc = (np.zeros(2), np.zeros(2))
    for part in t:
      acc = sqnr._dc_norm_merge(acc, part, (0,))
    # A power-of-two scale is exact, so the reference sum has no underflow.
    for k in range(2):
      vals = [float(v) * 2.0**700 for v in t[..., k].ravel()]
      want = math.sqrt(math.fsum(v * v for v in vals)) * 2.0**-700
      self.assertGreater(want, 0.0)
      self.assertLess(abs(self._root(acc)[k] / want - 1.0), 1e-12)
    # The plain sum of squares underflows to 0.
    self.assertEqual(float(np.sum(np.square(t[..., 1]))), 0.0)

  def test_zeros_and_empty_shards(self):
    zero = (np.zeros(2), np.zeros(2))
    acc = sqnr._dc_norm_merge(zero, np.zeros((64, 2)), (0,))
    np.testing.assert_array_equal(self._root(acc), [0.0, 0.0])
    acc = sqnr._dc_norm_merge(zero, np.zeros((0, 2)), (0,))
    np.testing.assert_array_equal(self._root(acc), [0.0, 0.0])
    t = np.array([[3.0, 0.0], [4.0, 1e-300]])
    acc = sqnr._dc_norm_merge(zero, t, (0,))
    for empty_first in (False, True):
      if empty_first:
        out = sqnr._dc_norm_merge(zero, np.zeros((0, 2)), (0,))
        out = sqnr._dc_norm_merge(out, t, (0,))
      else:
        out = sqnr._dc_norm_merge(acc, np.zeros((0, 2)), (0,))
      np.testing.assert_allclose(self._root(out), [5.0, 1e-300], rtol=1e-15)


class BoundedLsqTest(parameterized.TestCase):
  """`_dc_bounded_lsq3` gives the constrained minimum of the energy fit."""

  @parameterized.named_parameters(
      # Correlated v and h. The unconstrained solve gives A >= _DC_MIN_A and
      # a negative B or C, which a clamp would keep with the unrefit A.
      ("negative_b", 2.0, -0.2, 0.5, 1),
      ("negative_c", 2.0, 0.3, -0.1, 2),
      ("a_below_floor", 0.5, 1.0, 0.3, 0),
  )
  def test_minimum_meets_the_kkt_conditions(self, a0, b0, c0, violated):
    rng = np.random.default_rng(0)
    v = 2.0 ** rng.uniform(-30.0, -20.0, 4096)
    h = v * 2.0 ** rng.normal(0.0, 1.0, v.size)
    basis = np.stack([v, np.ones_like(v), h])
    e2 = a0 * v + b0 * np.mean(v) + c0 * h
    g, r = basis @ basis.T, basis @ e2
    free = np.linalg.solve(g, r)
    lo = np.array([sqnr._DC_MIN_A, 0.0, 0.0])
    # The case is the one named: only coordinate `violated` is out of bounds.
    self.assertLess(free[violated], lo[violated])
    for k in set(range(3)) - {violated}:
      self.assertGreaterEqual(free[k], lo[k])
    x = sqnr._dc_bounded_lsq3(g, r)
    self.assertTrue(np.all(x >= lo))
    self.assertEqual(x[violated], lo[violated])
    # KKT for min x g x - 2 r x: the gradient is 0 on the free coordinates
    # and >= 0 on the coordinates at their lower bound.
    grad = g @ x - r
    tol = 1e-9 * (np.abs(g) @ np.abs(x) + np.abs(r))
    for k in range(3):
      if x[k] > lo[k]:
        self.assertLessEqual(abs(grad[k]), tol[k])
      else:
        self.assertGreaterEqual(grad[k], -tol[k])
    # On the active face, the other two coordinates are the plain refit.
    f = [k for k in range(3) if k != violated]
    want = np.linalg.solve(
        g[np.ix_(f, f)], r[f] - g[f, violated] * lo[violated]
    )
    np.testing.assert_allclose(x[f], want, rtol=1e-9)

  def test_only_positive_definite_faces_are_solved(self):
    # g[0, 0] underflowed to 0 while g[0, 1] > 0, so the faces (A, B) and
    # (A, B, C) are indefinite. Their prod(diag) is 0, so
    # abs(det) > 1e-12 prod(diag) does not reject them.
    g = np.array([[0.0, 1.0, 0.0], [1.0, 4.0, 0.0], [0.0, 0.0, 1.0]])
    r = np.array([0.5, 4.0, 1.0])
    with mock.patch.object(np.linalg, "solve", wraps=np.linalg.solve) as solve:
      x = sqnr._dc_bounded_lsq3(g, r)
    self.assertNotEmpty(solve.call_args_list)
    for call in solve.call_args_list:
      g_ff = call.args[0]
      self.assertTrue(np.all(np.linalg.eigvalsh(g_ff) > 0.0), g_ff)
    # A stays at its floor, and (B, C) is the refit with A there.
    np.testing.assert_allclose(x, [1.0, 0.75, 1.0], rtol=1e-12)

  def test_saddle_of_an_indefinite_face_is_not_kept(self):
    # g[0, 0] underflowed to 0 while g[0, 1] > 0, so the faces (A, B) and
    # (A, B, C) are indefinite. A solve of these faces gives a stationary point
    # with A near 7e87, which is not a fit. Its objective in float64 can be the
    # smallest of the 8 points, so these faces must not be solved.
    h = float.fromhex
    g = np.array([
        [0.0, h("0x1.16a246e20df2ap-338"), h("0x1.3b11409238db0p-366")],
        [
            h("0x1.16a246e20df2ap-338"),
            h("0x1.47652115a9acbp+6"),
            h("0x1.4d0834db80c91p-140"),
        ],
        [
            h("0x1.3b11409238db0p-366"),
            h("0x1.4d0834db80c91p-140"),
            h("0x1.28193812740f5p-284"),
        ],
    ])
    r = np.array([
        h("0x1.70b10bdb5121ap-333"),
        h("0x1.21d3d975886bap+6"),
        -h("0x1.7a475565ec602p-275"),
    ])
    x = sqnr._dc_bounded_lsq3(g, r)
    # A and C stay at their floors, and B is the refit with them there.
    want_b = (r[1] - g[1, 0] * sqnr._DC_MIN_A) / g[1, 1]
    np.testing.assert_allclose(
        x, [sqnr._DC_MIN_A, want_b, 0.0], rtol=1e-12, atol=0.0
    )

  def test_small_scale_faces_are_kept(self):
    # A well-conditioned g at a small scale. Its diagonal is 1e-202, 4e3 and
    # 1e-137, so det(g) and prod(diag(g)) both underflow to 0. The free
    # minimum is feasible, so it is the result.
    s = np.sqrt([1e-202, 4e3, 1e-137])
    corr = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.4], [0.3, 0.4, 1.0]])
    g = corr * np.outer(s, s)
    want = np.array([3.0, 2.0, 5.0]) / s
    x = sqnr._dc_bounded_lsq3(g, g @ want)
    np.testing.assert_allclose(x, want, rtol=1e-9)

  def test_face_with_a_subnormal_diagonal_is_solved(self):
    # The (B, C) face has g[2, 2] = 200 * 2**-1074, a subnormal number. Its
    # equilibrated matrix is positive definite (det near 1e-3), but the Schur
    # complement of the unscaled face rounds to 0, so a solve of g_ff raises.
    o = 19.99 * 2.0**-537
    g = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, o], [0.0, o, 200 * 2.0**-1074]])
    want = np.array([1.0, 0.5, 1e161])
    x = sqnr._dc_bounded_lsq3(g, g @ want)
    np.testing.assert_allclose(x, want, rtol=1e-9)

  def test_face_with_a_wide_range_of_scales_is_solved(self):
    # On the unscaled (A, B) face, partial pivoting takes the 1e-101 row, and
    # the solve gives a wrong A. The free minimum is feasible, so it is the
    # result.
    g = np.array([[1e-202, 1e-101, 0.0], [1e-101, 1e30, 0.0], [0.0, 0.0, 1.0]])
    want = np.array([2.0, 1e-115, 0.0])
    x = sqnr._dc_bounded_lsq3(g, g @ want)
    np.testing.assert_allclose(x, want, rtol=1e-9)

  def test_tiny_float64_values_get_a_verdict(self):
    # One row of 0.75 sets the prescale, and the other shard holds +-2 t with
    # t near 1e-80. In that shard, y**2 and h have one value, the sum of h**2
    # is subnormal and the (B, C) face of the energy fit has a Schur
    # complement that rounds to 0. The one-row shard has no energy term.
    t = float.fromhex("0x1.369a1b37d11e4p-267")
    ref = np.full((4, 32), 2.0 * t)
    ref[1::2] *= -1.0
    cand = ref.copy()
    cand[0, :3] = np.nextafter(ref[0, :3], np.inf)
    big = np.full((1, 32), 0.75)
    report = _audit(None, None, shards=[(big, big.copy()), (ref, cand)])
    self.assertEqual(report.status, sqnr.STATUS_OK)
    self.assertTrue(report.passed, report.failed_reasons)


class QuickStartTest(absltest.TestCase):

  def test_fp8_matmul_passes_the_six_regimes(self):
    # The SQNR snippet of the SKILL.md Quick Start, with the six regimes of
    # references/sqnr.md. Without the energy term, per_channel_outliers fails
    # this honest kernel on the block and spike checks.
    def reference_fn(x, w):
      return np.matmul(x, w)

    def fp8_e4m3_matmul_fn(x, w):
      x8 = _round_to(x, "float8_e4m3fn")
      w8 = _round_to(w, "float8_e4m3fn")
      return (x8 @ w8).astype(np.float32)

    report = numerical_validator.validate_kernels(
        kernel_ref=reference_fn,
        kernel_candidate=fp8_e4m3_matmul_fn,
        shapes=[(128, 1024), (1024, 512)],
        dtype_str="float32",
        contract=numerical_validator.CONTRACT_SQNR,
        kernel_oracle="auto",
        regimes=(
            "normal",
            "scale_sweep",
            "student_t",
            "outliers",
            "boundary",
            "per_channel_outliers",
        ),
        sqnr_config=sqnr.SqnrConfig(min_sqnr_db=22.0),
    )
    summary = report.sqnr_summary
    self.assertTrue(summary["passed"], summary["failed_reasons"])
    self.assertEqual(summary["reports"], 15)


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
