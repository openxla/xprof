"""Head-to-head corpus and scorer for the SQNR check maths.

The corpus holds labelled cases. An honest case is a correct kernel output
with only rounding noise. A cheat case is a wrong output that a check must
reject. The scorer runs every case through `sqnr.audit_leaf` with the derived
math and with the fixed-constant `six_gate_v1` preset, under two threshold
policies, and counts false reds (honest cases that fail) and false greens
(cheats that pass). `decide` turns the counts into a default recommendation.

All data comes from `numpy.random.Generator` with fixed seeds, so a run is
deterministic.
"""

from collections.abc import Callable, Iterable, Mapping, Sequence
import concurrent.futures
import dataclasses
import functools
import itertools
import json
import math
import multiprocessing
import time
import types
from typing import Any

from absl import app
from absl import flags
import ml_dtypes
import numpy as np

from xprof.xparity import numerical_generator
from xprof.xparity import sqnr
from xprof.xparity.tests import sqnr_head_to_head_heldout as heldout

HONEST = "honest"
CHEAT = "cheat"

POLICY_LEGACY = "legacy_45"
POLICY_AUTO = "auto"
POLICIES = (POLICY_LEGACY, POLICY_AUTO)
GATE_MATHS = (sqnr.GATE_MATH_SIX_GATE_V1, sqnr.GATE_MATH_DERIVED)

# Explicit floor of the legacy policy, in dB.
LEGACY_FLOOR_DB = 45.0

# PrecisionSpec name of the 32-element block-scaled fp4 format.
MXFP4_PRECISION_NAME = "mxfp4"
_MX_BLOCK = 32
# Largest exponent of a float4_e2m1fn value (6.0 = 1.5 * 2**2).
_FP4_EMAX = 2

HONEST_SEEDS = 20
CHEAT_SEEDS = 3
FULL_SIZES = (1 << 12, 1 << 16, 1 << 20, 1 << 24)
REDUCED_SIZES = (1 << 12, 1 << 16)

_BF16 = np.dtype(ml_dtypes.bfloat16)
_F16 = np.dtype(np.float16)
_E4M3 = np.dtype(ml_dtypes.float8_e4m3fn)
_E5M2 = np.dtype(ml_dtypes.float8_e5m2)

# Honest cast formats: name -> dtype (None means MXFP4-like block scaling).
CAST_FORMATS: Mapping[str, np.dtype | None] = types.MappingProxyType({
    "bfloat16": _BF16,
    "float16": _F16,
    "float8_e4m3fn": _E4M3,
    "float8_e5m2": _E5M2,
    MXFP4_PRECISION_NAME: None,
})
DISTRIBUTIONS = (
    "gaussian",
    "student_t3",
    "mean_shift_4sigma",
    "lognormal",
    "outliers",
    "per_channel_outliers",
    "cancellation_pairs",
)

# Minor dimension for 2D honest cast shapes (rows = N / _MINOR).
_MINOR = 256


@dataclasses.dataclass(frozen=True)
class CaseData:
  """One generated instance of a case.

  Attributes:
    reference: Reference output.
    candidate: Candidate output, in the dtype the kernel writes.
    oracle: High-precision oracle, or None.
    input_views: Input arrays with the output shape, for the input-delta check.
    precision: Rounding sources for an AUTO threshold, or None.
  """

  reference: Any
  candidate: Any
  oracle: Any = None
  input_views: tuple[Any, ...] = ()
  precision: sqnr.PrecisionSpec | None = None


@dataclasses.dataclass(frozen=True)
class Case:
  """A labelled corpus case.

  Attributes:
    name: Unique case name.
    label: HONEST or CHEAT.
    group: Corpus group, for the summary table.
    generate: `generate(seed) -> CaseData`.
    seeds: Seeds to run.
    legacy_floor_db: Floor of the legacy policy for this case. The source tests
      pin a few cases to a floor other than 45 dB.
    atol: Absolute tolerance for an all-zero reference.
    explicit_floor_db: Floor for every policy, or None. A case whose formats
      AUTO does not model (for example int8) sets it.
  """

  name: str
  label: str
  group: str
  generate: Callable[[int], CaseData]
  seeds: tuple[int, ...]
  legacy_floor_db: float = LEGACY_FLOOR_DB
  atol: float | None = None
  explicit_floor_db: float | None = None


@dataclasses.dataclass(frozen=True)
class Outcome:
  """Verdict of one (case, seed, gate math, policy) run.

  Attributes:
    case: Case name.
    label: HONEST or CHEAT.
    group: Corpus group.
    seed: Seed.
    gate_math: Gate math of the config.
    policy: Threshold policy.
    passed: Report verdict.
    status: Report status.
    threshold_db: Resolved global floor.
    sqnr_db: Global SQNR.
    checks: Names of the checks that fired, in report order.
  """

  case: str
  label: str
  group: str
  seed: int
  gate_math: str
  policy: str
  passed: bool
  status: str
  threshold_db: float | None
  sqnr_db: float | None
  checks: tuple[str, ...]

  @property
  def wrong(self) -> bool:
    """True for a false red (honest fails) or a false green (cheat passes)."""
    return self.passed == (self.label == CHEAT)


def _rng(seed: int, *salt: int) -> np.random.Generator:
  return np.random.default_rng((seed, *salt))


def _gelu(x: np.ndarray) -> np.ndarray:
  return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def mxfp4_round(x: np.ndarray) -> np.ndarray:
  """Rounds float data to MXFP4-like values: e2m1 elements, 32-element blocks.

  Each block of 32 along the last axis shares a power-of-two scale
  2**(floor(log2(amax)) - 2), so the block maximum maps into [4, 8) and
  round-to-nearest saturates at 6.

  Args:
    x: Data whose last axis is a multiple of 32.

  Returns:
    Dequantized float32 values.
  """
  blocks = np.asarray(x, np.float64).reshape(-1, _MX_BLOCK)
  amax = np.max(np.abs(blocks), axis=1, keepdims=True)
  _, exp = np.frexp(np.where(amax > 0.0, amax, 1.0))
  scale = np.ldexp(1.0, exp - 1 - _FP4_EMAX)
  q = (blocks / scale).astype(ml_dtypes.float4_e2m1fn).astype(np.float64)
  return (q * scale).reshape(np.shape(x)).astype(np.float32)


def _clip_to(x: np.ndarray, dtype: np.dtype | None) -> np.ndarray:
  # An honest kernel output stays inside the output format range.
  if dtype is None:
    return x
  bound = 0.5 * float(ml_dtypes.finfo(dtype).max)
  return np.clip(x, -bound, bound)


def make_distribution(name: str, n: int, seed: int) -> np.ndarray:
  """Returns float32 data of shape (n // 256, 256) from a named regime."""
  shape = (n // _MINOR, _MINOR)
  rng = _rng(seed, 1)
  if name == "gaussian":
    return rng.standard_normal(shape).astype(np.float32)
  if name == "student_t3":
    return numerical_generator.generate_student_t_tensor(
        shape, "float32", df=3.0, seed=seed
    )
  if name == "mean_shift_4sigma":
    return (4.0 + rng.standard_normal(shape)).astype(np.float32)
  if name == "lognormal":
    return rng.lognormal(0.0, 1.0, shape).astype(np.float32)
  if name == "outliers":
    return numerical_generator.generate_outlier_tensor(
        shape, "float32", seed=seed
    )
  if name == "per_channel_outliers":
    return numerical_generator.generate_per_channel_outlier_tensor(
        shape, "float32", seed=seed
    )
  if name == "cancellation_pairs":
    pairs = numerical_generator.generate_cancellation_tensor(shape, "float32")
    return (pairs + rng.standard_normal(shape)).astype(np.float32)
  raise ValueError(f"unknown distribution {name!r}")


def _cast_case(fmt: str, dist: str, n: int, seed: int) -> CaseData:
  dtype = CAST_FORMATS[fmt]
  ref = _clip_to(make_distribution(dist, n, seed), dtype)
  if dtype is None:
    cand = mxfp4_round(ref)
  else:
    cand = ref.astype(dtype)
  return CaseData(
      reference=ref,
      candidate=cand,
      precision=sqnr.PrecisionSpec(output_dtype=fmt),
  )


def _matmul_operands(seed: int, k: int, m: int = 256):
  rng = _rng(seed, 2)
  a = (rng.standard_normal((m, k)) / math.sqrt(k)).astype(np.float32)
  b = rng.standard_normal((k, m)).astype(np.float32)
  return a, b


def _matmul_case(op_dtype: np.dtype, k: int, seed: int) -> CaseData:
  a, b = _matmul_operands(seed, k)
  oracle = a.astype(np.float64) @ b.astype(np.float64)
  # Products of two narrow operands are exact in float32. Accumulate in fp32.
  cand = a.astype(op_dtype).astype(np.float32) @ b.astype(op_dtype).astype(
      np.float32
  )
  return CaseData(
      reference=oracle.astype(np.float32),
      candidate=cand,
      oracle=oracle,
      precision=sqnr.PrecisionSpec(
          operand_dtypes=(op_dtype.name, op_dtype.name),
          output_dtype="float32",
          reduction_length=k,
      ),
  )


def _bf16_accumulator_case(seed: int, k: int = 256) -> CaseData:
  a, b = _matmul_operands(seed, k)
  a16 = a.astype(_BF16).astype(np.float32)
  b16 = b.astype(_BF16).astype(np.float32)
  acc = np.zeros((a.shape[0], b.shape[1]), _BF16)
  for i in range(k):
    prod = np.outer(a16[:, i], b16[i, :])
    acc = (acc.astype(np.float32) + prod).astype(_BF16)
  oracle = a.astype(np.float64) @ b.astype(np.float64)
  return CaseData(
      reference=oracle.astype(np.float32),
      candidate=acc,
      oracle=oracle,
      precision=sqnr.PrecisionSpec(
          operand_dtypes=("bfloat16", "bfloat16"),
          output_dtype="bfloat16",
          reduction_length=k,
          accumulator_dtype="bfloat16",
      ),
  )


def input_views(raw_inputs: Iterable[Any], shape: tuple[int, ...]):
  """Returns input views with the output shape, as the source harness does.

  Each input with the output element count gives a same-shape view, its
  transpose of the two minor axes when they are equal, or a transposed or
  reshaped view when the shape differs.

  Args:
    raw_inputs: Kernel inputs.
    shape: Output shape.

  Returns:
    A tuple of arrays with the output shape.
  """
  n_el = math.prod(shape)
  views = []
  for raw in raw_inputs:
    arr = np.asarray(raw)
    if arr.size != n_el:
      continue
    if arr.shape == shape:
      views.append(arr)
      if arr.ndim >= 2 and arr.shape[-2] == arr.shape[-1] > 1:
        views.append(np.swapaxes(arr, -2, -1))
      continue
    swapped = np.swapaxes(arr, -2, -1) if arr.ndim >= 2 else None
    if swapped is not None and swapped.shape == shape:
      views.append(swapped)
    reshaped = arr.reshape(shape)
    if (
        swapped is None
        or swapped.shape != shape
        or not np.array_equal(reshaped, swapped)
    ):
      views.append(reshaped)
  return tuple(views)


def _with_inputs(
    ref: Any, cand: Any, inputs: Iterable[Any], **kwargs: Any
) -> CaseData:
  return CaseData(
      reference=ref,
      candidate=cand,
      input_views=input_views(inputs, np.shape(ref)),
      **kwargs,
  )


# ---------------------------------------------------------------------------
# Honest cases ported from the source test (rounding noise only).
# ---------------------------------------------------------------------------


def _src_bf16_const_row(seed: int) -> CaseData:
  ref = _rng(seed, 11).standard_normal((128, 64))
  ref[0, :] = 1.3
  return CaseData(ref, ref.astype(_BF16))


def _src_bf16_broadcast(seed: int) -> CaseData:
  col = _rng(seed, 12).standard_normal((128, 1))
  ref = np.broadcast_to(col, (128, 64)).copy()
  return CaseData(ref, ref.astype(_BF16))


def _src_fp8x3_residual(shape: tuple[int, ...], seed: int) -> CaseData:
  rng = _rng(seed, 13)
  orc = rng.standard_normal(shape)
  axis = 1 if len(shape) == 2 else (1, 2)
  amax = np.max(np.abs(orc), axis=axis, keepdims=True)
  noise = rng.uniform(-1.0, 1.0, shape) * amax * 2.0**-9
  return CaseData(orc, (orc + noise).astype(np.float32))


def _src_bf16_matmul_f32_out(seed: int) -> CaseData:
  rng = _rng(seed, 14)
  a = rng.standard_normal((128, 64)).astype(np.float32)
  b = rng.standard_normal((64, 128)).astype(np.float32)
  cand = a.astype(_BF16).astype(np.float32) @ b.astype(_BF16).astype(np.float32)
  return CaseData(a @ b, cand)


def _src_near_constant_sums(seed: int) -> CaseData:
  p = np.exp(_rng(seed, 15).standard_normal((64, 512)))
  p /= p.sum(-1, keepdims=True)
  ref = p.astype(np.float32).sum(-1)
  cand = p.astype(np.float32).astype(np.float64).sum(-1).astype(np.float32)
  return CaseData(ref, cand)


def _src_bf16_dc_heavy(kind: str, seed: int) -> CaseData:
  """Builds the source bf16 cases with a large DC term."""
  rng = _rng(seed, 16)
  if kind == "dc5":
    orc = 5.0 + rng.standard_normal((64, 64))
  elif kind == "dc100":
    orc = 100.0 + rng.standard_normal((128, 256))
  elif kind == "log_softmax":
    logits = rng.standard_normal((16, 1024))
    orc = logits - np.log(np.sum(np.exp(logits), axis=-1, keepdims=True))
  else:
    sp_l = rng.standard_normal((8, 4096)) * 8.0
    sp_l -= np.max(sp_l, axis=-1, keepdims=True)
    orc = np.exp(sp_l) / np.sum(np.exp(sp_l), axis=-1, keepdims=True)
  return CaseData(orc, orc.astype(_BF16).astype(np.float32))


def _src_one_ulp_state(seed: int) -> CaseData:
  st = _rng(seed, 17).standard_normal((32, 32)).astype(np.float32)
  return _with_inputs(st, np.nextafter(st, np.float32(np.inf)), (st,))


def _src_bf16_residual_update(seed: int) -> CaseData:
  st = _rng(seed, 18).standard_normal((32, 32)).astype(np.float32)
  r_res = (st.astype(np.float64) + 1e-2 * _gelu(st.astype(np.float64))).astype(
      np.float32
  )
  return _with_inputs(r_res, r_res.astype(_BF16).astype(np.float32), (st,))


def _src_bf16_grid(dc_off: float, seed: int) -> CaseData:
  orc = dc_off + _rng(seed, 19).standard_normal((128, 256))
  return CaseData(orc, orc.astype(_BF16).astype(np.float32))


def _src_two_ulp_update(seed: int) -> CaseData:
  rng = _rng(seed, 20)
  mask = np.zeros((64, 64), np.float32)
  mask[:, 32:] = -np.inf
  x = np.ones((64, 64), np.float32)
  signs = rng.choice(np.array([-1.0, 1.0], np.float32), size=(64, 64))
  ref = (x + np.float32(6.0 * 2.0**-23) * signs).astype(np.float32)
  cand = (ref + np.float32(2.0 * 2.0**-23) * signs).astype(np.float32)
  return _with_inputs(ref, cand, (mask, x))


def _src_quiet_row(seed: int) -> CaseData:
  rng = _rng(seed, 21)
  ref = rng.standard_normal((128, 128)).astype(np.float32)
  ref[0, :] = (1e-5 * rng.standard_normal(128)).astype(np.float32)
  cand = ref + (1e-5 * rng.standard_normal((128, 128))).astype(np.float32)
  return CaseData(ref, cand)


def _src_bf16_cancelled_row(seed: int) -> CaseData:
  rng = _rng(seed, 22)
  base = rng.standard_normal((128, 128)).astype(_BF16)
  ref, cand = base.copy(), base.copy()
  cand[0, :] = (2e-3 * rng.standard_normal(128)).astype(_BF16)
  ref[0, :] = (1e-3 * rng.standard_normal(128)).astype(_BF16)
  return CaseData(ref, cand)


def _src_complex_fft_noise(seed: int) -> CaseData:
  rng = _rng(seed, 23)
  ref = rng.standard_normal(1024) + 1j * 1e-5 * rng.standard_normal(1024)
  ref = ref.astype(np.complex64)
  cand = (ref.real + 1j * 1e-5 * rng.standard_normal(1024)).astype(np.complex64)
  return CaseData(ref, cand)


def _src_tiny_reference(seed: int) -> CaseData:
  tiny = _rng(seed, 24).standard_normal((16, 16)) * 1e-200
  return _with_inputs(tiny, tiny * 1.00001, (np.zeros_like(tiny),))


def _src_quiet_slice(shape: tuple[int, int], ratio: float, seed: int):
  rng = _rng(seed, 25)
  ref = rng.standard_normal(shape).astype(np.float32)
  ref[0, :] = (math.sqrt(ratio) * 1e-3 * rng.standard_normal(shape[1])).astype(
      np.float32
  )
  cand = ref + (1e-3 * rng.standard_normal(shape)).astype(np.float32)
  return CaseData(ref, cand)


def _update_48db(seed: int) -> tuple[np.ndarray, np.ndarray]:
  rng = _rng(seed, 26)
  x_in = rng.standard_normal((128, 128)).astype(np.float32)
  upd = rng.standard_normal((128, 128))
  upd *= math.sqrt(float(np.sum(np.square(x_in, dtype=np.float64)))) / (
      math.sqrt(float(np.sum(np.square(upd)))) * 10.0 ** (48.0 / 20.0)
  )
  return x_in, (x_in.astype(np.float64) + upd).astype(np.float32)


def _src_bf16_update_48db(seed: int) -> CaseData:
  x_in, r_upd = _update_48db(seed)
  return _with_inputs(r_upd, r_upd.astype(_BF16).astype(np.float32), (x_in,))


def _cl_table_tensors(seed: int) -> dict[str, np.ndarray]:
  """Builds the shared tensors of the source comparison table cases."""
  rng = _rng(seed, 27)
  a = (rng.standard_normal((256, 512)) / math.sqrt(512.0)).astype(np.float32)
  b = rng.standard_normal((512, 256)).astype(np.float32)
  y_mm = a @ b
  bf16_mm = a.astype(_BF16).astype(np.float32) @ b.astype(_BF16).astype(
      np.float32
  )
  amax = np.max(np.abs(y_mm), axis=1, keepdims=True)
  fp8x3 = (y_mm + rng.uniform(-1.0, 1.0, y_mm.shape) * amax * 2.0**-9).astype(
      np.float32
  )
  return dict(
      y_b=(0.5 * y_mm).astype(np.float32),
      bf16_b=(0.5 * bf16_mm).astype(np.float32),
      fp8x3_b=(0.5 * fp8x3).astype(np.float32),
  )


def _src_cl_table_honest(key: str, seed: int) -> CaseData:
  t = _cl_table_tensors(seed)
  return CaseData(t["y_b"], t[key])


def _src_zero_ref_within_atol(seed: int) -> CaseData:
  del seed
  return CaseData(
      np.zeros((8, 8), np.float32), np.full((8, 8), 1e-9, np.float32)
  )


def _src_complex_zero_within_atol(seed: int) -> CaseData:
  del seed
  return CaseData(
      np.zeros(8, np.complex64), np.full(8, 0.9e-5 + 0j, np.complex64)
  )


def _src_zero_padded_shards(seed: int) -> CaseData:
  rng = _rng(seed, 28)
  sh0_r = rng.standard_normal((16, 32)).astype(np.float32)
  sh1_r = np.zeros((16, 32), np.float32)
  sh0_c = sh0_r + (1e-6 * rng.standard_normal((16, 32))).astype(np.float32)
  sh1_c = (1e-7 * rng.standard_normal((16, 32))).astype(np.float32)
  # Two shards: audit_leaf receives one (reference, candidate) pair each.
  return CaseData(reference=(sh0_r, sh1_r), candidate=(sh0_c, sh1_c))


# ---------------------------------------------------------------------------
# Cheat cases ported from the source test.
# ---------------------------------------------------------------------------


def _src_extreme(index: int, seed: int) -> CaseData:
  """Builds one of the source extreme-range pairs."""
  del seed
  pairs = (
      (np.ones(4), np.full(4, 1e-320)),
      (np.array([1e-10, 1e308]), np.full(2, 1e-10)),
      (np.array([0.8e308, -0.8e308]), np.array([-0.8e308, 0.8e308])),
      (
          np.array([1e38, -1e38], np.float32),
          np.array([-1e38, 1e38], np.float32),
      ),
  )
  ref, cand = pairs[index]
  return CaseData(ref, cand)


def _src_fp8_downcast(seed: int) -> CaseData:
  r0 = _rng(seed, 30).standard_normal((128, 128)).astype(np.float32)
  return CaseData(r0, r0.astype(_E4M3).astype(np.float32))


def _src_dc_constant(dc: float, shape: tuple[int, int], seed: int):
  ref = dc + _rng(seed, 31).standard_normal(shape)
  return CaseData(ref, np.full_like(ref, dc))


def _src_block_skip_3d(seed: int) -> CaseData:
  ref = _rng(seed, 32).standard_normal((32, 32, 32)).astype(np.float32)
  cand = ref.copy()
  cand[:, 0, 0] = 0.0
  return CaseData(ref, cand)


def _src_dc100_skip(kind: str, seed: int) -> CaseData:
  """Builds the source skip cheats on data with a DC term of 100."""
  rng = _rng(seed, 33)
  if kind == "tail_1d":
    ref = 100.0 + rng.standard_normal(200_000)
    cand = ref.copy()
    cand[-32:] = 100.0
    return CaseData(ref, cand)
  ref = 100.0 + rng.standard_normal((128, 256))
  cand = ref.copy()
  if kind == "row_tile":
    cand[0, :32] = 100.0
  elif kind == "col":
    cand[:32, 0] = 100.0
  else:
    cand[:64, :] = 100.0
  return CaseData(ref, cand)


def _src_spike(seed: int) -> CaseData:
  r0 = _rng(seed, 34).standard_normal((128, 128)).astype(np.float32)
  cand = r0.copy()
  cand[5, 9] += 0.3
  return CaseData(r0, cand)


def _src_row_col_bias(kind: str, seed: int) -> CaseData:
  """Builds the source row and column bias cheats."""
  rng = _rng(seed, 35)
  if kind == "nan_16":
    rows, cols = 16, 16
  elif kind == "thin_3x128":
    rows, cols = 3, 128
  else:
    rb = rng.standard_normal((64, 1)) * 1e6
    cand = np.broadcast_to(rb, (64, 64)).copy()
    return CaseData(rb + rng.standard_normal((64, 64)), cand)
  rb = rng.standard_normal((rows, 1)) * 200.0
  cb = rng.standard_normal((1, cols)) * 200.0
  ref = rb + cb + rng.standard_normal((rows, cols))
  cand = (rb + cb) * np.ones((rows, cols))
  if kind == "nan_16":
    ref[0, 0] = cand[0, 0] = np.nan
  return CaseData(ref, cand)


def _src_residual_passthrough_nan(seed: int) -> CaseData:
  x = _rng(seed, 36).standard_normal((32, 64)).astype(np.float32)
  ref = (x.astype(np.float64) + 1e-3 * _gelu(x.astype(np.float64))).astype(
      np.float32
  )
  ref[0, 0] = np.nan
  cand = x.copy()
  cand[0, 0] = np.nan
  return _with_inputs(ref, cand, (x,))


def _src_transposed_passthrough(shape: tuple[int, int], seed: int):
  rng = _rng(seed, 37)
  x = rng.standard_normal(shape).astype(np.float32)
  ref = x.T + 1e-3 * rng.standard_normal(shape[::-1]).astype(np.float32)
  return _with_inputs(ref, np.ascontiguousarray(x.T), (x,))


def _src_int_input_passthrough(seed: int) -> CaseData:
  rng = _rng(seed, 38)
  xi = rng.integers(10, 500, size=(32, 32), dtype=np.int32)
  ref = xi.astype(np.float32) + 0.25 * rng.standard_normal((32, 32)).astype(
      np.float32
  )
  return _with_inputs(ref, xi.astype(np.float32), (xi,))


def _src_moe_quiet_expert(seed: int) -> CaseData:
  rng = _rng(seed, 39)
  ref = (rng.standard_normal((64, 256)) / 32.0).astype(np.float32)
  ref[0, :] = rng.standard_normal(256).astype(np.float32) * 80.0
  cand = np.zeros_like(ref)
  cand[0, :] = ref[0, :]
  return CaseData(ref, cand)


def _src_complex_drop_imag(seed: int) -> CaseData:
  rng = _rng(seed, 40)
  ref = rng.standard_normal(128) + 1j * 1e-3 * rng.standard_normal(128)
  ref = ref.astype(np.complex64)
  return CaseData(ref, (ref.real + 0j).astype(np.complex64))


def _src_sharded_row_mean(seed: int) -> CaseData:
  rng = _rng(seed, 41)
  ref = rng.standard_normal((32, 32)).astype(np.float32) + rng.standard_normal(
      (32, 1)
  ).astype(np.float32) * np.float32(5000.0)
  cand = np.broadcast_to(np.mean(ref, axis=1, keepdims=True), (32, 32))
  cand = cand.astype(np.float32)
  return CaseData(
      reference=(ref[:16], ref[16:]), candidate=(cand[:16], cand[16:])
  )


def _src_scalar(kind: str, seed: int) -> CaseData:
  del seed
  if kind == "zero":
    return CaseData(np.float32(10.0), np.float32(0.0))
  if kind == "passthrough":
    return CaseData(
        np.float32(10.05),
        np.float32(10.0),
        input_views=(np.float32(10.0),),
    )
  ref = np.array(3.0 + 4.0j, np.complex64)
  return CaseData(ref, np.array(3.0 + 0.0j, np.complex64))


def _src_grid_cheat(dc_off: float, kind: str, seed: int) -> CaseData:
  orc = dc_off + _rng(seed, 19).standard_normal((128, 256))
  if kind == "half_rows":
    cand = orc.copy()
    cand[:64, :] = dc_off
  elif kind == "half_cols":
    cand = orc.copy()
    cand[:, :128] = dc_off
  else:
    cand = np.full_like(orc, float(kind))
  return CaseData(orc, cand)


def _src_odd_7d_skip(seed: int) -> CaseData:
  ref = (1.0 + 0.1 * _rng(seed, 42).standard_normal((9,) * 7)).astype(
      np.float32
  )
  cand = ref.copy()
  cand.reshape(-1)[:32] = 0.0
  return CaseData(ref, cand)


def _src_multiview_passthrough(seed: int) -> CaseData:
  rng = _rng(seed, 43)
  x = np.ones((64, 64), np.float32)
  signs = rng.choice(np.array([-1.0, 1.0], np.float32), size=(64, 64))
  ref = (x + np.float32(6.0 * 2.0**-23) * signs).astype(np.float32)
  x_other = (ref - np.float32(0.01) * signs).astype(np.float32)
  return _with_inputs(ref, x, (x, x_other))


def _src_plaquette_4d(seed: int) -> CaseData:
  rng = _rng(seed, 44)
  rb = (rng.standard_normal((2, 9, 64, 1)) * 200.0).astype(np.float32)
  cb = (rng.standard_normal((2, 9, 1, 128)) * 200.0).astype(np.float32)
  cand = (rb + cb) * np.ones((2, 9, 64, 128), np.float32)
  ref = cand + rng.standard_normal((2, 9, 64, 128)).astype(np.float32)
  return CaseData(ref, cand)


def _src_discrete(kind: str, seed: int) -> CaseData:
  del seed
  if kind == "dtype":
    return CaseData(
        np.asarray([0, 1, 2], ml_dtypes.int4),
        np.asarray([0, 1, 2], ml_dtypes.uint4),
    )
  dtype = ml_dtypes.int4 if kind == "int4" else ml_dtypes.uint4
  return CaseData(
      np.asarray([0, 1, 2, 3, 4, 5], dtype),
      np.asarray([0, 1, 7, 3, 4, 5], dtype),
  )


def _src_zero_ref_beyond_atol(kind: str, seed: int) -> CaseData:
  del seed
  if kind == "complex":
    return CaseData(
        np.zeros(8, np.complex64), np.full(8, 0.8e-5 + 0.8e-5j, np.complex64)
    )
  if kind == "tiny_no_atol":
    return CaseData(
        np.zeros((8, 8), np.float32), np.full((8, 8), 1e-9, np.float32)
    )
  return CaseData(
      np.zeros((8, 8), np.float32), np.full((8, 8), 1e-3, np.float32)
  )


def _src_noop_48db(seed: int) -> CaseData:
  x_in, r_upd = _update_48db(seed)
  return _with_inputs(r_upd, x_in, (x_in,))


def _src_cl_table_cheat(kind: str, seed: int) -> CaseData:
  """Builds one cheat of the source comparison table."""
  rng = _rng(seed, 45)
  if kind == "fp8_1pass_downcast":
    y_b = _cl_table_tensors(seed)["y_b"]
    return CaseData(y_b, y_b.astype(_E4M3).astype(np.float32))
  if kind == "isolated_spike":
    y_b = _cl_table_tensors(seed)["y_b"]
    cand = y_b.copy()
    cand[7, 13] += np.float32(0.30)
    return CaseData(y_b, cand)
  if kind == "quiet_32x32_skip":
    ref = (rng.standard_normal((128, 128)) * 0.04).astype(np.float32)
    ref[0, :] = (rng.standard_normal(128) * 4.0).astype(np.float32)
    cand = ref.copy()
    cand[1:33, :32] = 0.0
    return CaseData(ref, cand)
  if kind == "residual_noop":
    x = (rng.standard_normal((128, 128)) * 0.5).astype(np.float32)
    ref = (x.astype(np.float64) + 0.01 * _gelu(x.astype(np.float64))).astype(
        np.float32
    )
    return _with_inputs(ref, x, (x,))
  ref = (10.0 + 0.05 * rng.standard_normal((128, 128))).astype(np.float32)
  return CaseData(ref, np.full_like(ref, 10.0))


# ---------------------------------------------------------------------------
# New cheat cases.
# ---------------------------------------------------------------------------

_CHEAT_N = 1 << 16


def _bf16_gaussian(seed: int, shape=(_CHEAT_N // _MINOR, _MINOR)):
  ref = _rng(seed, 50).standard_normal(shape).astype(np.float32)
  return ref, ref.astype(_BF16)


_BF16_OUT = sqnr.PrecisionSpec(output_dtype="bfloat16")


def _new_half_scale(seed: int) -> CaseData:
  ref, _ = _bf16_gaussian(seed)
  return CaseData(ref, (0.5 * ref).astype(_BF16), precision=_BF16_OUT)


def _new_fp8_in_bf16_kernel(seed: int) -> CaseData:
  honest = _matmul_case(_E4M3, 4096, seed)
  assert honest.precision is not None
  # The kernel claims bf16 operands but rounds them to e4m3.
  return dataclasses.replace(
      honest,
      precision=dataclasses.replace(
          honest.precision, operand_dtypes=("bfloat16", "bfloat16")
      ),
  )


def _new_zero_block(kind: str, seed: int) -> CaseData:
  """Builds the zeroed-block and related block cheats."""
  rng = _rng(seed, 51)
  ref, cand = _bf16_gaussian(seed, (1024, 1024))
  cand = cand.copy()
  if kind == "single_32_block":
    row = int(rng.integers(0, 1024))
    col = 32 * int(rng.integers(0, 1024 // 32))
    cand[row, col : col + 32] = 0
  elif kind == "last_tile":
    cand[-8:, -128:] = 0
  else:
    cand[int(rng.integers(0, 1024)), :] = 0
  return CaseData(ref, cand, precision=_BF16_OUT)


def _residual_update(seed: int) -> tuple[np.ndarray, np.ndarray]:
  # y = x + update, with the update 50 dB below x, so `return x` meets 45 dB.
  rng = _rng(seed, 52)
  x = rng.standard_normal((256, 256)).astype(np.float32)
  upd = np.tanh(rng.standard_normal((256, 256)))
  upd *= 10.0 ** (-50.0 / 20.0) * float(np.std(x)) / float(np.std(upd))
  return x, (x.astype(np.float64) + upd).astype(np.float32)


def _new_return_x(kind: str, seed: int) -> CaseData:
  x, ref = _residual_update(seed)
  if kind == "plus_eps":
    cand = x + np.float32(1e-6)
  else:
    cand = x.copy()
  return _with_inputs(ref, cand, (x,))


def _new_constant_mean(seed: int) -> CaseData:
  ref, _ = _bf16_gaussian(seed)
  ref = ref + np.float32(8.0)
  return CaseData(ref, np.full(ref.shape, np.mean(ref), _BF16))


def _new_single_spike(seed: int) -> CaseData:
  rng = _rng(seed, 53)
  ref, cand = _bf16_gaussian(seed)
  cand = cand.copy()
  idx = tuple(int(rng.integers(0, s)) for s in cand.shape)
  cand[idx] = cand[idx] + ml_dtypes.bfloat16(0.5)
  return CaseData(ref, cand, precision=_BF16_OUT)


def _new_sign_flip_block(seed: int) -> CaseData:
  rng = _rng(seed, 54)
  ref, cand = _bf16_gaussian(seed, (1024, 1024))
  cand = cand.copy()
  row = int(rng.integers(0, 1024))
  col = 32 * int(rng.integers(0, 32))
  cand[row, col : col + 32] = -cand[row, col : col + 32]
  return CaseData(ref, cand, precision=_BF16_OUT)


def _new_shared_bug(seed: int) -> CaseData:
  rng = _rng(seed, 55)
  oracle = rng.standard_normal((512, 512))
  bug = oracle.copy()
  # Both reference and candidate drop the same 32x32 tile. Only the oracle
  # shows the fault.
  bug[64:96, 128:160] = 0.0
  ref = bug.astype(np.float32)
  return CaseData(
      ref,
      ref.astype(_BF16),
      oracle=oracle,
      precision=_BF16_OUT,
  )


# ---------------------------------------------------------------------------
# Corpus assembly.
# ---------------------------------------------------------------------------


def _seeds(count: int, base: int = 0) -> tuple[int, ...]:
  return tuple(range(base, base + count))


def honest_cases(
    sizes: Sequence[int] = FULL_SIZES, seeds: int = HONEST_SEEDS
) -> list[Case]:
  """Returns the honest corpus for the given output sizes and seed count."""
  seed_list = _seeds(seeds)
  cases = []
  for fmt in CAST_FORMATS:
    for dist in DISTRIBUTIONS:
      for n in sizes:
        cases.append(
            Case(
                name=f"cast_{fmt}_{dist}_n2^{int(math.log2(n))}",
                label=HONEST,
                group=f"cast_{fmt}",
                generate=functools.partial(_cast_case, fmt, dist, n),
                seeds=seed_list,
            )
        )
  for op in (_BF16, _E4M3):
    for k in (256, 4096):
      cases.append(
          Case(
              name=f"matmul_{op.name}_operands_f32acc_k{k}",
              label=HONEST,
              group="matmul",
              generate=functools.partial(_matmul_case, op, k),
              seeds=seed_list,
          )
      )
  cases.append(
      Case(
          name="matmul_bfloat16_acc_k256",
          label=HONEST,
          group="matmul",
          generate=_bf16_accumulator_case,
          seeds=seed_list,
      )
  )
  src = []

  def add(
      name: str,
      fn: Callable[[int], CaseData],
      floor: float = LEGACY_FLOOR_DB,
      atol: float | None = None,
  ) -> None:
    src.append(
        Case(
            name=f"src_{name}",
            label=HONEST,
            group="source_honest",
            generate=fn,
            seeds=seed_list,
            legacy_floor_db=floor,
            atol=atol,
        )
    )

  add("bf16_const_row", _src_bf16_const_row)
  add("bf16_broadcast", _src_bf16_broadcast)
  for shape in ((128, 128), (128, 128, 2), (128, 2), (2, 128)):
    tag = "x".join(map(str, shape))
    add(f"fp8x3_residual_{tag}", functools.partial(_src_fp8x3_residual, shape))
  add("bf16_matmul_f32_out", _src_bf16_matmul_f32_out, floor=44.0)
  add("near_constant_sums", _src_near_constant_sums)
  for kind in ("dc5", "dc100", "log_softmax", "sparse_softmax"):
    add(f"bf16_{kind}", functools.partial(_src_bf16_dc_heavy, kind))
  add("one_ulp_state", _src_one_ulp_state)
  add("bf16_residual_update", _src_bf16_residual_update)
  for dc_off in (1000.0, 1000.5, 1001.0, 1002.0):
    add(f"bf16_grid_{dc_off:g}", functools.partial(_src_bf16_grid, dc_off))
  add("two_ulp_update_views", _src_two_ulp_update)
  add("quiet_row", _src_quiet_row)
  add("bf16_cancelled_row", _src_bf16_cancelled_row, floor=40.0)
  add("complex_fft_noise", _src_complex_fft_noise)
  add("zero_padded_shards", _src_zero_padded_shards)
  add("tiny_reference", _src_tiny_reference)
  for shape in ((32, 32), (128, 32), (32, 64), (32, 128), (32, 256)):
    for ratio in (0.25, 0.5, 1.0):
      add(
          f"quiet_slice_{shape[0]}x{shape[1]}_r{ratio:g}",
          functools.partial(_src_quiet_slice, shape, ratio),
      )
  add("bf16_update_48db", _src_bf16_update_48db)
  add(
      "cl_table_bf16_mxu",
      functools.partial(_src_cl_table_honest, "bf16_b"),
      floor=44.0,
  )
  add(
      "cl_table_fp8x3_residual",
      functools.partial(_src_cl_table_honest, "fp8x3_b"),
      floor=44.0,
  )
  add("zero_ref_within_atol", _src_zero_ref_within_atol, atol=1e-5)
  add("complex_zero_within_atol", _src_complex_zero_within_atol, atol=1e-5)
  return cases + src


def cheat_cases(seeds: int = CHEAT_SEEDS) -> list[Case]:
  """Returns the cheat corpus."""
  seed_list = _seeds(seeds, base=1000)
  cases = []

  def add(
      name: str,
      fn: Callable[[int], CaseData],
      group: str,
      floor: float = LEGACY_FLOOR_DB,
      atol: float | None = None,
  ) -> None:
    cases.append(
        Case(
            name=name,
            label=CHEAT,
            group=group,
            generate=fn,
            seeds=seed_list,
            legacy_floor_db=floor,
            atol=atol,
        )
    )

  six = "source_six_cheats"
  for i, tag in enumerate(
      ("tiny_1e-320", "gap_1e308", "sign_swap_f64", "sign_swap_f32")
  ):
    add(f"src_extreme_{tag}", functools.partial(_src_extreme, i), six)
  add("src_fp8_downcast_f32", _src_fp8_downcast, six)
  add(
      "src_dc_1e15_constant",
      functools.partial(_src_dc_constant, 1e15, (64, 128)),
      six,
  )
  add("src_block_skip_3d", _src_block_skip_3d, six, floor=25.0)
  for kind in ("row_tile", "col", "tail_1d", "half_rows"):
    add(f"src_dc100_skip_{kind}", functools.partial(_src_dc100_skip, kind), six)
  add("src_spike_0p3", _src_spike, six)
  for kind in ("nan_16", "thin_3x128", "row_1e6"):
    add(f"src_bias_{kind}", functools.partial(_src_row_col_bias, kind), six)
  add("src_residual_passthrough_nan", _src_residual_passthrough_nan, six)
  add(
      "src_transposed_passthrough_32x64",
      functools.partial(_src_transposed_passthrough, (32, 64)),
      six,
  )
  add("src_int_input_passthrough", _src_int_input_passthrough, six)
  add("src_moe_quiet_expert", _src_moe_quiet_expert, six, floor=40.0)
  add("src_complex_drop_imag", _src_complex_drop_imag, six)
  add("src_sharded_row_mean", _src_sharded_row_mean, six)
  for kind in ("zero", "passthrough", "complex_drop_imag"):
    add(f"src_scalar_{kind}", functools.partial(_src_scalar, kind), six)
  for dc_off in (1000.0, 1000.5, 1001.0, 1002.0):
    # dict.fromkeys drops the repeated "1000" constant at dc_off = 1000.
    kinds = (f"{dc_off:g}", "1000", "1004", "half_rows", "half_cols")
    for kind in dict.fromkeys(kinds):
      add(
          f"src_grid_{dc_off:g}_{kind}",
          functools.partial(_src_grid_cheat, dc_off, kind),
          six,
      )
  add("src_odd_7d_skip", _src_odd_7d_skip, six)

  regression = "source_regression"
  add("src_multiview_passthrough", _src_multiview_passthrough, regression)
  add("src_plaquette_4d", _src_plaquette_4d, regression)
  for kind in ("int4", "uint4", "dtype"):
    add(
        f"src_discrete_{kind}",
        functools.partial(_src_discrete, kind),
        regression,
    )
  add(
      "src_zero_ref_no_atol",
      functools.partial(_src_zero_ref_beyond_atol, "tiny_no_atol"),
      regression,
  )
  add(
      "src_zero_ref_beyond_atol",
      functools.partial(_src_zero_ref_beyond_atol, "real"),
      regression,
      atol=1e-5,
  )
  add(
      "src_complex_zero_beyond_atol",
      functools.partial(_src_zero_ref_beyond_atol, "complex"),
      regression,
      atol=1e-5,
  )
  add("src_noop_48db", _src_noop_48db, regression)
  add(
      "src_transposed_passthrough_512",
      functools.partial(_src_transposed_passthrough, (512, 512)),
      regression,
  )
  table = "source_cl_table"
  for kind in (
      "fp8_1pass_downcast",
      "quiet_32x32_skip",
      "residual_noop",
      "dc_bias_shortcut",
      "isolated_spike",
  ):
    add(
        f"src_cl_table_{kind}",
        functools.partial(_src_cl_table_cheat, kind),
        table,
        floor=44.0,
    )

  new = "new_cheats"
  add("half_scale", _new_half_scale, new)
  add("fp8_in_bf16_kernel", _new_fp8_in_bf16_kernel, new)
  for kind in ("single_32_block", "last_tile", "dropped_row"):
    add(f"skip_{kind}", functools.partial(_new_zero_block, kind), new)
  for kind in ("return_x", "plus_eps"):
    add(kind, functools.partial(_new_return_x, kind), new)
  add("constant_mean", _new_constant_mean, new)
  add("single_element_spike", _new_single_spike, new)
  add("sign_flip_block", _new_sign_flip_block, new)
  add("shared_bug_oracle", _new_shared_bug, new)
  return cases


def build_corpus(
    *,
    full: bool,
    honest_seeds: int = HONEST_SEEDS,
    cheat_seeds: int = CHEAT_SEEDS,
) -> list[Case]:
  """Returns the full corpus, or the reduced one (no N >= 2**20)."""
  sizes = FULL_SIZES if full else REDUCED_SIZES
  return honest_cases(sizes, honest_seeds) + cheat_cases(cheat_seeds)


def _heldout_instance(
    generate: Callable[[int], heldout.HeldoutData], seed: int
) -> CaseData:
  d = generate(seed)
  return CaseData(
      reference=d.reference,
      candidate=d.candidate,
      oracle=d.oracle,
      input_views=d.input_views,
      precision=d.precision,
  )


def build_heldout_corpus(
    honest_seeds: int = HONEST_SEEDS, cheat_seeds: int = CHEAT_SEEDS
) -> list[Case]:
  """Returns the held-out cases as corpus cases. They run separately."""
  return [
      Case(
          name=c.name,
          label=c.label,
          group=c.group,
          generate=functools.partial(_heldout_instance, c.generate),
          seeds=c.seeds,
          explicit_floor_db=c.floor_db,
      )
      for c in heldout.heldout_cases(honest_seeds, cheat_seeds)
  ]


# ---------------------------------------------------------------------------
# Scorer.
# ---------------------------------------------------------------------------


def make_config(gate_math: str, policy: str, case: Case) -> sqnr.SqnrConfig:
  """Returns the config for a gate math and threshold policy."""
  if case.explicit_floor_db is not None:
    floor = case.explicit_floor_db
  elif policy == POLICY_AUTO:
    floor = sqnr.AUTO
  else:
    floor = case.legacy_floor_db
  if gate_math == sqnr.GATE_MATH_SIX_GATE_V1:
    return sqnr.SqnrConfig.six_gate_v1(min_sqnr_db=floor)
  return sqnr.SqnrConfig(min_sqnr_db=floor, gate_math=gate_math)


def fired_checks(report: sqnr.SqnrLeafReport) -> tuple[str, ...]:
  """Returns the metric name of each failed reason, or the bad status."""
  names = []
  for reason in report.failed_reasons:
    body = reason.split(": ", 1)[-1]
    names.append(body.split(" ", 1)[0] if body else "unknown")
  if not names and not report.passed:
    names.append(report.status)
  return tuple(dict.fromkeys(names))


def _shards(x: Any) -> tuple[Any, ...]:
  return tuple(x) if isinstance(x, tuple) else (x,)


def run_instance(case: Case, seed: int) -> list[Outcome]:
  """Generates one instance and audits it with every config and policy."""
  data = case.generate(seed)
  refs, cands = _shards(data.reference), _shards(data.candidate)
  pairs = list(zip(refs, cands))
  oracle = None if data.oracle is None else list(_shards(data.oracle))
  views = [[v] for v in data.input_views] if len(pairs) == 1 else []
  outcomes = []
  for gate_math in GATE_MATHS:
    for policy in POLICIES:
      try:
        report = sqnr.audit_leaf(
            pairs,
            config=make_config(gate_math, policy, case),
            precision=data.precision,
            input_views=views,
            oracle_pairs=oracle,
            atol=case.atol,
        )
      except Exception as e:  # pylint: disable=broad-exception-caught
        # A raise is not a verdict. Count it as a failed report, so it is a
        # false red on an honest case, and name it in the checks column.
        report = sqnr.SqnrLeafReport(
            path="",
            dtype=str(np.asarray(refs[0]).dtype),
            status=sqnr.STATUS_ERROR,
            passed=False,
            threshold_db=None,
            threshold_source="",
            gate_math=gate_math,
            failed_reasons=(f"output: raised:{type(e).__name__}",),
        )
      outcomes.append(
          Outcome(
              case=case.name,
              label=case.label,
              group=case.group,
              seed=seed,
              gate_math=gate_math,
              policy=policy,
              passed=bool(report.passed),
              status=report.status,
              threshold_db=report.threshold_db,
              sqnr_db=report.sqnr_db,
              checks=fired_checks(report),
          )
      )
  return outcomes


def run_corpus(cases: Sequence[Case], workers: int = 1) -> list[Outcome]:
  """Runs every (case, seed) instance.

  The result order does not depend on `workers`.

  Args:
    cases: The corpus.
    workers: Worker processes. 1 runs in this process.

  Returns:
    The outcomes, in corpus order, then seed order.
  """
  tasks = list(
      itertools.chain.from_iterable(
          ((case, seed) for seed in case.seeds) for case in cases
      )
  )
  if workers <= 1:
    chunks = [run_instance(case, seed) for case, seed in tasks]
  else:
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("fork")
    ) as pool:
      chunks = list(
          pool.map(
              run_instance,
              [case for case, _ in tasks],
              [seed for _, seed in tasks],
              chunksize=4,
          )
      )
  return list(itertools.chain.from_iterable(chunks))


@dataclasses.dataclass(frozen=True)
class Rates:
  """False-red and false-green counts for one (gate math, policy).

  Attributes:
    honest_failed: Honest instances that failed.
    honest_total: Honest instances.
    cheats_passed: Cheat instances that passed.
    cheats_total: Cheat instances.
    checks: Count of each fired check name over all instances.
  """

  honest_failed: int
  honest_total: int
  cheats_passed: int
  cheats_total: int
  checks: Mapping[str, int]

  @property
  def false_red(self) -> float:
    """Share of honest instances that failed."""
    return self.honest_failed / max(self.honest_total, 1)

  @property
  def false_green(self) -> float:
    """Share of cheat instances that passed."""
    return self.cheats_passed / max(self.cheats_total, 1)


def score(outcomes: Iterable[Outcome]) -> dict[tuple[str, str], Rates]:
  """Returns Rates per (gate_math, policy)."""
  acc: dict[tuple[str, str], list[Any]] = {}
  for o in outcomes:
    row = acc.setdefault((o.gate_math, o.policy), [0, 0, 0, 0, {}])
    if o.label == HONEST:
      row[1] += 1
      row[0] += int(not o.passed)
    else:
      row[3] += 1
      row[2] += int(o.passed)
    for name in o.checks:
      row[4][name] = row[4].get(name, 0) + 1
  return {k: Rates(*v) for k, v in acc.items()}


def wrong_cases(
    outcomes: Iterable[Outcome], gate_math: str, policy: str
) -> dict[str, int]:
  """Returns {case: wrong instance count} for one gate math and policy."""
  out: dict[str, int] = {}
  for o in outcomes:
    if o.gate_math == gate_math and o.policy == policy and o.wrong:
      out[o.case] = out.get(o.case, 0) + 1
  return out


def decide(
    outcomes: Sequence[Outcome], policy: str = POLICY_AUTO
) -> dict[str, Any]:
  """Applies the decision rule to the outcomes of one threshold policy.

  The derived math becomes the default only if it is no worse than
  six_gate_v1 on false reds AND on false greens. If derived loses only on
  false greens, the result names the six_gate_v1 checks that fired on the
  cheats derived let through, as "derived+keep:<checks>". Otherwise the
  default stays six_gate_v1.

  Args:
    outcomes: Outcomes from `run_corpus`.
    policy: Threshold policy to decide on.

  Returns:
    {"default": ..., "evidence": {...}}.
  """
  rates = score(o for o in outcomes if o.policy == policy)
  derived = rates[(sqnr.GATE_MATH_DERIVED, policy)]
  legacy = rates[(sqnr.GATE_MATH_SIX_GATE_V1, policy)]
  reds_ok = derived.false_red <= legacy.false_red
  greens_ok = derived.false_green <= legacy.false_green
  missed = {
      (o.case, o.seed)
      for o in outcomes
      if o.policy == policy
      and o.gate_math == sqnr.GATE_MATH_DERIVED
      and o.label == CHEAT
      and o.passed
  }
  caught = [
      o
      for o in outcomes
      if o.policy == policy
      and o.gate_math == sqnr.GATE_MATH_SIX_GATE_V1
      and (o.case, o.seed) in missed
      and not o.passed
  ]
  keep = sorted(set(itertools.chain.from_iterable(o.checks for o in caught)))
  if reds_ok and greens_ok:
    default = sqnr.GATE_MATH_DERIVED
  elif reds_ok and keep:
    default = f"{sqnr.GATE_MATH_DERIVED}+keep:{','.join(keep)}"
  else:
    default = sqnr.GATE_MATH_SIX_GATE_V1
  evidence = dict(
      policy=policy,
      derived_false_red=derived.false_red,
      six_gate_v1_false_red=legacy.false_red,
      derived_false_green=derived.false_green,
      six_gate_v1_false_green=legacy.false_green,
      derived_missed_cheats=sorted({case for case, _ in missed}),
      derived_false_red_cases=sorted(
          wrong_cases(
              (o for o in outcomes if o.label == HONEST),
              sqnr.GATE_MATH_DERIVED,
              policy,
          )
      ),
  )
  return dict(default=default, evidence=evidence)


def markdown_table(
    cases: Sequence[Case],
    outcomes: Sequence[Outcome],
    runtime_s: float,
    title: str = "SQNR head-to-head",
) -> str:
  """Returns the summary, the decisions, and per-case outcomes as markdown."""
  n_honest = sum(c.label == HONEST for c in cases)
  n_cheat = len(cases) - n_honest
  lines = [
      f"# {title}",
      "",
      (
          f"Cases: {n_honest} honest, {n_cheat} cheat. Instances:"
          f" {len(outcomes) // (len(GATE_MATHS) * len(POLICIES))}."
          f" Runtime: {runtime_s:.1f} s."
      ),
      "",
      "| gate math | policy | false reds | false greens | top checks |",
      "|---|---|---|---|---|",
  ]
  rates = score(outcomes)
  for (gate_math, policy), r in sorted(rates.items()):
    top = sorted(r.checks.items(), key=lambda kv: (-kv[1], kv[0]))[:6]
    top_s = ", ".join(f"{k}:{v}" for k, v in top)
    lines.append(
        f"| {gate_math} | {policy} |"
        f" {r.honest_failed}/{r.honest_total} ({100 * r.false_red:.1f}%) |"
        f" {r.cheats_passed}/{r.cheats_total} ({100 * r.false_green:.1f}%) |"
        f" {top_s} |"
    )
  lines += ["", "## False reds per group", ""]
  lines += _group_table(outcomes)
  lines += ["", "## Decision", ""]
  for policy in POLICIES:
    d = decide(outcomes, policy)
    lines.append(f"- `{policy}`: **{d['default']}**")
  lines += [
      "",
      "## Per-case outcomes",
      "",
      (
          "Cell = failed/instances for honest cases and passed/instances for"
          " cheats (wrong verdicts), then the checks that fired."
      ),
      "",
      "| case | label | group | "
      + " | ".join(
          f"{g} {p}" for g, p in itertools.product(GATE_MATHS, POLICIES)
      )
      + " |",
      "|---|---|---|" + "---|" * (len(GATE_MATHS) * len(POLICIES)),
  ]
  by_key: dict[tuple[str, str, str], list[Outcome]] = {}
  for o in outcomes:
    by_key.setdefault((o.case, o.gate_math, o.policy), []).append(o)
  for case in cases:
    cells = []
    for g in GATE_MATHS:
      for p in POLICIES:
        group = by_key.get((case.name, g, p), [])
        wrong = sum(o.wrong for o in group)
        checks = sorted(
            set(itertools.chain.from_iterable(o.checks for o in group))
        )
        cells.append(f"{wrong}/{len(group)} {' '.join(checks)}".strip())
    lines.append(
        f"| {case.name} | {case.label} | {case.group} | "
        + " | ".join(cells)
        + " |"
    )
  return "\n".join(lines) + "\n"


def _group_table(outcomes: Sequence[Outcome]) -> list[str]:
  """Returns wrong/instances per group for each (gate math, policy)."""
  keys = list(itertools.product(GATE_MATHS, POLICIES))
  counts: dict[str, dict[tuple[str, str], list[int]]] = {}
  for o in outcomes:
    cell = counts.setdefault(o.group, {}).setdefault(
        (o.gate_math, o.policy), [0, 0]
    )
    cell[0] += int(o.wrong)
    cell[1] += 1
  lines = [
      "| group | " + " | ".join(f"{g} {p}" for g, p in keys) + " |",
      "|---|" + "---|" * len(keys),
  ]
  for group in sorted(counts):
    cells = ["{}/{}".format(*counts[group].get(k, [0, 0])) for k in keys]
    lines.append(f"| {group} | " + " | ".join(cells) + " |")
  return lines


def results_json(
    cases: Sequence[Case], outcomes: Sequence[Outcome], runtime_s: float
) -> dict[str, Any]:
  """Returns a JSON-safe dict with rates, decisions and every outcome."""
  rates = score(outcomes)
  return dict(
      honest_cases=sum(c.label == HONEST for c in cases),
      cheat_cases=sum(c.label == CHEAT for c in cases),
      runtime_s=runtime_s,
      rates={
          f"{g}/{p}": dict(
              dataclasses.asdict(r),
              false_red=r.false_red,
              false_green=r.false_green,
          )
          for (g, p), r in sorted(rates.items())
      },
      decisions={p: decide(outcomes, p) for p in POLICIES},
      outcomes=[dataclasses.asdict(o) for o in outcomes],
  )


_FULL = flags.DEFINE_boolean(
    "full", True, "Run the full corpus (N up to 2**24)."
)
_HONEST_SEEDS = flags.DEFINE_integer(
    "honest_seeds", HONEST_SEEDS, "Seeds per honest case.", lower_bound=1
)
_CHEAT_SEEDS = flags.DEFINE_integer(
    "cheat_seeds", CHEAT_SEEDS, "Seeds per cheat case.", lower_bound=1
)
_WORKERS = flags.DEFINE_integer(
    "workers", 1, "Worker processes.", lower_bound=1
)
_HELDOUT = flags.DEFINE_boolean(
    "heldout",
    False,
    "Also run the held-out cases and report them in a separate table.",
)
_OUT_MD = flags.DEFINE_string("out_md", "", "Path for the markdown table.")
_OUT_JSON = flags.DEFINE_string("out_json", "", "Path for the JSON results.")


def run(
    *,
    full: bool,
    honest_seeds: int,
    cheat_seeds: int,
    workers: int,
    out_md: str,
    out_json: str,
    with_heldout: bool = False,
) -> str:
  """Runs the corpus, writes the outputs, and returns the markdown.

  With `with_heldout`, the held-out cases run after the main corpus. They get
  their own table, rates and decisions, under the JSON key "heldout". They do
  not change the main rates or decisions.

  Args:
    full: Run the full main corpus.
    honest_seeds: Seeds per honest case.
    cheat_seeds: Seeds per cheat case.
    workers: Worker processes.
    out_md: Path for the markdown, or "".
    out_json: Path for the JSON, or "".
    with_heldout: Also run the held-out cases.

  Returns:
    The markdown.
  """
  cases = build_corpus(
      full=full, honest_seeds=honest_seeds, cheat_seeds=cheat_seeds
  )
  start = time.monotonic()
  outcomes = run_corpus(cases, workers)
  runtime_s = time.monotonic() - start
  table = markdown_table(cases, outcomes, runtime_s)
  results = results_json(cases, outcomes, runtime_s)
  if with_heldout:
    h_cases = build_heldout_corpus(honest_seeds, cheat_seeds)
    start = time.monotonic()
    h_outcomes = run_corpus(h_cases, workers)
    h_runtime_s = time.monotonic() - start
    table += "\n" + markdown_table(
        h_cases, h_outcomes, h_runtime_s, title="SQNR head-to-head: held-out"
    )
    results["heldout"] = results_json(h_cases, h_outcomes, h_runtime_s)
  if out_md:
    with open(out_md, "w") as f:
      f.write(table)
  if out_json:
    with open(out_json, "w") as f:
      json.dump(results, f, indent=1)
  return table


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError(f"Too many positional arguments: {argv[1:]}")
  print(
      run(
          full=_FULL.value,
          honest_seeds=_HONEST_SEEDS.value,
          cheat_seeds=_CHEAT_SEEDS.value,
          workers=_WORKERS.value,
          out_md=_OUT_MD.value,
          out_json=_OUT_JSON.value,
          with_heldout=_HELDOUT.value,
      )
  )


if __name__ == "__main__":
  app.run(main)
