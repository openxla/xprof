"""Held-out cases for the SQNR head-to-head.

The main head-to-head corpus was used while the derived checks were designed.
These cases were not. The honest cases hold noise that does not come from one
round-to-nearest cast of a float32 reference: a different accumulation order,
two independent roundings, real NVFP4 scaling, int8 quantization, residual
adds and softmax. The cheats are faults that the main corpus does not hold.

Run them with `sqnr_head_to_head --heldout`. The harness reports them in a
separate table. This module depends only on NumPy, ml_dtypes and the
PrecisionSpec type of `sqnr`. It does not use the scale rules or the noise
model of `sqnr`.
"""

from collections.abc import Callable
import dataclasses
import functools
import math

import ml_dtypes
import numpy as np

from xprof.xparity import sqnr

HONEST = "honest"
CHEAT = "cheat"

HONEST_SEED_BASE = 5000
CHEAT_SEED_BASE = 6000

_BF16 = np.dtype(ml_dtypes.bfloat16)
_F16 = np.dtype(np.float16)
_E4M3 = np.dtype(ml_dtypes.float8_e4m3fn)
_E2M1 = np.dtype(ml_dtypes.float4_e2m1fn)
_F32 = np.dtype(np.float32)

# NVFP4: e2m1 elements, one e4m3 scale per 16 elements, one float32 scale per
# tensor. The tensor scale maps the tensor maximum to 6 * 448, the product of
# the largest e2m1 and e4m3 values.
_NVFP4_BLOCK = 16
_E2M1_MAX = 6.0
_E4M3_MAX = 448.0

# Explicit floor of the int8 cases, in dB. One symmetric int8 rounding with
# step amax / 127 has SQNR 10*log10(12 * 127**2 / c**2), where c = amax / rms.
# For Gaussian rows of 1024 to 4096 values c is about 4, so one operand gives
# about 41 dB and two operands about 38 dB. The floor keeps a 6 dB margin.
INT8_FLOOR_DB = 32.0

# Matmul output size of the held-out matmul cases.
_M = 256
_N = 256


@dataclasses.dataclass(frozen=True)
class HeldoutData:
  """One generated instance of a held-out case.

  Attributes:
    reference: Reference output.
    candidate: Candidate output, in the dtype the kernel writes.
    oracle: High-precision oracle, or None.
    input_views: Input arrays with the output shape.
    precision: Rounding sources for an AUTO threshold.
  """

  reference: np.ndarray
  candidate: np.ndarray
  oracle: np.ndarray | None = None
  input_views: tuple[np.ndarray, ...] = ()
  precision: sqnr.PrecisionSpec | None = None


@dataclasses.dataclass(frozen=True)
class HeldoutCase:
  """A labelled held-out case.

  Attributes:
    name: Unique case name.
    label: HONEST or CHEAT.
    group: Group name for the summary table.
    generate: `generate(seed) -> HeldoutData`.
    seeds: Seeds to run.
    floor_db: Explicit floor for every threshold policy, or None.
  """

  name: str
  label: str
  group: str
  generate: Callable[[int], HeldoutData]
  seeds: tuple[int, ...]
  floor_db: float | None = None


def _rng(seed: int, *salt: int) -> np.random.Generator:
  return np.random.default_rng((seed, 7, *salt))


def _spec(
    output: str,
    operands: tuple[str, ...] = (),
    k: int | None = None,
) -> sqnr.PrecisionSpec:
  return sqnr.PrecisionSpec(
      operand_dtypes=operands, output_dtype=output, reduction_length=k
  )


def _operands(
    rng: np.random.Generator, k: int, dist: str
) -> tuple[np.ndarray, np.ndarray]:
  """Returns float32 operands (M, K) and (K, N) from a named regime."""
  if dist == "gaussian":
    a = rng.standard_normal((_M, k))
    b = rng.standard_normal((k, _N))
  elif dist == "student_t5":
    a = rng.standard_t(5.0, (_M, k))
    b = rng.standard_t(5.0, (k, _N))
  elif dist == "relu":
    a = np.maximum(rng.standard_normal((_M, k)), 0.0)
    b = rng.standard_normal((k, _N))
  elif dist == "uniform":
    a = rng.uniform(-1.0, 1.0, (_M, k))
    b = rng.uniform(-1.0, 1.0, (k, _N))
  else:
    raise ValueError(f"unknown operand regime {dist!r}")
  return a.astype(np.float32), (b / math.sqrt(k)).astype(np.float32)


def chunked_matmul(
    a: np.ndarray, b: np.ndarray, chunks: int, order: str
) -> np.ndarray:
  """Returns a @ b with float32 partial sums over K chunks in a set order.

  Args:
    a: Float32 operand (M, K).
    b: Float32 operand (K, N).
    chunks: Number of K chunks. K must be a multiple of it.
    order: "reverse" adds the chunks from last to first. "pairwise" adds them as
      a balanced tree.

  Returns:
    The float32 product.
  """
  step = a.shape[1] // chunks
  parts = [
      a[:, i : i + step] @ b[i : i + step] for i in range(0, a.shape[1], step)
  ]
  if order == "reverse":
    acc = np.zeros((a.shape[0], b.shape[1]), np.float32)
    for part in reversed(parts):
      acc = acc + part
    return acc
  if order == "pairwise":
    while len(parts) > 1:
      parts = [parts[i] + parts[i + 1] for i in range(0, len(parts), 2)]
    return parts[0]
  raise ValueError(f"unknown order {order!r}")


# ---------------------------------------------------------------------------
# (a) Same-dtype outputs of one matmul in two summation orders.
# ---------------------------------------------------------------------------


def _accumulation_order(
    out: np.dtype, dist: str, chunks: int, order: str, seed: int
) -> HeldoutData:
  a, b = _operands(_rng(seed, 1), 1024, dist)
  ref = (a @ b).astype(out)
  cand = chunked_matmul(a, b, chunks, order).astype(out)
  return HeldoutData(ref, cand, precision=_spec(out.name, k=1024))


# ---------------------------------------------------------------------------
# (b) Two-sided rounding: reference and candidate each rounded independently.
# ---------------------------------------------------------------------------


def _elementwise(x: np.ndarray, kind: str) -> np.ndarray:
  """Applies a kernel body in the dtype of x (float32 or float64)."""
  if kind == "gelu":
    c = math.sqrt(2.0 / math.pi)
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x**3)))
  if kind == "silu":
    return x / (1.0 + np.exp(-x))
  if kind == "layernorm":
    y = x - x.mean(axis=-1, keepdims=True)
    return y / np.sqrt((y * y).mean(axis=-1, keepdims=True) + 1e-5)
  if kind == "rmsnorm":
    return x / np.sqrt((x * x).mean(axis=-1, keepdims=True) + 1e-5)
  raise ValueError(f"unknown kernel body {kind!r}")


def _two_sided(kind: str, out: np.dtype, seed: int) -> HeldoutData:
  rng = _rng(seed, 2)
  x = rng.standard_normal((256, 1024))
  if kind in ("layernorm", "rmsnorm"):
    x = x * rng.lognormal(0.0, 0.5, (256, 1)) + rng.normal(0.0, 2.0, (256, 1))
  x32 = x.astype(np.float32)
  ref = _elementwise(x32.astype(np.float64), kind).astype(out)
  cand = _elementwise(x32, kind).astype(out)
  return HeldoutData(ref, cand, precision=_spec(out.name))


def _two_sided_double_rounding(kind: str, seed: int) -> HeldoutData:
  # The reference rounds the float64 value once to e4m3. The candidate rounds
  # its float32 value to bfloat16 and then to e4m3, so a few values round the
  # other way.
  rng = _rng(seed, 10)
  x32 = (2.0 * rng.standard_normal((256, 1024))).astype(np.float32)
  ref = _elementwise(x32.astype(np.float64), kind).astype(_E4M3)
  cand = _elementwise(x32, kind).astype(_BF16).astype(_E4M3)
  return HeldoutData(ref, cand, precision=_spec(_E4M3.name))


def _two_sided_bf16_ops(seed: int) -> HeldoutData:
  # The reference rounds every op to bfloat16. The candidate computes in
  # float32 and rounds once at the end. Both are honest bfloat16 kernels.
  rng = _rng(seed, 3)
  x16 = rng.standard_normal((256, 1024)).astype(_BF16)
  ref = _elementwise(x16, "silu").astype(_BF16)
  cand = _elementwise(x16.astype(np.float32), "silu").astype(_BF16)
  return HeldoutData(ref, cand, precision=_spec("bfloat16", ("bfloat16",)))


# ---------------------------------------------------------------------------
# (c) NVFP4 with an e4m3 block scale and a float32 tensor scale.
# ---------------------------------------------------------------------------


def nvfp4_quantize(x: np.ndarray) -> np.ndarray:
  """Returns x rounded to NVFP4 and dequantized to float32.

  Each 16-element block along the last axis has an e4m3 scale
  e4m3(amax_block / (6 * s)). The float32 tensor scale is
  s = amax / (6 * 448). Elements round to nearest e2m1 and saturate at 6.

  Args:
    x: Data whose last axis is a multiple of 16.

  Returns:
    Dequantized float32 values with the shape of x.
  """
  x64 = np.asarray(x, np.float64)
  amax = float(np.max(np.abs(x64)))
  s = float(np.float32(amax / (_E2M1_MAX * _E4M3_MAX))) if amax > 0 else 1.0
  blocks = x64.reshape(-1, _NVFP4_BLOCK)
  b_amax = np.max(np.abs(blocks), axis=1, keepdims=True)
  b_scale = (b_amax / (_E2M1_MAX * s)).astype(_E4M3).astype(np.float64) * s
  safe = np.where(b_scale > 0.0, b_scale, 1.0)
  q = (blocks / safe).astype(_E2M1).astype(np.float64)
  return (q * b_scale).reshape(x64.shape).astype(np.float32)


def _nvfp4_data(dist: str, seed: int) -> np.ndarray:
  """Draws NVFP4 test data of the named distribution."""
  rng = _rng(seed, 4)
  shape = (1024, 256)
  if dist == "gaussian":
    x = rng.standard_normal(shape)
  elif dist == "student_t3":
    x = rng.standard_t(3.0, shape)
  elif dist == "channel_scaled":
    x = rng.standard_normal(shape) * rng.lognormal(0.0, 1.5, (1, shape[1]))
  elif dist == "lognormal":
    x = rng.lognormal(0.0, 1.0, shape)
  else:
    raise ValueError(f"unknown regime {dist!r}")
  return x.astype(np.float32)


def _nvfp4_cast(dist: str, seed: int) -> HeldoutData:
  ref = _nvfp4_data(dist, seed)
  return HeldoutData(ref, nvfp4_quantize(ref), precision=_spec("nvfp4"))


def _nvfp4_matmul(k: int, seed: int) -> HeldoutData:
  a, b = _operands(_rng(seed, 5), k, "gaussian")
  oracle = a.astype(np.float64) @ b.astype(np.float64)
  qa = nvfp4_quantize(a)
  qb = nvfp4_quantize(b.T).T
  return HeldoutData(
      oracle.astype(np.float32),
      qa @ qb,
      oracle=oracle,
      precision=_spec("float32", ("nvfp4", "nvfp4"), k),
  )


# ---------------------------------------------------------------------------
# (d) int8 per-channel symmetric quantization of a matmul.
# ---------------------------------------------------------------------------


def _int8_matmul(dist: str, k: int, seed: int) -> HeldoutData:
  a, b = _operands(_rng(seed, 6), k, dist)
  sa = np.max(np.abs(a), axis=1, keepdims=True) / 127.0
  sb = np.max(np.abs(b), axis=0, keepdims=True) / 127.0
  qa = np.clip(np.rint(a / sa), -127, 127).astype(np.int8)
  qb = np.clip(np.rint(b / sb), -127, 127).astype(np.int8)
  # int32 accumulation is exact. float64 holds every int32 sum exactly.
  acc = qa.astype(np.float64) @ qb.astype(np.float64)
  cand = (acc * sa * sb).astype(np.float32)
  ref = (a.astype(np.float64) @ b.astype(np.float64)).astype(np.float32)
  return HeldoutData(ref, cand, precision=_spec("float32", ("int8", "int8"), k))


# ---------------------------------------------------------------------------
# (e) float32 matmuls, float32 accumulation in different orders.
# ---------------------------------------------------------------------------


def _f32_order(dist: str, k: int, order: str, seed: int) -> HeldoutData:
  a, b = _operands(_rng(seed, 7), k, dist)
  return HeldoutData(
      a @ b,
      chunked_matmul(a, b, 16, order),
      precision=_spec("float32", k=k),
  )


# ---------------------------------------------------------------------------
# (f) Residual adds and softmax outputs in bfloat16.
# ---------------------------------------------------------------------------


def _residual(scale: float, ref_dtype: np.dtype, seed: int) -> HeldoutData:
  """Builds a residual-update case with an update of the given scale."""
  rng = _rng(seed, 8)
  x16 = rng.standard_normal((256, 1024)).astype(_BF16)
  upd = (scale * np.tanh(rng.standard_normal((256, 1024)))).astype(np.float32)
  exact = x16.astype(np.float64) + upd.astype(np.float64)
  if ref_dtype == _BF16:
    # Two-sided: the kernel rounds the update to bfloat16 before the add.
    cand = (
        x16.astype(np.float32) + upd.astype(_BF16).astype(np.float32)
    ).astype(_BF16)
  else:
    cand = (x16.astype(np.float32) + upd).astype(_BF16)
  return HeldoutData(
      exact.astype(ref_dtype),
      cand,
      input_views=(x16,),
      precision=_spec("bfloat16"),
  )


def _softmax(temperature: float, ref_dtype: np.dtype, seed: int) -> HeldoutData:
  z = temperature * _rng(seed, 9).standard_normal((256, 1024))
  z64 = z.astype(np.float32).astype(np.float64)
  e64 = np.exp(z64 - z64.max(axis=-1, keepdims=True))
  ref = (e64 / e64.sum(axis=-1, keepdims=True)).astype(ref_dtype)
  z32 = z64.astype(np.float32)
  e32 = np.exp(z32 - z32.max(axis=-1, keepdims=True))
  cand = (e32 / e32.sum(axis=-1, keepdims=True)).astype(_BF16)
  return HeldoutData(ref, cand, precision=_spec("bfloat16"))


# ---------------------------------------------------------------------------
# Held-out cheats.
# ---------------------------------------------------------------------------


def _bf16_output(seed: int, salt: int) -> tuple[np.ndarray, np.ndarray]:
  ref = _rng(seed, salt).standard_normal((1024, 1024)).astype(np.float32)
  return ref, ref.astype(_BF16)


def _cheat_tile_scale(seed: int) -> HeldoutData:
  # One 128x128 tile has a scale 1% too large.
  rng = _rng(seed, 20)
  ref, cand = _bf16_output(seed, 21)
  r, c = (128 * int(rng.integers(0, 8)) for _ in range(2))
  cand[r : r + 128, c : c + 128] = (
      1.01 * ref[r : r + 128, c : c + 128]
  ).astype(_BF16)
  return HeldoutData(ref, cand, precision=_spec("bfloat16"))


def _cheat_block_scale(seed: int) -> HeldoutData:
  # One 32-element block has a scale 1% too large.
  rng = _rng(seed, 22)
  ref, cand = _bf16_output(seed, 23)
  r = int(rng.integers(0, 1024))
  c = 32 * int(rng.integers(0, 32))
  cand[r, c : c + 32] = (1.01 * ref[r, c : c + 32]).astype(_BF16)
  return HeldoutData(ref, cand, precision=_spec("bfloat16"))


def _cheat_dropped_k_chunk(seed: int) -> HeldoutData:
  # The last 8-row tile skips the final 8-wide K chunk (a ragged K tail).
  m, k = 2048, 2048
  rng = _rng(seed, 24)
  a = rng.standard_normal((m, k)).astype(np.float32)
  b = (rng.standard_normal((k, _N)) / math.sqrt(k)).astype(np.float32)
  a16 = a.astype(_BF16).astype(np.float32)
  b16 = b.astype(_BF16).astype(np.float32)
  cand = a16 @ b16
  cand[-8:] -= a16[-8:, -8:] @ b16[-8:]
  ref = (a.astype(np.float64) @ b.astype(np.float64)).astype(np.float32)
  return HeldoutData(
      ref, cand, precision=_spec("float32", ("bfloat16", "bfloat16"), k)
  )


def _cheat_row_permutation(seed: int) -> HeldoutData:
  # Two rows of a bias-dominated output are swapped.
  rng = _rng(seed, 25)
  bias = 2.0 * rng.standard_normal((1, 256))
  ref = (bias + 0.05 * rng.standard_normal((1024, 256))).astype(np.float32)
  i, j = rng.choice(1024, size=2, replace=False)
  perm = np.arange(1024)
  perm[[i, j]] = perm[[j, i]]
  return HeldoutData(ref, ref[perm].astype(_BF16), precision=_spec("bfloat16"))


# ---------------------------------------------------------------------------
# Corpus.
# ---------------------------------------------------------------------------


def _seeds(count: int, base: int) -> tuple[int, ...]:
  return tuple(range(base, base + count))


def heldout_cases(
    honest_seeds: int = 20, cheat_seeds: int = 3
) -> list[HeldoutCase]:
  """Returns the held-out honest cases, then the held-out cheats."""
  h_seeds = _seeds(honest_seeds, HONEST_SEED_BASE)
  cases: list[HeldoutCase] = []

  def honest(
      name: str,
      group: str,
      fn: Callable[[int], HeldoutData],
      floor_db: float | None = None,
  ) -> None:
    cases.append(
        HeldoutCase(f"ho_{name}", HONEST, group, fn, h_seeds, floor_db)
    )

  for out in (_BF16, _F32):
    for dist, chunks, order in (
        ("gaussian", 4, "reverse"),
        ("student_t5", 4, "reverse"),
        ("gaussian", 8, "pairwise"),
    ):
      honest(
          f"acc_order_{out.name}_{dist}_{order}{chunks}_k1024",
          "ho_a_accumulation_order",
          functools.partial(_accumulation_order, out, dist, chunks, order),
      )
  for kind in ("gelu", "silu", "layernorm", "rmsnorm"):
    for out in (_BF16, _F16):
      honest(
          f"two_sided_{kind}_{out.name}",
          "ho_b_two_sided",
          functools.partial(_two_sided, kind, out),
      )
  honest(
      f"two_sided_layernorm_{_E4M3.name}",
      "ho_b_two_sided",
      functools.partial(_two_sided, "layernorm", _E4M3),
  )
  for kind in ("gelu", "silu"):
    honest(
        f"two_sided_{kind}_bf16_then_{_E4M3.name}",
        "ho_b_two_sided",
        functools.partial(_two_sided_double_rounding, kind),
    )
  honest("two_sided_silu_bf16_ops", "ho_b_two_sided", _two_sided_bf16_ops)
  for dist in ("gaussian", "student_t3", "channel_scaled", "lognormal"):
    honest(
        f"nvfp4_{dist}",
        "ho_c_nvfp4",
        functools.partial(_nvfp4_cast, dist),
    )
  for k in (1024, 4096):
    honest(
        f"nvfp4_matmul_k{k}",
        "ho_c_nvfp4",
        functools.partial(_nvfp4_matmul, k),
    )
  for dist in ("gaussian", "relu", "uniform"):
    for k in (1024, 4096):
      honest(
          f"int8_matmul_{dist}_k{k}",
          "ho_d_int8",
          functools.partial(_int8_matmul, dist, k),
          floor_db=INT8_FLOOR_DB,
      )
  for dist in ("gaussian", "relu"):
    for k in (1024, 16384):
      for order in ("reverse", "pairwise"):
        honest(
            f"f32_matmul_{dist}_k{k}_{order}16",
            "ho_e_f32_order",
            functools.partial(_f32_order, dist, k, order),
        )
  for ref_dtype in (_F32, _BF16):
    for scale in (0.1, 0.01):
      honest(
          f"residual_add_{scale:g}_ref_{ref_dtype.name}",
          "ho_f_residual_softmax",
          functools.partial(_residual, scale, ref_dtype),
      )
    for temperature in (1.0, 8.0):
      honest(
          f"softmax_t{temperature:g}_ref_{ref_dtype.name}",
          "ho_f_residual_softmax",
          functools.partial(_softmax, temperature, ref_dtype),
      )

  c_seeds = _seeds(cheat_seeds, CHEAT_SEED_BASE)
  for name, fn in (
      ("tile_scale_1p01", _cheat_tile_scale),
      ("block_scale_1p01", _cheat_block_scale),
      ("dropped_final_k_chunk_edge_rows", _cheat_dropped_k_chunk),
      ("row_permutation_bias_dominated", _cheat_row_permutation),
  ):
    cases.append(HeldoutCase(f"ho_{name}", CHEAT, "ho_cheats", fn, c_seeds))
  return cases
