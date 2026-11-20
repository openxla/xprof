"""Signal-to-quantization-noise ratio (SQNR) metrics and anti-cheat checks.

A ULP gate asks whether every element is within a few representable steps of
the reference. A kernel with deliberately lossy arithmetic (narrow operands,
block-scaled formats, a different accumulation order) fails that question and
can still be correct in aggregate. SQNR, `10*log10(sum(y**2) / sum((c-y)**2))`
in dB for a reference `y` and a candidate `c`, measures the aggregate error.
`expected_sqnr_db` turns the rounding formats of a kernel into the SQNR that
honest rounding gives, so the floor can follow from the formats.

Global SQNR alone cannot see local faults. A skipped tile, a constant or
mean-only output, a passthrough of an input, or a single spike barely moves
it. In strict mode `audit_leaf` therefore also runs local checks. Two check
families exist: "derived" (thresholds from a false-alarm rate) and
"six_gate_v1" (a fixed-constant preset).

With an oracle, the verdict uses the oracle SQNR of the candidate and of the
reference. This is a trade between false failures and false passes:
  - A reference at or above the floor cannot cause a false failure. The
    candidate passes at min(floor, reference + oracle_delta_db), so it can be
    up to 3 dB (the default) below the floor when the reference is also near
    the floor.
  - A reference below the floor fails the leaf (`reference_below_floor`),
    because it cannot vouch for the candidate. Without this rule, a candidate
    that is independently as bad as a lossy reference, or that shares its
    bug, would pass. The cost is a failure for a good candidate whose
    reference is lossy: fix the reference or lower the floor.

This module imports only the standard library, NumPy and ml_dtypes, and no
other xparity module, so standalone benchmark harnesses can carry its source
inline. Every array access goes through a `read(x, slices)` callback, and the
strict path holds at most one tile of a leaf as float64 at a time. Callers can
thus keep large leaves on a device and copy one tile at a time.

Public API:
  audit_leaf: Audits one output leaf (one or more shards) and runs the checks.
  compute_sqnr: Core SQNR metrics of one array pair, without the checks.
  merge_reports: Merges leaf or sample reports into a worst-case summary.
  expected_sqnr_db: Expected SQNR of honest rounding for a PrecisionSpec.
  precision_from_dtypes: The PrecisionSpec of a kernel from its dtypes.
  resolve_threshold: The SQNR floor and its source for a config.
  default_read: The default `read` callback for host arrays.
  as_real: Float64 view with a trailing (real[, imag]) axis.
  SqnrConfig, PrecisionSpec, SqnrLeafReport: Policy, rounding sources, result.
  BLOCK_SCALED_FORMATS: Block-scaled format names that PrecisionSpec accepts.
  STATUS_*, STATUS_ORDER: Leaf statuses, best to worst.
  METRIC_KEYS, MIN_MERGED_KEYS, MAX_MERGED_KEYS, REPORT_ONLY_KEYS: Metrics.
  AUTO, DEFAULT_*, GATE_MATH_*, THRESHOLD_SOURCE_*: Option values.
"""

import dataclasses
import itertools
import math
import statistics
import types
from typing import Any, Callable, Sequence

import ml_dtypes
import numpy as np

AUTO = "auto"
# Fixed floor when no format is known (and of six_gate_v1). One round-to-
# nearest gives 55.6 dB in bfloat16 and 31.5 dB in float8_e4m3fn (see the
# format table below). 45 dB is 10.6 dB below the first and 13.5 dB above the
# second, so an honest bfloat16 kernel passes and a float8 downcast fails.
DEFAULT_MIN_SQNR_DB = 45.0
# False-alarm rate per output leaf on honest rounding noise. Each derived check
# family splits it over its own tests (a Bonferroni bound), so each family
# stays at or below alpha. z(1 - 1e-6) = 4.75. The normal score grows as
# sqrt(2 ln(1 / alpha)), so a 1000x smaller alpha widens the bounds by only
# about 25% (z = 6.0) and costs little detection power.
# The rate holds for noise that the fitted model E[e**2] = A v + B describes.
# Two float32 summation orders of the same bf16 matmul fall outside it at
# outputs near zero, where cancellation leaves a few ulp of error: about 1
# false failure in 120 leaves of 65536 elements (5 of 620 seeds, on
# `max_error_ulp`).
DEFAULT_ALPHA = 1e-6
# Elements per block along the minor axis. 32 is the OCP MX scale block, so
# one block holds one shared scale. A 32-sample chi-square bound at
# z = 6.4 (alpha / n_blocks = 1e-10) is 3.5x the mean noise power (5.5 dB),
# narrow enough to see a block that is a few dB too noisy.
DEFAULT_BLOCK_SIZE = 32
# AUTO floor = expected SQNR - 6 dB (one bit, 6.02 dB). The margin covers the
# 2 dB error of the accumulator model and the 3 dB of one more rounding of the
# same size that the PrecisionSpec does not list.
DEFAULT_AUTO_MARGIN_DB = 6.0
# With an oracle, the candidate can trail the reference by 3 dB: twice the
# noise power, that is one more rounding as large as the reference error.
DEFAULT_ORACLE_DELTA_DB = -3.0

GATE_MATH_DERIVED = "derived"
GATE_MATH_SIX_GATE_V1 = "six_gate_v1"
_GATE_MATHS = frozenset({GATE_MATH_DERIVED, GATE_MATH_SIX_GATE_V1})

THRESHOLD_SOURCE_EXPLICIT = "explicit"
THRESHOLD_SOURCE_DEFAULT = "default"
THRESHOLD_SOURCE_AUTO_PREFIX = "auto:"

STATUS_DISCRETE_EXACT = "discrete_exact"
STATUS_NO_FINITE_ELEMENTS = "no_finite_elements"
STATUS_EXACT = "exact"
STATUS_OK = "ok"
STATUS_SKIPPED_OVER_BUDGET = "skipped_over_budget"
STATUS_ZERO_SIGNAL = "zero_signal"
STATUS_NON_FINITE_MISMATCH = "non_finite_mismatch"
STATUS_DISCRETE_MISMATCH = "discrete_mismatch"
STATUS_SHAPE_MISMATCH = "shape_mismatch"
STATUS_DTYPE_MISMATCH = "dtype_mismatch"
STATUS_TREE_MISMATCH = "tree_mismatch"
STATUS_ERROR = "error"
# Best to worst. merge_reports keeps the worst status.
STATUS_ORDER = (
    STATUS_DISCRETE_EXACT,
    STATUS_NO_FINITE_ELEMENTS,
    STATUS_EXACT,
    STATUS_OK,
    STATUS_SKIPPED_OVER_BUDGET,
    STATUS_ZERO_SIGNAL,
    STATUS_NON_FINITE_MISMATCH,
    STATUS_DISCRETE_MISMATCH,
    STATUS_SHAPE_MISMATCH,
    STATUS_DTYPE_MISMATCH,
    STATUS_TREE_MISMATCH,
    STATUS_ERROR,
)

# Metric keys. merge_reports takes the minimum of MIN_MERGED_KEYS and the
# maximum of MAX_MERGED_KEYS. The first five names and peak_error_rms_ratio
# match an existing benchmark report schema, so its consumers do not change.
MIN_MERGED_KEYS = (
    "sqnr_db",
    "centered_snr_db",
    "min_block_sqnr_db",
    "min_ac_snr_db",
    "min_delta_sqnr_db",
    "oracle_sqnr_db",
    "reference_oracle_sqnr_db",
    "oracle_delta_db",
    "si_snr_db",
    "cosine_similarity",
)
MAX_MERGED_KEYS = (
    "peak_error_rms_ratio",
    "max_block_noise_ratio_db",
    "max_error_ulp",
)
REPORT_ONLY_KEYS = ("best_fit_scale",)
METRIC_KEYS = MIN_MERGED_KEYS + MAX_MERGED_KEYS + REPORT_ONLY_KEYS

# Elements per tile. The strict path reads and converts one tile of a leaf at
# a time, so peak float64 memory stays bounded for any leaf size.
_TILE_ELEMENTS = 1 << 17

# Block-scaled format name -> (element dtype name, elements per scale block).
# OCP MX v1.0 (Rouhani et al., arXiv:2310.10537) uses an E8M0 scale per 32
# elements. NVFP4 uses an E4M3 scale per 16 elements.
BLOCK_SCALED_FORMATS: types.MappingProxyType[str, tuple[str, int]] = (
    types.MappingProxyType({
        "mxfp4": ("float4_e2m1fn", 32),
        "nvfp4": ("float4_e2m1fn", 16),
        "mxfp8_e4m3": ("float8_e4m3fn", 32),
        "mxfp8_e5m2": ("float8_e5m2", 32),
    })
)

# SQNR in dB of one round-to-nearest into each block-scaled format, from a
# simulation with 2**22 N(0, 1) samples, a float64 oracle and the block sizes
# above. The 6.02 p + 7.44 rule does not include the shared-scale loss.
# Simulation recipe (NumPy and ml_dtypes, seed 0, x = 2**22 N(0, 1) samples,
# reshaped to rows of one block):
#   MX: s = 2**(floor(log2(max|x_block|)) - emax), with emax 2 (float4_e2m1fn),
#     8 (float8_e4m3fn) or 15 (float8_e5m2). q = s * cast(clip(x / s, +-max)).
#   NVFP4: s = cast_e4m3(max|x_block| / 6). q = s * cast_e2m1(clip(x / s, +-6)).
#   SQNR = 10*log10(sum(x**2) / sum((q - x)**2)).
# The recipe gives 18.79, 20.44, 30.65 and 25.36 dB. The same recipe without a
# scale gives float4_e2m1fn 16.34, bfloat16 55.59, float8_e4m3fn 31.52 dB.
_BLOCK_SCALED_SQNR_DB: types.MappingProxyType[str, float] = (
    types.MappingProxyType({
        "mxfp4": 18.8,
        "nvfp4": 20.4,
        "mxfp8_e4m3": 30.7,
        "mxfp8_e5m2": 25.4,
    })
)

# Unscaled formats where the 6.02 p + 7.44 rule is wrong. float4_e2m1fn has
# only 2 exponent bits, so its scale-free region is too narrow and the rule
# overstates it by about 3 dB. The value is the same simulation as above,
# without a scale.
_UNSCALED_SQNR_DB: types.MappingProxyType[str, float] = types.MappingProxyType(
    {"float4_e2m1fn": 16.3}
)

# Widrow and Kollar, "Quantization Noise" (2008): one round-to-nearest into a
# float format with p significand bits (implicit bit included) adds noise of
# power about 0.180 * 2**(-2p) * E[x**2], that is SQNR = 6.02 p + 7.44 dB.
_DB_PER_BIT = 6.02
_DB_OFFSET = 7.44
_FLOAT32_PRECISION_BITS = 24

# Reduction length at which the accumulator term equals one rounding. With
# zero-mean independent terms the running sum s_k has E[s_k**2] = k * E[x**2].
# K roundings of s_1..s_K add noise in proportion to sum(k) = K**2 / 2, and the
# output power is K * E[x**2]. So SQNR_acc = (6.02 p + 7.44) - 10*log10(K / 2).
# This is the probabilistic sqrt(K) growth of Higham and Mary, "A New Approach
# to Probabilistic Rounding Error Analysis", SIAM J. Sci. Comput. 41(5), 2019.
# A sequential bfloat16 accumulation simulation (K = 256, 4096, 65536) agrees
# within 2 dB.
_ACCUMULATOR_K0 = 2


@dataclasses.dataclass(frozen=True)
class PrecisionSpec:
  """Rounding sources that `expected_sqnr_db` turns into an expected SQNR.

  Attributes:
    operand_dtypes: Dtype names of the inputs that the kernel rounds. Names in
      BLOCK_SCALED_FORMATS are accepted too.
    output_dtype: Dtype name of the output. None means the reference dtype.
    reduction_length: Reduction length K of the main contraction, if any.
    accumulator_dtype: Accumulator dtype name. None means float32.
  """

  operand_dtypes: tuple[str, ...] = ()
  output_dtype: str | None = None
  reduction_length: int | None = None
  accumulator_dtype: str | None = None


@dataclasses.dataclass(frozen=True)
class SqnrConfig:
  """Policy for the SQNR checks. Every field has a default.

  Attributes:
    min_sqnr_db: Global SQNR floor in dB (finite and positive), or AUTO. AUTO
      derives the floor from a PrecisionSpec (expected SQNR minus
      auto_margin_db) and falls back to DEFAULT_MIN_SQNR_DB when no format is
      known.
    strict: True runs the anti-cheat checks. False reports core metrics only.
    alpha: False-alarm rate per output on honest rounding noise that the fitted
      noise model describes. The derived check thresholds come from it. See
      DEFAULT_ALPHA for a case outside the model.
    block_size: Elements per block along the minor axis.
    auto_margin_db: Margin below the expected SQNR for AUTO.
    oracle_delta_db: Fail when oracle_delta_db is below this value.
    gate_math: GATE_MATH_DERIVED or GATE_MATH_SIX_GATE_V1.
  """

  min_sqnr_db: float | str = AUTO
  strict: bool = True
  alpha: float = DEFAULT_ALPHA
  block_size: int = DEFAULT_BLOCK_SIZE
  auto_margin_db: float = DEFAULT_AUTO_MARGIN_DB
  oracle_delta_db: float = DEFAULT_ORACLE_DELTA_DB
  gate_math: str = GATE_MATH_DERIVED

  def __post_init__(self):
    th = self.min_sqnr_db
    if isinstance(th, str):
      if th != AUTO:
        raise ValueError(f"min_sqnr_db={th!r} must be a float or {AUTO!r}.")
    else:
      try:
        th_f = float(th)
      except (TypeError, ValueError) as e:
        raise ValueError(f"min_sqnr_db={th!r} must be a float.") from e
      if isinstance(th, bool) or not math.isfinite(th_f) or th_f <= 0.0:
        raise ValueError(f"min_sqnr_db={th!r} must be a positive finite float.")
    if not 0.0 < self.alpha < 1.0:
      raise ValueError(f"alpha={self.alpha!r} must be in (0, 1).")
    if self.block_size < 2:
      raise ValueError(f"block_size={self.block_size!r} must be >= 2.")
    if self.gate_math not in _GATE_MATHS:
      raise ValueError(
          f"gate_math={self.gate_math!r} not in {sorted(_GATE_MATHS)}."
      )

  @classmethod
  def six_gate_v1(cls, **overrides: Any) -> "SqnrConfig":
    """Returns the fixed-constant six-check preset (45 dB, 6 dB margins)."""
    kwargs: dict[str, Any] = dict(
        min_sqnr_db=DEFAULT_MIN_SQNR_DB, gate_math=GATE_MATH_SIX_GATE_V1
    )
    kwargs.update(overrides)
    return cls(**kwargs)

  @classmethod
  def from_options(
      cls,
      *,
      min_sqnr_db: float | str = AUTO,
      alpha: float = DEFAULT_ALPHA,
      strict: bool = True,
      preset: str | None = None,
  ) -> "SqnrConfig":
    """Builds a config from flat CLI or flag values.

    Args:
      min_sqnr_db: A float, a numeric string, or AUTO.
      alpha: See the class attribute.
      strict: See the class attribute.
      preset: None or GATE_MATH_DERIVED for the default math,
        GATE_MATH_SIX_GATE_V1 for the fixed-constant preset.

    Returns:
      The config.

    Raises:
      ValueError: If a value is not valid.
    """
    if isinstance(min_sqnr_db, str) and min_sqnr_db != AUTO:
      try:
        min_sqnr_db = float(min_sqnr_db)
      except ValueError as e:
        raise ValueError(
            f"min_sqnr_db={min_sqnr_db!r} must be a float or {AUTO!r}."
        ) from e
    if preset in (None, GATE_MATH_DERIVED):
      return cls(min_sqnr_db=min_sqnr_db, alpha=alpha, strict=strict)
    if preset == GATE_MATH_SIX_GATE_V1:
      kw: dict[str, Any] = dict(alpha=alpha, strict=strict)
      if min_sqnr_db != AUTO:
        kw["min_sqnr_db"] = min_sqnr_db
      return cls.six_gate_v1(**kw)
    raise ValueError(f"preset={preset!r} not in {sorted(_GATE_MATHS)}.")


@dataclasses.dataclass(frozen=True)
class SqnrLeafReport:
  """SQNR result for one output leaf. Metric fields are None when not measured.

  All dB values use S(a, b) = 10*log10(sum(b**2) / sum((a - b)**2)), with y the
  reference, c the candidate and o the oracle.

  Attributes:
    path: Leaf name.
    dtype: Reference dtype name.
    status: One of STATUS_ORDER.
    passed: The verdict.
    threshold_db: The global SQNR floor.
    threshold_source: THRESHOLD_SOURCE_EXPLICIT, THRESHOLD_SOURCE_DEFAULT, or
      THRESHOLD_SOURCE_AUTO_PREFIX plus the derivation.
    gate_math: The check family of the config.
    sqnr_db: S(c, y) over finite elements.
    centered_snr_db: SNR after removal of the leaf mean of y and of c - y.
    min_block_sqnr_db: Worst S over blocks.
    min_ac_snr_db: Worst high-frequency (AC) SNR.
    min_delta_sqnr_db: Worst S(c - x, y - x) over input views x.
    peak_error_rms_ratio: Peak element error over the allowed noise RMS.
    max_block_noise_ratio_db: Worst block noise ratio (derived checks).
    max_error_ulp: Largest error in ULP (derived checks).
    oracle_sqnr_db: S(c, o).
    reference_oracle_sqnr_db: S(y, o).
    oracle_delta_db: oracle_sqnr_db - reference_oracle_sqnr_db.
    best_fit_scale: <c, y> / <y, y>.
    si_snr_db: Scale-invariant SNR of c against y.
    cosine_similarity: <c, y> / (|c| |y|). Report only.
    failed_reasons: One message per failed check.
    warnings: Messages that do not change the verdict.
  """

  path: str
  dtype: str
  status: str
  passed: bool
  threshold_db: float | None
  threshold_source: str
  gate_math: str
  sqnr_db: float | None = None
  centered_snr_db: float | None = None
  min_block_sqnr_db: float | None = None
  min_ac_snr_db: float | None = None
  min_delta_sqnr_db: float | None = None
  peak_error_rms_ratio: float | None = None
  max_block_noise_ratio_db: float | None = None
  max_error_ulp: float | None = None
  oracle_sqnr_db: float | None = None
  reference_oracle_sqnr_db: float | None = None
  oracle_delta_db: float | None = None
  best_fit_scale: float | None = None
  si_snr_db: float | None = None
  cosine_similarity: float | None = None
  failed_reasons: tuple[str, ...] = ()
  warnings: tuple[str, ...] = ()

  def to_dict(self) -> dict[str, Any]:
    """Returns a JSON-safe dict. The status key is named `sqnr_status`."""
    d = dataclasses.asdict(self)
    d["sqnr_status"] = d.pop("status")
    d["failed_reasons"] = list(self.failed_reasons)
    d["warnings"] = list(self.warnings)
    return d


def default_read(x: Any, sl: tuple[slice, ...]) -> np.ndarray:
  """Returns x[sl] as a host NumPy array."""
  return np.asarray(x[sl] if sl else x)


def _precision_bits(dtype_name: str) -> int | None:
  try:
    return int(ml_dtypes.finfo(np.dtype(dtype_name)).nmant) + 1
  except (TypeError, ValueError):
    return None


def _format_sqnr_db(name: str) -> float | None:
  """Returns the SQNR of one rounding into a named format, or None."""
  key = name.lower()
  if key in _BLOCK_SCALED_SQNR_DB:
    return _BLOCK_SCALED_SQNR_DB[key]
  p = _precision_bits(name)
  if p is None or p >= _FLOAT32_PRECISION_BITS:
    return None
  canon = str(np.dtype(name))
  if canon in _UNSCALED_SQNR_DB:
    return _UNSCALED_SQNR_DB[canon]
  return _DB_PER_BIT * p + _DB_OFFSET


def expected_sqnr_db(precision: PrecisionSpec) -> tuple[float, str] | None:
  """Returns (expected SQNR in dB, derivation), or None if no format is known.

  Each rounding source is one round-to-nearest into a format narrower than
  float32: each operand dtype, the output dtype, and the accumulator. Format
  SQNR is 6.02 p + 7.44 dB (Widrow and Kollar) for p significand bits, or the
  simulated value for block-scaled formats and float4_e2m1fn. Independent
  sources add in noise power: E = -10*log10(sum(10**(-S_i / 10))). An
  accumulator narrower than float32 with reduction length K adds the source
  S_acc - 10*log10(K / K0) with K0 = 2 (probabilistic sqrt(K) growth, Higham
  and Mary 2019). Names that are not known formats are skipped.

  Args:
    precision: The rounding sources.

  Returns:
    (E, derivation). The derivation lists each term as `name:dB` joined by
    `+`, then `=E dB`. None when no source is narrower than float32.
  """
  names = list(precision.operand_dtypes)
  if precision.output_dtype:
    names.append(precision.output_dtype)
  terms: list[tuple[str, float]] = []
  for name in names:
    db = _format_sqnr_db(name)
    if db is not None:
      terms.append((name, db))
  if precision.accumulator_dtype:
    db = _format_sqnr_db(precision.accumulator_dtype)
    if db is not None:
      label = f"acc_{precision.accumulator_dtype}"
      k = precision.reduction_length
      if k is not None and k > _ACCUMULATOR_K0:
        db -= 10.0 * math.log10(k / _ACCUMULATOR_K0)
        label += f"(K={k},K0={_ACCUMULATOR_K0})"
      terms.append((label, db))
  if not terms:
    return None
  noise = sum(10.0 ** (-db / 10.0) for _, db in terms)
  e_db = -10.0 * math.log10(noise)
  derivation = "+".join(f"{n}:{db:.1f}" for n, db in terms)
  return e_db, f"{derivation}={e_db:.1f}dB"


def _integer_operands(precision: PrecisionSpec | None) -> tuple[str, ...]:
  """Returns the operand dtype names of precision that are integer formats."""
  if precision is None:
    return ()
  out = []
  for name in precision.operand_dtypes:
    if name.lower() in BLOCK_SCALED_FORMATS:
      continue
    try:
      dt = np.dtype(name)
    except TypeError:
      continue
    if dt != np.bool_ and _is_discrete(dt):
      out.append(name)
  return tuple(out)


def resolve_threshold(
    config: SqnrConfig, precision: PrecisionSpec | None = None
) -> tuple[float, str]:
  """Returns (threshold_db, threshold_source) for a config.

  AUTO does not model integer quantization. With an integer operand (for
  example int8 or int4), AUTO falls back to DEFAULT_MIN_SQNR_DB, and
  `audit_leaf` adds a warning that asks for an explicit min_sqnr_db.

  Args:
    config: The policy.
    precision: Rounding sources for AUTO.
  """
  if config.min_sqnr_db != AUTO:
    return float(config.min_sqnr_db), THRESHOLD_SOURCE_EXPLICIT
  if _integer_operands(precision):
    return DEFAULT_MIN_SQNR_DB, THRESHOLD_SOURCE_DEFAULT
  exp = expected_sqnr_db(precision) if precision is not None else None
  if exp is None:
    return DEFAULT_MIN_SQNR_DB, THRESHOLD_SOURCE_DEFAULT
  e_db, derivation = exp
  margin = config.auto_margin_db
  return (
      e_db - margin,
      f"{THRESHOLD_SOURCE_AUTO_PREFIX}{derivation}-{margin:g}dB",
  )


def _dtype_name(dtype: Any) -> str:
  """Returns the canonical name of a dtype, or a format name unchanged."""
  if isinstance(dtype, str) and dtype.lower() in BLOCK_SCALED_FORMATS:
    return dtype.lower()
  return str(np.dtype(dtype))


def _narrower_than_float32(name: str) -> bool:
  """True for block-scaled formats, integer formats and narrow floats."""
  if name in BLOCK_SCALED_FORMATS:
    return True
  dt = np.dtype(name)
  if dt == np.bool_:
    return False
  if _is_discrete(dt):
    return True
  p = _precision_bits(name)
  return p is not None and p < _FLOAT32_PRECISION_BITS


def precision_from_dtypes(
    reference_dtype: Any,
    candidate_dtype: Any,
    input_dtypes: Sequence[Any] = (),
    accumulator_dtype: Any = None,
    reduction_length: int | None = None,
) -> PrecisionSpec:
  """Returns the PrecisionSpec of a kernel from its dtypes.

  Use this one rule wherever a PrecisionSpec is built, so that every caller
  gets the same AUTO floor:
    - operand_dtypes: the input dtypes narrower than float32, in order. These
      include integer formats, so that AUTO falls back to the default floor
      with a warning. Wider inputs (float32, float64) add no rounding source.
    - output_dtype: the candidate dtype, which is the format that the kernel
      rounds its output to.
    - accumulator_dtype and reduction_length: as given.

  Args:
    reference_dtype: Dtype of the reference output. It is not a rounding source
      of the candidate, and only its validity is checked.
    candidate_dtype: Dtype of the candidate output.
    input_dtypes: Dtypes of the kernel inputs. Names in BLOCK_SCALED_FORMATS are
      accepted too.
    accumulator_dtype: Accumulator dtype, or None for float32.
    reduction_length: Reduction length K of the main contraction, or None.

  Returns:
    The PrecisionSpec.

  Raises:
    TypeError: If a dtype is not valid.
  """
  _dtype_name(reference_dtype)
  operands = tuple(
      name
      for name in (_dtype_name(d) for d in input_dtypes)
      if _narrower_than_float32(name)
  )
  return PrecisionSpec(
      operand_dtypes=operands,
      output_dtype=_dtype_name(candidate_dtype),
      reduction_length=reduction_length,
      accumulator_dtype=(
          None if accumulator_dtype is None else _dtype_name(accumulator_dtype)
      ),
  )


def _db(num: float, den: float) -> float:
  if den <= 0.0:
    return math.inf
  return -math.inf if num <= 0.0 else 10.0 * (math.log10(num) - math.log10(den))


def _min_db(num: np.ndarray, den: np.ndarray) -> float:
  with np.errstate(all="ignore"):
    v = 10.0 * np.log10(np.maximum(num, 1e-300)) - 10.0 * np.log10(den)
  return float(np.min(v))


def _finite_or_none(v: float | None) -> float | None:
  return float(v) if v is not None and math.isfinite(v) else None


def _is_discrete(dt: Any) -> bool:
  """True for bool and integer dtypes, including ml_dtypes int2 and int4."""
  dt = np.dtype(dt)
  if dt == np.bool_:
    return True
  try:
    ml_dtypes.iinfo(dt)
  except (TypeError, ValueError):
    return False
  return True


def _is_complex(dt: Any) -> bool:
  return bool(np.issubdtype(np.dtype(dt), np.complexfloating))


def _eps_sq(dt: Any) -> float | None:
  """Returns finfo(dt).eps**2 for float and complex dtypes, else None."""
  if _is_discrete(dt):
    return None
  try:
    return float(ml_dtypes.finfo(np.dtype(dt)).eps) ** 2
  except (TypeError, ValueError):
    return None


def _shape_dtype(
    x: Any, read: Callable[..., Any]
) -> tuple[tuple[int, ...], np.dtype]:
  """Returns (shape, dtype) of x without a full read when x has metadata."""
  shape = getattr(x, "shape", None)
  dtype = getattr(x, "dtype", None)
  if shape is None or dtype is None:
    a = np.asarray(x)
    return a.shape, a.dtype
  shape = tuple(int(d) for d in shape)
  try:
    return shape, np.dtype(dtype)
  except TypeError:
    probe = tuple(slice(0, min(1, d)) for d in shape)
    return shape, np.asarray(read(x, probe)).dtype


@dataclasses.dataclass(frozen=True, eq=False)
class _SwappedView:
  """`base` with its two minor axes swapped. `_Reader` reads it lazily."""

  base: Any
  shape: tuple[int, ...]
  dtype: Any


class _Reader:
  """Wraps a `read` callback so that it also reads `_SwappedView` objects."""

  def __init__(self, read: Callable[[Any, tuple[slice, ...]], Any]):
    self._read = read

  def __call__(self, x: Any, sl: tuple[slice, ...]) -> np.ndarray:
    if isinstance(x, _SwappedView):
      if sl:
        sl = tuple(sl) + (slice(None),) * (len(x.shape) - len(sl))
        sl = sl[:-2] + (sl[-1], sl[-2])
      return np.swapaxes(np.asarray(self._read(x.base, sl)), -2, -1)
    return np.asarray(self._read(x, sl))


def _reader(
    read: Callable[[Any, tuple[slice, ...]], Any],
) -> Callable[[Any, tuple[slice, ...]], np.ndarray]:
  return read if isinstance(read, _Reader) else _Reader(read)


def _tile_slices(
    shape: tuple[int, ...], block: int
) -> tuple[list[tuple[slice, ...]], list[tuple[slice, ...]]]:
  """Returns disjoint and 1-element-overlapping tile slices for shape.

  A leaf with at most _TILE_ELEMENTS elements is one tile, given as (). Larger
  leaves are split along the leading axes first, then along the two minor
  axes (larger first), with steps rounded down to a multiple of `block`. The
  overlapping tiles extend one element back along the last two split axes, so
  neighbor checks see each tile edge. A short last tile extends back to one
  full block.

  Args:
    shape: Shard shape.
    block: Block size along the minor axes.
  """
  if math.prod(shape) <= _TILE_ELEMENTS:
    return [()], [()]
  ndim = len(shape)
  steps = list(shape)
  order = list(range(max(0, ndim - 2))) + sorted(
      range(max(0, ndim - 2), ndim), key=lambda k: shape[k], reverse=True
  )
  for k in order:
    cur = math.prod(steps)
    if cur <= _TILE_ELEMENTS:
      break
    rem = cur // max(steps[k], 1)
    q = max(1, _TILE_ELEMENTS // max(rem, 1))
    if q < steps[k]:
      steps[k] = max(block, (q // block) * block) if q >= block else q
  gt1 = [i for i, s in enumerate(steps) if s > 1]
  ov_set = set(gt1[-2:])
  if len(ov_set) < 2:
    split = [i for i, (s, d) in enumerate(zip(steps, shape)) if s < d]
    if split:
      ov_set.add(split[-1])
  d_axes, o_axes = [], []
  for k, (dim, step) in enumerate(zip(shape, steps)):
    if step >= dim:
      d_axes.append((slice(None),))
      o_axes.append((slice(None),))
      continue
    ds, os_k = [], []
    for i0 in range(0, dim, step):
      i1 = min(i0 + step, dim)
      if i1 - i0 < block <= dim and step >= block:
        ov0 = dim - block
      elif k in ov_set:
        ov0 = max(0, i0 - 1)
      else:
        ov0 = i0
      ds.append(slice(i0, i1))
      os_k.append(slice(ov0, i1))
    d_axes.append(tuple(ds))
    o_axes.append(tuple(os_k))
  return list(itertools.product(*d_axes)), list(itertools.product(*o_axes))


def _leaf_tiles(
    shapes: Sequence[tuple[int, ...]], block: int
) -> tuple[
    list[tuple[int, tuple[slice, ...]]], list[tuple[int, tuple[slice, ...]]]
]:
  """Returns (shard index, slices) disjoint and overlapping tiles of a leaf."""
  d_tiles: list[tuple[int, tuple[slice, ...]]] = []
  o_tiles: list[tuple[int, tuple[slice, ...]]] = []
  for s_i, shape in enumerate(shapes):
    ds, os_i = _tile_slices(shape, block)
    d_tiles.extend((s_i, sl) for sl in ds)
    o_tiles.extend((s_i, sl) for sl in os_i)
  return d_tiles, o_tiles


@dataclasses.dataclass(frozen=True)
class LeafContext:
  """Data that the checks read. Built once per leaf by `_build_context`.

  Sums are float64 over elements where both reference and candidate are
  finite, after scaling by 2**-shift. Complex leaves are split into a trailing
  axis of size 2 (real, imag) by `as_real`. A "component" is one entry of that
  axis. An "element" is finite when all its components are finite.

  Attributes:
    pairs: The (reference, candidate) shards.
    read: The accessor from `audit_leaf`, wrapped so that it also reads the lazy
      transposed input views. Use it for every array access.
    disjoint_tiles: (shard index, slices) that cover each shard once.
    overlap_tiles: (shard index, slices) with a 1-element overlap on the two
      minor split axes, for checks that look at neighbors.
    input_views: Input views that pass the filter (one array per shard each,
      same shard shapes, float or discrete dtype, same complexness). For a
      single-shard leaf, a square view is followed by its lazy transpose.
    oracle_pairs: Oracle shards, or None.
    ref_dtype: Reference dtype.
    cand_dtype: Candidate dtype.
    is_complex: True for complex leaves.
    eps_sq: finfo(ref_dtype).eps ** 2.
    full_n: Total element count over shards.
    shift: Power-of-two prescale exponent (frexp of the max magnitude of the
      reference and the candidate).
    sig: Sum of y**2 (y = scaled reference), over finite components.
    err: Sum of e**2 (e = scaled candidate - y), over finite components.
    n_finite: Count of finite scalar components.
    max_abs_err: Max |e| (scaled), over finite components.
    sqnr_db: 10 log10(sig / err).
    threshold_db: The resolved global floor.
    precision: The PrecisionSpec used for the threshold.
    block_size: Block size from the config. The tiles align to it.
    n_elements: Count of finite elements.
    sig_el: Sum of y**2 over finite elements.
    err_el: Sum of e**2 over finite elements.
    centered_sig: Sum over components of the centered sum of squares of y over
      finite elements (n_elements * variance).
    centered_cand: The same for the scaled candidate c.
    centered_err: The same for e.
    gmsq: Mean squared element magnitude of y: sig_el / n_elements, or sig /
      n_finite when no element is finite in all components.
    rel_eac: Centered error energy over sig_el (err / sig as a fallback). Equals
      err_el / sig_el when n_elements <= 1.
    sum_ey: Sum of e * y over finite components.
    ref_max: Max |reference| over finite components (unscaled).
    cand_max: Max |candidate| over finite components (unscaled).
  """

  pairs: Sequence[tuple[Any, Any]]
  read: Callable[[Any, tuple[slice, ...]], np.ndarray]
  disjoint_tiles: Sequence[tuple[int, tuple[slice, ...]]]
  overlap_tiles: Sequence[tuple[int, tuple[slice, ...]]]
  input_views: Sequence[Sequence[Any]]
  oracle_pairs: Sequence[Any] | None
  ref_dtype: Any
  cand_dtype: Any
  is_complex: bool
  eps_sq: float
  full_n: int
  shift: int
  sig: float
  err: float
  n_finite: int
  max_abs_err: float
  sqnr_db: float
  threshold_db: float
  precision: "PrecisionSpec"
  block_size: int = DEFAULT_BLOCK_SIZE
  n_elements: int = 0
  sig_el: float = 0.0
  err_el: float = 0.0
  centered_sig: float = 0.0
  centered_cand: float = 0.0
  centered_err: float = 0.0
  gmsq: float = 0.0
  rel_eac: float = 0.0
  sum_ey: float = 0.0
  ref_max: float = 0.0
  cand_max: float = 0.0


@dataclasses.dataclass(frozen=True)
class CheckOutcome:
  """What a check family returns.

  Attributes:
    metrics: Values for any of METRIC_KEYS that the family measures.
    reasons: One message per failed check, prefixed with the leaf name.
  """

  metrics: dict[str, float | None] = dataclasses.field(default_factory=dict)
  reasons: tuple[str, ...] = ()


def as_real(a: Any) -> np.ndarray:
  """Returns float64 data with a trailing (real[, imag]) axis."""
  with np.errstate(all="ignore"):
    if np.iscomplexobj(a):
      c = np.asarray(a, dtype=np.complex128)
      return np.stack([np.real(c), np.imag(c)], axis=-1)
    return np.asarray(a, dtype=np.float64)[..., None]


@dataclasses.dataclass(frozen=True)
class _LeafScan:
  """Result of the first pass over a leaf: finite counts and magnitudes."""

  n_finite: int
  n_elements: int
  ref_max: float
  cand_max: float


def _scan_leaf(
    pairs: Sequence[tuple[Any, Any]],
    read: Callable[[Any, tuple[slice, ...]], np.ndarray],
    d_tiles: Sequence[tuple[int, tuple[slice, ...]]],
) -> _LeafScan | None:
  """First pass. Returns None when the non-finite masks differ.

  NaN, +Inf and -Inf must sit at the same components in both arrays.

  Args:
    pairs: The (reference, candidate) shards.
    read: The accessor.
    d_tiles: Disjoint tiles of the leaf.

  Returns:
    The scan, or None when the non-finite masks differ.
  """
  n_f = n_el = 0
  r_max = c_max = 0.0
  eq = np.array_equal
  for s_i, d_sl in d_tiles:
    rc = as_real(read(pairs[s_i][0], d_sl))
    cc = as_real(read(pairs[s_i][1], d_sl))
    if not (
        eq(np.isnan(cc), np.isnan(rc))
        and eq(np.isposinf(cc), np.isposinf(rc))
        and eq(np.isneginf(cc), np.isneginf(rc))
    ):
      return None
    f = np.isfinite(rc) & np.isfinite(cc)
    if cnt := int(np.count_nonzero(f)):
      n_f += cnt
      n_el += int(np.count_nonzero(np.all(f, -1)))
      r_max = max(r_max, float(np.max(np.abs(rc[f]))))
      c_max = max(c_max, float(np.max(np.abs(cc[f]))))
  return _LeafScan(n_f, n_el, r_max, c_max)


def _filter_views(
    input_views: Sequence[Sequence[Any]],
    shapes: Sequence[tuple[int, ...]],
    is_complex: bool,
    full_n: int,
    read: Callable[[Any, tuple[slice, ...]], np.ndarray],
) -> list[list[Any]]:
  """Returns the input views that the input-delta checks can use.

  A view qualifies when it has one array per shard with the shard shapes, a
  float dtype (or a discrete dtype on a leaf of more than one element), and
  the leaf complexness. For a single-shard leaf, a qualifying view with equal
  minor dimensions is followed by its lazy transpose.

  Args:
    input_views: Candidate views, one array per shard each.
    shapes: Shard shapes of the leaf.
    is_complex: Leaf complexness.
    full_n: Element count of the leaf.
    read: The accessor.
  """
  out: list[list[Any]] = []
  for view in input_views:
    arrs = list(view)
    if len(arrs) != len(shapes):
      continue
    try:
      metas = [_shape_dtype(a, read) for a in arrs]
    except (TypeError, ValueError):
      continue
    if any(m[0] != s for m, s in zip(metas, shapes)):
      continue
    dt = metas[0][1]
    ok_dt = _eps_sq(dt) is not None or (_is_discrete(dt) and full_n > 1)
    if not ok_dt or _is_complex(dt) != is_complex:
      continue
    out.append(arrs)
    shp = metas[0][0]
    if len(arrs) == 1 and len(shp) >= 2 and shp[-2] == shp[-1] > 1:
      out.append([_SwappedView(arrs[0], shp, dt)])
  return out


def _build_context(
    pairs: Sequence[tuple[Any, Any]],
    read: Callable[[Any, tuple[slice, ...]], np.ndarray],
    input_views: Sequence[Sequence[Any]],
    oracle_pairs: Sequence[Any] | None,
    threshold_db: float,
    precision: PrecisionSpec,
    *,
    block_size: int = DEFAULT_BLOCK_SIZE,
    scan: _LeafScan | None = None,
) -> LeafContext:
  """Builds the LeafContext with one tiled pass of prescaled float64 sums.

  The caller has applied the leaf rules: shapes match, dtypes are float or
  complex and agree, the non-finite masks match, and the reference has a
  nonzero finite element.

  Args:
    pairs: The (reference, candidate) shards.
    read: The accessor.
    input_views: Candidate input views. They are filtered here.
    oracle_pairs: Oracle shards, or None.
    threshold_db: The resolved global floor.
    precision: The PrecisionSpec used for the threshold.
    block_size: Block size of the config.
    scan: The first-pass result, if the caller has it.

  Returns:
    The context.
  """
  read = _reader(read)
  metas = [_shape_dtype(r, read) for r, _ in pairs]
  shapes = [m[0] for m in metas]
  ref_dt = metas[0][1]
  cand_dt = _shape_dtype(pairs[0][1], read)[1]
  is_cplx = _is_complex(ref_dt)
  d_tiles, o_tiles = _leaf_tiles(shapes, block_size)
  if scan is None:
    scan = _scan_leaf(pairs, read, d_tiles)
    if scan is None:
      raise ValueError("non-finite masks differ")
  full_n = sum(math.prod(s) for s in shapes)
  eps_sq = _eps_sq(ref_dt)
  sh = math.frexp(max(scan.ref_max, scan.cand_max))[1]
  c_dim = 2 if is_cplx else 1
  sig = err = sig_el = err_el = max_e = sum_ey = 0.0
  # Chan et al. pairwise update of the centered sums of squares of (y, c, e)
  # per component. Each tile is shifted by its own minimum first.
  cn = 0
  ca = cmu = cm2 = np.zeros((3, c_dim))
  for s_i, d_sl in d_tiles:
    rc = as_real(read(pairs[s_i][0], d_sl))
    cc = as_real(read(pairs[s_i][1], d_sl))
    f = np.isfinite(rc) & np.isfinite(cc)
    if not np.any(f):
      continue
    yu = np.where(f, np.ldexp(rc, -sh), 0.0)
    cu = np.where(f, np.ldexp(cc, -sh), 0.0)
    eu = np.where(f, cu - yu, 0.0)
    max_e = max(max_e, float(np.max(np.abs(eu))))
    sig += float(np.sum(np.square(yu)))
    err += float(np.sum(np.square(eu)))
    sum_ey += float(np.sum(eu * yu))
    fel = np.all(f, -1)
    if np.any(fel):
      ye, ce, ee = yu[fel], cu[fel], eu[fel]
      sig_el += float(np.sum(np.square(ye)))
      err_el += float(np.sum(np.square(ee)))
      nt, st3 = ye.shape[0], np.stack([ye, ce, ee], axis=0)
      at = np.min(st3, axis=1)
      dv = st3 - at[:, None, :]
      mut = np.mean(dv, axis=1)
      m2t = np.sum(np.square(dv - mut[:, None, :]), axis=1)
      if cn == 0:
        cn, ca, cmu, cm2 = nt, at, mut, m2t
      else:
        nn, dl = cn + nt, (at - ca) + (mut - cmu)
        cmu = cmu + dl * (nt / nn)
        cm2 = cm2 + m2t + np.square(dl) * (cn * nt / nn)
        cn = nn
  n_el = scan.n_elements
  eac = float(np.sum(cm2[2])) if n_el > 1 else err_el
  if n_el > 0:
    gmsq = sig_el / float(n_el)
    rel_eac = eac / max(sig_el, 1e-300)
  else:
    gmsq = sig / float(scan.n_finite)
    rel_eac = err / max(sig, 1e-300)
  return LeafContext(
      pairs=pairs,
      read=read,
      disjoint_tiles=d_tiles,
      overlap_tiles=o_tiles,
      input_views=_filter_views(input_views, shapes, is_cplx, full_n, read),
      oracle_pairs=oracle_pairs,
      ref_dtype=ref_dt,
      cand_dtype=cand_dt,
      is_complex=is_cplx,
      eps_sq=0.0 if eps_sq is None else eps_sq,
      full_n=full_n,
      shift=sh,
      sig=sig,
      err=err,
      n_finite=scan.n_finite,
      max_abs_err=max_e,
      sqnr_db=_db(sig, err),
      threshold_db=threshold_db,
      precision=precision,
      block_size=block_size,
      n_elements=n_el,
      sig_el=sig_el,
      err_el=err_el,
      centered_sig=float(np.sum(cm2[0])),
      centered_cand=float(np.sum(cm2[1])),
      centered_err=float(np.sum(cm2[2])),
      gmsq=gmsq,
      rel_eac=rel_eac,
      sum_ey=sum_ey,
      ref_max=scan.ref_max,
      cand_max=scan.cand_max,
  )


# --- BEGIN SIX_GATE_V1 CHECKS (port of the fixed-constant gates) ---
# Fixed constants of the six_gate_v1 preset. The names describe each use.
_V1_BLOCK_MARGIN_DB = 6.0  # Block floor: threshold_db - 6 dB.
_V1_AC_MIN_DB = 6.0  # AC survival: every AC SNR must reach 6 dB.
_V1_CENTERED_MIN_DB = 6.0  # Lowest centered floor.
_V1_CENTERED_CORR_DB = 0.5  # Correlation floor when DC dominates.
_V1_MACRO_WINDOW = 1024  # Elements per macro-correlation window.
_V1_MACRO_CORR_DB = 0.2  # Macro-window correlation floor.
_V1_SLICE_CORR_DB = 0.5  # Quiet-slice and complex-part correlation floor.
_V1_PEAK_MAX = 16.0  # Peak error / allowed noise RMS cap.
_V1_FLOOR_FRACTION = 0.0625  # Block signal floor, as a fraction of gmsq.
_V1_DEAD_FRACTION = 1e-12  # Macro windows below this gmsq fraction are dead.
_V1_SLICE_ABS = 1e-12  # Absolute slice floor, as a fraction of gmsq.
_V1_RESOLVE_EPS_FACTOR = 32.0  # Window variance must exceed 32 eps^2 sig.
_V1_RESOLVE_REL = 1e-26  # Variance floor relative to the deviation energy.
_V1_EPS_FACTOR = 16.0  # 16 eps^2: the reference dtype resolution floor.
_V1_TNF_FACTOR = 4.0  # Noise-floor multiple for "high variance" tests.
_V1_REL_EAC_AC = 8.0  # AC window gate: var > min(4 tnf, 8 rel_eac) sig.
# A quarter of the honest centered noise. Gates the macro windows, the
# plaquettes and the centered check.
_V1_REL_EAC_LOW = 0.25
_V1_SLICE_MIN_SAMPLES = 32.0  # Slice correlation band: sqrt(32 / selem).
_V1_SLICE_REL = 0.25  # Lowest slice floor, as a fraction of rel_eac.
_V1_PLAQ_DIFF_RATIO = 64.0  # Plaquette: neighbor energy over 64 x variance.
_V1_COMPLEX_REL = 0.6  # Complex-part floor, as a fraction of rel_eac.
_V1_COMPLEX_ABS = 0.5e-12  # Absolute complex-part floor.
_V1_DELTA_P_FL_DB = 6.0205999  # Input views must be at least 1 bit from y.
_V1_DELTA_MAX_DB = 20.0  # Highest input-delta floor.
_V1_DELTA_MIN_DB = 6.0  # Input-delta floor far from the dtype limit.
_V1_DELTA_LIMIT_DB = 3.0  # Input-delta floor near the dtype limit.


def _v1_chunks(a: np.ndarray, w: int) -> np.ndarray:
  nf = (a.shape[0] // w) * w
  out = a[:nf].reshape(-1, w, a.shape[1])
  if nf == a.shape[0]:
    return out
  return np.concatenate([out, a[-w:].reshape(1, w, a.shape[1])], axis=0)


def _v1_win_stats(
    ycf: np.ndarray, ecf: np.ndarray, w: int, eps_sq: float, gmsq: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Per-window signal, error, centered signal, centered error, resolved."""
  yc, ec = _v1_chunks(ycf, w), _v1_chunks(ecf, w)
  sig, err = np.sum(np.square(yc), (1, 2)), np.sum(np.square(ec), (1, 2))
  d = yc - np.min(yc, axis=1, keepdims=True)
  var = np.sum(np.square(d - np.mean(d, 1, keepdims=True)), (1, 2))
  dev = np.sum(np.square(d), (1, 2))
  eac = np.sum(np.square(ec - np.mean(ec, 1, keepdims=True)), (1, 2))
  fl = np.maximum(
      _V1_RESOLVE_EPS_FACTOR * eps_sq * np.maximum(sig, w * gmsq),
      _V1_RESOLVE_REL * dev,
  )
  res = (var > fl) & (eac > 0.0)
  return sig, err, var, eac, res


def _v1_spatial_tile(
    r_arr: np.ndarray,
    c_arr: np.ndarray,
    ctx: LeafContext,
    target_db: float,
) -> tuple[float, float | None, bool, tuple[float, float] | None, float]:
  """Block, AC, correlation and peak metrics on one overlapping tile.

  Args:
    r_arr: Reference tile.
    c_arr: Candidate tile.
    ctx: The leaf context.
    target_db: The global floor.

  Returns:
    (min block SQNR, min AC SNR or None, AC survival, worst correlation
    (value, floor) below its floor or None, peak error / allowed noise RMS).
  """
  rc, cc = as_real(r_arr), as_real(c_arr)
  f = np.isfinite(rc) & np.isfinite(cc)
  if not np.any(f):
    return math.inf, None, True, None, 0.0
  sh, gmsq, rel_eac, g_sqnr = ctx.shift, ctx.gmsq, ctx.rel_eac, ctx.sqnr_db
  eps_sq, full_n, bs = ctx.eps_sq, ctx.full_n, ctx.block_size
  yu = np.where(f, np.ldexp(rc, -sh), 0.0)
  cu = np.where(f, np.ldexp(cc, -sh), 0.0)
  eu = np.where(f, cu - yu, 0.0)
  fel = np.all(f, -1)
  n_el = int(np.count_nonzero(fel))
  if float(np.max(np.abs(eu))) == 0.0 or n_el == 0:
    return math.inf, None, True, None, 0.0

  ye, ce, ee = yu[fel], cu[fel], eu[fel]
  sig_el, c_dim = float(np.sum(np.square(ye))), yu.shape[-1]
  fmsq = _V1_FLOOR_FRACTION * gmsq
  dmsq = _V1_DEAD_FRACTION * gmsq
  tnf = 10.0 ** (-target_db / 10.0)
  all_f = bool(np.all(fel))
  yo = yu if all_f else np.where(fel[..., None], yu, 0.0)
  co = cu if all_f else np.where(fel[..., None], cu, 0.0)
  eo = eu if all_f else np.where(fel[..., None], eu, 0.0)
  axes = [k for k in range(fel.ndim) if fel.shape[k] > 1]
  block_db = math.inf if axes else g_sqnr
  ac_db: float | None = None
  ac_ok = True
  m_cdb: tuple[float, float] | None = None
  ng_f = min(_V1_TNF_FACTOR * tnf, _V1_REL_EAC_AC * rel_eac)

  def _upd(val: float):
    nonlocal ac_db, ac_ok
    ac_db = val if ac_db is None else min(ac_db, val)
    ac_ok = ac_ok and (val >= _V1_AC_MIN_DB)

  def _upd_corr(val: float, fl: float):
    nonlocal m_cdb
    if val < fl and (m_cdb is None or (val - fl) < (m_cdb[0] - m_cdb[1])):
      m_cdb = (val, fl)

  mw = _V1_MACRO_WINDOW
  for k in axes:
    fl = fel.shape[k]
    fb = np.moveaxis(fel, k, -1).reshape(-1, fl)
    yb = np.moveaxis(yo, k, -2).reshape(-1, fl, c_dim)
    cb = np.moveaxis(co, k, -2).reshape(-1, fl, c_dim)
    eb = np.moveaxis(eo, k, -2).reshape(-1, fl, c_dim)
    if all_f:
      ycf, ccf, ecf = (t.reshape(-1, c_dim) for t in (yb, cb, eb))
    else:
      ycf, ccf, ecf = yb[fb], cb[fb], eb[fb]
    n = ycf.shape[0]
    if n < bs or (n == bs and full_n <= bs):
      if n > 0 and (et := float(np.sum(np.square(ecf)))) > 0.0:
        st_b = max(float(np.sum(np.square(ycf))), n * fmsq)
        block_db = min(block_db, _db(st_b, et))
    else:
      bsig, berr, bvar, beac, res = _v1_win_stats(ycf, ecf, bs, eps_sq, gmsq)
      if np.any(act := berr > 0.0):
        num = np.maximum(bsig[act], bs * fmsq)
        block_db = min(block_db, _min_db(num, berr[act]))
      res = res & (bvar > ng_f * np.maximum(bsig, bs * gmsq))
      if np.any(res):
        _upd(_min_db(bvar[res], beac[res]))
      if n >= mw:
        msig, _, mvar, meac, mres = _v1_win_stats(ycf, ecf, mw, eps_sq, gmsq)
        cd = _v1_chunks(ccf, mw)
        cd = cd - np.min(cd, axis=1, keepdims=True)
        mcvar = np.sum(np.square(cd - np.mean(cd, 1, keepdims=True)), (1, 2))
        m_th = min(
            _V1_TNF_FACTOR * tnf, _V1_REL_EAC_LOW * rel_eac
        ) * np.maximum(msig, mw * gmsq)
        mres &= (msig > mw * dmsq) & (mvar > m_th)
        if np.any(mres):
          mcov = np.maximum(0.5 * (mvar[mres] + mcvar[mres] - meac[mres]), 0.0)
          mden = np.maximum(mcvar[mres], 1e-300)
          mp = np.where(
              mcvar[mres] > 0.0, mvar[mres] - mcov * mcov / mden, mvar[mres]
          )
          _upd_corr(_min_db(mvar[mres], np.maximum(mp, 0.0)), _V1_MACRO_CORR_DB)
    if len(axes) >= 2 and (selem := n_el // fl) >= bs:
      ss, se = np.sum(np.square(yb), (0, 2)), np.sum(np.square(eb), (0, 2))
      # Honest noise correlates with a short quiet slice by chance. With r the
      # slice signal over the mean noise energy, the 0.5 dB band below fails
      # when the noise projection u ~ N(0, 1 / selem) is under 0.35 - sqrt(r).
      # r >= sqrt(32 / selem) keeps that at least 3.4 sigma away.
      q_fl = (
          max(_V1_SLICE_REL, math.sqrt(_V1_SLICE_MIN_SAMPLES / selem)) * rel_eac
      )
      sl_fl = selem * max(_V1_EPS_FACTOR * eps_sq, q_fl, _V1_SLICE_ABS) * gmsq
      if np.any(lv := (se > 0.0) & (ss > sl_fl)):
        hi = lv & (ss > selem * _V1_TNF_FACTOR * max(tnf, rel_eac) * gmsq)
        if np.any(hi):
          _upd(_min_db(ss[hi], se[hi]))
        if np.any(lo := lv & ~hi):
          sc = np.sum(np.square(cb), (0, 2))
          scov = np.maximum(0.5 * (ss[lo] + sc[lo] - se[lo]), 0.0)
          sden = np.maximum(sc[lo], 1e-300)
          sp = np.where(sc[lo] > 0.0, ss[lo] - scov * scov / sden, ss[lo])
          _upd_corr(_min_db(ss[lo], np.maximum(sp, 0.0)), _V1_SLICE_CORR_DB)

  if len(axes) >= 2:
    ax0, ax1 = axes[-2], axes[-1]
    sp = [ax for ax in range(fel.ndim) if ax not in (ax0, ax1)] + [ax0, ax1]
    h, w = fel.shape[ax0], fel.shape[ax1]
    fb2 = np.transpose(fel, sp).reshape(-1, h, w)
    yb2 = np.transpose(yo, sp + [fel.ndim]).reshape(-1, h, w, c_dim)
    eb2 = np.transpose(eo, sp + [fel.ndim]).reshape(-1, h, w, c_dim)
    q = fb2[:, :-1, :-1] & fb2[:, 1:, :-1] & fb2[:, :-1, 1:] & fb2[:, 1:, 1:]
    if np.any(q):
      qe = q[..., None]
      dy = np.diff(np.diff(yb2, 1, 2), 1, 1)
      de = np.diff(np.diff(eb2, 1, 2), 1, 1)
      pv = 0.25 * float(np.sum(np.square(np.where(qe, dy, 0.0))))
      pe = 0.25 * float(np.sum(np.square(np.where(qe, de, 0.0))))
      ys = np.sum(np.square(yb2), -1)
      q_ys = ys[:, :-1, :-1] + ys[:, 1:, :-1] + ys[:, :-1, 1:] + ys[:, 1:, 1:]
      ps = 0.25 * float(np.sum(np.where(q, q_ys, 0.0)))
      y00 = yb2[:, :-1, :-1]
      d01 = np.square(y00 - yb2[:, 1:, :-1]) + np.square(y00 - yb2[:, :-1, 1:])
      pd = 0.5 * float(np.sum(np.where(qe, d01, 0.0)))
      pq_ok = pv > max(_V1_EPS_FACTOR * eps_sq * ps, _V1_RESOLVE_REL * pd)
      pq_hi = pv > _V1_TNF_FACTOR * tnf * ps or (
          pv > _V1_REL_EAC_LOW * rel_eac * ps and pd > _V1_PLAQ_DIFF_RATIO * pv
      )
      if pe > 0.0 and pq_ok and pq_hi:
        _upd(_db(pv, pe))
  if c_dim == 2:
    c_fl = (
        max(
            _V1_EPS_FACTOR * eps_sq,
            _V1_COMPLEX_REL * rel_eac,
            _V1_COMPLEX_ABS,
        )
        * sig_el
    )
    for ci in (0, 1):
      cs_i = float(np.sum(np.square(ye[:, ci])))
      ce_i = float(np.sum(np.square(ee[:, ci])))
      if ce_i > 0.0 and cs_i > c_fl:
        if cs_i > _V1_TNF_FACTOR * max(tnf, rel_eac) * sig_el:
          _upd(_db(cs_i, ce_i))
        else:
          cc_i = float(np.sum(np.square(ce[:, ci])))
          cov_i = max(0.5 * (cs_i + cc_i - ce_i), 0.0)
          p_i = cs_i - cov_i * cov_i / cc_i if cc_i > 0.0 else cs_i
          _upd_corr(_db(cs_i, max(p_i, 0.0)), _V1_SLICE_CORR_DB)

  # Ratio of peak normalized error to allowed noise RMS at target_db.
  e_mag = np.sqrt(np.sum(np.square(ee), -1))
  y_mag = np.sqrt(np.sum(np.square(ye), -1))
  denom = np.maximum(y_mag, math.sqrt(max(gmsq, 1e-300)))
  max_rel = float(np.max(e_mag / denom))
  lp = -math.inf if max_rel <= 0.0 else math.log10(max_rel) + target_db / 20.0
  peak = 0.0 if lp < -300.0 else (math.inf if lp > 300.0 else 10.0**lp)
  return block_db, ac_db, ac_ok, m_cdb, peak


def _v1_centered(
    ctx: LeafContext, target_db: float, loc: str
) -> tuple[float | None, list[str]]:
  """Centered SNR and its check. Returns (centered_snr_db, reasons)."""
  n_el, gmsq, rel = ctx.n_elements, ctx.gmsq, ctx.rel_eac
  if n_el <= 1:
    return None, []
  g_var, g_cvar, g_eac = ctx.centered_sig, ctx.centered_cand, ctx.centered_err
  if g_var <= 0.0:
    return None, []
  g_cdb = _db(g_var, g_eac)
  gv, tnf = g_var / float(n_el), 10.0 ** (-target_db / 10.0)
  reasons: list[str] = []
  if gv > _V1_EPS_FACTOR * ctx.eps_sq * gmsq and (
      gv > tnf * gmsq or gv > _V1_REL_EAC_LOW * rel * gmsq
  ):
    hd = target_db - 10.0 * math.log10(max(gmsq / max(gv, 1e-300), 1.0))
    if hd >= 0.0:
      floor = max(_V1_CENTERED_MIN_DB, hd)
      if g_cdb < floor:
        reasons.append(f"{loc}: centered_snr_db {g_cdb:.2f} < {floor:.2f}")
    else:
      cov = 0.5 * (g_var + g_cvar - g_eac)
      p_err = (
          g_var - (cov * cov / g_cvar)
          if (cov > 0.0 and g_cvar > 0.0)
          else g_var
      )
      if (p_db := _db(g_var, max(p_err, 0.0))) < _V1_CENTERED_CORR_DB:
        reasons.append(
            f"{loc}: centered_corr_snr_db {p_db:.2f} <"
            f" {_V1_CENTERED_CORR_DB:.2f}"
        )
  return g_cdb, reasons


def _v1_input_delta(
    ctx: LeafContext, target_db: float, loc: str
) -> tuple[float | None, list[str]]:
  """Input-delta check. Returns (min_delta_sqnr_db, reasons).

  For an input view x with in_db = S(x, y), a passthrough `return x` gives
  S(c - x, y - x) = 0 dB. The check runs only for views with
  _V1_DELTA_P_FL_DB <= in_db < max_in, where max_in is the reference dtype
  limit, 16 eps^2.

  Args:
    ctx: The leaf context.
    target_db: The global floor.
    loc: Leaf name for messages.

  Returns:
    A tuple (min_delta_sqnr_db, reasons). min_delta_sqnr_db is None when
    no view qualifies.
  """
  read, pairs, d_tiles = ctx.read, ctx.pairs, ctx.disjoint_tiles
  max_in = -10.0 * math.log10(max(_V1_EPS_FACTOR * ctx.eps_sq, 1e-300))
  min_ddb: float | None = None
  bad_delta: tuple[float, float, float] | None = None
  for iv_shards in ctx.input_views:
    mi, bad_i = 0.0, False
    for s_i, d_sl in d_tiles:
      rc = as_real(read(pairs[s_i][0], d_sl))
      ic = as_real(read(iv_shards[s_i], d_sl))
      if np.any(m := np.isfinite(rc)):
        if not np.all(np.isfinite(ic)[m]):
          bad_i = True
          break
        mi = max(mi, float(np.max(np.abs(ic[m]))))
    if bad_i:
      continue
    psi = ctx.shift if mi == 0.0 else max(ctx.shift, math.frexp(mi)[1])
    rsig = dsig = derr = 0.0
    for s_i, d_sl in d_tiles:
      rc = as_real(read(pairs[s_i][0], d_sl))
      if np.any(m := np.isfinite(rc)):
        ru = np.ldexp(rc[m], -psi)
        dr = ru - np.ldexp(as_real(read(iv_shards[s_i], d_sl))[m], -psi)
        dc = np.ldexp(as_real(read(pairs[s_i][1], d_sl))[m], -psi) - ru
        rsig += float(np.sum(np.square(ru)))
        dsig += float(np.sum(np.square(dr)))
        derr += float(np.sum(np.square(dc)))
    in_db = math.inf if dsig == 0.0 else _db(rsig, dsig)
    if dsig == 0.0:
      d_db = math.inf if derr == 0.0 else -math.inf
    else:
      d_db = _db(dsig, derr)
    if _V1_DELTA_P_FL_DB <= in_db < max_in:
      min_ddb = d_db if min_ddb is None else min(min_ddb, d_db)
      v_fl = min(
          _V1_DELTA_MAX_DB,
          max(_V1_DELTA_MIN_DB, target_db - in_db),
          max(_V1_DELTA_LIMIT_DB, max_in - in_db),
      )
      if d_db < v_fl:
        gap = d_db - v_fl
        if bad_delta is None or gap < bad_delta[0]:
          bad_delta = (gap, d_db, v_fl)
  reasons = []
  if bad_delta is not None:
    reasons.append(
        f"{loc}: min_delta_sqnr_db {bad_delta[1]:.2f} < {bad_delta[2]:.2f}"
    )
  return min_ddb, reasons


def _six_gate_v1_checks(
    ctx: LeafContext, config: SqnrConfig, loc: str
) -> CheckOutcome:
  """Centered, block, AC, macro-correlation, input-delta and peak/RMS checks.

  The verdicts and metric values follow the fixed-constant six-check preset.
  The global SQNR check runs in `audit_leaf`.

  Args:
    ctx: The leaf context.
    config: The policy. Only `block_size` (through ctx) is used.
    loc: Leaf name for messages.

  Returns:
    Metrics centered_snr_db, min_block_sqnr_db, min_ac_snr_db,
    min_delta_sqnr_db and peak_error_rms_ratio, and the failed checks.
  """
  del config  # The preset uses fixed constants.
  target = ctx.threshold_db
  block_db, ac_db, ac_ok, m_cdb, pk = math.inf, None, True, None, 0.0
  for s_i, o_sl in ctx.overlap_tiles:
    tb, tac, tok, tm, tpk = _v1_spatial_tile(
        ctx.read(ctx.pairs[s_i][0], o_sl),
        ctx.read(ctx.pairs[s_i][1], o_sl),
        ctx,
        target,
    )
    block_db, ac_ok, pk = min(block_db, tb), ac_ok and tok, max(pk, tpk)
    if tac is not None:
      ac_db = tac if ac_db is None else min(ac_db, tac)
    if tm is not None and (
        m_cdb is None or (tm[0] - tm[1]) < (m_cdb[0] - m_cdb[1])
    ):
      m_cdb = tm
  g_cdb, reasons = _v1_centered(ctx, target, loc)
  if block_db < target - _V1_BLOCK_MARGIN_DB:
    reasons.append(
        f"{loc}: min_block_sqnr_db {block_db:.2f} <"
        f" {target - _V1_BLOCK_MARGIN_DB:.2f}"
    )
  if not ac_ok:
    reasons.append(f"{loc}: min_ac_snr_db failed AC survival")
  if m_cdb is not None:
    reasons.append(f"{loc}: macro_corr_snr_db {m_cdb[0]:.2f} < {m_cdb[1]:.2f}")
  min_ddb, d_reasons = _v1_input_delta(ctx, target, loc)
  reasons.extend(d_reasons)
  if pk > _V1_PEAK_MAX:
    reasons.append(f"{loc}: peak_error_rms_ratio {pk:.2f} > {_V1_PEAK_MAX:.2f}")
  return CheckOutcome(
      metrics=dict(
          centered_snr_db=g_cdb,
          min_block_sqnr_db=block_db,
          min_ac_snr_db=ac_db,
          min_delta_sqnr_db=min_ddb,
          peak_error_rms_ratio=pk,
      ),
      reasons=tuple(reasons),
  )


# --- END SIX_GATE_V1 CHECKS ---


# --- BEGIN DERIVED CHECKS (alpha-derived gates) ---
# Honest-noise model of the derived checks. Each output y_i is rounded once to
# a grid with step u_i (the candidate dtype or a PrecisionSpec format), so the
# error has E[e_i**2] = u_i**2 / 12 (Widrow and Kollar, "Quantization Noise",
# 2008). Accumulation and upstream rounding add noise that does not scale with
# u_i, so the checks fit E[e_i**2] = A * u_i**2 / 12 + B by least squares on
# the leaf and test the fit against the leaf structure. The B term also covers
# error near zero that comes from larger inputs, so u_i has no near-zero floor:
# a floor at 2**-8 max|y| overstates the step of small elements, biases A low,
# and makes outlier elements look like spikes.

# Safety factor on the block-noise bound (3 dB). The bound is the chi-square
# quantile for normal e. If e has kurtosis kappa (3 for normal e, 1.8 for one
# uniform rounding), the block sum of e**2 / sigma**2 matches chi2 with
# n * 2 / (kappa - 1) degrees of freedom instead of n. The Wilson-Hilferty
# quantile ratio of the two is at most 2.0 for kappa <= 4.76, at every n >= 1
# and every z <= 7.6 (alpha / n_blocks >= 1e-14). A sum of K products of
# normal terms has kappa = 3 + 6 / K, so the factor covers every K >= 4.
_DC_BLOCK_SAFETY = 2.0
# Safety factor on the spike bound (6 dB). The bound is z(alpha / 2N) for a
# normal e. Accumulated error is a sum of K products of normal terms, which has
# heavier tails. Its upper quantile over the normal one, at tail p, is 1.67 for
# K = 6 at p = 1e-9 and 2.07 for K = 6 at p = 1e-16 (exact chi-square difference
# form). A single rounding has |e| <= u / 2 = 1.7 sigma, far below the bound.
_DC_SPIKE_SAFETY = 2.0
# Elements per correlation window: 32 blocks of 32. The r**2 floor z**2 / n is
# then below 0.03 for z <= 5.5, far from the honest minimum k / (1 + k) = 0.41.
_DC_CORR_WINDOW = 1024
# Gate on the window AC energy of y over the AC energy of e. A candidate that
# drops the window AC gives 1. A candidate that is noise of the same variance as
# y, but independent of it, gives 1/2. 1/sqrt(2) is 1.5 dB from both.
_DC_CORR_GATE = 1.0 / math.sqrt(2.0)
# A window needs at least 4 z**2 samples, so that the r**2 floor z**2 / n is
# at most 1/4, below the honest minimum k / (1 + k) = 0.41.
_DC_MIN_SAMPLES_Z2 = 4.0
# The reference dtype resolves an AC energy above 16 eps**2 of the window
# energy: a standard deviation of 4 eps, that is 2 ulp at the top binade.
_DC_RESOLVE_EPS_FACTOR = 16.0
# An input view is a residual base only when it is at least one bit closer to
# y than zero: 20 log10(2) dB.
_DC_DELTA_MIN_IN_DB = 20.0 * math.log10(2.0)
# Correlation transforms per window position: centered and first difference
# in two orientations, and the 2-D mixed difference.
_DC_TRANSFORMS = 5
# Floor on the fitted A. One round-to-nearest into the candidate format has
# E[e**2] = u**2 / 12, so A = 1 is the prior for one honest rounding. The fit
# can go far below it when most errors are 0. This is the case when the
# reference and the candidate round values that are almost equal to the same
# grid (a different summation order): only a few elements differ, each by one
# step. With A < 1 such a step looks like a spike. At A = 1 the largest honest
# step |e| = u is sqrt(12) = 3.5 sigma, which is below the spike bound
# _DC_SPIKE_SAFETY * z(alpha / 2N) for every alpha <= 0.08 and N >= 1
# (2 * 4.9 = 9.8 at alpha = 1e-6, N = 1).
_DC_MIN_A = 1.0
# Offset of the frexp exponent of v in the fit bins. float64 exponents span
# -1073..1024, so 1100 holds them all.
_DC_BIN_OFFSET = 1100
# Reweighting rounds of the fit. Each round moves the weights to the current
# fit. Five rounds converge to 1e-3 relative on the corpus.
_DC_FIT_ITERATIONS = 5
# Floor of the fit variance relative to its largest bin. 1e-12 caps the
# weight ratio at 1e24, so the weighted sums stay finite in float64.
_DC_FIT_REL_FLOOR = 1e-12


def _dc_z(tail: float) -> float:
  """Returns z with P(N(0, 1) > z) = tail."""
  return -statistics.NormalDist().inv_cdf(min(max(tail, 1e-300), 0.5))


def _dc_chi2_upper(n: np.ndarray, z: float) -> np.ndarray:
  """Upper quantile of chi2_n / n at normal score z (Wilson-Hilferty)."""
  h = 2.0 / (9.0 * np.maximum(n, 1.0))
  return np.maximum(1.0 - h + z * np.sqrt(h), 0.0) ** 3


def _dc_formats(ctx: LeafContext) -> list[tuple[int, int, int, int]]:
  """Returns (nmant, minexp, emax, block) for each rounding format.

  The formats are the candidate dtype and the float names in the PrecisionSpec.
  block is 0 for an element format, else the elements per power-of-two scale
  along the last axis (the OCP MX rule: scale = 2**(floor(log2 amax) - emax)).

  Args:
    ctx: The leaf context.
  """
  p = ctx.precision
  names = [ctx.cand_dtype, p.output_dtype, p.accumulator_dtype]
  names.extend(p.operand_dtypes)
  out = set()
  for name in names:
    blk = 0
    if isinstance(name, str) and name in BLOCK_SCALED_FORMATS:
      name, blk = BLOCK_SCALED_FORMATS[name]
    if name is None:
      continue
    try:
      fi = ml_dtypes.finfo(np.dtype(name))
    except (TypeError, ValueError):
      continue
    out.add((int(fi.nmant), int(fi.minexp), int(fi.maxexp) - 1, blk))
  return sorted(out)


def _dc_var(
    y: np.ndarray, fmts: Sequence[tuple[int, int, int, int]], ctx: LeafContext
) -> np.ndarray:
  """Returns the honest rounding variance v = u**2 / 12 of each component.

  Args:
    y: Scaled reference tile, shape core + (components,).
    fmts: The output of `_dc_formats`.
    ctx: The leaf context.
  """
  a = np.abs(y)
  e_y = np.frexp(a)[1] - 1
  ex = np.full(y.shape, np.iinfo(np.int32).min // 2, np.int64)
  for nmant, minexp, emax, blk in fmts:
    lo = np.int64(minexp - ctx.shift)
    if blk and y.ndim >= 2:
      a_el = np.max(a, -1)
      starts = np.arange(0, a_el.shape[-1], blk)
      amax = np.maximum.reduceat(a_el, starts, axis=-1)
      cnt = np.diff(np.append(starts, a_el.shape[-1]))
      e_s = np.repeat(np.frexp(amax)[1] - 1 - emax, cnt, axis=-1)[..., None]
      # The scale maps amax into [2**emax, 2**(emax + 1)), above the largest
      # element value for E2M1 (6 of [4, 8)). The top binade saturates, with
      # error up to one step, so its step counts double.
      top = (e_y >= e_s + emax).astype(np.int64)
      ex = np.maximum(ex, np.maximum(e_y, e_s + minexp) - nmant + top)
      continue
    ex = np.maximum(ex, np.maximum(e_y, lo) - nmant)
  # An element below u / 2 can round to 0, so its error is at most |y|. This
  # also gives v = 0 at y = 0, where a rounding is exact.
  return np.minimum(np.ldexp(1.0 / 12.0, 2 * ex), np.square(y))


def _dc_tile(
    ctx: LeafContext, s_i: int, sl: tuple[slice, ...], fmts
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Returns scaled (y, c, v, mask) for a tile, shape (rows, L, components).

  y and c are zero where an element is not finite. v = u**2 / 12.

  Args:
    ctx: The leaf context.
    s_i: Shard index.
    sl: Tile slices.
    fmts: The output of `_dc_formats`.
  """
  rc = as_real(ctx.read(ctx.pairs[s_i][0], sl))
  cc = as_real(ctx.read(ctx.pairs[s_i][1], sl))
  if rc.ndim == 1:
    rc, cc = rc[None, None], cc[None, None]
  m = np.all(np.isfinite(rc) & np.isfinite(cc), -1)
  y = np.where(m[..., None], np.ldexp(rc, -ctx.shift), 0.0)
  c = np.where(m[..., None], np.ldexp(cc, -ctx.shift), 0.0)
  v = np.where(m[..., None], _dc_var(y, fmts, ctx), 0.0)
  shape = (-1,) + y.shape[-2:]
  return (
      y.reshape(shape),
      c.reshape(shape),
      v.reshape(shape),
      m.reshape(shape[:-1]),
  )


def _dc_block_sums(t: np.ndarray, starts: np.ndarray) -> np.ndarray:
  """Sums `t` over each block of the minor axis, flattened over the rows."""
  return np.add.reduceat(np.sum(t, -1), starts, axis=1).ravel()


def _dc_fit(ctx: LeafContext, fmts) -> tuple[float, float, int]:
  """Fits E[e**2] = A v + B. Returns (A, B, block count).

  The fit is weighted least squares with weights 1 / (A v + B)**2, the inverse
  variance of e**2 for normal e, so that a few large elements do not set A.
  The sums are binned by the binary exponent of v, which takes few values, so
  the reweighting runs on the bins and needs one pass over the leaf.

  Args:
    ctx: The leaf context.
    fmts: The output of `_dc_formats`.

  Returns:
    A tuple (A, B, block count).
  """
  # Bin 0 holds v = 0. Bin k > 0 holds frexp exponent k - _DC_BIN_OFFSET.
  sums = np.zeros((5, 2 * _DC_BIN_OFFSET))  # n, v, v**2, e**2, e**2 v
  n_blocks = 0
  for s_i, sl in ctx.disjoint_tiles:
    y, c, v, m = _dc_tile(ctx, s_i, sl, fmts)
    mk = np.broadcast_to(m[..., None], y.shape).ravel()
    v, e2 = v.ravel()[mk], np.square(c - y).ravel()[mk]
    k = np.where(v > 0.0, np.frexp(v)[1] + _DC_BIN_OFFSET, 0)
    k = np.clip(k, 0, 2 * _DC_BIN_OFFSET - 1)
    for row, w in enumerate((None, v, v * v, e2, e2 * v)):
      sums[row] += np.bincount(k, w, minlength=2 * _DC_BIN_OFFSET)
    n_blocks += y.shape[0] * (-(-y.shape[1] // ctx.block_size))
  n_k, v_k, v2_k, e_k, ev_k = sums[:, sums[0] > 0]
  # Start from A = 1 and B = mean(e**2). Then a single large element that is
  # exact in the output format (e = 0, large v) gets a small weight from the
  # first round, and cannot pull A to 0.
  a_fit, b_fit = 1.0, float(np.sum(e_k) / max(np.sum(n_k), 1.0))
  for _ in range(_DC_FIT_ITERATIONS):
    s2 = a_fit * v_k / n_k + b_fit
    s2_max = float(np.max(s2))
    if not s2_max > 0.0:
      break  # Every error is 0 and the fit is exact.
    # Weighted least squares does not change when all weights scale by the
    # same factor. Scale them by s2_max**2 so that they stay finite. The floor
    # caps the weight ratio at 1e24 (well inside float64). Without it a bin
    # with e = 0 and v = 0 (for example zeros in the reference) gets the
    # weight 1 / 0 once the fit sets B = 0.
    s2 = np.maximum(s2, _DC_FIT_REL_FLOOR * s2_max)
    w = np.square(s2_max / s2)
    sw, swv, swv2 = np.sum(w * n_k), np.sum(w * v_k), np.sum(w * v2_k)
    swe, swev = np.sum(w * e_k), np.sum(w * ev_k)
    det = sw * swv2 - swv * swv
    if det <= 1e-12 * sw * swv2:
      a_fit, b_fit = 0.0, float(swe / sw)
      continue
    a_fit = float((sw * swev - swv * swe) / det)
    b_fit = float((swv2 * swe - swv * swev) / det)
    if a_fit < 0.0:
      a_fit, b_fit = 0.0, float(swe / sw)
    elif b_fit < 0.0:
      a_fit, b_fit = float(swev / swv2), 0.0
  return a_fit, b_fit, n_blocks


def _dc_chunks(a: np.ndarray, w: int) -> np.ndarray:
  """Splits axis 0 into windows of w. The last window overlaps the previous."""
  n = a.shape[0]
  if n <= w:
    return a[None]
  nf = (n // w) * w
  out = a[:nf].reshape((-1, w) + a.shape[1:])
  if nf == n:
    return out
  return np.concatenate([out, a[n - w :][None]], axis=0)


def _dc_window_corr(
    y: np.ndarray,
    c: np.ndarray,
    v: np.ndarray,
    m: np.ndarray,
    z: float,
    eps_sq: float,
) -> tuple[np.ndarray, np.ndarray]:
  """AC correlation of candidate and reference on windows.

  Args:
    y: Transformed reference, shape (n, components).
    c: Transformed candidate, same shape.
    v: Honest rounding variance of the transformed values, same shape.
    m: Element mask, shape (n,).
    z: Normal score of the per-window false-alarm rate.
    eps_sq: finfo(reference dtype).eps ** 2.

  Returns:
    (p_db, p_fl) per window and component, NaN where the window is not
    tested. p_db = -10 log10(1 - r**2) is the share of the window AC of y that
    the candidate explains. p_fl is the same for r**2 = z**2 / n, the largest
    value that a candidate independent of y reaches at the alpha level.
  """
  mw = _dc_chunks(m, _DC_CORR_WINDOW)
  cnt = np.sum(mw, 1).astype(np.float64)
  den = np.maximum(cnt, 1.0)[:, None, None]
  mk = mw[..., None]

  def _dev(t):
    tw = np.where(mk, _dc_chunks(t, _DC_CORR_WINDOW), 0.0)
    return np.where(mk, tw - np.sum(tw, 1, keepdims=True) / den, 0.0), tw

  dy, yw = _dev(y)
  dc, _ = _dev(c)
  syy = np.sum(np.square(dy), 1)
  scc = np.sum(np.square(dc), 1)
  syc = np.sum(dy * dc, 1)
  see = np.sum(np.square(dc - dy), 1)
  q = np.sum(np.where(mk, _dc_chunks(v, _DC_CORR_WINDOW), 0.0), 1)
  res = _DC_RESOLVE_EPS_FACTOR * eps_sq * np.sum(np.square(yw), 1)
  ok = (cnt >= _DC_MIN_SAMPLES_Z2 * z * z)[:, None]
  ok = ok & (syy >= _DC_CORR_GATE * see) & (syy > _DC_CORR_GATE * q)
  ok &= syy > res
  with np.errstate(all="ignore"):
    r2 = np.where((syc > 0.0) & (scc > 0.0), syc * syc / (syy * scc), 0.0)
    p_db = -10.0 * np.log10(np.maximum(1.0 - np.minimum(r2, 1.0), 0.0))
    fl = np.minimum(z * z / den[:, :, 0], 1.0)
    p_fl = np.broadcast_to(-10.0 * np.log10(1.0 - fl + 1e-300), p_db.shape)
  return np.where(ok, p_db, np.nan), np.where(ok, p_fl, np.nan)


def _dc_transforms(y, c, v, m):
  """Yields (name, y, c, v, mask) for the correlation transforms of a tile.

  Args:
    y: Reference tile, shape (rows, L, components).
    c: Candidate tile.
    v: Rounding variance tile.
    m: Element mask, shape (rows, L).

  The centered transform tests the window mean-free part. The first
  difference also removes per-row constants, and the mixed difference removes
  any separable row-plus-column part. Each runs on row-major windows; the
  first two also on column-major windows.
  """
  k_c = y.shape[-1]

  def _flat(*ts):
    return tuple(t.reshape((-1, k_c) if t.ndim == 3 else (-1,)) for t in ts)

  yield ("centered_corr_snr_db",) + _flat(y, c, v, m)
  if y.shape[1] >= 2:
    d = lambda t: t[:, 1:] - t[:, :-1]
    yield ("diff_corr_snr_db",) + _flat(
        d(y), d(c), v[:, 1:] + v[:, :-1], m[:, 1:] & m[:, :-1]
    )
  if y.shape[0] >= 2:
    yt, ct, vt = (np.transpose(t, (1, 0, 2)) for t in (y, c, v))
    mt = m.T
    yield ("centered_corr_snr_db",) + _flat(yt, ct, vt, mt)
    d = lambda t: t[:, 1:] - t[:, :-1]
    yield ("diff_corr_snr_db",) + _flat(
        d(yt), d(ct), vt[:, 1:] + vt[:, :-1], mt[:, 1:] & mt[:, :-1]
    )
    if y.shape[1] >= 2:
      dd = lambda t: t[1:, 1:] - t[1:, :-1] - t[:-1, 1:] + t[:-1, :-1]
      v4 = v[1:, 1:] + v[1:, :-1] + v[:-1, 1:] + v[:-1, :-1]
      m4 = m[1:, 1:] & m[1:, :-1] & m[:-1, 1:] & m[:-1, :-1]
      yield ("plaquette_corr_snr_db",) + _flat(dd(y), dd(c), v4, m4)


def _dc_spatial(
    ctx: LeafContext,
    config: SqnrConfig,
    fmts,
    a_fit: float,
    b_fit: float,
    n_blocks: int,
    loc: str,
) -> tuple[dict[str, float | None], list[str]]:
  """Block noise, spike and AC correlation over the disjoint tiles.

  Each tile is reduced before the next one is read, so the memory use is
  O(tile) and does not grow with the leaf.

  Args:
    ctx: The leaf context.
    config: The policy (alpha).
    fmts: The output of `_dc_formats`.
    a_fit: The A term of sigma**2 = A v + B (already at its floor).
    b_fit: The B term.
    n_blocks: Block count of the leaf.
    loc: Leaf name for messages.

  Returns:
    A tuple (metrics, reasons). metrics maps report field names to values.
  """
  bs, alpha = ctx.block_size, config.alpha
  z_blk = _dc_z(alpha / max(n_blocks, 1))
  z_spk = _dc_z(alpha / (2.0 * max(ctx.n_finite, 1)))
  n_tests = (
      _DC_TRANSFORMS
      * (ctx.full_n // _DC_CORR_WINDOW + len(ctx.disjoint_tiles))
      * (2 if ctx.is_complex else 1)
  )
  z_cor = _dc_z(alpha / (2.0 * max(n_tests, 1)))
  min_blk: float | None = None
  max_ratio: float | None = None
  blk_worst = (0.0, 1.0)  # (R_b, bound) at the largest R_b / bound.
  spike = (0.0, 1.0)  # (|e| / sigma, sigma / u) at the worst element.
  max_ulp = 0.0
  worst: dict[str, tuple[float, float]] = {}
  min_ac: float | None = None
  for s_i, sl in ctx.disjoint_tiles:
    y, c, v, m = _dc_tile(ctx, s_i, sl, fmts)
    e = c - y
    mk = m[..., None]
    s2 = np.where(mk, a_fit * v + b_fit, 0.0)
    e2 = np.square(e)
    starts = np.arange(0, y.shape[1], bs)
    b_e = _dc_block_sums(e2, starts)
    b_s = _dc_block_sums(s2, starts)
    b_q = _dc_block_sums(np.square(s2), starts)
    if np.any(act := b_e > 0.0):
      t_db = _min_db(_dc_block_sums(np.square(y), starts)[act], b_e[act])
      min_blk = t_db if min_blk is None else min(min_blk, t_db)
    if np.any(pos := b_s > 0.0):
      b_e, b_s, b_q = b_e[pos], b_s[pos], b_q[pos]
      ratio = b_e / b_s
      with np.errstate(all="ignore"):
        # Effective sample count (sum s2)**2 / sum s2**2: the chi-square count
        # with the same mean and variance as the block sum of e**2 / s2.
        n_eff = np.where(b_q > 0.0, np.square(b_s) / b_q, 1.0)
      bound = _DC_BLOCK_SAFETY * _dc_chi2_upper(n_eff, z_blk)
      r_max = float(np.max(ratio))
      max_ratio = r_max if max_ratio is None else max(max_ratio, r_max)
      j = int(np.argmax(ratio / bound))
      if ratio[j] / bound[j] > blk_worst[0] / blk_worst[1]:
        blk_worst = (float(ratio[j]), float(bound[j]))
    with np.errstate(all="ignore"):
      u = np.sqrt(12.0 * v)
      # Elements with u = 0 have no step to compare with. Leave them out of
      # the argmax too, so that one of them cannot hide a spike elsewhere in
      # the tile.
      t = np.where((s2 > 0.0) & (u > 0.0), np.abs(e) / np.sqrt(s2), 0.0)
      max_ulp = max(
          max_ulp, float(np.max(np.where(u > 0.0, np.abs(e) / u, 0.0)))
      )
    i = int(np.argmax(t))
    if t.flat[i] > spike[0]:
      spike = (float(t.flat[i]), float(np.sqrt(s2.flat[i]) / u.flat[i]))
    for name, ty, tc, tv, tm in _dc_transforms(y, c, v, m):
      p_db, p_fl = _dc_window_corr(ty, tc, tv, tm, z_cor, ctx.eps_sq)
      if np.all(np.isnan(p_db)):
        continue
      gap = p_db - p_fl
      j = int(np.nanargmin(gap))
      t_min = float(np.nanmin(p_db))
      min_ac = t_min if min_ac is None else min(min_ac, t_min)
      if gap.flat[j] < 0.0 and (
          name not in worst or gap.flat[j] < worst[name][0] - worst[name][1]
      ):
        worst[name] = (float(p_db.flat[j]), float(p_fl.flat[j]))
  metrics: dict[str, float | None] = {}
  reasons: list[str] = []
  if min_blk is not None:
    metrics["min_block_sqnr_db"] = min_blk
  if max_ratio is not None:
    # The ratio uses the fitted variance directly. A ratio to the leaf mean
    # sum(e**2) / sum(s2) is wrong when the largest elements are exact in the
    # output format (e = 0): the mean then falls far below 1 and a block with
    # one honest large element looks noisy.
    metrics["max_block_noise_ratio_db"] = 10.0 * math.log10(
        max(max_ratio, 1e-300)
    )
    if blk_worst[0] > blk_worst[1]:
      reasons.append(
          f"{loc}: max_block_noise_ratio_db"
          f" {10.0 * math.log10(blk_worst[0]):.2f}"
          f" > {10.0 * math.log10(blk_worst[1]):.2f}"
      )
  metrics["max_error_ulp"] = max_ulp
  spk_bound = _DC_SPIKE_SAFETY * z_spk
  if spike[0] > spk_bound:
    # Both sides in ULP units of the worst element: |e| / u and its bound.
    k_u = spike[1]
    reasons.append(
        f"{loc}: max_error_ulp {spike[0] * k_u:.2f} > {spk_bound * k_u:.2f}"
    )
  metrics["min_ac_snr_db"] = min_ac
  for name, (p_db, p_fl) in sorted(worst.items()):
    reasons.append(f"{loc}: {name} {p_db:.2f} < {p_fl:.2f}")
  return metrics, reasons


def _dc_input_delta(
    ctx: LeafContext, config: SqnrConfig, fmts, loc: str
) -> tuple[float | None, list[str]]:
  """Input-delta check. Returns (min_delta_sqnr_db, reasons).

  For an input view x with in_db = S(x, y) in [_DC_DELTA_MIN_IN_DB, max_in),
  where max_in is the reference dtype limit, the update y - x must show in the
  candidate. A passthrough `return x` gives c - x = 0, and `x + const` gives a
  constant c - x. Honest noise e that is independent of the update gives
  r**2(c - x, y - x) >= k / (1 + k) once the gates below hold. The test fails
  when r**2 is below z**2 / n, or when c equals x on every element.

  Args:
    ctx: The leaf context.
    config: The policy (alpha).
    fmts: The output of `_dc_formats`.
    loc: Leaf name for messages.

  Returns:
    A tuple (min_delta_sqnr_db, reasons). min_delta_sqnr_db is None when
    no view qualifies.
  """
  read, pairs, d_tiles = ctx.read, ctx.pairs, ctx.disjoint_tiles
  max_in = -10.0 * math.log10(max(_DC_RESOLVE_EPS_FACTOR * ctx.eps_sq, 1e-300))
  k_c = 2 if ctx.is_complex else 1
  z = _dc_z(config.alpha / (2.0 * max(len(ctx.input_views) * k_c, 1)))
  min_ddb: float | None = None
  bad: tuple[float, str] | None = None
  for iv_shards in ctx.input_views:
    mi, bad_i = 0.0, False
    for s_i, d_sl in d_tiles:
      rc = as_real(read(pairs[s_i][0], d_sl))
      ic = as_real(read(iv_shards[s_i], d_sl))
      if np.any(m := np.isfinite(rc)):
        if not np.all(np.isfinite(ic)[m]):
          bad_i = True
          break
        mi = max(mi, float(np.max(np.abs(ic[m]))))
    if bad_i:
      continue
    psi = ctx.shift if mi == 0.0 else max(ctx.shift, math.frexp(mi)[1])
    sums = np.zeros((9, k_c))  # n, sy, sc, ryy, dyy, dee, dcc, q, unused
    tiles = []
    for s_i, d_sl in d_tiles:
      rc = as_real(read(pairs[s_i][0], d_sl))
      m = np.isfinite(rc)
      ru = np.where(m, np.ldexp(rc, -psi), 0.0)
      xu = np.where(m, np.ldexp(as_real(read(iv_shards[s_i], d_sl)), -psi), 0.0)
      cu = np.where(m, np.ldexp(as_real(read(pairs[s_i][1], d_sl)), -psi), 0.0)
      ys = np.where(m, np.ldexp(rc, -ctx.shift), 0.0)
      v = _dc_var(ys if ys.ndim >= 2 else ys[None], fmts, ctx)
      v = np.where(m, np.ldexp(v.reshape(m.shape), 2 * (ctx.shift - psi)), 0.0)
      ax = tuple(range(m.ndim - 1))
      dy, dc = ru - xu, cu - xu
      sums[0] += np.sum(m, ax)
      sums[1] += np.sum(dy, ax)
      sums[2] += np.sum(dc, ax)
      sums[3] += np.sum(np.square(ru), ax)
      sums[4] += np.sum(np.square(dy), ax)
      sums[5] += np.sum(np.square(cu - ru), ax)
      sums[6] += np.sum(np.square(dc), ax)
      sums[7] += np.sum(v, ax)
      tiles.append((s_i, d_sl))
    n, rsig, dsig, derr = sums[0], sums[3].sum(), sums[4].sum(), sums[5].sum()
    in_db = math.inf if dsig == 0.0 else _db(rsig, dsig)
    if not _DC_DELTA_MIN_IN_DB <= in_db < max_in:
      continue
    d_db = _db(dsig, derr)
    min_ddb = d_db if min_ddb is None else min(min_ddb, d_db)
    if sums[6].sum() == 0.0 and dsig > _DC_CORR_GATE * sums[7].sum():
      bad = (
          -math.inf,
          (
              f"{loc}: min_delta_sqnr_db {d_db:.2f} < inf"
              " (candidate equals an input view)"
          ),
      )
      continue
    mean_y = sums[1] / np.maximum(n, 1.0)
    mean_c = sums[2] / np.maximum(n, 1.0)
    cen = np.zeros((4, k_c))  # syy, scc, syc, see
    for s_i, d_sl in tiles:
      rc = as_real(read(pairs[s_i][0], d_sl))
      m = np.isfinite(rc)
      ru = np.ldexp(rc, -psi)
      xu = np.ldexp(as_real(read(iv_shards[s_i], d_sl)), -psi)
      cu = np.ldexp(as_real(read(pairs[s_i][1], d_sl)), -psi)
      ax = tuple(range(m.ndim - 1))
      dy = np.where(m, ru - xu - mean_y, 0.0)
      dc = np.where(m, cu - xu - mean_c, 0.0)
      cen[0] += np.sum(np.square(dy), ax)
      cen[1] += np.sum(np.square(dc), ax)
      cen[2] += np.sum(dy * dc, ax)
      cen[3] += np.sum(np.square(dc - dy), ax)
    syy, scc, syc, see = cen
    ok = (n >= _DC_MIN_SAMPLES_Z2 * z * z) & (syy >= _DC_CORR_GATE * see)
    ok &= (syy > _DC_CORR_GATE * sums[7]) & (
        syy > _DC_RESOLVE_EPS_FACTOR * ctx.eps_sq * sums[3]
    )
    with np.errstate(all="ignore"):
      r2 = np.where((syc > 0.0) & (scc > 0.0), syc * syc / (syy * scc), 0.0)
      p_db = -10.0 * np.log10(np.maximum(1.0 - np.minimum(r2, 1.0), 0.0))
      p_fl = -10.0 * np.log10(
          1.0 - np.minimum(z * z / np.maximum(n, 1.0), 1.0) + 1e-300
      )
    for k in np.flatnonzero(ok & (p_db < p_fl)):
      gap = float(p_db[k] - p_fl[k])
      if bad is None or gap < bad[0]:
        bad = (
            gap,
            (
                f"{loc}: min_delta_sqnr_db {p_db[k]:.2f} < {p_fl[k]:.2f}"
                " (correlation of candidate - input with reference - input)"
            ),
        )
  return min_ddb, [bad[1]] if bad is not None else []


def _derived_checks(
    ctx: LeafContext, config: SqnrConfig, loc: str
) -> CheckOutcome:
  """Checks with bounds from config.alpha and the honest-noise model.

  1. Block noise: R_b = sum_b e**2 / sum_b sigma**2 with sigma**2 = A v + B.
     Fails when R_b is above the 1 - alpha / n_blocks quantile of
     chi2_n / n times _DC_BLOCK_SAFETY, with n = (sum_b sigma**2)**2 /
     sum_b sigma**4, the effective sample count of the block.
  2. Spike: max |e| / sigma above z(1 - alpha / (2 N)) * _DC_SPIKE_SAFETY.
  3. AC correlation (centered, first difference, mixed difference) on windows
     of _DC_CORR_WINDOW elements, as in `_dc_window_corr`.
  4. Input delta, as in `_dc_input_delta`.

  Args:
    ctx: The leaf context.
    config: The policy (alpha).
    loc: Leaf name for messages.

  Returns:
    Metrics max_block_noise_ratio_db, min_block_sqnr_db, max_error_ulp,
    min_ac_snr_db, centered_snr_db and min_delta_sqnr_db, and the failed
    checks.
  """
  fmts = _dc_formats(ctx)
  a_fit, b_fit, n_blocks = _dc_fit(ctx, fmts)
  a_fit = max(a_fit, _DC_MIN_A)
  metrics, reasons = _dc_spatial(ctx, config, fmts, a_fit, b_fit, n_blocks, loc)
  if ctx.n_elements > 1 and ctx.centered_sig > 0.0:
    metrics["centered_snr_db"] = _db(ctx.centered_sig, ctx.centered_err)
  min_ddb, d_reasons = _dc_input_delta(ctx, config, fmts, loc)
  metrics["min_delta_sqnr_db"] = min_ddb
  reasons.extend(d_reasons)
  return CheckOutcome(metrics=metrics, reasons=tuple(reasons))


# --- END DERIVED CHECKS ---


def _oracle_reasons(
    o_c: float, o_r: float, threshold_db: float, config: SqnrConfig, loc: str
) -> list[str]:
  """Oracle rule. It fails closed when the reference is below the floor.

  A reference below the floor (o_r < threshold_db) cannot vouch for the
  candidate, so the leaf fails with `reference_below_floor`. Otherwise the
  rule fails only if the candidate misses BOTH bounds, the floor and the
  reference-relative bound: o_c < min(threshold_db, o_r + oracle_delta_db).

  Args:
    o_c: oracle_sqnr_db of the candidate.
    o_r: reference_oracle_sqnr_db.
    threshold_db: The global floor.
    config: The policy.
    loc: Leaf name for messages.

  Returns:
    The failure reasons. The list is empty when the rule passes.
  """
  if o_r < threshold_db:
    return [
        f"{loc}: reference_below_floor: reference_oracle_sqnr_db"
        f" {o_r:.2f} < {threshold_db:.2f}"
    ]
  bound = min(threshold_db, o_r + config.oracle_delta_db)
  if o_c < bound:
    return [f"{loc}: oracle_sqnr_db {o_c:.2f} < {bound:.2f}"]
  return []


def _oracle_sums(
    pairs: Sequence[tuple[Any, Any]],
    read: Callable[[Any, tuple[slice, ...]], np.ndarray],
    tiles: Sequence[tuple[int, tuple[slice, ...]]],
    data_max: float,
    oracle_pairs: Sequence[Any],
) -> tuple[float, float, int]:
  """Returns (oracle_sqnr_db, reference_oracle_sqnr_db, n_bad), tile by tile.

  The sums run over components where the oracle, the reference and the
  candidate are all finite. n_bad counts components where the reference and
  the candidate are finite but the oracle is not. Such components drop out of
  the sums, so the caller must fail the leaf when n_bad > 0.

  Args:
    pairs: (reference, candidate) shards.
    read: The array accessor.
    tiles: Disjoint tiles over the shards.
    data_max: max(|reference|, |candidate|) over finite components.
    oracle_pairs: Oracle shards with the shard shapes.
  """
  o_max = 0.0
  for s_i, d_sl in tiles:
    oc = as_real(read(oracle_pairs[s_i], d_sl))
    fo = np.isfinite(oc)
    if np.any(fo):
      o_max = max(o_max, float(np.max(np.abs(oc[fo]))))
  sh = math.frexp(max(data_max, o_max))[1]
  o_sig = e_c = e_r = 0.0
  n_bad = 0
  for s_i, d_sl in tiles:
    oc = as_real(read(oracle_pairs[s_i], d_sl))
    rc = as_real(read(pairs[s_i][0], d_sl))
    cc = as_real(read(pairs[s_i][1], d_sl))
    fd = np.isfinite(rc) & np.isfinite(cc)
    fo = np.isfinite(oc)
    n_bad += int(np.count_nonzero(fd & ~fo))
    f = fd & fo
    if not np.any(f):
      continue
    ou = np.ldexp(oc[f], -sh)
    o_sig += float(np.sum(np.square(ou)))
    e_c += float(np.sum(np.square(np.ldexp(cc[f], -sh) - ou)))
    e_r += float(np.sum(np.square(np.ldexp(rc[f], -sh) - ou)))
  return _db(o_sig, e_c), _db(o_sig, e_r), n_bad


def _core_metrics(ctx: LeafContext) -> dict[str, float | None]:
  """sqnr_db, best_fit_scale, si_snr_db, cosine_similarity from the sums.

  With s = <c, y> / <y, y> = 1 + <e, y> / <y, y>, the scale-invariant residual
  is |c - s y|**2 = |e|**2 - <e, y>**2 / |y|**2. This form subtracts error-
  sized terms only, so it stays accurate at high SQNR.

  Args:
    ctx: The leaf context.

  Returns:
    sqnr_db, best_fit_scale, si_snr_db and cosine_similarity by name.
  """
  sig, err, sum_ey = ctx.sig, ctx.err, ctx.sum_ey
  if sig <= 0.0:
    # The reference underflows after the prescale (the candidate is larger
    # by more than the float64 range). Only sqnr_db (-inf) is defined.
    return dict(sqnr_db=ctx.sqnr_db)
  cy = sig + sum_ey
  cc = max(sig + 2.0 * sum_ey + err, 0.0)
  scale = cy / sig
  resid = max(err - sum_ey * sum_ey / sig, 0.0)
  cos = cy / math.sqrt(sig * cc) if cc > 0.0 else None
  return dict(
      sqnr_db=ctx.sqnr_db,
      best_fit_scale=scale,
      si_snr_db=_db(scale * scale * sig, resid),
      cosine_similarity=cos,
  )


def compute_sqnr(
    actual: Any,
    expected: Any,
    *,
    oracle: Any = None,
    config: SqnrConfig | None = None,
    precision: PrecisionSpec | None = None,
    atol: float | None = None,
    path: str = "",
) -> SqnrLeafReport:
  """Measures core SQNR metrics of one array pair. Runs no anti-cheat checks.

  Args:
    actual: Candidate array (NumPy, ml_dtypes, or anything np.asarray takes).
    expected: Reference array.
    oracle: Optional high-precision oracle array of the same shape.
    config: Policy. Its `strict` field is ignored here.
    precision: Rounding sources for an AUTO threshold. None infers the output
      dtype from `expected`.
    atol: Absolute tolerance for an all-zero reference.
    path: Leaf name for messages.

  Returns:
    The report. `passed` covers the global floor and the oracle rule only.
  """
  config = dataclasses.replace(config or SqnrConfig(), strict=False)
  return audit_leaf(
      [(expected, actual)],
      config=config,
      precision=precision,
      oracle_pairs=None if oracle is None else [oracle],
      atol=atol,
      path=path,
  )


def audit_leaf(
    pairs: Sequence[tuple[Any, Any]],
    *,
    config: SqnrConfig | None = None,
    precision: PrecisionSpec | None = None,
    input_views: Sequence[Sequence[Any]] = (),
    oracle_pairs: Sequence[Any] | None = None,
    atol: float | None = None,
    path: str = "",
    read: Callable[[Any, tuple[slice, ...]], np.ndarray] = default_read,
) -> SqnrLeafReport:
  """Audits one output leaf, given as (reference, candidate) shards.

  The leaf rules run in this order: shape mismatch, discrete dtypes (exact
  match required), float dtypes, empty leaf, non-finite mask match, no finite
  elements, zero reference (allclose rule with `atol`), exact match. Then the
  global SQNR rule (or the oracle rule) and, in strict mode, the check family
  of `config.gate_math` run.

  Args:
    pairs: One (reference, candidate) pair per shard. Shapes must match per
      pair. Shards of one leaf are disjoint.
    config: Policy. None uses SqnrConfig().
    precision: Rounding sources for an AUTO threshold. None infers the output
      dtype from the reference.
    input_views: Optional input arrays for the input-delta check. Each entry is
      one view, given as one array per shard, with the shard shapes. Views of
      another shape, dtype or complexness are skipped. For a single-shard leaf,
      the transpose of a square view is added here.
    oracle_pairs: Optional oracle array per shard, same shapes as `pairs`.
    atol: Absolute tolerance for an all-zero reference.
    path: Leaf name for messages.
    read: Accessor `read(x, slices) -> np.ndarray`. Every array access goes
      through it, so callers can keep data on a device and copy one tile at a
      time. `slices` is () for the whole array.

  Returns:
    The report.
  """
  config = config or SqnrConfig()
  loc = path or "output"
  read = _reader(read)
  r_meta = [_shape_dtype(r, read) for r, _ in pairs]
  c_meta = [_shape_dtype(c, read) for _, c in pairs]
  ref_dt = r_meta[0][1] if pairs else None
  ref_name = str(ref_dt) if ref_dt is not None else "unknown"
  if precision is None:
    precision = PrecisionSpec(output_dtype=None if ref_dt is None else ref_name)
  th_db, th_src = resolve_threshold(config, precision)
  base: dict[str, Any] = dict(
      path=path,
      dtype=ref_name,
      threshold_db=th_db,
      threshold_source=th_src,
      gate_math=config.gate_math,
  )
  base_warnings: tuple[str, ...] = ()
  int_ops = _integer_operands(precision)
  if config.min_sqnr_db == AUTO and int_ops:
    base_warnings = (
        (
            f"{loc}: integer format ({', '.join(int_ops)}): AUTO does not model"
            f" integer quantization and uses {DEFAULT_MIN_SQNR_DB:g} dB; pass"
            " an explicit min_sqnr_db"
        ),
    )

  def _leaf(status: str, passed: bool, msg: str = "", **kw: Any):
    reasons = () if passed or not msg else (f"{loc}: {msg}",)
    return SqnrLeafReport(
        status=status,
        passed=passed,
        failed_reasons=reasons,
        warnings=base_warnings,
        **base,
        **kw,
    )

  def _oracle_check(
      tiles: Sequence[tuple[int, tuple[slice, ...]]], data_max: float
  ) -> tuple[dict[str, float], list[str]]:
    """Runs the oracle rule. Returns (metrics, failed reasons)."""
    if oracle_pairs is None:
      return {}, []
    o_c, o_r, n_bad = _oracle_sums(pairs, read, tiles, data_max, oracle_pairs)
    o_metrics = dict(
        oracle_sqnr_db=o_c,
        reference_oracle_sqnr_db=o_r,
        oracle_delta_db=o_c - o_r,
    )
    if n_bad:
      # The oracle cannot grade components that it does not cover, so the
      # leaf fails closed instead of skipping them.
      return o_metrics, [
          f"{loc}: oracle_non_finite: {n_bad} components where the reference"
          " and the candidate are finite"
      ]
    return o_metrics, _oracle_reasons(o_c, o_r, th_db, config, loc)

  def _exact_leaf(
      status: str,
      tiles: Sequence[tuple[int, tuple[slice, ...]]],
      data_max: float,
      **kw: Any,
  ) -> SqnrLeafReport:
    """Report for a candidate equal to the reference. The oracle still runs."""
    o_metrics, o_reasons = _oracle_check(tiles, data_max)
    oracle_fields: dict[str, Any] = {
        k: _finite_or_none(v) for k, v in o_metrics.items()
    }
    return SqnrLeafReport(
        status=status,
        passed=not o_reasons,
        failed_reasons=tuple(o_reasons),
        warnings=base_warnings,
        **base,
        **oracle_fields,
        **kw,
    )

  if not pairs:
    return _leaf(STATUS_NO_FINITE_ELEMENTS, True)
  shapes = [m[0] for m in r_meta]
  if any(rm[0] != cm[0] for rm, cm in zip(r_meta, c_meta)):
    return _leaf(STATUS_SHAPE_MISMATCH, False, "shape mismatch")
  if oracle_pairs is not None and (
      len(oracle_pairs) != len(pairs)
      or any(
          _shape_dtype(o, read)[0] != s for o, s in zip(oracle_pairs, shapes)
      )
  ):
    return _leaf(STATUS_SHAPE_MISMATCH, False, "oracle shape mismatch")
  d_tiles, _ = _leaf_tiles(shapes, config.block_size)
  dt, c_dt = r_meta[0][1], c_meta[0][1]
  if _is_discrete(dt) or _is_discrete(c_dt):
    ok = dt == c_dt and all(
        np.array_equal(read(pairs[s_i][1], sl), read(pairs[s_i][0], sl))
        for s_i, sl in d_tiles
    )
    bad = "dtype" if dt != c_dt else "discrete"
    if ok:
      return _leaf(STATUS_DISCRETE_EXACT, True)
    if bad == "dtype":
      return _leaf(STATUS_DTYPE_MISMATCH, False, "dtype mismatch")
    return _leaf(STATUS_DISCRETE_MISMATCH, False, "discrete mismatch")
  if (
      _eps_sq(dt) is None
      or _eps_sq(c_dt) is None
      or _is_complex(dt) != _is_complex(c_dt)
  ):
    return _leaf(STATUS_DTYPE_MISMATCH, False, "dtype mismatch")
  if sum(math.prod(s) for s in shapes) == 0:
    return _leaf(STATUS_NO_FINITE_ELEMENTS, True)
  scan = _scan_leaf(pairs, read, d_tiles)
  if scan is None:
    return _leaf(STATUS_NON_FINITE_MISMATCH, False, "non finite mask")
  if scan.n_finite == 0:
    return _leaf(STATUS_NO_FINITE_ELEMENTS, True)
  if scan.ref_max == 0.0:
    if scan.cand_max == 0.0:
      return _exact_leaf(STATUS_EXACT, d_tiles, 0.0)
    # Zero signal has no dB ratio, so apply the allclose rule at ref == 0:
    # every |cand| <= atol, where |.| is the modulus on complex leaves.
    g_abs = scan.cand_max
    if _is_complex(dt):
      g_abs = 0.0
      for s_i, sl in d_tiles:
        cc = as_real(read(pairs[s_i][1], sl))
        cc = np.where(np.isfinite(cc), cc, 0.0)
        mod = np.hypot(cc[..., 0], cc[..., 1])
        g_abs = max(g_abs, float(np.max(mod, initial=0.0)))
    ok_z = (
        atol is not None
        and math.isfinite(atol)
        and atol >= 0.0
        and g_abs <= atol
    )
    if not ok_z:
      return _leaf(STATUS_ZERO_SIGNAL, False, "nonzero error on zero reference")
    return _exact_leaf(STATUS_ZERO_SIGNAL, d_tiles, scan.cand_max)
  strict_v1 = config.strict and config.gate_math == GATE_MATH_SIX_GATE_V1
  ctx = _build_context(
      pairs,
      read,
      input_views,
      oracle_pairs,
      th_db,
      precision,
      block_size=config.block_size,
      scan=scan,
  )
  if ctx.max_abs_err == 0.0:
    exact_kw: dict[str, Any] = {}
    if strict_v1:
      exact_kw["peak_error_rms_ratio"] = 0.0
    return _exact_leaf(
        STATUS_EXACT, d_tiles, max(scan.ref_max, scan.cand_max), **exact_kw
    )

  metrics = _core_metrics(ctx)
  reasons: list[str] = []
  warnings: list[str] = list(base_warnings)
  if oracle_pairs is None:
    if ctx.sqnr_db < th_db:
      reasons.append(f"{loc}: sqnr_db {ctx.sqnr_db:.2f} < {th_db:.2f}")
  else:
    o_metrics, o_reasons = _oracle_check(
        ctx.disjoint_tiles, max(ctx.ref_max, ctx.cand_max)
    )
    metrics.update(o_metrics)
    reasons.extend(o_reasons)
  if config.strict:
    checks = _six_gate_v1_checks if strict_v1 else _derived_checks
    outcome = checks(ctx, config, loc)
    metrics.update(outcome.metrics)
    reasons.extend(outcome.reasons)
  metric_fields: dict[str, Any] = {
      k: _finite_or_none(v) for k, v in metrics.items()
  }
  return SqnrLeafReport(
      status=STATUS_OK,
      passed=not reasons,
      failed_reasons=tuple(reasons),
      warnings=tuple(warnings),
      **base,
      **metric_fields,
  )


def merge_reports(
    reports: Sequence[SqnrLeafReport | dict[str, Any]],
) -> dict[str, Any]:
  """Merges leaf or sample reports into one worst-case summary dict.

  Args:
    reports: SqnrLeafReport objects or dicts with the same keys (the status
      under `sqnr_status`).

  Returns:
    A dict with `sqnr_status` (worst), `passed` (all), `failed_reasons` and
    `warnings` (concatenated), the minimum of each MIN_MERGED_KEYS metric and
    the maximum of each MAX_MERGED_KEYS metric (None when no report has a
    value).
  """
  rank = {s: i for i, s in enumerate(STATUS_ORDER)}
  worst: str | None = None
  ok = True
  reasons: list[str] = []
  warnings: list[str] = []
  out: dict[str, Any] = {k: None for k in MIN_MERGED_KEYS + MAX_MERGED_KEYS}
  for r in reports:
    d = r.to_dict() if isinstance(r, SqnrLeafReport) else r
    ok = ok and bool(d.get("passed", False))
    reasons.extend(str(x) for x in d.get("failed_reasons") or ())
    warnings.extend(str(x) for x in d.get("warnings") or ())
    st = str(d.get("sqnr_status", STATUS_ERROR))
    if worst is None or rank.get(st, len(rank)) > rank.get(worst, len(rank)):
      worst = st
    for k in MIN_MERGED_KEYS + MAX_MERGED_KEYS:
      v = d.get(k)
      if isinstance(v, (int, float)) and math.isfinite(v):
        pick = min if k in MIN_MERGED_KEYS else max
        out[k] = float(v) if out[k] is None else pick(out[k], float(v))
  out.update(
      sqnr_status=worst or STATUS_EXACT,
      passed=ok,
      failed_reasons=reasons,
      warnings=warnings,
  )
  return out
