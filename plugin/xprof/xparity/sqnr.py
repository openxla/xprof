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

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
import dataclasses
import itertools
import math
import statistics
import types
from typing import Any

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
# The rate holds for noise that the fitted model E[e**2] = A v + B + C h of
# the derived checks describes (see the honest-noise model of the derived
# checks). Two float32 summation orders of the same bf16 matmul (K up to 4096,
# outputs up to 1048576 elements) gave 0 false failures in 620 seeds, a 95%
# upper bound of about 1 in 200 leaves.
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


def _canonical_format_name(
    name: Any, field: str, allow_none: bool = True
) -> str | None:
  """Returns the canonical name of a format, or None.

  A name in BLOCK_SCALED_FORMATS is returned in lower case. Any other name
  must be the canonical name of a float, complex, integer or bool dtype, that
  is, `str(np.dtype(name)) == name`. NumPy aliases are rejected: "i8" is
  int64 in NumPy, not int8.

  Args:
    name: A dtype or format name, or None.
    field: The PrecisionSpec field that holds `name`, for the error message.
    allow_none: Whether None is a valid value of the field.

  Returns:
    The canonical name, or None.

  Raises:
    TypeError: If `name` is not a string, or is None and `allow_none` is
      False.
    ValueError: If `name` is not a block-scaled format or the canonical name
      of a float, complex, integer or bool dtype. A misspelled name would
      otherwise change the AUTO floor without an error.
  """
  if name is None and allow_none:
    return None
  if not isinstance(name, str):
    raise TypeError(
        f"PrecisionSpec.{field} takes dtype names, not {name!r}. Use"
        " precision_from_dtypes to build a spec from dtype objects."
    )
  if name.lower() in BLOCK_SCALED_FORMATS:
    return name.lower()
  try:
    dt = np.dtype(name)
  except (TypeError, ValueError):
    dt = None
  if dt is not None and _numeric_dtype(dt) and str(dt) == name:
    return name
  hint = (
      f" NumPy reads it as {str(dt)!r}, so pass that name if you mean it."
      if dt is not None and _numeric_dtype(dt)
      else ""
  )
  raise ValueError(
      f"PrecisionSpec.{field}: {name!r} is not a known format.{hint} Use a"
      " dtype name such as 'bfloat16', 'float8_e4m3fn' or 'int8', or one of"
      f" {sorted(BLOCK_SCALED_FORMATS)}."
  )


def _numeric_dtype(dt: np.dtype) -> bool:
  """True for float, complex, integer (ml_dtypes included) and bool dtypes."""
  if dt == np.bool_:
    return True
  for info in (ml_dtypes.finfo, ml_dtypes.iinfo):
    try:
      info(dt)
    except (TypeError, ValueError):
      continue
    return True
  return False


@dataclasses.dataclass(frozen=True)
class PrecisionSpec:
  """Rounding sources that `expected_sqnr_db` turns into an expected SQNR.

  The fields take dtype names (strings). A dtype object raises TypeError:
  `precision_from_dtypes` builds a spec from dtype objects. Names in
  BLOCK_SCALED_FORMATS are case-insensitive. The spec stores them in lower
  case, so "MXFP4" and "mxfp4" give the same spec. Every other name must be
  the canonical NumPy or ml_dtypes name of a float, complex, integer or bool
  dtype ("bfloat16", "float8_e4m3fn", "int8"). Other names, such as "bf16",
  "fp8_e4m3" or the NumPy alias "i8" (int64), raise ValueError, because a
  name that `expected_sqnr_db` does not know would move the AUTO floor.

  Attributes:
    operand_dtypes: Dtype names of the inputs that the kernel rounds. Names in
      BLOCK_SCALED_FORMATS are accepted too.
    output_dtype: Dtype name of the output. None adds no output rounding source.
      `audit_leaf` and `compute_sqnr` use the reference dtype only when their
      `precision` argument is None.
    reduction_length: Reduction length K of the main contraction (an int >= 1),
      or None.
    accumulator_dtype: Accumulator dtype name. None means float32.
  """

  operand_dtypes: tuple[str, ...] = ()
  output_dtype: str | None = None
  reduction_length: int | None = None
  accumulator_dtype: str | None = None

  def __post_init__(self):
    # A bare string is iterable, and would become one name per character.
    if isinstance(self.operand_dtypes, str):
      raise TypeError(
          "PrecisionSpec.operand_dtypes takes a sequence of dtype names, not"
          f" the string {self.operand_dtypes!r}. Use"
          f" ({self.operand_dtypes!r},) for one operand."
      )
    # The dataclass is frozen, so the normalized names go in through
    # object.__setattr__.
    object.__setattr__(
        self,
        "operand_dtypes",
        tuple(
            _canonical_format_name(d, "operand_dtypes", allow_none=False)
            for d in self.operand_dtypes
        ),
    )
    object.__setattr__(
        self,
        "output_dtype",
        _canonical_format_name(self.output_dtype, "output_dtype"),
    )
    object.__setattr__(
        self,
        "accumulator_dtype",
        _canonical_format_name(self.accumulator_dtype, "accumulator_dtype"),
    )
    # A K <= 0 or a float K would be skipped silently by expected_sqnr_db.
    k = self.reduction_length
    if k is not None:
      if (
          isinstance(k, (bool, np.bool_))
          or not isinstance(k, (int, np.integer))
          or k < 1
      ):
        raise ValueError(f"reduction_length={k!r} must be None or an int >= 1.")
      object.__setattr__(self, "reduction_length", int(k))


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
    block_size: Elements per block along the minor axis. An int >= 2.
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
    for name in ("auto_margin_db", "oracle_delta_db"):
      # A NaN margin makes every floor comparison False, so a failing leaf
      # would pass.
      val = getattr(self, name)
      try:
        val_f = float(val)
      except (TypeError, ValueError) as e:
        raise ValueError(f"{name}={val!r} must be a finite float.") from e
      if isinstance(val, bool) or not math.isfinite(val_f):
        raise ValueError(f"{name}={val!r} must be a finite float.")
    # A str such as "false" is truthy, so only bool values are accepted.
    if not isinstance(self.strict, bool):
      raise ValueError(f"strict={self.strict!r} must be True or False.")
    # A float block size would pass `< 2` and fail later in tile slicing.
    bs = self.block_size
    if (
        isinstance(bs, (bool, np.bool_))
        or not isinstance(bs, (int, np.integer))
        or bs < 2
    ):
      raise ValueError(f"block_size={bs!r} must be an int >= 2.")
    object.__setattr__(self, "block_size", int(bs))
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
      min_sqnr_db: float | str | None = None,
      alpha: float = DEFAULT_ALPHA,
      strict: bool = True,
      preset: str | None = None,
  ) -> "SqnrConfig":
    """Builds a config from flat CLI or flag values.

    Args:
      min_sqnr_db: The global floor. None means not given: the default floor of
        the preset, AUTO for GATE_MATH_DERIVED and DEFAULT_MIN_SQNR_DB for
        GATE_MATH_SIX_GATE_V1. AUTO ("auto") means AUTO under every preset. A
        float or a numeric string sets that floor.
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
    kw: dict[str, Any] = dict(alpha=alpha, strict=strict)
    if min_sqnr_db is not None:
      kw["min_sqnr_db"] = min_sqnr_db
    if preset in (None, GATE_MATH_DERIVED):
      return cls(**kw)
    if preset == GATE_MATH_SIX_GATE_V1:
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
  and Mary 2019). Formats wider than float32 and integer formats add no
  source.

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


def _narrow_integer(dt: Any) -> bool:
  """Tells if dt is an integer dtype with fewer bits than float32 precision.

  A b-bit integer resolves 1 in 2**(b - 1) of its range, so from 32 bits up it
  is as fine as float32 (24 bits). int32 and int64 inputs are then indices,
  lengths or exact accumulators, not a rounding source. bool is a mask.

  Args:
    dt: A dtype.

  Returns:
    True for int16, int8, int4, int2 and the unsigned forms.
  """
  dt = np.dtype(dt)
  if dt == np.bool_ or not _is_discrete(dt):
    return False
  return int(ml_dtypes.iinfo(dt).bits) < _FLOAT32_PRECISION_BITS


def _integer_formats(precision: PrecisionSpec | None) -> tuple[str, ...]:
  """Returns the names in precision that are narrow integer formats.

  The operand dtypes, the output dtype and the accumulator dtype are scanned,
  in that order. The rule is `_narrow_integer`, the same as in
  `precision_from_dtypes`, so a hand-built spec and a spec from dtypes give
  the same floor. int32 and int64 are exact, so they are not in the result.
  Each name is listed once.

  Args:
    precision: The rounding sources, or None.
  """
  if precision is None:
    return ()
  out = []
  for name in (
      *precision.operand_dtypes,
      precision.output_dtype,
      precision.accumulator_dtype,
  ):
    if name is None or name.lower() in BLOCK_SCALED_FORMATS:
      continue
    try:
      dt = np.dtype(name)
    except TypeError:
      continue
    if _narrow_integer(dt) and name not in out:
      out.append(name)
  return tuple(out)


def resolve_threshold(
    config: SqnrConfig, precision: PrecisionSpec | None = None
) -> tuple[float, str]:
  """Returns (threshold_db, threshold_source) for a config.

  AUTO does not model integer quantization. With a narrow integer format
  (fewer than 24 bits, for example int16, int8 or int4) as an operand, output
  or accumulator dtype, AUTO falls back to DEFAULT_MIN_SQNR_DB, and
  `audit_leaf` adds a warning that asks for an explicit min_sqnr_db.

  Args:
    config: The policy.
    precision: Rounding sources for AUTO.

  Raises:
    ValueError: If the AUTO floor is not positive, for example from a narrow
      accumulator with a large reduction length. It is also raised for a
      discrete leaf, whose floor is not used.
  """
  if config.min_sqnr_db != AUTO:
    return float(config.min_sqnr_db), THRESHOLD_SOURCE_EXPLICIT
  if _integer_formats(precision):
    return DEFAULT_MIN_SQNR_DB, THRESHOLD_SOURCE_DEFAULT
  exp = expected_sqnr_db(precision) if precision is not None else None
  if exp is None:
    return DEFAULT_MIN_SQNR_DB, THRESHOLD_SOURCE_DEFAULT
  e_db, derivation = exp
  margin = config.auto_margin_db
  floor = e_db - margin
  # The same rule as for an explicit min_sqnr_db: a floor <= 0 dB passes any
  # candidate on the global floor.
  if floor <= 0.0:
    acc_note = (
        " The accumulator term is checked up to K = 65536."
        if "acc_" in derivation
        else ""
    )
    raise ValueError(
        f"AUTO floor {floor:.1f} dB ({derivation} - {margin:g} dB) is not"
        f" positive.{acc_note} Pass an explicit min_sqnr_db."
    )
  return (
      floor,
      f"{THRESHOLD_SOURCE_AUTO_PREFIX}{derivation}-{margin:g}dB",
  )


def _dtype_name(dtype: Any) -> str:
  """Returns the canonical name of a dtype, or a format name unchanged."""
  if isinstance(dtype, str) and dtype.lower() in BLOCK_SCALED_FORMATS:
    return dtype.lower()
  return str(np.dtype(dtype))


def _numeric_format_name(dtype: Any) -> str | None:
  """`_dtype_name`, or None for a dtype that is not a numeric format.

  The dtype is not parsed again from its name: the name of a structured or
  subarray dtype is not a valid `np.dtype` argument.

  Args:
    dtype: A dtype, dtype name or block-scaled format name.

  Returns:
    The canonical name, or None for a non-numeric dtype.

  Raises:
    TypeError: If `dtype` is not a valid dtype.
  """
  if isinstance(dtype, str) and dtype.lower() in BLOCK_SCALED_FORMATS:
    return dtype.lower()
  dt = np.dtype(dtype)
  return str(dt) if _numeric_dtype(dt) else None


def _narrower_than_float32(name: str) -> bool:
  """True for block-scaled formats, narrow integers and narrow floats."""
  if name.lower() in BLOCK_SCALED_FORMATS:
    return True
  dt = np.dtype(name)
  if _is_discrete(dt):
    return _narrow_integer(dt)
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
      include integer formats with fewer than 24 bits (int16, int8, int4,
      int2 and the unsigned forms), so that AUTO falls back to the default
      floor with a warning. Wider inputs add no rounding source: float32,
      float64, and int32 or int64 inputs such as indices, lengths or group
      sizes. bool inputs are masks and are left out too.
    - output_dtype: the candidate dtype, which is the format that the kernel
      rounds its output to.
    - accumulator_dtype and reduction_length: as given.
  A candidate or accumulator dtype that is not a float, complex, integer or
  bool dtype (for example object, str, structured or subarray) gives None. It
  adds no rounding source, and the audit fails such a leaf on its dtype. An
  input dtype of that kind is left out of operand_dtypes for the same reason.

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
    ValueError: If reduction_length is not None or an int >= 1.
  """
  _dtype_name(reference_dtype)
  operands = tuple(
      name
      for name in (_numeric_format_name(d) for d in input_dtypes)
      if name is not None and _narrower_than_float32(name)
  )
  return PrecisionSpec(
      operand_dtypes=operands,
      output_dtype=_numeric_format_name(candidate_dtype),
      reduction_length=reduction_length,
      accumulator_dtype=(
          None
          if accumulator_dtype is None
          else _numeric_format_name(accumulator_dtype)
      ),
  )


def _db(num: float, den: float) -> float:
  if den <= 0.0:
    return math.inf
  return -math.inf if num <= 0.0 else 10.0 * (math.log10(num) - math.log10(den))


def _min_db(num: np.ndarray, den: np.ndarray) -> float:
  """Smallest 10 log10(num / den), with the same edge cases as `_db`."""
  with np.errstate(all="ignore"):
    v = np.where(
        den <= 0.0,
        math.inf,
        np.where(
            num <= 0.0,
            -math.inf,
            10.0 * np.log10(num) - 10.0 * np.log10(den),
        ),
    )
  return float(np.min(v))


def _window_chunks(a: np.ndarray, w: int) -> np.ndarray:
  """Splits axis 0 into windows of w. The last window overlaps the previous.

  Args:
    a: Array with the window axis first.
    w: Window length.

  Returns:
    Shape (n_windows, w) + a.shape[1:], or a[None] when a.shape[0] <= w.
  """
  n = a.shape[0]
  if n <= w:
    return a[None]
  nf = (n // w) * w
  out = a[:nf].reshape((-1, w) + a.shape[1:])
  if nf == n:
    return out
  return np.concatenate([out, a[n - w :][None]], axis=0)


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


def _discrete_leaf(dtypes: Iterable[Any]) -> bool:
  """True if a leaf with these reference and candidate dtypes is discrete.

  A discrete leaf has a bool or integer reference or candidate dtype. It goes
  to the discrete rule in `_dtype_report` (equal dtypes and equal values).

  Args:
    dtypes: The reference and candidate dtypes of every shard.

  Returns:
    True if any dtype is bool or integer.
  """
  return any(_is_discrete(dt) for dt in dtypes)


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
  full block. The core sums and the six_gate_v1 checks use this plan, so that
  the preset keeps the tiles of the gates that it ports. The derived checks
  use `_row_tile_slices`.

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


def _row_tile_slices(
    shape: tuple[int, ...], block: int
) -> list[tuple[slice, ...]]:
  """Returns disjoint, row-contiguous tile slices for the derived checks.

  A leaf with at most _TILE_ELEMENTS elements is one tile, given as (). Larger
  leaves are split along the leading axes first, then along the rows (axis
  -2). The last axis is split only when two full rows do not fit in a tile.
  Then a tile has two rows, and its length is a multiple of `block` and of
  every BLOCK_SCALED_FORMATS block. Thus each tile reads contiguous rows, the
  scale blocks of a tile start at the same columns as in the leaf, and a tile
  has two or more rows and columns when the shard has them. A last piece of
  one row or one column joins the piece before it. When the alignment is at
  most _TILE_ELEMENTS // 2, a tile holds at most 1.5 * _TILE_ELEMENTS + 3
  elements.

  Args:
    shape: Shard shape.
    block: Block size along the last axis.
  """
  if math.prod(shape) <= _TILE_ELEMENTS:
    return [()]
  align = math.lcm(block, *(b for _, b in BLOCK_SCALED_FORMATS.values()))
  ndim = len(shape)
  steps = list(shape)
  for k in range(ndim):
    cur = math.prod(steps)
    if cur <= _TILE_ELEMENTS:
      break
    q = _TILE_ELEMENTS // (cur // steps[k])
    if k == ndim - 1:
      q = max(align, (q // align) * align)
    elif k == ndim - 2:
      q = max(2, q)
    steps[k] = min(steps[k], max(1, q))
  axes: list[tuple[slice, ...]] = []
  for k, (dim, step) in enumerate(zip(shape, steps)):
    starts = list(range(0, dim, step))
    if k >= ndim - 2 and len(starts) > 1 and dim - starts[-1] < 2:
      starts.pop()
    ends = starts[1:] + [dim]
    axes.append(tuple(slice(i0, i1) for i0, i1 in zip(starts, ends)))
  return list(itertools.product(*axes))


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
class _LeafContext:
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
    row_tiles: (shard index, slices) from `_row_tile_slices` that cover each
      shard once, for the derived checks.
    input_views: Input views that pass the filter (one array per shard each,
      same shard shapes, float or discrete dtype, same complexness). For a
      single-shard leaf, a square view is followed by its lazy transpose.
    cand_dtype: Candidate dtype.
    is_complex: True for complex leaves.
    eps_sq: finfo(reference dtype).eps ** 2.
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
    centered_sig: Sum over components of the centered sum of squares of y over
      finite elements (n_elements * variance).
    centered_cand: The same for the scaled candidate c.
    centered_err: The same for e.
    gmsq: Mean squared element magnitude of y: the sum of y**2 over finite
      elements divided by n_elements, or sig / n_finite when no element is
      finite in all components.
    rel_eac: Centered error energy over the sum of y**2 over finite elements
      (err / sig as a fallback). With n_elements <= 1 it uses the uncentered
      error energy.
    sum_ey: Sum of e * y over finite components.
    cand_sq: Sum of c**2 (c = scaled candidate), over finite components.
    sum_cy: Sum of c * y over finite components.
    ref_max: Max |reference| over finite components (unscaled).
    cand_max: Max |candidate| over finite components (unscaled).
  """

  pairs: Sequence[tuple[Any, Any]]
  read: Callable[[Any, tuple[slice, ...]], np.ndarray]
  disjoint_tiles: Sequence[tuple[int, tuple[slice, ...]]]
  overlap_tiles: Sequence[tuple[int, tuple[slice, ...]]]
  row_tiles: Sequence[tuple[int, tuple[slice, ...]]]
  input_views: Sequence[Sequence[Any]]
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
  centered_sig: float = 0.0
  centered_cand: float = 0.0
  centered_err: float = 0.0
  gmsq: float = 0.0
  rel_eac: float = 0.0
  sum_ey: float = 0.0
  cand_sq: float = 0.0
  sum_cy: float = 0.0
  ref_max: float = 0.0
  cand_max: float = 0.0


@dataclasses.dataclass(frozen=True)
class _CheckOutcome:
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
  """Result of the first pass over a leaf: finite counts and magnitudes.

  n_finite and n_non_finite count components, n_elements counts elements.
  """

  n_finite: int
  n_elements: int
  ref_max: float
  cand_max: float
  n_non_finite: int


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
  n_f = n_el = n_nf = 0
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
    cnt = int(np.count_nonzero(f))
    n_nf += f.size - cnt
    if cnt:
      n_f += cnt
      n_el += int(np.count_nonzero(np.all(f, -1)))
      r_max = max(r_max, float(np.max(np.abs(rc[f]))))
      c_max = max(c_max, float(np.max(np.abs(cc[f]))))
  return _LeafScan(n_f, n_el, r_max, c_max, n_nf)


def _filter_views(
    input_views: Sequence[Sequence[Any]],
    shapes: Sequence[tuple[int, ...]],
    is_complex: bool,
    full_n: int,
    read: Callable[[Any, tuple[slice, ...]], np.ndarray],
) -> list[list[Any]]:
  """Returns the input views that the input-delta checks can use.

  A view qualifies when it has one array per shard with the shard shapes, one
  dtype on every shard, a float dtype (or a discrete dtype on a leaf of more
  than one element), and the leaf complexness. For a single-shard leaf, a
  qualifying view with equal minor dimensions is followed by its lazy
  transpose.

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
    view_dts = {m[1] for m in metas}
    if len(view_dts) != 1:
      continue
    (dt,) = view_dts
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
    threshold_db: float,
    precision: PrecisionSpec,
    *,
    block_size: int = DEFAULT_BLOCK_SIZE,
    scan: _LeafScan | None = None,
) -> _LeafContext:
  """Builds the _LeafContext with one tiled pass of prescaled float64 sums.

  The caller has applied the leaf rules: shapes match, dtypes are float or
  complex and agree, the non-finite masks match, and the reference has a
  nonzero finite element.

  Args:
    pairs: The (reference, candidate) shards.
    read: The accessor.
    input_views: Candidate input views. They are filtered here.
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
  sig = err = sig_el = err_el = max_e = sum_ey = cand_sq = sum_cy = 0.0
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
    cand_sq += float(np.sum(np.square(cu)))
    sum_cy += float(np.sum(cu * yu))
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
  row_tiles = []
  for s_i, shape in enumerate(shapes):
    row_tiles.extend((s_i, sl) for sl in _row_tile_slices(shape, block_size))
  return _LeafContext(
      pairs=pairs,
      read=read,
      disjoint_tiles=d_tiles,
      overlap_tiles=o_tiles,
      row_tiles=row_tiles,
      input_views=_filter_views(input_views, shapes, is_cplx, full_n, read),
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
      centered_sig=float(np.sum(cm2[0])),
      centered_cand=float(np.sum(cm2[1])),
      centered_err=float(np.sum(cm2[2])),
      gmsq=gmsq,
      rel_eac=rel_eac,
      sum_ey=sum_ey,
      cand_sq=cand_sq,
      sum_cy=sum_cy,
      ref_max=scan.ref_max,
      cand_max=scan.cand_max,
  )


def _projection_residual(
    sig: np.ndarray | float,
    cand: np.ndarray | float,
    err: np.ndarray | float,
    *,
    den_floor: float = 1e-300,
) -> np.ndarray:
  """Energy of y that its projection on c leaves, from sums of squares.

  With <y, c> = max((|y|**2 + |c|**2 - |c - y|**2) / 2, 0), the residual is
  |y|**2 - <y, c>**2 / max(|c|**2, den_floor), and |y|**2 where |c|**2 is 0.
  The result is clamped at 0. All inputs broadcast elementwise.

  Args:
    sig: |y|**2.
    cand: |c|**2.
    err: |c - y|**2.
    den_floor: Lowest denominator. Pass 0.0 to divide by |c|**2 itself.

  Returns:
    The residual energy, with the broadcast shape of the inputs.
  """
  sig, cand, err = np.asarray(sig), np.asarray(cand), np.asarray(err)
  cov = np.maximum(0.5 * (sig + cand - err), 0.0)
  with np.errstate(divide="ignore", invalid="ignore"):
    resid = np.where(
        cand > 0.0, sig - cov * cov / np.maximum(cand, den_floor), sig
    )
  return np.maximum(resid, 0.0)


def _input_view_shift(
    ctx: _LeafContext, iv_shards: Sequence[Any]
) -> int | None:
  """Returns the prescale exponent for one input view, or None.

  The view is not usable (None) when one of its components is not finite
  where the reference component is finite.

  Args:
    ctx: The leaf context.
    iv_shards: The view, one array per shard.

  Returns:
    ctx.shift, raised to the frexp exponent of the largest view magnitude
    over finite reference components. None when the view is not usable.
  """
  view_max = 0.0
  for s_i, d_sl in ctx.disjoint_tiles:
    ref_c = as_real(ctx.read(ctx.pairs[s_i][0], d_sl))
    view_c = as_real(ctx.read(iv_shards[s_i], d_sl))
    if np.any(ref_finite := np.isfinite(ref_c)):
      if not np.all(np.isfinite(view_c)[ref_finite]):
        return None
      view_max = max(view_max, float(np.max(np.abs(view_c[ref_finite]))))
  if view_max == 0.0:
    return ctx.shift
  return max(ctx.shift, math.frexp(view_max)[1])


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


def _v1_win_stats(
    ycf: np.ndarray, ecf: np.ndarray, w: int, eps_sq: float, gmsq: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Per-window signal, error, centered signal, centered error, resolved."""
  yc, ec = _window_chunks(ycf, w), _window_chunks(ecf, w)
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


@dataclasses.dataclass(frozen=True)
class _V1Tile:
  """Prescaled data of one tile for the six_gate_v1 checks.

  Attributes:
    ref: Reference over 2**shift, 0 at elements that are not finite.
    cand: Candidate over 2**shift, 0 at elements that are not finite.
    err: cand - ref, 0 at elements that are not finite.
    ref_el: Finite elements of ref, flattened to (n_elements, components).
    cand_el: Finite elements of cand, flattened the same way.
    err_el: Finite elements of err, flattened the same way.
    finite: Mask of finite elements (all components finite).
    all_finite: True when every element of the tile is finite.
    n_elements: Count of finite elements.
    target_noise: Noise-to-signal energy ratio at target_db.
  """

  ref: np.ndarray
  cand: np.ndarray
  err: np.ndarray
  ref_el: np.ndarray
  cand_el: np.ndarray
  err_el: np.ndarray
  finite: np.ndarray
  all_finite: bool
  n_elements: int
  target_noise: float


@dataclasses.dataclass(frozen=True)
class _V1AxisRows:
  """Tile data with one axis moved last, as rows along that axis.

  Attributes:
    ref: Reference rows, shape (rows, axis length, components).
    cand: Candidate rows, the same shape.
    err: Error rows, the same shape.
    ref_flat: Finite reference elements of all rows, shape (n, components).
    cand_flat: Finite candidate elements, the same shape.
    err_flat: Finite error elements, the same shape.
  """

  ref: np.ndarray
  cand: np.ndarray
  err: np.ndarray
  ref_flat: np.ndarray
  cand_flat: np.ndarray
  err_flat: np.ndarray


def _v1_prepare_tile(
    r_arr: np.ndarray, c_arr: np.ndarray, ctx: _LeafContext, target_db: float
) -> _V1Tile | None:
  """Prescales one tile. Returns None when the tile has no error to check."""
  ref_c, cand_c = as_real(r_arr), as_real(c_arr)
  finite = np.isfinite(ref_c) & np.isfinite(cand_c)
  if not np.any(finite):
    return None
  ref_u = np.where(finite, np.ldexp(ref_c, -ctx.shift), 0.0)
  cand_u = np.where(finite, np.ldexp(cand_c, -ctx.shift), 0.0)
  err_u = np.where(finite, cand_u - ref_u, 0.0)
  finite_el = np.all(finite, -1)
  n_el = int(np.count_nonzero(finite_el))
  if float(np.max(np.abs(err_u))) == 0.0 or n_el == 0:
    return None
  all_finite = bool(np.all(finite_el))
  el_mask = finite_el[..., None]
  return _V1Tile(
      ref=ref_u if all_finite else np.where(el_mask, ref_u, 0.0),
      cand=cand_u if all_finite else np.where(el_mask, cand_u, 0.0),
      err=err_u if all_finite else np.where(el_mask, err_u, 0.0),
      ref_el=ref_u[finite_el],
      cand_el=cand_u[finite_el],
      err_el=err_u[finite_el],
      finite=finite_el,
      all_finite=all_finite,
      n_elements=n_el,
      target_noise=10.0 ** (-target_db / 10.0),
  )


def _v1_axis_rows(tile: _V1Tile, axis: int) -> _V1AxisRows:
  """Returns the tile data as rows along `axis`."""
  length, c_dim = tile.finite.shape[axis], tile.ref.shape[-1]
  rows_finite = np.moveaxis(tile.finite, axis, -1).reshape(-1, length)
  ref, cand, err = (
      np.moveaxis(t, axis, -2).reshape(-1, length, c_dim)
      for t in (tile.ref, tile.cand, tile.err)
  )
  if tile.all_finite:
    ref_flat, cand_flat, err_flat = (
        t.reshape(-1, c_dim) for t in (ref, cand, err)
    )
  else:
    ref_flat, cand_flat, err_flat = (
        ref[rows_finite],
        cand[rows_finite],
        err[rows_finite],
    )
  return _V1AxisRows(ref, cand, err, ref_flat, cand_flat, err_flat)


def _v1_is_windowed(n: int, ctx: _LeafContext) -> bool:
  """True when an axis with n finite elements is split into blocks."""
  bs = ctx.block_size
  return not (n < bs or (n == bs and ctx.full_n <= bs))


def _v1_axis_blocks(
    rows: _V1AxisRows, ctx: _LeafContext, tile: _V1Tile
) -> tuple[float, float | None]:
  """Block SQNR and AC SNR along one axis.

  Args:
    rows: The tile rows along the axis.
    ctx: The leaf context.
    tile: The tile.

  Returns:
    (min block SQNR, inf when no block has error; min AC SNR over the
    resolved high-variance blocks, or None when there is none).
  """
  ref_flat, err_flat = rows.ref_flat, rows.err_flat
  n, bs = ref_flat.shape[0], ctx.block_size
  floor_msq = _V1_FLOOR_FRACTION * ctx.gmsq
  if not _v1_is_windowed(n, ctx):
    # The axis is one short block.
    if n > 0 and (err_e := float(np.sum(np.square(err_flat)))) > 0.0:
      sig_e = max(float(np.sum(np.square(ref_flat))), n * floor_msq)
      return _db(sig_e, err_e), None
    return math.inf, None
  b_sig, b_err, b_var, b_eac, resolved = _v1_win_stats(
      ref_flat, err_flat, bs, ctx.eps_sq, ctx.gmsq
  )
  block_db = math.inf
  if np.any(active := b_err > 0.0):
    block_db = _min_db(np.maximum(b_sig[active], bs * floor_msq), b_err[active])
  var_floor = min(
      _V1_TNF_FACTOR * tile.target_noise, _V1_REL_EAC_AC * ctx.rel_eac
  )
  resolved = resolved & (b_var > var_floor * np.maximum(b_sig, bs * ctx.gmsq))
  if not np.any(resolved):
    return block_db, None
  return block_db, _min_db(b_var[resolved], b_eac[resolved])


def _v1_macro_corr(
    rows: _V1AxisRows, ctx: _LeafContext, tile: _V1Tile
) -> float | None:
  """Min correlation SNR over live macro windows along one axis, or None."""
  mw = _V1_MACRO_WINDOW
  if rows.ref_flat.shape[0] < mw:
    return None
  w_sig, _, w_var, w_eac, resolved = _v1_win_stats(
      rows.ref_flat, rows.err_flat, mw, ctx.eps_sq, ctx.gmsq
  )
  cand_w = _window_chunks(rows.cand_flat, mw)
  cand_w = cand_w - np.min(cand_w, axis=1, keepdims=True)
  w_cvar = np.sum(np.square(cand_w - np.mean(cand_w, 1, keepdims=True)), (1, 2))
  var_floor = min(
      _V1_TNF_FACTOR * tile.target_noise, _V1_REL_EAC_LOW * ctx.rel_eac
  ) * np.maximum(w_sig, mw * ctx.gmsq)
  dead_msq = _V1_DEAD_FRACTION * ctx.gmsq
  resolved &= (w_sig > mw * dead_msq) & (w_var > var_floor)
  if not np.any(resolved):
    return None
  resid = _projection_residual(
      w_var[resolved], w_cvar[resolved], w_eac[resolved]
  )
  return _min_db(w_var[resolved], resid)


def _v1_slice_checks(
    rows: _V1AxisRows, slice_n: int, ctx: _LeafContext, tile: _V1Tile
) -> tuple[float | None, float | None]:
  """Checks the slices across one axis of a tile with 2 or more axes.

  Loud slices get an AC SNR. Quiet slices get a projection correlation SNR.

  Args:
    rows: The tile rows along the axis.
    slice_n: Mean count of finite elements per slice.
    ctx: The leaf context.
    tile: The tile.

  Returns:
    (min AC SNR of loud slices or None, min correlation SNR of quiet slices
    or None).
  """
  gmsq, rel_eac, target_noise = ctx.gmsq, ctx.rel_eac, tile.target_noise
  s_sig = np.sum(np.square(rows.ref), (0, 2))
  s_err = np.sum(np.square(rows.err), (0, 2))
  # Honest noise correlates with a short quiet slice by chance. With r the
  # slice signal over the mean noise energy, the 0.5 dB band below fails
  # when the noise projection u ~ N(0, 1 / selem) is under 0.35 - sqrt(r).
  # r >= sqrt(32 / selem) keeps that at least 3.4 sigma away.
  quiet_fl = (
      max(_V1_SLICE_REL, math.sqrt(_V1_SLICE_MIN_SAMPLES / slice_n)) * rel_eac
  )
  live_fl = (
      slice_n * max(_V1_EPS_FACTOR * ctx.eps_sq, quiet_fl, _V1_SLICE_ABS) * gmsq
  )
  live = (s_err > 0.0) & (s_sig > live_fl)
  if not np.any(live):
    return None, None
  loud = live & (
      s_sig > slice_n * _V1_TNF_FACTOR * max(target_noise, rel_eac) * gmsq
  )
  loud_ac = _min_db(s_sig[loud], s_err[loud]) if np.any(loud) else None
  quiet_corr = None
  if np.any(quiet := live & ~loud):
    s_cand = np.sum(np.square(rows.cand), (0, 2))
    resid = _projection_residual(s_sig[quiet], s_cand[quiet], s_err[quiet])
    quiet_corr = _min_db(s_sig[quiet], resid)
  return loud_ac, quiet_corr


def _v1_plaquette_ac(
    tile: _V1Tile, axes: Sequence[int], ctx: _LeafContext
) -> float | None:
  """AC SNR of the 2D second differences over the two minor axes, or None."""
  eps_sq, rel_eac, target_noise = ctx.eps_sq, ctx.rel_eac, tile.target_noise
  finite, c_dim = tile.finite, tile.ref.shape[-1]
  ax0, ax1 = axes[-2], axes[-1]
  perm = [ax for ax in range(finite.ndim) if ax not in (ax0, ax1)] + [ax0, ax1]
  h, w = finite.shape[ax0], finite.shape[ax1]
  fin2 = np.transpose(finite, perm).reshape(-1, h, w)
  ref2 = np.transpose(tile.ref, perm + [finite.ndim]).reshape(-1, h, w, c_dim)
  err2 = np.transpose(tile.err, perm + [finite.ndim]).reshape(-1, h, w, c_dim)
  quad = (
      fin2[:, :-1, :-1] & fin2[:, 1:, :-1] & fin2[:, :-1, 1:] & fin2[:, 1:, 1:]
  )
  if not np.any(quad):
    return None
  quad_c = quad[..., None]
  d2_ref = np.diff(np.diff(ref2, 1, 2), 1, 1)
  d2_err = np.diff(np.diff(err2, 1, 2), 1, 1)
  p_var = 0.25 * float(np.sum(np.square(np.where(quad_c, d2_ref, 0.0))))
  p_err = 0.25 * float(np.sum(np.square(np.where(quad_c, d2_err, 0.0))))
  sq = np.sum(np.square(ref2), -1)
  quad_sq = sq[:, :-1, :-1] + sq[:, 1:, :-1] + sq[:, :-1, 1:] + sq[:, 1:, 1:]
  p_sig = 0.25 * float(np.sum(np.where(quad, quad_sq, 0.0)))
  corner = ref2[:, :-1, :-1]
  nbr_sq = np.square(corner - ref2[:, 1:, :-1]) + np.square(
      corner - ref2[:, :-1, 1:]
  )
  p_dev = 0.5 * float(np.sum(np.where(quad_c, nbr_sq, 0.0)))
  resolved = p_var > max(
      _V1_EPS_FACTOR * eps_sq * p_sig, _V1_RESOLVE_REL * p_dev
  )
  high_var = p_var > _V1_TNF_FACTOR * target_noise * p_sig or (
      p_var > _V1_REL_EAC_LOW * rel_eac * p_sig
      and p_dev > _V1_PLAQ_DIFF_RATIO * p_var
  )
  if p_err > 0.0 and resolved and high_var:
    return _db(p_var, p_err)
  return None


def _v1_complex_parts(
    tile: _V1Tile, ctx: _LeafContext
) -> tuple[list[float], list[float]]:
  """Checks the real and imaginary parts of a complex tile one by one.

  A loud part gets an AC SNR. A quiet part gets a projection correlation SNR.

  Args:
    tile: The tile.
    ctx: The leaf context.

  Returns:
    (AC SNRs of the loud parts, correlation SNRs of the quiet parts).
  """
  ref_el, cand_el, err_el = tile.ref_el, tile.cand_el, tile.err_el
  sig_el = float(np.sum(np.square(ref_el)))
  rel_eac = ctx.rel_eac
  part_fl = (
      max(
          _V1_EPS_FACTOR * ctx.eps_sq,
          _V1_COMPLEX_REL * rel_eac,
          _V1_COMPLEX_ABS,
      )
      * sig_el
  )
  loud_fl = _V1_TNF_FACTOR * max(tile.target_noise, rel_eac) * sig_el
  ac_values: list[float] = []
  corr_values: list[float] = []
  for part in (0, 1):
    p_sig = float(np.sum(np.square(ref_el[:, part])))
    p_err = float(np.sum(np.square(err_el[:, part])))
    if not (p_err > 0.0 and p_sig > part_fl):
      continue
    if p_sig > loud_fl:
      ac_values.append(_db(p_sig, p_err))
    else:
      p_cand = float(np.sum(np.square(cand_el[:, part])))
      resid = _projection_residual(p_sig, p_cand, p_err, den_floor=0.0)
      corr_values.append(_db(p_sig, float(resid)))
  return ac_values, corr_values


def _v1_peak_ratio(tile: _V1Tile, gmsq: float, target_db: float) -> float:
  """Ratio of peak normalized error to allowed noise RMS at target_db."""
  e_mag = np.sqrt(np.sum(np.square(tile.err_el), -1))
  y_mag = np.sqrt(np.sum(np.square(tile.ref_el), -1))
  denom = np.maximum(y_mag, math.sqrt(max(gmsq, 1e-300)))
  max_rel = float(np.max(e_mag / denom))
  lp = -math.inf if max_rel <= 0.0 else math.log10(max_rel) + target_db / 20.0
  return 0.0 if lp < -300.0 else (math.inf if lp > 300.0 else 10.0**lp)


def _v1_worst_corr(
    corrs: Sequence[tuple[float, float]],
) -> tuple[float, float] | None:
  """Returns the (value, floor) pair furthest below its floor, or None.

  The first pair wins a tie.

  Args:
    corrs: (correlation SNR, floor) pairs in check order.
  """
  worst: tuple[float, float] | None = None
  for val, floor in corrs:
    if val < floor and (worst is None or (val - floor) < (worst[0] - worst[1])):
      worst = (val, floor)
  return worst


def _v1_spatial_tile(
    r_arr: np.ndarray,
    c_arr: np.ndarray,
    ctx: _LeafContext,
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
  tile = _v1_prepare_tile(r_arr, c_arr, ctx, target_db)
  if tile is None:
    return math.inf, None, True, None, 0.0
  finite = tile.finite
  axes = [k for k in range(finite.ndim) if finite.shape[k] > 1]
  block_db = math.inf if axes else ctx.sqnr_db
  ac_values: list[float] = []
  corrs: list[tuple[float, float]] = []
  for k in axes:
    rows = _v1_axis_rows(tile, k)
    axis_block_db, axis_ac = _v1_axis_blocks(rows, ctx, tile)
    block_db = min(block_db, axis_block_db)
    if axis_ac is not None:
      ac_values.append(axis_ac)
    if _v1_is_windowed(rows.ref_flat.shape[0], ctx):
      if (macro := _v1_macro_corr(rows, ctx, tile)) is not None:
        corrs.append((macro, _V1_MACRO_CORR_DB))
    slice_n = tile.n_elements // finite.shape[k]
    if len(axes) >= 2 and slice_n >= ctx.block_size:
      loud_ac, quiet_corr = _v1_slice_checks(rows, slice_n, ctx, tile)
      if loud_ac is not None:
        ac_values.append(loud_ac)
      if quiet_corr is not None:
        corrs.append((quiet_corr, _V1_SLICE_CORR_DB))
  if len(axes) >= 2:
    if (plaq := _v1_plaquette_ac(tile, axes, ctx)) is not None:
      ac_values.append(plaq)
  if tile.ref.shape[-1] == 2:
    part_ac, part_corr = _v1_complex_parts(tile, ctx)
    ac_values.extend(part_ac)
    corrs.extend((val, _V1_SLICE_CORR_DB) for val in part_corr)
  ac_db = min(ac_values) if ac_values else None
  ac_ok = all(val >= _V1_AC_MIN_DB for val in ac_values)
  peak = _v1_peak_ratio(tile, ctx.gmsq, target_db)
  return block_db, ac_db, ac_ok, _v1_worst_corr(corrs), peak


def _v1_centered(
    ctx: _LeafContext, target_db: float, loc: str
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
      # A clamped covariance of 0 leaves g_var, as the unclamped rule does.
      p_err = float(_projection_residual(g_var, g_cvar, g_eac, den_floor=0.0))
      if (p_db := _db(g_var, p_err)) < _V1_CENTERED_CORR_DB:
        reasons.append(
            f"{loc}: centered_corr_snr_db {p_db:.2f} <"
            f" {_V1_CENTERED_CORR_DB:.2f}"
        )
  return g_cdb, reasons


def _v1_input_delta(
    ctx: _LeafContext, target_db: float, loc: str
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
    psi = _input_view_shift(ctx, iv_shards)
    if psi is None:
      continue
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
    ctx: _LeafContext, config: SqnrConfig, loc: str
) -> _CheckOutcome:
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
  return _CheckOutcome(
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
# Honest-noise model of the derived checks: E[e_i**2] = A v_i + B + C h_i.
# Each output y_i is rounded once to a grid with step u_i (the candidate dtype
# or a PrecisionSpec format), so the error has E[e_i**2] = v_i = u_i**2 / 12
# (Widrow and Kollar, "Quantization Noise", 2008). Accumulation and upstream
# rounding add noise that does not scale with u_i: the B term. The B term also
# covers error near zero that comes from larger inputs, so u_i has no near-zero
# floor: a floor at 2**-8 max|y| overstates the step of small elements, biases
# A low, and makes outlier elements look like spikes. The noise of a sum of
# products also scales with its energy sum_k a_k**2 b_k**2, which varies per
# row and column when one operand has per-channel outliers: the C h term, with
# h a row-column estimate of that energy (`_dc_energy`). The checks fit A, B
# and C by least squares on the leaf and test the fit against the leaf
# structure. The fit keeps A >= _DC_MIN_A, so every element keeps its rounding
# variance. The checks then grade each element with max(h, y**2) in place of
# h: when one product holds most of the energy of an element (heavy-tailed
# inputs), the row-column mean h is far below y**2, and the honest error of
# the input rounding is about eps_in |y|. The spike bound of an element with
# y**2 below the resolution of the reference dtype at the mean reference power
# also allows e**2 = y**2: an honest kernel can flush it to zero.
# A fitted term can take in extra noise of the candidate. On inputs with
# per-channel outliers, the fitted C can take in extra noise in the
# low-energy columns: only the global SQNR floor bounds it there.

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
# y, but independent of it, gives 1/2. 1/sqrt(2) is 1.5 dB from both, so the
# gate tests the first and skips the second: noise in place of a small part
# (for example the AC of a DC-dominated leaf, or a quiet complex part) is not
# tested, and passes when the block-noise check and the global floor pass. A
# skipped window with a significantly negative correlation on the centered
# transform is still tested (see `_dc_window_corr`).
_DC_CORR_GATE = 1.0 / math.sqrt(2.0)
# A window needs at least 4 z**2 samples, so that the r**2 floor z**2 / n is
# at most 1/4, below the honest minimum k / (1 + k) = 0.41.
_DC_MIN_SAMPLES_Z2 = 4.0
# The reference dtype resolves an AC energy above 16 eps**2 of the energy of
# the reference values that enter the window: a standard deviation of 4 eps,
# that is 2 ulp at the top binade.
_DC_RESOLVE_EPS_FACTOR = 16.0
# An input view is a residual base only when it is at least one bit closer to
# y than zero: 20 log10(2) dB.
_DC_DELTA_MIN_IN_DB = 20.0 * math.log10(2.0)
# By the same one-bit rule, the update is present in the candidate when the
# gain g = sum(d (c - x)) / sum(d**2) of c - x on d = y - x is within one bit
# of 1: |1 - g| < 1/2. An honest candidate with a small systematic gain error
# (an approximate function, a different normalization) stays inside it. A
# passthrough, an attenuated, wrong-sign or 2x update, or noise in place of
# the update is outside it.
_DC_DELTA_GAIN_TOL = 10.0 ** (-_DC_DELTA_MIN_IN_DB / 20.0)
# Tests of the input-delta family that use alpha: gain, sign count and
# correlation. Each gets alpha / _DC_DELTA_TESTS of the family (a Bonferroni
# split), so the family stays at or below alpha.
_DC_DELTA_TESTS = 3
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
# fit. Five rounds are a fixed cost, not a convergence test. The weights only
# set how the bins are balanced: each round is a valid weighted least-squares
# fit, and the checks grade against the last one.
_DC_FIT_ITERATIONS = 5
# Floor of the fit variance relative to its largest bin. 1e-12 caps the
# weight ratio at 1e24, so the weighted sums stay finite in float64.
_DC_FIT_REL_FLOOR = 1e-12

# A rounding format: (nmant, minexp, emax, block), see `_dc_formats`.
_DcFormat = tuple[int, int, int, int]


@dataclasses.dataclass(frozen=True)
class _DcTransform:
  """One correlation transform of a tile, flattened over the tile rows.

  Attributes:
    name: Report field name of the transform.
    y: Transformed reference, shape (n, components).
    c: Transformed candidate, same shape.
    v: Honest rounding variance of the transformed values, same shape.
    y2: Sum of the squares of the reference values that enter each transformed
      value, same shape.
    mask: Element mask, shape (n,).
  """

  name: str
  y: np.ndarray
  c: np.ndarray
  v: np.ndarray
  y2: np.ndarray
  mask: np.ndarray


def _dc_z(tail: float) -> float:
  """Returns z with P(N(0, 1) > z) = tail."""
  return -statistics.NormalDist().inv_cdf(min(max(tail, 1e-300), 0.5))


def _dc_chi2_upper(n: np.ndarray, z: float) -> np.ndarray:
  """Upper quantile of chi2_n / n at normal score z (Wilson-Hilferty)."""
  h = 2.0 / (9.0 * np.maximum(n, 1.0))
  return np.maximum(1.0 - h + z * np.sqrt(h), 0.0) ** 3


def _dc_formats(ctx: _LeafContext) -> list[_DcFormat]:
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
    y: np.ndarray,
    fmts: Sequence[_DcFormat],
    ctx: _LeafContext,
    capped: bool = True,
) -> np.ndarray:
  """Returns the honest rounding variance v = u**2 / 12 of each component.

  Args:
    y: Scaled reference tile, shape core + (components,).
    fmts: The output of `_dc_formats`.
    ctx: The leaf context.
    capped: Whether to cap v at y**2. The error of an element below u / 2 is at
      most |y|, so the cap gives the tighter bound on its error. The correlation
      gate uses the uncapped v: an element below u / 2 can round to 0, so its AC
      is not a signal that the candidate must keep.

  An exact zero has no exponent (frexp(0) gives 0, as for values in [0.5, 1)).
  It gets the exponent `low`, below every format, so its step is the subnormal
  step of the format: the limit of u as y goes to 0. A block of zeros has no
  scale, so its step is 0. The uncapped v of a zero is then not a top-binade
  step, which would mask the AC of a sparse or ReLU leaf from the correlation
  gate.
  """
  a = np.abs(y)
  # Far below every float64 exponent, and 2 * low still fits in int64.
  low = np.int64(np.iinfo(np.int32).min // 2)
  e_y = np.where(a > 0.0, np.frexp(a)[1] - 1, low)
  ex = np.full(y.shape, low, np.int64)
  for nmant, minexp, emax, blk in fmts:
    lo = np.int64(minexp - ctx.shift)
    if blk and y.ndim >= 2:
      a_el = np.max(a, -1)
      starts = np.arange(0, a_el.shape[-1], blk)
      amax = np.maximum.reduceat(a_el, starts, axis=-1)
      cnt = np.diff(np.append(starts, a_el.shape[-1]))
      e_amax = np.where(amax > 0.0, np.frexp(amax)[1] - 1 - emax, low)
      e_s = np.repeat(e_amax, cnt, axis=-1)[..., None]
      # The scale maps amax into [2**emax, 2**(emax + 1)), above the largest
      # element value for E2M1 (6 of [4, 8)). The top binade saturates, with
      # error up to one step, so its step counts double.
      top = (e_y >= e_s + emax).astype(np.int64)
      ex = np.maximum(ex, np.maximum(e_y, e_s + minexp) - nmant + top)
      continue
    ex = np.maximum(ex, np.maximum(e_y, lo) - nmant)
  if not capped:
    return np.ldexp(1.0 / 12.0, 2 * ex)
  # An element below u / 2 can round to 0, so its error is at most |y|. This
  # also gives v = 0 at y = 0, where a rounding is exact.
  return np.minimum(np.ldexp(1.0 / 12.0, 2 * ex), np.square(y))


def _dc_tile(
    ctx: _LeafContext,
    s_i: int,
    sl: tuple[slice, ...],
    fmts: Sequence[_DcFormat],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
  """Returns scaled (y, c, v_uncapped, mask, h) for a tile.

  y, c and v_uncapped have the shape (rows, L, components). The rows are the
  rows of the (h, L) planes of the tile, plane after plane, so rows is a
  multiple of h. h is the tile length along axis -2 of the leaf, or 1 for a
  leaf with fewer than two axes. y and c are zero where an element is not
  finite. v_uncapped = u**2 / 12, and 0 where an element is not finite. The
  capped v of `_dc_var` is np.minimum(v_uncapped, y**2). A tile of an empty
  shard gives size-0 arrays.

  Args:
    ctx: The leaf context.
    s_i: Shard index.
    sl: Tile slices.
    fmts: The output of `_dc_formats`.
  """
  rc = as_real(ctx.read(ctx.pairs[s_i][0], sl))
  cc = as_real(ctx.read(ctx.pairs[s_i][1], sl))
  if rc.size == 0:
    # An empty shard, for example (0,) or (4, 0). Callers skip size-0 tiles.
    empty = np.zeros((0, 0, rc.shape[-1]))
    return empty, empty, empty, np.zeros((0, 0), dtype=bool), 1
  if rc.ndim == 1:
    rc, cc = rc[None, None], cc[None, None]
  # The last axis of rc holds the components, so axis -3 is axis -2 of the
  # leaf.
  h = rc.shape[-3] if rc.ndim >= 3 else 1
  m = np.all(np.isfinite(rc) & np.isfinite(cc), -1)
  y = np.where(m[..., None], np.ldexp(rc, -ctx.shift), 0.0)
  c = np.where(m[..., None], np.ldexp(cc, -ctx.shift), 0.0)
  vu = np.where(m[..., None], _dc_var(y, fmts, ctx, capped=False), 0.0)
  shape = (-1,) + y.shape[-2:]
  return (
      y.reshape(shape),
      c.reshape(shape),
      vu.reshape(shape),
      m.reshape(shape[:-1]),
      h,
  )


def _dc_block_sums(t: np.ndarray, starts: np.ndarray) -> np.ndarray:
  """Sums `t` over each block of the minor axis, flattened over the rows."""
  return np.add.reduceat(np.sum(t, -1), starts, axis=1).ravel()


def _dc_subnormal_var(ctx: _LeafContext, fmts: Sequence[_DcFormat]) -> float:
  """Returns u**2 / 12 for the finest subnormal step u of the element formats.

  The step is in prescaled units. It is capped at 1, the prescaled leaf
  scale: every prescaled |y| and |c| is below 1, so a coarser step cannot
  apply. The cap also keeps `math.ldexp` and the squares of the result in
  range. Block-scaled formats are left out: their subnormal step is
  relative to a block scale, and a block of zeros has no scale.

  The step is positive, but for a float64 format at a normal leaf scale its
  square is below the float64 range (2**-2148 at shift 0) and rounds to 0.
  A block of reference zeros would then have sigma**2 = 0 and not be graded.
  The result is thus at least the smallest float64 subnormal. That value is
  above the exact u**2 / 12, so the floor can only make the bound more
  lenient.

  Args:
    ctx: The leaf context.
    fmts: The output of `_dc_formats`.
  """
  tiny = float(np.finfo(np.float64).smallest_subnormal)
  return min(
      (
          max(
              math.ldexp(1.0 / 12.0, 2 * min(minexp - ctx.shift - nmant, 0)),
              tiny,
          )
          for nmant, minexp, _, blk in fmts
          if not blk
      ),
      default=0.0,
  )


class _DcColEnergy:
  """Collects the column means of y**2 per shard, see `_dc_energy`.

  `_dc_fit` adds each tile in its pass over the leaf, so the means need no
  pass of their own. The mean of each column is over its finite elements. A
  component with fewer than two columns of positive mean gets zero means, so
  it has no energy term. A shard gets None when no component is left, when it
  has fewer than two rows with a finite element, or when the derived tiles
  split its last axis. Such a shard has no energy term.
  """

  def __init__(self, ctx: _LeafContext):
    n_shards = len(ctx.pairs)
    self._sums: list[np.ndarray | None] = [None] * n_shards
    self._counts: list[np.ndarray | None] = [None] * n_shards
    self._rows = [0] * n_shards
    self._bad = {
        s_i
        for s_i, sl in ctx.row_tiles
        if sl and sl[-1] != slice(None) and sl[-1].start != 0
    }

  def drop(self, s_i: int) -> None:
    """Gives shard s_i no energy term, for example for an empty tile."""
    self._bad.add(s_i)

  def add(self, s_i: int, y2: np.ndarray, m: np.ndarray) -> None:
    """Adds a tile of shard s_i.

    Args:
      s_i: Shard index.
      y2: y**2 for the y of `_dc_tile`.
      m: The mask of `_dc_tile`.
    """
    if s_i in self._bad:
      return
    # y is 0 where an element is not finite, so the sum is over finite ones.
    part, cnt = np.sum(y2, 0), np.sum(m, 0)
    prev, prev_cnt = self._sums[s_i], self._counts[s_i]
    if prev is not None and prev.shape != part.shape:
      self._bad.add(s_i)
      return
    self._sums[s_i] = part if prev is None else prev + part
    self._counts[s_i] = cnt if prev_cnt is None else prev_cnt + cnt
    self._rows[s_i] += int(np.count_nonzero(np.any(m, 1)))

  def means(self) -> list[np.ndarray | None]:
    """Returns the column means per shard, or None for a shard without them."""
    out: list[np.ndarray | None] = []
    for s_i, (total, cnt) in enumerate(zip(self._sums, self._counts)):
      if (
          s_i in self._bad
          or total is None
          or cnt is None
          or self._rows[s_i] < 2
      ):
        out.append(None)
        continue
      col = np.where(cnt[:, None] > 0, total / np.maximum(cnt, 1)[:, None], 0.0)
      col[:, np.sum(col > 0.0, 0) < 2] = 0.0
      out.append(col if np.any(col > 0.0) else None)
    return out


def _dc_energy(
    cols: Sequence[np.ndarray | None],
    s_i: int,
    sl: tuple[slice, ...],
    y: np.ndarray,
    m: np.ndarray,
) -> np.ndarray | None:
  """Returns the row-column energy h of a tile, or None for no energy term.

  h = r_i c_j, with c_j the mean of y**2 over the finite elements of column j
  of the shard and r_i the mean of y**2 / c_j over the finite elements of the
  tile row with c_j > 0, per component. Columns with c_j = 0 (for example zero
  padding) and elements that are not finite do not enter r_i, so they do not
  scale h. r_i is a mean of
  ratios, so a few high-energy columns do not set the row factor. For a sum
  y = sum_k a_k b_k of independent zero-mean terms, E[y**2] = sum_k a_k**2
  b_k**2 is the variance scale of its accumulated and operand rounding error.
  A rank-one product of row and column means follows it when the row and the
  column energies vary, for example per-channel outliers in one operand.

  Args:
    cols: Column means of y**2 per shard, shape (L, components), or None.
    s_i: Shard index.
    sl: Tile slices.
    y: Scaled reference tile, shape (rows, L, components).
    m: Element mask of the tile, shape (rows, L).
  """
  col = cols[s_i]
  if col is None:
    return None
  if sl:
    col = col[sl[-1]]
  valid = m[..., None] & (col > 0.0)
  with np.errstate(all="ignore"):
    ratio = np.where(valid, np.square(y) / col, 0.0)
  n_valid = np.maximum(np.sum(valid, 1, keepdims=True), 1)
  return np.sum(ratio, 1, keepdims=True) / n_valid * col


def _dc_fit_energy(
    ctx: _LeafContext,
    fmts: Sequence[_DcFormat],
    fit: tuple[float, float],
    cols: Sequence[np.ndarray | None],
) -> tuple[float, float, float]:
  """Fits E[e**2] = A v + B + C h with A >= _DC_MIN_A, B >= 0 and C >= 0.

  The fit is the reweighted least squares of `_dc_fit`, with one more term.
  It starts from the (A, B) of `_dc_fit` and C = 0. The sums are binned by
  the binary exponents of v and h. A joint fit is needed: with A and B held
  at the values of `_dc_fit`, B holds the noise of the high-energy columns,
  and is too large for the other columns.

  Args:
    ctx: The leaf context.
    fmts: The output of `_dc_formats`.
    fit: (A, B) from `_dc_fit`.
    cols: Column means of y**2 per shard, see `_dc_energy`.

  Returns:
    (A, B, C). When the constrained fit gives C = 0, C is 0 and (A, B) is the
    input.
  """
  acc: dict[int, np.ndarray] = {}
  for s_i, sl in ctx.row_tiles:
    if cols[s_i] is None:
      continue
    y, c, vu, m, _ = _dc_tile(ctx, s_i, sl, fmts)
    if y.size == 0:
      continue
    v = np.minimum(vu, np.square(y))
    h = _dc_energy(cols, s_i, sl, y, m)
    if h is None:
      continue
    mk = np.broadcast_to(m[..., None], y.shape).ravel()
    v, h, e2 = v.ravel()[mk], h.ravel()[mk], np.square(c - y).ravel()[mk]
    # Bin 0 of v (reference zeros) stays out, as in `_dc_fit`.
    keep = v > 0.0
    v, h, e2 = v[keep], h[keep], e2[keep]
    kv = np.frexp(v)[1] + _DC_BIN_OFFSET
    kh = np.where(h > 0.0, np.frexp(h)[1] + _DC_BIN_OFFSET, 0)
    keys, inv = np.unique(kv * (2 * _DC_BIN_OFFSET) + kh, return_inverse=True)
    rows = (None, v, h, v * v, v * h, h * h, e2, e2 * v, e2 * h)
    part = np.stack([np.bincount(inv, w, minlength=keys.size) for w in rows])
    for j, key in enumerate(keys.tolist()):
      if key in acc:
        acc[key] += part[:, j]
      else:
        acc[key] = part[:, j].copy()
  if not acc:
    return fit[0], fit[1], 0.0
  n_k, v_k, h_k, vv_k, vh_k, hh_k, e_k, ev_k, eh_k = np.stack(
      list(acc.values()), 1
  )
  if not np.any(hh_k > 0.0):
    return fit[0], fit[1], 0.0
  a_fit, b_fit, c_fit = fit[0], fit[1], 0.0
  for _ in range(_DC_FIT_ITERATIONS):
    s2 = (a_fit * v_k + c_fit * h_k) / n_k + b_fit
    s2_max = float(np.max(s2))
    if not s2_max > 0.0:
      break
    s2 = np.maximum(s2, _DC_FIT_REL_FLOOR * s2_max)
    w = np.square(s2_max / s2)
    g = np.array([
        [np.sum(w * vv_k), np.sum(w * v_k), np.sum(w * vh_k)],
        [np.sum(w * v_k), np.sum(w * n_k), np.sum(w * h_k)],
        [np.sum(w * vh_k), np.sum(w * h_k), np.sum(w * hh_k)],
    ])
    r = np.array([np.sum(w * ev_k), np.sum(w * e_k), np.sum(w * eh_k)])
    a_fit, b_fit, c_fit = (float(x) for x in _dc_bounded_lsq3(g, r))
  if not c_fit > 0.0:
    return fit[0], fit[1], 0.0
  return a_fit, b_fit, c_fit


def _dc_bounded_lsq3(g: np.ndarray, r: np.ndarray) -> np.ndarray:
  """Minimizes x g x - 2 r x over A >= _DC_MIN_A, B >= 0, C >= 0.

  x = (A, B, C), g and r are the weighted normal equations of
  `_dc_fit_energy`, so the objective is the weighted residual of the fit up
  to a constant. The problem is convex, so its minimum is the stationary
  point of one face of the box: the interior, a face, an edge or the vertex
  (_DC_MIN_A, 0, 0). Each of the 8 is solved with the other coordinates at
  their bounds, and the feasible point with the smallest objective is kept.
  A clamp of the unconstrained solution would keep an A that compensates
  for a negative B or C.

  Args:
    g: The 3x3 weighted normal matrix.
    r: The weighted right-hand side.

  Returns:
    The minimizer (A, B, C).
  """
  lo = np.array([_DC_MIN_A, 0.0, 0.0])
  best = lo
  best_obj = float(lo @ g @ lo - 2.0 * r @ lo)
  for n_free in (1, 2, 3):
    for free in itertools.combinations(range(3), n_free):
      f = list(free)
      fixed = [i for i in range(3) if i not in free]
      g_ff = g[np.ix_(f, f)]
      # By Sylvester's criterion, g_ff is positive definite when its diagonal
      # and the leading minors of the equilibrated matrix (unit diagonal) are
      # positive. The minors must be above a margin of 1e-12, so that a
      # near-singular face is not solved. The test runs on the equilibrated
      # matrix, so that it does not depend on the scale of g: at a small
      # scale, det(g_ff) and the product of its diagonal can underflow to 0.
      d = np.diag(g_ff)
      if not np.all(d > 0.0):
        continue
      s = np.sqrt(d)
      g_eq = g_ff / np.outer(s, s)
      if not all(
          np.linalg.det(g_eq[:k, :k]) > 1e-12 for k in range(2, n_free + 1)
      ):
        continue
      # The solve also runs on the equilibrated matrix. On g_ff, a subnormal
      # diagonal entry or a wide range of scales loses the pivot to rounding:
      # the solve then raises on a matrix that passed the test above, or
      # gives a wrong point.
      x = lo.copy()
      rhs = r[f] - g[np.ix_(f, fixed)] @ lo[fixed]
      with np.errstate(all="ignore"):
        x[f] = np.linalg.solve(g_eq, rhs / s) / s
      # A NaN or an infinity fails this test.
      if not np.all(np.isfinite(x[f]) & (x[f] >= lo[f])):
        continue
      obj = float(x @ g @ x - 2.0 * r @ x)
      if obj < best_obj:
        best, best_obj = x, obj
  return best


def _dc_fit(
    ctx: _LeafContext, fmts: Sequence[_DcFormat]
) -> tuple[float, float, float, int, list[np.ndarray | None]]:
  """Fits E[e**2] = A v + B. Returns (A, B, v_min, block count, cols).

  The fit is weighted least squares with weights 1 / (A v + B)**2, the inverse
  variance of e**2 for normal e, so that a few large elements do not set A.
  The sums are binned by the binary exponent of v, which takes few values, so
  the reweighting runs on the bins and needs one pass over the leaf. The same
  pass collects the column means of y**2 for the energy term.

  Args:
    ctx: The leaf context.
    fmts: The output of `_dc_formats`.

  Returns:
    A tuple (A, B, v_min, block count, cols). v_min is the smallest mean v of
    a fit bin. When no bin has v > 0 (for example a reference of zeros), it is
    the variance of the finest subnormal step, from `_dc_subnormal_var`. cols
    holds the column means of y**2 per shard, from `_DcColEnergy`.
  """
  # Bin 0 holds v = 0. Bin k > 0 holds frexp exponent k - _DC_BIN_OFFSET.
  sums = np.zeros((5, 2 * _DC_BIN_OFFSET))  # n, v, v**2, e**2, e**2 v
  n_blocks = 0
  cols = _DcColEnergy(ctx)
  for s_i, sl in ctx.row_tiles:
    y, c, vu, m, _ = _dc_tile(ctx, s_i, sl, fmts)
    if y.size == 0:
      cols.drop(s_i)
      continue
    y2 = np.square(y)
    cols.add(s_i, y2, m)
    v = np.minimum(vu, y2)
    mk = np.broadcast_to(m[..., None], y.shape).ravel()
    v, e2 = v.ravel()[mk], np.square(c - y).ravel()[mk]
    k = np.where(v > 0.0, np.frexp(v)[1] + _DC_BIN_OFFSET, 0)
    k = np.clip(k, 0, 2 * _DC_BIN_OFFSET - 1)
    for row, w in enumerate((None, v, v * v, e2, e2 * v)):
      sums[row] += np.bincount(k, w, minlength=2 * _DC_BIN_OFFSET)
    n_blocks += y.shape[0] * (-(-y.shape[1] // ctx.block_size))
  # Bin 0 (reference zeros) stays out of the fit. It holds no information on
  # A, and its weight 1 / B**2 grows without bound as B goes to 0. A few zero
  # elements with error then set B alone: the fit alternates between B = 0 and
  # their mean error, and grades them against themselves. `_dc_spatial` grades
  # them against the B of the other bins.
  fit_bins = sums[0] > 0.0
  fit_bins[0] = False
  if not np.any(fit_bins):
    return (
        _DC_MIN_A,
        0.0,
        _dc_subnormal_var(ctx, fmts),
        n_blocks,
        cols.means(),
    )
  n_k, v_k, v2_k, e_k, ev_k = sums[:, fit_bins]
  v_min = float(np.min(v_k / n_k))
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
    # The fit holds A >= _DC_MIN_A and B >= 0 inside the loop. A clamp of A
    # after an unconstrained fit would keep a B that already holds the v
    # term, and give sigma**2 = 2 v for one binade of rounding.
    det = sw * swv2 - swv * swv
    if det > 1e-12 * sw * swv2:
      a_fit = float((sw * swev - swv * swe) / det)
      b_fit = float((swv2 * swe - swv * swev) / det)
    if det <= 1e-12 * sw * swv2 or a_fit < _DC_MIN_A:
      # At the floor, A v is an upper bound on the rounding part. A bin with
      # mean e**2 below A v shows that the floor is above its noise, not
      # that B is small: B >= 0 already holds there. If such bins enter the
      # refit of B, their negative residuals cancel the excess of the bins
      # where the noise does not scale with v (accumulation error at outputs
      # near zero), and B falls to 0. So B is the weighted mean of the excess
      # max(0, e**2 - A v) per bin. One binade has one bin, so B is its
      # excess over v and sigma**2 stays at v, not 2 v.
      a_fit = _DC_MIN_A
      b_fit = float(np.sum(w * np.maximum(e_k - _DC_MIN_A * v_k, 0.0)) / sw)
    elif b_fit < 0.0:
      a_fit, b_fit = max(_DC_MIN_A, float(swev / swv2)), 0.0
  return a_fit, b_fit, v_min, n_blocks, cols.means()


def _dc_window_corr(
    tr: _DcTransform,
    z: float,
    eps_sq: float,
) -> tuple[np.ndarray, np.ndarray]:
  """AC correlation of candidate and reference on windows.

  A window is tested only when its centered reference sum of squares is above
  the resolution of the reference dtype. The resolution uses the squares of
  the untransformed reference values that enter each transformed value: a
  difference of two large, nearly equal values has a small result but the
  rounding error of the large values.

  Args:
    tr: One transform from `_dc_transforms`.
    z: Normal score of the per-window false-alarm rate.
    eps_sq: finfo(reference dtype).eps ** 2.

  Returns:
    (p_db, p_fl) per window and component, NaN where the window is not
    tested. p_db = -10 log10(1 - r**2) is the share of the window AC of y that
    the candidate explains. p_fl is the same for r**2 = z**2 / n, the largest
    value that a candidate independent of y reaches at the alpha level.
  """
  y, c, v, y2, m = tr.y, tr.c, tr.v, tr.y2, tr.mask
  mw = _window_chunks(m, _DC_CORR_WINDOW)
  cnt = np.sum(mw, 1).astype(np.float64)
  den = np.maximum(cnt, 1.0)[:, None, None]
  mk = mw[..., None]

  def _window_sum(t: np.ndarray) -> np.ndarray:
    return np.sum(np.where(mk, _window_chunks(t, _DC_CORR_WINDOW), 0.0), 1)

  def _dev(t: np.ndarray) -> np.ndarray:
    tw = np.where(mk, _window_chunks(t, _DC_CORR_WINDOW), 0.0)
    return np.where(mk, tw - np.sum(tw, 1, keepdims=True) / den, 0.0)

  dy, dc = _dev(y), _dev(c)
  syy = np.sum(np.square(dy), 1)
  scc = np.sum(np.square(dc), 1)
  syc = np.sum(dy * dc, 1)
  see = np.sum(np.square(dc - dy), 1)
  q = _window_sum(v)
  res = _DC_RESOLVE_EPS_FACTOR * eps_sq * _window_sum(y2)
  ok = (cnt >= _DC_MIN_SAMPLES_Z2 * z * z)[:, None]
  ok = ok & (syy > _DC_CORR_GATE * q) & (syy > res)
  # On the centered transform, a window with r < -z / sqrt(n) is tested even
  # when the see gate skips it. By the Chebyshev sum inequality,
  # cov(y, f(y)) >= 0 for any monotone elementwise f (round, cast, flush), so
  # honest rounding gives a gain g >= 0 on the window AC. Added noise that is
  # independent of y, or noise in place of a part (g = 0), then reaches
  # r < -z / sqrt(n) with probability at most Phi(-z), the false-alarm rate of
  # the window test. Each window gets one of the two tests, so the rate per
  # window stays Phi(-z). r is set to 0 for syc <= 0, so such a window fails.
  # Differences of a monotone map are not monotone in the differences, so this
  # does not hold for the other transforms: a correctly rounded cast of a
  # staircase can give r near -1 on the diff transform.
  with np.errstate(all="ignore"):
    neg = (syc < 0.0) & (
        -syc
        > z
        / np.sqrt(np.maximum(cnt, 1.0))[:, None]
        * np.sqrt(syy)
        * np.sqrt(scc)
    )
  neg &= tr.name == "centered_corr_snr_db"
  ok &= (syy >= _DC_CORR_GATE * see) | neg
  p_db, p_fl = _dc_corr_db(syy, scc, syc, cnt[:, None], z)
  return np.where(ok, p_db, np.nan), np.where(ok, p_fl, np.nan)


def _dc_corr_db(
    syy: np.ndarray, scc: np.ndarray, syc: np.ndarray, n: np.ndarray, z: float
) -> tuple[np.ndarray, np.ndarray]:
  """Correlation SNR of centered sums and its floor at the alpha level.

  Args:
    syy: Centered sum of squares of the reference part.
    scc: Centered sum of squares of the candidate part, same shape.
    syc: Centered cross sum, same shape.
    n: Sample count. Broadcasts against syy.
    z: Normal score of the false-alarm rate.

  Returns:
    (p_db, p_fl), each with the shape of syy. p_db = -10 log10(1 - r**2).
    p_fl is the same for r**2 = z**2 / n.
  """
  with np.errstate(all="ignore"):
    # r = syc / (sqrt(syy) sqrt(scc)) and not syc**2 / (syy scc): the squares
    # and the product underflow to 0 for prescaled sums near 1e-160, and a
    # window with r**2 = 0 / 0 would not be tested.
    denom = np.sqrt(syy) * np.sqrt(scc)
    r = np.where((syc > 0.0) & (denom > 0.0), syc / denom, 0.0)
    r2 = r * r
    p_db = -10.0 * np.log10(np.maximum(1.0 - np.minimum(r2, 1.0), 0.0))
    fl = np.minimum(z * z / np.maximum(n, 1.0), 1.0)
    p_fl = np.broadcast_to(-10.0 * np.log10(1.0 - fl + 1e-300), p_db.shape)
  return p_db, p_fl


def _dc_diff(t: np.ndarray) -> np.ndarray:
  """First difference along axis 1."""
  return t[:, 1:] - t[:, :-1]


def _dc_mixed_diff(t: np.ndarray) -> np.ndarray:
  """Mixed second difference along axes 0 and 1."""
  return t[1:, 1:] - t[1:, :-1] - t[:-1, 1:] + t[:-1, :-1]


def _dc_pair_sum(t: np.ndarray) -> np.ndarray:
  """Sum of neighbors along axis 1, the variance sum of `_dc_diff`."""
  return t[:, 1:] + t[:, :-1]


def _dc_corner_sum(t: np.ndarray) -> np.ndarray:
  """Sum of the 4 corners, the variance sum of `_dc_mixed_diff`."""
  return t[1:, 1:] + t[1:, :-1] + t[:-1, 1:] + t[:-1, :-1]


def _dc_transforms(
    y: np.ndarray, c: np.ndarray, v: np.ndarray, m: np.ndarray, h: int
) -> Iterator[_DcTransform]:
  """Yields the correlation transforms of a tile.

  The centered transform tests the window mean-free part. The first
  difference also removes per-row constants, and the mixed difference removes
  any separable row-plus-column part. Each runs on row-major windows. The
  first two also run on column-major windows. The rounding variance and the
  squared reference of each transformed value are the sums over the elements
  that enter it. The rows of a tile with leading axes are the rows of several
  (h, L) planes. A difference across rows uses only rows of the same plane,
  so that a per-plane separable part cancels, as in a 2-D leaf.

  Args:
    y: Reference tile, shape (rows, L, components).
    c: Candidate tile.
    v: Rounding variance tile.
    m: Element mask, shape (rows, L).
    h: Rows per plane, from `_dc_tile`. rows is a multiple of h.

  Yields:
    One `_DcTransform` per transform, with y, c, v and y2 of shape
    (n, components) and mask of shape (n,).
  """
  k_c = y.shape[-1]
  y2 = np.square(y)
  # same_plane[r] is True when rows r and r + 1 are in the same plane.
  same_plane = np.arange(1, y.shape[0]) % h != 0

  def _flat(name: str, *ts: np.ndarray) -> _DcTransform:
    ty, tc, tv, ty2, tm = (
        t.reshape((-1, k_c) if t.ndim == 3 else (-1,)) for t in ts
    )
    return _DcTransform(name, ty, tc, tv, ty2, tm)

  yield _flat("centered_corr_snr_db", y, c, v, y2, m)
  if y.shape[1] >= 2:
    yield _flat(
        "diff_corr_snr_db",
        _dc_diff(y),
        _dc_diff(c),
        _dc_pair_sum(v),
        _dc_pair_sum(y2),
        m[:, 1:] & m[:, :-1],
    )
  if y.shape[0] >= 2:
    yt, ct, vt, y2t = (np.transpose(t, (1, 0, 2)) for t in (y, c, v, y2))
    mt = m.T
    yield _flat("centered_corr_snr_db", yt, ct, vt, y2t, mt)
    yield _flat(
        "diff_corr_snr_db",
        _dc_diff(yt),
        _dc_diff(ct),
        _dc_pair_sum(vt),
        _dc_pair_sum(y2t),
        mt[:, 1:] & mt[:, :-1] & same_plane,
    )
    if y.shape[1] >= 2:
      m4 = m[1:, 1:] & m[1:, :-1] & m[:-1, 1:] & m[:-1, :-1]
      yield _flat(
          "plaquette_corr_snr_db",
          _dc_mixed_diff(y),
          _dc_mixed_diff(c),
          _dc_corner_sum(v),
          _dc_corner_sum(y2),
          m4 & same_plane[:, None],
      )


@dataclasses.dataclass(frozen=True)
class _DcBlockResult:
  """Block-noise reductions of one tile or of the leaf.

  Attributes:
    min_sqnr_db: Smallest block SQNR over the blocks with error, or None.
    max_ratio: Largest R_b = sum_b e**2 / sum_b sigma**2, or None.
    worst: (R_b, bound) at the largest R_b / bound.
  """

  min_sqnr_db: float | None = None
  max_ratio: float | None = None
  worst: tuple[float, float] = (0.0, 1.0)


@dataclasses.dataclass(frozen=True)
class _DcSpikeResult:
  """Spike reductions of one tile or of the leaf.

  Attributes:
    worst: (|e| / sigma, sigma / u) at the element with the largest |e| / sigma
      among the elements with u > 0.
    max_ulp: Largest |e| / u over the elements with u > 0.
  """

  worst: tuple[float, float] = (0.0, 1.0)
  max_ulp: float = 0.0


@dataclasses.dataclass(frozen=True)
class _DcCorrResult:
  """AC-correlation reductions of one tile or of the leaf.

  Attributes:
    min_ac_db: Smallest tested p_db, or None when no window is tested.
    worst: Per transform name, (p_db, p_fl) at the most negative p_db - p_fl,
      for the names with a negative gap.
  """

  min_ac_db: float | None = None
  worst: dict[str, tuple[float, float]] = dataclasses.field(
      default_factory=dict
  )


def _min_or_none(a: float | None, b: float | None) -> float | None:
  """Returns min(a, b), with None for a missing value."""
  if a is None:
    return b
  return a if b is None else min(a, b)


def _dc_tile_blocks(
    y: np.ndarray,
    e2: np.ndarray,
    s2: np.ndarray,
    mk: np.ndarray,
    s2_zero: float,
    bs: int,
    z_blk: float,
) -> _DcBlockResult:
  """Block-noise check of one tile.

  Args:
    y: Reference tile, shape (rows, L, components).
    e2: Squared error, same shape.
    s2: Allowed variance of each element, as in `_dc_spatial`, zero outside the
      mask, same shape.
    mk: Element mask, shape (rows, L, 1).
    s2_zero: Variance per component for a block of reference zeros that has
      sigma**2 = 0.
    bs: Block size.
    z_blk: Normal score of the per-block false-alarm rate.

  Returns:
    The reductions of the tile.
  """
  starts = np.arange(0, y.shape[1], bs)
  b_e = _dc_block_sums(e2, starts)
  b_s = _dc_block_sums(s2, starts)
  # The effective sample count uses s2 / max_b(s2), which is in [0, 1]. A sum
  # of s2**2 underflows to 0 for s2 below about 1e-154, and the count would
  # then fall to 1.
  b_mx = np.maximum.reduceat(np.max(s2, -1), starts, axis=1)
  cnt = np.diff(np.append(starts, y.shape[1]))
  mx = np.repeat(b_mx, cnt, axis=1)[..., None]
  with np.errstate(all="ignore"):
    s2n = np.where(mx > 0.0, s2 / mx, 0.0)
  b_sn = _dc_block_sums(s2n, starts)
  b_qn = _dc_block_sums(np.square(s2n), starts)
  min_db = None
  if np.any(act := b_e > 0.0):
    min_db = _min_db(_dc_block_sums(np.square(y), starts)[act], b_e[act])
  if np.any(zero := (b_s == 0.0) & act):
    # A block of reference zeros has sigma**2 = B, and B can be 0. Grade it
    # against s2_zero per component instead of skipping it. s2_zero is the
    # same for each component, so the effective count is the component count.
    b_n = _dc_block_sums(np.broadcast_to(mk, e2.shape).astype(float), starts)
    b_s = np.where(zero, b_n * s2_zero, b_s)
    b_sn = np.where(zero, b_n, b_sn)
    b_qn = np.where(zero, b_n, b_qn)
  if not np.any(pos := b_s > 0.0):
    return _DcBlockResult(min_sqnr_db=min_db)
  b_e, b_s, b_sn, b_qn = b_e[pos], b_s[pos], b_sn[pos], b_qn[pos]
  with np.errstate(all="ignore"):
    # b_s can be a count times the smallest subnormal (a zero block), so the
    # ratio can overflow to inf. That is a fail, as it should be.
    ratio = b_e / b_s
    # Effective sample count (sum s2)**2 / sum s2**2: the chi-square count
    # with the same mean and variance as the block sum of e**2 / s2.
    n_eff = np.where(b_qn > 0.0, np.square(b_sn) / b_qn, 1.0)
  bound = _DC_BLOCK_SAFETY * _dc_chi2_upper(n_eff, z_blk)
  j = int(np.argmax(ratio / bound))
  return _DcBlockResult(
      min_sqnr_db=min_db,
      max_ratio=float(np.max(ratio)),
      worst=(float(ratio[j]), float(bound[j])),
  )


def _dc_tile_spike(
    e: np.ndarray, v: np.ndarray, s2: np.ndarray
) -> _DcSpikeResult:
  """Spike check of one tile.

  An element is graded against its sigma**2 from `_dc_spatial`. An element
  with u = 0 (a reference zero) has no step of its own and is left to the
  block-noise check, which grades it against the variance of its block.

  Known gap: a spike at a reference zero that is small next to the block
  variance can pass both checks. In a bfloat16 ReLU leaf with 1e-3 noise, a
  0.02 error at a zero passes and a 0.05 error fails the block-noise check.
  A bound of B alone at the zeros would catch it, but B is not the variance
  of an error at a zero output that comes from a cancellation of large terms,
  and it gives false reds on honest masked float32 matmuls.

  Args:
    e: Error tile, shape (rows, L, components).
    v: Rounding variance tile, same shape.
    s2: Allowed variance of each element, as in `_dc_spatial`, zero outside the
      mask, same shape.

  Returns:
    The reductions of the tile.
  """
  with np.errstate(all="ignore"):
    u = np.sqrt(12.0 * v)
    # Elements with u = 0 have no step to compare with. Leave them out of
    # the argmax too, so that one of them cannot hide a spike elsewhere in
    # the tile.
    t = np.where((s2 > 0.0) & (u > 0.0), np.abs(e) / np.sqrt(s2), 0.0)
    ulp = np.where(u > 0.0, np.abs(e) / u, 0.0)
  i = int(np.argmax(t))
  t_max, k_u = float(t.flat[i]), 1.0
  if t_max > 0.0:
    k_u = float(np.sqrt(s2.flat[i]) / u.flat[i])
  return _DcSpikeResult((t_max, k_u), float(np.max(ulp, initial=0.0)))


def _dc_tile_corr(
    y: np.ndarray,
    c: np.ndarray,
    v: np.ndarray,
    m: np.ndarray,
    h: int,
    z_cor: float,
    eps_sq: float,
) -> _DcCorrResult:
  """AC-correlation checks of one tile.

  The gate variance v of a window is the noise that the window AC must be
  above before the window is tested (q in `_dc_window_corr`). `_dc_spatial`
  sets it to u**2 / 12 + b per element, with
  b = min(B, _DC_RESOLVE_EPS_FACTOR * eps**2 * sig / n). B is the noise
  floor of `_dc_fit`, not the B of `_dc_fit_energy`. sig is the sum of y**2 over
  the finite components of the
  prescaled reference (`_LeafContext.sig`), n is the finite component count,
  and eps is finfo(reference dtype).eps. Thus b is at most the resolution of
  the reference dtype at the mean reference power.

  Both terms remove false failures on windows that an honest candidate
  cannot keep. u**2 / 12 is not capped at y**2 here (the block-noise and
  spike checks keep the cap): an element below u / 2 can round to 0, so a
  window of such elements holds no AC that the candidate must keep. b covers
  values far below the mean power (for example outputs of 1e-40 next to
  outputs of 1e4). Their AC can be below the honest noise of an
  accumulation, so a candidate that flushes them to 0 is not wrong.

  The cap uses the reference only. A candidate error raises the fitted A and
  B, so a gate of A v + B or v + B grows with the error: a candidate that
  drops a low-power detail raises B by the power of that detail, and then
  the windows that would show the loss are not tested. The cap keeps b below
  a bound that no candidate can move.

  One gate of `_dc_window_corr` uses the candidate: a window whose error
  energy is above sqrt(2) times its AC energy (`_DC_CORR_GATE`) is not
  tested, unless its correlation is significantly negative on the centered
  transform. So noise that replaces a small part (a DC-dominated AC, or a
  quiet complex part) passes when the block-noise check and the global floor
  pass.

  Trade-off: a window whose AC per element is below b is not tested. A
  candidate can raise B with noise, but not b above the cap. The cap is
  about 126 dB (float32), 301 dB (float64) or 30 dB (bfloat16) below the
  mean reference power. For a bfloat16 reference, a cheat that adds noise
  near that level can thus flatten a region whose AC power is below it, and
  pass when the block-noise check and the global floor pass.

  Args:
    y: Reference tile, shape (rows, L, components).
    c: Candidate tile, same shape.
    v: Gate variance tile, same shape.
    m: Element mask, shape (rows, L).
    h: Rows per plane, from `_dc_tile`.
    z_cor: Normal score of the per-window false-alarm rate.
    eps_sq: finfo(reference dtype).eps ** 2.

  Returns:
    The reductions of the tile.
  """
  min_ac = None
  worst: dict[str, tuple[float, float]] = {}
  for tr in _dc_transforms(y, c, v, m, h):
    p_db, p_fl = _dc_window_corr(tr, z_cor, eps_sq)
    if np.all(np.isnan(p_db)):
      continue
    min_ac = _min_or_none(min_ac, float(np.nanmin(p_db)))
    _dc_keep_worst_gap(worst, tr.name, p_db, p_fl)
  return _DcCorrResult(min_ac, worst)


def _dc_keep_worst_gap(
    worst: dict[str, tuple[float, float]],
    name: str,
    p_db: Any,
    p_fl: Any,
) -> None:
  """Stores (p_db, p_fl) at the most negative gap p_db - p_fl under name.

  Args:
    worst: Per name, the (p_db, p_fl) kept so far. Updated in place.
    name: Transform name.
    p_db: Correlation SNR per window and component, NaN where not tested.
    p_fl: Its floor, same shape.
  """
  gap = np.asarray(p_db) - np.asarray(p_fl)
  j = int(np.nanargmin(gap))
  if gap.flat[j] < 0.0 and (
      name not in worst or gap.flat[j] < worst[name][0] - worst[name][1]
  ):
    worst[name] = (
        float(np.asarray(p_db).flat[j]),
        float(np.asarray(p_fl).flat[j]),
    )


def _dc_merge_blocks(a: _DcBlockResult, b: _DcBlockResult) -> _DcBlockResult:
  """Merges the block-noise reductions of two tile sets."""
  worst = a.worst
  if b.worst[0] / b.worst[1] > a.worst[0] / a.worst[1]:
    worst = b.worst
  max_ratio = a.max_ratio
  if b.max_ratio is not None:
    max_ratio = (
        b.max_ratio if max_ratio is None else max(max_ratio, b.max_ratio)
    )
  return _DcBlockResult(
      _min_or_none(a.min_sqnr_db, b.min_sqnr_db), max_ratio, worst
  )


def _dc_merge_spikes(a: _DcSpikeResult, b: _DcSpikeResult) -> _DcSpikeResult:
  """Merges the spike reductions of two tile sets."""
  worst = b.worst if b.worst[0] > a.worst[0] else a.worst
  return _DcSpikeResult(worst, max(a.max_ulp, b.max_ulp))


def _dc_merge_corr(a: _DcCorrResult, b: _DcCorrResult) -> _DcCorrResult:
  """Merges the AC-correlation reductions of two tile sets."""
  worst = dict(a.worst)
  for name, (p_db, p_fl) in b.worst.items():
    _dc_keep_worst_gap(worst, name, [p_db], [p_fl])
  return _DcCorrResult(_min_or_none(a.min_ac_db, b.min_ac_db), worst)


def _dc_spatial_report(
    blocks: _DcBlockResult,
    spikes: _DcSpikeResult,
    corr: _DcCorrResult,
    spk_bound: float,
    loc: str,
) -> tuple[dict[str, float | None], list[str]]:
  """Turns the leaf reductions of `_dc_spatial` into metrics and reasons.

  Args:
    blocks: Block-noise reductions of the leaf.
    spikes: Spike reductions of the leaf.
    corr: AC-correlation reductions of the leaf.
    spk_bound: Spike bound in sigma units.
    loc: Leaf name for messages.

  Returns:
    A tuple (metrics, reasons). metrics maps report field names to values.
  """
  metrics: dict[str, float | None] = {}
  reasons: list[str] = []
  if blocks.min_sqnr_db is not None:
    metrics["min_block_sqnr_db"] = blocks.min_sqnr_db
  if blocks.max_ratio is not None:
    # The ratio uses the fitted variance directly. A ratio to the leaf mean
    # sum(e**2) / sum(s2) is wrong when the largest elements are exact in the
    # output format (e = 0): the mean then falls far below 1 and a block with
    # one honest large element looks noisy.
    metrics["max_block_noise_ratio_db"] = 10.0 * math.log10(
        max(blocks.max_ratio, 1e-300)
    )
    r_b, bound = blocks.worst
    if r_b > bound:
      reasons.append(
          f"{loc}: max_block_noise_ratio_db {10.0 * math.log10(r_b):.2f}"
          f" > {10.0 * math.log10(bound):.2f}"
      )
  metrics["max_error_ulp"] = spikes.max_ulp
  t_max, k_u = spikes.worst
  if t_max > spk_bound:
    # Both sides in ULP units of the worst element: |e| / u and its bound.
    reasons.append(
        f"{loc}: max_error_ulp {t_max * k_u:.2f} > {spk_bound * k_u:.2f}"
    )
  metrics["min_ac_snr_db"] = corr.min_ac_db
  for name, (p_db, p_fl) in sorted(corr.worst.items()):
    reasons.append(f"{loc}: {name} {p_db:.2f} < {p_fl:.2f}")
  return metrics, reasons


@dataclasses.dataclass(frozen=True)
class _DcFit:
  """The fitted noise model of a leaf, sigma**2 = A v + B + C h.

  Attributes:
    a: A of `_dc_fit_energy`, at or above _DC_MIN_A.
    b: B of `_dc_fit_energy`.
    c: C of `_dc_fit_energy`, 0 when the leaf has no energy term.
    s2_zero: Variance per component for a block of reference zeros that has
      sigma**2 = 0, max(b, a v_min).
    cols: The column means of y**2 from `_dc_fit`, used for h as in
      `_dc_energy`.
    a_plain: A of `_dc_fit`, before `_dc_fit_energy`.
    b_plain: B of `_dc_fit`, before `_dc_fit_energy`.
    s2_zero_plain: s2_zero for (a_plain, b_plain).
  """

  a: float
  b: float
  c: float
  s2_zero: float
  cols: Sequence[np.ndarray | None]
  a_plain: float
  b_plain: float
  s2_zero_plain: float


def _dc_spatial(
    ctx: _LeafContext,
    config: SqnrConfig,
    fmts: Sequence[_DcFormat],
    fit: _DcFit,
    n_blocks: int,
    loc: str,
) -> tuple[dict[str, float | None], list[str]]:
  """Block noise, spike and AC correlation over the row tiles.

  Each tile is read once and reduced by the three checks before the next one
  is read, so the memory use is O(tile) and does not grow with the leaf.

  Args:
    ctx: The leaf context.
    config: The policy (alpha).
    fmts: The output of `_dc_formats`.
    fit: The fitted noise model. The checks grade each element with C max(h,
      y**2) in place of C h. A shard without an energy term uses the fit of
      `_dc_fit` when C > 0. The spike check adds y**2 where y**2 is below the
      resolution of the reference dtype. Only the block-noise and spike checks
      use the fit. The correlation gate uses fit.b_plain, as it does without the
      energy term.
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
      * (ctx.full_n // _DC_CORR_WINDOW + len(ctx.row_tiles))
      * (2 if ctx.is_complex else 1)
  )
  z_cor = _dc_z(alpha / (2.0 * max(n_tests, 1)))
  # The noise floor of the correlation gate, capped by the resolution of the
  # reference dtype at the mean reference power. See `_dc_tile_corr`. Below
  # b_res per element, the reference dtype does not resolve the error. The
  # gate uses the B of `_dc_fit`: the energy fit can move noise from B to
  # C h, and B = 0 then tests windows of flushed values (see the boundary
  # probe tests).
  b_res = _DC_RESOLVE_EPS_FACTOR * ctx.eps_sq * ctx.sig / max(ctx.n_finite, 1)
  b_gate = min(fit.b_plain, b_res)
  blocks, spikes, corr = _DcBlockResult(), _DcSpikeResult(), _DcCorrResult()
  for s_i, sl in ctx.row_tiles:
    y, c, vu, m, h = _dc_tile(ctx, s_i, sl, fmts)
    if y.size == 0:
      continue
    v = np.minimum(vu, np.square(y))
    e = c - y
    mk = m[..., None]
    # The allowed variance A v + B + C max(h, y**2). The fit stays on h, so
    # the y**2 term grades elements but does not set A, B or C.
    y2 = np.square(y)
    # With C = 0 the energy term adds nothing, so the energy is not computed.
    # No check depends on this condition.
    en = _dc_energy(fit.cols, s_i, sl, y, m) if fit.c > 0.0 else None
    if en is not None:
      s2, s2_zero = fit.a * v + fit.b + fit.c * np.maximum(en, y2), fit.s2_zero
    elif fit.c > 0.0:
      # The shard has no energy term, so the B of `_dc_fit_energy` lacks the
      # noise that C h holds. The fit of `_dc_fit` has it in B.
      s2, s2_zero = fit.a_plain * v + fit.b_plain, fit.s2_zero_plain
    else:
      s2, s2_zero = fit.a * v + fit.b, fit.s2_zero
    s2 = np.where(mk, s2, 0.0)
    e2 = np.square(e)
    blocks = _dc_merge_blocks(
        blocks, _dc_tile_blocks(y, e2, s2, mk, s2_zero, bs, z_blk)
    )
    # An honest kernel can flush an element with y**2 below b_res to zero, so
    # the spike bound of such an element also allows e**2 = y**2. The block
    # check does not get this term: a block of such elements next to larger
    # ones still has its own noise bound.
    s2_spk = s2 + np.where(mk & (y2 < b_res), y2, 0.0)
    spikes = _dc_merge_spikes(spikes, _dc_tile_spike(e, v, s2_spk))
    v_gate = np.where(mk, vu + b_gate, 0.0)
    corr = _dc_merge_corr(
        corr, _dc_tile_corr(y, c, v_gate, m, h, z_cor, ctx.eps_sq)
    )
  spk_bound = _DC_SPIKE_SAFETY * z_spk
  return _dc_spatial_report(blocks, spikes, corr, spk_bound, loc)


@dataclasses.dataclass(frozen=True)
class _DcDeltaRefSums:
  """Pass-1 sums of `_dc_input_delta`, per component.

  d = y - x is the update from the input view x to the reference y.

  Attributes:
    n: Finite element count.
    sum_d: sum(d).
    sum_y2: sum(y**2).
    sum_d2: sum(d**2).
  """

  n: np.ndarray
  sum_d: np.ndarray
  sum_y2: np.ndarray
  sum_d2: np.ndarray


@dataclasses.dataclass(frozen=True)
class _DcDeltaCandSums:
  """Pass-2 sums of `_dc_input_delta`, per component.

  d = y - x, e = c - y and v is the honest rounding variance.

  Attributes:
    sum_cx: sum(c - x).
    sum_e2: sum(e**2).
    sum_cx2: sum((c - x)**2).
    q: sum(v).
    q_norm: sum(v_norm), with v_norm the largest v of the components of each
      element. It equals q on a real leaf.
    sum_de: sum(d e).
    root_de2: sqrt(sum(d**2 e**2)), from `_dc_norm_merge`.
    root_d2v: sqrt(sum(d**2 v)), from `_dc_norm_merge`.
    n_sign: Count m of the elements with e**2 > 3 v and d != 0.
    n_neg: Count k of the elements in n_sign with d e < 0.
  """

  sum_cx: np.ndarray
  sum_e2: np.ndarray
  sum_cx2: np.ndarray
  q: np.ndarray
  q_norm: np.ndarray
  sum_de: np.ndarray
  root_de2: np.ndarray
  root_d2v: np.ndarray
  n_sign: np.ndarray
  n_neg: np.ndarray


@dataclasses.dataclass(frozen=True)
class _DcDeltaTests:
  """Per-leaf constants of the `_dc_input_delta` tests.

  Attributes:
    z: Normal score of the one-sided tail per test.
    log_tail: ln(1 / tail), the Chernoff bound of the sign test.
    t_gain: Hoeffding multiplier of the gain test, sqrt(2 ln(1 / tail)).
  """

  z: float
  log_tail: float
  t_gain: float


# A failed input-delta test: (gap, reason). The smallest gap is reported.
_DcDeltaFailure = tuple[float, str]


def _dc_delta_tile(
    ctx: _LeafContext, s_i: int, d_sl: tuple[slice, ...], x: Any, psi: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Reads one tile of the reference and of an input view.

  Args:
    ctx: The leaf context.
    s_i: Shard index.
    d_sl: Tile slices.
    x: The input view shard.
    psi: The view shift from `_input_view_shift`.

  Returns:
    A tuple (mask, rc, ru, xu): the finite mask of the reference, the raw
    reference, and the reference and the view scaled by 2**-psi (0 where the
    mask is False).
  """
  rc = as_real(ctx.read(ctx.pairs[s_i][0], d_sl))
  m = np.isfinite(rc)
  ru = np.where(m, np.ldexp(rc, -psi), 0.0)
  xu = np.where(m, np.ldexp(as_real(ctx.read(x, d_sl)), -psi), 0.0)
  return m, rc, ru, xu


def _dc_delta_ref_pass(
    ctx: _LeafContext, iv_shards: Sequence[Any], psi: int
) -> _DcDeltaRefSums:
  """Pass 1: reads the reference and the view only.

  A view outside the in_db range then costs no candidate read.

  Args:
    ctx: The leaf context.
    iv_shards: The input view, one array per shard.
    psi: The view shift.

  Returns:
    The pass-1 sums.
  """
  k_c = 2 if ctx.is_complex else 1
  n, sum_d, sum_y2, sum_d2 = (np.zeros(k_c) for _ in range(4))
  for s_i, d_sl in ctx.row_tiles:
    m, _, ru, xu = _dc_delta_tile(ctx, s_i, d_sl, iv_shards[s_i], psi)
    ax = tuple(range(m.ndim - 1))
    n += np.sum(m, ax)
    sum_d += np.sum(ru - xu, ax)
    sum_y2 += np.sum(np.square(ru), ax)
    sum_d2 += np.sum(np.square(ru - xu), ax)
  return _DcDeltaRefSums(n, sum_d, sum_y2, sum_d2)


def _dc_norm_merge(
    acc: tuple[np.ndarray, np.ndarray], t: np.ndarray, ax: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray]:
  """Adds sum(t**2) over ax to acc = (scale, ssq) without underflow.

  The norm is scale * sqrt(ssq), as in the BLAS dnrm2. The squares of
  prescaled products near 1e-200 underflow to 0, but t / scale does not.

  Args:
    acc: (scale, ssq) per component.
    t: The terms, with the components on the last axis.
    ax: The axes to sum over.

  Returns:
    The new (scale, ssq).
  """
  scale, ssq = acc
  t_max = np.max(np.abs(t), ax, initial=0.0)
  new = np.maximum(scale, t_max)
  safe = np.where(new > 0.0, new, 1.0)
  t_ssq = np.sum(np.square(t / safe), ax)
  return new, ssq * np.square(scale / safe) + t_ssq


def _dc_delta_cand_pass(
    ctx: _LeafContext,
    fmts: Sequence[_DcFormat],
    iv_shards: Sequence[Any],
    psi: int,
) -> _DcDeltaCandSums:
  """Pass 2: the sums of the equality, gain and sign tests.

  Args:
    ctx: The leaf context.
    fmts: The output of `_dc_formats`.
    iv_shards: The input view, one array per shard.
    psi: The view shift.

  Returns:
    The pass-2 sums.
  """
  k_c = 2 if ctx.is_complex else 1
  sums = [np.zeros(k_c) for _ in range(8)]
  # sum(d**2 e**2) and sum(d**2 v) as norms of d e and d sqrt(v): each square
  # underflows to 0 on a component near 1e-100 of a complex128 leaf near
  # 1e100, which would drop the random-sign term of the gain bound.
  n_de = n_dv = (np.zeros(k_c), np.zeros(k_c))
  for s_i, d_sl in ctx.row_tiles:
    m, rc, ru, xu = _dc_delta_tile(ctx, s_i, d_sl, iv_shards[s_i], psi)
    cc = as_real(ctx.read(ctx.pairs[s_i][1], d_sl))
    cu = np.where(m, np.ldexp(cc, -psi), 0.0)
    ys = np.where(m, np.ldexp(rc, -ctx.shift), 0.0)
    v = _dc_var(ys if ys.ndim >= 2 else ys[None], fmts, ctx)
    v = np.where(m, np.ldexp(v.reshape(m.shape), 2 * (ctx.shift - psi)), 0.0)
    ax = tuple(range(m.ndim - 1))
    d, e = ru - xu, cu - ru
    sel = (np.square(e) > 3.0 * v) & (d != 0.0)
    # Complex arithmetic mixes the components, so its error on each component
    # is bounded by a multiple of u |z| (Higham, "Accuracy and Stability of
    # Numerical Algorithms", 2002, Lemma 3.5), not by the step of that
    # component. The largest v of the components is u**2 / 12 at the binade of
    # max(|Re z|, |Im z|), which is within a factor 2 of |z|.
    v_norm = np.where(m, np.max(v, -1, keepdims=True), 0.0)
    terms = (
        cu - xu,
        np.square(cu - ru),
        np.square(cu - xu),
        v,
        v_norm,
        d * e,
        sel,
        sel & (d * e < 0.0),
    )
    for acc, term in zip(sums, terms):
      acc += np.sum(term, ax)
    n_de = _dc_norm_merge(n_de, d * e, ax)
    n_dv = _dc_norm_merge(n_dv, d * np.sqrt(v), ax)
  sum_cx, sum_e2, sum_cx2, q, q_norm, sum_de, n_sign, n_neg = sums
  return _DcDeltaCandSums(
      sum_cx=sum_cx,
      sum_e2=sum_e2,
      sum_cx2=sum_cx2,
      q=q,
      q_norm=q_norm,
      sum_de=sum_de,
      root_de2=n_de[0] * np.sqrt(n_de[1]),
      root_d2v=n_dv[0] * np.sqrt(n_dv[1]),
      n_sign=n_sign,
      n_neg=n_neg,
  )


def _dc_delta_gain_failure(
    ref: _DcDeltaRefSums,
    cand: _DcDeltaCandSums,
    tests: _DcDeltaTests,
    eps_sq: float,
    d_db: float,
    loc: str,
) -> _DcDeltaFailure | None:
  """Gain and sign tests (2 and 3 of `_dc_input_delta`).

  A component is tested only when its update is above the resolution of the
  reference dtype, as in the correlation test. An update below it can come
  from the rounding of y alone, so its gain holds no information.

  Args:
    ref: The pass-1 sums.
    cand: The pass-2 sums.
    tests: The test constants.
    eps_sq: finfo(reference dtype).eps ** 2.
    d_db: The delta SQNR of the view, for the message.
    loc: Leaf name for messages.

  Returns:
    The failure with the smallest gap, or None.
  """
  dev, d2, q = cand.sum_de, ref.sum_d2, cand.q
  # sqrt(3) sqrt(q) sqrt(d2), and not sqrt(3 q d2): the product underflows to
  # 0 for prescaled sums near 1e-160.
  g_bound = math.sqrt(3.0) * np.sqrt(q) * np.sqrt(d2) + tests.t_gain * (
      cand.root_de2 + math.sqrt(3.0) * cand.root_d2v
  )
  m_s = cand.n_sign
  p_s = np.minimum(cand.n_neg, m_s - cand.n_neg) / np.maximum(m_s, 1.0)
  with np.errstate(all="ignore"):
    # m D(p) with 0 ln 0 = 0.
    kl = m_s * (
        np.where(p_s > 0.0, p_s * np.log(2.0 * p_s), 0.0)
        + (1.0 - p_s) * np.log(2.0 * (1.0 - p_s))
    )
    g_gap = np.minimum(
        20.0 * np.log10(g_bound / np.abs(dev)),
        10.0 * np.log10(tests.log_tail / kl),
    )
  out = d2 > _DC_RESOLVE_EPS_FACTOR * eps_sq * ref.sum_y2
  out &= np.abs(dev) >= _DC_DELTA_GAIN_TOL * d2
  out &= (np.abs(dev) > g_bound) | (kl > tests.log_tail)
  bad: _DcDeltaFailure | None = None
  for k in np.flatnonzero(out):
    if bad is None or g_gap[k] < bad[0]:
      bad = (
          float(g_gap[k]),
          (
              f"{loc}: min_delta_sqnr_db {d_db:.2f} (gain"
              f" {1.0 + dev[k] / d2[k]:.3g} of candidate - input on"
              f" reference - input, outside 1 +- {_DC_DELTA_GAIN_TOL:.2g})"
          ),
      )
  return bad


def _dc_delta_corr_failure(
    ctx: _LeafContext,
    iv_shards: Sequence[Any],
    psi: int,
    ref: _DcDeltaRefSums,
    cand: _DcDeltaCandSums,
    tests: _DcDeltaTests,
    loc: str,
) -> _DcDeltaFailure | None:
  """Pass 3: the correlation test (4 of `_dc_input_delta`).

  The centered sums need a third pass with the global means. A one-pass
  merge of per-tile means makes the cross sum a difference of nearly equal
  terms when c - x is close to a constant, so r**2 is no longer bounded by 1
  and the `x + const` candidate could pass.

  Args:
    ctx: The leaf context.
    iv_shards: The input view, one array per shard.
    psi: The view shift.
    ref: The pass-1 sums.
    cand: The pass-2 sums.
    tests: The test constants.
    loc: Leaf name for messages.

  Returns:
    The failure with the smallest gap, or None.
  """
  n = ref.n
  mean_y = ref.sum_d / np.maximum(n, 1.0)
  mean_c = cand.sum_cx / np.maximum(n, 1.0)
  k_c = 2 if ctx.is_complex else 1
  syy, scc, syc = (np.zeros(k_c) for _ in range(3))
  for s_i, d_sl in ctx.row_tiles:
    m, _, ru, xu = _dc_delta_tile(ctx, s_i, d_sl, iv_shards[s_i], psi)
    cu = np.ldexp(as_real(ctx.read(ctx.pairs[s_i][1], d_sl)), -psi)
    ax = tuple(range(m.ndim - 1))
    dy = np.where(m, ru - xu - mean_y, 0.0)
    dc = np.where(m, cu - xu - mean_c, 0.0)
    syy += np.sum(np.square(dy), ax)
    scc += np.sum(np.square(dc), ax)
    syc += np.sum(dy * dc, ax)
  ok = n >= _DC_MIN_SAMPLES_Z2 * tests.z * tests.z
  # A rounding of y adds e_q with |e_q| <= u / 2, so its centered energy is at
  # most sum(e_q**2) <= 3 q. By Cauchy-Schwarz, r**2(d + e_q, d) >= 1 - 3 q /
  # syy. So the test cannot fail an honest rounding where syy (1 - z**2 / n) >
  # 3 q. Below that, c - x = 0 can be the rounding of y. On a complex leaf, q
  # is q_norm: the error of complex arithmetic on a quiet component scales
  # with |z|. The gate uses the reference only, so a candidate error cannot
  # turn the test off.
  fl = np.minimum(tests.z * tests.z / np.maximum(n, 1.0), 1.0)
  ok &= (syy * (1.0 - fl) > 3.0 * cand.q_norm) & (
      syy > _DC_RESOLVE_EPS_FACTOR * ctx.eps_sq * ref.sum_y2
  )
  p_db, p_fl = _dc_corr_db(syy, scc, syc, n, tests.z)
  bad: _DcDeltaFailure | None = None
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
  return bad


def _worse_failure(
    a: _DcDeltaFailure | None, b: _DcDeltaFailure | None
) -> _DcDeltaFailure | None:
  """Returns the failure with the smaller gap. a wins a tie."""
  if b is None or (a is not None and a[0] <= b[0]):
    return a
  return b


def _dc_input_delta(
    ctx: _LeafContext,
    config: SqnrConfig,
    fmts: Sequence[_DcFormat],
    loc: str,
) -> tuple[float | None, list[str]]:
  """Input-delta check. Returns (min_delta_sqnr_db, reasons).

  For an input view x with in_db = S(x, y) in [_DC_DELTA_MIN_IN_DB, max_in),
  where max_in is the reference dtype limit, the update d = y - x must show in
  the candidate. Four tests run per component:

  1. Equality: c equals x on every element while sum(d**2) is above 3 q, the
     largest value at which every |d| can be below u / 2.
  2. Gain: with e = c - y, the gain of c - x on d is g = 1 + sum(d e) /
     sum(d**2). The test fails when |1 - g| >= _DC_DELTA_GAIN_TOL and
     |sum(d e)| is above the honest bound. Honest e is the sum of a rounding
     part with |e_q| <= u / 2, so e_q**2 <= 3 v pointwise, and a part e_r
     whose sign does not depend on d. The bound uses 3 v and not the fitted
     A: an error that scales with |y| (for example a wrong multiplicative
     update) goes into the fitted A and would widen the bound around itself.
     Cauchy-Schwarz bounds sum(d e_q) by sqrt(3 q sum(d**2)), with
     q = sum(v). Given the magnitudes, sum(d e_r) is a sum of terms with
     random signs, so by Hoeffding P(|sum(d e_r)| > t sqrt(sum(d**2
     e_r**2))) <= 2 exp(-t**2 / 2) at every n. By Minkowski,
     sqrt(sum(d**2 e_r**2)) <= sqrt(sum(d**2 e**2)) + sqrt(3 sum(d**2 v)).
     The test is uncentered, so it also covers a constant d and small n.
  3. Sign count: where |e| > u / 2 = sqrt(3 v), |e_q| < |e|, so e has the
     sign of e_r, and d e is positive or negative with probability 1/2. Of
     the m such elements with d != 0, k have d e < 0. By the Chernoff bound,
     P(k or fewer, or m - k or more) <= 2 exp(-m D(min(k, m - k) / m)), with
     D(p) = p ln(2 p) + (1 - p) ln(2 (1 - p)) the relative entropy to 1/2.
     The bound is exact when all m signs agree, so the test resolves a
     one-sided error from m >= log2(1 / tail) elements, where Hoeffding
     needs about 2 ln(1 / tail) effective samples. It fails only when the
     gain is also outside 1 +- _DC_DELTA_GAIN_TOL, because an honest
     candidate with a small gain error (g = 0.8) has mostly d e < 0.
  4. Correlation: honest e that is independent of d gives r**2(c - x, d) >=
     k / (1 + k) once the gates below hold. The test fails when r**2 is below
     z**2 / n. It runs only when the centered energy of d is above
     3 q / (1 - z**2 / n), where a rounding alone cannot fail it.

  Tests 2-4 each use alpha / _DC_DELTA_TESTS per view and component.

  Args:
    ctx: The leaf context.
    config: The policy (alpha).
    fmts: The output of `_dc_formats`.
    loc: Leaf name for messages.

  Returns:
    A tuple (min_delta_sqnr_db, reasons). min_delta_sqnr_db is None when
    no view qualifies.
  """
  max_in = -10.0 * math.log10(max(_DC_RESOLVE_EPS_FACTOR * ctx.eps_sq, 1e-300))
  k_c = 2 if ctx.is_complex else 1
  # One side of each two-sided test, per view, component and test.
  tail = config.alpha / (
      2.0 * _DC_DELTA_TESTS * max(len(ctx.input_views) * k_c, 1)
  )
  # Hoeffding and Chernoff: the two-sided gain and sign tests fail at 2 tail.
  log_tail = math.log(1.0 / min(max(tail, 1e-300), 0.5))
  tests = _DcDeltaTests(
      z=_dc_z(tail), log_tail=log_tail, t_gain=math.sqrt(2.0 * log_tail)
  )
  min_ddb: float | None = None
  bad: _DcDeltaFailure | None = None
  for iv_shards in ctx.input_views:
    psi = _input_view_shift(ctx, iv_shards)
    if psi is None:
      continue
    ref = _dc_delta_ref_pass(ctx, iv_shards, psi)
    rsig, dsig = ref.sum_y2.sum(), ref.sum_d2.sum()
    in_db = math.inf if dsig == 0.0 else _db(rsig, dsig)
    if not _DC_DELTA_MIN_IN_DB <= in_db < max_in:
      continue
    cand = _dc_delta_cand_pass(ctx, fmts, iv_shards, psi)
    d_db = _db(dsig, cand.sum_e2.sum())
    min_ddb = d_db if min_ddb is None else min(min_ddb, d_db)
    # The rounding of y moves each element by at most u / 2, so c = x can be
    # honest while every |d| <= u / 2, that is, up to sum(d**2) = 3 q. Above
    # it, at least one |d| is above u / 2.
    if cand.sum_cx2.sum() == 0.0 and dsig > 3.0 * cand.q.sum():
      bad = (
          -math.inf,
          (
              f"{loc}: min_delta_sqnr_db {d_db:.2f} < inf"
              " (candidate equals an input view)"
          ),
      )
      continue
    bad = _worse_failure(
        bad, _dc_delta_gain_failure(ref, cand, tests, ctx.eps_sq, d_db, loc)
    )
    bad = _worse_failure(
        bad,
        _dc_delta_corr_failure(ctx, iv_shards, psi, ref, cand, tests, loc),
    )
  return min_ddb, [bad[1]] if bad is not None else []


def _derived_checks(
    ctx: _LeafContext, config: SqnrConfig, loc: str
) -> _CheckOutcome:
  """Checks with bounds from config.alpha and the honest-noise model.

  1. Block noise: R_b = sum_b e**2 / sum_b sigma**2 with sigma**2 = A v + B
     + C max(h, y**2), A, B and C as in `_dc_fit_energy` (`_dc_spatial`).
     Fails when R_b is above the
     1 - alpha / n_blocks quantile of chi2_n / n times _DC_BLOCK_SAFETY, with
     n = (sum_b sigma**2)**2 / sum_b sigma**4, the effective sample count of
     the block.
  2. Spike: max |e| / sigma above z(1 - alpha / (2 N)) * _DC_SPIKE_SAFETY.
     sigma**2 also allows e**2 = y**2 where y**2 is below the resolution of
     the reference dtype.
  3. AC correlation (centered, first difference, mixed difference) on windows
     of _DC_CORR_WINDOW elements, as in `_dc_window_corr`.
  4. Input delta: equality, gain, sign count and correlation, as in
     `_dc_input_delta`.

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
  a_fit, b_fit, v_min, n_blocks, cols = _dc_fit(ctx, fmts)
  # A block of reference zeros has sigma**2 = B, which is 0 when the other
  # elements show no noise beyond their rounding. Its error is then graded
  # against the smallest honest rounding variance of the leaf.
  a_plain, b_plain, c_fit = a_fit, b_fit, 0.0
  if any(col is not None for col in cols):
    a_fit, b_fit, c_fit = _dc_fit_energy(ctx, fmts, (a_fit, b_fit), cols)
  fit = _DcFit(
      a=a_fit,
      b=b_fit,
      c=c_fit,
      s2_zero=max(b_fit, a_fit * v_min),
      cols=cols,
      a_plain=a_plain,
      b_plain=b_plain,
      s2_zero_plain=max(b_plain, a_plain * v_min),
  )
  metrics, reasons = _dc_spatial(ctx, config, fmts, fit, n_blocks, loc)
  if ctx.n_elements > 1 and ctx.centered_sig > 0.0:
    metrics["centered_snr_db"] = _db(ctx.centered_sig, ctx.centered_err)
  min_ddb, d_reasons = _dc_input_delta(ctx, config, fmts, loc)
  metrics["min_delta_sqnr_db"] = min_ddb
  reasons.extend(d_reasons)
  return _CheckOutcome(metrics=metrics, reasons=tuple(reasons))


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


def _oracle_non_finite_mismatches(
    pairs: Sequence[tuple[Any, Any]],
    read: Callable[[Any, tuple[slice, ...]], np.ndarray],
    tiles: Sequence[tuple[int, tuple[slice, ...]]],
    oracle_pairs: Sequence[Any],
) -> int:
  """Counts non-finite reference components that the oracle does not match.

  The caller has checked that the reference and the candidate have the same
  NaN, +Inf and -Inf components. At each such component the oracle must have
  the same kind of value. Otherwise both outputs left the finite range (for
  example, by overflow together) where the oracle stays finite, and the
  oracle sums, which skip these components, cannot see it.

  Args:
    pairs: (reference, candidate) shards.
    read: The array accessor.
    tiles: Disjoint tiles over the shards.
    oracle_pairs: Oracle shards with the shard shapes.

  Returns:
    The number of non-finite reference components where the oracle has a
    different kind of value.
  """
  n_bad = 0
  for s_i, d_sl in tiles:
    rc = as_real(read(pairs[s_i][0], d_sl))
    if np.all(np.isfinite(rc)):
      continue
    oc = as_real(read(oracle_pairs[s_i], d_sl))
    bad = (
        (np.isnan(rc) & ~np.isnan(oc))
        | (np.isposinf(rc) & ~np.isposinf(oc))
        | (np.isneginf(rc) & ~np.isneginf(oc))
    )
    n_bad += int(np.count_nonzero(bad))
  return n_bad


def _oracle_sums(
    pairs: Sequence[tuple[Any, Any]],
    read: Callable[[Any, tuple[slice, ...]], np.ndarray],
    tiles: Sequence[tuple[int, tuple[slice, ...]]],
    data_max: float,
    oracle_pairs: Sequence[Any],
) -> tuple[float, float, int, bool]:
  """Returns (oracle_sqnr_db, reference_oracle_sqnr_db, n_bad, oracle_zero).

  The sums run tile by tile over components where the oracle, the reference
  and the candidate are all finite. n_bad counts components where the
  reference and the candidate are finite but the oracle is not. Such
  components drop out of the sums, so the caller must fail the leaf when
  n_bad > 0. oracle_zero is True when every finite oracle component is zero.
  Then the oracle has no signal and the two dB values have no meaning.

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
  return _db(o_sig, e_c), _db(o_sig, e_r), n_bad, o_max == 0.0


def _core_metrics(ctx: _LeafContext) -> dict[str, float | None]:
  """sqnr_db, best_fit_scale, si_snr_db, cosine_similarity from the sums.

  The scale is s = <c, y> / <y, y>, and the scale-invariant residual is
  |c - s y|**2. It has two equal forms: |e|**2 - <e, y>**2 / |y|**2 and
  |c|**2 - <c, y>**2 / |y|**2. Each form subtracts terms of its own size, so
  the residual uses the error form when |e| < |c| (high SQNR) and the candidate
  form otherwise (for example a candidate that is near zero). <c, y> and |c|**2
  are sums of their own, not values rebuilt from |y|**2, |e|**2 and <e, y>,
  which would cancel when |c| is much smaller than |y|.

  Args:
    ctx: The leaf context.

  Returns:
    sqnr_db, best_fit_scale, si_snr_db and cosine_similarity by name.
  """
  sig, err, sum_ey = ctx.sig, ctx.err, ctx.sum_ey
  cand_sq, sum_cy = ctx.cand_sq, ctx.sum_cy
  if sig <= 0.0:
    # The reference underflows after the prescale (the candidate is larger
    # by more than the float64 range). Only sqnr_db (-inf) is defined.
    return dict(sqnr_db=ctx.sqnr_db)
  scale = sum_cy / sig
  if err < cand_sq:
    resid = max(err - sum_ey * sum_ey / sig, 0.0)
  else:
    resid = max(cand_sq - sum_cy * sum_cy / sig, 0.0)
  cos: float | None = None
  # Two square roots: sig * cand_sq can round to 0 when one energy is
  # subnormal, while the product of the roots stays positive.
  denom = math.sqrt(sig) * math.sqrt(cand_sq)
  if denom > 0.0:
    cos = max(-1.0, min(1.0, sum_cy / denom))
  return dict(
      sqnr_db=ctx.sqnr_db,
      best_fit_scale=scale,
      # s**2 |y|**2 = s <c, y>: this form cannot overflow when sig is small.
      si_snr_db=_db(scale * sum_cy, resid),
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


@dataclasses.dataclass(frozen=True, eq=False)
class _LeafAudit:
  """Inputs and shared report fields of one `audit_leaf` call.

  Attributes:
    pairs: The (reference, candidate) shards.
    read: The wrapped accessor.
    oracle_pairs: Oracle shards, or None.
    config: The policy.
    precision: The PrecisionSpec used for the threshold.
    loc: Leaf name for messages.
    fields: Report fields that every report of the leaf shares.
    warnings: Warnings that every report of the leaf carries.
  """

  pairs: Sequence[tuple[Any, Any]]
  read: Callable[[Any, tuple[slice, ...]], np.ndarray]
  oracle_pairs: Sequence[Any] | None
  config: SqnrConfig
  precision: PrecisionSpec
  loc: str
  fields: dict[str, Any]
  warnings: tuple[str, ...]

  @property
  def threshold_db(self) -> float:
    """The resolved SQNR floor in dB."""
    return self.fields["threshold_db"]

  def rule_report(
      self, status: str, passed: bool, msg: str = ""
  ) -> SqnrLeafReport:
    """Report of a leaf rule. `msg` is the failure reason."""
    reasons = () if passed or not msg else (f"{self.loc}: {msg}",)
    return SqnrLeafReport(
        status=status,
        passed=passed,
        failed_reasons=reasons,
        warnings=self.warnings,
        **self.fields,
    )

  def metric_report(
      self, status: str, metrics: dict[str, Any], reasons: Sequence[str]
  ) -> SqnrLeafReport:
    """Report with metrics. It passes when `reasons` is empty."""
    metric_fields: dict[str, Any] = {
        k: _finite_or_none(v) for k, v in metrics.items()
    }
    return SqnrLeafReport(
        status=status,
        passed=not reasons,
        failed_reasons=tuple(reasons),
        warnings=self.warnings,
        **self.fields,
        **metric_fields,
    )


def _new_leaf_audit(
    pairs: Sequence[tuple[Any, Any]],
    read: Callable[[Any, tuple[slice, ...]], np.ndarray],
    oracle_pairs: Sequence[Any] | None,
    config: SqnrConfig,
    precision: PrecisionSpec | None,
    ref_dtype: np.dtype | None,
    path: str,
    discrete_leaf: bool,
) -> _LeafAudit:
  """Resolves the threshold and the shared report fields of a leaf.

  The integer-format warning is left out of a discrete leaf
  (`_discrete_leaf`): the discrete rule (or a shape mismatch, which fails
  first) does not use the floor.

  Args:
    pairs: The (reference, candidate) shards of the leaf.
    read: The callback that reads one tile of a shard.
    oracle_pairs: The oracle shards, or None.
    config: The SQNR policy.
    precision: The rounding sources, or None for the reference dtype only.
    ref_dtype: The reference dtype, or None if it is not known.
    path: The leaf path in the output tree, or "" for a single output.
    discrete_leaf: Whether the leaf goes to the discrete rule.

  Returns:
    The leaf audit with the threshold, the report fields and the warnings.
  """
  loc = path or "output"
  ref_name = str(ref_dtype) if ref_dtype is not None else "unknown"
  if precision is None:
    # An object or str reference adds no rounding source. The dtype rule
    # fails the leaf.
    precision = PrecisionSpec(
        output_dtype=(
            ref_name
            if ref_dtype is not None and _numeric_dtype(np.dtype(ref_dtype))
            else None
        )
    )
  th_db, th_src = resolve_threshold(config, precision)
  warnings: tuple[str, ...] = ()
  int_ops = _integer_formats(precision)
  if config.min_sqnr_db == AUTO and int_ops and not discrete_leaf:
    warnings = (
        (
            f"{loc}: integer format ({', '.join(int_ops)}): AUTO does not model"
            f" integer quantization and uses {DEFAULT_MIN_SQNR_DB:g} dB; pass"
            " an explicit min_sqnr_db"
        ),
    )
  return _LeafAudit(
      pairs=pairs,
      read=read,
      oracle_pairs=oracle_pairs,
      config=config,
      precision=precision,
      loc=loc,
      fields=dict(
          path=path,
          dtype=ref_name,
          threshold_db=th_db,
          threshold_source=th_src,
          gate_math=config.gate_math,
      ),
      warnings=warnings,
  )


def _shape_report(
    audit: _LeafAudit,
    ref_meta: Sequence[tuple[tuple[int, ...], np.dtype]],
    cand_meta: Sequence[tuple[tuple[int, ...], np.dtype]],
    oracle_meta: Sequence[tuple[tuple[int, ...], np.dtype]] | None,
) -> SqnrLeafReport | None:
  """Shape rule over every shard. Returns None when the shapes match."""
  if any(rm[0] != cm[0] for rm, cm in zip(ref_meta, cand_meta)):
    return audit.rule_report(STATUS_SHAPE_MISMATCH, False, "shape mismatch")
  if oracle_meta is not None and (
      len(oracle_meta) != len(ref_meta)
      or any(om[0] != rm[0] for om, rm in zip(oracle_meta, ref_meta))
  ):
    return audit.rule_report(
        STATUS_SHAPE_MISMATCH, False, "oracle shape mismatch"
    )
  return None


def _dtype_report(
    audit: _LeafAudit,
    ref_meta: Sequence[tuple[tuple[int, ...], np.dtype]],
    cand_meta: Sequence[tuple[tuple[int, ...], np.dtype]],
    oracle_meta: Sequence[tuple[tuple[int, ...], np.dtype]] | None,
    d_tiles: Sequence[tuple[int, tuple[slice, ...]]],
) -> SqnrLeafReport | None:
  """Dtype and discrete rules over every shard.

  All reference shards must share one dtype, and all candidate shards must
  share one dtype. A discrete leaf needs equal dtypes and equal values. A
  float leaf needs float or complex dtypes of the same complexness, and an
  oracle needs float or complex dtypes of the leaf complexness.

  Args:
    audit: The leaf audit.
    ref_meta: (shape, dtype) per reference shard.
    cand_meta: (shape, dtype) per candidate shard.
    oracle_meta: (shape, dtype) per oracle shard, or None.
    d_tiles: Disjoint tiles of the leaf.

  Returns:
    The report of a discrete leaf or of a dtype mismatch. None for a float
    leaf that can go on.
  """
  dtype_mismatch = audit.rule_report(
      STATUS_DTYPE_MISMATCH, False, "dtype mismatch"
  )
  ref_dts = {m[1] for m in ref_meta}
  cand_dts = {m[1] for m in cand_meta}
  if _discrete_leaf(ref_dts | cand_dts):
    if len(ref_dts | cand_dts) != 1:
      return dtype_mismatch
    read, pairs = audit.read, audit.pairs
    if all(
        np.array_equal(read(pairs[s_i][1], sl), read(pairs[s_i][0], sl))
        for s_i, sl in d_tiles
    ):
      return audit.rule_report(STATUS_DISCRETE_EXACT, True)
    return audit.rule_report(
        STATUS_DISCRETE_MISMATCH, False, "discrete mismatch"
    )
  if len(ref_dts) != 1 or len(cand_dts) != 1:
    return dtype_mismatch
  (ref_dt,), (cand_dt,) = ref_dts, cand_dts
  if (
      _eps_sq(ref_dt) is None
      or _eps_sq(cand_dt) is None
      or _is_complex(ref_dt) != _is_complex(cand_dt)
  ):
    return dtype_mismatch
  if oracle_meta is not None and any(
      _eps_sq(om[1]) is None or _is_complex(om[1]) != _is_complex(ref_dt)
      for om in oracle_meta
  ):
    return audit.rule_report(
        STATUS_DTYPE_MISMATCH, False, "oracle dtype mismatch"
    )
  return None


def _oracle_check(
    audit: _LeafAudit,
    tiles: Sequence[tuple[int, tuple[slice, ...]]],
    data_max: float,
    *,
    within_atol: bool = False,
) -> tuple[dict[str, float], list[str]]:
  """Runs the oracle rule. Returns (metrics, failed reasons).

  Args:
    audit: The leaf audit.
    tiles: Disjoint tiles of the leaf.
    data_max: max(|reference|, |candidate|) over finite components.
    within_atol: True when every |cand| is <= atol on a zero reference. If the
      oracle is zero too, the dB rule has no signal to grade. Then this flag
      decides the leaf instead of the -inf oracle_sqnr_db.

  Returns:
    The oracle metrics and the failure reasons.
  """
  if audit.oracle_pairs is None:
    return {}, []
  o_c, o_r, n_bad, oracle_zero = _oracle_sums(
      audit.pairs, audit.read, tiles, data_max, audit.oracle_pairs
  )
  o_metrics = dict(
      oracle_sqnr_db=o_c,
      reference_oracle_sqnr_db=o_r,
      oracle_delta_db=o_c - o_r,
  )
  if n_bad:
    # The oracle cannot grade components that it does not cover, so the
    # leaf fails closed instead of skipping them.
    return o_metrics, [
        f"{audit.loc}: oracle_non_finite: {n_bad} components where the"
        " reference and the candidate are finite"
    ]
  if within_atol and oracle_zero:
    # Zero reference, zero oracle and |cand| <= atol: the allclose rule
    # already graded the leaf. A nonzero oracle still goes to the dB rule,
    # where the zero reference fails with reference_below_floor.
    return o_metrics, []
  return o_metrics, _oracle_reasons(
      o_c, o_r, audit.threshold_db, audit.config, audit.loc
  )


def _exact_report(
    audit: _LeafAudit,
    status: str,
    tiles: Sequence[tuple[int, tuple[slice, ...]]],
    data_max: float,
    **metrics: float,
) -> SqnrLeafReport:
  """Report for a candidate equal to the reference. The oracle still runs."""
  o_metrics, o_reasons = _oracle_check(audit, tiles, data_max)
  return audit.metric_report(status, {**o_metrics, **metrics}, o_reasons)


def _zero_reference_report(
    audit: _LeafAudit,
    scan: _LeafScan,
    d_tiles: Sequence[tuple[int, tuple[slice, ...]]],
    is_complex: bool,
    atol: float | None,
) -> SqnrLeafReport:
  """Allclose rule for a leaf whose finite reference values are all zero.

  Zero signal has no dB ratio, so every |cand| must be <= atol, where |.| is
  the modulus on complex leaves.

  Args:
    audit: The leaf audit.
    scan: The first-pass result. scan.ref_max is 0.
    d_tiles: Disjoint tiles of the leaf.
    is_complex: Leaf complexness.
    atol: Absolute tolerance, or None.

  Returns:
    The report.
  """
  if scan.cand_max == 0.0:
    return _exact_report(audit, STATUS_EXACT, d_tiles, 0.0)
  cand_abs = scan.cand_max
  if is_complex:
    cand_abs = 0.0
    for s_i, sl in d_tiles:
      cand_c = as_real(audit.read(audit.pairs[s_i][1], sl))
      cand_c = np.where(np.isfinite(cand_c), cand_c, 0.0)
      modulus = np.hypot(cand_c[..., 0], cand_c[..., 1])
      cand_abs = max(cand_abs, float(np.max(modulus, initial=0.0)))
  within_atol = (
      atol is not None
      and math.isfinite(atol)
      and atol >= 0.0
      and cand_abs <= atol
  )
  if not within_atol:
    return audit.rule_report(
        STATUS_ZERO_SIGNAL, False, "nonzero error on zero reference"
    )
  # The oracle still runs. Only a zero oracle lets atol decide the leaf.
  o_metrics, o_reasons = _oracle_check(
      audit, d_tiles, scan.cand_max, within_atol=True
  )
  return audit.metric_report(STATUS_ZERO_SIGNAL, o_metrics, o_reasons)


def _measured_report(audit: _LeafAudit, ctx: _LeafContext) -> SqnrLeafReport:
  """Core metrics, the global or oracle rule, and the strict checks."""
  config, loc = audit.config, audit.loc
  metrics = _core_metrics(ctx)
  reasons: list[str] = []
  if audit.oracle_pairs is None:
    if ctx.sqnr_db < audit.threshold_db:
      reasons.append(
          f"{loc}: sqnr_db {ctx.sqnr_db:.2f} < {audit.threshold_db:.2f}"
      )
  else:
    o_metrics, o_reasons = _oracle_check(
        audit, ctx.disjoint_tiles, max(ctx.ref_max, ctx.cand_max)
    )
    metrics.update(o_metrics)
    reasons.extend(o_reasons)
  if config.strict:
    if config.gate_math == GATE_MATH_SIX_GATE_V1:
      outcome = _six_gate_v1_checks(ctx, config, loc)
    else:
      outcome = _derived_checks(ctx, config, loc)
    metrics.update(outcome.metrics)
    reasons.extend(outcome.reasons)
  return audit.metric_report(STATUS_OK, metrics, reasons)


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
  match required), float dtypes, empty leaf, non-finite mask match, oracle
  non-finite match (with an oracle, each NaN or Inf of the outputs needs the
  same value in the oracle), no finite elements, zero reference (allclose rule
  with `atol`), exact match. Then the global SQNR rule (or the oracle rule)
  and, in strict mode, the check family of `config.gate_math` run.

  Args:
    pairs: One (reference, candidate) pair per shard. Shapes must match per
      pair. Shards of one leaf are disjoint and share the leaf dtypes.
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
  read = _reader(read)
  ref_meta = [_shape_dtype(r, read) for r, _ in pairs]
  cand_meta = [_shape_dtype(c, read) for _, c in pairs]
  audit = _new_leaf_audit(
      pairs,
      read,
      oracle_pairs,
      config,
      precision,
      ref_meta[0][1] if pairs else None,
      path,
      discrete_leaf=_discrete_leaf([m[1] for m in ref_meta + cand_meta]),
  )
  if not pairs:
    return audit.rule_report(STATUS_NO_FINITE_ELEMENTS, True)
  oracle_meta = (
      None
      if oracle_pairs is None
      else [_shape_dtype(o, read) for o in oracle_pairs]
  )
  if rep := _shape_report(audit, ref_meta, cand_meta, oracle_meta):
    return rep
  shapes = [m[0] for m in ref_meta]
  d_tiles, _ = _leaf_tiles(shapes, config.block_size)
  if rep := _dtype_report(audit, ref_meta, cand_meta, oracle_meta, d_tiles):
    return rep
  if sum(math.prod(s) for s in shapes) == 0:
    return audit.rule_report(STATUS_NO_FINITE_ELEMENTS, True)
  scan = _scan_leaf(pairs, read, d_tiles)
  if scan is None:
    return audit.rule_report(
        STATUS_NON_FINITE_MISMATCH, False, "non finite mask"
    )
  if oracle_pairs is not None and scan.n_non_finite:
    # Runs before the no-finite pass: outputs that are all NaN or Inf where
    # the oracle is finite must not pass.
    if n_bad := _oracle_non_finite_mismatches(
        pairs, read, d_tiles, oracle_pairs
    ):
      return audit.rule_report(
          STATUS_NON_FINITE_MISMATCH,
          False,
          f"oracle_non_finite_mismatch: {n_bad} components where the"
          " reference and the candidate are non-finite and the oracle is"
          " finite or of another kind",
      )
  if scan.n_finite == 0:
    return audit.rule_report(STATUS_NO_FINITE_ELEMENTS, True)
  if scan.ref_max == 0.0:
    return _zero_reference_report(
        audit, scan, d_tiles, _is_complex(ref_meta[0][1]), atol
    )
  ctx = _build_context(
      pairs,
      read,
      input_views,
      audit.threshold_db,
      audit.precision,
      block_size=config.block_size,
      scan=scan,
  )
  if ctx.max_abs_err == 0.0:
    exact_metrics: dict[str, float] = {}
    if config.strict and config.gate_math == GATE_MATH_SIX_GATE_V1:
      exact_metrics["peak_error_rms_ratio"] = 0.0
    return _exact_report(
        audit,
        STATUS_EXACT,
        d_tiles,
        max(scan.ref_max, scan.cand_max),
        **exact_metrics,
    )
  return _measured_report(audit, ctx)


def merge_reports(
    reports: Iterable[SqnrLeafReport | Mapping[str, Any]],
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
