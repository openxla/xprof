"""Tests for the SQNR metrics and anti-cheat checks."""

import ast
import gc
import inspect
import math
import tracemalloc
import types
from typing import Any, TypeVar

from absl.testing import absltest
from absl.testing import parameterized
import ml_dtypes
import numpy as np

from xprof.xparity import sqnr

_T = TypeVar("_T")


def _nn(value: _T | None) -> _T:
  """Returns value after it checks that value is not None."""
  assert value is not None
  return value


def _gelu_np(x: np.ndarray) -> np.ndarray:
  return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def _cast(x: Any, dtype: Any, back: Any = None) -> np.ndarray:
  """Rounds x to dtype, then optionally casts it to `back`."""
  out = np.asarray(x).astype(dtype)
  return out if back is None else out.astype(back)


def _bf16(x: Any, back: Any = None) -> np.ndarray:
  return _cast(x, ml_dtypes.bfloat16, back)


def _bf16_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
  """bfloat16 operands, float32 accumulation, bfloat16 output, as float32."""
  prod = _bf16(a, np.float32) @ _bf16(b, np.float32)
  return _bf16(prod, np.float32)


def _input_views(
    inputs: Any, shape: tuple[int, ...], is_complex: bool
) -> list[list[np.ndarray]]:
  """Builds single-shard input views for a leaf of `shape`.

  This mirrors what a caller does: it keeps float or discrete inputs with the
  leaf size and complexness, and reshapes or swaps inputs of another shape.
  `audit_leaf` adds the transpose of a same-shape square view on its own.

  Args:
    inputs: Candidate inputs. Inputs that are not arrays are skipped.
    shape: Leaf shape.
    is_complex: Leaf complexness.

  Returns:
    One list of single-shard views per kept input.
  """
  n_el = math.prod(shape)
  views = []
  for raw in inputs:
    if isinstance(raw, (str, np.dtype)) or raw is None:
      continue
    raw = np.asarray(raw)
    if raw.size != n_el or np.iscomplexobj(raw) != is_complex:
      continue
    if raw.dtype == np.bool_ or (raw.dtype.kind in "iu" and n_el <= 1):
      continue
    if raw.shape == shape:
      views.append([raw])
      continue
    sw = np.swapaxes(raw, -2, -1) if raw.ndim >= 2 else None
    if sw is not None and sw.shape == shape:
      views.append([sw])
    rs = raw.reshape(shape)
    if sw is None or sw.shape != shape or not np.array_equal(rs, sw):
      views.append([rs])
  return views


def _gate(
    cand: Any,
    ref: Any,
    *,
    min_sqnr_db: float = 45.0,
    inputs: Any = (),
    atol: float | None = None,
) -> sqnr.SqnrLeafReport:
  """Runs the six_gate_v1 preset on one single-shard leaf."""
  r, c = np.asarray(ref), np.asarray(cand)
  return sqnr.audit_leaf(
      [(r, c)],
      config=sqnr.SqnrConfig.six_gate_v1(min_sqnr_db=min_sqnr_db),
      input_views=_input_views(inputs, r.shape, np.iscomplexobj(r)),
      atol=atol,
  )


class ModuleTest(absltest.TestCase):

  def test_imports_only_stdlib_numpy_and_ml_dtypes(self):
    imported = set()
    for node in ast.walk(ast.parse(inspect.getsource(sqnr))):
      if isinstance(node, ast.Import):
        imported.update(a.name.split(".")[0] for a in node.names)
      elif isinstance(node, ast.ImportFrom):
        imported.add((node.module or "").split(".")[0])
    self.assertLessEqual(
        imported,
        {
            "dataclasses",
            "itertools",
            "math",
            "statistics",
            "types",
            "typing",
            "ml_dtypes",
            "numpy",
        },
    )

  def test_block_scaled_formats_is_read_only(self):
    self.assertEqual(sqnr.BLOCK_SCALED_FORMATS["mxfp4"], ("float4_e2m1fn", 32))
    self.assertEqual(sqnr.BLOCK_SCALED_FORMATS["nvfp4"], ("float4_e2m1fn", 16))
    self.assertIsInstance(sqnr.BLOCK_SCALED_FORMATS, types.MappingProxyType)


class ConfigTest(parameterized.TestCase):

  @parameterized.parameters(float("nan"), float("inf"), 0.0, -5.0, True, "x")
  def test_rejects_bad_threshold(self, bad):
    with self.assertRaises(ValueError):
      sqnr.SqnrConfig(min_sqnr_db=bad)
    with self.assertRaises(ValueError):
      sqnr.SqnrConfig.six_gate_v1(min_sqnr_db=bad)

  @parameterized.parameters("nan", "0", "-1", "abc")
  def test_from_options_rejects_bad_threshold(self, bad):
    with self.assertRaises(ValueError):
      sqnr.SqnrConfig.from_options(min_sqnr_db=bad)

  def test_from_options(self):
    cfg = sqnr.SqnrConfig.from_options(min_sqnr_db="40", preset="six_gate_v1")
    self.assertEqual(
        (cfg.min_sqnr_db, cfg.gate_math), (40.0, sqnr.GATE_MATH_SIX_GATE_V1)
    )
    self.assertEqual(
        sqnr.SqnrConfig.from_options(preset="six_gate_v1").min_sqnr_db,
        sqnr.DEFAULT_MIN_SQNR_DB,
    )
    self.assertEqual(
        sqnr.SqnrConfig.from_options(min_sqnr_db="auto").min_sqnr_db, sqnr.AUTO
    )
    with self.assertRaises(ValueError):
      sqnr.SqnrConfig.from_options(preset="other")

  def test_accepts_numpy_float_threshold(self):
    cfg = sqnr.SqnrConfig.six_gate_v1(min_sqnr_db=np.float32(45.0))
    self.assertEqual(sqnr.resolve_threshold(cfg), (45.0, "explicit"))

  def test_rejects_bad_alpha_block_and_gate_math(self):
    bad_kwargs: list[dict[str, Any]] = [
        {"alpha": 0.0},
        {"block_size": 1},
        {"gate_math": "x"},
    ]
    for kw in bad_kwargs:
      with self.assertRaises(ValueError, msg=str(kw)):
        sqnr.SqnrConfig(**kw)


class ExpectedSqnrTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("bf16_operands", ("bfloat16", "bfloat16"), "float32", 52.6),
      ("e4m3_operands", ("float8_e4m3fn", "float8_e4m3fn"), None, 28.5),
      ("bf16_output", (), "bfloat16", 55.6),
      ("fp16_output", (), "float16", 73.7),
      ("e5m2_operand", ("float8_e5m2",), None, 25.5),
      ("fp4_unscaled", ("float4_e2m1fn",), None, 16.3),
      ("mxfp4", ("mxfp4",), "float32", 18.8),
      ("nvfp4", ("nvfp4",), None, 20.4),
      ("mxfp8_e4m3", ("mxfp8_e4m3",), None, 30.7),
      ("mxfp8_e5m2", (), "mxfp8_e5m2", 25.4),
  )
  def test_format_table(self, operands, output, expected_db):
    e_db, derivation = _nn(
        sqnr.expected_sqnr_db(
            sqnr.PrecisionSpec(operand_dtypes=operands, output_dtype=output)
        )
    )
    self.assertAlmostEqual(e_db, expected_db, delta=0.05)
    for name in operands:
      self.assertIn(name, derivation)
    self.assertTrue(derivation.endswith(f"={e_db:.1f}dB"), derivation)

  def test_no_narrow_format_gives_none_and_default_threshold(self):
    spec = sqnr.PrecisionSpec(
        operand_dtypes=("float32", "not_a_dtype", "int8"),
        output_dtype="float64",
    )
    self.assertIsNone(sqnr.expected_sqnr_db(spec))
    self.assertEqual(
        sqnr.resolve_threshold(sqnr.SqnrConfig(), spec),
        (sqnr.DEFAULT_MIN_SQNR_DB, sqnr.THRESHOLD_SOURCE_DEFAULT),
    )

  def test_accumulator_term_falls_3_db_per_doubling(self):
    def acc_db(k):
      return sqnr.expected_sqnr_db(
          sqnr.PrecisionSpec(accumulator_dtype="bfloat16", reduction_length=k)
      )

    db_4k, derivation = acc_db(4096)
    self.assertAlmostEqual(acc_db(8192)[0], db_4k - 3.01, delta=0.01)
    self.assertAlmostEqual(
        db_4k, 55.6 - 10.0 * math.log10(4096 / 2), delta=0.05
    )
    self.assertIn("acc_bfloat16(K=4096,K0=2)", derivation)
    # A float32 accumulator adds no term.
    self.assertIsNone(
        sqnr.expected_sqnr_db(
            sqnr.PrecisionSpec(accumulator_dtype="float32", reduction_length=8)
        )
    )

  def test_auto_threshold(self):
    th, src = sqnr.resolve_threshold(
        sqnr.SqnrConfig(),
        sqnr.PrecisionSpec(operand_dtypes=("bfloat16", "bfloat16")),
    )
    self.assertAlmostEqual(th, 52.6 - 6.0, delta=0.05)
    self.assertEqual(src, "auto:bfloat16:55.6+bfloat16:55.6=52.6dB-6dB")

  def test_auto_threshold_from_reference_dtype(self):
    ref = np.random.default_rng(0).standard_normal(256).astype(np.float16)
    rep = sqnr.compute_sqnr(ref, ref)
    self.assertAlmostEqual(_nn(rep.threshold_db), 73.7 - 6.0, delta=0.05)
    self.assertStartsWith(rep.threshold_source, "auto:float16:")

  @parameterized.named_parameters(("int8", "int8"), ("int4", "int4"))
  def test_auto_with_integer_operand_falls_back_and_warns(self, int_name):
    spec = sqnr.PrecisionSpec(
        operand_dtypes=(int_name, "bfloat16"), output_dtype="bfloat16"
    )
    self.assertEqual(
        sqnr.resolve_threshold(sqnr.SqnrConfig(), spec),
        (sqnr.DEFAULT_MIN_SQNR_DB, sqnr.THRESHOLD_SOURCE_DEFAULT),
    )
    rng = np.random.default_rng(1)
    ref = rng.standard_normal((32, 64))
    cand = ref * (1.0 + 1e-4 * rng.standard_normal(ref.shape))
    rep = sqnr.compute_sqnr(cand, ref, precision=spec)
    self.assertTrue(rep.passed, rep)
    self.assertEqual(rep.threshold_db, sqnr.DEFAULT_MIN_SQNR_DB)
    self.assertLen(rep.warnings, 1)
    self.assertIn(f"integer format ({int_name})", rep.warnings[0])
    self.assertIn("explicit min_sqnr_db", rep.warnings[0])
    # The warning also reaches the early leaf rules.
    self.assertLen(sqnr.compute_sqnr(ref, ref, precision=spec).warnings, 1)
    # An explicit floor removes the warning.
    rep_explicit = sqnr.compute_sqnr(
        cand, ref, precision=spec, config=sqnr.SqnrConfig(min_sqnr_db=30.0)
    )
    self.assertEqual(rep_explicit.warnings, ())
    self.assertEqual(rep_explicit.threshold_source, "explicit")


class PrecisionFromDtypesTest(parameterized.TestCase):

  def test_keeps_narrow_inputs_and_uses_candidate_output(self):
    spec = sqnr.precision_from_dtypes(
        np.float32,
        ml_dtypes.bfloat16,
        input_dtypes=(ml_dtypes.bfloat16, np.float32, "float8_e4m3fn"),
        accumulator_dtype="bfloat16",
        reduction_length=512,
    )
    self.assertEqual(
        spec,
        sqnr.PrecisionSpec(
            operand_dtypes=("bfloat16", "float8_e4m3fn"),
            output_dtype="bfloat16",
            reduction_length=512,
            accumulator_dtype="bfloat16",
        ),
    )

  def test_wide_dtypes_give_default_floor(self):
    spec = sqnr.precision_from_dtypes(
        "float64", "float32", input_dtypes=("float32", np.float64)
    )
    self.assertEqual(
        spec, sqnr.PrecisionSpec(operand_dtypes=(), output_dtype="float32")
    )
    self.assertEqual(
        sqnr.resolve_threshold(sqnr.SqnrConfig(), spec),
        (sqnr.DEFAULT_MIN_SQNR_DB, sqnr.THRESHOLD_SOURCE_DEFAULT),
    )

  def test_floor_follows_candidate_not_reference(self):
    # The reference dtype is not a rounding source of the candidate.
    th_f32_ref, _ = sqnr.resolve_threshold(
        sqnr.SqnrConfig(), sqnr.precision_from_dtypes("float32", "bfloat16")
    )
    th_bf16_ref, src = sqnr.resolve_threshold(
        sqnr.SqnrConfig(), sqnr.precision_from_dtypes("bfloat16", "bfloat16")
    )
    self.assertEqual(th_f32_ref, th_bf16_ref)
    self.assertEqual(src, "auto:bfloat16:55.6=55.6dB-6dB")

  def test_block_scaled_and_integer_inputs(self):
    spec = sqnr.precision_from_dtypes(
        "float32", "float32", input_dtypes=("MXFP4", "int8", ml_dtypes.int4)
    )
    self.assertEqual(spec.operand_dtypes, ("mxfp4", "int8", "int4"))
    # Integer operands make AUTO fall back to the default floor.
    self.assertEqual(
        sqnr.resolve_threshold(sqnr.SqnrConfig(), spec)[0],
        sqnr.DEFAULT_MIN_SQNR_DB,
    )
    # Booleans are masks, not rounding sources.
    self.assertEqual(
        sqnr.precision_from_dtypes(
            "float32", "float32", input_dtypes=(np.bool_,)
        ).operand_dtypes,
        (),
    )

  def test_rejects_unknown_dtype(self):
    with self.assertRaises(TypeError):
      sqnr.precision_from_dtypes("float32", "not_a_dtype")


class LeafRulesTest(parameterized.TestCase):

  @parameterized.parameters(
      ml_dtypes.int4, ml_dtypes.uint4, ml_dtypes.int2, ml_dtypes.uint2
  )
  def test_narrow_integers_are_discrete(self, dt):
    ref = np.array([0, 1, 1, 0], dtype=dt)
    rep = _gate(ref.copy(), ref)
    self.assertEqual(rep.status, sqnr.STATUS_DISCRETE_EXACT)
    self.assertTrue(rep.passed)
    bad = np.array([0, 1, 0, 0], dtype=dt)
    rep_bad = _gate(bad, ref)
    self.assertEqual(rep_bad.status, sqnr.STATUS_DISCRETE_MISMATCH)
    self.assertFalse(rep_bad.passed)
    self.assertEqual(rep_bad.failed_reasons, ("output: discrete mismatch",))

  def test_discrete_dtype_mismatch(self):
    rep = _gate(
        np.array([0, 1, 2], ml_dtypes.uint4),
        np.array([0, 1, 2], ml_dtypes.int4),
    )
    self.assertEqual(rep.status, sqnr.STATUS_DTYPE_MISMATCH)
    self.assertFalse(rep.passed)
    rep_f = _gate(np.array([1.0, 2.0], np.float32), np.array([1, 2], np.int32))
    self.assertEqual(rep_f.status, sqnr.STATUS_DTYPE_MISMATCH)

  def test_int32_and_bool(self):
    self.assertEqual(
        _gate(np.array([1, 3], np.int32), np.array([1, 2], np.int32)).status,
        sqnr.STATUS_DISCRETE_MISMATCH,
    )
    self.assertEqual(
        _gate(np.array([1, 2], np.int32), np.array([1, 2], np.int32)).status,
        sqnr.STATUS_DISCRETE_EXACT,
    )
    self.assertEqual(
        _gate(np.array([True, False]), np.array([True, False])).status,
        sqnr.STATUS_DISCRETE_EXACT,
    )

  def test_complex_and_real_mix_is_dtype_mismatch(self):
    rep = _gate(np.ones(4, np.complex64), np.ones(4, np.float32))
    self.assertEqual(rep.status, sqnr.STATUS_DTYPE_MISMATCH)

  def test_shape_mismatch(self):
    rep = _gate(np.ones((4, 4)), np.ones((4, 5)))
    self.assertEqual(rep.status, sqnr.STATUS_SHAPE_MISMATCH)
    self.assertFalse(rep.passed)
    rep_sh = sqnr.audit_leaf(
        [(np.ones((2, 4)), np.ones((2, 4))), (np.ones((2, 4)), np.ones(8))]
    )
    self.assertEqual(rep_sh.status, sqnr.STATUS_SHAPE_MISMATCH)

  def test_empty_and_all_nan_have_no_finite_elements(self):
    self.assertEqual(
        _gate(np.zeros((0, 3)), np.zeros((0, 3))).status,
        sqnr.STATUS_NO_FINITE_ELEMENTS,
    )
    rep = _gate(np.full(4, np.nan), np.full(4, np.nan))
    self.assertEqual(rep.status, sqnr.STATUS_NO_FINITE_ELEMENTS)
    self.assertTrue(rep.passed)
    self.assertEqual(sqnr.audit_leaf([]).status, sqnr.STATUS_NO_FINITE_ELEMENTS)

  @parameterized.named_parameters(
      ("nan_moves", [np.nan, 1.0, 2.0], [1.0, np.nan, 2.0]),
      ("inf_sign", [np.inf, 1.0, 2.0], [-np.inf, 1.0, 2.0]),
      ("inf_vs_nan", [np.inf, 1.0, 2.0], [np.nan, 1.0, 2.0]),
      ("finite_vs_nan", [0.5, 1.0, 2.0], [np.nan, 1.0, 2.0]),
  )
  def test_non_finite_mask_mismatch(self, cand, ref):
    rep = _gate(np.array(cand), np.array(ref))
    self.assertEqual(rep.status, sqnr.STATUS_NON_FINITE_MISMATCH)
    self.assertFalse(rep.passed)
    self.assertEqual(rep.failed_reasons, ("output: non finite mask",))

  def test_matching_non_finite_values_are_ignored(self):
    ref = np.array([np.nan, np.inf, -np.inf, 1.0, 2.0])
    rep = _gate(ref.copy(), ref)
    self.assertEqual(rep.status, sqnr.STATUS_EXACT)
    self.assertTrue(rep.passed)

  def test_zero_reference_uses_atol(self):
    z_ref = np.zeros((8, 8), np.float32)
    z_tiny = np.full((8, 8), 1e-9, np.float32)
    self.assertFalse(_gate(z_tiny, z_ref).passed)
    rep_z = _gate(z_tiny, z_ref, atol=1e-5)
    self.assertTrue(rep_z.passed)
    self.assertEqual(rep_z.status, sqnr.STATUS_ZERO_SIGNAL)
    self.assertEqual(rep_z.failed_reasons, ())
    rep_zb = _gate(np.full((8, 8), 1e-3, np.float32), z_ref, atol=1e-5)
    self.assertFalse(rep_zb.passed)
    self.assertEqual(rep_zb.status, sqnr.STATUS_ZERO_SIGNAL)
    self.assertEqual(
        rep_zb.failed_reasons, ("output: nonzero error on zero reference",)
    )
    self.assertFalse(_gate(z_tiny, z_ref, atol=float("nan")).passed)
    self.assertFalse(_gate(z_tiny, z_ref, atol=-1.0).passed)
    self.assertEqual(_gate(z_ref, z_ref).status, sqnr.STATUS_EXACT)

  def test_zero_reference_complex_uses_modulus(self):
    zc_ref = np.zeros(8, np.complex64)
    self.assertTrue(
        _gate(np.full(8, 0.9e-5 + 0j, np.complex64), zc_ref, atol=1e-5).passed
    )
    self.assertFalse(
        _gate(
            np.full(8, 0.8e-5 + 0.8e-5j, np.complex64), zc_ref, atol=1e-5
        ).passed
    )

  def test_exact_match(self):
    ref = np.random.default_rng(0).standard_normal((16, 16))
    rep = _gate(ref.copy(), ref)
    self.assertEqual(rep.status, sqnr.STATUS_EXACT)
    self.assertEqual(rep.peak_error_rms_ratio, 0.0)
    self.assertIsNone(rep.sqnr_db)
    self.assertIsNone(sqnr.compute_sqnr(ref, ref).peak_error_rms_ratio)

  def test_constant_leaf_has_no_centered_snr(self):
    rep = _gate(np.ones((4, 4), np.float32), np.ones((4, 4), np.float32))
    self.assertTrue(rep.passed)
    self.assertIsNone(rep.centered_snr_db)

  def test_junk_input_views_are_skipped(self):
    rng = np.random.default_rng(3)
    ref = rng.standard_normal((16, 16))
    views = [
        ["cfg"],
        [None],
        [np.int32(3)],
        [np.ones((4, 4))],
        [ref, ref],
        [np.ones((16, 16), np.complex64)],
    ]
    rep = sqnr.audit_leaf(
        [(ref, ref * 1.00001)],
        config=sqnr.SqnrConfig.six_gate_v1(),
        input_views=views,
    )
    self.assertTrue(rep.passed)
    self.assertIsNone(rep.min_delta_sqnr_db)


class CoreMetricsTest(parameterized.TestCase):

  def test_metrics_match_direct_formulas(self):
    rng = np.random.default_rng(5)
    ref = rng.standard_normal((64, 48))
    cand = 0.9 * ref + 1e-3 * rng.standard_normal(ref.shape)
    rep = sqnr.compute_sqnr(cand, ref, config=sqnr.SqnrConfig(min_sqnr_db=10))
    e = cand - ref
    scale = np.dot(cand.ravel(), ref.ravel()) / np.dot(ref.ravel(), ref.ravel())
    resid = cand - scale * ref
    self.assertAlmostEqual(
        rep.sqnr_db, 10 * np.log10(np.sum(ref**2) / np.sum(e**2)), places=9
    )
    self.assertAlmostEqual(_nn(rep.best_fit_scale), scale, places=12)
    self.assertAlmostEqual(
        rep.si_snr_db,
        10 * np.log10(np.sum((scale * ref) ** 2) / np.sum(resid**2)),
        places=6,
    )
    self.assertAlmostEqual(
        _nn(rep.cosine_similarity),
        np.dot(cand.ravel(), ref.ravel())
        / np.linalg.norm(cand)
        / np.linalg.norm(ref),
        places=12,
    )
    self.assertGreater(_nn(rep.si_snr_db), 55.0)
    self.assertLess(_nn(rep.sqnr_db), 21.0)
    self.assertTrue(rep.passed)
    self.assertEqual(rep.failed_reasons, ())

  def test_complex_metrics(self):
    rng = np.random.default_rng(6)
    ref = (rng.standard_normal(256) + 1j * rng.standard_normal(256)).astype(
        np.complex64
    )
    cand = (ref + 1e-3 * rng.standard_normal(256)).astype(np.complex64)
    rep = sqnr.compute_sqnr(cand, ref)
    r64, c64 = ref.astype(np.complex128), cand.astype(np.complex128)
    expect = 10 * np.log10(
        np.sum(np.abs(r64) ** 2) / np.sum(np.abs(c64 - r64) ** 2)
    )
    self.assertAlmostEqual(rep.sqnr_db, expect, places=6)

  def test_report_only_mode_has_no_check_metrics(self):
    rng = np.random.default_rng(8)
    ref = rng.standard_normal((32, 32))
    rep = sqnr.compute_sqnr(np.zeros_like(ref) + 1e-9, ref)
    self.assertFalse(rep.passed)
    self.assertIsNone(rep.min_block_sqnr_db)
    self.assertLen(rep.failed_reasons, 1)
    self.assertStartsWith(rep.failed_reasons[0], "output: sqnr_db")

  def test_to_dict_and_merge(self):
    rng = np.random.default_rng(9)
    ref = rng.standard_normal((32, 32))
    good = sqnr.compute_sqnr(_bf16(ref, np.float64), ref, path="a")
    bad = sqnr.compute_sqnr(ref * 0.5, ref, path="b")
    exact = sqnr.compute_sqnr(ref, ref, path="c")
    d = bad.to_dict()
    self.assertEqual(d["sqnr_status"], sqnr.STATUS_OK)
    self.assertNotIn("status", d)
    self.assertIsInstance(d["failed_reasons"], list)
    self.assertIsInstance(d["warnings"], list)
    merged = sqnr.merge_reports([good, bad.to_dict(), exact])
    self.assertFalse(merged["passed"])
    self.assertEqual(merged["sqnr_status"], sqnr.STATUS_OK)
    self.assertAlmostEqual(merged["sqnr_db"], bad.sqnr_db)
    self.assertEqual(merged["failed_reasons"], list(bad.failed_reasons))
    self.assertEqual(merged["warnings"], [])
    self.assertEqual(
        sqnr.merge_reports([exact, {"sqnr_status": "dtype_mismatch"}])[
            "sqnr_status"
        ],
        sqnr.STATUS_DTYPE_MISMATCH,
    )
    self.assertEqual(sqnr.merge_reports([])["sqnr_status"], sqnr.STATUS_EXACT)


class OracleTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.truth = np.random.default_rng(12).standard_normal((64, 128))

  def _noisy(self, snr_db: float, seed: int) -> np.ndarray:
    """Returns truth plus independent noise at snr_db against truth."""
    rng = np.random.default_rng(seed)
    rms = float(np.sqrt(np.mean(self.truth**2)))
    std = rms * 10.0 ** (-snr_db / 20.0)
    return self.truth + std * rng.standard_normal(self.truth.shape)

  def test_reference_above_floor_removes_false_failure(self):
    ref = _bf16(self.truth, np.float64)  # About 55.6 dB.
    cand = self._noisy(55.6, seed=20)  # Independent error of the same size.
    cfg = sqnr.SqnrConfig(min_sqnr_db=54.0)
    # Against the lossy reference alone, both errors add (about 52.6 dB).
    self.assertFalse(sqnr.compute_sqnr(cand, ref, config=cfg).passed)
    rep = sqnr.compute_sqnr(cand, ref, oracle=self.truth, config=cfg)
    self.assertTrue(rep.passed, rep)
    self.assertAlmostEqual(_nn(rep.reference_oracle_sqnr_db), 55.6, delta=0.5)
    self.assertGreater(_nn(rep.oracle_sqnr_db), 54.0)
    self.assertEqual(rep.warnings, ())

  def test_reference_below_floor_fails_closed(self):
    # A good candidate (float16, about 73.7 dB) with a reference that is
    # below the floor fails: the reference cannot vouch for it.
    ref = _bf16(self.truth, np.float64)  # About 55.6 dB.
    cand = _cast(self.truth, np.float16, np.float64)
    cfg = sqnr.SqnrConfig(min_sqnr_db=60.0)
    rep = sqnr.compute_sqnr(cand, ref, oracle=self.truth, config=cfg)
    self.assertFalse(rep.passed)
    self.assertGreater(_nn(rep.oracle_delta_db), 15.0)
    self.assertLen(rep.failed_reasons, 1)
    self.assertStartsWith(
        rep.failed_reasons[0], "output: reference_below_floor:"
    )

  def test_lossy_reference_does_not_vouch_for_bad_candidate(self):
    # Candidate 17.5 dB, reference 20 dB, floor 45 dB. The candidate is
    # within 3 dB of the reference, but the reference is below the floor.
    ref = self._noisy(20.0, seed=21)
    cand = self._noisy(17.5, seed=22)
    rep = sqnr.audit_leaf(
        [(ref, cand)],
        config=sqnr.SqnrConfig(min_sqnr_db=45.0),
        oracle_pairs=[self.truth],
    )
    self.assertFalse(rep.passed)
    self.assertAlmostEqual(_nn(rep.reference_oracle_sqnr_db), 20.0, delta=0.2)
    self.assertAlmostEqual(_nn(rep.oracle_sqnr_db), 17.5, delta=0.2)
    self.assertTrue(
        any("reference_below_floor" in r for r in rep.failed_reasons),
        rep.failed_reasons,
    )

  def test_honest_bf16_candidate_against_f32_reference_passes(self):
    ref = _cast(self.truth, np.float32)
    cand = _bf16(self.truth)
    spec = sqnr.precision_from_dtypes(ref.dtype, cand.dtype)
    rep = sqnr.audit_leaf(
        [(ref, cand)], precision=spec, oracle_pairs=[self.truth]
    )
    self.assertTrue(rep.passed, rep)
    self.assertAlmostEqual(_nn(rep.threshold_db), 55.6 - 6.0, delta=0.05)
    self.assertGreater(_nn(rep.reference_oracle_sqnr_db), 140.0)
    self.assertAlmostEqual(_nn(rep.oracle_sqnr_db), 55.6, delta=0.5)
    self.assertEqual(rep.warnings, ())

  def test_candidate_worse_than_reference_and_floor_fails(self):
    ref = _cast(self.truth, np.float16, np.float64)
    cand = _cast(self.truth, ml_dtypes.float8_e4m3fn, np.float64)
    rep = sqnr.compute_sqnr(
        cand, ref, oracle=self.truth, config=sqnr.SqnrConfig(min_sqnr_db=45.0)
    )
    self.assertFalse(rep.passed)
    self.assertEqual(rep.warnings, ())
    self.assertLess(_nn(rep.oracle_delta_db), -3.0)
    self.assertEqual(
        rep.failed_reasons,
        (f"output: oracle_sqnr_db {rep.oracle_sqnr_db:.2f} < 45.00",),
    )

  def test_shared_error_fails_when_reference_below_floor(self):
    ref = self.truth.copy()
    ref[:8] = 0.0  # A bug that the reference shares with the candidate.
    rng = np.random.default_rng(13)
    cand = ref * (1.0 + 1e-6 * rng.standard_normal(ref.shape))
    rep = sqnr.compute_sqnr(
        cand, ref, oracle=self.truth, config=sqnr.SqnrConfig(min_sqnr_db=45.0)
    )
    self.assertFalse(rep.passed)
    self.assertLess(_nn(rep.reference_oracle_sqnr_db), 45.0)
    self.assertLess(_nn(rep.oracle_sqnr_db), 45.0)
    self.assertEqual(
        rep.failed_reasons,
        (
            (
                "output: reference_below_floor: reference_oracle_sqnr_db"
                f" {rep.reference_oracle_sqnr_db:.2f} < 45.00"
            ),
        ),
    )
    merged = sqnr.merge_reports([rep])
    self.assertFalse(merged["passed"])
    self.assertEqual(merged["failed_reasons"], list(rep.failed_reasons))

  def test_non_finite_oracle_fails_closed(self):
    # A 20 dB candidate with a 45 dB floor. The oracle has no finite element,
    # so it cannot grade the leaf. The leaf must fail, not pass with +inf dB.
    ref = self.truth.copy()
    cand = self._noisy(20.0, seed=23)
    for strict in (False, True):
      rep = sqnr.compute_sqnr(
          cand,
          ref,
          oracle=np.full_like(self.truth, np.nan),
          config=sqnr.SqnrConfig(min_sqnr_db=45.0, strict=strict),
      )
      self.assertFalse(rep.passed, strict)
      self.assertTrue(
          any("oracle_non_finite" in r for r in rep.failed_reasons),
          rep.failed_reasons,
      )

  def test_partly_non_finite_oracle_fails_closed(self):
    oracle = self.truth.copy()
    oracle[3, 7] = np.inf
    ref = _cast(self.truth, np.float32)
    rep = sqnr.compute_sqnr(_bf16(self.truth), ref, oracle=oracle)
    self.assertFalse(rep.passed)
    self.assertEqual(
        rep.failed_reasons,
        (
            (
                "output: oracle_non_finite: 1 components where the reference"
                " and the candidate are finite"
            ),
        ),
    )

  def test_bit_exact_candidate_still_checks_the_oracle(self):
    # The reference is 6 dB from the oracle. A candidate equal to it gets no
    # pass from the exact path: the reference cannot vouch for it.
    ref = 0.5 * self.truth
    rep = sqnr.compute_sqnr(
        ref.copy(), ref, oracle=self.truth, config=sqnr.SqnrConfig()
    )
    self.assertEqual(rep.status, sqnr.STATUS_EXACT)
    self.assertFalse(rep.passed)
    self.assertAlmostEqual(_nn(rep.reference_oracle_sqnr_db), 6.02, delta=0.01)
    self.assertStartsWith(
        rep.failed_reasons[0], "output: reference_below_floor:"
    )

  def test_bit_exact_candidate_with_a_good_reference_passes(self):
    ref = _cast(self.truth, np.float32)
    rep = sqnr.compute_sqnr(ref.copy(), ref, oracle=self.truth)
    self.assertEqual(rep.status, sqnr.STATUS_EXACT)
    self.assertTrue(rep.passed, rep)
    self.assertGreater(_nn(rep.reference_oracle_sqnr_db), 140.0)

  def test_all_zero_outputs_still_check_the_oracle(self):
    zeros = np.zeros_like(self.truth)
    rep = sqnr.compute_sqnr(zeros, zeros, oracle=self.truth)
    self.assertEqual(rep.status, sqnr.STATUS_EXACT)
    self.assertFalse(rep.passed)
    self.assertStartsWith(
        rep.failed_reasons[0], "output: reference_below_floor:"
    )
    self.assertTrue(
        sqnr.compute_sqnr(zeros, zeros, oracle=np.zeros_like(zeros)).passed
    )

  def test_oracle_shape_mismatch(self):
    rep = sqnr.compute_sqnr(self.truth, self.truth * 1.1, oracle=np.ones(3))
    self.assertEqual(rep.status, sqnr.STATUS_SHAPE_MISMATCH)


class TilingTest(absltest.TestCase):

  def test_custom_read_sees_one_tile_at_a_time(self):
    shape = (1024, 512)
    rng = np.random.default_rng(14)
    ref = rng.standard_normal(shape).astype(np.float32)
    cand = _bf16(ref, np.float32)
    x = rng.standard_normal(shape).astype(np.float32)
    sizes = []

    def read(a, sl):
      out = np.asarray(a[sl] if sl else a)
      sizes.append(out.size)
      return out

    rep = sqnr.audit_leaf(
        [(ref, cand)],
        config=sqnr.SqnrConfig.six_gate_v1(),
        input_views=[[x]],
        oracle_pairs=[ref.astype(np.float64)],
        read=read,
    )
    self.assertEqual(rep.status, sqnr.STATUS_OK)
    self.assertGreater(len(sizes), 16)
    self.assertLess(max(sizes), 2 * (1 << 17))
    self.assertNotIn(math.prod(shape), sizes)

  def test_lazy_transposed_view_reads_through_callback(self):
    rng = np.random.default_rng(15)
    x = rng.standard_normal((512, 512)).astype(np.float32)
    r_big = x.T + 1e-3 * rng.standard_normal((512, 512)).astype(np.float32)
    c_big = np.ascontiguousarray(x.T)
    sizes = []

    def read(a, sl):
      out = np.asarray(a[sl] if sl else a)
      sizes.append(out.size)
      return out

    rep = sqnr.audit_leaf(
        [(r_big, c_big)],
        config=sqnr.SqnrConfig.six_gate_v1(),
        input_views=[[x]],
        read=read,
    )
    self.assertFalse(rep.passed)
    self.assertTrue(
        any("min_delta_sqnr_db" in r for r in rep.failed_reasons),
        rep.failed_reasons,
    )
    self.assertLess(max(sizes), 2 * (1 << 17))
    self.assertEqual(rep, _gate(c_big, r_big, inputs=(x,)))

  def test_peak_memory_is_bounded(self):
    for shape in ((2048, 2048), (32, 131072), (64, 50257), (33, 64, 64, 32)):
      big_r = (
          np.ones(shape, dtype=np.float32)
          + np.float32(0.01)
          * np.arange(shape[0], dtype=np.float32)[
              (slice(None),) + (None,) * (len(shape) - 1)
          ]
      )
      big_c = (big_r * np.float32(1.00001)).astype(np.float32)
      gc.collect()
      tracemalloc.start()
      try:
        rep = _gate(big_c, big_r)
        _, peak_bytes = tracemalloc.get_traced_memory()
      finally:
        tracemalloc.stop()
      self.assertTrue(rep.passed, msg=str(shape))
      self.assertLess(peak_bytes / big_r.size, 20.0, msg=str(shape))


class SixGateV1HonestTest(parameterized.TestCase):
  """Honest-precision cases that the preset must pass."""

  def test_honest_precisions_thin_axes_dc_sparse_and_state_leaves(self):
    rng = np.random.default_rng(11)
    ref = rng.standard_normal((128, 64))
    ref[0, :] = 1.3
    self.assertTrue(_gate(_bf16(ref), ref).passed)
    bcast = np.broadcast_to(rng.standard_normal((128, 1)), (128, 64)).copy()
    self.assertTrue(_gate(_bf16(bcast), bcast).passed)
    for shape, ax in (
        ((128, 128), 1),
        ((128, 128, 2), (1, 2)),
        ((128, 2), 1),
        ((2, 128), 1),
    ):
      orc = rng.standard_normal(shape)
      amax = np.max(np.abs(orc), axis=ax, keepdims=True)
      fp8x3 = (orc + rng.uniform(-1.0, 1.0, shape) * amax * 2.0**-9).astype(
          np.float32
      )
      self.assertTrue(_gate(fp8x3, orc).passed, msg=str(shape))
      a = rng.standard_normal((shape[0], 64)).astype(np.float32)
      b = rng.standard_normal((64, shape[-1])).astype(np.float32)
      self.assertTrue(
          _gate(_bf16_matmul(a, b), a @ b, min_sqnr_db=44.0).passed,
          msg=str(shape),
      )

    p = np.exp(rng.standard_normal((64, 512)))
    p /= p.sum(-1, keepdims=True)
    self.assertTrue(
        _gate(
            p.astype(np.float32).astype(np.float64).sum(-1).astype(np.float32),
            p.astype(np.float32).sum(-1),
        ).passed
    )
    logits = rng.standard_normal((16, 1024))
    lsm = logits - np.log(np.sum(np.exp(logits), axis=-1, keepdims=True))
    sp_l = rng.standard_normal((8, 4096)) * 8.0
    sp_l -= np.max(sp_l, axis=-1, keepdims=True)
    sm_sp = np.exp(sp_l) / np.sum(np.exp(sp_l), axis=-1, keepdims=True)
    for orc in (
        5.0 + rng.standard_normal((64, 64)),
        100.0 + rng.standard_normal((128, 256)),
        lsm,
        sm_sp,
    ):
      self.assertTrue(_gate(_bf16(orc, np.float32), orc).passed)
    st = rng.standard_normal((32, 32)).astype(np.float32)
    self.assertTrue(
        _gate(np.nextafter(st, np.float32(np.inf)), st, inputs=(st,)).passed
    )
    r_res = (
        st.astype(np.float64) + 1e-2 * _gelu_np(st.astype(np.float64))
    ).astype(np.float32)
    self.assertTrue(_gate(_bf16(r_res, np.float32), r_res, inputs=(st,)).passed)
    self.assertFalse(_gate(st, r_res, inputs=(st,)).passed)
    bf16_ref = _bf16(ref)
    self.assertTrue(_gate(bf16_ref, bf16_ref).passed)
    bf16_pert = bf16_ref.copy()
    bf16_pert[0, 0] = ml_dtypes.bfloat16(1.3046875)
    rep_bf16 = _gate(bf16_pert, bf16_ref, min_sqnr_db=40.0)
    self.assertEqual(rep_bf16.status, sqnr.STATUS_OK)
    self.assertTrue(rep_bf16.passed)

  @parameterized.parameters(1000.0, 1000.5, 1001.0, 1002.0)
  def test_bf16_grid_phases(self, dc_off):
    rng = np.random.default_rng(int(dc_off * 2))
    orc_dc = dc_off + rng.standard_normal((128, 256))
    self.assertTrue(_gate(_bf16(orc_dc, np.float32), orc_dc).passed)
    for c_const in (dc_off, 1000.0, 1004.0):
      self.assertFalse(_gate(np.full_like(orc_dc, c_const), orc_dc).passed)
    half_rows_dc = orc_dc.copy()
    half_rows_dc[:64, :] = dc_off
    rep_hr = _gate(half_rows_dc, orc_dc)
    self.assertFalse(rep_hr.passed)
    self.assertGreater(_nn(rep_hr.min_ac_snr_db), 50.0)
    self.assertTrue(
        any("macro_corr_snr_db" in r for r in rep_hr.failed_reasons),
        rep_hr.failed_reasons,
    )
    half_cols_dc = orc_dc.copy()
    half_cols_dc[:, :128] = dc_off
    self.assertFalse(_gate(half_cols_dc, orc_dc).passed)

  def test_quiet_rows_and_complex_rounding(self):
    rng = np.random.default_rng(23)
    r_qrow = rng.standard_normal((128, 128)).astype(np.float32)
    r_qrow[0, :] = (1e-5 * rng.standard_normal(128)).astype(np.float32)
    c_qrow = r_qrow + (1e-5 * rng.standard_normal((128, 128))).astype(
        np.float32
    )
    self.assertTrue(_gate(c_qrow, r_qrow).passed)
    r_bf_q = _bf16(r_qrow)
    c_bf_q = r_bf_q.copy()
    c_bf_q[0, :] = _bf16(2e-3 * rng.standard_normal(128))
    r_bf_q[0, :] = _bf16(1e-3 * rng.standard_normal(128))
    self.assertTrue(_gate(c_bf_q, r_bf_q, min_sqnr_db=40.0).passed)
    c_fft_r = (
        rng.standard_normal(1024) + 1j * 1e-5 * rng.standard_normal(1024)
    ).astype(np.complex64)
    c_fft_c = (c_fft_r.real + 1j * 1e-5 * rng.standard_normal(1024)).astype(
        np.complex64
    )
    self.assertTrue(_gate(c_fft_c, c_fft_r).passed)

  def test_multi_view_delta_and_ulp_updates(self):
    rng = np.random.default_rng(24)
    attn_mask = np.zeros((64, 64), dtype=np.float32)
    attn_mask[:, 32:] = -np.inf
    x_ulp = np.ones((64, 64), dtype=np.float32)
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(64, 64))
    ref_6ulp = (x_ulp + np.float32(6.0 * 2.0**-23) * signs).astype(np.float32)
    cand_2ulp = (ref_6ulp + np.float32(2.0 * 2.0**-23) * signs).astype(
        np.float32
    )
    self.assertTrue(
        _gate(cand_2ulp, ref_6ulp, inputs=(attn_mask, x_ulp)).passed
    )
    x_other = (ref_6ulp - np.float32(0.01) * signs).astype(np.float32)
    self.assertFalse(_gate(x_ulp, ref_6ulp, inputs=(x_ulp, x_other)).passed)

  def test_honest_quiet_slices_across_seeds(self):
    for seed in range(20):
      for shape in ((32, 32), (128, 32), (32, 64), (32, 128), (32, 256)):
        for ratio in (0.25, 0.5, 1.0):
          rs = np.random.default_rng(1000 + seed)
          q_ref = rs.standard_normal(shape).astype(np.float32)
          q_ref[0, :] = (
              np.sqrt(ratio) * 1e-3 * rs.standard_normal(shape[1])
          ).astype(np.float32)
          q_cand = q_ref + (1e-3 * rs.standard_normal(shape)).astype(np.float32)
          rep_q = _gate(q_cand, q_ref)
          self.assertTrue(
              rep_q.passed, msg=f"seed={seed} shape={shape} ratio={ratio}"
          )

  def test_tiny_reference_with_zero_input_does_not_underflow(self):
    tiny_r = np.random.default_rng(25).standard_normal((16, 16)) * 1e-200
    self.assertTrue(
        _gate(tiny_r * 1.00001, tiny_r, inputs=(np.zeros_like(tiny_r),)).passed
    )

  def test_scalar_leaves(self):
    rep_0d = _gate(np.float32(10.0001), np.float32(10.0))
    self.assertTrue(rep_0d.passed)
    self.assertAlmostEqual(
        _nn(rep_0d.sqnr_db), _nn(rep_0d.min_block_sqnr_db), places=5
    )
    self.assertFalse(_gate(np.float32(0.0), np.float32(10.0)).passed)
    self.assertTrue(_gate(10.00001, 10.0).passed)
    self.assertFalse(
        _gate(
            np.float32(10.0), np.float32(10.05), inputs=(np.float32(10.0),)
        ).passed
    )
    c0_ref = np.array(3.0 + 4.0j, dtype=np.complex64)
    self.assertTrue(
        _gate(np.array(3.0 + 4.00002j, dtype=np.complex64), c0_ref).passed
    )
    self.assertFalse(
        _gate(np.array(3.0 + 0.0j, dtype=np.complex64), c0_ref).passed
    )
    self.assertTrue(
        _gate(np.float32(10.0001), np.float32(10.0), min_sqnr_db=44.0).passed
    )

  def test_odd_7d_and_zero_padded_leaves(self):
    rng = np.random.default_rng(26)
    odd_7d = (1.0 + 0.1 * rng.standard_normal((9,) * 7)).astype(np.float32)
    self.assertTrue(
        _gate((odd_7d * np.float32(1.0001)).astype(np.float32), odd_7d).passed
    )
    odd_bad = odd_7d.copy()
    odd_bad.reshape(-1)[:32] = 0.0
    self.assertFalse(_gate(odd_bad, odd_7d).passed)
    z_pad = np.ones((2048, 2048), dtype=np.float32)
    z_pad[1100:, :] = 0.0
    self.assertTrue(_gate(z_pad + np.float32(1e-30), z_pad).passed)

  def test_two_shard_leaf_with_zero_shard(self):
    rng = np.random.default_rng(27)
    sh0_r = rng.standard_normal((16, 32)).astype(np.float32)
    sh1_r = np.zeros((16, 32), dtype=np.float32)
    sh0_c = sh0_r + (1e-6 * rng.standard_normal((16, 32))).astype(np.float32)
    sh1_c = (1e-7 * rng.standard_normal((16, 32))).astype(np.float32)
    # Input shards of another shape are skipped.
    wrong = [sh0_r.reshape(32, 16), sh1_r.reshape(32, 16)]
    rep = sqnr.audit_leaf(
        [(sh0_r, sh0_c), (sh1_r, sh1_c)],
        config=sqnr.SqnrConfig.six_gate_v1(),
        input_views=[wrong],
    )
    self.assertTrue(rep.passed, rep)


class SixGateV1CheatTest(parameterized.TestCase):
  """Cheats that the preset must reject."""

  @parameterized.named_parameters(
      ("denormal", np.ones(4), np.full(4, 1e-320)),
      ("float64_range", np.array([1e-10, 1e308]), np.full(2, 1e-10)),
      (
          "float64_max",
          np.array([0.8e308, -0.8e308]),
          np.array([-0.8e308, 0.8e308]),
      ),
      (
          "float32_max",
          np.array([1e38, -1e38], np.float32),
          np.array([-1e38, 1e38], np.float32),
      ),
  )
  def test_extreme_ranges(self, ref, cand):
    rep = _gate(cand, ref)
    self.assertEqual(rep.status, sqnr.STATUS_OK)
    self.assertIs(type(rep.passed), bool)
    self.assertFalse(rep.passed)

  def test_downcast_dc_skip_and_spike(self):
    rng = np.random.default_rng(7)
    r0 = rng.standard_normal((128, 128)).astype(np.float32)
    self.assertFalse(
        _gate(_cast(r0, ml_dtypes.float8_e4m3fn, np.float32), r0).passed
    )
    r_dc = 1e15 + rng.standard_normal((64, 128))
    v_dc = _gate(np.full_like(r_dc, 1e15), r_dc)
    self.assertFalse(v_dc.passed)
    self.assertAlmostEqual(_nn(v_dc.centered_snr_db), 0.0, delta=0.05)
    r_3d = rng.standard_normal((32, 32, 32)).astype(np.float32)
    c_3d = r_3d.copy()
    c_3d[:, 0, 0] = 0.0
    self.assertFalse(_gate(c_3d, r_3d, min_sqnr_db=25.0).passed)
    r_dc100 = 100.0 + rng.standard_normal((128, 256))
    c_dc100 = r_dc100.copy()
    c_dc100[0, :32] = 100.0
    self.assertFalse(_gate(c_dc100, r_dc100).passed)
    c_col100 = r_dc100.copy()
    c_col100[:32, 0] = 100.0
    self.assertFalse(_gate(c_col100, r_dc100).passed)
    r_1d100 = 100.0 + rng.standard_normal(200_000)
    c_1d100 = r_1d100.copy()
    c_1d100[-32:] = 100.0
    self.assertFalse(_gate(c_1d100, r_1d100).passed)
    c_sp = r0.copy()
    c_sp[5, 9] += 0.3
    rep_sp = _gate(c_sp, r0)
    self.assertFalse(rep_sp.passed)
    self.assertGreater(_nn(rep_sp.peak_error_rms_ratio), 16.0)
    half_skip_100 = r_dc100.copy()
    half_skip_100[:64, :] = 100.0
    self.assertFalse(_gate(half_skip_100, r_dc100).passed)

  def test_bias_shortcuts(self):
    rng = np.random.default_rng(8)
    rb16 = rng.standard_normal((16, 1)) * 200.0
    cb16 = rng.standard_normal((1, 16)) * 200.0
    r16 = rb16 + cb16 + rng.standard_normal((16, 16))
    c16 = (rb16 + cb16) * np.ones((16, 16))
    r16[0, 0] = c16[0, 0] = np.nan
    self.assertFalse(_gate(c16, r16).passed)
    rb3 = rng.standard_normal((3, 1)) * 200.0
    cb3 = rng.standard_normal((1, 128)) * 200.0
    self.assertFalse(
        _gate(
            (rb3 + cb3) * np.ones((3, 128)),
            rb3 + cb3 + rng.standard_normal((3, 128)),
        ).passed
    )
    rb_1e6 = rng.standard_normal((64, 1)) * 1e6
    self.assertFalse(
        _gate(
            np.broadcast_to(rb_1e6, (64, 64)).copy(),
            rb_1e6 + rng.standard_normal((64, 64)),
        ).passed
    )
    rb4 = (rng.standard_normal((2, 9, 64, 1)) * 200.0).astype(np.float32)
    cb4 = (rng.standard_normal((2, 9, 1, 128)) * 200.0).astype(np.float32)
    c4_cheat = (rb4 + cb4) * np.ones((2, 9, 64, 128), dtype=np.float32)
    r4_true = c4_cheat + rng.standard_normal((2, 9, 64, 128)).astype(np.float32)
    self.assertFalse(_gate(c4_cheat, r4_true).passed)

  def test_row_mean_on_two_shards(self):
    rng = np.random.default_rng(9)
    r_na = (
        rng.standard_normal((32, 32)).astype(np.float32)
        + rng.standard_normal((32, 1)).astype(np.float32) * 5000.0
    )
    c_na = np.broadcast_to(r_na.mean(axis=1, keepdims=True), (32, 32))
    v1 = sqnr.SqnrConfig.six_gate_v1()

    def shards(a):
      return [a[:16], a[16:]]

    self.assertFalse(
        sqnr.audit_leaf(list(zip(shards(r_na), shards(c_na))), config=v1).passed
    )
    self.assertTrue(
        sqnr.audit_leaf(
            list(zip(shards(r_na), shards(r_na.copy()))), config=v1
        ).passed
    )

  def test_passthroughs_quiet_expert_and_complex_part(self):
    rng = np.random.default_rng(10)
    x = rng.standard_normal((32, 64)).astype(np.float32)
    r_res = (
        x.astype(np.float64) + 1e-3 * _gelu_np(x.astype(np.float64))
    ).astype(np.float32)
    r_res[0, 0] = np.nan
    c_res = x.copy()
    c_res[0, 0] = np.nan
    self.assertFalse(
        _gate(
            c_res,
            r_res,
            inputs=(x, "cfg", np.dtype("float32"), None, np.int32(3)),
        ).passed
    )
    # Noise 1e-3 gives a global SQNR of about 60 dB > 45 dB, so only the
    # transposed input view catches x.T.
    r_tr = x.T + 1e-3 * rng.standard_normal((64, 32)).astype(np.float32)
    self.assertTrue(_gate(x.T, r_tr).passed)
    self.assertFalse(_gate(x.T, r_tr, inputs=(x,)).passed)
    xi = rng.integers(10, 500, size=(32, 32), dtype=np.int32)
    self.assertFalse(
        _gate(
            xi.astype(np.float32),
            xi.astype(np.float32)
            + 0.25 * rng.standard_normal((32, 32)).astype(np.float32),
            inputs=(xi,),
        ).passed
    )
    r_moe = (rng.standard_normal((64, 256)) / 32.0).astype(np.float32)
    r_moe[0, :] = rng.standard_normal(256).astype(np.float32) * 80.0
    c_moe = np.zeros_like(r_moe)
    c_moe[0, :] = r_moe[0, :]
    self.assertFalse(_gate(c_moe, r_moe, min_sqnr_db=40.0).passed)
    rc_1d = (
        rng.standard_normal(128) + 1j * 1e-3 * rng.standard_normal(128)
    ).astype(np.complex64)
    self.assertFalse(
        _gate((rc_1d.real + 0j).astype(np.complex64), rc_1d).passed
    )

  def test_return_input_above_floor_fails_only_on_delta(self):
    rng = np.random.default_rng(31)
    x_in = rng.standard_normal((128, 128)).astype(np.float32)
    upd = rng.standard_normal((128, 128))
    upd *= np.sqrt(np.sum(np.square(x_in, dtype=np.float64))) / (
        np.sqrt(np.sum(np.square(upd))) * 10.0 ** (48.0 / 20.0)
    )
    r_upd = (x_in.astype(np.float64) + upd).astype(np.float32)
    self.assertTrue(
        _gate(_bf16(r_upd, np.float32), r_upd, inputs=(x_in,)).passed
    )
    rep_noop = _gate(x_in, r_upd, inputs=(x_in,))
    self.assertFalse(rep_noop.passed)
    self.assertGreater(_nn(rep_noop.sqnr_db), 45.0)
    self.assertTrue(
        any("min_delta_sqnr_db" in r for r in rep_noop.failed_reasons)
    )
    self.assertTrue(_gate(x_in, r_upd).passed)

  def test_comparison_table_verdicts(self):
    rng = np.random.default_rng(42)
    a = (rng.standard_normal((256, 512)) / np.sqrt(512.0)).astype(np.float32)
    b = rng.standard_normal((512, 256)).astype(np.float32)
    y_mm = a @ b
    bf16_mm = _bf16_matmul(a, b)
    amax = np.max(np.abs(y_mm), axis=1, keepdims=True)
    fp8x3_mm = (
        y_mm + rng.uniform(-1.0, 1.0, y_mm.shape) * amax * (2.0**-9)
    ).astype(np.float32)
    y_b = (0.5 * y_mm).astype(np.float32)
    bf16_b = (0.5 * bf16_mm).astype(np.float32)
    fp8x3_b = (0.5 * fp8x3_mm).astype(np.float32)
    fp8_1p = _cast(y_b, ml_dtypes.float8_e4m3fn, np.float32)
    y_moe = (rng.standard_normal((128, 128)) * 0.04).astype(np.float32)
    y_moe[0, :] = (rng.standard_normal(128) * 4.0).astype(np.float32)
    c_moe_skip = y_moe.copy()
    c_moe_skip[1:33, :32] = 0.0
    x_res = (rng.standard_normal((128, 128)) * 0.5).astype(np.float32)
    y_res = (
        x_res.astype(np.float64) + 0.01 * _gelu_np(x_res.astype(np.float64))
    ).astype(np.float32)
    y_dc = (10.0 + 0.05 * rng.standard_normal((128, 128))).astype(np.float32)
    c_dc = np.full_like(y_dc, 10.0)
    c_spike = y_b.copy()
    c_spike[7, 13] += np.float32(0.30)
    cases = [
        ("bf16_mxu", bf16_b, y_b, (), True),
        ("fp8x3_residual", fp8x3_b, y_b, (), True),
        ("fp8_1pass_downcast", fp8_1p, y_b, (), False),
        ("quiet_32x32_skip", c_moe_skip, y_moe, (), False),
        ("residual_noop", x_res, y_res, (x_res,), False),
        ("dc_bias_shortcut", c_dc, y_dc, (), False),
        ("isolated_spike", c_spike, y_b, (), False),
    ]
    for name, cand, ref, inps, expected in cases:
      rep = _gate(cand, ref, min_sqnr_db=44.0, inputs=inps)
      self.assertEqual(rep.passed, expected, msg=f"{name} {rep}")


if __name__ == "__main__":
  absltest.main()
