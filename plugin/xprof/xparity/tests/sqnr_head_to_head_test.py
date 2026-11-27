"""Tests for the SQNR head-to-head corpus, scorer and decision rule."""

from collections.abc import Sequence
import functools

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from xprof.xparity import sqnr
from xprof.xparity.tests import sqnr_head_to_head as h2h
from xprof.xparity.tests import sqnr_head_to_head_heldout as heldout

_DERIVED = sqnr.GATE_MATH_DERIVED
_V1 = sqnr.GATE_MATH_SIX_GATE_V1


def _outcome(
    label: str,
    gate_math: str,
    passed: bool,
    checks: Sequence[str] = (),
    case: str = "c",
    seed: int = 0,
) -> h2h.Outcome:
  return h2h.Outcome(
      case=case,
      label=label,
      group="g",
      seed=seed,
      gate_math=gate_math,
      policy=h2h.POLICY_AUTO,
      passed=passed,
      status=sqnr.STATUS_OK,
      threshold_db=45.0,
      sqnr_db=50.0,
      checks=tuple(checks),
  )


@functools.cache
def _reduced_run() -> tuple[list[h2h.Case], list[h2h.Outcome]]:
  cases = h2h.build_corpus(full=False, honest_seeds=2, cheat_seeds=1)
  return cases, h2h.run_corpus(cases)


class DecideTest(parameterized.TestCase):

  def test_derived_wins_when_no_worse_on_both_rates(self):
    outcomes = [
        _outcome(h2h.HONEST, _DERIVED, True),
        _outcome(h2h.HONEST, _V1, False, ["min_block_sqnr_db"]),
        _outcome(h2h.CHEAT, _DERIVED, False, ["max_error_ulp"], case="x"),
        _outcome(h2h.CHEAT, _V1, False, ["peak_error_rms_ratio"], case="x"),
    ]
    decision = h2h.decide(outcomes)
    self.assertEqual(decision["default"], _DERIVED)
    self.assertEqual(decision["evidence"]["derived_false_red"], 0.0)
    self.assertEqual(decision["evidence"]["six_gate_v1_false_red"], 1.0)

  def test_derived_keeps_v1_checks_that_catch_its_false_greens(self):
    outcomes = [
        _outcome(h2h.HONEST, _DERIVED, True),
        _outcome(h2h.HONEST, _V1, True),
        _outcome(h2h.CHEAT, _DERIVED, True, case="x"),
        _outcome(h2h.CHEAT, _V1, False, ["min_ac_snr_db"], case="x"),
    ]
    decision = h2h.decide(outcomes)
    self.assertEqual(decision["default"], "derived+keep:min_ac_snr_db")
    self.assertEqual(decision["evidence"]["derived_missed_cheats"], ["x"])

  def test_v1_stays_default_when_derived_has_more_false_reds(self):
    outcomes = [
        _outcome(h2h.HONEST, _DERIVED, False, ["max_error_ulp"]),
        _outcome(h2h.HONEST, _V1, True),
        _outcome(h2h.CHEAT, _DERIVED, False, ["max_error_ulp"]),
        _outcome(h2h.CHEAT, _V1, False, ["sqnr_db"]),
    ]
    decision = h2h.decide(outcomes)
    self.assertEqual(decision["default"], _V1)
    self.assertEqual(decision["evidence"]["derived_false_red_cases"], ["c"])


class CorpusTest(parameterized.TestCase):

  def test_case_names_are_unique_and_labels_are_known(self):
    cases = h2h.build_corpus(full=True)
    names = [c.name for c in cases]
    self.assertLen(set(names), len(names))
    self.assertEqual({c.label for c in cases}, {h2h.HONEST, h2h.CHEAT})
    self.assertTrue(
        all(len(c.seeds) >= 20 for c in cases if c.label == h2h.HONEST)
    )
    self.assertIn("cast_mxfp4_gaussian_n2^24", names)

  def test_generation_is_deterministic(self):
    for case in h2h.build_corpus(full=False, honest_seeds=1, cheat_seeds=1):
      first, second = case.generate(case.seeds[0]), case.generate(case.seeds[0])
      for field in ("reference", "candidate"):
        a, b = getattr(first, field), getattr(second, field)
        for x, y in zip(h2h._shards(a), h2h._shards(b)):
          np.testing.assert_array_equal(
              np.asarray(x), np.asarray(y), err_msg=f"{case.name} {field}"
          )

  @parameterized.parameters(
      ("bfloat16", 55.6), ("float8_e4m3fn", 31.5), ("mxfp4", 18.8)
  )
  def test_honest_cast_sqnr_matches_format(self, fmt, expected_db):
    data = h2h._cast_case(fmt, "gaussian", 1 << 16, 0)
    ref = np.asarray(data.reference, np.float64)
    err = np.asarray(data.candidate, np.float64) - ref
    measured = 10.0 * np.log10(np.sum(ref**2) / np.sum(err**2))
    self.assertAlmostEqual(measured, expected_db, delta=0.3)

  def test_transposed_input_view_matches_source_rule(self):
    x = np.arange(6.0).reshape(2, 3)
    views = h2h.input_views((x, np.ones(5)), (3, 2))
    self.assertLen(views, 2)
    np.testing.assert_array_equal(views[0], x.T)
    np.testing.assert_array_equal(views[1], x.reshape(3, 2))
    # sqnr.audit_leaf adds the transpose of a square view, so input_views
    # returns a same-shape square input once, without its transpose.
    square = np.arange(9.0).reshape(3, 3)
    views = h2h.input_views((square,), (3, 3))
    self.assertLen(views, 1)
    np.testing.assert_array_equal(views[0], square)


def _sqnr_db(reference: np.ndarray, candidate: np.ndarray) -> float:
  ref = np.asarray(reference, np.float64)
  err = np.asarray(candidate, np.float64) - ref
  return float(10.0 * np.log10(np.sum(ref**2) / np.sum(err**2)))


class HeldoutTest(parameterized.TestCase):

  def test_heldout_cases_are_separate_from_the_main_corpus(self):
    main = {c.name for c in h2h.build_corpus(full=True)}
    cases = h2h.build_heldout_corpus()
    names = [c.name for c in cases]
    self.assertLen(set(names), len(names))
    self.assertEmpty(main & set(names))
    honest = [c for c in cases if c.label == h2h.HONEST]
    cheats = [c for c in cases if c.label == h2h.CHEAT]
    self.assertGreaterEqual(len(cheats), 3)
    self.assertTrue(all(len(c.seeds) >= 20 for c in honest))
    self.assertEqual(
        {c.group for c in honest},
        {
            "ho_a_accumulation_order",
            "ho_b_two_sided",
            "ho_c_nvfp4",
            "ho_d_int8",
            "ho_e_f32_order",
            "ho_f_residual_softmax",
        },
    )

  def test_heldout_generation_is_deterministic(self):
    for case in h2h.build_heldout_corpus(honest_seeds=1, cheat_seeds=1):
      first, second = case.generate(case.seeds[0]), case.generate(case.seeds[0])
      np.testing.assert_array_equal(
          np.asarray(first.candidate), np.asarray(second.candidate), case.name
      )

  def test_nvfp4_matches_the_format_sqnr(self):
    data = heldout._nvfp4_cast("gaussian", 0)
    self.assertAlmostEqual(
        _sqnr_db(data.reference, data.candidate), 20.4, delta=0.3
    )

  def test_nvfp4_block_has_at_most_16_distinct_magnitudes_per_scale(self):
    x = np.random.default_rng(0).standard_normal((4, 64)).astype(np.float32)
    q = heldout.nvfp4_quantize(x).reshape(-1, 16).astype(np.float64)
    for block in q:
      mags = np.unique(np.abs(block[block != 0]))
      ratios = mags / mags.max()
      # e2m1 magnitudes relative to 6: {0.5, 1, 1.5, 2, 3, 4, 6} / 6.
      allowed = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]) / 6.0
      self.assertTrue(
          np.all(np.min(np.abs(ratios[:, None] - allowed), axis=1) < 1e-6)
      )

  def test_accumulation_order_differs_in_few_elements(self):
    data = heldout._accumulation_order(
        np.dtype(heldout._BF16), "gaussian", 4, "reverse", 0
    )
    n_diff = int(np.count_nonzero(data.reference != data.candidate))
    self.assertBetween(n_diff, 1, data.reference.size // 100)

  def test_int8_cases_use_an_explicit_floor_below_their_sqnr(self):
    data = heldout._int8_matmul("gaussian", 4096, 0)
    self.assertGreater(
        _sqnr_db(data.reference, data.candidate),
        heldout.INT8_FLOOR_DB + 3.0,
    )
    case = next(c for c in h2h.build_heldout_corpus() if c.group == "ho_d_int8")
    for policy in h2h.POLICIES:
      config = h2h.make_config(sqnr.GATE_MATH_DERIVED, policy, case)
      self.assertEqual(config.min_sqnr_db, heldout.INT8_FLOOR_DB)


class ReducedRunTest(absltest.TestCase):

  def test_run_reports_every_config_and_policy(self):
    cases, outcomes = _reduced_run()
    instances = sum(len(c.seeds) for c in cases)
    self.assertLen(outcomes, instances * 4)
    rates = h2h.score(outcomes)
    self.assertLen(rates, 4)
    table = h2h.markdown_table(cases, outcomes, 0.0)
    self.assertIn("| derived | auto |", table)
    self.assertIn("## False reds per group", table)

  def test_default_gate_math_follows_the_decision_rule(self):
    _, outcomes = _reduced_run()
    decision = h2h.decide(outcomes, h2h.POLICY_AUTO)
    # "derived+keep:<checks>" still means the derived math is the default.
    expected = decision["default"].split("+", 1)[0]
    self.assertEqual(sqnr.SqnrConfig().gate_math, expected, msg=str(decision))


if __name__ == "__main__":
  absltest.main()
