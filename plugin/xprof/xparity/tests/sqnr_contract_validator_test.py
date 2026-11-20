"""Tests for the SQNR contract and the SQNR report in the validator."""

from collections.abc import Callable
from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import ml_dtypes
import numpy as np

from xprof import xparity
from xprof.xparity import numerical_validator
from xprof.xparity import sqnr

_SQNR = numerical_validator.CONTRACT_SQNR
# One round-to-nearest to bfloat16 (8 significand bits) gives about
# 6.02 * 8 + 7.44 = 55.6 dB on normal data, above the 45 dB default floor.
_BF16_ROUNDING_DB = 55.6


def _sqnr(result: numerical_validator.BatchValidationResult) -> dict[str, Any]:
  """Returns the SQNR report of a batch or leaf result."""
  assert result.sqnr is not None
  return result.sqnr


def _leaves(
    batch: numerical_validator.BatchValidationResult,
) -> dict[str, numerical_validator.BatchValidationResult]:
  assert batch.leaf_results is not None
  return batch.leaf_results


def _ref(inputs: Any) -> np.ndarray:
  """Halves the input and keeps its dtype, so float64 inputs stay float64."""
  return np.multiply(np.asarray(inputs), 0.5)


def _oracle(inputs: Any) -> np.ndarray:
  return np.asarray(inputs, dtype=np.float64) * 0.5


def _bf16_rounded(inputs: Any) -> np.ndarray:
  """_ref with each output rounded once to bfloat16, as float32."""
  return _ref(inputs).astype(ml_dtypes.bfloat16).astype(np.float32)


def _half_scale(inputs: Any) -> np.ndarray:
  return _ref(inputs) * np.float32(0.5)


def _zero_block(inputs: Any) -> np.ndarray:
  """_ref with the first half of each row left at zero, a skipped tile."""
  out = _ref(inputs).copy()
  out[..., : out.shape[-1] // 2] = 0.0
  return out


def _argmax(inputs: Any) -> np.ndarray:
  return np.argmax(np.asarray(inputs), axis=-1)


def _argmax_shifted(inputs: Any) -> np.ndarray:
  return _argmax(inputs) + 1


def _tree_ref(inputs: Any) -> tuple[np.ndarray, np.ndarray]:
  return _argmax(inputs), _ref(inputs)


def _tree_cand(inputs: Any) -> tuple[np.ndarray, np.ndarray]:
  return _argmax(inputs), _bf16_rounded(inputs)


def _validate(
    candidate: Callable[..., Any],
    **kwargs: Any,
) -> numerical_validator.KernelValidationReport:
  kwargs.setdefault("shapes", (64, 64))
  kwargs.setdefault("dtype_str", "float32")
  kwargs.setdefault("regimes", "normal")
  kwargs.setdefault("contract", _SQNR)
  return numerical_validator.validate_kernels(_ref, candidate, **kwargs)


class SqnrContractTest(parameterized.TestCase):

  def test_honest_bf16_rounding_passes(self):
    report = _validate(_bf16_rounded)
    self.assertTrue(report.is_numerically_equivalent, report.summary_message)
    self.assertIn("contract: sqnr", report.summary_message)
    summary = report.sqnr_summary
    self.assertAlmostEqual(summary["sqnr_db"], _BF16_ROUNDING_DB, delta=2.0)
    self.assertEqual(
        (
            summary["passed"],
            summary["threshold_db"],
            summary["threshold_source"],
            summary["gate_math"],
            summary["failed_reports"],
        ),
        (
            True,
            sqnr.DEFAULT_MIN_SQNR_DB,
            sqnr.THRESHOLD_SOURCE_DEFAULT,
            sqnr.GATE_MATH_DERIVED,
            0,
        ),
    )

  @parameterized.named_parameters(
      ("half_scale", _half_scale),
      ("zero_block", _zero_block),
  )
  def test_cheating_candidate_fails(self, candidate: Callable[..., Any]):
    report = _validate(candidate)
    self.assertFalse(report.is_numerically_equivalent)
    self.assertStartsWith(report.summary_message, "FAILED")
    self.assertIn("contract: sqnr", report.summary_message)
    first = _sqnr(report.batch_results[0])
    self.assertFalse(first["passed"])
    self.assertLess(first["sqnr_db"], 10.0)
    self.assertNotEmpty(first["failed_reasons"])
    self.assertEqual(
        report.sqnr_summary["failed_reports"], report.failed_batches_count
    )

  def test_explicit_floor_is_honored(self):
    report = _validate(
        _bf16_rounded, sqnr_config=sqnr.SqnrConfig(min_sqnr_db=70.0)
    )
    self.assertFalse(report.is_numerically_equivalent)
    first = _sqnr(report.batch_results[0])
    self.assertEqual(
        (first["threshold_db"], first["threshold_source"]),
        (70.0, sqnr.THRESHOLD_SOURCE_EXPLICIT),
    )
    self.assertIn("sqnr_db", first["failed_reasons"][0])
    self.assertIn("sqnr_db", report.summary_message)

  def test_run_config_and_summary_carry_the_config(self):
    config = sqnr.SqnrConfig(min_sqnr_db=40.0, strict=False)
    report = _validate(_bf16_rounded, sqnr_config=config)
    self.assertEqual(
        (
            report.run_config["sqnr_config"]["min_sqnr_db"],
            report.run_config["sqnr_config"]["strict"],
            report.sqnr_summary["config"]["min_sqnr_db"],
        ),
        (40.0, False, 40.0),
    )
    self.assertEqual(report.sqnr_summary["reports"], report.total_batches_count)
    self.assertIn(
        report.sqnr_summary["worst_sqnr_batch"],
        [b.batch_name for b in report.batch_results],
    )

  def test_oracle_is_used_for_the_sqnr_rule(self):
    report = _validate(_bf16_rounded, kernel_oracle=_oracle)
    self.assertTrue(report.is_numerically_equivalent, report.summary_message)
    first = _sqnr(report.batch_results[0])
    self.assertAlmostEqual(
        first["oracle_sqnr_db"], _BF16_ROUNDING_DB, delta=2.0
    )
    self.assertIsNotNone(report.sqnr_summary["oracle_sqnr_db"])

  def test_auto_oracle_is_used_for_the_sqnr_rule(self):
    report = _validate(_half_scale, kernel_oracle="auto")
    self.assertFalse(report.is_numerically_equivalent)
    first = _sqnr(report.batch_results[0])
    self.assertLess(first["oracle_sqnr_db"], 10.0)
    self.assertIn("oracle_sqnr_db", first["failed_reasons"][0])

  @parameterized.named_parameters(
      ("match", _argmax, True),
      ("mismatch", _argmax_shifted, False),
  )
  def test_discrete_output_needs_exact_match(
      self, candidate: Callable[..., Any], expected: bool
  ):
    report = numerical_validator.validate_kernels(
        _argmax,
        candidate,
        shapes=(16, 16),
        dtype_str="float32",
        regimes="normal",
        contract=_SQNR,
    )
    self.assertEqual(report.is_numerically_equivalent, expected)
    self.assertIsNone(report.batch_results[0].sqnr)
    self.assertEqual(report.sqnr_summary, {})
    if not expected:
      self.assertIn("discrete output differs", report.summary_message)

  def test_contract_by_leaf(self):
    report = numerical_validator.validate_kernels(
        _tree_ref,
        _tree_cand,
        shapes=(64, 64),
        dtype_str="float32",
        regimes="normal",
        contract_by_leaf={"[1]": _SQNR},
    )
    self.assertTrue(report.is_numerically_equivalent, report.summary_message)
    leaves = _leaves(report.batch_results[0])
    self.assertEqual(
        (
            leaves["[0]"].sqnr,
            _sqnr(leaves["[1]"])["passed"],
            _sqnr(leaves["[1]"])["path"],
            report.sqnr_summary["worst_sqnr_leaf"],
            report.run_config["sqnr_config"]["strict"],
        ),
        (None, True, "[1]", "[1]", True),
    )

  def test_replay_restores_and_overrides_the_config(self):
    tmp = self.create_tempdir().full_path
    config = sqnr.SqnrConfig(min_sqnr_db=10.0, strict=False)
    report = _validate(_half_scale, sqnr_config=config, dump_failures_to=tmp)
    self.assertFalse(report.is_numerically_equivalent)
    dump = report.failure_dumps[0]
    self.assertEqual(
        xparity.read_suite_metadata(dump)["sqnr_config"]["min_sqnr_db"], 10.0
    )
    again = numerical_validator.replay_failure_dump(dump, _ref, _half_scale)
    self.assertFalse(again.is_numerically_equivalent)
    self.assertEqual(again.run_config["sqnr_config"]["min_sqnr_db"], 10.0)
    relaxed = numerical_validator.replay_failure_dump(
        dump,
        _ref,
        _half_scale,
        sqnr_config=sqnr.SqnrConfig(min_sqnr_db=5.0, strict=False),
    )
    self.assertTrue(relaxed.is_numerically_equivalent, relaxed.summary_message)


class SqnrReportOnlyTest(parameterized.TestCase):
  """Every contract reports SQNR, and only CONTRACT_SQNR gates on it."""

  @parameterized.named_parameters(
      ("ulp", numerical_validator.CONTRACT_ULP),
      ("bitwise", numerical_validator.CONTRACT_BITWISE),
  )
  def test_other_contract_reports_but_does_not_gate(self, contract: str):
    report = _validate(_bf16_rounded, contract=contract)
    self.assertFalse(report.is_numerically_equivalent)
    self.assertTrue(_sqnr(report.batch_results[0])["passed"])
    self.assertTrue(report.sqnr_summary["passed"])
    self.assertIn("sqnr_config", report.run_config)

  def test_row_rel_l2_contract_reports_sqnr(self):
    report = _validate(
        _bf16_rounded,
        contract=numerical_validator.CONTRACT_ROW_REL_L2,
        rel_l2_tol=1e-2,
    )
    self.assertTrue(report.is_numerically_equivalent, report.summary_message)
    self.assertAlmostEqual(
        report.sqnr_summary["sqnr_db"], _BF16_ROUNDING_DB, delta=2.0
    )

  def test_exact_candidate_reports_exact_status(self):
    report = _validate(_ref, contract=numerical_validator.CONTRACT_ULP)
    self.assertTrue(report.is_numerically_equivalent, report.summary_message)
    self.assertEqual(
        (
            _sqnr(report.batch_results[0])["sqnr_status"],
            report.sqnr_summary["sqnr_status"],
        ),
        (sqnr.STATUS_EXACT, sqnr.STATUS_EXACT),
    )

  def test_ulp_contract_keeps_its_summary(self):
    report = _validate(_half_scale, contract=numerical_validator.CONTRACT_ULP)
    self.assertFalse(report.is_numerically_equivalent)
    self.assertFalse(report.sqnr_summary["passed"])
    self.assertNotIn("contract: sqnr", report.summary_message)

  @parameterized.named_parameters(
      ("ulp", numerical_validator.CONTRACT_ULP, False),
      ("bitwise", numerical_validator.CONTRACT_BITWISE, False),
      ("sqnr", _SQNR, True),
  )
  def test_default_config_is_strict_under_sqnr_only(
      self, contract: str, strict: bool
  ):
    report = _validate(_bf16_rounded, contract=contract)
    first = _sqnr(report.batch_results[0])
    self.assertEqual(
        (
            report.run_config["sqnr_config"]["strict"],
            report.sqnr_summary["config"]["strict"],
            first["max_block_noise_ratio_db"] is not None,
        ),
        (strict, strict, strict),
    )

  def test_explicit_strict_config_runs_strict_under_ulp(self):
    report = _validate(
        _bf16_rounded,
        contract=numerical_validator.CONTRACT_ULP,
        sqnr_config=sqnr.SqnrConfig(strict=True),
    )
    self.assertTrue(report.run_config["sqnr_config"]["strict"])
    self.assertIsNotNone(
        _sqnr(report.batch_results[0])["max_block_noise_ratio_db"]
    )

  @parameterized.named_parameters(
      ("ulp", numerical_validator.CONTRACT_ULP, True),
      ("sqnr", _SQNR, False),
  )
  def test_sqnr_error_gives_error_status(self, contract: str, expected: bool):
    with mock.patch.object(
        numerical_validator.sqnr,
        "audit_leaf",
        autospec=True,
        side_effect=RuntimeError("boom"),
    ):
      report = _validate(_ref, contract=contract)
    self.assertEqual(
        report.is_numerically_equivalent, expected, report.summary_message
    )
    first = _sqnr(report.batch_results[0])
    self.assertEqual(
        (first["sqnr_status"], first["passed"], report.sqnr_summary["passed"]),
        (sqnr.STATUS_ERROR, False, False),
    )
    self.assertIn("RuntimeError: boom", first["failed_reasons"][0])

  def test_auto_threshold_uses_the_candidate_dtype(self):
    def bf16_candidate(inputs: Any) -> np.ndarray:
      out = _ref(inputs).astype(np.float32) * np.float32(1.0 + 2.0**-8)
      return out.astype(ml_dtypes.bfloat16)

    report = _validate(bf16_candidate, dtype_str="bfloat16")
    first = _sqnr(report.batch_results[0])
    self.assertStartsWith(
        first["threshold_source"], sqnr.THRESHOLD_SOURCE_AUTO_PREFIX
    )
    self.assertIn("bfloat16", first["threshold_source"])


class SqnrValidateArraysTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.ref = (
        np.random.default_rng(0).standard_normal((64, 32)).astype(np.float32)
    )

  @parameterized.named_parameters(
      ("bf16_rounded", lambda x: x.astype(ml_dtypes.bfloat16), True),
      ("half_scale", lambda x: x * np.float32(0.5), False),
  )
  def test_sqnr_contract(
      self, make_actual: Callable[[np.ndarray], Any], expected: bool
  ):
    actual = np.asarray(make_actual(self.ref)).astype(np.float32)
    res = numerical_validator.validate_arrays(
        actual, self.ref, dtype_str="float32", contract=_SQNR
    )
    self.assertEqual((res.passed, _sqnr(res)["passed"]), (expected, expected))

  def test_ulp_contract_reports_sqnr(self):
    actual = self.ref.astype(ml_dtypes.bfloat16).astype(np.float32)
    res = numerical_validator.validate_arrays(
        actual, self.ref, dtype_str="float32"
    )
    self.assertFalse(res.passed)
    self.assertAlmostEqual(_sqnr(res)["sqnr_db"], _BF16_ROUNDING_DB, delta=2.0)

  @parameterized.named_parameters(
      ("ulp", numerical_validator.CONTRACT_ULP, True),
      ("sqnr", _SQNR, False),
  )
  def test_sqnr_error_gives_error_status(self, contract: str, expected: bool):
    with mock.patch.object(
        numerical_validator.sqnr,
        "audit_leaf",
        autospec=True,
        side_effect=ValueError("bad"),
    ):
      res = numerical_validator.validate_arrays(
          self.ref, self.ref, dtype_str="float32", contract=contract
      )
    self.assertEqual(
        (res.passed, _sqnr(res)["sqnr_status"]),
        (expected, sqnr.STATUS_ERROR),
    )

  def test_explicit_config(self):
    actual = self.ref.astype(ml_dtypes.bfloat16).astype(np.float32)
    res = numerical_validator.validate_arrays(
        actual,
        self.ref,
        dtype_str="float32",
        contract=_SQNR,
        sqnr_config=sqnr.SqnrConfig(min_sqnr_db=70.0),
    )
    self.assertFalse(res.passed)
    self.assertEqual(_sqnr(res)["threshold_db"], 70.0)

  @parameterized.named_parameters(
      ("match", 0, True),
      ("mismatch", 1, False),
  )
  def test_discrete_arrays(self, offset: int, expected: bool):
    ref = np.arange(16, dtype=np.int32)
    res = numerical_validator.validate_arrays(
        ref + offset, ref, dtype_str="int32", contract=_SQNR
    )
    self.assertEqual((res.passed, res.sqnr), (expected, None))


class SqnrExportsTest(absltest.TestCase):

  def test_package_exports(self):
    self.assertEqual(
        (xparity.CONTRACT_SQNR, xparity.SqnrConfig, xparity.compute_sqnr),
        (_SQNR, sqnr.SqnrConfig, sqnr.compute_sqnr),
    )
    self.assertIs(xparity.sqnr, sqnr)

  def test_unknown_contract_names_sqnr(self):
    with self.assertRaisesRegex(ValueError, "sqnr"):
      _validate(_ref, contract="snr")


if __name__ == "__main__":
  absltest.main()
