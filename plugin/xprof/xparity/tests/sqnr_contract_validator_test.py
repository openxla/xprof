"""Tests for the SQNR contract and the SQNR report in the validator."""

from collections.abc import Callable
import dataclasses
import json
from typing import Any
from unittest import mock
import warnings

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


# The residual update is 50 dB below the input, so a candidate that returns
# the input has a global SQNR of about 50 dB, above the 45 dB floor. Only the
# input-delta check of the strict policy can catch it.
_RESIDUAL_SCALE = 10.0 ** (-50.0 / 20.0)


def _residual(inputs: Any) -> np.ndarray:
  """Adds a small update to the input in float64, rounded to float32."""
  x = np.asarray(inputs, dtype=np.float64)
  return (x + _RESIDUAL_SCALE * np.tanh(x)).astype(np.float32)


def _residual_f32(inputs: Any) -> np.ndarray:
  """_residual computed in float32, an honest candidate."""
  x = np.asarray(inputs, dtype=np.float32)
  return x + np.float32(_RESIDUAL_SCALE) * np.tanh(x)


def _passthrough(inputs: Any) -> np.ndarray:
  """Returns the input, a cheat on a residual kernel."""
  return np.asarray(inputs, dtype=np.float32)


def _tree_residual(inputs: Any) -> tuple[np.ndarray, np.ndarray]:
  return _argmax(inputs), _residual(inputs)


def _tree_passthrough(inputs: Any) -> tuple[np.ndarray, np.ndarray]:
  return _argmax(inputs), _passthrough(inputs)


def _dequant(inputs: Any) -> np.ndarray:
  """A float32 output from any input dtype, such as a dequantize kernel."""
  return np.asarray(inputs, dtype=np.float32) * np.float32(0.5)


def _dequant_near(inputs: Any) -> np.ndarray:
  """_dequant with a relative error of 2**-10, about 60 dB."""
  return _dequant(inputs) * np.float32(1.0 + 2.0**-10)


def _dequant_half(inputs: Any) -> np.ndarray:
  """_dequant at half scale: about 6 dB, far below the 45 dB floor."""
  return _dequant(inputs) * np.float32(0.5)


def _tree_dequant(inputs: Any) -> tuple[np.ndarray, np.ndarray]:
  return _dequant(inputs), _dequant(inputs)


def _noisy(inputs: Any) -> np.ndarray:
  """_dequant with 1% relative noise: about 40 dB, below the 45 dB floor."""
  out = _dequant(inputs)
  noise = np.random.default_rng(1).standard_normal(out.shape)
  return out + np.float32(1e-2 * np.std(out)) * noise.astype(np.float32)


def _zeros(inputs: Any) -> np.ndarray:
  return np.zeros(np.shape(inputs), dtype=np.float32)


def _tiny(inputs: Any) -> np.ndarray:
  """A small nonzero output, far below 1e-3 in magnitude."""
  return np.full(np.shape(inputs), 1e-6, dtype=np.float32)


class _Unreadable:
  """An array-like input that cannot be copied to the host.

  It stands for a deleted or donated device buffer.
  """

  shape = (64, 64)
  dtype = np.dtype(np.float32)

  def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
    del dtype, copy  # Unused.
    raise RuntimeError("buffer was deleted")


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

  def test_report_only_atol_does_not_pass_a_zero_reference(self):
    # The candidate is within atol of the all-zero reference, but atol only
    # fills tolerance_headroom. It must not change the CONTRACT_SQNR verdict.
    report = numerical_validator.validate_kernels(
        _zeros,
        _tiny,
        shapes=(64, 64),
        dtype_str="float32",
        regimes="normal",
        contract=_SQNR,
        atol=1e-3,
    )
    self.assertFalse(report.is_numerically_equivalent)
    first = _sqnr(report.batch_results[0])
    self.assertEqual(
        (first["sqnr_status"], first["passed"]),
        (sqnr.STATUS_ZERO_SIGNAL, False),
    )
    self.assertEqual(report.tolerance_headroom["regimes_over_tolerance"], [])

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

  def test_shared_non_finite_with_finite_oracle_fails(self):
    def ref(inputs: Any) -> np.ndarray:
      out = _ref(inputs).copy()
      out[0, 0], out[1, 1] = np.inf, np.nan
      return out

    # The candidate equals the reference, so only the oracle sees that the
    # Inf and NaN elements are wrong.
    report = numerical_validator.validate_kernels(
        ref,
        ref,
        kernel_oracle=_oracle,
        shapes=(64, 64),
        dtype_str="float32",
        regimes="normal",
        contract=_SQNR,
    )
    self.assertFalse(report.is_numerically_equivalent, report.summary_message)
    first = _sqnr(report.batch_results[0])
    self.assertEqual(
        (first["sqnr_status"], first["passed"]),
        (sqnr.STATUS_NON_FINITE_MISMATCH, False),
    )
    self.assertIn("oracle_non_finite_mismatch", first["failed_reasons"][0])

  def test_oracle_judges_float_output_of_integer_run(self):
    def quarter(inputs: Any) -> np.ndarray:
      return np.asarray(inputs, dtype=np.float64) * 0.25

    # The reference and the candidate agree, and both are 6 dB from the
    # oracle. Only the oracle check can fail the run.
    report = numerical_validator.validate_kernels(
        _dequant,
        _dequant,
        kernel_oracle=quarter,
        shapes=(64, 64),
        dtype_str="int32",
        regimes="normal",
        contract=_SQNR,
    )
    self.assertFalse(report.is_numerically_equivalent, report.summary_message)
    first = _sqnr(report.batch_results[0])
    self.assertLess(first["oracle_sqnr_db"], 10.0)
    self.assertIn("reference_below_floor", first["failed_reasons"][0])

  @parameterized.named_parameters(
      ("integer_run", _SQNR, "int32", False, "AGREEMENT_AND_ORACLE"),
      ("non_finite_outputs", _SQNR, "float32", True, "AGREEMENT_AND_ORACLE"),
      (
          "integer_run_auto_oracle",
          _SQNR,
          "int32",
          False,
          "AGREEMENT_ONLY",
          numerical_validator.ORACLE_AUTO,
      ),
      (
          "non_finite_outputs_auto_oracle",
          _SQNR,
          "float32",
          True,
          "AGREEMENT_ONLY",
          numerical_validator.ORACLE_AUTO,
      ),
      (
          "integer_run_ulp",
          numerical_validator.CONTRACT_ULP,
          "int32",
          False,
          "AGREEMENT_ONLY",
      ),
      (
          "non_finite_outputs_ulp",
          numerical_validator.CONTRACT_ULP,
          "float32",
          True,
          "AGREEMENT_ONLY",
      ),
  )
  def test_sqnr_oracle_checks_set_the_correctness_basis(
      self,
      contract: str,
      dtype_str: str,
      non_finite: bool,
      basis: str,
      oracle: Any = _oracle,
  ):
    def kernel(inputs: Any) -> np.ndarray:
      out = _dequant(inputs)
      if non_finite:
        out[0, 0] = np.inf
      return out

    # The ULP oracle statistics need finite outputs of a float run, so here
    # the oracle runs only for the SQNR checks of CONTRACT_SQNR. Those checks
    # gate the verdict. Under another contract the oracle does not run. An
    # ORACLE_AUTO oracle does not count: in an integer run it re-runs the
    # reference on the same inputs, and nothing audits its precision.
    report = numerical_validator.validate_kernels(
        kernel,
        kernel,
        kernel_oracle=oracle,
        shapes=(64, 64),
        dtype_str=dtype_str,
        regimes="normal",
        contract=contract,
    )
    self.assertEqual(
        (report.correctness_basis, report.oracle_audit), (basis, None)
    )

  @parameterized.named_parameters(
      ("ulp", numerical_validator.CONTRACT_ULP, {}),
      ("bitwise", numerical_validator.CONTRACT_BITWISE, {}),
      (
          "row_rel_l2",
          numerical_validator.CONTRACT_ROW_REL_L2,
          {"rel_l2_tol": 0.05},
      ),
  )
  def test_sqnr_leaf_failure_under_another_run_contract_gives_its_reason(
      self, contract: str, gate_kwargs: dict[str, Any]
  ):
    def cand(inputs: Any) -> tuple[np.ndarray, np.ndarray]:
      return _dequant(inputs), _dequant_half(inputs)

    report = numerical_validator.validate_kernels(
        _tree_dequant,
        cand,
        shapes=(64, 64),
        dtype_str="float32",
        regimes="normal",
        contract=contract,
        contract_by_leaf={"[1]": _SQNR},
        **gate_kwargs,
    )
    self.assertFalse(report.is_numerically_equivalent)
    reason = _sqnr(_leaves(report.batch_results[0])["[1]"])["failed_reasons"][0]
    self.assertIn(
        f"output leaf [1] (contract: sqnr): {reason}", report.summary_message
    )

  @parameterized.named_parameters(
      ("bfloat16", "bfloat16"),
      ("float8_e4m3fn", "float8_e4m3fn"),
      ("float8_e5m2", "float8_e5m2"),
  )
  def test_widened_narrow_output_keeps_its_dtype(self, dtype_name: str):
    dtype = np.dtype(dtype_name)
    ref = np.random.default_rng(0).standard_normal((64, 64)).astype(dtype)
    # A PyTorch bfloat16 or float8 tensor reaches the SQNR report as float32
    # with its original dtype name, as `rel_l2.as_host_array` returns it.
    widened = ref.astype(np.float32)
    with mock.patch.object(
        numerical_validator.sqnr,
        "audit_leaf",
        autospec=True,
        wraps=sqnr.audit_leaf,
    ) as audit_leaf:
      report = numerical_validator._sqnr_report(  # pylint: disable=protected-access
          widened,
          widened.copy(),
          dtype_name,
          dtype_name,
          contract=_SQNR,
          oracle=None,
          config=None,
          path="",
      )
    ((pair_ref, pair_cand),) = audit_leaf.call_args.args[0]
    self.assertEqual((pair_ref.dtype, pair_cand.dtype), (dtype, dtype))
    np.testing.assert_array_equal(pair_ref, ref)
    assert report is not None
    self.assertEqual(report["dtype"], dtype_name)

  def test_widened_torch_output_gets_the_numpy_report(self):
    x = np.random.default_rng(0).standard_normal((128, 256))
    ref = x.astype(ml_dtypes.bfloat16)
    cand = x.astype(np.float16).astype(ml_dtypes.bfloat16)
    expected = numerical_validator.validate_arrays(
        cand, ref, dtype_str="bfloat16", contract=_SQNR
    )
    as_host_array = numerical_validator.rel_l2.as_host_array

    def widen(value: Any) -> tuple[np.ndarray, str]:
      # What `as_host_array` returns for a torch.bfloat16 tensor.
      arr, name = as_host_array(value)
      return arr.astype(np.float32), name

    with mock.patch.object(
        numerical_validator.rel_l2,
        "as_host_array",
        autospec=True,
        side_effect=widen,
    ):
      widened = numerical_validator.validate_arrays(
          cand, ref, dtype_str="bfloat16", contract=_SQNR
      )
    self.assertEqual(_sqnr(widened), _sqnr(expected))
    self.assertEqual(_sqnr(widened)["dtype"], "bfloat16")

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

  @parameterized.named_parameters(
      ("int64_vs_int32", np.int64, np.int32),
      ("bool_vs_int32", np.bool_, np.int32),
      ("float32_vs_int32", np.float32, np.int32),
  )
  def test_dtype_mismatch_with_equal_values_fails(self, ref_dtype, cand_dtype):
    def ref(inputs: Any) -> np.ndarray:
      return (_argmax(inputs) % 2).astype(ref_dtype)

    def cand(inputs: Any) -> np.ndarray:
      return ref(inputs).astype(cand_dtype)

    report = numerical_validator.validate_kernels(
        ref,
        cand,
        shapes=(16, 16),
        dtype_str="float32",
        regimes="normal",
        contract=_SQNR,
    )
    self.assertFalse(report.is_numerically_equivalent)
    self.assertEqual(
        _sqnr(report.batch_results[0])["sqnr_status"],
        sqnr.STATUS_DTYPE_MISMATCH,
    )
    self.assertIn("dtype mismatch", report.summary_message)

  @parameterized.named_parameters(
      ("ulp", numerical_validator.CONTRACT_ULP, True),
      ("bitwise", numerical_validator.CONTRACT_BITWISE, False),
  )
  def test_dtype_mismatch_report_does_not_gate_other_contracts(
      self, contract: str, expected: bool
  ):
    def cand(inputs: Any) -> np.ndarray:
      return _argmax(inputs).astype(np.int32)

    report = numerical_validator.validate_kernels(
        _argmax,
        cand,
        shapes=(16, 16),
        dtype_str="float32",
        regimes="normal",
        contract=contract,
    )
    # ULP distance is 0, so only the bitwise dtype rule fails the run.
    self.assertEqual(
        report.is_numerically_equivalent, expected, report.summary_message
    )
    self.assertEqual(
        _sqnr(report.batch_results[0])["sqnr_status"],
        sqnr.STATUS_DTYPE_MISMATCH,
    )
    self.assertNotIn("contract: sqnr", report.summary_message)

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

  @parameterized.named_parameters(
      ("int32_near", "int32", _dequant_near, True),
      ("int32_half", "int32", _dequant_half, False),
      ("int8_near", "int8", _dequant_near, True),
      ("int8_half", "int8", _dequant_half, False),
  )
  def test_float_output_of_integer_run_gets_sqnr(
      self, dtype_str: str, candidate: Callable[..., Any], expected: bool
  ):
    # The output arrays, not dtype_str, decide if SQNR applies. Without SQNR,
    # the near candidate would fail the exact-match fallback.
    report = numerical_validator.validate_kernels(
        _dequant,
        candidate,
        shapes=(64, 64),
        dtype_str=dtype_str,
        regimes="normal",
        contract=_SQNR,
    )
    self.assertEqual(
        report.is_numerically_equivalent, expected, report.summary_message
    )
    first = _sqnr(report.batch_results[0])
    self.assertEqual(
        (first["sqnr_status"], first["passed"], report.sqnr_summary["passed"]),
        (sqnr.STATUS_OK, expected, expected),
    )

  def test_summary_merges_only_sqnr_leaves(self):
    def cand(inputs: Any) -> tuple[np.ndarray, np.ndarray]:
      return _noisy(inputs), _bf16_rounded(inputs)

    report = numerical_validator.validate_kernels(
        _tree_dequant,
        cand,
        shapes=(64, 64),
        dtype_str="float32",
        regimes="normal",
        contract=_SQNR,
        contract_by_leaf={"[0]": numerical_validator.CONTRACT_ROW_REL_L2},
        rel_l2_tol=0.05,
    )
    self.assertTrue(report.is_numerically_equivalent, report.summary_message)
    leaves = _leaves(report.batch_results[0])
    summary = report.sqnr_summary
    self.assertEqual(
        (
            _sqnr(leaves["[0]"])["passed"],
            summary["passed"],
            summary["reports"],
            summary["worst_sqnr_leaf"],
        ),
        (False, True, 1, "[1]"),
    )

  def test_failure_of_non_sqnr_leaf_names_its_contract(self):
    def cand(inputs: Any) -> tuple[np.ndarray, np.ndarray]:
      return _dequant_half(inputs), _bf16_rounded(inputs)

    report = numerical_validator.validate_kernels(
        _tree_dequant,
        cand,
        shapes=(64, 64),
        dtype_str="float32",
        regimes="normal",
        contract=_SQNR,
        contract_by_leaf={"[0]": numerical_validator.CONTRACT_ULP},
    )
    self.assertFalse(report.is_numerically_equivalent)
    # Leaf [0] fails ULP and its report-only SQNR. Only ULP decides it, so the
    # summary must not give the SQNR reasons as the cause.
    self.assertFalse(_sqnr(_leaves(report.batch_results[0])["[0]"])["passed"])
    self.assertIn(
        "output leaf [0], failed its ulp contract.", report.summary_message
    )
    self.assertNotIn("sqnr_db", report.summary_message)

  def test_failure_of_discrete_non_sqnr_leaf_names_its_contract(self):
    def ref(inputs: Any) -> tuple[np.ndarray, np.ndarray]:
      return _argmax(inputs), _dequant(inputs)

    def cand(inputs: Any) -> tuple[np.ndarray, np.ndarray]:
      return _argmax_shifted(inputs), _bf16_rounded(inputs)

    report = numerical_validator.validate_kernels(
        ref,
        cand,
        shapes=(64, 64),
        dtype_str="float32",
        regimes="normal",
        contract=_SQNR,
        contract_by_leaf={"[0]": numerical_validator.CONTRACT_BITWISE},
    )
    self.assertFalse(report.is_numerically_equivalent)
    self.assertIn(
        "output leaf [0], failed its bitwise contract.", report.summary_message
    )

  def test_summary_merges_only_sqnr_leaves_under_another_run_contract(self):
    def cand(inputs: Any) -> tuple[np.ndarray, np.ndarray]:
      return _noisy(inputs), _bf16_rounded(inputs)

    report = numerical_validator.validate_kernels(
        _tree_dequant,
        cand,
        shapes=(64, 64),
        dtype_str="float32",
        regimes="normal",
        contract=numerical_validator.CONTRACT_ROW_REL_L2,
        contract_by_leaf={"[1]": _SQNR},
        rel_l2_tol=0.05,
    )
    self.assertTrue(report.is_numerically_equivalent, report.summary_message)
    summary = report.sqnr_summary
    self.assertEqual(
        (summary["passed"], summary["reports"], summary["worst_sqnr_leaf"]),
        (True, 1, "[1]"),
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

  def test_tool_replay_keeps_the_stored_auto_floor(self):
    tmp = self.create_tempdir().full_path
    config = sqnr.SqnrConfig.six_gate_v1(min_sqnr_db=sqnr.AUTO)
    report = _validate(_half_scale, sqnr_config=config, dump_failures_to=tmp)
    dump = report.failure_dumps[0]
    # Only sqnr_strict changes. The stored gate math must not act as a preset
    # switch that replaces the AUTO floor with the fixed 45 dB.
    again = json.loads(
        xparity.replay(dump, _ref, _half_scale, sqnr_strict=False)
    )
    self.assertEqual(
        again["run_config"]["sqnr_config"],
        dataclasses.asdict(dataclasses.replace(config, strict=False)),
    )

  def test_tool_replay_reads_a_partial_stored_config(self):
    tmp = self.create_tempdir().full_path
    report = _validate(
        _half_scale,
        sqnr_config=sqnr.SqnrConfig(min_sqnr_db=10.0, strict=False),
        dump_failures_to=tmp,
    )
    dump = report.failure_dumps[0]
    # A dump from another version: keys are missing and one key is unknown.
    info = xparity.read_suite_metadata(dump)
    info["sqnr_config"] = {"min_sqnr_db": 10.0, "unknown_field": 1}
    partial = f"{tmp}/partial.npz"
    xparity.save_test_suite(
        xparity.load_test_suite(dump), partial, extra_metadata=info
    )
    again = json.loads(
        xparity.replay(partial, _ref, _half_scale, sqnr_alpha=1e-4)
    )
    self.assertEqual(
        again["run_config"]["sqnr_config"],
        dataclasses.asdict(sqnr.SqnrConfig(min_sqnr_db=10.0, alpha=1e-4)),
    )
    self.assertFalse(again["is_numerically_equivalent"])


class SqnrInputDeltaTest(parameterized.TestCase):
  """CONTRACT_SQNR gives the kernel inputs to the input-delta check."""

  def test_passthrough_on_residual_kernel_fails(self):
    report = numerical_validator.validate_kernels(
        _residual,
        _passthrough,
        shapes=(64, 64),
        dtype_str="float32",
        regimes="normal",
        contract=_SQNR,
    )
    self.assertFalse(report.is_numerically_equivalent)
    first = _sqnr(report.batch_results[0])
    # The global SQNR passes. Only the input-delta check fails.
    self.assertGreater(first["sqnr_db"], first["threshold_db"])
    self.assertIsNotNone(first["min_delta_sqnr_db"])
    self.assertLen(first["failed_reasons"], 1)
    self.assertIn("min_delta_sqnr_db", first["failed_reasons"][0])

  def test_passthrough_passes_without_input_views(self):
    # validate_arrays has no kernel inputs, so the global SQNR decides.
    rng = np.random.default_rng(0)
    x = rng.standard_normal((64, 64)).astype(np.float32)
    res = numerical_validator.validate_arrays(
        _passthrough(x), _residual(x), dtype_str="float32", contract=_SQNR
    )
    self.assertTrue(res.passed)
    self.assertIsNone(_sqnr(res)["min_delta_sqnr_db"])

  def test_honest_residual_kernel_passes(self):
    report = numerical_validator.validate_kernels(
        _residual,
        _residual_f32,
        shapes=(64, 64),
        dtype_str="float32",
        regimes="normal",
        contract=_SQNR,
    )
    self.assertTrue(report.is_numerically_equivalent, report.summary_message)
    self.assertIsNotNone(_sqnr(report.batch_results[0])["min_delta_sqnr_db"])

  @parameterized.named_parameters(
      ("ulp", numerical_validator.CONTRACT_ULP, {}, False),
      ("bitwise", numerical_validator.CONTRACT_BITWISE, {}, False),
      (
          "row_rel_l2",
          numerical_validator.CONTRACT_ROW_REL_L2,
          {"rel_l2_tol": 1e-2},
          True,
      ),
  )
  def test_report_only_contracts_get_no_input_views(
      self, contract: str, extra: dict[str, Any], expected: bool
  ):
    with mock.patch.object(
        numerical_validator.sqnr,
        "audit_leaf",
        autospec=True,
        side_effect=sqnr.audit_leaf,
    ) as audit:
      report = numerical_validator.validate_kernels(
          _residual,
          _passthrough,
          shapes=(64, 64),
          dtype_str="float32",
          regimes="normal",
          contract=contract,
          **extra,
      )
    self.assertEqual(
        report.is_numerically_equivalent, expected, report.summary_message
    )
    self.assertNotEmpty(audit.call_args_list)
    for call in audit.call_args_list:
      self.assertEqual(call.kwargs["input_views"], ())
    first = _sqnr(report.batch_results[0])
    self.assertEqual(
        (first["passed"], first["min_delta_sqnr_db"]), (True, None)
    )

  def test_contract_by_leaf_gets_input_views(self):
    report = numerical_validator.validate_kernels(
        _tree_residual,
        _tree_passthrough,
        shapes=(64, 64),
        dtype_str="float32",
        regimes="normal",
        contract_by_leaf={"[1]": _SQNR},
    )
    self.assertFalse(report.is_numerically_equivalent)
    leaves = _leaves(report.batch_results[0])
    self.assertIsNone(leaves["[0]"].sqnr)
    self.assertIn(
        "min_delta_sqnr_db", _sqnr(leaves["[1]"])["failed_reasons"][0]
    )

  def test_replay_keeps_the_input_delta_check(self):
    tmp = self.create_tempdir().full_path
    report = numerical_validator.validate_kernels(
        _residual,
        _passthrough,
        shapes=(64, 64),
        dtype_str="float32",
        regimes="normal",
        contract=_SQNR,
        dump_failures_to=tmp,
    )
    self.assertNotEmpty(report.failure_dumps)
    again = numerical_validator.replay_failure_dump(
        report.failure_dumps[0], _residual, _passthrough
    )
    self.assertFalse(again.is_numerically_equivalent)
    self.assertIn(
        "min_delta_sqnr_db",
        _sqnr(again.batch_results[0])["failed_reasons"][0],
    )
    fixed = numerical_validator.replay_failure_dump(
        report.failure_dumps[0], _residual, _residual_f32
    )
    self.assertTrue(fixed.is_numerically_equivalent, fixed.summary_message)

  def test_input_views_keep_same_size_inputs(self):
    # pylint: disable=protected-access
    out = np.zeros((32, 64), np.float32)
    same = np.ones((32, 64), np.float32)
    swapped = np.arange(64 * 32, dtype=np.float32).reshape(64, 32)
    flat = np.arange(32 * 64, dtype=np.float32)
    skipped = (
        np.ones((8, 8), np.float32),
        np.ones((32, 64), np.bool_),
        np.ones((32, 64), np.complex64),
        "name",
        3.0,
    )
    views = numerical_validator._sqnr_input_views(
        numerical_validator._SqnrInputs(
            (same, swapped, *skipped), {"flat": flat}
        ),
        out,
    )
    np.testing.assert_array_equal(views[0][0], same)
    np.testing.assert_array_equal(views[1][0], swapped.T)
    np.testing.assert_array_equal(views[2][0], swapped.reshape(32, 64))
    np.testing.assert_array_equal(views[3][0], flat.reshape(32, 64))
    self.assertLen(views, 4)
    for view in views:
      self.assertEqual((len(view), view[0].shape), (1, (32, 64)))

  def test_unreadable_input_is_skipped(self):
    x = np.random.default_rng(0).standard_normal((64, 64)).astype(np.float32)
    suite = [{
        "name": "unreadable",
        "args": (x, _Unreadable()),
        "kwargs": {},
        "regime": "normal",
    }]
    with self.assertLogs(level="WARNING") as logs:
      report = numerical_validator.validate_kernels(
          lambda x, _: _residual(x),
          lambda x, _: _passthrough(x),
          shapes=(64, 64),
          dtype_str="float32",
          test_suite=suite,
          contract=_SQNR,
      )
    self.assertIn("RuntimeError: buffer was deleted", "\n".join(logs.output))
    self.assertFalse(report.is_numerically_equivalent)
    self.assertIn(
        "min_delta_sqnr_db",
        _sqnr(report.batch_results[0])["failed_reasons"][0],
    )

  def test_nested_inputs_are_found(self):
    x = np.random.default_rng(0).standard_normal((64, 64)).astype(np.float32)
    suite = [{
        "name": "nested",
        "args": ({"x": x, "scale": 2.0},),
        "kwargs": {},
        "regime": "normal",
    }]
    report = numerical_validator.validate_kernels(
        lambda d: _residual(d["x"]),
        lambda d: _passthrough(d["x"]),
        shapes=(64, 64),
        dtype_str="float32",
        test_suite=suite,
        contract=_SQNR,
    )
    self.assertFalse(report.is_numerically_equivalent)
    self.assertIn(
        "min_delta_sqnr_db",
        _sqnr(report.batch_results[0])["failed_reasons"][0],
    )

  def test_inputs_are_copied_once_per_batch(self):
    x = np.random.default_rng(0).standard_normal((64, 64)).astype(np.float32)
    suite = [{"name": "tree", "args": (x,), "kwargs": {}, "regime": "normal"}]
    host_lib = numerical_validator.rel_l2
    with mock.patch.object(
        host_lib,
        "as_host_array",
        autospec=True,
        side_effect=host_lib.as_host_array,
    ) as as_host:
      # The candidate returns copies, so that only the input is `x`.
      report = numerical_validator.validate_kernels(
          lambda x: (_residual(x), _residual(x)),
          lambda x: (np.array(x), np.array(x)),
          shapes=(64, 64),
          dtype_str="float32",
          test_suite=suite,
          contract=_SQNR,
      )
    self.assertFalse(report.is_numerically_equivalent)
    self.assertLen([c for c in as_host.call_args_list if c.args[0] is x], 1)
    leaves = _leaves(report.batch_results[0])
    self.assertLen(leaves, 2)
    for leaf in leaves.values():
      self.assertIn("min_delta_sqnr_db", _sqnr(leaf)["failed_reasons"][0])


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

  @parameterized.named_parameters(
      ("bf16_rounded", lambda x: x.astype(ml_dtypes.bfloat16), True),
      ("half_scale", lambda x: x * np.float32(0.5), False),
  )
  def test_float_arrays_of_integer_run_get_sqnr(
      self, make_actual: Callable[[np.ndarray], Any], expected: bool
  ):
    actual = np.asarray(make_actual(self.ref)).astype(np.float32)
    res = numerical_validator.validate_arrays(
        actual, self.ref, dtype_str="int32", contract=_SQNR
    )
    self.assertEqual(
        (res.passed, _sqnr(res)["sqnr_status"], _sqnr(res)["passed"]),
        (expected, sqnr.STATUS_OK, expected),
    )

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

  @parameterized.named_parameters(
      ("int64_vs_int32", np.int64, np.int32, "int32"),
      ("bool_vs_int32", np.bool_, np.int32, "int32"),
      ("float32_vs_int32", np.float32, np.int32, "float32"),
  )
  def test_dtype_mismatch_with_equal_values_fails(
      self, actual_dtype, expected_dtype, dtype_str: str
  ):
    expected = (np.arange(16) % 2).astype(expected_dtype)
    res = numerical_validator.validate_arrays(
        expected.astype(actual_dtype),
        expected,
        dtype_str=dtype_str,
        contract=_SQNR,
    )
    self.assertEqual(
        (res.passed, _sqnr(res)["sqnr_status"]),
        (False, sqnr.STATUS_DTYPE_MISMATCH),
    )

  def test_sqnr_passed_requires_equal_dtypes(self):
    # The public paths no longer reach this case, so call the helper directly.
    sqnr_passed = numerical_validator._sqnr_passed  # pylint: disable=protected-access
    ref = np.arange(4, dtype=np.int32)
    self.assertEqual(
        (
            sqnr_passed(None, ref.copy(), ref),
            sqnr_passed(None, ref.astype(np.int64), ref),
        ),
        (True, False),
    )


def _fft(inputs: Any) -> np.ndarray:
  """A complex64 output from a float input, such as an FFT kernel."""
  return np.fft.fft(np.asarray(inputs, dtype=np.float32)).astype(np.complex64)


def _fft_near(inputs: Any) -> np.ndarray:
  """_fft with a relative error of 2**-10, about 60 dB."""
  return _fft(inputs) * np.complex64(1.0 + 2.0**-10)


def _fft_oracle(inputs: Any) -> np.ndarray:
  return np.fft.fft(np.asarray(inputs, dtype=np.float64))


def _fft_imag_nan(inputs: Any) -> np.ndarray:
  """_fft with a NaN imaginary part at [0, 0]."""
  out = _fft(inputs)
  out[0, 0] = complex(1.0, np.nan)
  return out


def _dequant_nan(inputs: Any) -> np.ndarray:
  """_dequant with a NaN at [0, 0]."""
  out = _dequant(inputs)
  out[0, 0] = np.nan
  return out


def _complex_arrays() -> tuple[np.ndarray, np.ndarray]:
  """Returns a complex64 reference and a candidate about 60 dB from it."""
  rng = np.random.default_rng(0)
  ref = rng.standard_normal((16, 8)) + 1j * rng.standard_normal((16, 8))
  ref = ref.astype(np.complex64)
  return ref * np.complex64(1.0 + 2.0**-10), ref


class SqnrDiagnosticsTest(parameterized.TestCase):
  """The ULP and non-finite diagnostics of SQNR leaves follow the outputs."""

  def test_float_output_of_integer_run_reports_non_finite_values(self):
    report = numerical_validator.validate_kernels(
        _dequant_nan,
        _dequant_nan,
        shapes=(64, 64),
        dtype_str="int32",
        regimes="normal",
        contract=_SQNR,
    )
    first = report.batch_results[0]
    self.assertEqual(
        (first.has_nan_or_inf, first.nan_count, first.first_non_finite_index),
        (True, 1, (0, 0)),
    )

  @parameterized.named_parameters(
      ("sqnr", _SQNR, True),
      ("ulp", numerical_validator.CONTRACT_ULP, False),
  )
  def test_float_output_of_integer_run_measures_float_ulps(
      self, contract: str, float_ulps: bool
  ):
    report = numerical_validator.validate_kernels(
        _dequant,
        _dequant_near,
        shapes=(64, 64),
        dtype_str="int32",
        regimes="normal",
        contract=contract,
    )
    # A relative error of 2**-10 is 2**13 float32 ULPs. Other contracts keep
    # the integer distance of the run dtype.
    self.assertEqual(
        report.batch_results[0].max_ulp_distance > 1000, float_ulps
    )

  def test_float_arrays_of_integer_run_report_non_finite_values(self):
    ref = np.random.default_rng(0).standard_normal((64, 32)).astype(np.float32)
    ref[0, 0] = np.nan
    res = numerical_validator.validate_arrays(
        ref * np.float32(1.0 + 2.0**-10),
        ref,
        dtype_str="int32",
        contract=_SQNR,
    )
    self.assertEqual(
        (res.has_nan_or_inf, res.nan_count, res.first_non_finite_index),
        (True, 1, (0, 0)),
    )
    assert res.finite_max_ulp is not None
    self.assertGreater(res.finite_max_ulp, 1000)

  @parameterized.named_parameters(
      ("complex_run", "complex64"),
      ("float_run", "float32"),
  )
  def test_complex_arrays(self, dtype_str: str):
    actual, expected = _complex_arrays()
    with warnings.catch_warnings():
      warnings.simplefilter("error", np.exceptions.ComplexWarning)
      res = numerical_validator.validate_arrays(
          actual, expected, dtype_str=dtype_str, contract=_SQNR
      )
    self.assertTrue(res.passed)
    self.assertTrue(_sqnr(res)["passed"])
    # The diagnostics compare the real and imaginary parts on a trailing axis.
    assert res.worst_offender is not None
    self.assertLen(res.worst_offender.max_ulp_index, 3)
    self.assertGreater(res.max_ulp_distance, 1000)

  def test_complex_arrays_report_imaginary_nan(self):
    actual, expected = _complex_arrays()
    actual[3, 4] = expected[3, 4] = complex(1.0, np.nan)
    with warnings.catch_warnings():
      warnings.simplefilter("error", np.exceptions.ComplexWarning)
      res = numerical_validator.validate_arrays(
          actual, expected, dtype_str="complex64", contract=_SQNR
      )
    self.assertEqual(
        (res.has_nan_or_inf, res.nan_count, res.first_non_finite_index),
        (True, 1, (3, 4, 1)),
    )

  def test_complex_kernel_output(self):
    with warnings.catch_warnings():
      warnings.simplefilter("error", np.exceptions.ComplexWarning)
      report = numerical_validator.validate_kernels(
          _fft,
          _fft_near,
          kernel_oracle=_fft_oracle,
          shapes=(64, 64),
          dtype_str="float32",
          regimes="normal",
          contract=_SQNR,
      )
    self.assertTrue(report.is_numerically_equivalent, report.summary_message)
    self.assertGreater(report.batch_results[0].max_ulp_distance, 1000)
    # A complex128 oracle runs in float64.
    assert report.oracle_audit is not None
    self.assertTrue(report.oracle_audit.oracle_executed_in_float64)

  def test_complex_kernel_output_reports_imaginary_nan(self):
    with warnings.catch_warnings():
      warnings.simplefilter("error", np.exceptions.ComplexWarning)
      report = numerical_validator.validate_kernels(
          _fft_imag_nan,
          _fft_imag_nan,
          shapes=(64, 64),
          dtype_str="float32",
          regimes="normal",
          contract=_SQNR,
      )
    first = report.batch_results[0]
    self.assertEqual(
        (first.has_nan_or_inf, first.nan_count, first.first_non_finite_index),
        (True, 1, (0, 0, 1)),
    )

  @parameterized.product(
      dtype_name=("float8_e8m0fnu", "float8_e4m3fnuz"),
      scale=(1.0, 2.0),
  )
  def test_unsupported_float_output_of_integer_run(
      self, dtype_name: str, scale: float
  ):
    dtype = np.dtype(getattr(ml_dtypes, dtype_name))

    def ref(inputs: Any) -> np.ndarray:
      # Positive values from 1 to 64, which both dtypes hold.
      return (np.abs(np.asarray(inputs, dtype=np.float32)) % 64 + 1).astype(
          dtype
      )

    def cand(inputs: Any) -> np.ndarray:
      return (ref(inputs).astype(np.float32) * np.float32(scale)).astype(dtype)

    report = numerical_validator.validate_kernels(
        ref,
        cand,
        shapes=(64, 64),
        dtype_str="int32",
        regimes="normal",
        contract=_SQNR,
    )
    # The ULP code does not support these dtypes, so the ULP figures use
    # float32 units. The SQNR report still gives the verdict.
    first = report.batch_results[0]
    self.assertEqual(
        (_sqnr(first)["passed"], first.has_nan_or_inf), (scale == 1.0, False)
    )
    self.assertEqual(report.is_numerically_equivalent, scale == 1.0)
    expected = ref(np.arange(-8, 8, dtype=np.int32).reshape(4, 4))
    res = numerical_validator.validate_arrays(
        cand(np.arange(-8, 8, dtype=np.int32).reshape(4, 4)),
        expected,
        dtype_str="int32",
        contract=_SQNR,
    )
    self.assertEqual(
        (res.passed, _sqnr(res)["passed"], res.has_nan_or_inf),
        (scale == 1.0, scale == 1.0, False),
    )

  def test_unsupported_float_output_reports_non_finite_values(self):
    expected = np.array(
        [[1.0, 2.0], [4.0, 8.0]], dtype=ml_dtypes.float8_e8m0fnu
    )
    actual = expected.copy()
    actual[1, 0] = expected[1, 0] = np.nan
    res = numerical_validator.validate_arrays(
        actual, expected, dtype_str="int32", contract=_SQNR
    )
    self.assertEqual(
        (res.has_nan_or_inf, res.nan_count, res.first_non_finite_index),
        (True, 1, (1, 0)),
    )

  @parameterized.product(
      dtype_name=(
          "float8_e8m0fnu",
          "float8_e4m3fnuz",
          "float8_e5m2fnuz",
          "float6_e2m3fn",
      ),
      scale=(1.0, 0.5),
  )
  def test_unsupported_float_run_dtype(self, dtype_name: str, scale: float):
    dtype = np.dtype(getattr(ml_dtypes, dtype_name))
    # Powers of two from 0.5 to 4, which all four dtypes hold.
    x = np.random.default_rng(0).choice([1.0, 2.0, 4.0], size=(64, 64))
    x = x.astype(dtype)

    def cand(inputs: Any) -> np.ndarray:
      return (np.asarray(inputs, dtype=np.float32) * scale).astype(dtype)

    # The ULP code does not support these run dtypes, so the ULP figures use
    # float32 units. The SQNR report still gives the verdict.
    res = numerical_validator.validate_arrays(
        cand(x), x, dtype_str=dtype_name, contract=_SQNR
    )
    self.assertEqual(
        (res.passed, _sqnr(res)["passed"], res.has_nan_or_inf),
        (scale == 1.0, scale == 1.0, False),
    )
    suite = [{"name": "powers", "args": (x,), "kwargs": {}, "regime": "normal"}]
    report = numerical_validator.validate_kernels(
        lambda inputs: inputs,
        cand,
        shapes=(64, 64),
        dtype_str=dtype_name,
        test_suite=suite,
        contract=_SQNR,
    )
    first = report.batch_results[0]
    self.assertEqual(
        (_sqnr(first)["passed"], first.has_nan_or_inf), (scale == 1.0, False)
    )
    self.assertEqual(report.is_numerically_equivalent, scale == 1.0)
    # The automatic oracle measures its ULP figures in the same units.
    auto_report = numerical_validator.validate_kernels(
        lambda inputs: inputs,
        cand,
        shapes=(64, 64),
        dtype_str=dtype_name,
        test_suite=suite,
        contract=_SQNR,
        kernel_oracle="auto",
    )
    self.assertEqual(auto_report.is_numerically_equivalent, scale == 1.0)
    audit = auto_report.oracle_audit
    assert audit is not None
    self.assertEqual(audit.reference_max_ulp_from_oracle, 0)
    self.assertEqual(audit.candidate_max_ulp_from_oracle == 0, scale == 1.0)
    # The automatic oracle promotes these ml_dtypes floats to float64 too.
    self.assertTrue(audit.oracle_executed_in_float64)
    self.assertTrue(audit.oracle_precision_verified)

  @parameterized.parameters(
      "float8_e8m0fnu", "float8_e4m3fnuz", "float6_e2m3fn"
  )
  def test_auto_oracle_promotes_ml_dtypes_float_argument(
      self, scale_dtype: str
  ):
    # A ULP-contract run with integer codes and per-row scales in an ml_dtypes
    # float. The reference returns the product, so its dtype follows the
    # scales: float32 if the scales stay narrow, float64 if they are promoted.
    rng = np.random.default_rng(0)
    codes = rng.integers(-8, 8, size=(64, 64)).astype(np.int8)
    scales = np.full((64, 1), 2.0).astype(getattr(ml_dtypes, scale_dtype))

    def ref(q: np.ndarray, s: np.ndarray) -> np.ndarray:
      return q.astype(np.float32) * s

    def cand(q: np.ndarray, s: np.ndarray) -> np.ndarray:
      return q.astype(np.float32) * s.astype(np.float32)

    report = numerical_validator.validate_kernels(
        ref,
        cand,
        shapes=(64, 64),
        dtype_str="float32",
        test_suite=[{
            "name": "scaled",
            "args": (codes, scales),
            "kwargs": {},
            "regime": "normal",
        }],
        contract=numerical_validator.CONTRACT_ULP,
        kernel_oracle=numerical_validator.ORACLE_AUTO,
    )
    self.assertTrue(report.is_numerically_equivalent)
    audit = report.oracle_audit
    assert audit is not None
    self.assertEqual(
        (
            audit.oracle_executed_in_float64,
            audit.oracle_precision_verified,
            audit.reference_max_ulp_from_oracle,
            audit.candidate_max_ulp_from_oracle,
        ),
        (True, True, 0, 0),
    )

  @parameterized.parameters(
      ("float8_e4m3fnuz", 8),
      ("float8_e5m2fnuz", 4),
      ("float6_e2m3fn", 8),
  )
  def test_zero_ulp_probe_measures_unsupported_float_run_dtype(
      self, dtype_name: str, want_ulp: int
  ):
    dtype = np.dtype(getattr(ml_dtypes, dtype_name))
    x = np.random.default_rng(0).choice([1.0, 2.0, 4.0], size=(64, 64))
    x = x.astype(dtype)

    def lossy(inputs: np.ndarray) -> np.ndarray:
      # Adds 1/16 sixteen times. In the run dtype each step rounds back, so
      # the result stays at the input. In float64 the result is input + 1.
      wide = np.asarray(inputs).dtype == np.float64
      acc = np.asarray(inputs, dtype=np.float64 if wide else np.float32)
      for _ in range(16):
        acc = acc + 1 / 16
        if not wide:
          acc = acc.astype(dtype).astype(np.float32)
      return acc if wide else acc.astype(dtype)

    suite = [{"name": "powers", "args": (x,), "kwargs": {}, "regime": "normal"}]
    report = numerical_validator.validate_kernels(
        lossy,
        lossy,
        shapes=(64, 64),
        dtype_str=dtype_name,
        test_suite=suite,
        contract=_SQNR,
    )
    # The kernels agree exactly, but the reference is many steps of the run
    # dtype away from its float64 result. The probe reports it.
    self.assertTrue(report.is_numerically_equivalent)
    self.assertEqual(
        report.run_config.get("probed_baseline_oracle_ulp"), want_ulp
    )
    self.assertNotIn("zero_ulp_probe_skip_reason", report.run_config)
    self.assertStartsWith(
        report.summary_message, "⚠️ 0-ULP AGREEMENT WITH LOSSY BASELINE"
    )
    exact = numerical_validator.validate_kernels(
        lambda inputs: inputs,
        lambda inputs: inputs,
        shapes=(64, 64),
        dtype_str=dtype_name,
        test_suite=suite,
        contract=_SQNR,
    )
    self.assertNotIn("probed_baseline_oracle_ulp", exact.run_config)
    self.assertNotIn("zero_ulp_probe_skip_reason", exact.run_config)
    self.assertNotIn("0-ULP AGREEMENT", exact.summary_message)

  def test_zero_ulp_probe_records_why_it_was_skipped(self):
    x = np.random.default_rng(0).standard_normal((16, 16)).astype(np.float32)

    def ref(inputs: np.ndarray) -> np.ndarray:
      if np.asarray(inputs).dtype == np.float64:
        raise TypeError("float64 inputs are not supported")
      return inputs

    report = numerical_validator.validate_kernels(
        ref,
        lambda inputs: inputs,
        shapes=(16, 16),
        dtype_str="float32",
        test_suite=[
            {"name": "normal", "args": (x,), "kwargs": {}, "regime": "normal"}
        ],
    )
    self.assertTrue(report.is_numerically_equivalent)
    self.assertEqual(
        report.run_config.get("zero_ulp_probe_skip_reason"),
        "TypeError: float64 inputs are not supported",
    )

  def test_zero_ulp_probe_records_a_non_float64_oracle(self):
    x = np.random.default_rng(0).standard_normal((16, 16)).astype(np.float32)

    def ref(inputs: np.ndarray) -> np.ndarray:
      return np.asarray(inputs).astype(np.float32)

    report = numerical_validator.validate_kernels(
        ref,
        ref,
        shapes=(16, 16),
        dtype_str="float32",
        test_suite=[
            {"name": "normal", "args": (x,), "kwargs": {}, "regime": "normal"}
        ],
    )
    self.assertEqual(
        report.run_config.get("zero_ulp_probe_skip_reason"),
        "kernel_ref returned dtype 'float32' on float64 inputs, not float64.",
    )

  @parameterized.named_parameters(
      # Cancellation: 2**-6 is many steps from 0 in a one-byte float, but
      # small next to the output scale, as in compute_ulp_distance.
      (
          "near_zero_fnuz",
          "float8_e4m3fnuz",
          [64.0, -64.0, 2**-6],
          [64.0, -64.0, 0.0],
          1,
      ),
      (
          "near_zero_supported",
          "float8_e4m3fn",
          [64.0, -64.0, 2**-6],
          [64.0, -64.0, 0.0],
          1,
      ),
      # A fnuz reference that overflows to NaN against a finite oracle.
      (
          "overflow_to_nan",
          "float8_e4m3fnuz",
          [64.0, 1e6, 1.0],
          [64.0, 64.0, 1.0],
          255,
      ),
      # float6 has no NaN, so a NaN oracle rounds to -0.0. It is not a 0-step
      # match for a zero reference.
      (
          "nan_oracle",
          "float6_e2m3fn",
          [64.0, 0.0, 1.0],
          [64.0, np.nan, 1.0],
          63,
      ),
      # Two NaN values match.
      (
          "nan_both",
          "float8_e4m3fnuz",
          [64.0, np.nan, 1.0],
          [64.0, np.nan, 1.0],
          0,
      ),
  )
  def test_baseline_probe_ulp_one_byte_steps(
      self,
      dtype_name: str,
      values: list[float],
      oracle: list[float],
      want_ulp: int,
  ):
    reference = np.array(values).astype(getattr(ml_dtypes, dtype_name))
    probe = numerical_validator._baseline_probe_ulp  # pylint: disable=protected-access
    self.assertEqual(
        probe(reference, np.array(oracle), dtype_name), (want_ulp, dtype_name)
    )

  def test_baseline_probe_ulp_uses_run_dtype_steps_for_wide_output(self):
    # An fp8 kernel with a float32 output: float32 rounding is not a step of
    # the run dtype.
    oracle = np.array([1.0, 2.0, 3.0])
    reference = (oracle + 1e-7).astype(np.float32)
    probe = numerical_validator._baseline_probe_ulp  # pylint: disable=protected-access
    self.assertEqual(
        probe(reference, oracle, "float8_e4m3fnuz"), (0, "float8_e4m3fnuz")
    )
    # fnuz rounds overflow of both signs to its one NaN. Opposite signs must
    # still count as a full-grid difference.
    overflow = np.array([1e6, -1e6, 1e6], dtype=np.float32)
    self.assertEqual(
        probe(overflow, np.array([-1e6, 1e6, 1e6]), "float8_e4m3fnuz"),
        (255, "float8_e4m3fnuz"),
    )

  def test_baseline_probe_ulp_raises_without_a_unit(self):
    reference = np.zeros(4, dtype=np.complex64)
    probe = numerical_validator._baseline_probe_ulp  # pylint: disable=protected-access
    with self.assertRaisesRegex(ValueError, "No ULP unit for run dtype"):
      probe(reference, np.zeros(4), "complex64")

  def test_ulp_table_dtypes_give_zero_distance_on_equal_arrays(self):
    values = np.array([0.5, 1.0, 2.0], dtype=np.float32)
    for name in sorted(xparity.ulp.DTYPE_TOLERANCES):
      with self.subTest(name):
        arr = values.astype(getattr(ml_dtypes, name, None) or np.dtype(name))
        self.assertEqual(xparity.compute_ulp_distance(arr, arr, name).max(), 0)

  def test_summary_threshold_belongs_to_the_worst_leaf(self):
    def ref(inputs: Any) -> tuple[np.ndarray, np.ndarray]:
      out = _dequant(inputs)
      return out, out.astype(ml_dtypes.bfloat16)

    def cand(inputs: Any) -> tuple[np.ndarray, np.ndarray]:
      out = _dequant(inputs)
      noise = np.random.default_rng(1).standard_normal(out.shape)
      noisy = out + np.float32(4e-3 * np.std(out)) * noise.astype(np.float32)
      near = out * np.float32(1.0 + 2.0**-10)
      return noisy, near.astype(ml_dtypes.bfloat16)

    report = numerical_validator.validate_kernels(
        ref,
        cand,
        shapes=(64, 64),
        dtype_str="float32",
        regimes="normal",
        contract=_SQNR,
    )
    summary = report.sqnr_summary
    # Leaf [0] has the lowest SQNR and the 45 dB default floor. Leaf [1] has
    # the higher bfloat16 auto floor, which must not pair with leaf [0].
    self.assertEqual(
        (
            summary["worst_sqnr_leaf"],
            summary["threshold_db"],
            summary["threshold_source"],
            len(summary["thresholds"]),
        ),
        ("[0]", 45.0, "default", 2),
    )
    self.assertGreater(
        max(t["threshold_db"] for t in summary["thresholds"]), 45.0
    )
    self.assertIn("threshold 45.00 dB (default)", report.summary_message)

  def test_replay_with_an_oracle(self):
    tmp = self.create_tempdir().full_path
    # The reference and the candidate agree, and both are 6 dB from the
    # oracle. Only the oracle check can fail the run and its replay.
    report = numerical_validator.validate_kernels(
        _half_scale,
        _half_scale,
        kernel_oracle=_oracle,
        shapes=(64, 64),
        dtype_str="float32",
        regimes="normal",
        contract=_SQNR,
        dump_failures_to=tmp,
    )
    self.assertFalse(report.is_numerically_equivalent)
    dump = report.failure_dumps[0]
    again = numerical_validator.replay_failure_dump(
        dump, _half_scale, _half_scale, kernel_oracle=_oracle
    )
    self.assertEqual(
        (again.is_numerically_equivalent, again.correctness_basis),
        (False, "AGREEMENT_AND_ORACLE"),
    )
    self.assertLess(_sqnr(again.batch_results[0])["oracle_sqnr_db"], 10.0)
    without_oracle = numerical_validator.replay_failure_dump(
        dump, _half_scale, _half_scale
    )
    self.assertTrue(
        without_oracle.is_numerically_equivalent, without_oracle.summary_message
    )


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
