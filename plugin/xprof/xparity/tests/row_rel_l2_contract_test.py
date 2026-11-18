"""Tests for the row_rel_l2 contract in the validator, tool and CLI."""

from collections.abc import Callable, Sequence
import contextlib
import io
import json
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np

from xprof.xparity import numerical_validator
from xprof.xparity import rel_l2
from xprof.xparity import xparity_cli
from xprof.xparity import xparity_tool

_ROW = numerical_validator.CONTRACT_ROW_REL_L2
_BAD_ROW = 5


def _row(
    result: numerical_validator.BatchValidationResult,
) -> rel_l2.RowRelL2Comparison:
  """Returns the row_rel_l2 comparison of a batch or leaf result."""
  assert result.row_rel_l2 is not None
  return result.row_rel_l2


def _leaves(
    batch: numerical_validator.BatchValidationResult,
) -> dict[str, numerical_validator.BatchValidationResult]:
  assert batch.leaf_results is not None
  return batch.leaf_results


def _ref_kernel(inputs: Any) -> np.ndarray:
  return np.asarray(inputs, dtype=np.float32) * np.float32(0.5)


def _fp8_row_scaled_kernel(inputs: Any) -> np.ndarray:
  """_ref_kernel with each row round-tripped through scaled FP8 E4M3."""
  out = _ref_kernel(inputs)
  scale = np.max(np.abs(out), axis=-1, keepdims=True) / np.float32(448.0)
  scale = np.where(scale > 0, scale, np.float32(1.0)).astype(np.float32)
  quantized = (out / scale).astype(ml_dtypes.float8_e4m3fn)
  return quantized.astype(np.float32) * scale


def _bad_row_kernel(inputs: Any) -> np.ndarray:
  out = _ref_kernel(inputs).copy()
  out[_BAD_ROW] *= np.float32(1.5)
  return out


def _nan_kernel(inputs: Any) -> np.ndarray:
  out = _ref_kernel(inputs).copy()
  out[0, 0] = np.nan
  return out


def _short_kernel(inputs: Any) -> np.ndarray:
  return _ref_kernel(inputs)[..., :-1]


def _ref_bf16(inputs: Any) -> np.ndarray:
  return _ref_kernel(inputs).astype(ml_dtypes.bfloat16)


def _bad_bf16(inputs: Any) -> np.ndarray:
  return _bad_row_kernel(inputs).astype(ml_dtypes.bfloat16)


def _jax_ref(inputs: Any) -> jax.Array:
  return jnp.asarray(inputs, jnp.float32) * 0.5


def _jax_bad(inputs: Any) -> jax.Array:
  return _jax_ref(inputs).at[_BAD_ROW].multiply(1.5)


def _ref_tree(inputs: Any) -> dict[str, np.ndarray]:
  return {"idx": np.argmax(inputs, axis=-1), "out": _ref_kernel(inputs)}


def _cand_tree(inputs: Any) -> dict[str, np.ndarray]:
  return {
      "idx": np.argmax(inputs, axis=-1),
      "out": _fp8_row_scaled_kernel(inputs),
  }


def _validate(
    candidate: Callable[..., Any],
    *,
    shapes: Sequence[int] = (256, 128),
    **kwargs: Any,
) -> numerical_validator.KernelValidationReport:
  kwargs.setdefault("dtype_str", "float32")
  kwargs.setdefault("regimes", "normal")
  return numerical_validator.validate_kernels(
      _ref_kernel,
      candidate,
      shapes=shapes,
      contract=_ROW,
      rel_l2_tol=kwargs.pop("rel_l2_tol", 5e-2),
      **kwargs,
  )


class RowRelL2ContractTest(parameterized.TestCase):

  def test_lossy_fp8_candidate_passes_row_rel_l2(self):
    report = _validate(_fp8_row_scaled_kernel)
    self.assertTrue(report.is_numerically_equivalent, report.summary_message)
    self.assertIn("contract: row_rel_l2", report.summary_message)
    self.assertGreater(report.overall_max_ulp, 1000)

  def test_lossy_fp8_candidate_populates_rel_l2_summary_and_config(self):
    report = _validate(_fp8_row_scaled_kernel)
    summary = report.rel_l2_summary
    self.assertLess(summary["max_rel_l2"], 5e-2)
    self.assertGreater(summary["max_rel_l2"], 1e-3)
    self.assertEqual(
        (
            summary["failed_checks"],
            report.run_config["rel_l2_tol"],
            report.run_config["row_factor"],
        ),
        (0, 5e-2, 5.0),
    )

  def test_lossy_fp8_candidate_fails_ulp_contract(self):
    ulp_report = numerical_validator.validate_kernels(
        _ref_kernel,
        _fp8_row_scaled_kernel,
        shapes=(256, 128),
        dtype_str="float32",
        regimes="normal",
    )
    self.assertFalse(ulp_report.is_numerically_equivalent)

  def test_lossy_fp8_candidate_passes_on_every_regime(self):
    report = _validate(_fp8_row_scaled_kernel, regimes="all")
    self.assertTrue(report.is_numerically_equivalent, report.summary_message)
    self.assertGreater(report.total_batches_count, 1)

  def test_single_bad_row_fails_and_is_located(self):
    report = _validate(_bad_row_kernel)
    batch = report.batch_results[0]
    row = _row(batch)
    self.assertFalse(report.is_numerically_equivalent)
    self.assertLess(row.rel_l2, 5e-2)
    self.assertEqual(
        (batch.passed, row.reason, row.worst_row_index),
        (False, "row_rel_l2", (_BAD_ROW,)),
    )

  def test_single_bad_row_is_reported_in_summary(self):
    report = _validate(_bad_row_kernel)
    self.assertIn(f"row ({_BAD_ROW},)", report.summary_message)
    self.assertEqual(report.rel_l2_summary["worst_row_index"], [_BAD_ROW])

  def test_row_factor_and_floor_are_honored(self):
    self.assertTrue(
        _validate(_bad_row_kernel, row_factor=20.0).is_numerically_equivalent
    )

  def test_nan_candidate_fails(self):
    report = _validate(_nan_kernel, rel_l2_tol=1.0)
    self.assertFalse(report.is_numerically_equivalent)
    self.assertEqual(
        _row(report.batch_results[0]).reason, "non_finite_candidate"
    )

  def test_missing_tolerance_raises(self):
    with self.assertRaisesRegex(ValueError, "rel_l2_tol"):
      numerical_validator.validate_kernels(
          _ref_kernel, _ref_kernel, shapes=(8, 8), contract=_ROW
      )

  def test_invalid_row_factor_raises(self):
    with self.assertRaisesRegex(ValueError, "row_factor"):
      _validate(_ref_kernel, row_factor=0.0)

  @parameterized.named_parameters(
      ("matching_bf16", _ref_bf16, True, None),
      ("bad_row_bf16", _bad_bf16, False, "row_rel_l2"),
      # The output dtype must match the reference, as in the ULP contract's
      # bitwise detail.
      ("widened_f32", _ref_kernel, False, "dtype_mismatch"),
  )
  def test_bfloat16_outputs(
      self,
      candidate: Callable[..., Any],
      expected_equiv: bool,
      expected_reason: str | None,
  ):
    report = numerical_validator.validate_kernels(
        _ref_bf16,
        candidate,
        shapes=(256, 128),
        dtype_str="bfloat16",
        regimes="normal",
        contract=_ROW,
        rel_l2_tol=5e-2,
    )
    self.assertEqual(report.is_numerically_equivalent, expected_equiv)
    self.assertEqual(_row(report.batch_results[0]).reason, expected_reason)

  def test_jax_kernels(self):
    report = numerical_validator.validate_kernels(
        _jax_ref,
        _jax_bad,
        shapes=(256, 128),
        dtype_str="float32",
        regimes="normal",
        contract=_ROW,
        rel_l2_tol=5e-2,
    )
    self.assertFalse(report.is_numerically_equivalent)
    self.assertEqual(_row(report.batch_results[0]).worst_row_index, (_BAD_ROW,))

  def test_contract_by_leaf(self):
    report = numerical_validator.validate_kernels(
        _ref_tree,
        _cand_tree,
        shapes=(256, 128),
        dtype_str="float32",
        regimes="normal",
        contract_by_leaf={"['out']": _ROW},
        rel_l2_tol=5e-2,
    )
    self.assertTrue(report.is_numerically_equivalent, report.summary_message)
    leaves = _leaves(report.batch_results[0])
    self.assertEqual(
        (
            leaves["['idx']"].row_rel_l2,
            _row(leaves["['out']"]).passed,
            report.rel_l2_summary["worst_row_leaf"],
        ),
        (None, True, "['out']"),
    )

  def test_validate_arrays(self):
    ref = np.random.default_rng(0).standard_normal((64, 32)).astype(np.float32)
    bad = ref.copy()
    # Row 7 is off by 1.5x its norm (bound 5 * 0.25), while the whole output
    # stays near 1.5 / sqrt(64) = 0.19, under the 0.25 tolerance.
    bad[7] *= 2.5
    res = numerical_validator.validate_arrays(
        bad, ref, dtype_str="float32", contract=_ROW, rel_l2_tol=0.25
    )
    row = _row(res)
    self.assertEqual(
        (res.passed, row.reason, row.worst_row_index),
        (False, rel_l2.REASON_ROW, (7,)),
    )

  def test_validate_arrays_missing_tolerance_raises(self):
    ref = np.zeros((4, 4), dtype=np.float32)
    with self.assertRaisesRegex(ValueError, "rel_l2_tol"):
      numerical_validator.validate_arrays(ref, ref, contract=_ROW)


class RowRelL2ToolTest(absltest.TestCase):

  def test_compact_verdict_names_the_bad_row(self):
    verdict = json.loads(
        xparity_tool.verify_numerical_parity(
            _ref_kernel,
            _bad_row_kernel,
            shapes="(256, 128)",
            dtype_str="float32",
            regimes="normal",
            contract=_ROW,
            rel_l2_tol=5e-2,
            verdict_only=True,
        )
    )
    failure = verdict["failures"][0]
    self.assertEqual(
        {
            "equiv": verdict["is_numerically_equivalent"],
            "contract": verdict["contract"],
            "reason": failure["reason"],
            "worst_row_index": failure["worst_row_index"],
            "summary_worst_row": verdict["rel_l2"]["worst_row_index"],
        },
        {
            "equiv": False,
            "contract": _ROW,
            "reason": "row_rel_l2",
            "worst_row_index": [_BAD_ROW],
            "summary_worst_row": [_BAD_ROW],
        },
    )
    self.assertAlmostEqual(failure["worst_row_rel_l2"], 0.5, delta=1e-4)
    self.assertStartsWith(verdict["verdict"], "FAILED")

  def test_full_report_serializes_row_rel_l2(self):
    payload = json.loads(
        xparity_tool.verify_numerical_parity(
            _ref_kernel,
            _nan_kernel,
            shapes="(16, 16)",
            dtype_str="float32",
            regimes="normal",
            contract=_ROW,
            rel_l2_tol=1e-2,
        )
    )
    row = payload["batch_results"][0]["row_rel_l2"]
    self.assertEqual(row["reason"], "non_finite_candidate")
    self.assertIsNone(row["rel_l2"])  # NaN is serialized as null.
    self.assertEqual(row["first_non_finite_index"], [0, 0])

  def test_shape_mismatch(self):
    verdict = json.loads(
        xparity_tool.verify_numerical_parity(
            _ref_kernel,
            _short_kernel,
            shapes="(16, 16)",
            dtype_str="float32",
            contract=_ROW,
            rel_l2_tol=1e-2,
            verdict_only=True,
        )
    )
    self.assertEqual(verdict["correctness_basis"], "SHAPE_MISMATCH")

  def test_replay_keeps_and_overrides_the_tolerance(self):
    tmp = self.create_tempdir().full_path
    payload = json.loads(
        xparity_tool.verify_numerical_parity(
            _ref_kernel,
            _bad_row_kernel,
            shapes="(256, 128)",
            dtype_str="float32",
            regimes="normal",
            contract=_ROW,
            rel_l2_tol=5e-2,
            dump_failures_to=tmp,
        )
    )
    dump = payload["failure_dumps"][0]
    again = json.loads(xparity_tool.replay(dump, _ref_kernel, _bad_row_kernel))
    self.assertEqual(
        (again["is_numerically_equivalent"], again["run_config"]["rel_l2_tol"]),
        (False, 5e-2),
    )
    relaxed = json.loads(
        xparity_tool.replay(dump, _ref_kernel, _bad_row_kernel, row_factor=20.0)
    )
    self.assertTrue(relaxed["is_numerically_equivalent"])

  def test_cli_flags(self):
    module = __name__
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
      code = xparity_cli.main([
          "xparity_cli",
          "verify",
          f"--kernel_ref={module}._ref_kernel",
          f"--kernel_candidate={module}._fp8_row_scaled_kernel",
          "--shapes=(256, 128)",
          "--dtype_str=float32",
          "--regimes=normal",
          "--contract=row_rel_l2",
          "--rel_l2_tol=0.05",
          "--row_factor=5",
          "--row_norm_floor=0.1",
          "--verdict_only",
      ])
    self.assertEqual(code, xparity_cli.EXIT_OK)
    verdict = json.loads(stdout.getvalue())
    self.assertTrue(verdict["is_numerically_equivalent"], verdict)
    self.assertEqual(verdict["rel_l2"]["rel_l2_tol"], 0.05)


if __name__ == "__main__":
  absltest.main()
