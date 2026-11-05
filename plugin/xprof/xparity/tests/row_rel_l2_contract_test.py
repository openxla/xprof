"""Tests for the row_rel_l2 contract in the validator, tool and CLI."""

import contextlib
import io
import json

from absl.testing import absltest
import jax.numpy as jnp
import ml_dtypes
import numpy as np

from xprof.xparity import numerical_validator
from xprof.xparity import rel_l2
from xprof.xparity import xparity_cli
from xprof.xparity import xparity_tool

_ROW = numerical_validator.CONTRACT_ROW_REL_L2
_BAD_ROW = 5


def _row(result) -> rel_l2.RowRelL2Comparison:
  """Returns the row_rel_l2 comparison of a batch or leaf result."""
  assert result.row_rel_l2 is not None
  return result.row_rel_l2


def _leaves(batch) -> dict[str, numerical_validator.BatchValidationResult]:
  assert batch.leaf_results is not None
  return batch.leaf_results


def ref_kernel(x):
  return np.asarray(x, dtype=np.float32) * np.float32(0.5)


def fp8_row_scaled_kernel(x):
  """ref_kernel with each row round-tripped through scaled FP8 E4M3."""
  out = ref_kernel(x)
  scale = np.max(np.abs(out), axis=-1, keepdims=True) / np.float32(448.0)
  scale = np.where(scale > 0, scale, np.float32(1.0)).astype(np.float32)
  quantized = (out / scale).astype(ml_dtypes.float8_e4m3fn)
  return quantized.astype(np.float32) * scale


def bad_row_kernel(x):
  out = ref_kernel(x).copy()
  out[_BAD_ROW] *= np.float32(1.5)
  return out


def nan_kernel(x):
  out = ref_kernel(x).copy()
  out[0, 0] = np.nan
  return out


def short_kernel(x):
  return ref_kernel(x)[..., :-1]


def _validate(candidate, shapes=(256, 128), **kwargs):
  kwargs.setdefault("dtype_str", "float32")
  kwargs.setdefault("regimes", "normal")
  return numerical_validator.validate_kernels(
      ref_kernel,
      candidate,
      shapes=shapes,
      contract=_ROW,
      rel_l2_tol=kwargs.pop("rel_l2_tol", 5e-2),
      **kwargs,
  )


class RowRelL2ContractTest(absltest.TestCase):

  def test_lossy_fp8_candidate_passes_row_rel_l2_but_not_ulp(self):
    report = _validate(fp8_row_scaled_kernel)
    self.assertTrue(report.is_numerically_equivalent, report.summary_message)
    self.assertIn("contract: row_rel_l2", report.summary_message)
    self.assertGreater(report.overall_max_ulp, 1000)
    summary = report.rel_l2_summary
    self.assertLess(summary["max_rel_l2"], 5e-2)
    self.assertGreater(summary["max_rel_l2"], 1e-3)
    self.assertEqual(summary["failed_checks"], 0)
    self.assertEqual(report.run_config["rel_l2_tol"], 5e-2)
    self.assertEqual(report.run_config["row_factor"], 5.0)

    ulp_report = numerical_validator.validate_kernels(
        ref_kernel,
        fp8_row_scaled_kernel,
        shapes=(256, 128),
        dtype_str="float32",
        regimes="normal",
    )
    self.assertFalse(ulp_report.is_numerically_equivalent)

  def test_lossy_fp8_candidate_passes_on_every_regime(self):
    report = _validate(fp8_row_scaled_kernel, regimes="all")
    self.assertTrue(report.is_numerically_equivalent, report.summary_message)
    self.assertGreater(report.total_batches_count, 1)

  def test_single_bad_row_fails_and_is_located(self):
    report = _validate(bad_row_kernel)
    self.assertFalse(report.is_numerically_equivalent)
    batch = report.batch_results[0]
    self.assertFalse(batch.passed)
    self.assertIsNotNone(batch.row_rel_l2)
    self.assertEqual(batch.row_rel_l2.reason, "row_rel_l2")
    self.assertEqual(batch.row_rel_l2.worst_row_index, (_BAD_ROW,))
    self.assertLess(batch.row_rel_l2.rel_l2, 5e-2)
    self.assertIn(f"row ({_BAD_ROW},)", report.summary_message)
    self.assertEqual(report.rel_l2_summary["worst_row_index"], [_BAD_ROW])

  def test_row_factor_and_floor_are_honored(self):
    self.assertTrue(
        _validate(bad_row_kernel, row_factor=20.0).is_numerically_equivalent
    )

  def test_nan_candidate_fails(self):
    report = _validate(nan_kernel, rel_l2_tol=1.0)
    self.assertFalse(report.is_numerically_equivalent)
    self.assertEqual(
        report.batch_results[0].row_rel_l2.reason, "non_finite_candidate"
    )

  def test_missing_tolerance_raises(self):
    with self.assertRaisesRegex(ValueError, "rel_l2_tol"):
      numerical_validator.validate_kernels(
          ref_kernel, ref_kernel, shapes=(8, 8), contract=_ROW
      )
    with self.assertRaisesRegex(ValueError, "row_factor"):
      _validate(ref_kernel, row_factor=0.0)

  def test_bfloat16_outputs(self):
    def ref_bf16(x):
      return ref_kernel(x).astype(ml_dtypes.bfloat16)

    def bad_bf16(x):
      return bad_row_kernel(x).astype(ml_dtypes.bfloat16)

    def f32_out(x):
      return ref_kernel(x)

    kwargs = dict(
        shapes=(256, 128),
        dtype_str="bfloat16",
        regimes="normal",
        contract=_ROW,
        rel_l2_tol=5e-2,
    )
    self.assertTrue(
        numerical_validator.validate_kernels(
            ref_bf16, ref_bf16, **kwargs
        ).is_numerically_equivalent
    )
    bad = numerical_validator.validate_kernels(ref_bf16, bad_bf16, **kwargs)
    self.assertFalse(bad.is_numerically_equivalent)
    # The output dtype must match the reference, as in the ULP contract's
    # bitwise detail.
    widened = numerical_validator.validate_kernels(ref_bf16, f32_out, **kwargs)
    self.assertFalse(widened.is_numerically_equivalent)
    self.assertEqual(_row(widened.batch_results[0]).reason, "dtype_mismatch")

  def test_jax_kernels(self):
    def jax_ref(x):
      return jnp.asarray(x, jnp.float32) * 0.5

    def jax_bad(x):
      return jax_ref(x).at[_BAD_ROW].multiply(1.5)

    report = numerical_validator.validate_kernels(
        jax_ref,
        jax_bad,
        shapes=(256, 128),
        dtype_str="float32",
        regimes="normal",
        contract=_ROW,
        rel_l2_tol=5e-2,
    )
    self.assertFalse(report.is_numerically_equivalent)
    self.assertEqual(_row(report.batch_results[0]).worst_row_index, (_BAD_ROW,))

  def test_contract_by_leaf(self):
    def ref_tree(x):
      return {"idx": np.argmax(x, axis=-1), "out": ref_kernel(x)}

    def cand_tree(x):
      return {"idx": np.argmax(x, axis=-1), "out": fp8_row_scaled_kernel(x)}

    report = numerical_validator.validate_kernels(
        ref_tree,
        cand_tree,
        shapes=(256, 128),
        dtype_str="float32",
        regimes="normal",
        contract_by_leaf={"['out']": _ROW},
        rel_l2_tol=5e-2,
    )
    self.assertTrue(report.is_numerically_equivalent, report.summary_message)
    leaves = _leaves(report.batch_results[0])
    self.assertIsNone(leaves["['idx']"].row_rel_l2)
    self.assertTrue(_row(leaves["['out']"]).passed)
    self.assertEqual(report.rel_l2_summary["worst_row_leaf"], "['out']")

  def test_validate_arrays(self):
    ref = np.random.default_rng(0).standard_normal((64, 32)).astype(np.float32)
    bad = ref.copy()
    # Row 7 is off by 1.5x its norm (bound 5 * 0.25), while the whole output
    # stays near 1.5 / sqrt(64) = 0.19, under the 0.25 tolerance.
    bad[7] *= 2.5
    res = numerical_validator.validate_arrays(
        bad, ref, dtype_str="float32", contract=_ROW, rel_l2_tol=0.25
    )
    self.assertFalse(res.passed)
    self.assertEqual(_row(res).reason, rel_l2.REASON_ROW)
    self.assertEqual(_row(res).worst_row_index, (7,))
    with self.assertRaisesRegex(ValueError, "rel_l2_tol"):
      numerical_validator.validate_arrays(bad, ref, contract=_ROW)


class RowRelL2ToolTest(absltest.TestCase):

  def test_compact_verdict_names_the_bad_row(self):
    verdict = json.loads(
        xparity_tool.verify_numerical_parity(
            ref_kernel,
            bad_row_kernel,
            shapes="(256, 128)",
            dtype_str="float32",
            regimes="normal",
            contract=_ROW,
            rel_l2_tol=5e-2,
            verdict_only=True,
        )
    )
    self.assertFalse(verdict["is_numerically_equivalent"])
    self.assertEqual(verdict["contract"], _ROW)
    failure = verdict["failures"][0]
    self.assertEqual(failure["reason"], "row_rel_l2")
    self.assertEqual(failure["worst_row_index"], [_BAD_ROW])
    self.assertAlmostEqual(failure["worst_row_rel_l2"], 0.5, delta=1e-4)
    self.assertEqual(verdict["rel_l2"]["worst_row_index"], [_BAD_ROW])
    self.assertTrue(verdict["verdict"].startswith("FAILED"))

  def test_full_report_serializes_row_rel_l2(self):
    payload = json.loads(
        xparity_tool.verify_numerical_parity(
            ref_kernel,
            nan_kernel,
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
            ref_kernel,
            short_kernel,
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
            ref_kernel,
            bad_row_kernel,
            shapes="(256, 128)",
            dtype_str="float32",
            regimes="normal",
            contract=_ROW,
            rel_l2_tol=5e-2,
            dump_failures_to=tmp,
        )
    )
    dump = payload["failure_dumps"][0]
    again = json.loads(xparity_tool.replay(dump, ref_kernel, bad_row_kernel))
    self.assertFalse(again["is_numerically_equivalent"])
    self.assertEqual(again["run_config"]["rel_l2_tol"], 5e-2)
    relaxed = json.loads(
        xparity_tool.replay(dump, ref_kernel, bad_row_kernel, row_factor=20.0)
    )
    self.assertTrue(relaxed["is_numerically_equivalent"])

  def test_cli_flags(self):
    module = __name__
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
      code = xparity_cli.main([
          "xparity_cli",
          "verify",
          f"--kernel_ref={module}.ref_kernel",
          f"--kernel_candidate={module}.fp8_row_scaled_kernel",
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
