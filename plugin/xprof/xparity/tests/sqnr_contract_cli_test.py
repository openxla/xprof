"""Tests for the SQNR flags and report fields of the xparity tool and CLI."""

import contextlib
import io
import json
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import ml_dtypes
import numpy as np

from xprof.xparity import sqnr
from xprof.xparity import xparity_cli
from xprof.xparity import xparity_tool

_VERDICT_SQNR_KEYS = {
    "sqnr_status",
    "passed",
    "sqnr_db",
    "oracle_sqnr_db",
    "threshold_db",
    "threshold_source",
    "gate_math",
    "failed_reports",
    "warnings",
    "failed_reasons",
}


def _ref(inputs: Any) -> np.ndarray:
  return np.asarray(inputs, dtype=np.float32) * np.float32(0.5)


def _bf16_rounded(inputs: Any) -> np.ndarray:
  return _ref(inputs).astype(ml_dtypes.bfloat16).astype(np.float32)


def _half_scale(inputs: Any) -> np.ndarray:
  return _ref(inputs) * np.float32(0.5)


def _short(inputs: Any) -> np.ndarray:
  return _ref(inputs)[..., :-1]


def _tree_ref(inputs: Any) -> tuple[np.ndarray, np.ndarray]:
  return np.argmax(np.asarray(inputs), axis=-1), _ref(inputs)


def _tree_bf16_rounded(inputs: Any) -> tuple[np.ndarray, np.ndarray]:
  return np.argmax(np.asarray(inputs), axis=-1), _bf16_rounded(inputs)


def _tree_half_scale(inputs: Any) -> tuple[np.ndarray, np.ndarray]:
  return np.argmax(np.asarray(inputs), axis=-1), _half_scale(inputs)


def _verify(candidate: Any, **kwargs: Any) -> dict[str, Any]:
  kwargs.setdefault("shapes", "(64, 64)")
  kwargs.setdefault("dtype_str", "float32")
  kwargs.setdefault("regimes", "normal")
  kwargs.setdefault("contract", "sqnr")
  return json.loads(
      xparity_tool.verify_numerical_parity(_ref, candidate, **kwargs)
  )


def _run_cli(*flags: str) -> tuple[int, str]:
  module = __name__
  stdout = io.StringIO()
  with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(
      io.StringIO()
  ):
    code = xparity_cli.main([
        "xparity_cli",
        "verify",
        f"--kernel_ref={module}._ref",
        "--shapes=(64, 64)",
        "--dtype_str=float32",
        "--regimes=normal",
        *flags,
    ])
  return code, stdout.getvalue()


class SqnrToolTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("auto", sqnr.AUTO, sqnr.DEFAULT_MIN_SQNR_DB),
      ("float", 40.0, 40.0),
      ("int", 40, 40.0),
      ("string", "40", 40.0),
  )
  def test_compact_verdict_passes_honest_rounding(
      self, min_sqnr_db: Any, threshold_db: float
  ):
    verdict = _verify(_bf16_rounded, min_sqnr_db=min_sqnr_db, verdict_only=True)
    self.assertTrue(verdict["is_numerically_equivalent"], verdict)
    self.assertEqual(set(verdict["sqnr"]), _VERDICT_SQNR_KEYS)
    self.assertEqual(
        (verdict["sqnr"]["passed"], verdict["sqnr"]["threshold_db"]),
        (True, threshold_db),
    )

  def test_compact_verdict_names_the_failure(self):
    verdict = _verify(_half_scale, verdict_only=True)
    self.assertFalse(verdict["is_numerically_equivalent"])
    self.assertStartsWith(verdict["verdict"], "FAILED")
    failure = verdict["failures"][0]
    self.assertEqual(failure["sqnr_status"], sqnr.STATUS_OK)
    self.assertLess(failure["sqnr_db"], 10.0)
    self.assertNotEmpty(failure["sqnr_failed_reasons"])
    self.assertNotEmpty(verdict["sqnr"]["failed_reasons"])

  def test_default_contract_has_no_sqnr_verdict_key(self):
    verdict = _verify(_ref, contract="ulp", verdict_only=True)
    self.assertTrue(verdict["is_numerically_equivalent"], verdict)
    self.assertNotIn("sqnr", verdict)

  def test_failure_of_other_contract_has_no_sqnr_fields(self):
    verdict = _verify(_half_scale, contract="ulp", verdict_only=True)
    self.assertFalse(verdict["is_numerically_equivalent"])
    failure = verdict["failures"][0]
    self.assertIn("max_ulp", failure)
    self.assertNoCommonElements(
        failure, {"sqnr_db", "sqnr_status", "sqnr_failed_reasons"}
    )

  def test_failure_of_non_sqnr_leaf_has_no_sqnr_fields(self):
    verdict = json.loads(
        xparity_tool.verify_numerical_parity(
            _tree_ref,
            _tree_half_scale,
            shapes="(64, 64)",
            dtype_str="float32",
            regimes="normal",
            contract="sqnr",
            contract_by_leaf="{'[1]': 'ulp'}",
            verdict_only=True,
        )
    )
    self.assertFalse(verdict["is_numerically_equivalent"])
    failure = verdict["failures"][0]
    self.assertEqual(failure["leaf_path"], "[1]")
    self.assertNoCommonElements(
        failure, {"sqnr_db", "sqnr_status", "sqnr_failed_reasons"}
    )

  def test_failure_of_sqnr_leaf_keeps_sqnr_fields(self):
    verdict = json.loads(
        xparity_tool.verify_numerical_parity(
            _tree_ref,
            _tree_half_scale,
            shapes="(64, 64)",
            dtype_str="float32",
            regimes="normal",
            contract="ulp",
            contract_by_leaf="{'[1]': 'sqnr'}",
            verdict_only=True,
        )
    )
    self.assertFalse(verdict["is_numerically_equivalent"])
    failure = verdict["failures"][0]
    self.assertEqual(
        (failure["leaf_path"], failure["sqnr_status"]), ("[1]", sqnr.STATUS_OK)
    )
    self.assertLess(failure["sqnr_db"], 10.0)
    self.assertNotEmpty(failure["sqnr_failed_reasons"])

  def test_compact_verdict_has_sqnr_for_a_sqnr_leaf(self):
    verdict = json.loads(
        xparity_tool.verify_numerical_parity(
            _tree_ref,
            _tree_bf16_rounded,
            shapes="(64, 64)",
            dtype_str="float32",
            regimes="normal",
            contract="ulp",
            contract_by_leaf="{'[1]': 'sqnr'}",
            verdict_only=True,
        )
    )
    self.assertTrue(verdict["is_numerically_equivalent"], verdict)
    self.assertEqual(set(verdict["sqnr"]), _VERDICT_SQNR_KEYS)
    self.assertTrue(verdict["sqnr"]["passed"])

  @parameterized.named_parameters(
      ("ulp", dict(contract="ulp"), False),
      ("sqnr", dict(contract="sqnr"), True),
      ("explicit_strict", dict(contract="ulp", sqnr_strict=True), True),
      ("explicit_floor", dict(contract="ulp", min_sqnr_db=40), False),
  )
  def test_default_strict_follows_the_contract(
      self, kwargs: dict[str, Any], strict: bool
  ):
    payload = _verify(_bf16_rounded, **kwargs)
    self.assertEqual(payload["run_config"]["sqnr_config"]["strict"], strict)

  def test_replay_override_keeps_the_report_only_default(self):
    tmp = self.create_tempdir().full_path
    payload = _verify(_half_scale, contract="ulp", dump_failures_to=tmp)
    dump = payload["failure_dumps"][0]
    again = json.loads(
        xparity_tool.replay(dump, _ref, _half_scale, min_sqnr_db=5)
    )
    self.assertEqual(
        (
            again["run_config"]["sqnr_config"]["min_sqnr_db"],
            again["run_config"]["sqnr_config"]["strict"],
        ),
        (5.0, False),
    )

  def test_preset_and_strict_reach_the_report(self):
    payload = _verify(
        _bf16_rounded, sqnr_preset="six_gate_v1", sqnr_strict=False
    )
    config = payload["run_config"]["sqnr_config"]
    self.assertEqual(
        (config["gate_math"], config["strict"], config["min_sqnr_db"]),
        (sqnr.GATE_MATH_SIX_GATE_V1, False, sqnr.DEFAULT_MIN_SQNR_DB),
    )
    self.assertEqual(
        payload["sqnr_summary"]["gate_math"], sqnr.GATE_MATH_SIX_GATE_V1
    )

  def test_explicit_auto_floor_with_six_gate_preset_stays_auto(self):
    payload = _verify(
        _bf16_rounded,
        sqnr_preset="six_gate_v1",
        min_sqnr_db="auto",
        sqnr_strict=False,
    )
    config = payload["run_config"]["sqnr_config"]
    self.assertEqual(
        (config["gate_math"], config["min_sqnr_db"]),
        (sqnr.GATE_MATH_SIX_GATE_V1, sqnr.AUTO),
    )

  @parameterized.named_parameters(
      ("explicit_auto", sqnr.AUTO, sqnr.AUTO),
      ("omitted", None, sqnr.DEFAULT_MIN_SQNR_DB),
  )
  def test_replay_preset_switch_floor(
      self, min_sqnr_db: str | None, expected_floor: float | str
  ):
    tmp = self.create_tempdir().full_path
    payload = _verify(_half_scale, sqnr_strict=False, dump_failures_to=tmp)
    self.assertEqual(
        payload["run_config"]["sqnr_config"]["min_sqnr_db"], "auto"
    )
    again = json.loads(
        xparity_tool.replay(
            payload["failure_dumps"][0],
            _ref,
            _half_scale,
            sqnr_preset="six_gate_v1",
            min_sqnr_db=min_sqnr_db,
        )
    )
    config = again["run_config"]["sqnr_config"]
    self.assertEqual(
        (config["gate_math"], config["min_sqnr_db"]),
        (sqnr.GATE_MATH_SIX_GATE_V1, expected_floor),
    )

  @parameterized.named_parameters(
      (
          "six_gate_auto_same_preset",
          dict(sqnr_preset="six_gate_v1", min_sqnr_db="auto"),
          "six_gate_v1",
          sqnr.AUTO,
      ),
      (
          "six_gate_default_to_derived",
          dict(sqnr_preset="six_gate_v1"),
          "derived",
          sqnr.AUTO,
      ),
      (
          "six_gate_explicit_floor_to_derived",
          dict(sqnr_preset="six_gate_v1", min_sqnr_db=30.0),
          "derived",
          30.0,
      ),
  )
  def test_replay_preset_switch_keeps_only_a_given_stored_floor(
      self,
      dump_kwargs: dict[str, Any],
      replay_preset: str,
      expected_floor: float | str,
  ):
    tmp = self.create_tempdir().full_path
    payload = _verify(
        _half_scale, sqnr_strict=False, dump_failures_to=tmp, **dump_kwargs
    )
    again = json.loads(
        xparity_tool.replay(
            payload["failure_dumps"][0],
            _ref,
            _half_scale,
            sqnr_preset=replay_preset,
        )
    )
    config = again["run_config"]["sqnr_config"]
    self.assertEqual(
        (config["gate_math"], config["min_sqnr_db"]),
        (replay_preset, expected_floor),
    )

  def test_full_report_serializes_sqnr(self):
    payload = _verify(_half_scale, contract="ulp")
    batch = payload["batch_results"][0]["sqnr"]
    self.assertEqual(batch["sqnr_status"], sqnr.STATUS_OK)
    self.assertFalse(batch["passed"])
    self.assertIsInstance(batch["failed_reasons"], list)
    self.assertIsInstance(batch["warnings"], list)
    self.assertIn("sqnr_db", payload["sqnr_summary"])

  def test_shape_mismatch_payload_has_sqnr_fields(self):
    verdict = _verify(_short, verdict_only=True)
    self.assertEqual(verdict["correctness_basis"], "SHAPE_MISMATCH")
    payload = _verify(_short)
    self.assertEqual(payload["sqnr_summary"], {})
    self.assertIn("sqnr_config", payload["run_config"])

  @parameterized.named_parameters(
      ("negative_floor", dict(min_sqnr_db=-1.0)),
      ("zero_floor", dict(min_sqnr_db=0)),
      ("bad_string", dict(min_sqnr_db="loud")),
      ("bool_floor", dict(min_sqnr_db=True)),
      ("bad_preset", dict(sqnr_preset="seven_gate")),
      ("bad_alpha", dict(sqnr_alpha=2.0)),
      ("bad_strict", dict(sqnr_strict="yes")),
  )
  def test_invalid_options_raise(self, kwargs: dict[str, Any]):
    with self.assertRaises(ValueError):
      _verify(_ref, **kwargs)

  def test_replay_keeps_and_overrides_the_config(self):
    tmp = self.create_tempdir().full_path
    payload = _verify(
        _half_scale, min_sqnr_db=10.0, sqnr_strict=False, dump_failures_to=tmp
    )
    dump = payload["failure_dumps"][0]
    again = json.loads(xparity_tool.replay(dump, _ref, _half_scale))
    self.assertFalse(again["is_numerically_equivalent"])
    self.assertEqual(
        (
            again["run_config"]["sqnr_config"]["min_sqnr_db"],
            again["run_config"]["sqnr_config"]["strict"],
        ),
        (10.0, False),
    )
    relaxed = json.loads(
        xparity_tool.replay(dump, _ref, _half_scale, min_sqnr_db=5)
    )
    self.assertTrue(relaxed["is_numerically_equivalent"], relaxed)
    self.assertEqual(
        (
            relaxed["run_config"]["sqnr_config"]["min_sqnr_db"],
            relaxed["run_config"]["sqnr_config"]["strict"],
        ),
        (5.0, False),
    )


class SqnrCliTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("auto", "--min_sqnr_db=auto"),
      ("number", "--min_sqnr_db=40"),
  )
  def test_flags_pass_honest_rounding(self, floor_flag: str):
    code, out = _run_cli(
        f"--kernel_candidate={__name__}._bf16_rounded",
        "--contract=sqnr",
        floor_flag,
        "--sqnr_alpha=1e-6",
        "--sqnr_preset=derived",
        "--sqnr_strict=True",
        "--verdict_only",
    )
    self.assertEqual(code, xparity_cli.EXIT_OK)
    verdict = json.loads(out)
    self.assertTrue(verdict["is_numerically_equivalent"], verdict)
    self.assertTrue(verdict["sqnr"]["passed"])

  @parameterized.named_parameters(
      ("explicit_auto", ("--min_sqnr_db=auto",), sqnr.AUTO),
      ("omitted", (), sqnr.DEFAULT_MIN_SQNR_DB),
  )
  def test_six_gate_preset_floor_flag(
      self, floor_flags: tuple[str, ...], expected_floor: float | str
  ):
    code, out = _run_cli(
        f"--kernel_candidate={__name__}._bf16_rounded",
        "--contract=sqnr",
        "--sqnr_preset=six_gate_v1",
        "--sqnr_strict=False",
        *floor_flags,
    )
    self.assertEqual(code, xparity_cli.EXIT_OK)
    config = json.loads(out)["run_config"]["sqnr_config"]
    self.assertEqual(
        (config["gate_math"], config["min_sqnr_db"]),
        (sqnr.GATE_MATH_SIX_GATE_V1, expected_floor),
    )

  def test_failing_candidate_reports_not_equivalent(self):
    code, out = _run_cli(
        f"--kernel_candidate={__name__}._half_scale",
        "--contract=sqnr",
        "--verdict_only",
    )
    self.assertEqual(code, xparity_cli.EXIT_OK)
    self.assertFalse(json.loads(out)["is_numerically_equivalent"])

  @parameterized.named_parameters(
      ("negative_floor", "--min_sqnr_db=-1"),
      ("bad_preset", "--sqnr_preset=seven_gate"),
  )
  def test_invalid_flag_value(self, flag: str):
    code, _ = _run_cli(
        f"--kernel_candidate={__name__}._bf16_rounded",
        "--contract=sqnr",
        flag,
    )
    self.assertEqual(code, xparity_cli.EXIT_INVALID_VALUE)


if __name__ == "__main__":
  absltest.main()
