"""Tests for saving failing inputs and replaying them."""

import json
import os

from absl.testing import absltest
import numpy as np

from xprof.xparity import numerical_generator
from xprof.xparity import numerical_validator
from xprof.xparity import xparity_cli
from xprof.xparity import xparity_tool

_SHAPES = (8, 8)


def _ref(x):
  return np.asarray(x, dtype=np.float32) * 2.0


def _broken(x):
  return _ref(x) + np.float32(1e-3)


def _validate(tmp_dir, candidate=_broken, **kwargs):
  return numerical_validator.validate_kernels(
      _ref,
      candidate,
      shapes=_SHAPES,
      dtype_str="float32",
      tier="fast_agent",
      dump_failures_to=tmp_dir,
      **kwargs,
  )


class FailureDumpTest(absltest.TestCase):

  def test_failing_batches_are_saved(self):
    tmp_dir = self.create_tempdir().full_path
    report = _validate(tmp_dir)
    self.assertFalse(report.is_numerically_equivalent)
    self.assertLen(report.failure_dumps, report.failed_batches_count)
    for path in report.failure_dumps:
      self.assertTrue(os.path.exists(path))
      self.assertEqual(os.path.dirname(path), tmp_dir)
    self.assertIn("Failing inputs saved", report.summary_message)

  def test_passing_run_saves_nothing(self):
    tmp_dir = os.path.join(self.create_tempdir().full_path, "dumps")
    report = _validate(tmp_dir, candidate=_ref)
    self.assertTrue(report.is_numerically_equivalent)
    self.assertEmpty(report.failure_dumps)
    self.assertFalse(os.path.exists(tmp_dir))

  def test_dump_records_run_context(self):
    report = _validate(self.create_tempdir().full_path)
    first = report.batch_results[0]
    info = numerical_generator.read_suite_metadata(report.failure_dumps[0])
    self.assertEqual(info["dtype_str"], "float32")
    self.assertEqual(info["contract"], numerical_validator.CONTRACT_ULP)
    self.assertEqual(info["max_allowed_ulp"], 2)
    self.assertEqual(info["failure"]["batch_name"], first.batch_name)
    self.assertEqual(info["failure"]["max_ulp"], first.max_ulp_distance)

  def test_replay_reproduces_failure_and_confirms_fix(self):
    report = _validate(self.create_tempdir().full_path)
    first = report.batch_results[0]
    path = report.failure_dumps[0]

    again = numerical_validator.replay_failure_dump(path, _ref, _broken)
    self.assertFalse(again.is_numerically_equivalent)
    self.assertEqual(again.total_batches_count, 1)
    self.assertEqual(again.batch_results[0].batch_name, first.batch_name)
    self.assertEqual(again.overall_max_ulp, first.max_ulp_distance)

    fixed = numerical_validator.replay_failure_dump(path, _ref, _ref)
    self.assertTrue(fixed.is_numerically_equivalent)

  def test_replay_uses_stored_contract_unless_overridden(self):
    def one_ulp_off(x):
      return (_ref(x).view(np.uint32) + 1).view(np.float32)

    report = _validate(
        self.create_tempdir().full_path,
        candidate=one_ulp_off,
        contract=numerical_validator.CONTRACT_BITWISE,
    )
    self.assertNotEmpty(report.failure_dumps)
    path = report.failure_dumps[0]

    strict = numerical_validator.replay_failure_dump(path, _ref, one_ulp_off)
    self.assertEqual(
        strict.run_config["contract"], numerical_validator.CONTRACT_BITWISE
    )
    self.assertFalse(strict.is_numerically_equivalent)

    relaxed = numerical_validator.replay_failure_dump(
        path, _ref, one_ulp_off, contract=numerical_validator.CONTRACT_ULP
    )
    self.assertTrue(relaxed.is_numerically_equivalent)

  def test_plain_suite_is_rejected(self):
    path = os.path.join(self.create_tempdir().full_path, "suite.npz")
    numerical_generator.save_test_suite(
        numerical_generator.generate_test_suite(
            _SHAPES, dtype_str="float32", tier="fast_agent"
        ),
        path,
    )
    with self.assertRaisesRegex(ValueError, "not an Xparity failure dump"):
      numerical_validator.replay_failure_dump(path, _ref, _ref)


class ReplayToolTest(absltest.TestCase):

  def test_tool_round_trip(self):
    payload = json.loads(
        xparity_tool.verify_numerical_parity(
            _ref,
            _broken,
            shapes=_SHAPES,
            dtype_str="float32",
            tier="fast_agent",
            dump_failures_to=self.create_tempdir().full_path,
        )
    )
    self.assertFalse(payload["is_numerically_equivalent"])
    self.assertNotEmpty(payload["failure_dumps"])

    replayed = json.loads(
        xparity_tool.replay(payload["failure_dumps"][0], _ref, _ref)
    )
    self.assertTrue(replayed["is_numerically_equivalent"])
    self.assertEqual(replayed["total_batches_count"], 1)
    self.assertEqual(replayed["failure_dumps"], [])

  def test_cli_registers_replay(self):
    self.assertIs(xparity_cli.cli_main()["replay"], xparity_tool.replay)


if __name__ == "__main__":
  absltest.main()
