"""Tests for the compact verdict output of the Xparity tool and CLI."""

import contextlib
import io
import json
import os
import tempfile

from absl.testing import absltest
import numpy as np

from xprof.xparity import xparity_cli
from xprof.xparity import xparity_tool


def ref_kernel(x):
  return np.asarray(x, dtype=np.float32) * np.float32(2.0)


def clipped_kernel(x):
  """Matches ref_kernel for |x| <= 8 and is wrong for larger inputs."""
  return np.clip(np.asarray(x, dtype=np.float32), -8.0, 8.0) * np.float32(2.0)


def wrong_shape_kernel(x):
  return ref_kernel(x)[..., :-1]


_KWARGS = dict(shapes="(32, 32)", dtype_str="float32", tier="presubmit")

_VERDICT_KEYS = {
    "is_numerically_equivalent",
    "correctness_basis",
    "contract",
    "overall_max_ulp",
    "failed_batches_count",
    "total_batches_count",
    "failures",
    "failures_omitted",
    "regimes_not_run",
    "regimes_over_tolerance",
    "failure_dumps",
    "verdict",
}


class CompactVerdictTest(absltest.TestCase):

  def test_passing_verdict(self):
    out = xparity_tool.verify_numerical_parity(
        ref_kernel, ref_kernel, verdict_only=True, **_KWARGS
    )
    self.assertNotIn("\n", out)
    verdict = json.loads(out)
    self.assertEqual(set(verdict), _VERDICT_KEYS)
    self.assertTrue(verdict["is_numerically_equivalent"])
    self.assertEqual(verdict["failures"], [])
    self.assertTrue(verdict["verdict"].startswith("PASSED"))

  def test_failing_verdict_is_small_and_names_failures(self):
    full = xparity_tool.verify_numerical_parity(
        ref_kernel, clipped_kernel, atol=1e-3, rtol=1e-3, **_KWARGS
    )
    compact = xparity_tool.verify_numerical_parity(
        ref_kernel,
        clipped_kernel,
        atol=1e-3,
        rtol=1e-3,
        verdict_only=True,
        **_KWARGS,
    )
    self.assertLess(len(compact), 2048)
    self.assertLess(len(compact), len(full) // 4)
    verdict = json.loads(compact)
    full_payload = json.loads(full)
    self.assertFalse(verdict["is_numerically_equivalent"])
    self.assertEqual(
        verdict["failed_batches_count"], full_payload["failed_batches_count"]
    )
    self.assertNotEmpty(verdict["failures"])
    self.assertLessEqual(len(verdict["failures"]), 5)
    self.assertEqual(
        len(verdict["failures"]) + verdict["failures_omitted"],
        verdict["failed_batches_count"],
    )
    self.assertNotIn("normal", {f["regime"] for f in verdict["failures"]})
    self.assertNotEmpty(verdict["regimes_over_tolerance"])
    self.assertTrue(verdict["verdict"].startswith("FAILED"))

  def test_shape_mismatch_verdict(self):
    verdict = json.loads(
        xparity_tool.verify_numerical_parity(
            ref_kernel, wrong_shape_kernel, verdict_only=True, **_KWARGS
        )
    )
    self.assertFalse(verdict["is_numerically_equivalent"])
    self.assertEqual(verdict["correctness_basis"], "SHAPE_MISMATCH")
    self.assertIn("Shape mismatch", verdict["verdict"])

  def test_cli_flag(self):
    module = __name__
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
      code = xparity_cli.main([
          "xparity_cli",
          "verify",
          f"--kernel_ref={module}.ref_kernel",
          f"--kernel_candidate={module}.clipped_kernel",
          "--shapes=(32, 32)",
          "--dtype_str=float32",
          "--verdict_only",
      ])
    self.assertEqual(code, xparity_cli.EXIT_OK)
    verdict = json.loads(stdout.getvalue())
    self.assertFalse(verdict["is_numerically_equivalent"])

  def test_replay_verdict(self):
    with tempfile.TemporaryDirectory() as tmp:
      full = json.loads(
          xparity_tool.verify_numerical_parity(
              ref_kernel, clipped_kernel, dump_failures_to=tmp, **_KWARGS
          )
      )
      dump = full["failure_dumps"][0]
      self.assertTrue(os.path.exists(dump))
      verdict = json.loads(
          xparity_tool.replay(
              dump, ref_kernel, clipped_kernel, verdict_only=True
          )
      )
    self.assertFalse(verdict["is_numerically_equivalent"])
    self.assertLen(verdict["failures"], 1)


if __name__ == "__main__":
  absltest.main()
