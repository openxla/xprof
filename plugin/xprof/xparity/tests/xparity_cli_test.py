"""Tests for xparity_tool and xparity_cli subcommands."""

import json
import re
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from xprof.xparity import xparity_cli
from xprof.xparity import xparity_tool


def sample_ref_fn(x: np.ndarray) -> np.ndarray:
  return x * 2.0


def sample_candidate_fn(x: np.ndarray) -> np.ndarray:
  return x + x


class XparityCliTest(parameterized.TestCase):

  def test_cli_main_exposes_all_subcommands(self):
    """Verifies cli_main registers verify, generate_suite, inspect_suite, probe_precision."""
    cmds = xparity_cli.cli_main()
    self.assertIn("verify", cmds)
    self.assertIn("verify_numerical_parity", cmds)
    self.assertIn("generate_suite", cmds)
    self.assertIn("inspect_suite", cmds)
    self.assertIn("probe_precision", cmds)

  def test_bug_link_matches_build_variant(self):
    """The reported bug target differs between the 1P and OSS builds.

    `xparity_cli.py` is exported to github.com/openxla/xprof, where a `go/`
    shortlink is unresolvable, so the constant is copybara-replaced. Without
    this assertion the replacement branch is never executed by any test.
    """
    expected_bug_link = "https://github.com/openxla/xprof/issues"
    self.assertEqual(xparity_cli._BUG_LINK, expected_bug_link)  # pylint: disable=protected-access

  def test_verify_with_direct_callables_pass(self):
    """Verifies tool with direct Python callable functions."""
    report_json = xparity_tool.verify_numerical_parity(
        kernel_ref=sample_ref_fn,
        kernel_candidate=sample_candidate_fn,
        shapes=[(16, 16)],
        dtype_str="float32",
        tier="fast_agent",
    )
    report = json.loads(report_json)
    self.assertTrue(report["is_numerically_equivalent"])
    self.assertEqual(report["overall_max_ulp"], 0)
    self.assertEqual(report["failed_batches_count"], 0)

  def test_verify_with_string_dotted_import_paths(self):
    """Verifies tool dynamically resolves module-qualified dotted strings."""
    report_json = xparity_tool.verify_numerical_parity(
        kernel_ref="numpy.sin",
        kernel_candidate="numpy.sin",
        shapes=[(8, 8)],
        dtype_str="float32",
        tier="fast_agent",
    )
    report = json.loads(report_json)
    self.assertTrue(report["is_numerically_equivalent"])
    self.assertEqual(report["overall_max_ulp"], 0)

  def test_verify_with_string_colon_import_paths(self):
    """Verifies tool dynamically resolves colon syntax (module:attribute)."""
    report_json = xparity_tool.verify_numerical_parity(
        kernel_ref="numpy:cos",
        kernel_candidate="numpy:cos",
        shapes=[(8, 8)],
        dtype_str="float32",
        tier="fast_agent",
    )
    report = json.loads(report_json)
    self.assertTrue(report["is_numerically_equivalent"])
    self.assertEqual(report["overall_max_ulp"], 0)

  def test_verify_with_string_shapes_literal(self):
    """Verifies string literal shapes from CLI flags are parsed cleanly."""
    report_json = xparity_tool.verify_numerical_parity(
        kernel_ref=sample_ref_fn,
        kernel_candidate=sample_candidate_fn,
        shapes="[(16, 32)]",
        dtype_str="float32",
        tier="fast_agent",
    )
    report = json.loads(report_json)
    self.assertTrue(report["is_numerically_equivalent"])

  def test_resolve_callable_invalid_module_raises(self):
    """Verifies non-existent module name raises ImportError."""
    with self.assertRaises(ImportError):
      xparity_tool._resolve_callable("non_existent_module_xyz.some_fn")

  def test_resolve_callable_invalid_attribute_raises(self):
    """Verifies missing attribute on existing module raises AttributeError."""
    with self.assertRaises(AttributeError):
      xparity_tool._resolve_callable("numpy.non_existent_function_12345")

  def test_resolve_callable_non_callable_attribute_raises(self):
    """Verifies resolving to a non-callable variable raises TypeError."""
    with self.assertRaises(TypeError):
      xparity_tool._resolve_callable("numpy.pi")

  def test_resolve_callable_empty_string_raises(self):
    """Verifies empty string path raises ValueError."""
    with self.assertRaises(ValueError):
      xparity_tool._resolve_callable("   ")

  def test_verify_tool_emits_tolerance_audit_json(self):
    """Verifies CLI tool output JSON contains full tolerance_audit metadata."""
    report_json = xparity_tool.verify_numerical_parity(
        kernel_ref=sample_ref_fn,
        kernel_candidate=sample_candidate_fn,
        shapes=[(16, 16)],
        dtype_str="bfloat16",
        tier="fast_agent",
        max_allowed_ulp=4,  # Relaxed override above 2
    )
    report = json.loads(report_json)
    self.assertTrue(report["is_numerically_equivalent"])
    self.assertIn("tolerance_audit", report)
    audit = report["tolerance_audit"]
    self.assertTrue(audit["is_relaxed_override"])
    self.assertEqual(audit["recommended_contract_ulp"], 2)
    self.assertEqual(audit["configured_max_ulp"], 4)
    self.assertIn("caution_banner", audit)
    self.assertIn("⚠️ CAUTION", audit["caution_banner"])

  def test_verify_tool_discrete_integer_and_boolean(self):
    """Verifies CLI tool verification on discrete integer and boolean functions."""

    def int_ref_fn(x):
      return x

    def int_cand_fn(x):
      return x

    report_json = xparity_tool.verify_numerical_parity(
        kernel_ref=int_ref_fn,
        kernel_candidate=int_cand_fn,
        shapes=[(8, 8)],
        dtype_str="int32",
        tier="fast_agent",
    )
    report = json.loads(report_json)
    self.assertTrue(report["is_numerically_equivalent"])
    self.assertEqual(report["overall_max_ulp"], 0)

  def test_verify_tool_hard_ceiling_error_json(self):
    """Verifies that exceeding hard safety ceiling raises ValueError."""
    with self.assertRaises(ValueError) as ctx:
      xparity_tool.verify_numerical_parity(
          kernel_ref=sample_ref_fn,
          kernel_candidate=sample_candidate_fn,
          shapes=[(8, 8)],
          dtype_str="bfloat16",
          tier="fast_agent",
          max_allowed_ulp=12,  # Hard ceiling for bfloat16 is 8
      )
    self.assertIn("exceeds immutable safety ceiling", str(ctx.exception))

  def test_verify_nan_output_produces_strict_rfc8259_json(self):
    """Verifies that NaN outputs produce valid RFC 8259 JSON with nulls."""

    def nan_candidate_fn(x: np.ndarray) -> np.ndarray:
      out = np.array(x * 2.0)
      out[0, 0] = np.nan
      return out

    report_json = xparity_tool.verify_numerical_parity(
        kernel_ref=sample_ref_fn,
        kernel_candidate=nan_candidate_fn,
        shapes=[(8, 8)],
        dtype_str="float32",
        tier="fast_agent",
    )

    def _reject_non_standard_constants(val: str) -> None:
      raise ValueError(f"Encountered non-standard JSON token: {val}")

    parsed = json.loads(
        report_json, parse_constant=_reject_non_standard_constants
    )
    self.assertFalse(parsed["is_numerically_equivalent"])
    self.assertGreater(parsed["failed_batches_count"], 0)
    self.assertIsNone(
        re.search(r":\s*(?:NaN|Infinity|-Infinity)\b", report_json)
    )
    self.assertIsNone(parsed["ulp_context"]["p50"])

  def test_verify_structured_shape_mismatch_verdict(self):
    """Verifies shape mismatch returns structured JSON failure report (G4)."""

    def wrong_shape_cand(x: np.ndarray) -> np.ndarray:
      return x.reshape(-1)

    report_json = xparity_tool.verify_numerical_parity(
        kernel_ref=sample_ref_fn,
        kernel_candidate=wrong_shape_cand,
        shapes=[(8, 8)],
        dtype_str="float32",
        tier="fast_agent",
    )
    parsed = json.loads(report_json)
    self.assertFalse(parsed["is_numerically_equivalent"])
    self.assertEqual(parsed["correctness_basis"], "SHAPE_MISMATCH")
    self.assertIsNotNone(parsed["shape_mismatch"])
    self.assertIn("Shape mismatch in batch", parsed["shape_mismatch"]["error"])

  def test_generate_and_inspect_suite_commands(self):
    """Verifies generate_suite and inspect_suite CLI subcommands (G6)."""
    npz_path = self.create_tempfile("suite.npz").full_path
    gen_json = xparity_tool.generate_suite(
        shapes=[(8, 16)],
        output_path=npz_path,
        dtype_str="bfloat16",
        tier="fast_agent",
    )
    gen_meta = json.loads(gen_json)
    self.assertEqual(gen_meta["output_path"], npz_path)
    self.assertGreater(gen_meta["num_batches"], 0)

    insp_json = xparity_tool.inspect_suite(npz_path)
    insp_meta = json.loads(insp_json)
    self.assertEqual(insp_meta["num_batches"], gen_meta["num_batches"])

  def test_probe_precision_command(self):
    """Verifies probe_precision CLI subcommand (G6)."""
    probe_json = xparity_tool.probe_precision(
        kernel_fn=sample_ref_fn,
        shapes=[(8, 8)],
        dtype_str="float32",
        device_kind="cpu",
    )
    probe_meta = json.loads(probe_json)
    self.assertTrue(probe_meta["is_pinned_at_highest"])
    self.assertFalse(probe_meta["reference_pin_inert"])


if __name__ == "__main__":
  absltest.main()
