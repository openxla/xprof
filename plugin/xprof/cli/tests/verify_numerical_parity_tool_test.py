"""Tests for verify_numerical_parity_tool CLI interface and resolution."""

import json
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from xprof.cli.tools import verify_numerical_parity_tool


def sample_ref_fn(x: np.ndarray) -> np.ndarray:
  return x * 2.0


def sample_candidate_fn(x: np.ndarray) -> np.ndarray:
  return x + x


class VerifyNumericalParityToolTest(parameterized.TestCase):

  def test_verify_with_direct_callables_pass(self):
    """Verifies tool with direct Python callable functions."""
    report_json = verify_numerical_parity_tool.verify_numerical_parity(
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
    report_json = verify_numerical_parity_tool.verify_numerical_parity(
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
    report_json = verify_numerical_parity_tool.verify_numerical_parity(
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
    report_json = verify_numerical_parity_tool.verify_numerical_parity(
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
      verify_numerical_parity_tool._resolve_callable(
          "non_existent_module_xyz.some_fn"
      )

  def test_resolve_callable_invalid_attribute_raises(self):
    """Verifies missing attribute on existing module raises AttributeError."""
    with self.assertRaises(AttributeError):
      verify_numerical_parity_tool._resolve_callable(
          "numpy.non_existent_function_12345"
      )

  def test_resolve_callable_non_callable_attribute_raises(self):
    """Verifies resolving to a non-callable variable raises TypeError."""
    with self.assertRaises(TypeError):
      verify_numerical_parity_tool._resolve_callable("numpy.pi")

  def test_resolve_callable_empty_string_raises(self):
    """Verifies empty string path raises ValueError."""
    with self.assertRaises(ValueError):
      verify_numerical_parity_tool._resolve_callable("   ")


if __name__ == "__main__":
  absltest.main()
