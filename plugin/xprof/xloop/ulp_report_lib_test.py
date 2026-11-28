# Copyright 2026 The XProf Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Unit tests for the ULP report."""

from typing import Any

from absl.testing import absltest
import ml_dtypes
import numpy as np

from xprof.xloop import ulp_report_lib


def _bf16(values: object) -> np.ndarray:
  return np.asarray(values, dtype=np.float32).astype(ml_dtypes.bfloat16)


class CompareOutputsTest(absltest.TestCase):

  def test_identical_outputs_are_bit_identical(self):
    leaf = _bf16([1.0, -2.5, 3.0])
    report = ulp_report_lib.compare_outputs([[leaf]], [[leaf.copy()]])
    self.assertEqual(report["status"], "ok")
    self.assertEqual(report["max_ulp"], 0)
    self.assertEqual(report["median_ulp"], 0.0)
    self.assertTrue(report["bit_identical"])
    self.assertTrue(report["within_contract"])
    self.assertNotIn("worst_index", report)

  def test_one_ulp_off_reports_the_worst_element(self):
    reference = _bf16([1.0, 2.0, 4.0, 8.0])
    candidate = reference.copy()
    # The next bfloat16 above 4.0.
    candidate[2] = _bf16([4.03125])[0]
    report = ulp_report_lib.compare_outputs([[reference]], [[candidate]])
    self.assertEqual(report["max_ulp"], 1)
    self.assertEqual(report["median_ulp"], 0.0)
    self.assertEqual(report["contract_ulp"], 2)
    self.assertEqual(report["dtype"], "bfloat16")
    self.assertTrue(report["within_contract"])
    self.assertFalse(report["bit_identical"])
    self.assertEqual(report["worst_sample"], 0)
    self.assertEqual(report["worst_leaf"], 0)
    self.assertEqual(report["worst_index"], [2])
    self.assertEqual(report["worst_reference_value"], 4.0)
    self.assertEqual(report["worst_output_value"], 4.03125)

  def test_a_zeroed_tail_is_far_over_the_contract(self):
    reference = _bf16(np.linspace(1.0, 2.0, 16))
    candidate = reference.copy()
    candidate[8:] = 0
    report = ulp_report_lib.compare_outputs([[reference]], [[candidate]])
    self.assertGreater(report["max_ulp"], 1000)
    self.assertFalse(report["within_contract"])
    self.assertGreaterEqual(report["worst_index"][0], 8)
    self.assertEqual(report["worst_output_value"], 0.0)

  def test_nan_and_inf_are_counted_and_left_out_of_the_distances(self):
    reference = np.asarray([1.0, 2.0, np.inf, 4.0], dtype=np.float32)
    candidate = np.asarray([1.0, np.nan, np.inf, -np.inf], dtype=np.float32)
    report = ulp_report_lib.compare_outputs([[reference]], [[candidate]])
    self.assertEqual(report["status"], "ok")
    self.assertEqual(report["nan_count"], 1)
    self.assertEqual(report["inf_count"], 2)
    self.assertEqual(report["reference_nan_count"], 0)
    self.assertEqual(report["reference_inf_count"], 1)
    # The matching +inf is not a mismatch. The NaN and the -inf are.
    self.assertEqual(report["non_finite_mismatch_count"], 2)
    self.assertEqual(report["max_ulp"], 0)

  def test_a_wider_output_is_measured_in_the_reference_dtype(self):
    reference = _bf16([1.0, 2.0])
    candidate = np.asarray([1.0, 2.0], dtype=np.float32)
    report = ulp_report_lib.compare_outputs([[reference]], [[candidate]])
    self.assertEqual(report["status"], "ok")
    self.assertEqual(report["dtype_mismatch_count"], 1)
    self.assertEqual(report["max_ulp"], 0)
    self.assertFalse(report["bit_identical"])

  def test_an_integer_output_against_a_float_reference_is_not_measured(self):
    reference = np.asarray([1.0, 2.0], dtype=np.float32)
    candidate = np.asarray([1, 2], dtype=np.int32)
    report = ulp_report_lib.compare_outputs([[reference]], [[candidate]])
    self.assertEqual(report["status"], "not_measured")
    self.assertIn("dtype_mismatch", report["detail"])
    self.assertIsNone(report["max_ulp"])

  def test_integers_measure_the_difference(self):
    reference = np.asarray([0, 10, 200], dtype=np.uint8)
    candidate = np.asarray([3, 10, 190], dtype=np.uint8)
    report = ulp_report_lib.compare_outputs([[reference]], [[candidate]])
    self.assertEqual(report["max_ulp"], 10)
    self.assertEqual(report["worst_index"], [2])

  def test_one_unmeasured_leaf_makes_the_report_partial(self):
    good = np.asarray([1.0], dtype=np.float32)
    report = ulp_report_lib.compare_outputs(
        [[good, np.zeros((2,), np.float32)]],
        [[good, np.zeros((3,), np.float32)]],
    )
    self.assertEqual(report["status"], "partial")
    self.assertIn("shape_mismatch", report["detail"])
    self.assertEqual(report["leaves_total"], 2)
    self.assertEqual(report["leaves_measured"], 1)

  def test_the_worst_leaf_and_sample_are_reported(self):
    reference = np.asarray([1.0, 1.0], dtype=np.float32)
    off = np.nextafter(np.float32(1.0), np.float32(2.0))
    report = ulp_report_lib.compare_outputs(
        [[reference, reference], [reference, reference]],
        [
            [reference, reference],
            [reference, np.asarray([1.0, off], np.float32)],
        ],
    )
    self.assertEqual(report["max_ulp"], 1)
    self.assertEqual(report["worst_sample"], 1)
    self.assertEqual(report["worst_leaf"], 1)
    self.assertEqual(report["worst_index"], [1])

  def test_a_sample_count_mismatch_is_not_measured(self):
    leaf = np.zeros((1,), np.float32)
    report = ulp_report_lib.compare_outputs([[leaf]], [[leaf], [leaf]])
    self.assertEqual(report["status"], "not_measured")

  def test_bad_input_is_an_error_not_an_exception(self):
    # Typed as Any on purpose: the input breaks the declared contract.
    bad: Any = [[object()]]
    report = ulp_report_lib.compare_outputs(bad, bad)
    self.assertIn(report["status"], ("error", "not_measured"))


if __name__ == "__main__":
  absltest.main()
