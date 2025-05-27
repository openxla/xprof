# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for input_pipeline_proto_to_gviz."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import io
import enum

import gviz_api
import tensorflow as tf
# pylint: disable=g-importing-member
from google3.google.protobuf.any_pb2 import Any
# pylint: enable=g-importing-member

from xprof.convert import input_pipeline_proto_to_gviz
from xprof.protobuf import input_pipeline_pb2
from xprof.protobuf import tpu_input_pipeline_pb2


class StrEnum(str, enum.Enum):
  pass


class MockValues(StrEnum):
  STEP_NUMBER = 1
  STEP_TIME_MS = 2
  UNKNOWN_TIME_MS = 3
  HOST_WAIT_INPUT_MS = 11
  HOST_TO_DEVICE_MS = 12
  OUTPUT_MS = 5
  DEVICE_COMPUTE_MS = 6
  DEVICE_TO_DEVICE_MS = 7
  DEVICE_COLLECTIVES_MS = 13
  HOST_COMPUTE_MS = 8
  HOST_PREPARE_MS = 9
  HOST_COMPILE_MS = 10
  TOOLTIP = """Step 1, duration: 2.00 ms
-All others: 3.00 ms
-Compilation: 10.00 ms
-Output: 5.00 ms
-Input: 23.00 ms
-Kernel launch: 9.00 ms
-Host compute: 8.00 ms
-Device collectives: 13.00 ms
-Device to device: 7.00 ms
-Device compute: 6.00 ms"""
  # mock values for tpu input pipeline
  TC_COMPUTE_TIME_MS = 14.0
  TC_INFEED_TIME_MS = 6.0
  TC_OUTFEED_TIME_MS = 8.0
  TC_IDLE_TIME_MS = 2.0
  SCV0_COMPUTE_TIME_MS = 10.0
  SCV0_INFEED_TIME_MS = 7.0
  SCV0_OUTFEED_TIME_MS = 6.0
  SCV0_IDLE_TIME_MS = 7.0
  SC_COMPUTE_TIME_MS = 11.0
  SC_INFEED_TIME_MS = 8.0
  SC_OUTFEED_TIME_MS = 6.0
  SC_IDLE_TIME_MS = 5.0
  HOST_TRANSFER_MS = 12.0
  INFEED_PERCENT_AVERAGE = 17.0
  INFEED_PERCENT_MINIMUM = 16.0
  INFEED_PERCENT_MAXIMUM = 18.0
  TOOLTIP_TPU = (
      "step {}: \nTime waiting for input data = {:.3f} ms, Step time ="
      " {:.3f} ms".format(
          1,
          13.0,
          47.0,
      )
  )


class ProtoToGvizTest(tf.test.TestCase):

  def create_empty_input_pipeline(self):
    return input_pipeline_pb2.InputPipelineAnalysisResult()

  def create_mock_step_summary(self, base):
    step_summary = input_pipeline_pb2.StepSummary()
    step_summary.average = 1 + base
    step_summary.standard_deviation = 2 + base
    step_summary.minimum = 3 + base
    step_summary.maximum = 4 + base
    return step_summary

  def create_mock_input_op_data(
      self, ipa: input_pipeline_pb2.InputPipelineAnalysisResult
  ):
    input_time_breakdown = input_pipeline_pb2.InputTimeBreakdown()
    input_time_breakdown.demanded_file_read_us = 1
    input_time_breakdown.advanced_file_read_us = 2
    input_time_breakdown.preprocessing_us = 3
    input_time_breakdown.enqueue_us = 4
    input_time_breakdown.unclassified_non_enqueue_us = 5
    ipa.input_time_breakdown.CopyFrom(input_time_breakdown)

    for _ in range(0, 3):
      input_op_details = input_pipeline_pb2.InputOpDetails()
      input_op_details.op_name = str(1)
      input_op_details.count = 2
      input_op_details.time_in_ms = 3
      input_op_details.time_in_percent = 4
      input_op_details.self_time_in_ms = 5
      input_op_details.self_time_in_percent = 6
      input_op_details.category = str(7)
      ipa.input_op_details.append(input_op_details)

  def create_mock_recommendation(
      self, ipa: input_pipeline_pb2.InputPipelineAnalysisResult
  ):
    recommendation = input_pipeline_pb2.InputPipelineAnalysisRecommendation()
    for ss in ["a", "b", "c", "d", "e"]:
      recommendation.details.append(ss)
    ipa.recommendation.CopyFrom(recommendation)

  def create_mock_input_pipeline_for_tpu(self, has_legacy_sc=False):
    ipa = input_pipeline_pb2.InputPipelineAnalysisResult()
    ipa.hardware_type = "TPU"
    ipa.tag = True
    ipa.step_time_summary.CopyFrom(self.create_mock_step_summary(10))
    ipa.input_percent_summary.CopyFrom(self.create_mock_step_summary(20))

    # create mock step details for tpu
    # Add 3 rows
    for _ in range(0, 3):
      step_details = tpu_input_pipeline_pb2.PerTpuStepDetails()
      step_details.step_number = int(MockValues.STEP_NUMBER)
      step_details.tc_compute_time_ms = float(MockValues.TC_COMPUTE_TIME_MS)
      step_details.tc_infeed_time_ms = float(MockValues.TC_INFEED_TIME_MS)
      step_details.tc_outfeed_time_ms = float(MockValues.TC_OUTFEED_TIME_MS)
      step_details.tc_idle_time_ms = float(MockValues.TC_IDLE_TIME_MS)
      step_details.scv0_compute_time_ms = float(MockValues.SCV0_COMPUTE_TIME_MS)
      step_details.scv0_infeed_time_ms = float(MockValues.SCV0_INFEED_TIME_MS)
      step_details.host_transfer_ms = float(MockValues.HOST_TRANSFER_MS)
      step_details.infeed_percent_average = float(
          MockValues.INFEED_PERCENT_AVERAGE
      )
      step_details.infeed_percent_minimum = float(
          MockValues.INFEED_PERCENT_MINIMUM
      )
      step_details.infeed_percent_maximum = float(
          MockValues.INFEED_PERCENT_MAXIMUM
      )

      step_details_any = Any()
      step_details_any.Pack(step_details)
      ipa.step_details.append(step_details_any)

    self.create_mock_input_op_data(ipa)
    self.create_mock_recommendation(ipa)

    step_time_breakdown = tpu_input_pipeline_pb2.TpuStepTimeBreakdown()
    step_time_breakdown.tc_compute_ms_summary.CopyFrom(
        self.create_mock_step_summary(1)
    )
    step_time_breakdown.tc_infeed_ms_summary.CopyFrom(
        self.create_mock_step_summary(2)
    )
    step_time_breakdown.tc_outfeed_ms_summary.CopyFrom(
        self.create_mock_step_summary(3)
    )
    step_time_breakdown.tc_idle_ms_summary.CopyFrom(
        self.create_mock_step_summary(4)
    )
    step_time_breakdown.scv0_compute_ms_summary.CopyFrom(
        self.create_mock_step_summary(5)
    )
    step_time_breakdown.scv0_infeed_ms_summary.CopyFrom(
        self.create_mock_step_summary(6)
    )

    if not has_legacy_sc:
      sparse_core_step_summary = tpu_input_pipeline_pb2.SparseCoreStepSummary()
      sparse_core_step_summary.sc_compute_ms_summary.CopyFrom(
          self.create_mock_step_summary(11)
      )
      sparse_core_step_summary.sc_infeed_ms_summary.CopyFrom(
          self.create_mock_step_summary(12)
      )
      sparse_core_step_summary.sc_outfeed_ms_summary.CopyFrom(
          self.create_mock_step_summary(13)
      )
      sparse_core_step_summary.sc_idle_ms_summary.CopyFrom(
          self.create_mock_step_summary(14)
      )
      sparse_core_step_summary.sc_step_time_ms_summary.CopyFrom(
          self.create_mock_step_summary(15)
      )
      step_time_breakdown.sparse_core_step_summary.CopyFrom(
          sparse_core_step_summary
      )
    step_time_breakdown_any = Any()
    step_time_breakdown_any.Pack(step_time_breakdown)
    ipa.step_time_breakdown.CopyFrom(step_time_breakdown_any)

    return ipa

  def create_mock_input_pipeline(self):
    ipa = input_pipeline_pb2.InputPipelineAnalysisResult()
    ipa.hardware_type = "CPU_ONLY"
    ipa.step_time_summary.CopyFrom(self.create_mock_step_summary(10))
    ipa.input_percent_summary.CopyFrom(self.create_mock_step_summary(20))

    # Add 3 rows
    for _ in range(0, 3):
      step_details = input_pipeline_pb2.PerGenericStepDetails()
      step_details.step_number = int(MockValues.STEP_NUMBER)
      step_details.step_name = MockValues.STEP_NUMBER
      step_details.step_time_ms = int(MockValues.STEP_TIME_MS)
      step_details.unknown_time_ms = int(MockValues.UNKNOWN_TIME_MS)
      step_details.host_wait_input_ms = int(MockValues.HOST_WAIT_INPUT_MS)
      step_details.host_to_device_ms = int(MockValues.HOST_TO_DEVICE_MS)
      step_details.output_ms = int(MockValues.OUTPUT_MS)
      step_details.device_compute_ms = int(MockValues.DEVICE_COMPUTE_MS)
      step_details.device_to_device_ms = int(MockValues.DEVICE_TO_DEVICE_MS)
      step_details.device_collectives_ms = int(MockValues.DEVICE_COLLECTIVES_MS)
      step_details.host_compute_ms = int(MockValues.HOST_COMPUTE_MS)
      step_details.host_prepare_ms = int(MockValues.HOST_PREPARE_MS)
      step_details.host_compile_ms = int(MockValues.HOST_COMPILE_MS)

      step_details_any = Any()
      step_details_any.Pack(step_details)
      ipa.step_details.append(step_details_any)

    self.create_mock_input_op_data(ipa)
    self.create_mock_recommendation(ipa)

    step_time_breakdown = input_pipeline_pb2.GenericStepTimeBreakdown()
    step_time_breakdown.unknown_time_ms_summary.CopyFrom(
        self.create_mock_step_summary(1))
    step_time_breakdown.host_wait_input_ms_summary.CopyFrom(
        self.create_mock_step_summary(9))
    step_time_breakdown.host_to_device_ms_summary.CopyFrom(
        self.create_mock_step_summary(10))
    step_time_breakdown.input_ms_summary.CopyFrom(
        self.create_mock_step_summary(11))
    step_time_breakdown.output_ms_summary.CopyFrom(
        self.create_mock_step_summary(3))
    step_time_breakdown.device_compute_ms_summary.CopyFrom(
        self.create_mock_step_summary(4))
    step_time_breakdown.device_to_device_ms_summary.CopyFrom(
        self.create_mock_step_summary(5))
    step_time_breakdown.device_collectives_ms_summary.CopyFrom(
        self.create_mock_step_summary(12))
    step_time_breakdown.host_compute_ms_summary.CopyFrom(
        self.create_mock_step_summary(6))
    step_time_breakdown.host_prepare_ms_summary.CopyFrom(
        self.create_mock_step_summary(7))
    step_time_breakdown.host_compile_ms_summary.CopyFrom(
        self.create_mock_step_summary(8))

    step_time_breakdown_any = Any()
    step_time_breakdown_any.Pack(step_time_breakdown)
    ipa.step_time_breakdown.CopyFrom(step_time_breakdown_any)

    return ipa

  def test_input_pipeline_empty(self):
    ipa = self.create_empty_input_pipeline()
    data_table = input_pipeline_proto_to_gviz.generate_step_breakdown_table(ipa)

    self.assertEqual(0, data_table.NumberOfRows(),
                     "Empty table should have 0 rows.")
    # Input pipeline chart data table has 12 columns.
    self.assertLen(data_table.columns, 12)

  def test_input_pipeline_step_breakdown(self):
    ipa = self.create_mock_input_pipeline()
    (table_description, data, custom_properties
    ) = input_pipeline_proto_to_gviz.get_step_breakdown_table_args(ipa)
    data_table = gviz_api.DataTable(table_description, data, custom_properties)

    # Data is a list of 3 rows.
    self.assertLen(data, 3)
    self.assertEqual(3, data_table.NumberOfRows(), "Simple table has 3 rows.")
    # Table descriptor is a list of 12 columns.
    self.assertLen(table_description, 12)
    # DataTable also has 12 columns.
    self.assertLen(data_table.columns, 12)
    # The step time graph column ids property is a string
    #  with valid column name separated by comma.
    print(custom_properties)
    self.assertEqual(
        custom_properties["step_time_graph_column_ids"],
        "stepnum,deviceComputeTimeMs,deviceToDeviceTimeMs,deviceCollectivesTimeMs,hostComputeTimeMs,kernelLaunchTimeMs,infeedTimeMs,outfeedTimeMs,compileTimeMs,otherTimeMs,tooltip",
    )

    csv_file = io.StringIO(data_table.ToCsv())
    reader = csv.reader(csv_file)

    expected = [
        str(int(MockValues.STEP_NUMBER)),
        int(MockValues.DEVICE_COMPUTE_MS),
        int(MockValues.DEVICE_TO_DEVICE_MS),
        int(MockValues.DEVICE_COLLECTIVES_MS),
        int(MockValues.HOST_COMPUTE_MS),
        int(MockValues.HOST_PREPARE_MS),
        int(MockValues.HOST_WAIT_INPUT_MS) + int(MockValues.HOST_TO_DEVICE_MS),
        int(MockValues.OUTPUT_MS),
        int(MockValues.HOST_COMPILE_MS),
        int(MockValues.UNKNOWN_TIME_MS),
        MockValues.TOOLTIP.value,
        int(MockValues.STEP_TIME_MS),
    ]

    for (rr, row_values) in enumerate(reader):
      if rr == 0:
        # DataTable columns match schema defined in table_description.
        for (cc, column_header) in enumerate(row_values):
          self.assertEqual(table_description[cc][2], column_header)
      else:
        for (cc, cell_str) in enumerate(row_values):
          raw_value = data[rr - 1][cc]
          value_type = table_description[cc][1]

          # Only number and strings are used in our DataTable schema.
          self.assertIn(value_type, ["number", "string"])

          # Encode in similar fashion as DataTable.ToCsv().
          expected_value = gviz_api.DataTable.CoerceValue(raw_value, value_type)
          self.assertNotIsInstance(expected_value, tuple)
          self.assertEqual(expected_value, raw_value)
          self.assertEqual(str(expected_value), cell_str)

          # Check against expected values we have set in our mock table.
          if isinstance(expected[cc], str):
            self.assertEqual(expected[cc], cell_str)
          else:
            self.assertEqual(str(float(expected[cc])), cell_str)

  def test_input_pipeline_step_breakdown_for_tpu(self):
    ipa = self.create_mock_input_pipeline_for_tpu()
    (table_description, data, custom_properties) = (
        input_pipeline_proto_to_gviz.get_step_breakdown_table_args_for_tpu(ipa)
    )
    data_table = gviz_api.DataTable(table_description, data, custom_properties)

    # Data is a list of 3 rows.
    self.assertLen(data, 3)
    self.assertEqual(3, data_table.NumberOfRows(), "Simple table has 3 rows.")
    # Table descriptor is a list of 12 columns
    # because has_sc_summary_legacy is true.
    self.assertLen(table_description, 10)
    # DataTable also has 12 columns.
    self.assertLen(data_table.columns, 10)
    # The step time graph column ids property is a string
    # with valid column name separated by comma.
    self.assertEqual(
        custom_properties["step_time_graph_column_ids"],
        "stepnum,tcComputeTimeMs,tcInfeedTimeMs,tcOutfeedTimeMs,tcIdleTimeMs,hostTransferTimeMs,tooltip",
    )

    csv_file = io.StringIO(data_table.ToCsv())
    reader = csv.reader(csv_file)

    expected = [
        int(MockValues.STEP_NUMBER),
        float(MockValues.TC_COMPUTE_TIME_MS),
        float(MockValues.TC_INFEED_TIME_MS),
        float(MockValues.TC_OUTFEED_TIME_MS),
        float(MockValues.TC_IDLE_TIME_MS),
        float(MockValues.HOST_TRANSFER_MS),
        MockValues.TOOLTIP_TPU.value,
        float(MockValues.INFEED_PERCENT_AVERAGE),
        float(MockValues.INFEED_PERCENT_MINIMUM),
        float(MockValues.INFEED_PERCENT_MAXIMUM),
    ]

    for rr, row_values in enumerate(reader):
      if rr == 0:
        # DataTable columns match schema defined in table_description.
        for cc, column_header in enumerate(row_values):
          self.assertEqual(table_description[cc][2], column_header)
      else:
        for cc, cell_str in enumerate(row_values):
          raw_value = data[rr - 1][cc]
          value_type = table_description[cc][1]

          # Only number and strings are used in our DataTable schema.
          self.assertIn(value_type, ["number", "string"])

          # Encode in similar fashion as DataTable.ToCsv().
          expected_value = gviz_api.DataTable.CoerceValue(raw_value, value_type)
          self.assertNotIsInstance(expected_value, tuple)
          self.assertEqual(expected_value, raw_value)
          self.assertEqual(str(expected_value), cell_str)

          # Check against expected values we have set in our mock table.
          if isinstance(expected[cc], str):
            self.assertEqual(expected[cc], cell_str)
          else:
            self.assertEqual(str(expected[cc]), cell_str)

  def test_input_pipeline_step_breakdown_for_tpu_with_legacy_sc(self):
    ipa = self.create_mock_input_pipeline_for_tpu(has_legacy_sc=True)
    (table_description, data, custom_properties) = (
        input_pipeline_proto_to_gviz.get_step_breakdown_table_args_for_tpu(ipa)
    )
    data_table = gviz_api.DataTable(table_description, data, custom_properties)

    # Data is a list of 3 rows.
    self.assertLen(data, 3)
    self.assertEqual(3, data_table.NumberOfRows(), "Simple table has 3 rows.")
    # Table descriptor is a list of 12 columns
    # because has_sc_summary_legacy is true.
    self.assertLen(table_description, 12)
    # DataTable also has 12 columns.
    self.assertLen(data_table.columns, 12)

    csv_file = io.StringIO(data_table.ToCsv())
    reader = csv.reader(csv_file)

    expected = [
        int(MockValues.STEP_NUMBER),
        float(MockValues.TC_COMPUTE_TIME_MS),
        float(MockValues.SCV0_COMPUTE_TIME_MS),
        float(MockValues.SCV0_INFEED_TIME_MS),
        float(MockValues.TC_INFEED_TIME_MS),
        float(MockValues.TC_OUTFEED_TIME_MS),
        float(MockValues.TC_IDLE_TIME_MS),
        float(MockValues.HOST_TRANSFER_MS),
        MockValues.TOOLTIP_TPU.value,
        float(MockValues.INFEED_PERCENT_AVERAGE),
        float(MockValues.INFEED_PERCENT_MINIMUM),
        float(MockValues.INFEED_PERCENT_MAXIMUM),
    ]

    for rr, row_values in enumerate(reader):
      if rr == 0:
        # DataTable columns match schema defined in table_description.
        for cc, column_header in enumerate(row_values):
          self.assertEqual(table_description[cc][2], column_header)
      else:
        for cc, cell_str in enumerate(row_values):
          raw_value = data[rr - 1][cc]
          value_type = table_description[cc][1]

          # Only number and strings are used in our DataTable schema.
          self.assertIn(value_type, ["number", "string"])

          # Encode in similar fashion as DataTable.ToCsv().
          expected_value = gviz_api.DataTable.CoerceValue(raw_value, value_type)
          self.assertNotIsInstance(expected_value, tuple)
          self.assertEqual(expected_value, raw_value)
          self.assertEqual(str(expected_value), cell_str)

          # Check against expected values we have set in our mock table.
          if isinstance(expected[cc], str):
            self.assertEqual(expected[cc], cell_str)
          else:
            self.assertEqual(str(expected[cc]), cell_str)

  def test_generate_max_infeed_core_table(self):
    ipa = input_pipeline_pb2.InputPipelineAnalysisResult()
    ipa.hardware_type = "TPU"
    ipa.tag = True
    for step_number in range(0, 3):
      step_details = tpu_input_pipeline_pb2.PerTpuStepDetails()
      step_details.step_number = step_number
      step_details.max_infeed_time_core_name = "core_name"
      step_details_any = Any()
      step_details_any.Pack(step_details)
      ipa.step_details.append(step_details_any)
    data_table = input_pipeline_proto_to_gviz.generate_max_infeed_core_table(
        ipa
    )
    self.assertEqual(2, data_table.NumberOfRows())
    for row in csv.reader(io.StringIO(data_table.ToCsv())):
      self.assertIn(row[0], ["Index", "Step Number", "Core Name"])
      if row[0] == "Index":
        # Process column headers.
        for idx, col in enumerate(row[1:]):
          self.assertEqual(col, str(idx))
      elif row[0] == "Step Number":
        for idx, step_number in enumerate(row[1:]):
          self.assertEqual(step_number, str(idx))
      else:
        self.assertEqual(row[0], "Core Name")
        for core_name in row[1:]:
          self.assertEqual(core_name, "core_name")


if __name__ == "__main__":
  tf.test.main()
