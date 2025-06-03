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

"""Tests for dcn_collective_stats_proto_to_gviz."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import enum
import io
import sys

import gviz_api
import tensorflow as tf

from xprof.convert import dcn_collective_stats_proto_to_gviz
from xprof.protobuf import dcn_slack_analysis_pb2


class StrEnum(str, enum.Enum):
  pass


class MockValues(StrEnum):
  DCN_COLLECTIVE_NAME = "collective-1"
  RECV_OP_NAME = "recv-done"
  SEND_OP_NAME = "send"
  SLACK_US = 2
  OBSERVED_DURATION_US = 12
  STALL_DURATION_MS = 5
  OCCURRENCES = 6
  BYTES_TRANSMITTED_OVER_NETWORK = 576012
  REQUIRED_BANDWIDTH = 2.304048


class ProtoToGvizTest(tf.test.TestCase):

  def create_empty_dcn_slack_analysis(self):
    return dcn_slack_analysis_pb2.DcnSlackAnalysis()

  def create_mock_dcn_slack_summary(self, slack=int(MockValues.SLACK_US)):
    dcn_slack_summary = dcn_slack_analysis_pb2.DcnSlackSummary(
        rendezvous=MockValues.DCN_COLLECTIVE_NAME,
        recv_op_name=MockValues.RECV_OP_NAME,
        send_op_name=MockValues.SEND_OP_NAME,
        slack_us=slack * 1000,
        observed_duration_us=int(MockValues.OBSERVED_DURATION_US) * 1000,
        stall_duration_us=int(MockValues.STALL_DURATION_MS) * 1000,
        occurrences=int(MockValues.OCCURRENCES),
        bytes_transmitted_over_network=int(
            MockValues.BYTES_TRANSMITTED_OVER_NETWORK
        ),
    )
    return dcn_slack_summary

  def create_mock_dcn_slack_analysis(self, slack=int(MockValues.SLACK_US)):
    dcn_slack_analysis = dcn_slack_analysis_pb2.DcnSlackAnalysis()
    for _ in range(0, 3):
      dcn_slack_analysis.dcn_slack_summary.append(
          self.create_mock_dcn_slack_summary(slack)
      )
    return dcn_slack_analysis

  def validate_actual_with_expected(
      self, reader, table_description, data, expected
  ):
    for rr, row_values in enumerate(reader):
      if rr == 0:
        # DataTable columns match schema defined in table_description.
        for cc, column_header in enumerate(row_values):
          self.assertEqual(table_description[cc][2], column_header)
      else:
        for cc, cell_str in enumerate(row_values):
          raw_value = data[rr - 1][cc]
          value_type = table_description[cc][1]

          # Only number and strings are used in the DataTable schema.
          self.assertIn(value_type, ["number", "string"])

          # Encode in similar fashion as DataTable.ToCsv().
          expected_value = gviz_api.DataTable.CoerceValue(raw_value, value_type)
          self.assertNotIsInstance(expected_value, tuple)
          self.assertEqual(expected_value, raw_value)

          # Check against expected values we have set in our mock table.
          if value_type == "string":
            self.assertEqual(expected[cc], cell_str)
          else:
            if expected[cc] == MockValues.OCCURRENCES:
              self.assertEqual(str(int(expected[cc])), cell_str)
            else:
              self.assertEqual(str(float(expected[cc])), cell_str)

  def test_dcn_collective_stats_empty(self):
    dcn_slack_analysis = self.create_empty_dcn_slack_analysis()
    data_table = (
        dcn_collective_stats_proto_to_gviz.generate_dcn_collective_stats_table(
            dcn_slack_analysis
        )
    )

    self.assertEqual(0, data_table.NumberOfRows())
    self.assertLen(data_table.columns, 10)

  def test_dcn_collective_stats_table(self):
    dcn_slack_analysis = self.create_mock_dcn_slack_analysis()
    (table_description, data, custom_properties) = (
        dcn_collective_stats_proto_to_gviz.get_dcn_collective_stats_table_args(
            dcn_slack_analysis
        )
    )
    data_table = gviz_api.DataTable(table_description, data, custom_properties)

    self.assertLen(data, 3)
    self.assertEqual(3, data_table.NumberOfRows())
    self.assertLen(table_description, 10)
    self.assertLen(data_table.columns, 10)

    csv_file = io.StringIO(data_table.ToCsv())
    reader = csv.reader(csv_file)

    expected = [
        MockValues.DCN_COLLECTIVE_NAME,
        MockValues.RECV_OP_NAME,
        MockValues.SEND_OP_NAME,
        MockValues.SLACK_US,
        MockValues.OBSERVED_DURATION_US,
        MockValues.STALL_DURATION_MS,
        MockValues.OCCURRENCES,
        int(MockValues.STALL_DURATION_MS) * int(MockValues.OCCURRENCES),
        "576.012 KB",
        MockValues.REQUIRED_BANDWIDTH,
    ]

    self.validate_actual_with_expected(
        reader, table_description, data, expected
    )

  def test_dcn_collective_stats_table_with_zero_slack(self):
    dcn_slack_analysis = self.create_mock_dcn_slack_analysis(0)
    (table_description, data, custom_properties) = (
        dcn_collective_stats_proto_to_gviz.get_dcn_collective_stats_table_args(
            dcn_slack_analysis
        )
    )
    data_table = gviz_api.DataTable(table_description, data, custom_properties)

    self.assertLen(data, 3)
    self.assertEqual(3, data_table.NumberOfRows())
    self.assertLen(table_description, 10)
    self.assertLen(data_table.columns, 10)

    csv_file = io.StringIO(data_table.ToCsv())
    reader = csv.reader(csv_file)

    expected = [
        MockValues.DCN_COLLECTIVE_NAME,
        MockValues.RECV_OP_NAME,
        MockValues.SEND_OP_NAME,
        0,
        MockValues.OBSERVED_DURATION_US,
        MockValues.STALL_DURATION_MS,
        MockValues.OCCURRENCES,
        int(MockValues.STALL_DURATION_MS) * int(MockValues.OCCURRENCES),
        "576.012 KB",
        sys.maxsize,
    ]

    self.validate_actual_with_expected(
        reader, table_description, data, expected
    )


if __name__ == "__main__":
  tf.test.main()
