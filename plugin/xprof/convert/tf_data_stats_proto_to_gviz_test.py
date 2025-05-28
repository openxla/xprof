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

"""Tests for tf_data_stats_proto_to_gviz."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import io
import sys

import gviz_api
import tensorflow as tf

from xprof.convert import tf_data_stats_proto_to_gviz
from xprof.protobuf import tf_data_stats_pb2


class MockValues:
  IS_INPUT_BOUND = True
  SUMMARY = "Summary"
  BOTTLENECK_HOST_NAME = "bottleneck_host"
  BOTTLENECK_INPUT_PIPELINE_NAME = "bottleneck_input_pipeline"
  BOTTLENECK_MAX_LATENCY_PS = 100_000_000
  BOTTLENECK_ITERATOR_NAME = "Iterator::Map"
  BOTTLENECK_ITERATOR_LONG_NAME = "Iterator::Prefetch::Map"
  BOTTLENECK_ITERATOR_LATENCY_PS = 80_000_000
  SUGGESTION = "Suggestion"
  HOST_NAME = "host1"
  FIRST_ITERATOR_ID = "123"
  FIRST_ITERATOR_NAME = "Iterator::Prefetch"
  FIRST_ITERATOR_LONG_NAME = "Iterator::Prefetch"
  SECOND_ITERATOR_ID = "456"
  SECOND_ITERATOR_NAME = "Iterator::Map"
  SECOND_ITERATOR_LONG_NAME = "Iterator::Prefetch::Map"
  INPUT_PIPELINE_ID = "123"
  INPUT_PIPELINE_NAME = "Host:0"
  AVG_LATENCY_PS = 100_000_000
  MIN_LATENCY_PS = 100_000_000
  MAX_LATENCY_PS = 100_000_000
  NUM_SLOW_CALLS = 1
  FIRST_START_TIME_PS = 10_000_000
  FIRST_DURATION_PS = 100_000_000
  FIRST_SELF_TIME_PS = 30_000_000
  SECOND_START_TIME_PS = 20_000_000
  SECOND_DURATION_PS = 70_000_000
  SECOND_SELF_TIME_PS = 70_000_000
  IS_BLOCKING = True
  NUM_CALLS = 1


class ProtoToGvizTest(tf.test.TestCase):

  def create_empty_combined_tf_data_stats(self):
    return tf_data_stats_pb2.CombinedTfDataStats()

  def create_mock_combined_tf_data_stats(self):
    combined_tf_data_stats = tf_data_stats_pb2.CombinedTfDataStats()
    combined_tf_data_stats.is_input_bound = MockValues.IS_INPUT_BOUND
    combined_tf_data_stats.summary = MockValues.SUMMARY

    bottleneck_analysis = tf_data_stats_pb2.TfDataBottleneckAnalysis()
    bottleneck_analysis.host = MockValues.BOTTLENECK_HOST_NAME
    bottleneck_analysis.input_pipeline = MockValues.BOTTLENECK_INPUT_PIPELINE_NAME
    bottleneck_analysis.max_latency_ps = int(
        MockValues.BOTTLENECK_MAX_LATENCY_PS)
    bottleneck_analysis.iterator_name = MockValues.BOTTLENECK_ITERATOR_NAME
    bottleneck_analysis.iterator_long_name = MockValues.BOTTLENECK_ITERATOR_LONG_NAME
    bottleneck_analysis.iterator_latency_ps = int(
        MockValues.BOTTLENECK_ITERATOR_LATENCY_PS)
    bottleneck_analysis.suggestion = MockValues.SUGGESTION
    combined_tf_data_stats.bottleneck_analysis.append(bottleneck_analysis)

    tf_data_stats = combined_tf_data_stats.tf_data_stats[MockValues.HOST_NAME]
    iterator_metadata = tf_data_stats.iterator_metadata[int(
        MockValues.FIRST_ITERATOR_ID)]
    iterator_metadata.id = int(MockValues.FIRST_ITERATOR_ID)
    iterator_metadata.name = MockValues.FIRST_ITERATOR_NAME
    iterator_metadata.long_name = MockValues.FIRST_ITERATOR_LONG_NAME
    iterator_metadata = tf_data_stats.iterator_metadata[int(
        MockValues.SECOND_ITERATOR_ID)]
    iterator_metadata.id = int(MockValues.SECOND_ITERATOR_ID)
    iterator_metadata.parent_id = int(MockValues.FIRST_ITERATOR_ID)
    iterator_metadata.name = MockValues.SECOND_ITERATOR_NAME
    iterator_metadata.long_name = MockValues.SECOND_ITERATOR_LONG_NAME

    input_pipeline_stats = tf_data_stats.input_pipelines[int(
        MockValues.INPUT_PIPELINE_ID)]
    metadata = input_pipeline_stats.metadata
    metadata.id = int(MockValues.INPUT_PIPELINE_ID)
    metadata.type = tf_data_stats_pb2.InputPipelineMetadata.InputPipelineType.HOST
    metadata.name = MockValues.INPUT_PIPELINE_NAME
    input_pipeline_stats.avg_latency_ps = int(MockValues.AVG_LATENCY_PS)
    input_pipeline_stats.min_latency_ps = int(MockValues.MIN_LATENCY_PS)
    input_pipeline_stats.max_latency_ps = int(MockValues.MAX_LATENCY_PS)
    input_pipeline_stats.num_slow_calls = int(MockValues.NUM_SLOW_CALLS)

    input_pipeline_stat = input_pipeline_stats.stats.add()
    input_pipeline_stat.bottleneck_iterator_id = int(
        MockValues.SECOND_ITERATOR_ID)
    iterator_stat = input_pipeline_stat.iterator_stats[int(
        MockValues.FIRST_ITERATOR_ID)]
    iterator_stat.id = int(MockValues.FIRST_ITERATOR_ID)
    iterator_stat.start_time_ps = int(MockValues.FIRST_START_TIME_PS)
    iterator_stat.duration_ps = int(MockValues.FIRST_DURATION_PS)
    iterator_stat.self_time_ps = int(MockValues.FIRST_SELF_TIME_PS)
    iterator_stat.is_blocking = MockValues.IS_BLOCKING
    iterator_stat.num_calls = int(MockValues.NUM_CALLS)
    iterator_stat = input_pipeline_stat.iterator_stats[int(
        MockValues.SECOND_ITERATOR_ID)]
    iterator_stat.id = int(MockValues.SECOND_ITERATOR_ID)
    iterator_stat.start_time_ps = int(MockValues.SECOND_START_TIME_PS)
    iterator_stat.duration_ps = int(MockValues.SECOND_DURATION_PS)
    iterator_stat.self_time_ps = int(MockValues.SECOND_SELF_TIME_PS)
    iterator_stat.is_blocking = MockValues.IS_BLOCKING
    iterator_stat.num_calls = int(MockValues.NUM_CALLS)

    return combined_tf_data_stats

  def check_data_table(self, table_description, data, data_table, expected):
    self.assertLen(data, len(expected))
    self.assertEqual(data_table.NumberOfRows(), len(expected),
                     "Bottleneck table should have 1 row")
    if not expected:
      return
    self.assertLen(table_description, len(expected[0]))
    self.assertLen(data_table.columns, len(expected[0]))

    csv_file = io.StringIO(data_table.ToCsv())
    reader = csv.reader(csv_file)
    for (rr, row_values) in enumerate(reader):
      if rr == 0:
        # DataTable columns match schema defined in table_description.
        for (cc, column_header) in enumerate(row_values):
          self.assertEqual(table_description[cc][2], column_header)
      else:
        for (cc, cell_str) in enumerate(row_values):
          raw_value = data[rr - 1][cc]
          value_type = table_description[cc][1]
          print(raw_value)
          print(str(expected[rr - 1][cc]))

          # Only number and strings are used in our DataTable schema.
          self.assertIn(value_type, ["boolean", "number", "string"])

          # Encode in similar fashion as DataTable.ToCsv().
          expected_value = gviz_api.DataTable.CoerceValue(raw_value, value_type)
          self.assertEqual(expected_value, raw_value)
          if isinstance(expected_value, tuple):
            expected_value = expected_value[0]

          # Check against expected values we have set in our mock table.
          if value_type == "boolean":
            # CSV converts boolean to lower case.
            self.assertEqual(str(expected_value).lower(), cell_str)
            self.assertEqual(expected[rr - 1][cc].lower(), cell_str)
          elif value_type == "string":
            self.assertEqual(expected[rr - 1][cc], cell_str)
          elif value_type == "number":
            self.assertEqual(str(expected_value), cell_str)
            if "." in cell_str:
              self.assertEqual(str(float(expected[rr - 1][cc])), cell_str)
            else:
              self.assertEqual(str(int(expected[rr - 1][cc])), cell_str)

  def test_empty_tf_data_stats(self):
    combined_tf_data_stats = self.create_empty_combined_tf_data_stats()
    data_table = tf_data_stats_proto_to_gviz.generate_summary_table(
        combined_tf_data_stats)

    self.assertEqual(0, data_table.NumberOfRows(),
                     "Empty table should have 0 rows.")
    # Kernel stats chart data table has 7 columns.
    self.assertLen(data_table.columns, 7)

  def test_graph_table(self):
    combined_tf_data_stats = self.create_mock_combined_tf_data_stats()
    (table_description, data, custom_properties
    ) = tf_data_stats_proto_to_gviz.get_graph_table_args(combined_tf_data_stats)
    data_table = gviz_api.DataTable(table_description, data, custom_properties)
    expected = [
        [
            MockValues.HOST_NAME,
            MockValues.INPUT_PIPELINE_NAME,
            0,
            MockValues.FIRST_ITERATOR_ID,
            "",
            1,
        ],
        [
            MockValues.HOST_NAME,
            MockValues.INPUT_PIPELINE_NAME,
            0,
            MockValues.SECOND_ITERATOR_ID,
            MockValues.FIRST_ITERATOR_ID,
            2,
        ],
    ]
    self.check_data_table(table_description, data, data_table, expected)

  def test_summary_table(self):
    combined_tf_data_stats = self.create_mock_combined_tf_data_stats()
    (table_description, data,
     custom_properties) = tf_data_stats_proto_to_gviz.get_summary_table_args(
         combined_tf_data_stats)
    data_table = gviz_api.DataTable(table_description, data, custom_properties)
    expected = [[
        MockValues.HOST_NAME,
        MockValues.INPUT_PIPELINE_NAME,
        int(MockValues.MIN_LATENCY_PS) / 1000_000,
        int(MockValues.AVG_LATENCY_PS) / 1000_000,
        int(MockValues.MAX_LATENCY_PS) / 1000_000,
        int(MockValues.NUM_CALLS),
        int(MockValues.NUM_SLOW_CALLS),
    ]]
    self.check_data_table(table_description, data, data_table, expected)

  def test_bottleneck_analysis_table(self):
    combined_tf_data_stats = self.create_mock_combined_tf_data_stats()
    (table_description, data, custom_properties
    ) = tf_data_stats_proto_to_gviz.get_bottleneck_analysis_table_args(
        combined_tf_data_stats)
    data_table = gviz_api.DataTable(table_description, data, custom_properties)
    expected = [[
        MockValues.BOTTLENECK_HOST_NAME,
        MockValues.BOTTLENECK_INPUT_PIPELINE_NAME,
        int(MockValues.BOTTLENECK_MAX_LATENCY_PS) / 1000_000,
        tf_data_stats_proto_to_gviz.format_bottleneck(
            MockValues.BOTTLENECK_ITERATOR_NAME,
            MockValues.BOTTLENECK_ITERATOR_LONG_NAME,
            int(MockValues.BOTTLENECK_ITERATOR_LATENCY_PS)),
        MockValues.SUGGESTION
    ]]
    self.check_data_table(table_description, data, data_table, expected)


if __name__ == "__main__":
  tf.test.main()
