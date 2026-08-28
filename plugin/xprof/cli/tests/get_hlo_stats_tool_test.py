"""Unit tests for get_hlo_stats_tool.

These tests verify that get_hlo_stats parses the hlo_stats tool response
correctly, handles sorting by different metrics, supports Markdown and JSON
formatting, and has proper error response paths.
"""

import json
from unittest import mock

from google.protobuf import json_format

from absl.testing import absltest
from absl.testing import parameterized
from xprof.cli.internal import decorators
from xprof.cli.internal.oss import xprof_client
from xprof.cli.tools import get_hlo_stats_tool
from xprof.protobuf import hlo_stats_pb2


class GetHloStatsToolTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    mock_cache = mock.create_autospec(
        decorators.Cache, instance=True, spec_set=True
    )
    mock_cache.get.return_value = decorators.Cache.UNKNOWN
    self.enter_context(
        mock.patch.object(
            decorators,
            "get_cache",
            return_value=mock_cache,
            autospec=True,
            spec_set=True,
        )
    )
    self.mock_client = mock.create_autospec(
        xprof_client.CachedXprofClient, instance=True, spec_set=True
    )
    self.enter_context(
        mock.patch.object(
            xprof_client,
            "get_client",
            return_value=self.mock_client,
            autospec=True,
            spec_set=True,
        )
    )

  def _create_fake_database(self) -> hlo_stats_pb2.HloStatsDatabase:
    db = hlo_stats_pb2.HloStatsDatabase()

    # Record 1: Self Time = 1000us, occurrences = 10, category = Convolution
    r1 = db.hlo_stats_record.add()
    r1.rank = 1
    r1.program_id = 1111111111111111111
    r1.hlo_category = "Convolution"
    r1.hlo_expression = "%convolution.1 = ... calls=conv_sub"
    r1.tf_op_name = "conv2d"
    r1.occurrences = 10
    r1.total_time_in_us = 1200.0
    r1.avg_time_in_us = 120.0
    r1.total_self_time_in_us = 1000.0
    r1.avg_self_time_in_us = 100.0
    r1.total_self_time_as_fraction = 0.5
    r1.measured_flop_rate = 0.8
    r1.flops_v2 = 500.0
    r1.measured_memory_bw = 50.0
    r1.bound_by = "Compute"
    r1.source_info.file_name = "model.py"
    r1.source_info.line_number = 42

    # Record 2: Self Time = 800us, occurrences = 5, category = Fusion
    r2 = db.hlo_stats_record.add()
    r2.rank = 2
    r2.program_id = 2222222222222222222
    r2.hlo_category = "Fusion"
    r2.hlo_expression = "%fusion.2 = ... calls=fusion_sub"
    r2.tf_op_name = "gelu"
    r2.occurrences = 5
    r2.total_time_in_us = 900.0
    r2.avg_time_in_us = 180.0
    r2.total_self_time_in_us = 800.0
    r2.avg_self_time_in_us = 160.0
    r2.total_self_time_as_fraction = 0.4
    r2.measured_flop_rate = 0.2
    r2.flops_v2 = 100.0
    r2.measured_memory_bw = 80.0
    r2.bound_by = "HBM"

    # Record 3: Self Time = 200us, occurrences = 20, category = Tuple
    r3 = db.hlo_stats_record.add()
    r3.rank = 3
    r3.program_id = 3333333333333333333
    r3.hlo_category = "Tuple"
    r3.hlo_expression = "%tuple.3 = tuple()"
    r3.tf_op_name = ""
    r3.occurrences = 20
    r3.total_time_in_us = 200.0
    r3.avg_time_in_us = 10.0
    r3.total_self_time_in_us = 200.0
    r3.avg_self_time_in_us = 10.0
    r3.total_self_time_as_fraction = 0.1
    r3.measured_flop_rate = 0.0
    r3.flops = 10  # Test fallback to flops
    r3.measured_memory_bw = 10.0
    r3.bound_by = "Unknown"

    return db

  def test_get_hlo_stats_markdown_default(self):
    db = self._create_fake_database()
    self.mock_client.fetch.return_value = (None, db.SerializeToString())

    result = get_hlo_stats_tool.get_hlo_stats("session_123")

    self.assertIn("| Rank | Category | Op Name | Occurrences |", result)
    self.assertIn("convolution.1", result)
    self.assertIn("fusion.2", result)
    self.assertIn("tuple.3", result)
    self.assertIn("model.py:42", result)

  def test_get_hlo_stats_json_success(self):
    db = self._create_fake_database()
    self.mock_client.fetch.return_value = (None, db.SerializeToString())

    result_str = get_hlo_stats_tool.get_hlo_stats(
        "session_123", output_format="json"
    )
    records = json.loads(result_str)

    self.assertLen(records, 3)
    self.assertEqual(records[0]["op_name"], "convolution.1")
    self.assertEqual(records[1]["op_name"], "fusion.2")
    self.assertEqual(records[2]["op_name"], "tuple.3")
    self.assertEqual(records[0]["self_time_percent"], 50.0)
    self.assertEqual(records[0]["flops"], 500.0)
    self.assertEqual(records[2]["flops"], 10.0)

  def test_json_fallback_parsing(self):
    db = self._create_fake_database()
    json_str = json_format.MessageToJson(db)
    self.mock_client.fetch.return_value = (None, json_str.encode("utf-8"))

    result_str = get_hlo_stats_tool.get_hlo_stats(
        "session_123", output_format="json"
    )
    records = json.loads(result_str)

    self.assertLen(records, 3)
    self.assertEqual(records[0]["op_name"], "convolution.1")

  def test_direct_return_value_non_tuple(self):
    db = self._create_fake_database()
    self.mock_client.fetch.return_value = db.SerializeToString()

    result_str = get_hlo_stats_tool.get_hlo_stats(
        "session_123", output_format="json"
    )
    records = json.loads(result_str)

    self.assertLen(records, 3)
    self.assertEqual(records[0]["op_name"], "convolution.1")

  def test_op_name_fallback(self):
    db = hlo_stats_pb2.HloStatsDatabase()
    r = db.hlo_stats_record.add()
    r.rank = 1
    r.program_id = 100
    r.hlo_category = "Custom"
    r.hlo_expression = (
        "custom_op_instruction_without_standard_hlo_format_that_is_very_long"
    )
    r.total_self_time_in_us = 500.0

    self.mock_client.fetch.return_value = (None, db.SerializeToString())

    result_str = get_hlo_stats_tool.get_hlo_stats(
        "session_123", output_format="json"
    )
    records = json.loads(result_str)

    self.assertLen(records, 1)
    self.assertEqual(
        records[0]["op_name"],
        "custom_op_instruction_without_standard_hlo_format_that_is_very_long"[
            :80
        ],
    )

  def test_op_name_regex_no_redos_on_multiple_percents(self):
    db = hlo_stats_pb2.HloStatsDatabase()
    r = db.hlo_stats_record.add()
    r.rank = 1
    r.program_id = 100
    r.hlo_category = "Custom"
    r.hlo_expression = "%foo %bar %baz %qux = f32[10] add(%foo, %bar)"
    r.total_self_time_in_us = 500.0

    self.mock_client.fetch.return_value = (None, db.SerializeToString())

    result_str = get_hlo_stats_tool.get_hlo_stats(
        "session_123", output_format="json"
    )
    records = json.loads(result_str)

    self.assertLen(records, 1)
    # %qux is the single token immediately preceding ' =' without internal '%'
    self.assertEqual(records[0]["op_name"], "qux")

  @parameterized.named_parameters(
      ("self_time", "self_time", ["convolution.1", "fusion.2", "tuple.3"]),
      ("total_time", "total_time", ["convolution.1", "fusion.2", "tuple.3"]),
      ("occurrences", "occurrences", ["tuple.3", "convolution.1", "fusion.2"]),
      ("flops", "flops", ["convolution.1", "fusion.2", "tuple.3"]),
      ("bandwidth", "bandwidth", ["fusion.2", "convolution.1", "tuple.3"]),
  )
  def test_sorting(self, sort_by, expected_order):
    db = self._create_fake_database()
    self.mock_client.fetch.return_value = (None, db.SerializeToString())

    result_str = get_hlo_stats_tool.get_hlo_stats(
        "session_123", sort_by=sort_by, output_format="json"
    )
    records = json.loads(result_str)
    actual_order = [r["op_name"] for r in records]
    self.assertEqual(actual_order, expected_order)

  def test_category_filter(self):
    db = self._create_fake_database()
    self.mock_client.fetch.return_value = (None, db.SerializeToString())

    result_str = get_hlo_stats_tool.get_hlo_stats(
        "session_123", category_filter="Convolution", output_format="json"
    )
    records = json.loads(result_str)
    self.assertLen(records, 1)
    self.assertEqual(records[0]["op_name"], "convolution.1")

  def test_limit(self):
    db = self._create_fake_database()
    self.mock_client.fetch.return_value = (None, db.SerializeToString())

    result_str = get_hlo_stats_tool.get_hlo_stats(
        "session_123", limit=2, output_format="json"
    )
    records = json.loads(result_str)
    self.assertLen(records, 2)

  def test_error_fetch_failure(self):
    self.mock_client.fetch.side_effect = RuntimeError("Connection Refused")

    with self.assertRaises(RuntimeError) as cm:
      get_hlo_stats_tool.get_hlo_stats("session_123")
    self.assertIn("Connection Refused", str(cm.exception))

  def test_error_empty_result(self):
    self.mock_client.fetch.return_value = None

    with self.assertRaises(RuntimeError) as cm:
      get_hlo_stats_tool.get_hlo_stats("session_123")
    self.assertIn("Failed to fetch hlo_stats", str(cm.exception))

  def test_error_empty_records(self):
    db = hlo_stats_pb2.HloStatsDatabase()
    self.mock_client.fetch.return_value = (None, db.SerializeToString())

    with self.assertRaises(FileNotFoundError) as cm:
      get_hlo_stats_tool.get_hlo_stats("session_123")
    self.assertIn("No HLO stats records found", str(cm.exception))


if __name__ == "__main__":
  absltest.main()
