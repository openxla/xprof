# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for the csv_writer module."""

import json
import unittest
from xprof.convert import csv_writer


class TestJsonToCsv(unittest.TestCase):

  def test_valid_json(self):
    json_string = json.dumps({"a": 1, "b": "z", "c": 2})
    expected_csv = "a,b,c\n1,z,2\n"
    self.assertEqual(csv_writer.json_to_csv(json_string), expected_csv)

  def test_valid_json_with_columns_order(self):
    json_string = json.dumps({"a": 1, "b": "z", "c": 2})
    columns_order = ["c", "a", "b"]
    expected_csv = "c,a,b\n2,1,z\n"
    self.assertEqual(
        csv_writer.json_to_csv(json_string, columns_order=columns_order),
        expected_csv,
    )

  def test_valid_json_with_separator(self):
    json_string = json.dumps({"a": 1, "b": "z", "c": 2})
    expected_csv = "a;b;c\n1;z;2\n"
    self.assertEqual(
        csv_writer.json_to_csv(json_string, separator=";"), expected_csv
    )

  def test_empty_json(self):
    json_string = json.dumps({})
    expected_csv = ""
    self.assertEqual(csv_writer.json_to_csv(json_string), expected_csv)

  def test_invalid_json(self):
    json_string = "invalid json"
    with self.assertRaisesRegex(ValueError, "Invalid JSON string"):
      csv_writer.json_to_csv(json_string)

  def test_json_not_a_dict(self):
    json_string = json.dumps([1, 2, 3])
    with self.assertRaisesRegex(
        ValueError, "JSON data must be a single object"
    ):
      csv_writer.json_to_csv(json_string)

  def test_json_with_boolean(self):
    json_string = json.dumps({"a": True, "b": False})
    expected_csv = "a,b\ntrue,false\n"
    self.assertEqual(csv_writer.json_to_csv(json_string), expected_csv)

  def test_invalid_columns_order(self):
    json_string = json.dumps({"a": 1, "b": "z"})
    columns_order = ["a", "b", "c"]
    with self.assertRaisesRegex(
        ValueError, "columns_order must be a list of all column IDs"
    ):
      csv_writer.json_to_csv(json_string, columns_order=columns_order)

  def test_json_with_none_value(self):
    json_string = json.dumps({"a": 1, "b": None})
    expected_csv = "a,b\n1,\n"
    self.assertEqual(csv_writer.json_to_csv(json_string), expected_csv)

  def test_valid_json_list_input(self):
    """Test standard input structure: a list containing one data dictionary."""
    input_data = [{
        "cols": [{"label": "Rank"}, {"label": "Op"}],
        "rows": [{"c": [{"v": 1.0}, {"v": "Add"}]}]
    }]
    result = csv_writer.json_to_csv_string(input_data)
    # Expect quoted values because of csv.QUOTE_ALL
    expected = '"Rank","Op"\n"1.0","Add"\n'
    self.assertEqual(result, expected)

  def test_byte_string_input(self):
    """Test that byte strings (b'...') are correctly decoded and parsed."""
    input_bytes = b'[{"cols":[{"label":"ID"}],"rows":[{"c":[{"v":"A1"}]}]}]'
    result = csv_writer.json_to_csv_string(input_bytes)
    expected = '"ID"\n"A1"\n'
    self.assertEqual(result, expected)

  def test_handle_multiline_stack_traces(self):
    """Test that quoting correctly handles newlines within a cell."""
    multiline_val = "line1\nline2"
    input_data = {
        "cols": [{"label": "Trace"}],
        "rows": [{"c": [{"v": multiline_val}]}]
    }
    result = csv_writer.json_to_csv_string(input_data)
    # The newline should be contained within the double quotes
    expected = '"Trace"\n"line1\nline2"\n'
    self.assertEqual(result, expected)

  def test_missing_cols_error(self):
    """Test that the function returns an error string if 'cols' is missing."""
    input_data = [{"invalid": "data"}]
    with self.assertRaisesRegex(ValueError, "Data format not suitable"):
      csv_writer.json_to_csv_string(input_data)

  def test_invalid_json_string(self):
    """Test behavior when an unparseable string is provided."""
    with self.assertRaisesRegex(ValueError, "Invalid JSON string"):
      csv_writer.json_to_csv_string("{ invalid json ]")

  def test_none_and_missing_values(self):
    """Test that null values in JSON are converted to empty quoted strings."""
    input_data = {
        "cols": [{"label": "Value"}],
        "rows": [{"c": [{"v": None}]}, {"c": []}]
    }
    result = csv_writer.json_to_csv_string(input_data)
    # Row 1 is empty string, Row 2 is empty list resulting in no row writer call
    # (per your logic)
    expected = '"Value"\n""\n'
    self.assertEqual(result, expected)

  def test_empty_list(self):
    """Test that an empty list returns an empty string."""
    self.assertEqual(csv_writer.json_to_csv_string([]), "")


if __name__ == "__main__":
  unittest.main()
