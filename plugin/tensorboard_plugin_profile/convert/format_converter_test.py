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

"""Tests for the format_converter module."""

import textwrap
import unittest

from tensorboard_plugin_profile.convert import format_converter


class TestJsonToCsv(unittest.TestCase):

  def test_empty_json(self):
    with self.assertRaises(ValueError):
      format_converter.json_to_csv("")

  def test_valid_json_list(self):
    json_data = """
      [
        {"name": "Alice", "age": 30},
        {"name": "Bob", "age": 25}
      ]
    """
    expected_csv = textwrap.dedent("""\
      name,age
      Alice,30
      Bob,25
      """)
    self.assertEqual(format_converter.json_to_csv(json_data), expected_csv)

  def test_valid_json_list_with_columns_order(self):
    json_data = """
      [
        {"name": "Alice", "age": 30, "city": "Wonderland"},
        {"name": "Bob", "age": 25, "city": "Neverland"}
      ]
    """
    expected_csv = textwrap.dedent("""\
      city,name,age
      Wonderland,Alice,30
      Neverland,Bob,25
      """)
    self.assertEqual(
        format_converter.json_to_csv(
            json_data, columns_order=["city", "name", "age"]
        ),
        expected_csv,
    )

  def test_invalid_json(self):
    json_data = "{'name': 'Alice', 'age': 30}"
    with self.assertRaises(ValueError):
      format_converter.json_to_csv(json_data)

  def test_empty_list(self):
    self.assertEqual(format_converter.json_to_csv("[]"), "")

  def test_invalid_columns_order(self):
    json_data = """
      [
        {"name": "Alice", "age": 30},
        {"name": "Bob", "age": 25}
      ]
    """
    with self.assertRaises(ValueError):
      format_converter.json_to_csv(json_data, columns_order=["name", "city"])

  def test_json_with_nested_objects(self):
    json_data = """
      [
        {"name": "Alice", "address": {"street": "123 Main St", "city": "Anytown"}},
        {"name": "Bob", "address": {"street": "456 Oak Ave", "city": "Otherville"}}
      ]
    """
    expected_csv = textwrap.dedent("""\
      name,address
      Alice,"{'street': '123 Main St', 'city': 'Anytown'}"
      Bob,"{'street': '456 Oak Ave', 'city': 'Otherville'}"
      """)
    self.assertEqual(format_converter.json_to_csv(json_data), expected_csv)

  def test_json_with_boolean(self):
    json_data = """
      [
        {"name": "Alice", "is_active": true},
        {"name": "Bob", "is_active": false}
      ]
    """
    expected_csv = textwrap.dedent("""\
      name,is_active
      Alice,true
      Bob,false
      """)
    self.assertEqual(format_converter.json_to_csv(json_data), expected_csv)


if __name__ == "__main__":
  unittest.main()
