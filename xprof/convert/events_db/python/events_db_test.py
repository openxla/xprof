# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for events_db Python bindings (Schema and Record)."""

import operator

from absl.testing import absltest
from absl.testing import parameterized

from xprof.convert.events_db.python import events_db


class EventsDbSchemaAndRecordTest(parameterized.TestCase):

  def test_schema_initial_state(self):
    schema = events_db.Schema()
    self.assertEqual((len(schema), schema.size), (0, 0))

  def test_schema_register_duplicate_returns_same_index(self):
    schema = events_db.Schema()
    fn1 = schema.register_field_name("name")
    fn2 = schema.register_field_name("name")
    self.assertEqual((fn1, len(schema)), (fn2, 1))

  @parameterized.named_parameters(
      ("name", "name", "name"),
      ("duration", "duration_ns", "duration_ns"),
  )
  def test_schema_get_field_name(self, field_name, expected):
    schema = events_db.Schema()
    fn = schema.register_field_name(field_name)
    self.assertEqual(schema.get_field_name(fn), expected)

  def test_schema_get_field_name_invalid(self):
    schema = events_db.Schema()
    self.assertIsNone(schema.get_field_name(events_db.FieldIndex()))

  @parameterized.named_parameters(
      ("existing", "name", True),
      ("missing", "missing", False),
  )
  def test_schema_lookup_field_index(self, lookup_name, should_exist):
    schema = events_db.Schema()
    registered = schema.register_field_name("name")
    result = schema.lookup_field_index(lookup_name)
    if should_exist:
      self.assertEqual(result, registered)
    else:
      self.assertIsNone(result)

  def test_schema_max_field_count(self):
    schema = events_db.Schema(max_field_count=2)
    f1 = schema.register_field_name("f1")
    f2 = schema.register_field_name("f2")
    f3 = schema.register_field_name("f3")
    self.assertEqual(
        (schema.max_field_count, f1.is_valid(), f2.is_valid(), f3.is_valid()),
        (2, True, True, False),
    )

  @parameterized.named_parameters(
      ("equal_self", operator.eq, True, True),
      ("equal_distinct", operator.eq, False, False),
      ("not_equal", operator.ne, False, True),
      ("less", operator.lt, False, True),
      ("less_equal", operator.le, False, True),
      ("greater", operator.gt, False, False),
      ("greater_equal", operator.ge, False, False),
  )
  def test_field_index_comparison(self, op, compare_self, expected):
    schema = events_db.Schema()
    f1 = schema.register_field_name("f1")
    f2 = schema.register_field_name("f2")
    other = f1 if compare_self else f2
    self.assertEqual(op(f1, other), expected)

  @parameterized.named_parameters(
      ("valid", True, "valid"),
      ("invalid", False, "invalid"),
  )
  def test_field_index_validity_and_repr(self, make_valid, expected_substr):
    schema = events_db.Schema()
    field = (
        schema.register_field_name("f")
        if make_valid
        else events_db.FieldIndex()
    )
    self.assertEqual(
        (field.is_valid(), expected_substr in repr(field)),
        (make_valid, True),
    )

  def test_field_index_usable_as_dict_key(self):
    schema = events_db.Schema()
    fn = schema.register_field_name("name")
    test_dict = {fn: "name_val"}
    self.assertEqual(test_dict[fn], "name_val")

if __name__ == "__main__":
  absltest.main()
