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
"""Tests for events_db Python bindings."""

from collections.abc import Sequence
import copy
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

  def test_record_empty_initial_state(self):
    schema = events_db.Schema()
    fn = schema.register_field_name("name")
    record = events_db.Record()
    self.assertEqual(
        (len(record), record.has_field(fn), fn in record, record.get(fn)),
        (0, False, False, None),
    )

  def test_record_get_default_value(self):
    schema = events_db.Schema()
    fn = schema.register_field_name("name")
    record = events_db.Record()
    self.assertEqual(record.get(fn, default="fallback"), "fallback")

  def test_record_getitem_missing_field_raises_key_error(self):
    schema = events_db.Schema()
    fn = schema.register_field_name("name")
    record = events_db.Record()
    with self.assertRaisesRegex(KeyError, "Field not found in Record"):
      _ = record[fn]

  def test_record_setitem_invalid_field_raises_value_error(self):
    record = events_db.Record()
    invalid_field = events_db.FieldIndex()
    with self.assertRaisesRegex(
        ValueError, "Cannot set field on invalid FieldIndex"
    ):
      record[invalid_field] = "value"

  @parameterized.named_parameters(
      ("bool", "is_kernel", True),
      ("str", "kernel_name", "matmul_kernel"),
      ("int", "duration_ns", 5000),
      ("float", "utilization", 0.85),
  )
  def test_record_scalar_roundtrip(self, field_name, value):
    schema = events_db.Schema()
    fn = schema.register_field_name(field_name)
    record = events_db.Record()
    record[fn] = value
    self.assertEqual(
        (record.has_field(fn), record.get(fn), record[fn]),
        (True, value, value),
    )

  def test_record_overwrite_field(self):
    schema = events_db.Schema()
    fn = schema.register_field_name("device")
    record = events_db.Record()
    record[fn] = "/device:TPU:0"
    record[fn] = "/device:GPU:0"
    self.assertEqual(record[fn], "/device:GPU:0")

  def test_record_clear(self):
    schema = events_db.Schema()
    fn = schema.register_field_name("name")
    record = events_db.Record()
    record[fn] = "kernel"
    record.clear()
    self.assertEqual((len(record), record.has_field(fn)), (0, False))

  @parameterized.named_parameters(
      ("string_tuple", ("tensor_x", "tensor_y"), ["tensor_x", "tensor_y"]),
      ("int_list", [10, 20, 30], [10, 20, 30]),
      ("range_sequence", range(4), [0, 1, 2, 3]),
  )
  def test_record_repeated_field_roundtrip(self, input_val, expected_list):
    schema = events_db.Schema()
    fn = schema.register_field_name("repeated_field")
    record = events_db.Record()
    record[fn] = input_val
    self.assertEqual(
        (isinstance(record[fn], Sequence), list(record[fn])),
        (True, expected_list),
    )

  @parameterized.named_parameters(
      ("equal_to_list", [10, 20, 30], True),
      ("equal_to_tuple", (10, 20, 30), True),
      ("equal_to_float_list", [10.0, 20.0, 30.0], True),
      ("unequal_float_list", [10.5, 20.0, 30.0], False),
      ("unequal_str_list", ["10", "20", "30"], False),
      ("unequal_other_type", "not_a_seq", False),
  )
  def test_sequence_equality(self, compare_target, should_equal):
    schema = events_db.Schema()
    fn = schema.register_field_name("counts")
    record = events_db.Record()
    record[fn] = [10, 20, 30]
    if should_equal:
      self.assertEqual(record[fn], compare_target)
    else:
      self.assertNotEqual(record[fn], compare_target)

  def test_cross_sequence_view_equality(self):
    schema = events_db.Schema()
    fn_counts = schema.register_field_name("counts")
    fn_ratios = schema.register_field_name("ratios")
    record = events_db.Record()
    record[fn_counts] = [10, 20, 30]
    record[fn_ratios] = [10.0, 20.0, 30.0]
    self.assertEqual(record[fn_counts], record[fn_ratios])

  def test_cross_sequence_view_inequality(self):
    schema = events_db.Schema()
    fn_counts = schema.register_field_name("counts")
    fn_tensors = schema.register_field_name("tensors")
    record = events_db.Record()
    record[fn_counts] = [10, 20, 30]
    record[fn_tensors] = ["tensor_x", "tensor_y", "tensor_z"]
    self.assertNotEqual(record[fn_counts], record[fn_tensors])

  def test_record_equality_same_fields(self):
    schema = events_db.Schema()
    fn = schema.register_field_name("name")
    r1, r2 = events_db.Record(), events_db.Record()
    r1[fn] = "op_a"
    r2[fn] = "op_a"
    self.assertEqual(r1, r2)

  def test_record_equality_different_values(self):
    schema = events_db.Schema()
    fn = schema.register_field_name("name")
    r1, r2 = events_db.Record(), events_db.Record()
    r1[fn] = "op_a"
    r2[fn] = "op_b"
    self.assertNotEqual(r1, r2)

  def test_sequence_views_are_abc_sequences(self):
    schema = events_db.Schema()
    fn = schema.register_field_name("field")
    record = events_db.Record()
    record.set_int32_sequence(fn, [1, 2, 3])
    seq = record[fn]
    self.assertIsInstance(seq, Sequence)
    self.assertEqual(list(seq), [1, 2, 3])

  def test_events_db_exports(self):
    self.assertCountEqual(
        events_db.__all__,
        [
            "FieldIndex",
            "FieldValue",
            "Record",
            "Schema",
            "StepControl",
            "ParseStatus",
            "RecordConsumerRef",
        ],
    )

  def test_bytes_rejected_with_type_error(self):
    schema = events_db.Schema()
    fn = schema.register_field_name("bytes_field")
    record = events_db.Record()
    with self.assertRaisesRegex(
        TypeError, "bytes is not a supported FieldValue; use str"
    ):
      record[fn] = b"abc"

  def test_unsupported_type_rejected_with_type_error(self):
    schema = events_db.Schema()
    fn = schema.register_field_name("unsupported_field")
    record = events_db.Record()
    with self.assertRaisesRegex(
        TypeError, "Unsupported value type for FieldValue"
    ):
      record[fn] = object()

  @parameterized.named_parameters(
      ("schema", events_db.Schema, "mapping between string names and indices"),
      ("record", events_db.Record, "row-oriented record"),
      (
          "field_index",
          events_db.FieldIndex,
          "Opaque token representing a field",
      ),
      (
          "record_consumer_ref",
          events_db.RecordConsumerRef,
          "Wrapper and adapter for a record consumer",
      ),
  )
  def test_class_docstrings(self, cls, expected_phrase):
    self.assertIn(expected_phrase, cls.__doc__)

  def test_module_docstring_contains_record_consumer_lifecycle(self):
    self.assertIn("Record Consumer Semantics & Lifecycle", events_db.__doc__)

  @parameterized.named_parameters(
      ("bool", "set_bool", True, True),
      ("int32", "set_int32", -42, -42),
      ("uint32", "set_uint32", 100, 100),
      ("int64", "set_int64", -100000, -100000),
      ("uint64", "set_uint64", 18446744073709551615, 18446744073709551615),
      ("double", "set_double", 3.14159, 3.14159),
      ("string", "set_string", "test_str", "test_str"),
      ("int32_seq", "set_int32_sequence", [-1, 0, 1], [-1, 0, 1]),
      ("int32_seq_range", "set_int32_sequence", range(3), [0, 1, 2]),
      ("uint32_seq", "set_uint32_sequence", [10, 20], [10, 20]),
      ("int64_seq", "set_int64_sequence", [-100, 200], [-100, 200]),
      (
          "uint64_seq",
          "set_uint64_sequence",
          [1000, 18446744073709551615],
          [1000, 18446744073709551615],
      ),
      ("double_seq", "set_double_sequence", [0.1, 0.2], [0.1, 0.2]),
      ("string_seq", "set_string_sequence", ["a", "b"], ["a", "b"]),
  )
  def test_typed_setter_methods(self, method_name, input_val, expected_val):
    schema = events_db.Schema()
    fn = schema.register_field_name("field")
    record = events_db.Record()
    getattr(record, method_name)(fn, input_val)
    if isinstance(expected_val, float):
      self.assertAlmostEqual(record[fn], expected_val, places=5)
    elif isinstance(expected_val, list):
      self.assertEqual(list(record[fn]), expected_val)
    else:
      self.assertEqual(record[fn], expected_val)

  def test_typed_setter_overwrite_different_types(self):
    schema = events_db.Schema()
    fn = schema.register_field_name("field")
    record = events_db.Record()

    record.set_int32(fn, 123)
    self.assertEqual(record[fn], 123)

    record.set_string(fn, "hello")
    self.assertEqual(record[fn], "hello")

    record.set_int32_sequence(fn, [1, 2, 3])
    self.assertEqual(list(record[fn]), [1, 2, 3])

  @parameterized.named_parameters(
      ("set_bool", "set_bool", True),
      ("set_int32", "set_int32", 1),
      ("set_uint32", "set_uint32", 1),
      ("set_int64", "set_int64", 1),
      ("set_uint64", "set_uint64", 1),
      ("set_double", "set_double", 1.0),
      ("set_string", "set_string", "a"),
      ("set_int32_sequence", "set_int32_sequence", [1]),
      ("set_uint32_sequence", "set_uint32_sequence", [1]),
      ("set_int64_sequence", "set_int64_sequence", [1]),
      ("set_uint64_sequence", "set_uint64_sequence", [1]),
      ("set_double_sequence", "set_double_sequence", [1.0]),
      ("set_string_sequence", "set_string_sequence", ["a"]),
  )
  def test_typed_setter_invalid_field_raises_value_error(
      self, method_name, val
  ):
    record = events_db.Record()
    invalid_field = events_db.FieldIndex()
    with self.assertRaisesRegex(
        ValueError, "Cannot set field on invalid FieldIndex"
    ):
      getattr(record, method_name)(invalid_field, val)

  @parameterized.named_parameters(
      ("set_string", "set_string"),
      ("set_int32_sequence", "set_int32_sequence"),
      ("set_string_sequence", "set_string_sequence"),
  )
  def test_typed_setter_bytes_rejected_with_type_error(self, method_name):
    schema = events_db.Schema()
    fn = schema.register_field_name("field")
    record = events_db.Record()
    with self.assertRaisesRegex(
        TypeError, "bytes is not a supported FieldValue; use str"
    ):
      getattr(record, method_name)(fn, b"abc")

  @parameterized.named_parameters(
      ("set_int32_sequence", "set_int32_sequence", {1, 2}),
      ("set_uint32_sequence", "set_uint32_sequence", {1, 2}),
      ("set_int64_sequence", "set_int64_sequence", {1, 2}),
      ("set_uint64_sequence", "set_uint64_sequence", {1, 2}),
      ("set_double_sequence", "set_double_sequence", {1.0, 2.0}),
      ("set_string_sequence", "set_string_sequence", {"a", "b"}),
  )
  def test_typed_sequence_setter_rejects_set_with_type_error(
      self, method_name, val
  ):
    schema = events_db.Schema()
    fn = schema.register_field_name("field")
    record = events_db.Record()
    with self.assertRaisesRegex(TypeError, "Expected a sequence"):
      getattr(record, method_name)(fn, val)

  def test_typed_sequence_setter_rejects_string_with_type_error(self):
    schema = events_db.Schema()
    fn = schema.register_field_name("field")
    record = events_db.Record()
    with self.assertRaisesRegex(
        TypeError, "str is not a supported sequence; use list or tuple"
    ):
      record.set_string_sequence(fn, "hello")

  @parameterized.named_parameters(
      ("empty_int", "set_int32_sequence", [], "[]"),
      ("int_sequence", "set_int32_sequence", [1, 2], "[1, 2]"),
      ("string_sequence", "set_string_sequence", ["a", "b"], '["a", "b"]'),
  )
  def test_sequence_view_repr(self, setter, val, expected_repr):
    schema = events_db.Schema()
    fn = schema.register_field_name("field")
    record = events_db.Record()
    getattr(record, setter)(fn, val)
    self.assertEqual(repr(record[fn]), expected_repr)

  def test_sequence_view_direct_equality(self):
    schema = events_db.Schema()
    fn1 = schema.register_field_name("fn1")
    fn2 = schema.register_field_name("fn2")
    r1, r2 = events_db.Record(), events_db.Record()
    r1.set_int32_sequence(fn1, [1, 2, 3])
    r2.set_int32_sequence(fn2, [1, 2, 3])
    self.assertEqual(r1[fn1], r2[fn2])

    r2.set_int32_sequence(fn2, [1, 2, 4])
    self.assertNotEqual(r1[fn1], r2[fn2])

  def test_sequence_view_equality_exception_returns_false(self):
    class ErrorOnEq:

      def __eq__(self, other: object) -> bool:
        raise RuntimeError("comparison error")

    schema = events_db.Schema()
    fn = schema.register_field_name("field")
    record = events_db.Record()
    record.set_int32_sequence(fn, [1])
    self.assertNotEqual(record[fn], [ErrorOnEq()])

  def test_sequence_view_equality_keyboard_interrupt_propagates(self):
    class InterruptOnEq:

      def __eq__(self, other: object) -> bool:
        raise KeyboardInterrupt()

    schema = events_db.Schema()
    fn = schema.register_field_name("field")
    record = events_db.Record()
    record.set_int32_sequence(fn, [1])
    with self.assertRaises(KeyboardInterrupt):
      _ = record[fn] == [InterruptOnEq()]

  def test_sequence_view_slicing(self):
    schema = events_db.Schema()
    fn = schema.register_field_name("seq")
    record = events_db.Record()
    record.set_int32_sequence(fn, [0, 1, 2, 3, 4, 5])
    seq = record[fn]

    with self.subTest(name="basic_slice"):
      sliced = seq[1:4]
      self.assertIs(type(sliced), type(seq))
      self.assertEqual(sliced, [1, 2, 3])

    with self.subTest(name="open_bounds_and_step"):
      self.assertEqual(seq[:3], [0, 1, 2])
      self.assertEqual(seq[3:], [3, 4, 5])
      self.assertEqual(seq[:], [0, 1, 2, 3, 4, 5])
      self.assertEqual(seq[::2], [0, 2, 4])

    with self.subTest(name="negative_indices_and_reversed_slice"):
      self.assertEqual(seq[-3:-1], [3, 4])
      self.assertEqual(seq[::-1], [5, 4, 3, 2, 1, 0])

    with self.subTest(name="empty_slice"):
      self.assertEqual(seq[4:2], [])
      self.assertEmpty(seq[4:2])

    with self.subTest(name="reassign_sliced_sequence"):
      fn2 = schema.register_field_name("seq2")
      record2 = events_db.Record()
      record2[fn2] = sliced
      self.assertEqual(record2[fn2], [1, 2, 3])

  def test_large_uint64_via_generic_setter(self):
    schema = events_db.Schema()
    fn = schema.register_field_name("field")
    record = events_db.Record()
    record[fn] = 18446744073709551615
    self.assertEqual(record[fn], 18446744073709551615)

  @parameterized.named_parameters(
      ("int32", "set_int32_sequence", [1, 2]),
      ("uint32", "set_uint32_sequence", [10, 20]),
      ("int64", "set_int64_sequence", [-100, 200]),
      ("uint64", "set_uint64_sequence", [1000, 2000]),
      ("double", "set_double_sequence", [1.5, 2.5]),
      ("string", "set_string_sequence", ["a", "b"]),
  )
  def test_assign_existing_sequence_view_to_record(self, setter, val):
    schema = events_db.Schema()
    fn1 = schema.register_field_name("fn1")
    fn2 = schema.register_field_name("fn2")
    r1, r2 = events_db.Record(), events_db.Record()
    getattr(r1, setter)(fn1, val)
    r2[fn2] = r1[fn1]
    if isinstance(val[0], float):
      for actual, expected in zip(r2[fn2], val):
        self.assertAlmostEqual(actual, expected)
    else:
      self.assertEqual(list(r2[fn2]), val)

  def test_record_set_method(self):
    schema = events_db.Schema()
    fn = schema.register_field_name("field")
    record = events_db.Record()
    record.set(fn, 123)
    self.assertEqual(record[fn], 123)

  def test_record_set_invalid_field_raises_value_error(self):
    record = events_db.Record()
    invalid_field = events_db.FieldIndex()
    with self.assertRaisesRegex(
        ValueError, "Cannot set field on invalid FieldIndex"
    ):
      record.set(invalid_field, 123)

  def test_set_string_non_string_raises_type_error(self):
    schema = events_db.Schema()
    fn = schema.register_field_name("field")
    record = events_db.Record()
    with self.assertRaisesRegex(TypeError, "Expected str"):
      record.set_string(fn, 123)


class StepControlTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("continue", events_db.StepControl.CONTINUE, "CONTINUE", 0),
      ("stop", events_db.StepControl.STOP, "STOP", 1),
  )
  def test_member_properties_and_lookup(
      self,
      member: events_db.StepControl,
      expected_name: str,
      expected_value: int,
  ):
    self.assertEqual(member.name, expected_name)
    self.assertEqual(member.value, expected_value)
    self.assertIs(events_db.StepControl[expected_name], member)
    self.assertIs(events_db.StepControl(expected_value), member)

  def test_members_are_distinct(self):
    self.assertNotEqual(
        events_db.StepControl.CONTINUE, events_db.StepControl.STOP
    )

  def test_usable_in_set(self):
    control_set = {events_db.StepControl.CONTINUE, events_db.StepControl.STOP}
    self.assertIn(events_db.StepControl.CONTINUE, control_set)
    self.assertIn(events_db.StepControl.STOP, control_set)


class ParseStatusTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("complete", events_db.ParseStatus.COMPLETE, "COMPLETE", 0),
      (
          "stopped_early",
          events_db.ParseStatus.STOPPED_EARLY,
          "STOPPED_EARLY",
          1,
      ),
  )
  def test_member_properties_and_lookup(
      self,
      member: events_db.ParseStatus,
      expected_name: str,
      expected_value: int,
  ):
    self.assertEqual(member.name, expected_name)
    self.assertEqual(member.value, expected_value)
    self.assertIs(events_db.ParseStatus[expected_name], member)
    self.assertIs(events_db.ParseStatus(expected_value), member)

  def test_members_are_distinct(self):
    self.assertNotEqual(
        events_db.ParseStatus.COMPLETE, events_db.ParseStatus.STOPPED_EARLY
    )

  def test_usable_in_dict(self):
    status_dict = {
        events_db.ParseStatus.COMPLETE: "complete",
        events_db.ParseStatus.STOPPED_EARLY: "stopped_early",
    }
    self.assertEqual(status_dict[events_db.ParseStatus.COMPLETE], "complete")
    self.assertEqual(
        status_dict[events_db.ParseStatus.STOPPED_EARLY], "stopped_early"
    )


class RecordConsumerRefTest(parameterized.TestCase):

  def test_construct_from_callable_function(self):
    def consumer(record: events_db.Record) -> events_db.StepControl:
      del record  # Unused.
      return events_db.StepControl.CONTINUE

    ref = events_db.RecordConsumerRef(consumer)
    record = events_db.Record()
    self.assertEqual(ref.consume(record), events_db.StepControl.CONTINUE)
    self.assertEqual(ref(record), events_db.StepControl.CONTINUE)

  def test_construct_from_lambda(self):
    ref = events_db.RecordConsumerRef(lambda r: events_db.StepControl.STOP)
    record = events_db.Record()
    self.assertEqual(ref(record), events_db.StepControl.STOP)

  def test_construct_from_consumer_method(self):
    class MethodConsumer:

      def __init__(self):
        self.call_count = 0

      def consume(self, record: events_db.Record) -> events_db.StepControl:
        del record  # Unused.
        self.call_count += 1
        return events_db.StepControl.CONTINUE

    consumer = MethodConsumer()
    ref = events_db.RecordConsumerRef(consumer)
    record = events_db.Record()
    self.assertEqual(ref(record), events_db.StepControl.CONTINUE)
    self.assertEqual(consumer.call_count, 1)

  def test_prioritizes_consume_over_call(self):
    class DualConsumer:

      def consume(self, record: events_db.Record) -> events_db.StepControl:
        del self, record  # Unused.
        return events_db.StepControl.CONTINUE

      def __call__(self, record: events_db.Record) -> events_db.StepControl:
        del self, record  # Unused.
        return events_db.StepControl.STOP

    ref = events_db.RecordConsumerRef(DualConsumer())
    record = events_db.Record()
    self.assertEqual(ref(record), events_db.StepControl.CONTINUE)

  @parameterized.named_parameters(
      ("return_none", lambda r: None, events_db.StepControl.CONTINUE),
      ("return_true", lambda r: True, events_db.StepControl.CONTINUE),
      ("return_false", lambda r: False, events_db.StepControl.STOP),
      (
          "return_continue",
          lambda r: events_db.StepControl.CONTINUE,
          events_db.StepControl.CONTINUE,
      ),
      (
          "return_stop",
          lambda r: events_db.StepControl.STOP,
          events_db.StepControl.STOP,
      ),
  )
  def test_consumer_return_values(self, callback, expected_control):
    ref = events_db.RecordConsumerRef(callback)
    record = events_db.Record()
    self.assertEqual(ref(record), expected_control)

  @parameterized.named_parameters(
      ("int", 42),
      ("string", "unexpected"),
      ("list", [1, 2, 3]),
  )
  def test_invalid_return_type_raises_type_error(self, return_value):
    ref = events_db.RecordConsumerRef(lambda r: return_value)
    record = events_db.Record()
    with self.assertRaisesRegex(
        TypeError, "Record consumer must return StepControl, bool, or None"
    ):
      ref(record)

  @parameterized.named_parameters(
      ("int", 123),
      ("none", None),
      ("string", "not_callable"),
      ("object_without_consume", object()),
  )
  def test_invalid_target_raises_type_error(self, target):
    with self.assertRaisesRegex(
        TypeError,
        "RecordConsumerRef target must be callable or provide a callable"
        " 'consume' method",
    ):
      events_db.RecordConsumerRef(target)

  def test_target_property(self):
    def my_fn(record: events_db.Record) -> None:
      del record  # Unused.

    ref = events_db.RecordConsumerRef(my_fn)
    self.assertIs(ref.target, my_fn)

  def test_repr(self):
    def custom_target(record: events_db.Record) -> None:
      del record  # Unused.

    ref = events_db.RecordConsumerRef(custom_target)
    self.assertIn("RecordConsumerRef(target=", repr(ref))
    self.assertIn("custom_target", repr(ref))

  def test_copy_constructor(self):
    ref1 = events_db.RecordConsumerRef(lambda r: events_db.StepControl.STOP)
    ref2 = events_db.RecordConsumerRef(ref1)
    self.assertIs(ref2.target, ref1.target)
    ref3 = copy.copy(ref1)
    self.assertIs(ref3.target, ref1.target)
    record = events_db.Record()
    self.assertEqual(ref2(record), events_db.StepControl.STOP)
    self.assertEqual(ref3(record), events_db.StepControl.STOP)

  def test_finalize_signature_inspection_failure_falls_back_to_taking_arg(self):
    class UninspectableFinalize:

      def __init__(self) -> None:
        self.called_with: events_db.ParseStatus | None = None

      @property
      def __signature__(self) -> None:
        raise ValueError("Cannot inspect signature")

      def __call__(self, arg: events_db.ParseStatus) -> None:
        self.called_with = arg

    class Target:

      def __init__(self, finalize_fn: UninspectableFinalize) -> None:
        self.finalize = finalize_fn

      def consume(self, record: events_db.Record) -> None:
        del self, record  # Unused.

    fin = UninspectableFinalize()
    ref = events_db.RecordConsumerRef(Target(fin))
    ref.finalize(events_db.ParseStatus.COMPLETE)
    self.assertEqual(fin.called_with, events_db.ParseStatus.COMPLETE)

  def test_unhandled_finalize_attribute_raises_type_error(self):
    class BadFinalize:

      def consume(self, record: events_db.Record) -> None:
        del self, record  # Unused.

      finalize = 123

    with self.assertRaisesRegex(
        TypeError, "'finalize' attribute must be callable"
    ):
      events_db.RecordConsumerRef(BadFinalize())

  def test_noop_finalize_when_target_has_no_finalize(self):
    ref = events_db.RecordConsumerRef(lambda r: events_db.StepControl.CONTINUE)
    ref.finalize()
    ref.finalize(events_db.ParseStatus.STOPPED_EARLY)

  @parameterized.named_parameters(
      ("default", None, events_db.ParseStatus.COMPLETE),
      (
          "complete",
          events_db.ParseStatus.COMPLETE,
          events_db.ParseStatus.COMPLETE,
      ),
      (
          "stopped_early",
          events_db.ParseStatus.STOPPED_EARLY,
          events_db.ParseStatus.STOPPED_EARLY,
      ),
  )
  def test_finalize_with_status_argument(self, pass_status, expected_received):
    class FinalizeConsumer:

      def __init__(self):
        self.received = None

      def consume(self, record: events_db.Record) -> events_db.StepControl:
        del self, record  # Unused.
        return events_db.StepControl.CONTINUE

      def finalize(self, status: events_db.ParseStatus) -> None:
        self.received = status

    consumer = FinalizeConsumer()
    ref = events_db.RecordConsumerRef(consumer)
    if pass_status is None:
      ref.finalize()
    else:
      ref.finalize(pass_status)
    self.assertEqual(consumer.received, expected_received)

  @parameterized.named_parameters(
      ("default", None),
      ("complete", events_db.ParseStatus.COMPLETE),
      ("stopped_early", events_db.ParseStatus.STOPPED_EARLY),
  )
  def test_parameterless_finalize(self, pass_status):
    class ParameterlessFinalizeConsumer:

      def __init__(self):
        self.finalize_calls = 0

      def consume(self, record: events_db.Record) -> events_db.StepControl:
        del self, record  # Unused.
        return events_db.StepControl.CONTINUE

      def finalize(self) -> None:
        self.finalize_calls += 1

    consumer = ParameterlessFinalizeConsumer()
    ref = events_db.RecordConsumerRef(consumer)
    if pass_status is None:
      ref.finalize()
    else:
      ref.finalize(pass_status)
    self.assertEqual(consumer.finalize_calls, 1)

  def test_consume_exception_propagates(self):
    def faulty_consumer(record: events_db.Record) -> None:
      del record  # Unused.
      raise ValueError("custom consume failure")

    ref = events_db.RecordConsumerRef(faulty_consumer)
    with self.assertRaisesRegex(ValueError, "custom consume failure"):
      ref(events_db.Record())

  def test_finalize_exception_propagates(self):
    class FaultyFinalizeConsumer:

      def consume(self, record: events_db.Record) -> None:
        del self, record  # Unused.

      def finalize(self) -> None:
        raise RuntimeError("custom finalize failure")

    ref = events_db.RecordConsumerRef(FaultyFinalizeConsumer())
    with self.assertRaisesRegex(RuntimeError, "custom finalize failure"):
      ref.finalize()

  def test_finalize_with_exception_argument(self):
    class FinalizeWithOutcomeConsumer:

      def __init__(self):
        self.received = None

      def consume(self, record: events_db.Record) -> events_db.StepControl:
        del self, record  # Unused.
        return events_db.StepControl.CONTINUE

      def finalize(self, outcome: Exception | events_db.ParseStatus) -> None:
        self.received = outcome

    consumer = FinalizeWithOutcomeConsumer()
    ref = events_db.RecordConsumerRef(consumer)
    err = RuntimeError("worker thread failed")
    ref.finalize(err)
    self.assertIs(consumer.received, err)

  def test_parameterless_finalize_skipped_on_exception(self):
    class ParameterlessFinalizeConsumer:

      def __init__(self):
        self.finalize_calls = 0

      def consume(self, record: events_db.Record) -> events_db.StepControl:
        del self, record  # Unused.
        return events_db.StepControl.CONTINUE

      def finalize(self) -> None:
        self.finalize_calls += 1

    consumer = ParameterlessFinalizeConsumer()
    ref = events_db.RecordConsumerRef(consumer)
    ref.finalize(RuntimeError("worker failed"))
    self.assertEqual(consumer.finalize_calls, 0)

  @parameterized.named_parameters(
      ("int", 123),
      ("str", "invalid"),
      ("dict", {}),
  )
  def test_invalid_finalize_argument_raises_type_error(self, arg):
    ref = events_db.RecordConsumerRef(lambda r: events_db.StepControl.CONTINUE)
    with self.assertRaisesRegex(
        TypeError,
        "finalize argument must be a ParseStatus, an Exception, or None",
    ):
      ref.finalize(arg)

  def test_record_mutation_in_consumer(self):
    schema = events_db.Schema()
    tag_field = schema.register_field_name("tag")

    def mutating_consumer(record: events_db.Record) -> events_db.StepControl:
      record[tag_field] = "mutated"
      return events_db.StepControl.CONTINUE

    ref = events_db.RecordConsumerRef(mutating_consumer)
    record = events_db.Record()
    self.assertEqual(ref(record), events_db.StepControl.CONTINUE)
    self.assertEqual(record[tag_field], "mutated")


if __name__ == "__main__":
  absltest.main()
