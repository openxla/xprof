"""Unit tests for XLoopItem loading, saving, and wire compatibility."""

import os

from absl.testing import absltest

from xprof.protobuf import xloop_item_pb2
from xprof.xloop import input_utils


def _item(name: str = "matmul") -> xloop_item_pb2.XLoopItem:
  item = xloop_item_pb2.XLoopItem(
      id="item-1",
      name=name,
      kernel_type=xloop_item_pb2.KERNEL_TYPE_JAX,
      reference_code="def get_model_identifier(): return lambda x: x",
      custom_code="def get_model_identifier(): return lambda x: x + 0",
  )
  config = item.input_generation_configs["default"]
  config.name = "default"
  config.input_generation_code = "def get_input_generator(): ..."
  config.atol = 1e-4
  return item


class RoleHelpersTest(absltest.TestCase):

  def test_code_for_role_returns_source(self):
    item = _item()
    self.assertEqual(
        input_utils.code_for_role(item, "reference"), item.reference_code
    )
    self.assertEqual(
        input_utils.code_for_role(item, "custom"), item.custom_code
    )

  def test_code_for_role_returns_empty_for_absent_role(self):
    self.assertEmpty(input_utils.code_for_role(_item(), "optimized"))

  def test_code_for_role_rejects_unknown_role(self):
    with self.assertRaises(KeyError):
      input_utils.code_for_role(_item(), "baseline")

  def test_present_roles_reports_only_populated_roles(self):
    self.assertEqual(
        input_utils.present_roles(_item()), ("reference", "custom")
    )

  def test_present_roles_preserves_canonical_order(self):
    item = xloop_item_pb2.XLoopItem(optimized_code="o", reference_code="r")
    self.assertEqual(
        input_utils.present_roles(item), ("reference", "optimized")
    )

  def test_auxiliary_modules_for_role(self):
    item = _item()
    item.custom_auxiliary_modules.modules["pkg.helper"] = "X = 1"
    self.assertEqual(
        input_utils.auxiliary_modules_for_role(item, "custom"),
        {"pkg.helper": "X = 1"},
    )
    self.assertEmpty(input_utils.auxiliary_modules_for_role(item, "reference"))


class ToleranceTest(absltest.TestCase):

  def test_set_tolerance_is_returned(self):
    config = xloop_item_pb2.InputGenerationConfig(atol=1e-4)
    atol, rtol = input_utils.tolerances(config)
    self.assertIsNotNone(atol)
    assert atol is not None  # Narrowing for the type checker.
    self.assertAlmostEqual(atol, 1e-4)
    self.assertIsNone(rtol)

  def test_explicit_zero_is_distinguished_from_unset(self):
    # Presence matters: an explicit 0.0 means "require exactness", whereas
    # unset means "defer to the input module".
    config = xloop_item_pb2.InputGenerationConfig(atol=0.0)
    atol, _ = input_utils.tolerances(config)
    self.assertEqual(atol, 0.0)
    self.assertIsNotNone(atol)

  def test_unset_tolerances_are_none(self):
    atol, rtol = input_utils.tolerances(xloop_item_pb2.InputGenerationConfig())
    self.assertIsNone(atol)
    self.assertIsNone(rtol)


class RoundTripTest(absltest.TestCase):

  def _round_trip(self, filename: str) -> list[xloop_item_pb2.XLoopItem]:
    path = os.path.join(self.create_tempdir().full_path, filename)
    original = [_item("a"), _item("b")]
    input_utils.save_output_items(original, path)
    loaded = input_utils.load_input_items(path)
    self.assertEqual(loaded, original)
    return loaded

  def test_json_round_trip(self):
    self._round_trip("items.json")

  def test_jsonl_round_trip(self):
    self._round_trip("items.jsonl")

  def test_textproto_round_trip(self):
    self._round_trip("items.txtpb")

  def test_binary_round_trip(self):
    self._round_trip("items.pb")

  def test_json_uses_snake_case_field_names(self):
    # Snake_case keeps hand-written JSON identical to the .proto source.
    text = input_utils.to_json(_item())
    self.assertIn('"reference_code"', text)
    self.assertNotIn('"referenceCode"', text)

  def test_json_accepts_lower_camel_case(self):
    path = os.path.join(self.create_tempdir().full_path, "item.json")
    with open(path, "wt") as f:
      f.write('{"name": "m", "referenceCode": "code"}')
    (loaded,) = input_utils.load_input_items(path)
    self.assertEqual(loaded.reference_code, "code")

  def test_bare_item_loads_without_batch_wrapper(self):
    path = os.path.join(self.create_tempdir().full_path, "item.json")
    with open(path, "wt") as f:
      f.write(input_utils.to_json(_item()))
    self.assertLen(input_utils.load_input_items(path), 1)

  def test_binary_accepts_bare_item(self):
    path = os.path.join(self.create_tempdir().full_path, "item.pb")
    with open(path, "wb") as f:
      f.write(_item().SerializeToString())
    (loaded,) = input_utils.load_input_items(path)
    self.assertEqual(loaded, _item())


class WireCompatibilityTest(absltest.TestCase):
  """XLoopItem wire layout and reserved field numbers are pinned."""

  def test_reserved_field_numbers_are_not_reused(self):
    fields_by_number = {
        f.number for f in xloop_item_pb2.XLoopItem.DESCRIPTOR.fields
    }
    reserved = (
        set(range(7, 17))
        | set(range(19, 23))
        | {24, 25}
        | set(range(28, 32))
        | {34, 35}
        | set(range(37, 40))
    )
    self.assertEmpty(fields_by_number & reserved)

  def test_field_numbers_are_pinned(self):
    # Pinning these makes an accidental renumbering a test failure rather than
    # a silent cross-version data corruption.
    expected = {
        "id": 1,
        "name": 2,
        "kernel_type": 3,
        "reference_code": 4,
        "custom_code": 5,
        "split": 6,
        "optimized_code": 23,
        "task_specific_metadata": 26,
        "input_generation_configs": 27,
        "execution_report": 32,
        "execution_results": 33,
        "task_type": 36,
        "base_changelist_number": 41,
        "reference_auxiliary_modules": 42,
        "custom_auxiliary_modules": 43,
        "optimized_auxiliary_modules": 44,
        "execution_environments": 47,
    }
    actual = {
        f.name: f.number
        for f in xloop_item_pb2.XLoopItem.DESCRIPTOR.fields
        if f.name in expected
    }
    self.assertEqual(actual, expected)

  def test_unknown_fields_survive_parsing(self):
    item = _item()
    raw = bytearray(item.SerializeToString())
    # Append field 20 (a reserved field), varint wire type.
    raw.extend(b"\xa0\x01\x04")
    parsed = xloop_item_pb2.XLoopItem()
    parsed.ParseFromString(bytes(raw))
    self.assertEqual(parsed.name, item.name)
    self.assertEqual(parsed.reference_code, item.reference_code)


class ErrorTest(absltest.TestCase):

  def test_missing_file_raises(self):
    with self.assertRaisesRegex(input_utils.InputError, "no such input file"):
      input_utils.load_input_items("/nonexistent/items.json")

  def test_malformed_json_raises(self):
    path = os.path.join(self.create_tempdir().full_path, "items.json")
    with open(path, "wt") as f:
      f.write("{not json")
    with self.assertRaises(input_utils.InputError):
      input_utils.load_input_items(path)

  def test_unknown_json_field_raises(self):
    path = os.path.join(self.create_tempdir().full_path, "items.json")
    with open(path, "wt") as f:
      f.write('{"name": "m", "no_such_field": 1}')
    with self.assertRaises(input_utils.InputError):
      input_utils.load_input_items(path)

  def test_jsonl_error_reports_line_number(self):
    path = os.path.join(self.create_tempdir().full_path, "items.jsonl")
    with open(path, "wt") as f:
      f.write(input_utils.to_json(_item(), indent=None) + "\n")
      f.write("{oops\n")
    with self.assertRaisesRegex(input_utils.InputError, r":2:"):
      input_utils.load_input_items(path)

  def test_jsonl_skips_blank_lines(self):
    path = os.path.join(self.create_tempdir().full_path, "items.jsonl")
    with open(path, "wt") as f:
      f.write("\n")
      f.write(input_utils.to_json(_item(), indent=None) + "\n")
      f.write("\n")
    self.assertLen(input_utils.load_input_items(path), 1)

  def test_unsupported_output_extension_raises(self):
    path = os.path.join(self.create_tempdir().full_path, "items.yaml")
    with self.assertRaisesRegex(input_utils.InputError, "unsupported output"):
      input_utils.save_output_items([_item()], path)

  def test_save_creates_parent_directories(self):
    path = os.path.join(
        self.create_tempdir().full_path, "nested", "deeper", "items.json"
    )
    input_utils.save_output_items([_item()], path)
    self.assertTrue(os.path.exists(path))


if __name__ == "__main__":
  absltest.main()
