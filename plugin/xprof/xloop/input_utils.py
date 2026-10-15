"""Loading and saving XLoop benchmark items.

Supported formats:

===========  ========================================================
`.json`      A single `XLoopItem`, or an `XLoopItems` batch, in proto3
             JSON. Field names may be snake_case or lowerCamelCase.
`.jsonl`     One `XLoopItem` per line.
`.txtpb`     An `XLoopItems` batch in text format.
`.pb`        An `XLoopItems` batch (or a single `XLoopItem`) in binary
             wire format.
===========  ========================================================
"""

from __future__ import annotations

import json
import os
from typing import Any

from google.protobuf import json_format
from google.protobuf import text_format

from xprof.protobuf import xloop_item_pb2

# Conventional implementation roles, in the order they are reported.
#
# `reference` is the correctness oracle and the denominator for speedups;
# `custom` is the implementation under test; `optimized` is a known-good
# implementation used to measure how much headroom is left.
ROLES: tuple[str, ...] = ("reference", "custom", "optimized")
REFERENCE_ROLE = "reference"

# Fallback comparison tolerance, used when neither the item nor the
# input-generation module specifies one.
DEFAULT_ATOL = 1e-3
DEFAULT_RTOL = 1e-3

_JSON_EXTS = frozenset({".json"})
_JSONL_EXTS = frozenset({".jsonl"})
_TEXTPROTO_EXTS = frozenset({".txtpb", ".textproto", ".pbtxt"})
_BINARY_EXTS = frozenset({".pb", ".binarypb", ".recordio"})


class InputError(ValueError):
  """Raised when an input file cannot be parsed into items."""


def code_for_role(item: xloop_item_pb2.XLoopItem, role: str) -> str:
  """Returns the source code for `role`.

  Args:
    item: Item to read.
    role: One of `ROLES`.

  Returns:
    The source text, empty if the item does not define that role.

  Raises:
    KeyError: If `role` is not a known role.
  """
  _check_role(role)
  return getattr(item, f"{role}_code")


def auxiliary_modules_for_role(
    item: xloop_item_pb2.XLoopItem, role: str
) -> dict[str, str]:
  """Returns the auxiliary module sources for `role`, keyed by module name.

  Args:
    item: Item to read.
    role: One of `ROLES`.

  Returns:
    Mapping of fully qualified module name to source text.

  Raises:
    KeyError: If `role` is not a known role.
  """
  _check_role(role)
  return dict(getattr(item, f"{role}_auxiliary_modules").modules)


def present_roles(item: xloop_item_pb2.XLoopItem) -> tuple[str, ...]:
  """Returns the roles for which `item` actually carries code."""
  return tuple(r for r in ROLES if code_for_role(item, r))


def tolerances(
    config: xloop_item_pb2.InputGenerationConfig,
) -> tuple[float | None, float | None]:
  """Returns `(atol, rtol)` for a config, `None` where unset.

  An unset tolerance means "defer to the input-generation module's
  `get_atol()` / `get_rtol()`". Only if those are also absent does the engine
  fall back to `DEFAULT_ATOL` / `DEFAULT_RTOL`.

  Args:
    config: Config to read.

  Returns:
    Tuple of optional absolute and relative tolerance.
  """
  atol = config.atol if config.HasField("atol") else None
  rtol = config.rtol if config.HasField("rtol") else None
  return atol, rtol


def _check_role(role: str) -> None:
  if role not in ROLES:
    raise KeyError(f"unknown role {role!r}, expected one of {ROLES}")


def _item_from_json_obj(
    raw: Any, source: str
) -> list[xloop_item_pb2.XLoopItem]:
  """Parses a decoded JSON value as either a batch or a single item."""
  if not isinstance(raw, dict):
    raise InputError(
        f"{source}: expected a JSON object, got {type(raw).__name__}"
    )
  try:
    # A batch is distinguished by its single repeated `items` field. Anything
    # else is a bare item.
    if set(raw) == {"items"}:
      batch = json_format.ParseDict(raw, xloop_item_pb2.XLoopItems())
      return list(batch.items)
    return [json_format.ParseDict(raw, xloop_item_pb2.XLoopItem())]
  except json_format.ParseError as e:
    raise InputError(f"{source}: {e}") from e


def _load_json(path: str) -> list[xloop_item_pb2.XLoopItem]:
  with open(path, "rt") as f:
    try:
      raw = json.load(f)
    except json.JSONDecodeError as e:
      raise InputError(f"{path}: not valid JSON: {e}") from e
  return _item_from_json_obj(raw, path)


def _load_jsonl(path: str) -> list[xloop_item_pb2.XLoopItem]:
  """Loads one or more XLoopItems from a JSON Lines file."""
  items: list[xloop_item_pb2.XLoopItem] = []
  with open(path, "rt") as f:
    for lineno, line in enumerate(f, start=1):
      line = line.strip()
      if not line:
        continue
      try:
        raw = json.loads(line)
      except json.JSONDecodeError as e:
        raise InputError(f"{path}:{lineno}: not valid JSON: {e}") from e
      items.extend(_item_from_json_obj(raw, f"{path}:{lineno}"))
  return items


def _load_textproto(path: str) -> list[xloop_item_pb2.XLoopItem]:
  """Loads an XLoopItem or XLoopItems batch from a textproto file."""
  with open(path, "rt") as f:
    text = f.read()
  batch = xloop_item_pb2.XLoopItems()
  try:
    text_format.Parse(text, batch)
    return list(batch.items)
  except text_format.ParseError:
    pass
  item = xloop_item_pb2.XLoopItem()
  try:
    text_format.Parse(text, item)
  except text_format.ParseError as e:
    raise InputError(f"{path}: not a valid XLoopItem(s) text proto: {e}") from e
  return [item]


def _load_binary(path: str) -> list[xloop_item_pb2.XLoopItem]:
  """Loads an XLoopItem or XLoopItems batch from a binary proto file."""
  with open(path, "rb") as f:
    data = f.read()
  batch = xloop_item_pb2.XLoopItems()
  try:
    batch.ParseFromString(data)
    if batch.items:
      return list(batch.items)
  except Exception:  # pylint: disable=broad-exception-caught
    pass
  # Not a batch, or an empty one; try parsing as a single XLoopItem.
  item = xloop_item_pb2.XLoopItem()
  try:
    item.ParseFromString(data)
  except Exception as e:  # pylint: disable=broad-exception-caught
    raise InputError(f"{path}: not a valid XLoopItem(s) wire proto: {e}") from e
  return [item]


def load_input_items(input_path: str) -> list[xloop_item_pb2.XLoopItem]:
  """Loads benchmark items from a file.

  Args:
    input_path: Path to a `.json`, `.jsonl`, `.txtpb` or `.pb` file. The format
      is chosen by extension; unrecognized extensions are tried as JSON and then
      as binary.

  Returns:
    The items in file order. May be empty if the file holds an empty batch.

  Raises:
    InputError: If the file is missing or cannot be parsed.
  """
  if not os.path.exists(input_path):
    raise InputError(f"no such input file: {input_path}")

  ext = os.path.splitext(input_path)[1].lower()
  if ext in _JSONL_EXTS:
    return _load_jsonl(input_path)
  if ext in _TEXTPROTO_EXTS:
    return _load_textproto(input_path)
  if ext in _BINARY_EXTS:
    return _load_binary(input_path)
  if ext in _JSON_EXTS:
    return _load_json(input_path)

  try:
    return _load_json(input_path)
  except InputError:
    return _load_binary(input_path)


def save_output_items(
    items: list[xloop_item_pb2.XLoopItem], output_path: str
) -> None:
  """Writes items to a file, choosing the format by extension.

  Args:
    items: Items to write.
    output_path: Destination. Parent directories are created.

  Raises:
    InputError: If the extension is not a supported output format.
  """
  parent = os.path.dirname(output_path)
  if parent:
    os.makedirs(parent, exist_ok=True)

  ext = os.path.splitext(output_path)[1].lower()
  batch = xloop_item_pb2.XLoopItems(items=items)

  if ext in _JSONL_EXTS:
    with open(output_path, "wt") as f:
      for item in items:
        f.write(_to_json(item, indent=None) + "\n")
    return
  if ext in _TEXTPROTO_EXTS:
    with open(output_path, "wt") as f:
      f.write(text_format.MessageToString(batch))
    return
  if ext in _BINARY_EXTS:
    with open(output_path, "wb") as f:
      f.write(batch.SerializeToString())
    return
  if ext in _JSON_EXTS:
    with open(output_path, "wt") as f:
      f.write(_to_json(batch, indent=2))
    return

  raise InputError(
      f"unsupported output extension {ext!r} for {output_path}; expected one"
      " of .json, .jsonl, .txtpb, .pb"
  )


def _to_json(message: Any, *, indent: int | None) -> str:
  """Serializes a proto to JSON with snake_case field names.

  Snake_case matches the `.proto` field names, so hand-written and generated
  JSON look the same.

  Args:
    message: Proto message.
    indent: Indentation, or None for a single line.

  Returns:
    The JSON document.
  """
  return json_format.MessageToJson(
      message,
      indent=indent,
      preserving_proto_field_name=True,
      always_print_fields_with_no_presence=False,
  )


def to_json(item: xloop_item_pb2.XLoopItem, *, indent: int | None = 2) -> str:
  """Serializes one item to proto3 JSON with snake_case field names."""
  return _to_json(item, indent=indent)
