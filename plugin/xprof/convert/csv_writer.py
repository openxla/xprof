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

"""Converts data between different formats."""

import csv
import io
import json
from typing import List, Optional, Any


def json_to_csv(
    json_string: str,
    columns_order: Optional[List[str]] = None,
    separator: str = ",",
) -> str:
  """Converts a JSON string to a CSV string.

  Args:
    json_string: The JSON string to convert.
    columns_order: Optional. Specifies the order of columns in the output table.
      Specify a list of all column IDs in the order in which you want the table
      created. Note that you must list all column IDs in this parameter, if you
      use it.
    separator: Optional. The separator to use between the values.

  Returns:
    A CSV string representing the data.
    Example result:
     'a','b','c'
     1,'z',2
     3,'w',''

  Raises:
    ValueError: The json_string is not a valid json or is not a single object.
  """
  try:
    data = json.loads(json_string)
  except json.JSONDecodeError as e:
    raise ValueError(f"Invalid JSON string: {e}") from e

  if not isinstance(data, dict):
    raise ValueError("JSON data must be a single object")

  headers = list(data.keys())
  if columns_order is None:
    columns_order = headers

  if not all(col in headers for col in columns_order):
    raise ValueError("columns_order must be a list of all column IDs")

  if not columns_order:
    return ""

  csv_buffer = io.StringIO(newline="")
  writer = csv.writer(csv_buffer, delimiter=separator, lineterminator="\n")

  # Write header
  writer.writerow([col for col in columns_order])

  # Write row
  cells_list = []
  for col in columns_order:
    value = ""
    if col in data:
      value = data.get(col)
      if value is not None:
        if isinstance(value, bool):
          value = "true" if value else "false"
        else:
          value = str(value)
    cells_list.append(value)
  writer.writerow(cells_list)

  return csv_buffer.getvalue()


def json_to_csv_string(json_data: Any) -> str:
  """Converts profile JSON to CSV using strict quoting to handle stack traces."""
  # 1. Handle Byte Strings
  if isinstance(json_data, (bytes, bytearray)):
    json_data = json_data.decode("utf-8")

  # 2. Parse JSON
  try:
    data = json.loads(json_data) if isinstance(json_data, str) else json_data
  except json.JSONDecodeError as e:
    raise ValueError(f"Invalid JSON string: {e}") from e

  # 3. Handle list structure (Take first element)
  if isinstance(data, list):
    if not data: return ""
    data = data[0]

  if not isinstance(data, dict) or "cols" not in data:
    raise ValueError("Data format not suitable for CSV (missing 'cols')")

  csv_buffer = io.StringIO()
  writer = csv.writer(
      csv_buffer,
      delimiter=",",
      quoting=csv.QUOTE_ALL,
      lineterminator="\n"
  )

  # 4. Write Headers
  headers = [col.get("label", col.get("id", "")) for col in data["cols"]]
  writer.writerow(headers)

  # 5. Write Rows
  for row in data.get("rows", []):
    cells = row.get("c", [])
    row_values = []
    for cell in cells:
      if isinstance(cell, dict):
        val = cell.get("v")
        if val is None:
          row_values.append("")
        else:
          row_values.append(str(val))
      else:
        row_values.append("")

    if row_values:
      writer.writerow(row_values)

  return csv_buffer.getvalue()
