"""Tool to fetch detailed HLO performance statistics from XProf."""

import dataclasses
import json
import logging
import re
from typing import Any

from google.protobuf import json_format
from google.protobuf import message as proto_message

from xprof.cli.internal import decorators
from xprof.cli.internal.oss import xprof_client
from xprof.protobuf import hlo_stats_pb2

try:
  from google3.net.rpc.python import pywraprpc  # pylint: disable=g-import-not-at-top  # pyrefly: ignore[missing-module-attribute]
except ImportError:
  pywraprpc = None

_OP_NAME_REGEX = re.compile(r"%([^%=]+) =")


@dataclasses.dataclass
class HloOperationStats:
  """Statistics for an HLO operation.

  Attributes:
    rank: The rank of the operation.
    program_id: Program ID for the operation.
    category: HLO category.
    op_name: Extracted HLO operation name.
    tf_op_name: Framework operation name.
    occurrences: Number of occurrences.
    total_time_us: Total accumulated time in microseconds.
    total_self_time_us: Total self time in microseconds.
    self_time_percent: Self time as a percentage.
    measured_flop_rate: Measured FLOP rate.
    flops: Number of FLOPs.
    measured_memory_bw_gbs: Measured memory bandwidth in GiB/s.
    bound_by: Bottleneck resource according to Roofline model.
    source_file: Source file path if available.
    source_line: Source line number if available.
  """

  rank: int
  program_id: int
  category: str
  op_name: str
  tf_op_name: str
  occurrences: int
  total_time_us: float
  total_self_time_us: float
  self_time_percent: float
  measured_flop_rate: float
  flops: float
  measured_memory_bw_gbs: float
  bound_by: str
  source_file: str
  source_line: int


def _extract_records_from_proto(
    hlo_stats_db: hlo_stats_pb2.HloStatsDatabase,
    category_filter: str | None = None,
) -> list[HloOperationStats]:
  """Extracts HloOperationStats instances from HloStatsDatabase proto."""
  records = hlo_stats_db.hlo_stats_record
  extracted_records: list[HloOperationStats] = []
  for row in records:
    if category_filter:
      row_cat = row.hlo_category or ""
      if category_filter.strip().lower() not in row_cat.lower():
        continue

    hlo_expr = row.hlo_expression or ""
    op_name_matches = _OP_NAME_REGEX.findall(hlo_expr)
    op_name = op_name_matches[0] if op_name_matches else hlo_expr[:80]

    source_file = ""
    source_line = 0
    if row.HasField("source_info"):
      source_file = row.source_info.file_name
      source_line = row.source_info.line_number

    self_time_fraction = row.total_self_time_as_fraction
    flops_val = row.flops_v2 if row.HasField("flops_v2") else float(row.flops)

    extracted_records.append(
        HloOperationStats(
            rank=row.rank,
            program_id=row.program_id,
            category=row.hlo_category,
            op_name=op_name.strip(),
            tf_op_name=row.tf_op_name.strip() if row.tf_op_name else "",
            occurrences=row.occurrences,
            total_time_us=row.total_time_in_us,
            total_self_time_us=row.total_self_time_in_us,
            self_time_percent=self_time_fraction * 100.0,
            measured_flop_rate=row.measured_flop_rate,
            flops=flops_val,
            measured_memory_bw_gbs=row.measured_memory_bw,
            bound_by=row.bound_by,
            source_file=source_file,
            source_line=source_line,
        )
    )
  return extracted_records


def _safe_float(val: Any, default: float = 0.0) -> float:
  """Safely converts a value to float, returning default on failure."""
  if val is None:
    return default
  try:
    return float(val)
  except (ValueError, TypeError):
    return default


def _safe_int(val: Any, default: int = 0) -> int:
  """Safely converts a value to int without 64-bit float precision loss."""
  if val is None:
    return default
  try:
    if isinstance(val, int):
      return val
    if isinstance(val, str) and "." not in val:
      return int(val)
    return int(float(val))
  except (ValueError, TypeError):
    return default


def _get_cell_val(
    row_cells: list[Any],
    col_indices: dict[str, int],
    *keys: str,
    default: Any = None,
) -> Any:
  """Returns the first non-empty cell value matching any candidate key."""
  for key in keys:
    idx = col_indices.get(key.lower())
    if idx is not None and idx < len(row_cells):
      cell = row_cells[idx]
      val = cell.get("v", default) if isinstance(cell, dict) else cell
      if val is not None:
        if isinstance(val, str) and not val.strip():
          continue
        return val
  return default


def _parse_hlo_stats_datatable(
    table_json: dict[str, Any],
    category_filter: str | None = None,
) -> list[HloOperationStats]:
  """Parses HLO operation statistics from Google DataTable JSON."""
  cols = [
      c.get("id") or c.get("label", f"col_{i}")
      for i, c in enumerate(table_json.get("cols", []))
  ]
  col_indices = {col.lower(): i for i, col in enumerate(cols)}

  records: list[HloOperationStats] = []
  for row in table_json.get("rows", []):
    cells = row.get("c", [])
    category = str(
        _get_cell_val(
            cells, col_indices, "hlo_category", "category", default=""
        )
    )
    if category_filter:
      if category_filter.strip().lower() not in category.lower():
        continue

    hlo_expr = str(
        _get_cell_val(
            cells,
            col_indices,
            "hlo_op_expression",
            "hlo_expression",
            "expression",
            default="",
        )
    )
    op_name_matches = _OP_NAME_REGEX.findall(hlo_expr)
    if op_name_matches:
      op_name = op_name_matches[0]
    elif hlo_expr:
      op_name = hlo_expr[:80]
    else:
      op_name = str(
          _get_cell_val(
              cells, col_indices, "hlo_op_name", "op_name", default=""
          )
      )

    rank = _safe_int(_get_cell_val(cells, col_indices, "rank"))
    program_id = _safe_int(_get_cell_val(cells, col_indices, "program_id"))
    tf_op_name = str(
        _get_cell_val(
            cells,
            col_indices,
            "tf_op_name",
            "framework_op_name",
            default="",
        )
    )
    occurrences = _safe_int(_get_cell_val(cells, col_indices, "occurrences"))
    total_time_us = _safe_float(
        _get_cell_val(
            cells,
            col_indices,
            "total_time_in_us",
            "total_time_us",
            "total_time",
        )
    )
    total_self_time_us = _safe_float(
        _get_cell_val(
            cells,
            col_indices,
            "total_self_time_in_us",
            "total_self_time_us",
            "total_self_time",
            "self_time",
        )
    )

    self_time_frac_val = _get_cell_val(
        cells, col_indices, "total_self_time_as_fraction"
    )
    if self_time_frac_val is not None:
      self_time_percent = _safe_float(self_time_frac_val) * 100.0
    else:
      self_time_percent = _safe_float(
          _get_cell_val(
              cells,
              col_indices,
              "total_self_time_percent",
              "self_time_percent",
              "self_time_pct",
          )
      )

    measured_flop_rate = _safe_float(
        _get_cell_val(
            cells,
            col_indices,
            "measured_flop_rate",
            "model_flop_rate",
            "normalized_flop_rate",
            "flop_rate",
        )
    )
    flops = _safe_float(_get_cell_val(cells, col_indices, "flops_v2", "flops"))
    measured_memory_bw_gbs = _safe_float(
        _get_cell_val(
            cells,
            col_indices,
            "measured_memory_bw",
            "hbm_bw",
            "memory_bw",
            "bandwidth",
        )
    )
    bound_by = str(
        _get_cell_val(cells, col_indices, "bound_by", default="Unknown")
    )
    source_file = str(
        _get_cell_val(
            cells, col_indices, "source_file", "file_name", default=""
        )
    )
    source_line = _safe_int(
        _get_cell_val(cells, col_indices, "source_line", "line_number")
    )
    if not source_file:
      source_info = str(
          _get_cell_val(cells, col_indices, "source_info", default="")
      )
      if ":" in source_info:
        parts = source_info.rsplit(":", 1)
        source_file = parts[0]
        if not source_line and parts[1].isdigit():
          source_line = int(parts[1])
      elif source_info:
        source_file = source_info

    records.append(
        HloOperationStats(
            rank=rank,
            program_id=program_id,
            category=category,
            op_name=op_name.strip(),
            tf_op_name=tf_op_name.strip(),
            occurrences=occurrences,
            total_time_us=total_time_us,
            total_self_time_us=total_self_time_us,
            self_time_percent=self_time_percent,
            measured_flop_rate=measured_flop_rate,
            flops=flops,
            measured_memory_bw_gbs=measured_memory_bw_gbs,
            bound_by=bound_by,
            source_file=source_file,
            source_line=source_line,
        )
    )
  return records


def _parse_hlo_stats_payload(
    data: bytes | str,
    category_filter: str | None = None,
) -> list[HloOperationStats]:
  """Parses raw payload into HloOperationStats records."""
  hlo_stats_db = hlo_stats_pb2.HloStatsDatabase()
  if isinstance(data, bytes):
    try:
      hlo_stats_db.ParseFromString(data)
      return _extract_records_from_proto(hlo_stats_db, category_filter)
    except proto_message.DecodeError:
      decoded_data = data.decode("utf-8", errors="replace")
  else:
    decoded_data = data

  decoded_data_trimmed = decoded_data.strip()
  if decoded_data_trimmed.startswith("{"):
    try:
      parsed_json = json.loads(decoded_data_trimmed)
      if isinstance(parsed_json, dict) and "cols" in parsed_json:
        return _parse_hlo_stats_datatable(parsed_json, category_filter)
    except (ValueError, TypeError, json.JSONDecodeError):
      pass

  try:
    json_format.Parse(decoded_data, hlo_stats_db)
    return _extract_records_from_proto(hlo_stats_db, category_filter)
  except (json_format.ParseError, json.JSONDecodeError) as parse_err:
    logging.exception("Failed to parse data as HloStatsDatabase proto")
    raise ValueError(
        f"Failed to parse HloStatsDatabase proto: {parse_err!r}"
    ) from parse_err


@decorators.cached(expire=86400)
def get_hlo_stats(
    session_id: str,
    *,
    limit: int = 20,
    sort_by: str = "self_time",
    category_filter: str | None = None,
    bypass_cache: bool = False,
) -> str:
  """Fetches detailed performance statistics for HLO operations.

  Args:
    session_id: The unique XProf session ID.
    limit: The maximum number of records to return. Defaults to 20.
    sort_by: The metric to sort by. Options: 'self_time', 'total_time',
      'occurrences', 'flops', 'bandwidth'. Defaults to 'self_time'.
    category_filter: Optional category name to filter operations.
    bypass_cache: Whether to bypass cache and recompute metrics.

  Returns:
    A JSON-formatted string representing the list of HLO operation statistics.

  Raises:
    FileNotFoundError: If no HLO stats records are found for the session.
    RuntimeError: If fetching or parsing HLO stats fails.
  """
  fetch_errors: list[type[Exception]] = [ValueError, OSError, RuntimeError]
  if pywraprpc is not None:
    fetch_errors.append(pywraprpc.RPCException)

  client = xprof_client.get_client()
  fetch_kwargs: dict[str, Any] = dict(
      tool_name="hlo_stats.json",
      session_id=session_id,
      format="json",
      tqx="out:pb",
  )
  if bypass_cache:
    fetch_kwargs["bypass_cache"] = True

  try:
    result = client.fetch(**fetch_kwargs)
  except tuple(fetch_errors) as e:
    logging.exception("Error fetching HLO stats for session %s", session_id)
    raise RuntimeError(
        f"Error fetching HLO stats for session {session_id}: {e!r}"
    ) from e

  if result is None:
    raise RuntimeError(f"Failed to fetch hlo_stats for session {session_id}")

  if isinstance(result, tuple) and len(result) == 2:
    _, data = result
  else:
    data = result

  if not isinstance(data, (bytes, str)):
    raise RuntimeError(f"Unexpected data type returned: {type(data)}")

  extracted_records = _parse_hlo_stats_payload(
      data, category_filter=category_filter
  )

  if not extracted_records:
    raise FileNotFoundError("No HLO stats records found")

  # Sort the records
  sort_key_map = {
      "self_time": lambda x: x.total_self_time_us,
      "total_time": lambda x: x.total_time_us,
      "occurrences": lambda x: x.occurrences,
      "flops": lambda x: x.flops,
      "bandwidth": lambda x: x.measured_memory_bw_gbs,
  }
  sort_fn = sort_key_map.get(
      sort_by.strip().lower(), lambda x: x.total_self_time_us
  )
  extracted_records.sort(key=sort_fn, reverse=True)

  # Truncate to limit
  if limit > 0:
    extracted_records = extracted_records[:limit]

  return json.dumps(
      [dataclasses.asdict(record) for record in extracted_records], indent=2
  )
