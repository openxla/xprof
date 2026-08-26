"""Tool to fetch detailed HLO performance statistics from XProf."""

import dataclasses
import json
import logging
import re
import traceback

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


@decorators.cached(expire=86400)
def get_hlo_stats(
    session_id: str,
    *,
    limit: int = 20,
    sort_by: str = "self_time",
    category_filter: str | None = None,
    output_format: str = "markdown",
) -> str:
  """Fetches detailed performance statistics for HLO operations.

  Args:
    session_id: The unique XProf session ID.
    limit: The maximum number of records to return. Defaults to 20.
    sort_by: The metric to sort by. Options: 'self_time', 'total_time',
      'occurrences', 'flops', 'bandwidth'. Defaults to 'self_time'.
    category_filter: Optional category name to filter operations.
    output_format: The format of the output. Options: 'markdown', 'json'.
      Defaults to 'markdown'.

  Returns:
    A Markdown table or JSON-formatted string of HLO operation statistics.
  """
  fetch_errors: list[type[Exception]] = [ValueError, OSError, RuntimeError]
  if pywraprpc is not None:
    fetch_errors.append(pywraprpc.RPCException)

  client = xprof_client.get_client()
  try:
    result = client.fetch(
        tool_name="hlo_stats.json",
        session_id=session_id,
        format="json",
        tqx="out:pb",
    )
  except tuple(fetch_errors) as e:
    logging.exception("Error fetching HLO stats for session %s", session_id)
    return json.dumps(
        dict(
            error=f"Error fetching HLO stats for session {session_id}: {e!r}",
            traceback=traceback.format_exc(),
        ),
        indent=2,
    )

  if result is None:
    return json.dumps(
        dict(error=f"Failed to fetch hlo_stats for session {session_id}"),
        indent=2,
    )

  if isinstance(result, tuple) and len(result) == 2:
    _, data = result
  else:
    data = result

  if not isinstance(data, (bytes, str)):
    return json.dumps(
        dict(error=f"Unexpected data type returned: {type(data)}"),
        indent=2,
    )

  hlo_stats_db = hlo_stats_pb2.HloStatsDatabase()
  if isinstance(data, bytes):
    try:
      hlo_stats_db.ParseFromString(data)
    except proto_message.DecodeError:
      # Fallback to json parsing if it was a json representation
      decoded_data = data.decode("utf-8", errors="replace")
      try:
        json_format.Parse(decoded_data, hlo_stats_db)
      except (json_format.ParseError, json.JSONDecodeError) as parse_err:
        logging.exception("Failed to parse data as HloStatsDatabase proto")
        return json.dumps(
            dict(
                error=f"Failed to parse HloStatsDatabase proto: {parse_err!r}"
            ),
            indent=2,
        )
  else:  # data is str
    try:
      json_format.Parse(data, hlo_stats_db)
    except (json_format.ParseError, json.JSONDecodeError) as parse_err:
      logging.exception("Failed to parse data as HloStatsDatabase proto")
      return json.dumps(
          dict(error=f"Failed to parse HloStatsDatabase proto: {parse_err!r}"),
          indent=2,
      )

  # Check if records exist
  records = hlo_stats_db.hlo_stats_record
  if not records:
    return json.dumps(dict(error="No HLO stats records found"), indent=2)

  # Extract, filter, and format the records using HloOperationStats dataclass
  extracted_records: list[HloOperationStats] = []
  for row in records:
    # Filter by category if specified
    if category_filter:
      row_cat = row.hlo_category or ""
      if category_filter.strip().lower() not in row_cat.lower():
        continue

    # Extract clean op name from hlo_expression
    hlo_expr = row.hlo_expression or ""
    op_name_matches = _OP_NAME_REGEX.findall(hlo_expr)
    op_name = op_name_matches[0] if op_name_matches else hlo_expr[:80]

    # Handle source provenance
    source_file = ""
    source_line = 0
    if row.HasField("source_info"):
      source_file = row.source_info.file_name
      source_line = row.source_info.line_number

    # total_self_time_as_fraction is a fraction
    self_time_fraction = row.total_self_time_as_fraction

    # Flops mapping
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

  if not extracted_records:
    return json.dumps(
        dict(error="No records matched the filter criteria"), indent=2
    )

  if output_format.strip().lower() == "json":
    return json.dumps(
        [dataclasses.asdict(record) for record in extracted_records], indent=2
    )

  # Markdown format
  md_lines = [
      (
          "| Rank | Category | Op Name | Occurrences | Self Time (us) | Self"
          " Time (%) | FLOP Rate | Memory BW (GiB/s) | Bound By | Source |"
      ),
      "|---:|---|---|---:|---:|---:|---:|---:|---|---|",
  ]
  for record in extracted_records:
    source_str = ""
    if record.source_file:
      source_file_basename = record.source_file.split("/")[-1]
      source_str = f"{source_file_basename}:{record.source_line}"

    md_lines.append(
        f"| {record.rank} "
        f"| {record.category} "
        f"| {record.op_name} "
        f"| {record.occurrences} "
        f"| {record.total_self_time_us:.2f} "
        f"| {record.self_time_percent:.2f}% "
        f"| {record.measured_flop_rate:.4f} "
        f"| {record.measured_memory_bw_gbs:.2f} "
        f"| {record.bound_by} "
        f"| {source_str} |"
    )

  return "\n".join(md_lines)
