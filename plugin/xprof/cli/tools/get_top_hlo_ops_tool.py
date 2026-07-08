"""Tool to fetch top HLO operations from XProf sorted by different criteria.

This tool flattens the HLO operation profile tree and returns the top operations
sorted by Self Time, FLOPs, and Bytes Accessed.
"""

import heapq
import json
import logging
import traceback
from typing import Any, Dict, Generator

from google.protobuf import json_format
from google.protobuf.message import DecodeError

from xprof.cli.internal import decorators
from xprof.cli.internal.oss import xprof_client
from xprof.protobuf import op_profile_pb2


@decorators.cached(expire=86400)
def get_top_hlo_ops(session_id: str, *, limit: int = 10) -> str:
  """Fetches top HLO operations sorted by Time, FLOPs, and Bytes Accessed.

  Args:
      session_id: The unique XProf session ID.
      limit: Number of top operations to return per list (default is 10).

  Returns:
      A JSON-formatted string containing three lists of top operations.
  """
  client = xprof_client.get_client()
  try:
    op_profile_result = client.fetch(
        tool_name="hlo_op_profile.json",
        session_id=session_id,
        format="json",
    )
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.exception("Error fetching top HLO ops for session %s", session_id)
    return json.dumps(
        dict(
            error=f"Error fetching top HLO ops for session {session_id}: {e!r}",
            traceback=traceback.format_exc(),
        ),
        indent=2,
    )

  op_profile = op_profile_pb2.Profile()

  # Guard clauses for op_profile_result processing
  if not isinstance(op_profile_result, (tuple, bytes)):
    return json.dumps(
        dict(error=f"Failed to fetch op_profile: {op_profile_result}"), indent=2
    )

  if isinstance(op_profile_result, tuple):
    if len(op_profile_result) != 2:
      return json.dumps(dict(error="Unexpected tuple length"), indent=2)
    content_type, data = op_profile_result
    if not isinstance(data, bytes):
      return json.dumps(
          dict(error=f"Unexpected data type: {type(data)}"), indent=2
      )

    if content_type == 81 or data.strip().startswith(b"{"):
      try:
        json_format.Parse(data.decode("utf-8", errors="replace"), op_profile)
      except json_format.ParseError as e:
        return json.dumps(
            dict(error=f"Failed to parse JSON proto: {e}"), indent=2
        )
    else:
      try:
        op_profile = op_profile_pb2.Profile.FromString(data)
      except DecodeError as e:
        return json.dumps(
            dict(error=f"Failed to parse binary proto: {e}"), indent=2
        )
  else:  # bytes
    try:
      op_profile = op_profile_pb2.Profile.FromString(op_profile_result)
    except DecodeError as e:
      return json.dumps(
          dict(error=f"Failed to parse binary proto: {e}"), indent=2
      )

  def traverse(
      node: op_profile_pb2.Node, current_name_prefix: str = ""
  ) -> Generator[Dict[str, Any], None, None]:
    name = node.name
    full_name = f"{current_name_prefix}/{name}" if current_name_prefix else name
    metrics = node.metrics

    # Only add leaf nodes (instructions) that have XLA info
    if node.HasField("xla") and metrics.raw_time > 0:
      total_bytes = (
          sum(metrics.raw_bytes_accessed_array)
          if metrics.raw_bytes_accessed_array
          else 0
      )
      yield {
          "name": full_name,
          "category": node.xla.category,
          "raw_time": metrics.raw_time,
          "occurrences": metrics.occurrences,
          "flops": metrics.raw_flops,
          "bytes_accessed": total_bytes,
      }

    for child in node.children:
      yield from traverse(child, full_name)

  if (
      op_profile.HasField("by_category")
      and op_profile.by_category.metrics.raw_time > 0
  ):
    ops_iterable = traverse(op_profile.by_category)
  elif op_profile.HasField("by_program"):
    ops_iterable = traverse(op_profile.by_program)
  else:
    ops_iterable = []

  flat_ops = list(ops_iterable)

  if not flat_ops:
    return json.dumps(
        {
            "error": "No ops found",
            "has_by_program": op_profile.HasField("by_program"),
        },
        indent=2,
    )

  top_by_time = heapq.nlargest(limit, flat_ops, key=lambda x: x["raw_time"])
  top_by_flops = heapq.nlargest(limit, flat_ops, key=lambda x: x["flops"])
  top_by_bytes = heapq.nlargest(
      limit, flat_ops, key=lambda x: x["bytes_accessed"]
  )

  # Convert raw_time (ps) to total_self_time_ms for output
  for op in top_by_time + top_by_flops + top_by_bytes:
    if "raw_time" in op:
      op["total_self_time_ms"] = op["raw_time"] / 1e9
      del op["raw_time"]

  return json.dumps(
      {
          "top_by_time": top_by_time,
          "top_by_flops": top_by_flops,
          "top_by_bytes_accessed": top_by_bytes,
      },
      indent=2,
  )
