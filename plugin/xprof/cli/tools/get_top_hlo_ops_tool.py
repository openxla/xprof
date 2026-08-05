"""Tool to fetch top HLO operations from XProf sorted by different criteria.

This tool flattens the HLO operation profile tree and returns the top operations
sorted by Self Time, FLOPs, and Bytes Accessed.
"""

import heapq
import json
import logging
import traceback
from typing import Any, Dict, List

from google.protobuf import json_format
from google.protobuf import message

from xprof.cli.internal import decorators
from xprof.cli.internal.oss import xprof_client
from xprof.protobuf import op_profile_pb2

DecodeError = message.DecodeError


@decorators.cached(expire=86400)
def get_top_hlo_ops(
    session_id: str, *, limit: int = 10, category_filter: str | None = None
) -> str:
  """Fetches top HLO operations sorted by Time, FLOPs, and Bytes Accessed.

  Args:
      session_id: The unique XProf session ID.
      limit: Number of top operations to return per list (default is 10).
      category_filter: Optional HLO op category name to filter by (e.g.,
        'convolution' or 'fusion').

  Returns:
      A JSON-formatted string containing three lists of top operations with
      mandatory source provenance metadata whenever available.
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

  target_cat = category_filter.strip().lower() if category_filter else None

  def traverse(root_node: op_profile_pb2.Node) -> List[Dict[str, Any]]:
    results = []
    stack = [(root_node, "")]
    while stack:
      node, current_name_prefix = stack.pop()
      name = node.name
      full_name = (
          f"{current_name_prefix}/{name}" if current_name_prefix else name
      )

      # Only add leaf nodes (instructions) that have XLA info
      if node.HasField("xla") and node.HasField("metrics"):
        metrics = node.metrics
        if metrics.raw_time > 0:
          xla = node.xla
          category = xla.category
          if target_cat is None or category.strip().lower() == target_cat:
            bytes_array = metrics.raw_bytes_accessed_array
            total_bytes = sum(bytes_array) if bytes_array else 0
            item = {
                "name": full_name,
                "category": category,
                "total_self_time_ms": metrics.raw_time / 1e9,
                "occurrences": metrics.occurrences,
                "flops": metrics.raw_flops,
                "bytes_accessed": total_bytes,
            }
            if xla.HasField("source_info"):
              source_info = xla.source_info
              item["source_file"] = source_info.file_name
              item["source_line"] = source_info.line_number
              stack_frame = source_info.stack_frame
              if stack_frame:
                item["stack_frame"] = stack_frame
            results.append(item)

      children = node.children
      if children:
        for child in reversed(children):
          stack.append((child, full_name))

    return results

  if (
      op_profile.HasField("by_category")
      and op_profile.by_category.metrics.raw_time > 0
  ):
    flat_ops = traverse(op_profile.by_category)
  elif op_profile.HasField("by_program"):
    flat_ops = traverse(op_profile.by_program)
  else:
    flat_ops = []

  if not flat_ops:
    return json.dumps(
        {
            "error": "No ops found",
            "has_by_program": op_profile.HasField("by_program"),
        },
        indent=2,
    )

  top_by_time = heapq.nlargest(
      limit, flat_ops, key=lambda x: x["total_self_time_ms"]
  )
  top_by_flops = heapq.nlargest(limit, flat_ops, key=lambda x: x["flops"])
  top_by_bytes = heapq.nlargest(
      limit, flat_ops, key=lambda x: x["bytes_accessed"]
  )

  return json.dumps(
      {
          "top_by_time": top_by_time,
          "top_by_flops": top_by_flops,
          "top_by_bytes_accessed": top_by_bytes,
      },
      indent=2,
  )
