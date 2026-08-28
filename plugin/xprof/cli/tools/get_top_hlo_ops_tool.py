"""Tool to fetch top HLO operations from XProf sorted by different criteria.

This tool flattens the HLO operation profile tree and returns the top operations
sorted by Self Time, FLOPs, and Bytes Accessed.
"""

import heapq
import json
import logging
import re
from typing import Any, Dict, Generator

from google.protobuf import json_format
from google.protobuf import message

from xprof.cli.internal import decorators
from xprof.cli.internal.oss import xprof_client
from xprof.protobuf import op_profile_pb2

DecodeError = message.DecodeError


_CUSTOM_CALL_TARGET_RE = re.compile(r'custom_call_target="([^"]+)"')


@decorators.cached(expire=86400)
def get_top_hlo_ops(
    session_id: str,
    *,
    limit: int = 10,
    category_filter: str | None = None,
    bypass_cache: bool = False,
) -> str:
  """Fetches top HLO operations sorted by Time, FLOPs, and Bytes Accessed.

  Args:
      session_id: The unique XProf session ID.
      limit: Number of top operations to return per list (default is 10).
      category_filter: Optional HLO op category name to filter by (e.g.,
        'convolution' or 'fusion').
      bypass_cache: Whether to bypass cache and recompute metrics.

  Returns:
      A JSON-formatted string containing three lists of top operations with
      mandatory source provenance metadata whenever available.

  Raises:
      FileNotFoundError: If no HLO op_profile data is found in the trace.
      RuntimeError: If fetching or parsing op_profile data fails.
  """
  client = xprof_client.get_client()
  fetch_kwargs: dict[str, Any] = {
      "tool_name": "op_profile",
      "session_id": session_id,
      "format": "pb",
  }
  if bypass_cache:
    fetch_kwargs["bypass_cache"] = True
  try:
    op_profile_result = client.fetch(**fetch_kwargs)
    if not op_profile_result or (
        isinstance(op_profile_result, tuple) and not op_profile_result[1]
    ):
      fetch_kwargs["tool_name"] = "hlo_op_profile.json"
      op_profile_result = client.fetch(**fetch_kwargs)
  except (FileNotFoundError, ValueError):
    raise
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.exception("Error fetching top HLO ops for session %s", session_id)
    raise RuntimeError(
        f"Error fetching top HLO ops for session {session_id}: {e!r}"
    ) from e

  op_profile = op_profile_pb2.Profile()

  # Guard clauses for op_profile_result processing
  if not isinstance(op_profile_result, (tuple, bytes)):
    raise RuntimeError(f"Failed to fetch op_profile: {op_profile_result}")

  if isinstance(op_profile_result, tuple):
    if len(op_profile_result) != 2:
      raise RuntimeError("Unexpected tuple length")
    content_type, data = op_profile_result
    if isinstance(data, str):
      data = data.encode("utf-8")
    if not isinstance(data, bytes):
      if data is None or not data:
        raise FileNotFoundError(
            "No HLO op_profile found in trace. For JAX traces, ensure"
            " compilation is captured in the trace or pass"
            " XLA_FLAGS='--xla_dump_to=<path> --xla_dump_hlo_as_proto'."
        )
      raise RuntimeError(f"Unexpected data type: {type(data)}")

    if content_type == 81 or data.strip().startswith(b"{"):
      try:
        json_format.Parse(data.decode("utf-8", errors="replace"), op_profile)
      except json_format.ParseError as e:
        return json.dumps(
            dict(error=f"Failed to parse JSON proto: {e}"), indent=2
        )
    else:
      try:
        op_profile.ParseFromString(data)
      except DecodeError as e:
        return json.dumps(
            dict(error=f"Failed to parse binary proto: {e}"), indent=2
        )
  else:  # bytes
    try:
      op_profile.ParseFromString(op_profile_result)
    except DecodeError as e:
      return json.dumps(
          dict(error=f"Failed to parse binary proto: {e}"), indent=2
      )

  def traverse(
      node: op_profile_pb2.Node, current_name_prefix: str = ""
  ) -> Generator[Dict[str, Any], None, None]:
    name = node.name
    metrics = node.metrics

    # Only add leaf nodes (instructions) that have XLA info
    if node.HasField("xla") and metrics.raw_time > 0:
      category = node.xla.category
      op_label = name
      if node.xla.provenance:
        if name in ("IDLE", "idle", "unknown", "") or category.lower() in (
            "custom-call",
            "custom_call",
        ):
          op_label = (
              f"{name} [{node.xla.provenance}]"
              if name != node.xla.provenance
              else name
          )
      elif node.xla.expression and category.lower() in (
          "custom-call",
          "custom_call",
      ):
        match = _CUSTOM_CALL_TARGET_RE.search(node.xla.expression)
        if match:
          target = match.group(1)
          op_label = f"{name} [{target}]" if name != target else name

      full_name = (
          f"{current_name_prefix}/{op_label}"
          if current_name_prefix
          else op_label
      )
      total_bytes = (
          sum(metrics.raw_bytes_accessed_array)
          if metrics.raw_bytes_accessed_array
          else 0
      )
      item = {
          "name": full_name,
          "category": category,
          "total_self_time_ms": metrics.raw_time / 1e9,
          "occurrences": metrics.occurrences,
          "flops": metrics.raw_flops,
          "bytes_accessed": total_bytes,
      }
      if node.xla.HasField("source_info"):
        item["source_file"] = node.xla.source_info.file_name
        item["source_line"] = node.xla.source_info.line_number
        if node.xla.source_info.stack_frame:
          item["stack_frame"] = node.xla.source_info.stack_frame
      yield item

    for child in node.children:
      child_prefix = (
          f"{current_name_prefix}/{name}" if current_name_prefix else name
      )
      yield from traverse(child, child_prefix)

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
  if category_filter:
    target_cat = category_filter.strip().lower()
    flat_ops = [
        op
        for op in flat_ops
        if op.get("category", "").strip().lower() == target_cat
    ]

  if not flat_ops:
    return json.dumps(
        {
            "top_by_time": [],
            "top_by_flops": [],
            "top_by_bytes_accessed": [],
            "total_matched": 0,
            "has_by_program": op_profile.HasField("by_program"),
        },
        indent=2,
    )

  if limit > 0:
    top_by_time = heapq.nlargest(
        limit, flat_ops, key=lambda x: x["total_self_time_ms"]
    )
    top_by_flops = heapq.nlargest(limit, flat_ops, key=lambda x: x["flops"])
    top_by_bytes = heapq.nlargest(
        limit, flat_ops, key=lambda x: x["bytes_accessed"]
    )
  else:
    top_by_time = sorted(
        flat_ops, key=lambda x: x["total_self_time_ms"], reverse=True
    )
    top_by_flops = sorted(flat_ops, key=lambda x: x["flops"], reverse=True)
    top_by_bytes = sorted(
        flat_ops, key=lambda x: x["bytes_accessed"], reverse=True
    )

  has_custom_call = any(
      op.get("category", "").lower() in ("custom-call", "custom_call")
      or "custom-call" in op.get("name", "").lower()
      for op in top_by_time + top_by_flops + top_by_bytes
  )

  result_payload: dict[str, Any] = {
      "top_by_time": top_by_time,
      "top_by_flops": top_by_flops,
      "top_by_bytes_accessed": top_by_bytes,
      "total_matched": len(flat_ops),
  }
  if has_custom_call:
    result_payload["guidance"] = (
        "Op-level metrics unavailable for custom calls. Use get_llo_analysis,"
        " get_llo_debug_string, and aggregate_xplane_events for Pallas kernels."
    )

  return json.dumps(result_payload, indent=2)
