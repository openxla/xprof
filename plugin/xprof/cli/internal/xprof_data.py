"""Data fetching tools for XProf MCP."""

import json
import logging
import traceback

from google.protobuf import json_format

from xprof.cli.internal import decorators
from xprof.cli.internal.oss import xprof_client
from xprof.protobuf import op_profile_pb2


@decorators.cached(expire=86400)
def get_profile_summary(session_id: str) -> str:
  """Provides a high-level performance summary of an XProf session.

  **START HERE.** This tool identifies the top bottlenecks, step time, and HBM
  usage. Use its output to decide which operations deserve a deeper dive.

  Args:
    session_id: The unique XProf session ID.

  Returns:
    A executive-level text summary of the profile's performance landscape.
  """

  client = xprof_client.get_client()
  try:
    # Fetch Op Profile for Top Ops
    # Use hlo_op_profile.json as op_profile.pb is not available in some
    # environments.
    op_profile_result = client.fetch(
        tool_name="hlo_op_profile.json",
        session_id=session_id,
        # usually auto-detected
        format="json",
    )
    op_profile = op_profile_pb2.Profile()

    if isinstance(op_profile_result, tuple) and len(op_profile_result) == 2:
      content_type, data = op_profile_result
      if isinstance(data, bytes):
        # type 81 is JSON, type 80/None might be PB.
        # Check if it looks like JSON or if content_type matches.
        if content_type == 81 or data.strip().startswith(b"{"):
          json_str = data.decode("utf-8", errors="replace")
          json_format.Parse(json_str, op_profile)
        else:
          op_profile.ParseFromString(data)
      else:
        return (
            f"Unexpected data type for op_profile: {type(data)} (data={data})"
        )
    elif isinstance(op_profile_result, bytes):
      # Fallback for unexpected return type
      op_profile.ParseFromString(op_profile_result)
    else:
      return f"Failed to fetch op_profile: {op_profile_result}"

    # Analyze Op Profile
    def extract_top_ops(node, limit=10):
      # Traverse to find leaf nodes or interesting nodes
      all_nodes = []

      def walk(n):
        if len(all_nodes) >= limit:
          return  # Stop if limit is reached

        if n.metrics.raw_time > 0:
          all_nodes.append(n)
          if len(all_nodes) >= limit:
            return  # Stop after appending if limit is reached

        for child in n.children:
          if len(all_nodes) >= limit:
            return  # Stop before recursing if limit is reached
          walk(child)

      walk(node)
      return sorted(all_nodes, key=lambda n: n.metrics.raw_time, reverse=True)[
          :limit
      ]

    root = None
    if op_profile.by_category and op_profile.by_category.metrics.raw_time > 0:
      root = op_profile.by_category
    elif op_profile.by_program:
      root = op_profile.by_program

    if not root:
      return (
          f"Profile Summary for {session_id}\nNo performance data found in"
          " op_profile."
      )

    total_time_ps = root.metrics.raw_time

    lines = []
    lines.append(f"Profile Summary for {session_id}")
    if total_time_ps > 0:
      lines.append(f"Total Time: {total_time_ps / 1e12:.4f} s")

    lines.append("\nTop Operations (by self time):")
    lines.append("| Name | Self Time (s) | Fraction |")
    lines.append("|---|---|---|")

    top_nodes = extract_top_ops(root)

    for child in top_nodes:
      name = child.name if child.name else "Unknown"
      # Escape pipes in name to avoid breaking table
      name = name.replace("|", "\\|")
      time_s = child.metrics.raw_time / 1e12
      fraction = child.metrics.raw_time / total_time_ps if total_time_ps else 0
      lines.append(f"| {name} | {time_s:.4f} | {fraction:.1%} |")

    return "\n".join(lines)

  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.exception("Error analyzing profile")
    return (
        "Error analyzing profile:"
        f" {''.join(traceback.format_exception_only(e)).strip()}"
    )


@decorators.cached(expire=86400)
def get_hlo_op_profile(session_id: str, top_n: int = 15) -> str:
  """Summarizes the most expensive HLO operations in the session.

  **Use this** to identify candidates for optimization. It provides
  self-time, FLOPS, and bytes accessed for the top N operations.

  Args:
      session_id: The unique XProf session ID.
      top_n: Number of top operations to return (default is 15).

  Returns:
      A JSON-formatted list of operations with detailed performance metrics.
  """
  client = xprof_client.get_client()
  try:
    op_profile_result = client.fetch(
        tool_name="hlo_op_profile.json",
        session_id=session_id,
        format="json",
    )
    op_profile = op_profile_pb2.Profile()

    if isinstance(op_profile_result, tuple) and len(op_profile_result) == 2:
      content_type, data = op_profile_result
      if isinstance(data, bytes):
        if content_type == 81 or data.strip().startswith(b"{"):
          json_str = data.decode("utf-8", errors="replace")
          json_format.Parse(json_str, op_profile)
        else:
          op_profile.ParseFromString(data)
      else:
        return f"Unexpected data type for op_profile: {type(data)}"
    elif isinstance(op_profile_result, bytes):
      op_profile.ParseFromString(op_profile_result)
    else:
      return f"Failed to fetch op_profile: {op_profile_result}"

    # Traverse the tree to flatten ops
    # This is a bit complex as OpProfile is hierarchical.
    # We will collect all leaf nodes (or nodes with self time > 0)

    flat_ops = []

    def traverse(node, current_name_prefix=""):
      name = node.name
      full_name = (
          f"{current_name_prefix}/{name}" if current_name_prefix else name
      )

      # Helper to get metrics safe
      metrics = node.metrics
      if metrics.raw_time > 0:
        # Sum bytes accessed across all memory types if array exists
        total_bytes = (
            sum(metrics.raw_bytes_accessed_array)
            if metrics.raw_bytes_accessed_array
            else 0
        )

        category_str = "unknown"
        if node.HasField("xla"):
          category_str = node.xla.category
        elif node.HasField("category"):
          category_str = "Category: " + name

        flat_ops.append({
            "name": full_name,
            "category": category_str,
            "total_self_time_ps": metrics.raw_time,
            "occurrences": metrics.occurrences,
            "flops": metrics.raw_flops,
            "bytes_accessed": total_bytes,
        })

      for child in node.children:
        traverse(child, full_name)

    if op_profile.by_category and op_profile.by_category.metrics.raw_time > 0:
      traverse(op_profile.by_category)
    elif op_profile.by_program:
      traverse(op_profile.by_program)

    # If still empty, return error
    if not flat_ops:
      return json.dumps(
          {
              "error": "No ops found",
              "has_by_program": bool(op_profile.by_program),
          },
          indent=2,
      )

    # Sort by self_time descending
    flat_ops.sort(key=lambda x: x["total_self_time_ps"], reverse=True)

    top_ops = flat_ops[:top_n]

    # Convert picoseconds to seconds/ms for readability
    for op in top_ops:
      op["total_self_time_ms"] = op["total_self_time_ps"] / 1e9
      del op["total_self_time_ps"]

    return json.dumps(top_ops, indent=2)

  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.exception("Error fetching HLO op profile")
    return (
        "Error fetching HLO op profile:"
        f" {''.join(traceback.format_exception_only(e)).strip()}"
    )


@decorators.cached(expire=86400)
def get_hosts(session_id: str) -> str:
  """Returns the list of hosts profiled in the session.

  **Use this** to see which machines participated in the profile, including
  metadata like hostnames.

  Args:
      session_id: The unique XProf session ID.

  Returns:
      A JSON-formatted dict containing a list of hosts or an error.
  """
  client = xprof_client.get_client()
  try:
    hosts_data = client.get_hosts(session_id, with_metadata=True)
    if not hosts_data:
      return json.dumps(dict(error="No hosts found"), indent=2)

    return json.dumps(dict(hosts=hosts_data), indent=2)
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.exception("Error fetching hosts")
    return json.dumps(
        dict(
            error=(
                "Error fetching hosts:"
                f" {''.join(traceback.format_exception_only(e)).strip()}"
            )
        ),
        indent=2,
    )


@decorators.cached(expire=86400)
def get_device_information(session_id: str) -> str:
  """Returns hardware device information from the Roofline Model analysis.

  **Use this** to retrieve device specs such as the accelerator type,
  peak FLOP rate, peak memory bandwidths, and ridge points.

  Args:
      session_id: The unique XProf session ID.

  Returns:
      A JSON-formatted dict of device information properties extracted
      from the Roofline Model DataTable. Numeric values are auto-converted
      to floats.
  """
  client = xprof_client.get_client()
  try:
    result = client.fetch(
        tool_name="roofline_model.json",
        session_id=session_id,
    )

    if isinstance(result, tuple) and len(result) == 2:
      _, data = result
    else:
      data = result

    if not data:
      return json.dumps(dict(error="No roofline model data returned"), indent=2)

    if isinstance(data, bytes):
      data = data.decode("utf-8", errors="replace")

    roofline_data = json.loads(data)

    if not isinstance(roofline_data, list) or not roofline_data:
      return json.dumps(
          dict(error="Unexpected roofline model data format"), indent=2
      )

    table_props = roofline_data[0].get("p", {})

    device_info = {}
    for key, value in table_props.items():
      try:
        value = float(value)
      except (ValueError, TypeError):
        pass
      device_info[key] = value

    return json.dumps(device_info, indent=2)

  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.exception("Error fetching device information")
    return json.dumps(
        dict(
            error=(
                "Error fetching device information:"
                f" {''.join(traceback.format_exception_only(e)).strip()}"
            )
        ),
        indent=2,
    )
