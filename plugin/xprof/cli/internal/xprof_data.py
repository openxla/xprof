"""Data fetching tools for XProf MCP."""

import json
import logging

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

  Raises:
    FileNotFoundError: If no HLO op_profile data is found in the profile.
    ValueError: If op_profile proto parsing fails.
    RuntimeError: If fetching or parsing the profile fails.
  """
  client = xprof_client.get_client()
  try:
    # Fetch Op Profile for Top Ops
    op_profile_result = client.fetch(
        tool_name="op_profile",
        session_id=session_id,
        format="json",
    )
    if not op_profile_result or (
        isinstance(op_profile_result, tuple) and not op_profile_result[1]
    ):
      op_profile_result = client.fetch(
          tool_name="hlo_op_profile.json",
          session_id=session_id,
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
          try:
            json_format.Parse(json_str, op_profile)
          except (json_format.ParseError, json.JSONDecodeError) as e:
            raise ValueError(f"Failed to parse op_profile proto: {e!r}") from e
        else:
          try:
            op_profile.ParseFromString(data)
          except Exception as e:
            raise ValueError(f"Failed to parse op_profile proto: {e!r}") from e
      else:
        if data is None:
          raise FileNotFoundError(
              "No HLO op_profile data found in profile for session"
              f" {session_id}."
          )
        raise RuntimeError(
            f"Unexpected data type for op_profile: {type(data)} (data={data})"
        )
    elif isinstance(op_profile_result, bytes):
      try:
        op_profile.ParseFromString(op_profile_result)
      except Exception as e:
        raise ValueError(f"Failed to parse op_profile proto: {e!r}") from e
    else:
      raise RuntimeError(f"Failed to fetch op_profile: {op_profile_result}")

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
      raise FileNotFoundError(
          f"No performance data found in op_profile for session {session_id}."
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

  except (FileNotFoundError, ValueError):
    raise
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.exception("Error analyzing profile for session %s", session_id)
    raise RuntimeError(
        f"Error analyzing profile for session {session_id}: {e!r}"
    ) from e


@decorators.cached(expire=86400)
def get_hlo_op_profile(
    session_id: str,
    top_n: int = 15,
    view: str = "grouped",
    category: str | None = None,
    path: str | None = None,
    depth: int = 2,
    sort_by: str = "time",
    bypass_cache: bool = False,
) -> str:
  """Summarizes HLO operations with progressive hierarchy and navigation hints.

  **Use this** to identify candidates for optimization. It provides
  progressive category breakdowns, grouped operations, subtree tree navigation,
  and micro-level leaf operation metrics.

  Args:
      session_id: The unique XProf session ID.
      top_n: Number of top operations to return (per category or overall).
      view: View mode ('grouped', 'category', 'summary', 'flat', 'tree').
        Default is 'grouped'.
      category: Optional category name filter (e.g. 'convolution',
        'all-gather').
      path: Subtree path for tree exploration when view='tree' (e.g.
        'by_category').
      depth: Max traversal depth when view='tree' (default is 2).
      sort_by: Metric to sort by ('time', 'flops', 'bytes'). Default is 'time'.
      bypass_cache: Whether to bypass cache and recompute metrics.

  Returns:
      A JSON-formatted string with performance data and structured navigation
      hints.

  Raises:
      FileNotFoundError: If no HLO op_profile operations match or path/category
      is not found.
      ValueError: If an invalid view mode or invalid arguments are passed.
      RuntimeError: If fetching or processing op_profile fails.
  """
  normalized_view = view.lower().strip()
  if normalized_view not in ("grouped", "category", "summary", "flat", "tree"):
    raise ValueError(
        f"Invalid view mode '{view}'. Must be one of: 'grouped', 'category',"
        " 'summary', 'flat', 'tree'."
    )

  normalized_sort_by = sort_by.lower().strip()
  if normalized_sort_by not in ("time", "flops", "bytes"):
    raise ValueError(
        f"Invalid sort_by '{sort_by}'. Must be one of: 'time', 'flops',"
        " 'bytes'."
    )

  client = xprof_client.get_client()
  try:
    op_profile_result = client.fetch(
        tool_name="op_profile",
        session_id=session_id,
        format="json",
        bypass_cache=bypass_cache,
    )
    if not op_profile_result or (
        isinstance(op_profile_result, tuple) and not op_profile_result[1]
    ):
      op_profile_result = client.fetch(
          tool_name="hlo_op_profile.json",
          session_id=session_id,
          format="json",
          bypass_cache=bypass_cache,
      )
    op_profile = op_profile_pb2.Profile()

    if isinstance(op_profile_result, tuple) and len(op_profile_result) == 2:
      content_type, data = op_profile_result
      if isinstance(data, bytes):
        if content_type == 81 or data.strip().startswith(b"{"):
          json_str = data.decode("utf-8", errors="replace")
          try:
            json_format.Parse(json_str, op_profile)
          except (json_format.ParseError, json.JSONDecodeError) as e:
            raise ValueError(f"Failed to parse op_profile proto: {e!r}") from e
        else:
          try:
            op_profile.ParseFromString(data)
          except Exception as e:
            raise ValueError(f"Failed to parse op_profile proto: {e!r}") from e
      else:
        if data is None:
          raise FileNotFoundError(
              f"No HLO op_profile found in trace for session {session_id}."
          )
        raise RuntimeError(f"Unexpected data type for op_profile: {type(data)}")
    elif isinstance(op_profile_result, bytes):
      try:
        op_profile.ParseFromString(op_profile_result)
      except Exception as e:
        raise ValueError(f"Failed to parse op_profile proto: {e!r}") from e
    else:
      raise RuntimeError(f"Failed to fetch op_profile: {op_profile_result}")

    root = None
    if (
        op_profile.HasField("by_category")
        and op_profile.by_category.metrics.raw_time > 0
    ):
      root = op_profile.by_category
    elif (
        op_profile.HasField("by_program")
        and op_profile.by_program.metrics.raw_time > 0
    ):
      root = op_profile.by_program

    if not root:
      raise FileNotFoundError(
          "No HLO op_profile operations found in profile for session"
          f" {session_id}."
      )

    total_time_ms = root.metrics.raw_time / 1e9

    # Traverse the tree to extract leaf operations
    flat_ops = []

    def traverse(node, current_name_prefix=""):
      name = node.name
      full_name = (
          f"{current_name_prefix}/{name}" if current_name_prefix else name
      )

      metrics = node.metrics
      # Only emit leaf instructions with non-zero execution time
      is_xla_leaf = node.HasField("xla") and metrics.raw_time > 0
      is_other_leaf = not node.children and metrics.raw_time > 0

      if is_xla_leaf or is_other_leaf:
        total_bytes = (
            sum(metrics.raw_bytes_accessed_array)
            if metrics.raw_bytes_accessed_array
            else 0
        )

        if node.HasField("xla") and node.xla.category:
          category_str = node.xla.category
        elif node.HasField("category"):
          category_str = f"Category: {name}"
        else:
          category_str = "unknown"
          lower_name = name.lower()
          for cat in (
              "custom-call",
              "fusion",
              "convolution",
              "dot",
              "reduce",
              "copy",
              "reshape",
              "broadcast",
              "while",
              "tuple",
          ):
            if cat in lower_name:
              category_str = cat
              break
          if category_str == "unknown" and name.startswith("%"):
            category_str = name.lstrip("%").split(".")[0].split("_")[0]

        occurrences = metrics.occurrences if metrics.occurrences > 0 else 1

        item = {
            "name": full_name,
            "category": category_str,
            "total_self_time_ms": round(metrics.raw_time / 1e9, 4),
            "occurrences": occurrences,
            "flops": metrics.raw_flops,
            "bytes_accessed": total_bytes,
        }
        if node.HasField("xla") and node.xla.HasField("source_info"):
          if node.xla.source_info.file_name:
            item["source_file"] = node.xla.source_info.file_name
          if node.xla.source_info.line_number > 0:
            item["source_line"] = node.xla.source_info.line_number
          if node.xla.source_info.stack_frame:
            item["stack_frame"] = node.xla.source_info.stack_frame
        flat_ops.append(item)

      for child in node.children:
        traverse(child, full_name)

    traverse(root)

    # If still empty, return error
    if not flat_ops:
      raise FileNotFoundError(
          "No HLO op_profile operations found in profile for session"
          f" {session_id}."
      )

    def _sort_key(op_dict):
      if normalized_sort_by == "flops":
        return op_dict.get("flops", 0)
      elif normalized_sort_by == "bytes":
        return op_dict.get("bytes_accessed", 0)
      return op_dict.get("total_self_time_ms", 0.0)

    # Group leaf operations by category
    grouped_by_cat = {}
    for op in flat_ops:
      cat = op["category"]
      if cat not in grouped_by_cat:
        grouped_by_cat[cat] = []
      grouped_by_cat[cat].append(op)

    # Build category_summary
    category_summary = []
    for cat_name, cat_ops in grouped_by_cat.items():
      cat_time = sum(o["total_self_time_ms"] for o in cat_ops)
      fraction = (
          round(cat_time / total_time_ms, 4) if total_time_ms > 0 else 0.0
      )
      total_flops = sum(o["flops"] for o in cat_ops)
      total_bytes = sum(o["bytes_accessed"] for o in cat_ops)
      sorted_cat_ops = sorted(cat_ops, key=_sort_key, reverse=True)
      top_op_name = (
          sorted_cat_ops[0]["name"].split("/")[-1] if sorted_cat_ops else ""
      )

      category_summary.append({
          "category": cat_name,
          "total_self_time_ms": round(cat_time, 4),
          "fraction_of_total_time": fraction,
          "total_flops": total_flops,
          "bytes_accessed": total_bytes,
          "op_count": len(cat_ops),
          "top_op": top_op_name,
      })

    # Sort category_summary
    if normalized_sort_by == "flops":
      category_summary.sort(key=lambda x: x["total_flops"], reverse=True)
    elif normalized_sort_by == "bytes":
      category_summary.sort(key=lambda x: x["bytes_accessed"], reverse=True)
    else:
      category_summary.sort(key=lambda x: x["total_self_time_ms"], reverse=True)

    available_categories = [c["category"] for c in category_summary]

    # Category drill-down view
    if category is not None:
      target_cat = category.strip()
      matched_cat_name = None
      for c_name in available_categories:
        if target_cat.lower() == c_name.lower():
          matched_cat_name = c_name
          break
      if not matched_cat_name:
        for c_name in available_categories:
          if target_cat.lower() in c_name.lower():
            matched_cat_name = c_name
            break

      if not matched_cat_name:
        raise FileNotFoundError(
            f"Category '{category}' not found in HLO op profile. Available"
            f" categories: {available_categories}"
        )

      cat_ops = grouped_by_cat[matched_cat_name]
      matched_cat_time = sum(o["total_self_time_ms"] for o in cat_ops)
      limit = top_n if top_n > 0 else len(cat_ops)
      sorted_ops = sorted(cat_ops, key=_sort_key, reverse=True)[:limit]
      ops_with_fraction = []
      for o in sorted_ops:
        o_copy = dict(o)
        o_copy["category_fraction"] = (
            round(o["total_self_time_ms"] / matched_cat_time, 4)
            if matched_cat_time > 0
            else 0.0
        )
        ops_with_fraction.append(o_copy)

      top_op_name = (
          ops_with_fraction[0]["name"].split("/")[-1]
          if ops_with_fraction
          else ""
      )
      result = {
          "category": matched_cat_name,
          "total_self_time_ms": round(matched_cat_time, 4),
          "fraction_of_total_time": (
              round(matched_cat_time / total_time_ms, 4)
              if total_time_ms > 0
              else 0.0
          ),
          "operations": ops_with_fraction,
          "navigation_hints": {
              "inspect_top_op_ast": (
                  "xprof get_hlo_neighborhood <trace>"
                  f" --op_name='{top_op_name}'"
              ),
              "inspect_graph": (
                  f"xprof get_graph_viewer <trace> --node_name='{top_op_name}'"
              ),
              "back_to_categories": (
                  "xprof get_hlo_op_profile <trace> --view=category"
              ),
          },
      }
      return json.dumps(result, indent=2)

    # Legacy flat view
    if normalized_view == "flat":
      flat_ops.sort(key=_sort_key, reverse=True)
      limit = top_n if top_n > 0 else len(flat_ops)
      return json.dumps(flat_ops[:limit], indent=2)

    # Category summary view
    if normalized_view in ("category", "summary"):
      top_cat = category_summary[0]["category"] if category_summary else ""
      comm_cat = next(
          (
              c["category"]
              for c in category_summary
              if any(
                  k in c["category"].lower()
                  for k in (
                      "all-gather",
                      "all-reduce",
                      "reduce-scatter",
                      "collective",
                  )
              )
          ),
          None,
      )
      hints = {
          "drill_down_into_top_category": (
              f"xprof get_hlo_op_profile <trace> --category='{top_cat}'"
          ),
          "view_grouped_ops": "xprof get_hlo_op_profile <trace> --view=grouped",
          "available_categories": available_categories,
      }
      if comm_cat:
        hints["drill_down_into_communication"] = (
            f"xprof get_hlo_op_profile <trace> --category='{comm_cat}'"
        )
      result = {
          "category_summary": category_summary,
          "navigation_hints": hints,
      }
      return json.dumps(result, indent=2)

    # Hierarchical tree view
    if normalized_view == "tree":
      target_node = root
      target_path_str = root.name if root.name else "root"
      if path:
        path_parts = [p for p in path.strip("/").split("/") if p]
        curr = root
        root_name_str = root.name if root.name else "root"
        matched_prefix = [root_name_str]
        root_aliases = {
            "root",
            "by_category",
            "by_program",
            root_name_str.lower(),
        }
        if path_parts and path_parts[0].lower() in root_aliases:
          path_parts = path_parts[1:]
        for part in path_parts:
          found_child = None
          for ch in curr.children:
            if (
                ch.name.lower() == part.lower()
                or part.lower() in ch.name.lower()
            ):
              found_child = ch
              break
          if found_child is None:
            raise FileNotFoundError(
                f"Path '{path}' not found in HLO op profile tree."
            )
          curr = found_child
          matched_prefix.append(curr.name if curr.name else "node")
        target_node = curr
        target_path_str = "/".join(matched_prefix)

      max_depth = depth if depth > 0 else 2

      def build_tree_dict(node, curr_depth, limit_depth, prefix=""):
        node_name = node.name if node.name else "root"
        node_path = f"{prefix}/{node_name}" if prefix else node_name
        self_time_ms = round(node.metrics.raw_time / 1e9, 4)
        child_dicts = []
        if curr_depth < limit_depth:
          sorted_children = sorted(
              node.children, key=lambda c: c.metrics.raw_time, reverse=True
          )
          for ch in sorted_children:
            if ch.metrics.raw_time > 0:
              child_dicts.append(
                  build_tree_dict(ch, curr_depth + 1, limit_depth, node_path)
              )
        res = {
            "name": node_name,
            "path": node_path,
            "total_self_time_ms": self_time_ms,
        }
        if node.HasField("xla"):
          if node.xla.category:
            res["category"] = node.xla.category
          if node.metrics.raw_flops > 0:
            res["flops"] = node.metrics.raw_flops
        if child_dicts:
          res["children"] = child_dicts
          res["child_count"] = len(node.children)
          res["has_children"] = True
        elif node.children:
          res["has_children"] = True
          res["child_count"] = len(node.children)
        return res

      tree_dict = build_tree_dict(target_node, 1, max_depth, "")
      child_paths = [
          f"{target_path_str}/{ch.name}"
          for ch in target_node.children
          if ch.metrics.raw_time > 0
      ]

      if child_paths:
        child_hint = (
            "xprof get_hlo_op_profile <trace> --view=tree"
            f" --path='{child_paths[0]}' --depth={max_depth}"
        )
      else:
        child_hint = "No further child paths."
      result = {
          "current_path": target_path_str,
          "depth_limit": max_depth,
          "total_self_time_ms": round(target_node.metrics.raw_time / 1e9, 4),
          "tree": tree_dict,
          "navigation_hints": {
              "navigate_deeper_into_child": child_hint,
              "switch_to_program_tree": (
                  "xprof get_hlo_op_profile <trace> --view=tree"
                  " --path='by_program' --depth=2"
              ),
              "available_child_paths": child_paths,
          },
      }
      return json.dumps(result, indent=2)

    # Grouped view (Default)
    grouped_operations = {}
    ops_per_cat_limit = min(top_n, 5) if top_n > 5 else top_n
    if top_n <= 0:
      ops_per_cat_limit = None
    for cat_summary in category_summary:
      c_name = cat_summary["category"]
      c_ops = grouped_by_cat[c_name]
      sorted_c_ops = sorted(c_ops, key=_sort_key, reverse=True)
      if ops_per_cat_limit is not None:
        sorted_c_ops = sorted_c_ops[:ops_per_cat_limit]
      cat_time = cat_summary["total_self_time_ms"]
      c_ops_formatted = []
      for op in sorted_c_ops:
        op_formatted = dict(op)
        op_formatted["category_fraction"] = (
            round(op["total_self_time_ms"] / cat_time, 4)
            if cat_time > 0
            else 0.0
        )
        c_ops_formatted.append(op_formatted)
      grouped_operations[c_name] = c_ops_formatted

    result = {
        "category_summary": category_summary,
        "grouped_operations": grouped_operations,
        "navigation_hints": {
            "drill_down_category": (
                "xprof get_hlo_op_profile <trace> --category='<category_name>'"
            ),
            "inspect_op_neighborhood": (
                "xprof get_hlo_neighborhood <trace> --op_name='<op_name>'"
            ),
            "inspect_roofline": "xprof get_roofline_model <trace>",
            "explore_tree": (
                "xprof get_hlo_op_profile <trace> --view=tree"
                " --path='by_category' --depth=2"
            ),
            "available_categories": available_categories,
        },
    }
    return json.dumps(result, indent=2)

  except (FileNotFoundError, ValueError):
    raise
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.exception(
        "Error fetching HLO op profile for session %s", session_id
    )
    raise RuntimeError(
        f"Error fetching HLO op profile for session {session_id}: {e!r}"
    ) from e


@decorators.cached(expire=86400)
def get_hosts(session_id: str) -> str:
  """Returns the list of hosts profiled in the session.

  **Use this** to see which machines participated in the profile, including
  metadata like hostnames.

  Args:
      session_id: The unique XProf session ID.

  Returns:
      A JSON-formatted dict containing a list of hosts or an error.

  Raises:
      FileNotFoundError: If no hosts are found for the session.
      RuntimeError: If fetching hosts fails.
  """
  client = xprof_client.get_client()
  try:
    hosts_data = client.get_hosts(session_id, with_metadata=True)
    if not hosts_data:
      raise FileNotFoundError(f"No hosts found for session {session_id}.")

    return json.dumps(dict(hosts=hosts_data), indent=2)
  except (FileNotFoundError, ValueError):
    raise
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.exception("Error fetching hosts for session %s", session_id)
    raise RuntimeError(
        f"Error fetching hosts for session {session_id}: {e!r}"
    ) from e


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

  Raises:
      FileNotFoundError: If no roofline model data is returned.
      ValueError: If roofline model data is malformed.
      RuntimeError: If fetching device information fails.
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
      raise FileNotFoundError(
          f"No roofline model data returned for session {session_id}."
      )

    if isinstance(data, bytes):
      data = data.decode("utf-8", errors="replace")

    try:
      roofline_data = json.loads(data)
    except Exception as e:
      raise ValueError(f"Failed to parse roofline model data: {e!r}") from e

    if not isinstance(roofline_data, list) or not roofline_data:
      raise ValueError("Unexpected roofline model data format")

    table_props = roofline_data[0].get("p", {})

    device_info = {}
    for key, value in table_props.items():
      try:
        value = float(value)
      except (ValueError, TypeError):
        pass
      device_info[key] = value

    return json.dumps(device_info, indent=2)

  except (FileNotFoundError, ValueError):
    raise
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.exception(
        "Error fetching device information for session %s", session_id
    )
    raise RuntimeError(
        f"Error fetching device information for session {session_id}: {e!r}"
    ) from e
