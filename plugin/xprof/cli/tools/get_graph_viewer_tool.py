"""Tool to fetch graph viewer data from XProf."""

import json
import logging
import traceback

from xprof.cli.internal.oss import xprof_client


def get_graph_viewer(
    *,
    session_id: str = "",
    symbol_id: str = "",
    symbol_type: str = "",
    graph_type: str = "xla",
    module_name: str = "",
    output_type: str = "short_txt",
    show_metadata: bool = True,
    node_name: str = "",
    graph_width: int = 1,
    merge_fusion: bool = False,
    tag: str = "",
    tool: str = "",
    op_profile_limit: int = 0,
    use_xplane: int = 0,
) -> str:
  """Gets graph viewer data from XProf.

  Args:
      session_id: Optional XProf session ID (cannot be set with symbol_id).
      symbol_id: Optional symbol ID (cannot be set with session_id).
      symbol_type: Optional symbol type.
      graph_type: Optional graph type (defaults to 'xla').
      module_name: Optional module name.
      output_type: Optional output type (defaults to 'short_txt'). Maps to
        'type' in URL.
      show_metadata: Optional show metadata flag (defaults to True).
      node_name: Optional node name for type=graph.
      graph_width: Optional graph width for type=graph (defaults to 1).
      merge_fusion: Optional merge fusion flag for type=graph (defaults to
        False).
      tag: Optional tag (e.g., 'graph_viewer').
      tool: Optional tool name query param.
      op_profile_limit: Optional limit for op profile (e.g., 1).
      use_xplane: Optional flag to use xplane (e.g., 1).

  Returns:
      The content returned by XProf.
  """
  if session_id and symbol_id:
    return json.dumps(
        dict(error="Cannot set both session_id and symbol_id"), indent=2
    )

  if not symbol_id and not session_id:
    return json.dumps(
        dict(error="Either session_id or symbol_id must be provided"), indent=2
    )

  if symbol_id:
    session_id = "xsymbol"
  # copybara:uncomment_begin(external)
  #   else:
  #     try:
  #       hlo_tools.generate_hlo_protos(session_id)
  #     except Exception as e:  # pylint: disable=broad-exception-caught
  #       logging.warning("Failed to generate HLO protos: %s", e)
  #   # copybara:uncomment_end

  client = xprof_client.get_client()

  # Construct parameters, ensuring strings for boolean-like flags
  params = {
      "graph_type": graph_type,
      "type": output_type,
      "show_metadata": str(show_metadata).lower(),
  }

  if symbol_id:
    params["symbol_id"] = symbol_id
  if symbol_type:
    params["symbol_type"] = symbol_type
  if module_name:
    params["module_name"] = module_name
  if node_name:
    params["node_name"] = node_name
  if graph_width != 1:
    params["graph_width"] = str(graph_width)
  if merge_fusion:
    params["merge_fusion"] = str(merge_fusion).lower()
  if tag:
    params["tag"] = tag
  if tool:
    params["tool"] = tool
  if op_profile_limit > 0:
    params["op_profile_limit"] = str(op_profile_limit)
  if use_xplane > 0:
    params["use_xplane"] = str(use_xplane)

  params = {
      "tool_name": "graph_viewer.json",
      "session_id": session_id,
      "graph_viewer_options": params,
  }

  try:
    result = client.fetch(**params)
    if isinstance(result, tuple) and len(result) == 2:
      _, data = result
    else:
      data = result

    if not data:
      return json.dumps(
          dict(error="No data returned for graph_viewer.json"), indent=2
      )

    if isinstance(data, bytes):
      data = data.decode("utf-8", errors="replace")

    return data
  except Exception as e:  # pylint: disable=broad-except
    logging.exception("Error fetching data for graph_viewer.json")
    return json.dumps(
        dict(
            error=f"Error fetching data for graph_viewer.json: {e!r}",
            traceback=traceback.format_exc(),
        ),
        indent=2,
    )
