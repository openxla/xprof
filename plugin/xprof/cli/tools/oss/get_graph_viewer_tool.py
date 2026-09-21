"""Tool to fetch graph viewer data from XProf in OSS."""

import json
import logging
import tempfile
from typing import Any

from xprof.cli.internal.oss import hlo_tools
from xprof.cli.internal.oss import xprof_client


def get_graph_viewer(
    session_id: str = "",
    *,
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
    max_lines: int = -1,
    bypass_cache: bool = False,
) -> str:
  """Gets graph viewer data from XProf in OSS.

  Args:
    session_id: Optional XProf session ID, run name, or logdir path.
    symbol_id: Optional symbol ID (cannot be set with session_id).
    symbol_type: Optional symbol type.
    graph_type: Optional graph type (defaults to 'xla').
    module_name: Optional module name.
    output_type: Optional output type (defaults to 'short_txt'). Maps to 'type'
      in URL.
    show_metadata: Optional show metadata flag (defaults to True).
    node_name: Optional node name for type=graph.
    graph_width: Optional graph width for type=graph (defaults to 1).
    merge_fusion: Optional merge fusion flag for type=graph (defaults to False).
    tag: Optional tag (e.g., 'graph_viewer').
    tool: Optional tool name query param.
    op_profile_limit: Optional limit for op profile (e.g., 1).
    use_xplane: Optional flag to use xplane (e.g., 1).
    max_lines: Optional maximum number of lines to return (-1 for unlimited).
    bypass_cache: Whether to bypass cache and recompute metrics.

  Returns:
    The content returned by XProf.

  Raises:
    ValueError: If both or neither of session_id and symbol_id are provided.
    FileNotFoundError: If graph viewer or HLO proto data is not found.
    RuntimeError: If fetching data from XProf fails.
  """
  if session_id and symbol_id:
    raise ValueError("Cannot set both session_id and symbol_id")

  if not symbol_id and not session_id:
    raise ValueError("Either session_id or symbol_id must be provided")

  if symbol_id:
    session_id = "xsymbol"
  else:
    try:
      files = hlo_tools.get_hlo_proto_files(session_id)
      if files:
        module_name = hlo_tools.resolve_module_name(
            session_id, module_name or None
        )
    except ValueError:
      raise
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.warning("Failed to auto-discover HLO module: %s", e)

  client = xprof_client.get_client()

  # Construct parameters, ensuring strings for boolean-like flags
  options: dict[str, Any] = {
      "graph_type": graph_type,
      "type": output_type,
      "show_metadata": str(show_metadata).lower(),
  }
  if bypass_cache:
    options["bypass_cache"] = True

  if symbol_id:
    options["symbol_id"] = symbol_id
  if symbol_type:
    options["symbol_type"] = symbol_type
  if module_name:
    options["module_name"] = module_name
  if node_name:
    options["node_name"] = node_name
  if graph_width != 1:
    options["graph_width"] = str(graph_width)
  if merge_fusion:
    options["merge_fusion"] = str(merge_fusion).lower()
  if tag:
    options["tag"] = tag
  if tool:
    options["tool"] = tool
  if op_profile_limit > 0:
    options["op_profile_limit"] = str(op_profile_limit)
  if use_xplane > 0:
    options["use_xplane"] = str(use_xplane)

  params = {
      "tool_name": "graph_viewer",
      "session_id": session_id,
      "graph_viewer_options": options,
  }

  try:
    result = client.fetch(**params)  # pyrefly: ignore[missing-argument]
  except Exception as e:
    err_str = str(e)
    if (
        "Can not load hlo proto" in err_str
        or "No HLO" in err_str
        or "No such file or directory" in err_str
        or ".hlo_proto.pb" in err_str
    ):
      if module_name:
        raise FileNotFoundError(
            f"Compiled HLO module '{module_name}' not found in profile"
            " session. Run 'xprof list_hlo_modules' to inspect available"
            " modules, or export XLA_FLAGS='--xla_dump_to=<logdir>"
            " --xla_dump_hlo_as_proto' before profiling."
        ) from e
      raise FileNotFoundError(
          "No compiled HLO module proto found in profile session. To capture"
          " HLO graphs, export XLA_FLAGS='--xla_dump_to=<logdir>"
          " --xla_dump_hlo_as_proto' before profiling."
      ) from e
    if isinstance(e, ValueError):
      raise
    logging.exception("Error fetching data for graph_viewer")
    raise RuntimeError(
        f"Error fetching data for graph_viewer: {e!r}"
    ) from e

  if isinstance(result, tuple) and len(result) == 2:
    _, data = result
  else:
    data = result

  if not data:
    raise FileNotFoundError(
        "No graph_viewer data found in profile session. Ensure workload was"
        " profiled with XLA_FLAGS='--xla_dump_to=<logdir>"
        " --xla_dump_hlo_as_proto'."
    )

  if isinstance(data, bytes):
    data = data.decode("utf-8", errors="replace")

  if max_lines > 0 and isinstance(data, str):
    lines = data.splitlines()
    if len(lines) > max_lines:
      data = "\n".join(lines[:max_lines]) + (
          f"\n... [Truncated to {max_lines} of {len(lines)} lines] ..."
      )

  if isinstance(data, str):
    byte_len = len(data.encode("utf-8"))
    if byte_len > 10 * 1024 * 1024:
      ext = ".html" if output_type == "graph" else ".txt"
      with tempfile.NamedTemporaryFile(
          mode="w",
          encoding="utf-8",
          delete=False,
          prefix="xprof_spill_get_graph_viewer_",
          suffix=ext,
      ) as spill_tmp:
        spill_tmp.write(data)
        spill_path = spill_tmp.name
      return json.dumps(
          {
              "status": "SAVED_TO_FILE",
              "size_bytes": byte_len,
              "size_mib": round(byte_len / (1024 * 1024), 2),
              "file_path": spill_path,
              "message": (
                  f"Output payload ({round(byte_len / (1024 * 1024), 2)} MB)"
                  " exceeded 10 MB threshold. Saved to file to prevent"
                  " buffer overflow."
              ),
          },
          indent=2,
      )

  return data
