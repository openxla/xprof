"""HLO-related tools for OSS XProf."""

from collections.abc import Sequence
import functools
import json
import logging
import pathlib

# pylint: disable=g-import-not-at-top,g-direct-tensorflow-import
try:
  from xprof.convert import raw_to_tool_data as convert  # pyrefly: ignore[missing-import]
except ImportError:
  from xprof.convert import raw_to_tool_data as convert  # pyrefly: ignore[missing-import]

from xprof.cli.internal import decorators
from xprof.cli.internal import hlo_graph_db

from . import xprof_client


def generate_hlo_protos(session_id: str) -> str:
  """Generates local <module_name>.hlo_proto.pb files from the XPlane traces.

  Args:
    session_id: The unique XProf session ID.

  Returns:
    A string indicating if the HLO protos were generated or already existed.
  """
  client = xprof_client.get_client()
  run_dir = client.get_run_dir(session_id)

  if any(run_dir.glob("*.hlo_proto.pb")) or any(
      run_dir.glob("**/*.hlo_proto.pb")
  ):
    return "Skipped: Already exist."

  convert.xspace_to_tool_names(client.get_xspace_paths(run_dir))
  return "Generated HLO protos."


def get_hlo_proto_files(session_id: str) -> Sequence[pathlib.Path]:
  """Finds all HLO proto files for the session.

  Args:
    session_id: The unique XProf session ID.

  Returns:
    A sequence of pathlib.Path objects pointing to the found HLO proto files.
  """
  generate_hlo_protos(session_id)
  client = xprof_client.get_client()
  run_dir = client.get_run_dir(str(session_id))
  files = list(run_dir.glob("*.hlo_proto.pb"))
  if not files:
    files = list(run_dir.glob("**/*.hlo_proto.pb"))
  return sorted(set(f for f in files if f.name != "NO_MODULE.hlo_proto.pb"))


_get_hlo_proto_files = get_hlo_proto_files


def _resolve_from_available_modules(
    available_modules: Sequence[str], module_name: str | None = None
) -> str:
  """Resolves module_name against available_modules (exact, base-name, prefix)."""
  if not available_modules:
    raise FileNotFoundError("No HLO proto found.")
  if not module_name:
    return available_modules[0]
  if module_name in available_modules:
    return module_name

  base_matches = [
      m for m in available_modules if m.split("(")[0] == module_name
  ]
  if len(base_matches) == 1:
    return base_matches[0]
  if len(base_matches) > 1:
    raise ValueError(
        f"Ambiguous module name '{module_name}'. Matches:"
        f" {', '.join(base_matches)}"
    )

  prefix_matches = [m for m in available_modules if m.startswith(module_name)]
  if len(prefix_matches) == 1:
    return prefix_matches[0]
  if len(prefix_matches) > 1:
    raise ValueError(
        f"Ambiguous module prefix '{module_name}'. Matches:"
        f" {', '.join(prefix_matches)}"
    )

  raise ValueError(
      f"Module '{module_name}' not found. Available modules:"
      f" {', '.join(available_modules)}"
  )


def resolve_module_name(session_id: str, module_name: str | None = None) -> str:
  """Resolves a short, prefix, or full module name against available HLO modules.

  Args:
    session_id: The unique XProf session ID.
    module_name: Optional module name, base name (without program ID), or
      prefix.

  Returns:
    The resolved full module name.

  Raises:
    FileNotFoundError: If no HLO proto files are found.
    ValueError: If module_name is not found or ambiguous.
  """
  files = _get_hlo_proto_files(session_id)
  if not files:
    raise FileNotFoundError("No HLO proto found.")
  available_modules = [f.name.removesuffix(".hlo_proto.pb") for f in files]
  return _resolve_from_available_modules(available_modules, module_name)


@decorators.cached(expire=86_400)
def list_hlo_modules(session_id: str) -> str:
  """Lists all HLO modules available in the XProf session.

  **Use this first** to discover which modules (e.g., JIT-ed vs. compiled) are
  available for deep-dive analysis.

  Args:
    session_id: The unique XProf session ID.

  Returns:
    A JSON-formatted string containing the list of module names.
  """
  try:
    files = _get_hlo_proto_files(session_id)
    modules = [f.name.removesuffix(".hlo_proto.pb") for f in files]
    return json.dumps(
        {
            "status": "SUCCESS",
            "count": len(modules),
            "modules": modules,
        },
        indent=2,
    )
  except (ValueError, FileNotFoundError):
    raise
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.exception(
        "Error listing HLO modules for session_id: %s", session_id
    )
    raise RuntimeError(f"Error listing HLO modules: {e!r}") from e


@decorators.cached(expire=86_400)
def get_hlo_module_content(
    session_id: str,
    fmt: str = "text",
    module_name: str | None = None,
    max_lines: int = 2000,
    *,
    print_metadata: bool = False,
) -> str:
  """Returns the full HLO module content (instruction graph) as a JSON envelope.

  **Use this** after `list_hlo_modules` to inspect the full program logic for a
  specific module. This is the primary tool for detailed code review of
  the compiled HLO.

  Args:
    session_id: The unique XProf session ID.
    fmt: Desired output format. Only 'text' (human-readable HLO) is supported.
    module_name: Optional name of the module (from list_hlo_modules). If
      omitted, defaults to the first module found.
    max_lines: Maximum number of lines to return (default 2000). Set to -1 for
      unlimited.
    print_metadata: Whether to include op metadata in output.

  Returns:
    A JSON string containing the HLO text representation and metadata.

  Raises:
    FileNotFoundError: If no HLO proto files are found.
    ValueError: If module_name is not found or fmt is unsupported.
    RuntimeError: If fetching HLO content fails.
  """
  try:
    target_module = resolve_module_name(session_id, module_name)

    if fmt != "text":
      raise ValueError(f"Unsupported format: {fmt}")

    client = xprof_client.get_client()
    _, raw_text = client.fetch(
        tool_name="graph_viewer.json",
        session_id=str(session_id),
        graph_viewer_options={
            "type": "long_txt" if print_metadata else "short_txt",
            "module_name": target_module,
        },
    )
    text = raw_text.decode("utf-8") if isinstance(raw_text, bytes) else raw_text

    lines = text.splitlines()
    is_truncated = False
    if max_lines > 0 and len(lines) > max_lines:
      is_truncated = True
      truncated_text = "\n".join(lines[:max_lines])
      truncated_text += (
          f"\n... (truncated after {max_lines} lines, total {len(lines)})."
          " Use 'max_lines=-1' to see all)"
      )
      text = truncated_text

    return json.dumps(
        {
            "status": "SUCCESS",
            "module_name": target_module,
            "line_count": len(lines),
            "truncated": is_truncated,
            "content": text,
        },
        indent=2,
    )
  except (ValueError, FileNotFoundError):
    raise
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.exception(
        "Error fetching HLO module content for session_id: %s, module_name: %s,"
        " fmt: %s, max_lines: %d",
        session_id,
        module_name,
        fmt,
        max_lines,
    )
    raise RuntimeError(f"Error fetching HLO module content: {e!r}") from e


def get_hlo_text(
    session_id: str,
    path: str | None = None,
    module_name: str | None = None,
    op_name: str | None = None,
    max_lines: int = 2000,
    bypass_cache: bool = False,
) -> str:
  """Retrieves or exports HLO module text with resolved metadata.

  Unlike `get_hlo_module_content` (which returns a raw bounded text view for
  interactive inspection), `get_hlo_text` resolves the target module name,
  writes the complete untruncated HLO module text to `path` when specified, and
  returns a structured JSON envelope with line/byte counts.

  Args:
    session_id: XProf session ID.
    path: Optional path to save the full untruncated HLO text file.
    module_name: Name of the module (auto-resolved if omitted).
    op_name: Name of the operation to focus on (optional).
    max_lines: Maximum lines to include in `content` when `path` is not set
      (default 2000; set to -1 for unlimited).
    bypass_cache: Whether to bypass cache.

  Returns:
    A JSON string containing the retrieved HLO text content and metadata.

  Raises:
    FileNotFoundError: If no HLO proto is found.
    ValueError: If the requested module is not found.
    RuntimeError: If fetching HLO module content or neighborhood fails.
  """
  try:
    if op_name:
      raw_output = get_hlo_neighborhood(
          session_id,
          op_name,
          radius=2,
          module_name=module_name,
          bypass_cache=bypass_cache,
      )
    else:
      raw_output = get_hlo_module_content(
          session_id,
          module_name=module_name,
          max_lines=-1,
          bypass_cache=bypass_cache,
      )

    text = raw_output
    resolved_module = module_name
    try:
      parsed = json.loads(raw_output)
      if isinstance(parsed, dict):
        if "content" in parsed:
          text = parsed["content"]
        if parsed.get("module_name"):
          resolved_module = parsed["module_name"]
    except Exception:  # pylint: disable=broad-exception-caught
      pass

    if not resolved_module:
      try:
        resolved_module = resolve_module_name(session_id, module_name)
      except Exception:  # pylint: disable=broad-exception-caught
        resolved_module = module_name

    lines = text.splitlines()
    line_count = len(lines)
    byte_count = len(text.encode("utf-8"))
    is_truncated = False
    returned_line_count = line_count

    if path:
      path_obj = pathlib.Path(path)
      path_obj.parent.mkdir(parents=True, exist_ok=True)
      path_obj.write_text(text, encoding="utf-8")
      logging.info("Saved HLO text to %s", path)
      if line_count > 50:
        is_truncated = True
        returned_line_count = 50
        preview_lines = lines[:50]
        content_field = (
            "\n".join(preview_lines)
            + f"\n... [Saved full {line_count} lines ({byte_count} bytes) to"
            f" {path}] ..."
        )
      else:
        content_field = text
    elif max_lines > 0 and line_count > max_lines:
      is_truncated = True
      returned_line_count = max_lines
      content_field = (
          "\n".join(lines[:max_lines])
          + f"\n... [Truncated to {max_lines} of {line_count} lines"
          f" ({byte_count} bytes); pass --max_lines=-1 or --path=<file> for"
          " full HLO] ..."
      )
    else:
      content_field = text

    return json.dumps(
        {
            "status": "SUCCESS",
            "module_name": resolved_module,
            "op_name": op_name,
            "saved_to_path": str(path) if path else None,
            "truncated": is_truncated,
            "line_count": line_count,
            "returned_line_count": returned_line_count,
            "byte_count": byte_count,
            "content": content_field,
        },
        indent=2,
    )
  except (ValueError, FileNotFoundError):
    raise
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.exception(
        "Error in get_hlo_text for session_id: %s, path: %s, module_name: %s,"
        " op_name: %s",
        session_id,
        path,
        module_name,
        op_name,
    )
    raise RuntimeError("Error retrieving HLO text") from e


@functools.lru_cache(maxsize=64)
def _get_cached_oss_graph_db(
    target_module: str, raw_text: str | bytes
) -> hlo_graph_db.HloGraphDb:
  """Builds and caches HloGraphDb from raw HLO text or bytes."""
  full_text = (
      raw_text.decode("utf-8") if isinstance(raw_text, bytes) else raw_text
  )
  return hlo_graph_db.HloGraphDb.from_hlo_text(
      full_text, module_name=target_module
  )


@decorators.cached(expire=86_400)
def get_hlo_neighborhood(
    session_id: str,
    instruction_name: str | None = None,
    radius: int = 2,
    module_name: str | None = None,
    fmt: str = "text",
    *,
    op_name: str | None = None,
    print_metadata: bool = False,
    direction: str = "both",
    follow_calls: bool = False,
    detect_fusion_blockers: bool = False,
    opcode_filter: str | None = None,
) -> str:
  """Returns the neighborhood of a specific HLO instruction (BFS traversal).

  **Crucial for debugging regressions.** Use this to root-cause why a specific
  op is slow by inspecting its immediate producers and consumers. Often, a slow
  op is caused by a `bitcast` or `copy` in its neighborhood that blocks fusion.

  Args:
    session_id: The unique XProf session ID.
    instruction_name: The name of the instruction to center the neighborhood on
      (e.g., %fused_computation.1).
    radius: How many steps to traverse up (operands) and down (users). Default
      is 2.
    module_name: Optional name of the module to search in.
    fmt: Desired output format ('text' or 'markdown').
    op_name: Alias for instruction_name for backwards compatibility.
    print_metadata: Whether to include op metadata in output.
    direction: Traversal direction ('both', 'operands', or 'users').
    follow_calls: Whether to traverse into called sub-computations.
    detect_fusion_blockers: Whether to annotate bitcast/copy/reshape/convert ops
      with [FUSION_BLOCKER].
    opcode_filter: Optional substring filter on neighbor opcodes.

  Returns:
    A textual description of the neighborhood with high-fidelity formatting.
  """
  if instruction_name is not None and op_name is not None:
    if instruction_name != op_name:
      raise ValueError(
          f"Conflicting arguments: instruction_name='{instruction_name}'"
          f" and op_name='{op_name}' cannot both be specified with"
          " different values."
      )
  target_instr = instruction_name or op_name
  if not target_instr:
    raise ValueError(
        "Either instruction_name or op_name must be provided to"
        " get_hlo_neighborhood."
    )
  if target_instr.startswith("%"):
    target_instr = target_instr[1:]

  try:
    try:
      target_module = resolve_module_name(session_id, module_name)
    except FileNotFoundError:
      return "No HLO proto found."
    except ValueError as e:
      return str(e)

    # Fetch full text from native graph_viewer.
    client = xprof_client.get_client()
    _, raw_text = client.fetch(
        tool_name="graph_viewer.json",
        session_id=str(session_id),
        graph_viewer_options={
            "type": "long_txt" if print_metadata else "short_txt",
            "module_name": target_module,
        },
    )
    db = _get_cached_oss_graph_db(target_module, raw_text)
    return db.get_neighborhood(
        target_instr,
        radius=radius,
        fmt=fmt,
        print_metadata=print_metadata,
        direction=direction,
        follow_calls=follow_calls,
        detect_fusion_blockers=detect_fusion_blockers,
        opcode_filter=opcode_filter,
        oss_style_suggestions=True,
    )

  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.exception(
        "Error analyzing neighborhood for session_id: %s, instruction_name: %s,"
        " radius: %d, module_name: %s",
        session_id,
        instruction_name,
        radius,
        module_name,
    )
    return f"Error analyzing neighborhood: {e!r}"


@decorators.cached(expire=86_400)
def query_hlo_graph(
    session_id: str,
    mode: str = "summary",
    module_name: str | None = None,
    *,
    opcode: str | None = None,
    category: str | None = None,
    comp_name: str | None = None,
    name_pattern: str | None = None,
    src_op: str | None = None,
    dst_op: str | None = None,
    sql: str | None = None,
    limit: int = 50,
) -> str:
  """Executes relational, structural, or SQL queries against the HLO Graph DB.

  Args:
    session_id: The unique XProf session ID.
    mode: Query mode ('summary', 'opcode_stats', 'fusion_blockers',
      'shortest_path', 'call_tree', 'sql', 'search').
    module_name: Optional name of the HLO module to query.
    opcode: Optional opcode filter for 'search'.
    category: Optional category filter for 'search'.
    comp_name: Optional computation filter.
    name_pattern: Optional substring or SQL LIKE pattern for 'search'.
    src_op: Source instruction name for mode='shortest_path'.
    dst_op: Destination instruction name for mode='shortest_path'.
    sql: Custom read-only SQL SELECT query when mode='sql'.
    limit: Maximum number of rows to return (default 50).

  Returns:
    A JSON-formatted string containing the structured query results.
  """
  try:
    target_module = resolve_module_name(session_id, module_name)
    client = xprof_client.get_client()
    _, raw_text = client.fetch(
        tool_name="graph_viewer.json",
        session_id=str(session_id),
        graph_viewer_options={
            "type": "short_txt",
            "module_name": target_module,
        },
    )
    db = _get_cached_oss_graph_db(target_module, raw_text)
    result = db.query(
        mode=mode,
        opcode=opcode,
        category=category,
        comp_name=comp_name,
        name_pattern=name_pattern,
        src_op=src_op,
        dst_op=dst_op,
        sql=sql,
        limit=limit,
    )
    return json.dumps(result, indent=2)
  except (ValueError, FileNotFoundError):
    raise
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.exception("Error querying HLO graph in session %s", session_id)
    raise RuntimeError(f"Error querying HLO graph: {e!r}") from e


def get_hlo_stats(session_id: str) -> str:
  """Fetches HLO stats containing HLO text expressions.

  Args:
    session_id: The unique XProf session ID.

  Returns:
    A string containing the HLO stats data.
  """
  client = xprof_client.get_client()
  _, data = client.fetch(tool_name="hlo_stats", session_id=str(session_id))
  if isinstance(data, bytes):
    return data.decode("utf-8", errors="ignore")
  return data
