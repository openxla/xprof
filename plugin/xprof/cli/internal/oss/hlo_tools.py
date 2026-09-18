"""HLO-related tools for OSS XProf."""

import collections
from collections.abc import Sequence
import json
import logging
import operator
import pathlib
import re

# pylint: disable=g-import-not-at-top,g-direct-tensorflow-import
try:
  from xprof.convert import raw_to_tool_data as convert  # pyrefly: ignore[missing-import]
except ImportError:
  from xprof.convert import raw_to_tool_data as convert  # pyrefly: ignore[missing-import]

from xprof.cli.internal import decorators

from . import xprof_client

# Pre-compile regexes to improve performance.
# Computation header: "ENTRY entry {" (short_txt) or "%fused_computation (..) {"
# (long_txt). short_txt emits bare names with no leading "%"; anchoring on the
# trailing "{" reliably distinguishes headers from instruction/module lines.
_COMP_NAME_RE = re.compile(r"(?:ENTRY\s+)?%?([a-zA-Z0-9._-]+)\b.*\{\s*$")
_INSTR_RE = re.compile(r"%?([a-zA-Z0-9._-]+)\s*=(.*)")
_METADATA_RE = re.compile(r"metadata={.*?}", re.DOTALL)
_OPERAND_RE = re.compile(r"(?:^|[\s,(])%?([a-zA-Z0-9._-]+)(?=[\s,)]|$)")


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


def resolve_module_name(
    session_id: str, module_name: str | None = None
) -> str:
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
    text = (
        raw_text.decode("utf-8") if isinstance(raw_text, bytes) else raw_text
    )

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
    bypass_cache: bool = False,
) -> str:
  """Retrieves HLO module content for static analysis as JSON.

  Args:
    session_id: XProf session ID.
    path: Path to save the HLO text file.
    module_name: Name of the module.
    op_name: Name of the operation to focus on (optional).
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
          bypass_cache=bypass_cache,
      )

    text = raw_output
    try:
      parsed = json.loads(raw_output)
      if isinstance(parsed, dict) and "content" in parsed:
        text = parsed["content"]
    except Exception:  # pylint: disable=broad-exception-caught
      pass

    if path:
      path_obj = pathlib.Path(path)
      path_obj.parent.mkdir(parents=True, exist_ok=True)
      path_obj.write_text(text, encoding="utf-8")
      logging.info("Saved HLO text to %s", path)

    return json.dumps(
        {
            "status": "SUCCESS",
            "module_name": module_name,
            "op_name": op_name,
            "saved_to_path": str(path) if path else None,
            "content": text,
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


@decorators.cached(expire=86_400)
def get_hlo_neighborhood(
    session_id: str,
    instruction_name: str | None = None,
    radius: int = 2,
    module_name: str | None = None,
    *,
    op_name: str | None = None,
    print_metadata: bool = False,
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
    op_name: Alias for instruction_name for backwards compatibility.
    print_metadata: Whether to include op metadata in output.

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
    _, full_text = client.fetch(
        tool_name="graph_viewer.json",
        session_id=str(session_id),
        graph_viewer_options={
            "type": "long_txt" if print_metadata else "short_txt",
            "module_name": target_module,
        },
    )
    full_text = (
        full_text.decode("utf-8") if isinstance(full_text, bytes) else full_text
    )

    # 1. Build graph from text.
    # Map naming convention: X_by_Y.
    line_by_name = {}
    operands_by_name = {}
    users_by_name = collections.defaultdict(list)
    comp_name_by_instr_name = {}

    current_comp = "unknown"

    for line in full_text.splitlines():
      stripped = line.strip()

      # Detect computation headers in both renderers. short_txt emits bare names
      # (e.g. "ENTRY entry {") with no leading "%", so header detection must not
      # gate on "%". Instruction lines contain "=" and are handled below.
      m_comp = _COMP_NAME_RE.match(stripped)
      if m_comp and "=" not in stripped and not stripped.startswith("ROOT "):
        current_comp = m_comp.group(1)
        continue

      clean_line = stripped[5:] if stripped.startswith("ROOT ") else stripped

      m = _INSTR_RE.fullmatch(clean_line)
      if m:
        instr_name = m.group(1)
        rhs = m.group(2)
        line_by_name[instr_name] = line.strip()
        comp_name_by_instr_name[instr_name] = current_comp

        rhs_no_metadata = _METADATA_RE.sub("", rhs)
        operands = _OPERAND_RE.findall(rhs_no_metadata)

        operands_by_name[instr_name] = operands
        for op in operands:
          users_by_name[op].append(instr_name)

    if target_instr not in line_by_name:
      msg = f"Instruction '{target_instr}' not found in HLO module."
      top_instrs = list(line_by_name.keys())[:10]
      if top_instrs:
        msg += f" Suggestions: {', '.join(top_instrs)}"
      return msg

    # 2. Perform BFS.
    visited = {target_instr}
    queue = collections.deque([(target_instr, 0)])
    neighborhood = []

    while queue:
      curr_name, dist = queue.popleft()
      neighborhood.append((dist, curr_name))

      if dist < radius:
        for operand_name in operands_by_name.get(curr_name, []):
          if operand_name not in visited and operand_name in line_by_name:
            visited.add(operand_name)
            queue.append((operand_name, dist + 1))
        for user_name in users_by_name.get(curr_name, []):
          if user_name not in visited and user_name in line_by_name:
            visited.add(user_name)
            queue.append((user_name, dist + 1))

    # 3. Format the output.
    # Unpack tuples using operator.itemgetter for sorting.
    neighborhood.sort(key=operator.itemgetter(0, 1))
    output_lines = [f"Neighborhood of '{target_instr}' (radius={radius}):"]

    for dist, name in neighborhood:
      prefix = "  " * (dist + 1)
      dist_str = f"[dist={dist}]"
      comp_name = comp_name_by_instr_name.get(name, "unknown")
      context_str = f" [{comp_name}]"
      text_line = line_by_name[name]

      output_lines.append(f"{prefix}{dist_str}{context_str} {text_line}")

    return "\n".join(output_lines)

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
