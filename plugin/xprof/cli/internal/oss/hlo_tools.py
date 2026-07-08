"""HLO-related tools for OSS XProf."""

import collections
from collections.abc import Sequence
import logging
import operator
import pathlib
import re

from xprof.convert import raw_to_tool_data as convert  # pytype: disable=import-error

from xprof.cli.internal import decorators

from . import xprof_client

# Pre-compile regexes to improve performance.
_COMP_NAME_RE = re.compile(r"%([a-zA-Z0-9._-]+)\s*\(")
_INSTR_RE = re.compile(r"%([a-zA-Z0-9._-]+)\s*=(.*)")
_METADATA_RE = re.compile(r"metadata={.*?}", re.DOTALL)
_OPERAND_RE = re.compile(r"%([a-zA-Z0-9._-]+)")


def generate_hlo_protos(session_id: str) -> str:
  """Generates local <module_name>.hlo_proto.pb files from the XPlane traces.

  Args:
    session_id: The unique XProf session ID.

  Returns:
    A string indicating if the HLO protos were generated or already existed.
  """
  client = xprof_client.get_client()
  run_dir = client.get_run_dir(session_id)

  if any(run_dir.glob("*.hlo_proto.pb")):
    return "Skipped: Already exist."

  convert.xspace_to_tool_names(client.get_xspace_paths(run_dir))
  return "Generated HLO protos."


def _get_hlo_proto_files(session_id: str) -> Sequence[pathlib.Path]:
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
  return sorted(f for f in files if f.name != "NO_MODULE.hlo_proto.pb")


@decorators.cached(expire=86_400)
def list_hlo_modules(session_id: str) -> str:
  """Lists all HLO modules available in the XProf session.

  **Use this first** to discover which modules (e.g., JIT-ed vs. compiled) are
  available for deep-dive analysis.

  Args:
    session_id: The unique XProf session ID.

  Returns:
    A human-readable list of module names.
  """
  try:
    files = _get_hlo_proto_files(session_id)
    if not files:
      return (
          "No HLO modules found. Ensure you have run a compilation or imported"
          " traces with HLO."
      )

    lines = [f"Found {len(files)} HLO modules:"]
    for i, f in enumerate(files):
      # E.g., module_name.hlo_proto.pb -> module_name.
      module_name = f.name.removesuffix(".hlo_proto.pb")
      lines.append(f"{i}. {module_name}")
    return "\n".join(lines)
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.exception(
        "Error listing HLO modules for session_id: %s", session_id
    )
    return f"Error listing HLO modules: {e!r}"


@decorators.cached(expire=86_400)
def get_hlo_module_content(
    session_id: str,
    fmt: str = "text",
    module_name: str | None = None,
    max_lines: int = 2000,
    *,
    print_metadata: bool = False,
) -> str:
  """Returns the full HLO module content (instruction graph) as text.

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
    The full HLO text representation for the selected module.
  """
  try:
    files = _get_hlo_proto_files(session_id)
    if not files:
      return "No HLO proto found."

    available_modules = [f.name.removesuffix(".hlo_proto.pb") for f in files]

    if module_name:
      if module_name not in available_modules:
        return (
            f"Module '{module_name}' not found. Available:"
            f" {', '.join(available_modules)}"
        )
      target_module = module_name
    else:
      target_module = available_modules[0]

    if fmt == "text":
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

      if max_lines > 0:
        lines = text.splitlines()
        if len(lines) > max_lines:
          truncated_text = "\n".join(lines[:max_lines])
          truncated_text += (
              f"\n... (truncated after {max_lines} lines, total {len(lines)})."
              " Use 'max_lines=-1' to see all)"
          )
          return truncated_text
      return text
    return f"Unsupported format: {fmt}"
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.exception(
        "Error fetching HLO module content for session_id: %s, module_name: %s,"
        " fmt: %s, max_lines: %d",
        session_id,
        module_name,
        fmt,
        max_lines,
    )
    return f"Error fetching HLO module content: {e!r}"


def get_hlo_text(
    session_id: str,
    path: str | None = None,
    module_name: str | None = None,
    op_name: str | None = None,
) -> str:
  """Retrieves HLO module content for static analysis.

  Args:
    session_id: XProf session ID.
    path: Path to save the HLO text file.
    module_name: Name of the module.
    op_name: Name of the operation to focus on (optional).

  Returns:
    The retrieved HLO text content.

  Raises:
    RuntimeError: If fetching HLO module content or neighborhood fails.
  """
  try:
    if op_name:
      text = get_hlo_neighborhood(
          session_id, op_name, radius=2, module_name=module_name
      )
    else:
      text = get_hlo_module_content(session_id, module_name=module_name)

    if path:
      path_obj = pathlib.Path(path)
      path_obj.parent.mkdir(parents=True, exist_ok=True)
      path_obj.write_text(text, encoding="utf-8")
      logging.info("Saved HLO text to %s", path)

    return text
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
    instruction_name: str,
    radius: int = 2,
    module_name: str | None = None,
    *,
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
    print_metadata: Whether to include op metadata in output.

  Returns:
    A textual description of the neighborhood with high-fidelity formatting.
  """
  try:
    if instruction_name.startswith("%"):
      instruction_name = instruction_name[1:]

    files = _get_hlo_proto_files(session_id)
    if not files:
      return "No HLO proto found."

    available_modules = [f.name.removesuffix(".hlo_proto.pb") for f in files]
    if module_name:
      if module_name not in available_modules:
        return (
            f"Module '{module_name}' not found. Available:"
            f" {', '.join(available_modules)}"
        )
      target_module = module_name
    else:
      target_module = available_modules[0]

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
      if line.startswith("%"):
        m = _COMP_NAME_RE.match(line)
        if m:
          current_comp = m.group(1)
        continue

      stripped = line.strip()
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

    if instruction_name not in line_by_name:
      msg = f"Instruction '{instruction_name}' not found in HLO module."
      top_instrs = list(line_by_name.keys())[:10]
      if top_instrs:
        msg += f" Suggestions: {', '.join(top_instrs)}"
      return msg

    # 2. Perform BFS.
    visited = {instruction_name}
    queue = collections.deque([(instruction_name, 0)])
    neighborhood = []

    while queue:
      curr_name, dist = queue.popleft()
      neighborhood.append((dist, curr_name))

      if dist < radius:
        for op_name in operands_by_name.get(curr_name, []):
          if op_name not in visited and op_name in line_by_name:
            visited.add(op_name)
            queue.append((op_name, dist + 1))
        for user_name in users_by_name.get(curr_name, []):
          if user_name not in visited and user_name in line_by_name:
            visited.add(user_name)
            queue.append((user_name, dist + 1))

    # 3. Format the output.
    # Unpack tuples using operator.itemgetter for sorting.
    neighborhood.sort(key=operator.itemgetter(0, 1))
    output_lines = [f"Neighborhood of '{instruction_name}' (radius={radius}):"]

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
