"""Tool to fetch peak memory allocations data from XProf, ordered by HBM usage."""

import collections
from collections.abc import Mapping, Sequence
import dataclasses
import heapq
import json
import logging
import re
import traceback
from typing import Any, Literal
from xprof.cli.internal import decorators
from xprof.cli.internal.oss import xprof_client

_INSTRUCTION_NAME_REGEX = re.compile(r"([a-zA-Z_][a-zA-Z0-9_]*)\.\d+")


@dataclasses.dataclass
class BufferAllocation:
  """A buffer allocation.

  Attributes:
      instruction: The name of the HLO instruction.
      size_mib: The size of the allocation in MiB.
  """

  instruction: str
  size_mib: float


@dataclasses.dataclass
class ModulePeakAllocations:
  """Peak memory allocations for a module.

  Attributes:
      module_name: The name of the HLO module.
      total_hbm_mib: The total peak HBM usage in MiB.
      top_buffers: A sequence of top buffer allocations.
  """

  module_name: str
  total_hbm_mib: float
  top_buffers: Sequence[BufferAllocation]


@dataclasses.dataclass
class LargestBuffer:
  """Largest buffer allocation info.

  Attributes:
      module: The name of the HLO module.
      instruction: The name of the HLO instruction.
      size_mib: The size of the allocation in MiB.
  """

  module: str
  instruction: str
  size_mib: float


@dataclasses.dataclass
class ModuleSummary:
  """Summary of a module's memory usage.

  Attributes:
      module_name: The name of the HLO module.
      total_hbm_mib: The total peak HBM usage in MiB.
  """

  module_name: str
  total_hbm_mib: float


@dataclasses.dataclass
class SummaryData:
  """Summary metrics across all modules.

  Attributes:
      total_modules: Total number of modules.
      top_modules: Summary of top modules by usage.
  """

  total_modules: int
  top_modules: Sequence[ModuleSummary]


def _format_markdown(
    modules_data: Sequence[ModulePeakAllocations],
    summary_data: SummaryData | None = None,
    aggregation_applied: bool = True,
) -> str:
  """Formats modules data as a markdown table.

  Args:
      modules_data: A sequence of ModulePeakAllocations objects.
      summary_data: Optional SummaryData object with summary metrics.
      aggregation_applied: Whether aggregation was applied to the data.

  Returns:
      A string containing the formatted markdown table.
  """
  lines = []
  lines.append("# Peak Memory Allocations by Module")
  lines.append("")
  if aggregation_applied:
    lines.append("> [!NOTE]")
    lines.append("> **Aggregation Logic:**")
    lines.append(
        "> - Buffers with similar names (e.g., `name.1`, `name.2`) and"
        " identical sizes are aggregated into `name.*`."
    )
    lines.append(
        "> - Buffers smaller than the threshold are aggregated into 'Others'."
    )
    lines.append("")
  if summary_data:
    lines.append("## Session Summary")
    lines.append(f"- Total Modules: {summary_data.total_modules}")
    if summary_data.top_modules:
      lines.append("")
      lines.append("| Module | Total HBM (MiB) |")
      lines.append("| :--- | ---: |")
      for mod in summary_data.top_modules:
        lines.append(f"| `{mod.module_name}` | {mod.total_hbm_mib:.2f} |")
    lines.append("")

  for mod in modules_data:
    lines.append(f"## Module: `{mod.module_name}`")
    lines.append(f"Total HBM: {mod.total_hbm_mib:.2f} MiB")
    lines.append("")
    lines.append("| Instruction | Size (MiB) |")
    lines.append("| :--- | ---: |")
    for buf in mod.top_buffers:
      lines.append(f"| `{buf.instruction}` | {buf.size_mib:.2f} |")
    lines.append("")
  return "\n".join(lines)


def _calculate_summary(
    modules_data: Sequence[ModulePeakAllocations],
    top_modules_data: Sequence[ModulePeakAllocations],
) -> SummaryData:
  """Calculates summary metrics across all modules.

  Args:
      modules_data: A sequence of ModulePeakAllocations objects (all modules).
      top_modules_data: A sequence of ModulePeakAllocations objects (top
        modules).

  Returns:
      A SummaryData object containing summary metrics.
  """
  return SummaryData(
      total_modules=len(modules_data),
      top_modules=[
          ModuleSummary(
              module_name=m.module_name, total_hbm_mib=m.total_hbm_mib
          )
          for m in top_modules_data
      ],
  )


def _parse_and_aggregate_buffers(
    parsed_mod_data: Mapping[str, Any],
    min_size_mib: float,
    aggregate_instructions: bool = True,
) -> Sequence[BufferAllocation]:
  """Parses and aggregates buffer allocations for a module.

  Args:
      parsed_mod_data: A dictionary containing the parsed JSON data for a
        module. It is expected to have 'bufferAssignment' or 'maxHeap' keys.
      min_size_mib: Buffers smaller than this threshold (in MiB) will be
        aggregated into an "Others" category.
      aggregate_instructions: Whether to aggregate instructions.

  Returns:
      A sequence of BufferAllocation objects, ordered by size in descending
      order.
  """
  top_buffers = []
  others_size = 0.0
  buffer_assignment = parsed_mod_data.get("bufferAssignment", {})
  logical_buffers = buffer_assignment.get("logicalBuffers", [])

  if logical_buffers:
    for buf in logical_buffers:
      size_bytes = int(buf.get("size", 0))
      size_mib = size_bytes / (1024 * 1024)
      defined_at = buf.get("definedAt", {})
      instruction = defined_at.get("instructionName", "unknown")

      if aggregate_instructions and size_mib < min_size_mib:
        others_size += size_mib
        continue

      top_buffers.append(
          BufferAllocation(instruction=instruction, size_mib=size_mib)
      )
  else:
    # Fallback to maxHeap if bufferAssignment is missing
    max_heap = parsed_mod_data.get("maxHeap", [])
    if isinstance(max_heap, list):
      for buf in max_heap:
        size_mib = float(buf.get("logicalBufferSizeMib", 0))
        instruction = buf.get("instructionName", "unknown")

        if aggregate_instructions and size_mib < min_size_mib:
          others_size += size_mib
          continue

        top_buffers.append(
            BufferAllocation(instruction=instruction, size_mib=size_mib)
        )

  if not aggregate_instructions:
    return sorted(top_buffers, key=lambda x: x.size_mib, reverse=True)

  # Aggregate similar buffers (e.g., param.1, param.2 -> param.*)
  instructions_by_base_name_and_size = collections.defaultdict(list)
  for buf in top_buffers:
    inst = buf.instruction
    size = buf.size_mib
    match = _INSTRUCTION_NAME_REGEX.fullmatch(inst)
    if match:
      base_name = match.group(1) + ".*"
    else:
      base_name = inst
    instructions_by_base_name_and_size[(base_name, size)].append(inst)

  aggregated_buffers = []
  for (base_name, size), insts in instructions_by_base_name_and_size.items():
    count = len(insts)
    if count > 1:
      aggregated_buffers.append(
          BufferAllocation(
              instruction=(
                  f"{base_name} ({count} occurrences of size {size:g} MiB)"
              ),
              size_mib=size * count,
          )
      )
    else:
      aggregated_buffers.append(
          BufferAllocation(instruction=insts[0], size_mib=size)
      )

  if others_size > 0:
    aggregated_buffers.append(
        BufferAllocation(
            instruction=f"Others (< {min_size_mib} MiB)",
            size_mib=others_size,
        )
    )

  return sorted(aggregated_buffers, key=lambda x: x.size_mib, reverse=True)


def _get_module_names(
    client: xprof_client.CachedXprofClient, session_id: str
) -> Sequence[str]:
  """Fetches the list of HLO module names for a session.

  Args:
      client: The CachedXprofClient instance to use for fetching data.
      session_id: The unique XProf session ID.

  Returns:
      A sequence of HLO module names.

  Raises:
      ValueError: If no memory viewer data is returned or if no HLO modules
        are found in the data.
  """
  result = client.fetch(
      tool_name="memory_viewer.json",
      session_id=session_id,
      format="json",
  )
  if isinstance(result, tuple) and len(result) == 2:
    _, data = result
  else:
    data = result

  if not data:
    raise ValueError("No memory viewer data returned for the session")

  if isinstance(data, bytes):
    data = data.decode("utf-8", errors="replace")

  module_names = [m.strip() for m in data.split(",") if m.strip()]
  if not module_names:
    raise ValueError(
        "No HLO modules found in memory viewer data for the session"
    )

  return module_names


def _fetch_modules_data(
    client: xprof_client.CachedXprofClient,
    session_id: str,
    module_names: Sequence[str],
    min_size_mib: float,
    aggregate_instructions: bool = True,
) -> Sequence[ModulePeakAllocations]:
  """Fetches and processes peak allocation data for all modules.

  Args:
      client: The CachedXprofClient instance to use for fetching data.
      session_id: The unique XProf session ID.
      module_names: A sequence of HLO module names to fetch data for.
      min_size_mib: Buffers smaller than this threshold (in MiB) will be
        aggregated into an "Others" category.
      aggregate_instructions: Whether to aggregate instructions.

  Returns:
      A sequence of ModulePeakAllocations objects containing processed peak
      allocation data for each module.
  """
  modules_data = []
  for module in module_names:
    module_result = client.fetch(
        tool_name="memory_viewer.json",
        session_id=session_id,
        format="json",
        module_name=module,
    )
    if isinstance(module_result, tuple) and len(module_result) == 2:
      _, mod_data = module_result
    else:
      mod_data = module_result

    if not mod_data:
      logging.warning(
          "No data returned for module %s in session %s", module, session_id
      )
      continue

    if isinstance(mod_data, bytes):
      mod_data = mod_data.decode("utf-8", errors="replace")

    try:
      parsed_mod_data = json.loads(mod_data)
    except json.JSONDecodeError:
      logging.exception(
          "Failed to parse JSON for module %s in session %s",
          module,
          session_id,
      )
      continue

    total_hbm_mib = parsed_mod_data.get("totalBufferAllocationMib", 0)

    aggregated_buffers = _parse_and_aggregate_buffers(
        parsed_mod_data, min_size_mib, aggregate_instructions
    )

    modules_data.append(
        ModulePeakAllocations(
            module_name=module,
            total_hbm_mib=total_hbm_mib,
            top_buffers=aggregated_buffers,
        )
    )
  return modules_data


def _format_error(session_id: str, error: Exception, output_format: str) -> str:
  """Formats an error message in the requested format."""
  formatted_error = "".join(
      traceback.format_exception_only(type(error), error)
  ).strip()
  error_msg = (
      f"Failed to get peak allocations for session {session_id}: "
      f"{formatted_error}"
  )
  if output_format == "markdown":
    return f"# Error\n{error_msg}\n"
  else:
    return json.dumps({"error": error_msg}, indent=2)


@decorators.cached(expire=86400)
def get_peak_allocations(
    session_id: str,
    *,
    limit: int = 10,
    min_size_mib: float = 1.0,
    output_format: Literal["json", "markdown"] = "json",
    include_summary: bool = True,
    aggregate_instructions: bool = True,
) -> str:
  """Fetches HLO modules and their peak memory allocations for a session.

  Retrieves a list of HLO modules ordered by peak HBM usage and includes
  the top buffers contributing to that usage for each module.

  Args:
      session_id: The unique XProf session ID.
      limit: The maximum number of modules to return (default: 10). Use 0 for no
        limit.
      min_size_mib: Buffers smaller than this threshold (in MiB) will be
        aggregated into an "Others" category (default: 1.0).
      output_format: The output format, either 'json' or 'markdown' (default:
        'json').
      include_summary: Whether to include a high-level summary at the top
        (default: True).
      aggregate_instructions: Whether to aggregate instructions (default: True).

  Returns:
      HLO modules and their top buffers in the requested format.
  """
  client = xprof_client.get_client()

  try:
    module_names = _get_module_names(client, session_id)
  except ValueError as e:
    return _format_error(session_id, e, output_format)

  try:
    modules_data = _fetch_modules_data(
        client, session_id, module_names, min_size_mib, aggregate_instructions
    )
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.exception(
        "Failed to get peak allocations for session %s", session_id
    )
    return _format_error(session_id, e, output_format)

  sorted_modules_data = (
      heapq.nlargest(limit, modules_data, key=lambda x: x.total_hbm_mib)
      if limit > 0
      else sorted(modules_data, key=lambda x: x.total_hbm_mib, reverse=True)
  )

  summary_data = (
      _calculate_summary(modules_data, sorted_modules_data)
      if include_summary
      else None
  )

  if output_format == "markdown":
    return _format_markdown(
        sorted_modules_data,
        summary_data,
        aggregation_applied=aggregate_instructions,
    )

  if include_summary:
    return json.dumps(
        {
            "summary": dataclasses.asdict(summary_data),
            "modules": [dataclasses.asdict(m) for m in sorted_modules_data],
        },
        indent=2,
    )

  return json.dumps(
      [dataclasses.asdict(m) for m in sorted_modules_data], indent=2
  )
