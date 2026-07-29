"""OSS hermetic backend to compute kernel stats and step times from local XPlane traces."""

import collections
import json
import logging
import re
import statistics as stats_mod
from typing import Any, Dict, List, Literal

from xprof.cli.internal.oss import xplane_tools

# Standard device line inclusion filters for active hardware compute events.
# These lines represent actual device execution (compute + transfers).
_DEVICE_LINE_INCLUDE = (
    "XLA OPS",
    "PALLAS",
    "LLO OPS",
)

# Standard device line exclusion filters for non-computational metadata.
_DEVICE_LINE_EXCLUDE = (
    "COUNTER",
    "MODULES",
    "OVERLAY",
    "SYNC FLAG",
    "SENSOR",
)


def compute_disjoint_interval_union_ns(intervals: list[tuple[int, int]]) -> int:
  """Computes total duration from overlapping intervals using sweep-line merge.

  This prevents double-counting concurrent operations across MXU compute and
  DMA/ICI transfer planes, which inflates active hardware compute time by
  up to 47.3% on modern accelerators (b/537646336).

  Args:
    intervals: A list of (start_ns, end_ns) tuples.

  Returns:
    The total non-overlapping duration in nanoseconds.
  """
  if not intervals:
    return 0
  sorted_intervals = sorted(intervals, key=lambda x: x[0])
  merged = []
  curr_start, curr_end = sorted_intervals[0]
  for start, end in sorted_intervals[1:]:
    if start <= curr_end:
      curr_end = max(curr_end, end)
    else:
      merged.append((curr_start, curr_end))
      curr_start, curr_end = start, end
  merged.append((curr_start, curr_end))
  return sum(end - start for start, end in merged)


def format_markdown_table(
    df_records: List[Dict[str, Any]], kernel_name: str | None
) -> str:
  """Formats kernel stats as a markdown table."""
  lines = []
  if kernel_name:
    lines.append(f"# Kernel Stats for `{kernel_name}`")
  else:
    lines.append("# Top Kernels by Duration")
  lines.append("")
  lines.append(
      "| Kernel | Total Duration (us) | Execution Count | Avg Duration (us) |"
  )
  lines.append("| :--- | ---: | ---: | ---: |")
  for row in df_records:
    total_dur = float(row.get("total_duration_us", 0.0) or 0.0)
    avg_dur = float(row.get("avg_duration_us", 0.0) or 0.0)
    lines.append(
        f"| `{row.get('kernel_name', 'unknown')}` | {total_dur:.2f} |"
        f" {row.get('execution_count', 0)} | {avg_dur:.2f} |"
    )
  return "\n".join(lines)


def _matches(pattern: str, text: str) -> bool:
  """Matches text against a regex pattern or literal substring."""
  try:
    if re.search(pattern, text):
      return True
  except re.error:
    pass
  return pattern in text


def get_kernel_stats(
    source: Any,
    *,
    kernel_name: str | None = None,
    limit: int = 10,
    output_format: Literal["json", "markdown", "dict"] = "json",
    include_summary: bool = False,
    device_to_use: str | None = "TPU:0",  # pylint: disable=unused-argument
    trace_matchers: tuple[str, ...] | None = None,
) -> Any:
  """Computes performance metrics for operations from local XPlanes in OSS.

  Supports polymorphic inputs: session IDs, file paths, bytes, or in-memory
  ProfileData/XSpace objects. When include_summary=True (or output_format="dict"
  with include_summary=True), returns an enriched dict containing ground-truth
  device durations via Disjoint Interval Union, step durations, and stats.

  Args:
    source: XProf session ID, local file/directory path, serialized XSpace
      bytes, in-memory ProfileData/XSpace object, or pre-computed records.
    kernel_name: Optional specific tf_op_name / kernel name to filter by.
    limit: Number of top kernels to return when kernel_name is not provided.
    output_format: Output format - 'json' (JSON string), 'markdown' (markdown
      table string), or 'dict' (raw Python dict/list).
    include_summary: If True, computes and returns ground-truth timing via
      Disjoint Interval Union alongside per-kernel records.
    device_to_use: Device plane to target (e.g., "TPU:0").
    trace_matchers: Optional tuple of event name matchers for filtering.

  Returns:
      A formatted string, list of dict records, or enriched summary dict.
  """
  try:
    if isinstance(source, (dict, list)):
      records = (
          source.get("kernel_records", source)
          if isinstance(source, dict)
          else source
      )
      if not isinstance(records, list):
        records = [records] if isinstance(records, dict) else []
      if not kernel_name:
        records = records[:limit]
      if output_format == "dict":
        return source
      if output_format == "markdown":
        return format_markdown_table(records, kernel_name)
      return json.dumps(source if include_summary else records, indent=2)

    kernel_durations_us = collections.defaultdict(list)
    all_intervals: list[tuple[int, int]] = []
    step_durations_us: list[float] = []

    for plane in xplane_tools.iter_planes(source):
      if not re.search(r"^/device:.*", plane.name):
        continue

      is_tpu = "TPU" in plane.name.upper()

      for line in plane.lines:
        line_name_upper = line.name.upper()
        # Restrict TPU compute events to XLA/Pallas/LLO
        # to avoid timing inflation.
        if is_tpu and not any(
            w in line_name_upper for w in _DEVICE_LINE_INCLUDE
        ):
          # Still check XLA Modules for step durations if include_summary.
          if include_summary and "XLA MODULES" in line_name_upper:
            for event in line.events:
              step_durations_us.append(float(event.duration_ns) / 1000.0)
          continue
        elif not is_tpu and any(
            w in line_name_upper
            for w in _DEVICE_LINE_EXCLUDE
        ):
          continue

        for event in line.events:
          name_info = event.name
          tf_op_name = None
          for stat_name, stat_val in event.stats:
            if stat_name == "tf_op_name" and stat_val:
              tf_op_name = str(stat_val)
              break
          if tf_op_name:
            name_info = tf_op_name
          elif name_info.isdigit():
            for stat_name, stat_val in event.stats:
              if stat_val and stat_name in (
                  "msg",
                  "message",
                  "annotation",
                  "label",
              ):
                name_info = str(stat_val)
                break

          if kernel_name and name_info != kernel_name:
            continue

          # Apply trace matchers if provided.
          if trace_matchers:
            if not any(_matches(m, name_info) for m in trace_matchers):
              continue

          dur_us = float(event.duration_ns) / 1000.0
          kernel_durations_us[name_info].append(dur_us)

          if include_summary:
            start_ns = int(event.start_ns)
            end_ns = start_ns + int(event.duration_ns)
            all_intervals.append((start_ns, end_ns))

    if not kernel_durations_us:
      msg = f"No kernel stats found for session {source}"
      if kernel_name:
        msg += f" and kernel {kernel_name}"
      if output_format == "dict" and include_summary:
        return {
            "total_device_duration_ns": 0,
            "total_device_duration_us": 0.0,
            "total_device_duration_ms": 0.0,
            "kernel_records": [],
            "step_durations_us": [],
            "stats": {"mean_us": 0.0, "std_us": 0.0},
        }
      if output_format == "dict":
        return []
      if output_format == "markdown":
        return f"# Info\n{msg}\n"
      return json.dumps({"info": msg}, indent=2)

    records = []
    for name, dur_list in kernel_durations_us.items():
      count = len(dur_list)
      total_us = sum(dur_list)
      avg_us = total_us / count if count > 0 else 0.0
      records.append({
          "kernel_name": name,
          "total_duration_us": round(total_us, 4),
          "execution_count": count,
          "avg_duration_us": round(avg_us, 4),
      })

    records.sort(key=lambda x: x["total_duration_us"], reverse=True)
    if not kernel_name:
      records = records[:limit]

    if include_summary:
      total_ns = compute_disjoint_interval_union_ns(all_intervals)
      total_us = float(total_ns / 1000.0)
      total_ms = float(total_ns / 1_000_000.0)
      mean_us = total_us if not step_durations_us else (
          sum(step_durations_us) / len(step_durations_us)
      )
      std_us = 0.0
      if len(step_durations_us) > 1:
        std_us = stats_mod.stdev(step_durations_us)
      summary = {
          "total_device_duration_ns": total_ns,
          "total_device_duration_us": total_us,
          "total_device_duration_ms": total_ms,
          "kernel_records": records,
          "step_durations_us": step_durations_us,
          "stats": {"mean_us": round(mean_us, 4), "std_us": round(std_us, 4)},
      }
      if output_format == "dict":
        return summary
      if output_format == "markdown":
        return json.dumps(summary, indent=2)
      return json.dumps(summary, indent=2)

    if output_format == "dict":
      return records
    if output_format == "markdown":
      return format_markdown_table(records, kernel_name)
    return json.dumps(records, indent=2)

  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.exception(
        "Error in OSS get_kernel_stats for source %r", source
    )
    if output_format == "dict":
      raise
    error_msg = f"Failed to get kernel stats: {e!r}"
    if output_format == "markdown":
      return f"# Error\n{error_msg}\n"
    return json.dumps({"error": error_msg}, indent=2)


def get_avg_step_time(
    source: Any,
    *,
    func_name: str | None = None,
    output_format: Literal["json", "dict"] = "json",
) -> Any:
  """Computes average step time from local XPlane 'XLA Modules' envelopes in OSS."""
  try:
    step_durations_ms = []

    for plane in xplane_tools.iter_planes(source):
      if not re.search(r"^/device:.*", plane.name):
        continue

      for line in plane.lines:
        if "XLA MODULES" not in line.name.upper():
          continue

        for event in line.events:
          ev_name = event.name
          if func_name and func_name not in ev_name:
            continue

          step_durations_ms.append(float(event.duration_ns) / 1_000_000.0)

    if not step_durations_ms:
      res_err = {
          "error": (
              f"No steps matching func_name '{func_name}' found in"
              f" {source}."
          )
      }
      if output_format == "dict":
        return res_err
      return json.dumps(res_err, indent=2)

    step_count = len(step_durations_ms)
    avg_ms = sum(step_durations_ms) / step_count
    res = {
        "avg_step_time_ms": round(avg_ms, 4),
        "step_count": step_count,
    }
    return res if output_format == "dict" else json.dumps(res, indent=2)

  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.exception("Error across OSS get_avg_step_time")
    if output_format == "dict":
      raise
    return json.dumps(
        {"error": f"Failed to get average step time: {e!r}"}, indent=2
    )
