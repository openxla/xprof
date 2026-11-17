"""OSS hermetic backend to compute kernel stats from local XPlane traces."""

import collections
import json
import logging
import re
import statistics as stats_mod
from typing import Any, Dict, List, Literal

from xprof.cli.internal import decorators
from xprof.cli.internal.oss import xplane_tools

_DEVICE_PLANE_RE = re.compile(r"^/device:.*")

# Top-level accelerator chip planes, e.g. "/device:TPU:0" or "/device:GPU:3".
# Sub-planes of a chip (e.g. "/device:TPU:0 SparseCore 0") do not match.
_ACCELERATOR_PLANE_RE = re.compile(r"^/device:(TPU|GPU):(\d+)$", re.IGNORECASE)

# `device_to_use` value that aggregates every device plane.
ALL_DEVICES = "all"

# Exact XLine names carrying *top-level* device kernels. Substring matching is
# unsafe here: with --xla_xprof_enable_custom_call_tracing=true the TPU tracer
# adds "Pallas Primitives" and "LLO Ops" lines holding the regions *inside* a
# kernel, and a substring test for "PALLAS" / "LLO OPS" matches those too,
# promoting every intra-kernel region to a top-level kernel. This mirrors the
# exact-name check already used by the kernel utilization converter.
_TPU_KERNEL_LINES = frozenset(("XLA OPS", "PALLAS"))

# Exact XLine names (and a suffix) carrying regions *within* a kernel. These
# overlap their containing kernel, so reporting them alongside it double counts
# and can rank a region above the kernel that contains it.
_TPU_INTRA_KERNEL_LINES = frozenset(("LLO OPS", "PALLAS PRIMITIVES"))
_TPU_INTRA_KERNEL_LINE_SUFFIX = " INSTRUCTIONS"

# Standard device line exclusion filters for non-computational metadata.
_DEVICE_LINE_EXCLUDE = (
    "COUNTER",
    "MODULES",
    "OVERLAY",
    "SYNC FLAG",
    "SENSOR",
)


def classify_tpu_line(line_name: str) -> str:
  """Classifies a TPU device XLine by the granularity of events it carries.

  Args:
    line_name: The XLine name, in any case.

  Returns:
    'kernel' for top-level kernel lines, 'intra_kernel' for lines holding
    regions inside a kernel, and 'other' for everything else.
  """
  upper = line_name.upper()
  if upper in _TPU_KERNEL_LINES:
    return "kernel"
  if upper in _TPU_INTRA_KERNEL_LINES or upper.endswith(
      _TPU_INTRA_KERNEL_LINE_SUFFIX
  ):
    return "intra_kernel"
  return "other"


def resolve_device_planes(
    plane_names: list[str], device_to_use: str | None
) -> tuple[str, list[str]]:
  """Resolves `device_to_use` to the device plane names to aggregate.

  Args:
    plane_names: Names of every plane in the profile, in order. A name may
      repeat when the source holds several trace files (one per host).
    device_to_use: None picks the first accelerator chip plane, TPU before GPU
      and then by device index. "all" selects every device plane. Any other
      value names one chip, e.g. "GPU:0" or "/device:TPU:1" (case-insensitive
      device type); its sub-planes such as "/device:TPU:0 SparseCore 0" are
      included too.

  Returns:
    A (device, plane_names) pair: the resolved device plane name (or "all"),
    and the de-duplicated device plane names to aggregate.

  Raises:
    ValueError: If `device_to_use` names a device that is not in the profile.
  """
  device_names = list(
      dict.fromkeys(n for n in plane_names if _DEVICE_PLANE_RE.search(n))
  )
  if not device_names:
    return ALL_DEVICES, []
  if device_to_use is not None and device_to_use.strip().lower() == ALL_DEVICES:
    return ALL_DEVICES, device_names
  if device_to_use is None:
    chips = []
    for name in device_names:
      match = _ACCELERATOR_PLANE_RE.match(name)
      if match:
        device_type_rank = 0 if match.group(1).upper() == "TPU" else 1
        chips.append((device_type_rank, int(match.group(2)), name))
    if not chips:
      return ALL_DEVICES, device_names
    chosen = min(chips)[2]
  else:
    target = device_to_use.strip()
    if not target.lower().startswith("/device:"):
      target = "/device:" + target
    chosen = next(
        (n for n in device_names if n.lower() == target.lower()), None
    )
    if chosen is None:
      raise ValueError(
          f"device_to_use={device_to_use!r} matches no device plane in the"
          f" profile. Available device planes: {device_names}. Pass"
          f" device_to_use={ALL_DEVICES!r} to aggregate every device plane."
      )
  selected = [
      n for n in device_names if n == chosen or n.startswith(chosen + " ")
  ]
  return chosen, selected


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
    df_records: List[Dict[str, Any]],
    kernel_name: str | None,
    device: str | None = None,
) -> str:
  """Formats kernel stats as a markdown table.

  Args:
    df_records: Kernel records to render.
    kernel_name: The kernel the records were filtered to, if any.
    device: The device the records were aggregated over, if known; shown
      under the heading.

  Returns:
    The markdown table.
  """
  lines = []
  if kernel_name:
    lines.append(f"# Kernel Stats for `{kernel_name}`")
  else:
    lines.append("# Top Kernels by Duration")
  lines.append("")
  if device:
    lines.append(f"Device: `{device}`")
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


def _apply_kernel_truncation(
    records: list[dict[str, Any]],
    *,
    kernel_name: str | None,
    limit: int,
    total_duration_us_all: float | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
  """Computes truncation fields and applies `limit` to kernel records."""
  total_records = len(records)
  non_region = [
      r for r in records if not r.get("is_intra_kernel_region", False)
  ]
  sliced = records[:limit] if not kernel_name and limit > 0 else list(records)
  truncation_fields = decorators.build_truncation_fields(
      total_records=total_records,
      returned_records=len(sliced),
      top_level_records=len(non_region),
      total_field="total_duration_us_all_records",
      total_value=total_duration_us_all,
  )
  return truncation_fields, sliced


def _format_kernel_records_output(
    records: list[dict[str, Any]],
    *,
    truncation_fields: dict[str, Any],
    kernel_name: str | None,
    limit: int,
    output_format: Literal["json", "markdown"],
    device: str | None = None,
) -> str:
  """Formats kernel records as markdown or JSON and emits truncation notes."""
  truncated = bool(truncation_fields["truncated"])
  total_records = int(truncation_fields["total_records"])
  raw_total_dur = truncation_fields.get("total_duration_us_all_records")
  total_dur_all = float(raw_total_dur) if raw_total_dur is not None else None
  if output_format == "markdown":
    table = format_markdown_table(records, kernel_name, device)
    if truncated:
      if total_dur_all is not None:
        table += (
            f"\n\nShowing {len(records)} of {total_records} kernels"
            f" (--limit={limit}). Total duration across all outermost"
            f" kernels: {total_dur_all:.2f} us (sums across device planes,"
            " excluding nested loop body ops and intra-kernel regions; pass"
            " --include_summary=True for wall-clock"
            " total_device_duration_us). Do not sum the rows above to get a"
            " total."
        )
      else:
        table += (
            f"\n\nShowing {len(records)} of {total_records} kernels"
            f" (--limit={limit}). Total duration across all kernels is"
            " unavailable for pre-computed records without event timestamps."
            " Do not sum the rows above to get a total."
        )
    return table
  if truncated:
    # The JSON result is a plain list, so report truncation on stderr, in
    # the same style as the `xprof-capture:` line, instead of changing its
    # shape. `include_summary=True` returns these fields in the object.
    if total_dur_all is not None:
      decorators.emit_stderr_note(
          f"xprof-note: truncated: returned {len(records)} of"
          f" {total_records} kernels (--limit={limit});"
          f" total_duration_us_all_records={total_dur_all}"
          " (sum of outermost kernel durations across device planes,"
          " excluding nested loop body ops and intra-kernel regions). Do"
          " not sum the rows to get a total; pass --include_summary=True"
          " for total_device_duration_us and truncation fields.\n"
      )
    else:
      decorators.emit_stderr_note(
          f"xprof-note: truncated: returned {len(records)} of"
          f" {total_records} kernels (--limit={limit});"
          " total_duration_us_all_records=null (unavailable for"
          " pre-computed records without event timestamps). Do not sum the"
          " rows to get a total; pass --include_summary=True for"
          " truncation fields.\n"
      )
  return json.dumps(records, indent=2)


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
    output_format: Literal["json", "markdown"] = "json",
    include_summary: bool = False,
    device_to_use: str | None = None,
    trace_matchers: tuple[str, ...] | None = None,
    include_intra_kernel_regions: bool = False,
    bypass_cache: bool = False,
) -> Any:
  """Computes performance metrics for operations from local XPlanes in OSS.

  Supports polymorphic inputs: session IDs, file paths, bytes, or in-memory
  ProfileData/XSpace objects. When include_summary=True, the JSON output is an
  enriched summary object containing ground-truth device durations via Disjoint
  Interval Union, step durations, and stats.

  Args:
    source: XProf session ID, local file/directory path, serialized XSpace
      bytes, in-memory ProfileData/XSpace object, or pre-computed records.
    kernel_name: Optional specific tf_op_name / kernel name to filter by.
    limit: Number of top kernels to return when kernel_name is not provided.
    output_format: Output format - 'json' (JSON string) or 'markdown' (markdown
      table string).
    include_summary: If True, computes and returns ground-truth timing via
      Disjoint Interval Union alongside per-kernel records.
    device_to_use: Device to aggregate. None (default) picks the first
      accelerator chip plane in the profile, TPU before GPU and then by index
      (e.g. "/device:GPU:0" on a GPU profile). Pass a device such as "TPU:1"
      or "GPU:3" to select it, or "all" to sum every device plane. Raises
      ValueError if the named device is not in the profile. Ignored for
      pre-computed record inputs.
    trace_matchers: Optional tuple of event name matchers for filtering.
    include_intra_kernel_regions: If True, also emits events from TPU lines that
      carry regions *inside* a kernel ("LLO Ops", "Pallas Primitives", "<unit>
      Instructions"). These overlap the kernel that contains them, so they are
      excluded by default on trace inputs to keep the ranking over kernels only.
      Included records are tagged with is_intra_kernel_region=True. Pre-computed
      record inputs keep all supplied rows unchanged.
    bypass_cache: Whether to bypass cache.

  Returns:
      A JSON string (a list of kernel records, or a summary object when
      include_summary=True) or a markdown table string. The summary object
      also has `truncated`, `total_records`, `top_level_records` (count of
      kernel records excluding intra-kernel regions), `returned_records`, and
      `total_duration_us_all_records` (sum of outermost kernel event durations
      across device planes before `limit`, excluding ops nested inside
      control-flow kernels such as `%while` and excluding intra-kernel regions,
      or `None` when called with pre-computed records that lack event
      timestamps; see `total_device_duration_us` for wall-clock device time).
      When `limit` drops rows, the list form writes an `xprof-note: truncated`
      line to stderr and the markdown form appends a "Showing N of M kernels"
      line.
  """
  del bypass_cache
  if isinstance(source, (int, float)):
    source = str(source)
  try:
    if isinstance(source, (dict, list)):
      raw_records = (
          source.get("kernel_records", source)
          if isinstance(source, dict)
          else source
      )
      if not isinstance(raw_records, list):
        raw_records = [raw_records] if isinstance(raw_records, dict) else []
      # Pre-computed records carry no event start times, so nested loop body
      # ops cannot be distinguished from outermost kernels; leave
      # total_duration_us_all_records as None rather than summing records.
      records = [
          r
          for r in raw_records
          if isinstance(r, dict)
          and (not kernel_name or r.get("kernel_name") == kernel_name)
      ]
      truncation_fields, records = _apply_kernel_truncation(
          records,
          kernel_name=kernel_name,
          limit=limit,
          total_duration_us_all=None,
      )
      if include_summary:
        summary_out = dict(source) if isinstance(source, dict) else {}
        summary_out["kernel_records"] = records
        summary_out.update(truncation_fields)
        return json.dumps(summary_out, indent=2)
      return _format_kernel_records_output(
          records,
          truncation_fields=truncation_fields,
          kernel_name=kernel_name,
          limit=limit,
          output_format=output_format,
      )

    kernel_durations_us = collections.defaultdict(list)
    top_level_total_us = 0.0
    all_intervals: list[tuple[int, int]] = []
    line_intervals: dict[str, list[tuple[int, int]]] = collections.defaultdict(
        list
    )
    custom_call_intervals: list[tuple[int, int]] = []
    step_durations_us: list[float] = []
    excluded_region_lines: set[str] = set()

    planes = list(xplane_tools.iter_planes(source))
    device, device_plane_names = resolve_device_planes(
        [plane.name for plane in planes], device_to_use
    )
    selected_planes = frozenset(device_plane_names)
    if device_to_use is None and device != ALL_DEVICES:
      num_chips = len({
          p.name for p in planes if _ACCELERATOR_PLANE_RE.match(p.name)
      })
      if num_chips > 1:
        decorators.emit_stderr_note(
            f"xprof-note: device_to_use=None picked {device} of {num_chips}"
            " accelerator devices; pass device_to_use='<TYPE>:<N>' for"
            " another device or 'all' to sum every device.\n"
        )

    for plane in planes:
      if plane.name not in selected_planes:
        continue

      is_tpu = "TPU" in plane.name.upper()

      for line in plane.lines:
        line_name_upper = line.name.upper()
        if include_summary:
          for event in line.events:
            start_ns = int(event.start_ns)
            end_ns = start_ns + int(event.duration_ns)
            line_intervals[line.name].append((start_ns, end_ns))

        is_region_line = False
        if is_tpu:
          line_kind = classify_tpu_line(line.name)
          if line_kind == "intra_kernel":
            # Regions inside a kernel overlap their container, so they are not
            # top-level kernels and must not be ranked against one.
            if not include_intra_kernel_regions:
              excluded_region_lines.add(line.name)
              continue
            is_region_line = True
          elif line_kind != "kernel":
            # Still check XLA Modules for step durations if include_summary.
            if include_summary and "XLA MODULES" in line_name_upper:
              for event in line.events:
                step_durations_us.append(float(event.duration_ns) / 1000.0)
            continue
        elif any(w in line_name_upper for w in _DEVICE_LINE_EXCLUDE):
          continue

        # Per (plane, line): an event is outermost unless an earlier event on
        # the same line still covers its start (e.g. a `%while` op covering the
        # ops in its body on `XLA Ops`).
        cover_end_ns = -1
        for event in sorted(
            line.events,
            key=lambda e: (int(e.start_ns), -int(e.duration_ns)),
        ):
          start_ns = int(event.start_ns)
          end_ns = start_ns + int(event.duration_ns)
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
          kernel_durations_us[(name_info, is_region_line)].append(dur_us)
          if not is_region_line and start_ns >= cover_end_ns:
            top_level_total_us += dur_us
          cover_end_ns = max(cover_end_ns, end_ns)

          # Region intervals are nested inside kernel intervals, so including
          # them would not change the union but would blur its meaning.
          if include_summary and not is_region_line:
            all_intervals.append((start_ns, end_ns))
            name_lower = name_info.lower()
            if (
                name_lower.startswith("custom-call")
                or "custom_call" in name_lower
                or "pallas" in name_lower
                or "PALLAS" in line_name_upper
            ):
              custom_call_intervals.append((start_ns, end_ns))

    if not kernel_durations_us:
      msg = f"No kernel stats found for session {source}"
      if kernel_name:
        msg += f" and kernel {kernel_name}"
      if device != ALL_DEVICES:
        msg += f" on device {device}"
      if output_format == "markdown":
        return f"# Info\n{msg}\n"
      return json.dumps({"info": msg}, indent=2)

    records = []
    for (name, is_region), dur_list in kernel_durations_us.items():
      count = len(dur_list)
      total_us = sum(dur_list)
      avg_us = total_us / count if count > 0 else 0.0
      # Extract compact canonical name (e.g. 'fusion.668')
      canonical = name.strip().split(":")[0].strip().split()[0] if name else ""
      records.append({
          "kernel_name": name,
          "canonical_name": canonical,
          "short_name": canonical,
          "hlo_op_name": canonical,
          "raw_name": name,
          "total_duration_us": round(total_us, 4),
          "execution_count": count,
          "avg_duration_us": round(avg_us, 4),
          "is_intra_kernel_region": is_region,
      })

    records.sort(key=lambda x: x["total_duration_us"], reverse=True)
    # `top_level_total_us` sums outermost kernel events per (plane, line)
    # before `limit` drops any rows, so ops inside `%while` bodies and
    # intra-kernel regions are not double-counted.
    truncation_fields, records = _apply_kernel_truncation(
        records,
        kernel_name=kernel_name,
        limit=limit,
        total_duration_us_all=top_level_total_us,
    )

    if include_summary:
      total_ns = compute_disjoint_interval_union_ns(all_intervals)
      total_us = float(total_ns / 1000.0)
      total_ms = float(total_ns / 1_000_000.0)
      by_line_duration_ns = {
          line_name: compute_disjoint_interval_union_ns(intervals)
          for line_name, intervals in line_intervals.items()
      }
      custom_call_ns = compute_disjoint_interval_union_ns(custom_call_intervals)
      custom_call_us = round(float(custom_call_ns / 1000.0), 4)
      custom_call_pct = (
          round((custom_call_ns / total_ns) * 100.0, 2) if total_ns > 0 else 0.0
      )
      mean_us = total_us if not step_durations_us else (
          sum(step_durations_us) / len(step_durations_us)
      )
      std_us = 0.0
      if len(step_durations_us) > 1:
        std_us = stats_mod.stdev(step_durations_us)
      summary: dict[str, Any] = {
          "total_device_duration_ns": total_ns,
          "total_device_duration_us": total_us,
          "total_device_duration_ms": total_ms,
          "by_line_duration_ns": by_line_duration_ns,
          "custom_call_duration_us": custom_call_us,
          "custom_call_share_pct": custom_call_pct,
          "kernel_records": records,
          "step_durations_us": step_durations_us,
          "stats": {"mean_us": round(mean_us, 4), "std_us": round(std_us, 4)},
          "device": device,
          "device_planes": device_plane_names,
          **truncation_fields,
      }
      if excluded_region_lines:
        summary["excluded_intra_kernel_region_lines"] = sorted(
            excluded_region_lines
        )
        summary["note"] = (
            "Events on intra-kernel region lines were excluded because they"
            " overlap the kernel that contains them and would otherwise be"
            " ranked as top-level kernels. Pass"
            " include_intra_kernel_regions=True to include them, tagged with"
            " is_intra_kernel_region."
        )
      return json.dumps(summary, indent=2)

    return _format_kernel_records_output(
        records,
        truncation_fields=truncation_fields,
        kernel_name=kernel_name,
        limit=limit,
        output_format=output_format,
        device=device,
    )

  except (FileNotFoundError, ValueError):
    raise
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.exception(
        "Error in OSS get_kernel_stats for source %r", source
    )
    raise RuntimeError(f"Failed to get kernel stats: {e}") from e
