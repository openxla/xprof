"""LLO Runtime Report Generator for XProf CLI.

Generates structured JSON and GitHub-flavored Markdown/HTML reports from
runtime LLO events stored in SQLite. Provides empirical feedback for RL agents
and performance engineers on scheduling efficiency, concurrency, and
bottlenecks.
"""

import sqlite3
from typing import Any
from xprof.cli.internal.oss import llo_detectors  # pyrefly: ignore[missing-import]


def generate_structured_report(
    db_path: str,
    kernel_filter: str | None = None,
    session_id: str = "",
) -> dict[str, Any]:
  """Generates a complete structured report dictionary conforming to XprofLloRuntimeReport schema.

  Args:
    db_path: Path to the SQLite database containing llo_events.
    kernel_filter: Optional substring filter across scopes, details, and events.
    session_id: Optional XProf session ID for report metadata.

  Returns:
    Dictionary exactly conforming to XprofLloRuntimeReport schema.
  """
  conn = sqlite3.connect(db_path)
  try:
    c = conn.cursor()
    filter_sql, filter_params = llo_detectors._build_filter_sql(kernel_filter)  # pylint: disable=protected-access
    c.execute(
        f"SELECT COUNT(*) FROM llo_events WHERE 1=1{filter_sql}", filter_params
    )
    total_event_count = (c.fetchone() or (0,))[0]

    concurrency = llo_detectors.analyze_runtime_concurrency_and_overlap(
        c, time_slice_ps=1000, window_size_ps=50000, kernel_filter=kernel_filter
    )
    bubbles = llo_detectors.detect_runtime_scheduling_bubbles(
        c,
        util_threshold_pct=25.0,
        min_consecutive_bundles=10,
        min_duration_ps=6666,
        kernel_filter=kernel_filter,
    )
    zero_init = llo_detectors.detect_runtime_blocked_zero_init(
        c,
        max_start_delay_pct=5.0,
        min_vst_ratio_pct=30.0,
        kernel_filter=kernel_filter,
    )
    findings = llo_detectors.synthesize_findings(
        concurrency, bubbles, zero_init
    )

    total_runtime_duration_ps = concurrency.get("total_duration_ps", 0)

    report = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "XprofLloRuntimeReport",
        "report_metadata": {
            "db_path": db_path,
            "session_id": session_id or "",
            "kernel_name": kernel_filter or "ALL_KERNELS",
            "total_runtime_duration_ps": int(total_runtime_duration_ps),
            "total_event_count": int(total_event_count),
        },
        "concurrency_and_overlap": concurrency,
        "bottleneck_findings": {
            "scheduling_bubbles": bubbles,
            "blocked_zero_init": zero_init,
        },
        "findings": findings,
    }
    return report
  finally:
    conn.close()


def _render_bar(percentage: float, width: int = 20) -> str:
  """Renders a text progress bar representing a percentage [0, 100]."""
  pct = max(0.0, min(100.0, percentage))
  filled = int(round((pct / 100.0) * width))
  empty = width - filled
  return f"[`{'#' * filled}{'-' * empty}`] {percentage:.1f}%"


def _format_ps(ps: int | float) -> str:
  """Formats picoseconds into human-readable units (ps, ns, us, ms)."""
  if ps < 1_000:
    return f"{ps:.0f} ps"
  if ps < 1_000_000:
    return f"{ps / 1_000:.2f} ns"
  if ps < 1_000_000_000:
    return f"{ps / 1_000_000:.2f} us"
  return f"{ps / 1_000_000_000:.2f} ms"


def _render_table(headers: list[str], rows: list[list[Any]]) -> str:
  """Renders a clean GitHub-flavored Markdown table."""
  if not headers:
    return ""
  header_line = "| " + " | ".join(str(h) for h in headers) + " |"
  sep_line = "| " + " | ".join("---" for _ in headers) + " |"
  row_lines = [
      "| " + " | ".join(str(item) for item in row) + " |" for row in rows
  ]
  return "\n".join([header_line, sep_line] + row_lines)


def _format_ps_exact(ps: int | float) -> str:
  """Formats picoseconds with exact ps range plus human-readable units."""
  human = _format_ps(ps)
  return f"{int(ps):,} ps ({human})"


def generate_markdown_html_report(
    report_dict: dict[str, Any], collapsible_timelines: bool = True
) -> str:
  """Renders a GitHub-flavored Markdown + HTML report from structured LLO report data.

  Args:
    report_dict: The structured dictionary from generate_structured_report.
    collapsible_timelines: Whether to wrap long lists in collapsible <details>
      blocks.

  Returns:
    Formatted Markdown + HTML string.
  """
  meta = report_dict.get("report_metadata", {})
  conc = report_dict.get("concurrency_and_overlap", {})
  bot = report_dict.get("bottleneck_findings", {})
  findings = report_dict.get("findings", [])

  lines = []
  lines.append("# LLO Runtime Execution & Bottleneck Report\n")

  # 1. Executive Summary
  lines.append("## Executive Summary\n")
  summary_rows = [
      ["**Kernel Filter**", meta.get("kernel_name", "ALL_KERNELS")],
      [
          "**Total Runtime Duration**",
          _format_ps_exact(meta.get("total_runtime_duration_ps", 0)),
      ],
      ["**Total LLO Event Count**", f"{meta.get('total_event_count', 0):,}"],
  ]
  lines.append(_render_table(["Metric", "Value"], summary_rows))
  lines.append("\n")

  # 2. Concurrency & Overlap Analysis
  lines.append("## Concurrency & Overlap Analysis\n")
  conc_rows = [
      [
          "Overlapped Compute + Memory (`MXU` + `VLD`/`VST`)",
          _format_ps(conc.get("overlapped_mxu_mem_duration_ps", 0)),
          _render_bar(conc.get("overlapped_percentage", 0.0)),
      ],
      [
          "Pure Compute (`MXU`/`VALU` without Memory)",
          _format_ps(conc.get("pure_compute_duration_ps", 0)),
          _render_bar(conc.get("pure_compute_percentage", 0.0)),
      ],
      [
          "Pure Memory (`VLD`/`VST` without Compute)",
          _format_ps(conc.get("pure_memory_duration_ps", 0)),
          _render_bar(conc.get("pure_memory_percentage", 0.0)),
      ],
      [
          "Exposed Vector (`VALU` without `MXU`)",
          _format_ps(conc.get("exposed_vector_duration_ps", 0)),
          (
              f"{conc.get('exposed_vector_duration_ps', 0) / max(1, conc.get('total_duration_ps', 1)) * 100.0:.1f}%"
          ),
      ],
      [
          "Idle / Stalled Core Units",
          _format_ps(conc.get("idle_duration_ps", 0)),
          _render_bar(conc.get("idle_percentage", 0.0)),
      ],
  ]
  lines.append(
      _render_table(
          ["Execution State", "Duration", "Share of Timeline"], conc_rows
      )
  )
  lines.append("\n")

  # Lowest Overlap Region
  lor = conc.get("lowest_overlap_region")
  if lor and lor.get("duration_ps", 0) > 0:
    lines.append("### Lowest Compute/Memory Overlap Region\n")
    lor_rows = [[
        (
            f"`{_format_ps(lor.get('start_offset_ps', 0))} -"
            f" {_format_ps(lor.get('end_offset_ps', 0))}`"
        ),
        _format_ps(lor.get("duration_ps", 0)),
        f"{lor.get('overlap_percentage', 0.0):.1f}%",
    ]]
    lines.append(
        _render_table(
            ["Time Window", "Duration", "Co-Issuance Overlap"], lor_rows
        )
    )
    lines.append("\n")
    if lor.get("attributed_scopes") or lor.get("source_locations"):
      if collapsible_timelines:
        lines.append("<details>")
        lines.append(
            "<summary><strong>Active Scopes & Source Locations in Lowest"
            " Overlap Region</strong></summary>\n"
        )
      if lor.get("attributed_scopes"):
        lines.append("**Attributed Scopes:**")
        for sc in lor.get("attributed_scopes", []):
          lines.append(f"- `{sc}`")
        lines.append("")
      if lor.get("source_locations"):
        lines.append("**Source Locations:**")
        loc_rows = [
            [
                loc.get("file", ""),
                str(loc.get("line", "")),
                loc.get("line_name", ""),
            ]
            for loc in lor.get("source_locations", [])
        ]
        lines.append(_render_table(["File", "Line", "Operation"], loc_rows))
        lines.append("")
      if collapsible_timelines:
        lines.append("</details>\n")

  # 3. Bottleneck Findings
  lines.append("## Bottleneck Findings\n")

  # Scheduling Bubbles
  bubbles = bot.get("scheduling_bubbles", [])
  lines.append(f"### Scheduling Bubbles ({len(bubbles)} detected)\n")
  if bubbles:
    if collapsible_timelines and len(bubbles) > 3:
      lines.append("<details>")
      lines.append(
          f"<summary><strong>View all {len(bubbles)} scheduling"
          " bubbles</strong></summary>\n"
      )
    bubble_rows = []
    for b in bubbles:
      b_range = (
          f"{b.get('start_bundle', 0)} - {b.get('end_bundle', 0)}"
          if b.get("start_bundle", 0) > 0
          else "N/A"
      )
      bubble_rows.append([
          (
              f"`{_format_ps(b.get('start_offset_ps', 0))} -"
              f" {_format_ps(b.get('end_offset_ps', 0))}`"
          ),
          _format_ps(b.get("duration_ps", 0)),
          b_range,
      ])
    lines.append(
        _render_table(
            ["Time Interval", "Duration", "Bundle Range"], bubble_rows
        )
    )
    lines.append("\n")

    # Sample top 5 bubbles for detailed scopes/locations display
    for idx, b in enumerate(bubbles[:5]):
      lines.append(
          f"**Bubble #{idx + 1} Context"
          f" (`{_format_ps(b.get('duration_ps', 0))}` | Interval:"
          f" `{int(b.get('start_offset_ps', 0)):,} ps ->"
          f" {int(b.get('end_offset_ps', 0)):,} ps`)**"
      )
      if b.get("preceding_op") or b.get("following_op"):
        lines.append(
            "- **Surrounding Execution Boundary**: Preceded by"
            f" `{b.get('preceding_op', 'None')}`, Followed by"
            f" `{b.get('following_op', 'None')}`"
        )
      if b.get("attributed_scopes"):
        lines.append(
            "- **Enclosing Scopes**: "
            + ", ".join(f"`{s}`" for s in b.get("attributed_scopes", [])[:3])
        )
      else:
        lines.append(
            "- **Enclosing Scopes**: No explicit `llo_scopes` recorded inside"
            " this stall window."
        )
      if b.get("source_locations"):
        locs = [
            f"`{l.get('file', '')}:{l.get('line', '')}`"
            for l in b.get("source_locations", [])[:3]
        ]
        lines.append("- **Source Locations**: " + ", ".join(locs))
      else:
        lines.append(
            "- **Source Locations**: No explicit source line symbols recorded"
            " inside this stall window."
        )
      lines.append("")
    if collapsible_timelines and len(bubbles) > 5:
      lines.append("</details>\n")
  else:
    lines.append(
        "No severe scheduling bubbles (<25% utilization across all core tracks)"
        " detected.\n"
    )

  # Blocked Zero-Initialization
  zinit = bot.get("blocked_zero_init", {})
  lines.append("### Blocked Zero-Initialization Phase\n")
  if zinit and zinit.get("detected"):
    zinit_rows = [[
        _format_ps(zinit.get("start_delay_ps", 0)),
        f"{zinit.get('start_delay_pct', 0.0):.1f}%",
        f"`{zinit.get('first_compute_op', '')}`",
        f"{zinit.get('write_ratio_pct', 0.0):.1f}%",
    ]]
    lines.append(
        _render_table(
            [
                "Startup Delay",
                "Share of Timeline",
                "First Compute Op",
                "Store Write Ratio in Delay",
            ],
            zinit_rows,
        )
    )
    lines.append("\n")
    if zinit.get("attributed_scopes") or zinit.get("source_locations"):
      if collapsible_timelines:
        lines.append("<details>")
        lines.append(
            "<summary><strong>Zero-Init Store Operations"
            " Context</strong></summary>\n"
        )
      if zinit.get("attributed_scopes"):
        lines.append("**Attributed Scopes:**")
        for sc in zinit.get("attributed_scopes", []):
          lines.append(f"- `{sc}`")
        lines.append("")
      if zinit.get("source_locations"):
        loc_rows = [
            [
                loc.get("file", ""),
                str(loc.get("line", "")),
                loc.get("line_name", ""),
            ]
            for loc in zinit.get("source_locations", [])
        ]
        lines.append(_render_table(["File", "Line", "Operation"], loc_rows))
        lines.append("")
      if collapsible_timelines:
        lines.append("</details>\n")
  else:
    lines.append(
        "No significant startup delay caused by pre-compute store operations"
        " detected.\n"
    )

  # 4. Summary of Bottleneck Flags
  lines.append("## Summary of Bottleneck Flags\n")
  if not findings:
    lines.append(
        "No concurrency anomalies, scheduling bubbles, or zero-init blocks"
        " flagged (`findings = []`).\n"
    )
  else:
    lines.append(
        "The following objective bottleneck flags (`findings`) triggered based"
        " on empirical execution thresholds:\n"
    )
    for f in findings:
      if f == "runtime_scheduling_bubbles":
        lines.append(
            f"- **`{f}`**: `{len(bubbles)}` scheduling bubbles detected where"
            " all five execution tracks (`MXU`, `VALU`, `VLD`, `VST`, `XLU`)"
            " dropped below 25.0% utilization."
        )
      elif f == "blocked_zero_init":
        lines.append(
            f"- **`{f}`**: Kernel startup delayed by"
            f" `{_format_ps_exact(zinit.get('start_delay_ps', 0))}`"
            f" ({zinit.get('start_delay_pct', 0.0):.1f}% of total duration)"
            " before the first compute instruction, with `VST` stores"
            f" comprising {zinit.get('write_ratio_pct', 0.0):.1f}% of"
            " pre-compute memory ops."
        )
      elif f == "low_memory_compute_overlap":
        lines.append(
            f"- **`{f}`**: High memory activity with low `MXU` concurrency"
            " (Overlapped `MXU` + `Mem` ratio:"
            f" {conc.get('overlapped_percentage', 0.0):.1f}% vs Pure Memory:"
            f" {conc.get('pure_memory_percentage', 0.0):.1f}%)."
        )
      elif f == "exposed_vector_bubbles":
        lines.append(
            f"- **`{f}`**: Exposed vector ALU (`VALU` without `MXU`) occupies"
            f" `{_format_ps_exact(conc.get('exposed_vector_duration_ps', 0))}`"
            f" ({conc.get('exposed_vector_duration_ps', 0) / max(1, conc.get('total_duration_ps', 1)) * 100.0:.1f}%"
            " of total duration)."
        )
      elif f == "exposed_memory_bubbles":
        lines.append(
            f"- **`{f}`**: Pure memory load/store operations without compute"
            " occupy"
            f" `{_format_ps_exact(conc.get('pure_memory_duration_ps', 0))}`"
            f" ({conc.get('pure_memory_percentage', 0.0):.1f}% of total"
            " duration)."
        )
      else:
        lines.append(f"- **`{f}`**: Objective threshold exceeded.")
    lines.append("")

  return "\n".join(lines)
