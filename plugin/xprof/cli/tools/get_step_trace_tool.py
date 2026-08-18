"""Tool to retrieve step execution breakdowns and timing data from XProf."""

import collections
from collections.abc import Sequence
import dataclasses
import json
import logging
import statistics
from typing import Any, Literal

from xprof.cli.internal import decorators

from xprof.cli.internal.oss import xprof_client

pywraprpc = None

_FETCH_EXCEPTIONS_LIST: list[type[BaseException]] = [
    RuntimeError,
    ValueError,
    TypeError,
    AttributeError,
    OSError,
]
if pywraprpc is not None:
  _FETCH_EXCEPTIONS_LIST.append(pywraprpc.RPCException)

_FETCH_EXCEPTIONS: tuple[type[BaseException], ...] = tuple(
    _FETCH_EXCEPTIONS_LIST
)

_DEFAULT_STEP_LIMIT = 20


@dataclasses.dataclass
class CommunicationBreakdown:
  """Breakdown of communication time into components.

  Attributes:
      all_reduce_ms: Time spent in All-Reduce (Cross-Replica Sum) in ms.
      send_ms: Time spent in Send operations in ms.
      recv_ms: Time spent in Recv operations in ms.
  """

  all_reduce_ms: float
  send_ms: float
  recv_ms: float


@dataclasses.dataclass
class StepInfo:
  """Step execution breakdown and timing information.

  Attributes:
      step_num: The step number (or None if unavailable).
      step_time_ms: Total step duration in ms.
      compute_time_ms: Time spent in device compute in ms.
      compute_percent: Percentage of step time spent in compute.
      communication_time_ms: Time spent in communication in ms.
      communication_percent: Percentage of step time spent in communication.
      infeed_time_ms: Time spent in host infeed in ms.
      infeed_percent: Percentage of step time spent in infeed.
      outfeed_time_ms: Time spent in host outfeed in ms.
      outfeed_percent: Percentage of step time spent in outfeed.
      bottleneck: Primary bottleneck identified for the step.
      communication_breakdown_ms: Detailed breakdown of communication
        components.
  """

  step_num: int | None
  step_time_ms: float
  compute_time_ms: float
  compute_percent: float
  communication_time_ms: float
  communication_percent: float
  infeed_time_ms: float
  infeed_percent: float
  outfeed_time_ms: float
  outfeed_percent: float
  bottleneck: str
  communication_breakdown_ms: CommunicationBreakdown | None = None


@dataclasses.dataclass
class SummaryData:
  """Aggregate summary metrics across all steps in the session.

  Attributes:
      total_steps: Total number of steps analyzed.
      step_time_ms_average: Average step duration in ms.
      step_time_ms_min: Minimum step duration in ms.
      step_time_ms_max: Maximum step duration in ms.
      step_time_ms_stddev: Standard deviation of step duration in ms.
      compute_time_ms_average: Average compute duration in ms.
      compute_percent: Percentage of average step time spent in compute.
      communication_time_ms_average: Average communication duration in ms.
      communication_percent: Percentage of average step time spent in
        communication.
      infeed_time_ms_average: Average infeed duration in ms.
      infeed_percent: Percentage of average step time spent in infeed.
      outfeed_time_ms_average: Average outfeed duration in ms.
      outfeed_percent: Percentage of average step time spent in outfeed.
      primary_bottleneck: Primary bottleneck identified across steps.
      conclusion: Optional summary conclusion or note.
      note: Optional additional note.
  """

  total_steps: int
  step_time_ms_average: float
  step_time_ms_min: float
  step_time_ms_max: float
  step_time_ms_stddev: float
  compute_time_ms_average: float
  compute_percent: float
  communication_time_ms_average: float
  communication_percent: float
  infeed_time_ms_average: float
  infeed_percent: float
  outfeed_time_ms_average: float
  outfeed_percent: float
  primary_bottleneck: str
  conclusion: str | None = None
  note: str | None = None


def _safe_float(val: Any) -> float:
  """Safely converts a value to float."""
  if val is None:
    return 0.0
  try:
    val_str = str(val).strip().lower()
    if val_str in ("nan", "none", "", "null"):
      return 0.0
    return float(val)
  except (ValueError, TypeError):
    return 0.0


def _build_summary(
    steps: Sequence[StepInfo],
    extra_props: dict[str, Any] | None = None,
) -> SummaryData | None:
  """Computes aggregate summary metrics across all steps."""
  if not steps:
    return None
  step_times = [s.step_time_ms for s in steps]
  compute_times = [s.compute_time_ms for s in steps]
  comm_times = [s.communication_time_ms for s in steps]
  infeed_times = [s.infeed_time_ms for s in steps]
  outfeed_times = [s.outfeed_time_ms for s in steps]

  avg_step = round(sum(step_times) / len(step_times), 4)
  avg_comp = round(sum(compute_times) / len(compute_times), 4)
  avg_comm = round(sum(comm_times) / len(comm_times), 4)
  avg_infeed = round(sum(infeed_times) / len(infeed_times), 4)
  avg_outfeed = round(sum(outfeed_times) / len(outfeed_times), 4)

  bottlenecks = [s.bottleneck for s in steps if s.bottleneck]
  primary_b = (
      collections.Counter(bottlenecks).most_common(1)[0][0]
      if bottlenecks
      else ("Communication" if avg_comm > avg_comp else "Compute")
  )

  conclusion = extra_props.get("summary_conclusion") if extra_props else None

  return SummaryData(
      total_steps=len(steps),
      step_time_ms_average=avg_step,
      step_time_ms_min=round(min(step_times), 4),
      step_time_ms_max=round(max(step_times), 4),
      step_time_ms_stddev=(
          round(statistics.stdev(step_times), 4) if len(step_times) > 1 else 0.0
      ),
      compute_time_ms_average=avg_comp,
      compute_percent=(
          round(avg_comp / avg_step * 100, 2) if avg_step > 0 else 0.0
      ),
      communication_time_ms_average=avg_comm,
      communication_percent=(
          round(avg_comm / avg_step * 100, 2) if avg_step > 0 else 0.0
      ),
      infeed_time_ms_average=avg_infeed,
      infeed_percent=(
          round(avg_infeed / avg_step * 100, 2) if avg_step > 0 else 0.0
      ),
      outfeed_time_ms_average=avg_outfeed,
      outfeed_percent=(
          round(avg_outfeed / avg_step * 100, 2) if avg_step > 0 else 0.0
      ),
      primary_bottleneck=primary_b,
      conclusion=conclusion,
  )


def _format_markdown(
    steps_data: Sequence[StepInfo],
    summary_data: SummaryData | None = None,
) -> str:
  """Formats step trace data as markdown tables."""
  lines = []
  lines.append("# Step Execution Trace & Breakdown")
  lines.append("")
  if summary_data:
    lines.append("## Session Summary")
    lines.append(f"- Total Steps: {summary_data.total_steps}")
    lines.append(
        f"- Step Time (ms): avg={summary_data.step_time_ms_average:.4f}, "
        f"min={summary_data.step_time_ms_min:.4f}, "
        f"max={summary_data.step_time_ms_max:.4f}, "
        f"stddev={summary_data.step_time_ms_stddev:.4f}"
    )
    lines.append(
        f"- Compute: {summary_data.compute_time_ms_average:.4f} ms "
        f"({summary_data.compute_percent:.2f}%)"
    )
    lines.append(
        f"- Communication: {summary_data.communication_time_ms_average:.4f} ms "
        f"({summary_data.communication_percent:.2f}%)"
    )
    lines.append(
        f"- Infeed: {summary_data.infeed_time_ms_average:.4f} ms "
        f"({summary_data.infeed_percent:.2f}%)"
    )
    lines.append(
        f"- Outfeed: {summary_data.outfeed_time_ms_average:.4f} ms "
        f"({summary_data.outfeed_percent:.2f}%)"
    )
    lines.append(f"- Primary Bottleneck: `{summary_data.primary_bottleneck}`")
    if summary_data.conclusion:
      lines.append(f"- Conclusion: {summary_data.conclusion}")
    if summary_data.note:
      lines.append(f"- Note: {summary_data.note}")
    lines.append("")

  if steps_data:
    lines.append("## Step Breakdown")
    lines.append("")
    lines.append(
        "| Step | Total (ms) | Compute (ms) | Comm (ms) | Infeed (ms) |"
        " Outfeed (ms) | Bottleneck |"
    )
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | :--- |")
    for step in steps_data:
      s_num = str(step.step_num) if step.step_num is not None else "-"
      lines.append(
          f"| {s_num} | {step.step_time_ms:.4f} | {step.compute_time_ms:.4f} "
          f"({step.compute_percent:.1f}%) | {step.communication_time_ms:.4f} "
          f"({step.communication_percent:.1f}%) | {step.infeed_time_ms:.4f} "
          f"({step.infeed_percent:.1f}%) | {step.outfeed_time_ms:.4f} "
          f"({step.outfeed_percent:.1f}%) | `{step.bottleneck}` |"
      )
    lines.append("")

  return "\n".join(lines)


def _format_error(
    error_msg: str, output_format: Literal["json", "markdown"]
) -> str:
  """Formats an error message in requested format."""
  if output_format == "markdown":
    return f"# Error\n{error_msg}\n"
  return json.dumps({"error": error_msg}, indent=2)


def _step_to_dict(step: StepInfo) -> dict[str, Any]:
  """Converts StepInfo dataclass to JSON-serializable dictionary."""
  res = dataclasses.asdict(step)
  if step.communication_breakdown_ms is None:
    res.pop("communication_breakdown_ms", None)
  return res


def _parse_pod_viewer(
    raw_data: str,
    step_num: int | None = None,
    device_core: int | None = None,
) -> tuple[list[StepInfo], dict[str, Any] | None]:
  """Extracts per-step breakdowns from pod_viewer.json."""
  try:
    data = json.loads(raw_data)
  except json.JSONDecodeError:
    return [], None

  pod_map = data.get("podStatsSequence", {}).get("podStatsMap", [])
  if not isinstance(pod_map, list):
    return [], None

  steps = []
  for entry in pod_map:
    if not isinstance(entry, dict):
      continue

    s_num = entry.get("stepNum")
    if s_num is not None:
      try:
        s_num = int(s_num)
      except (ValueError, TypeError):
        pass

    if step_num is not None and s_num != step_num:
      continue

    cores = entry.get("podStatsPerCore", {})
    if not isinstance(cores, dict):
      continue

    if device_core is not None:
      core_key = str(device_core)
      core_stats = [cores[core_key]] if core_key in cores else []
    else:
      core_stats = list(cores.values())

    if not core_stats:
      continue

    n = len(core_stats)
    total_us = (
        sum(_safe_float(c.get("totalDurationUs")) for c in core_stats) / n
    )
    comp_us = (
        sum(_safe_float(c.get("highFlopsComputeUs")) for c in core_stats) / n
    )
    crs_us = sum(_safe_float(c.get("crsDurationUs")) for c in core_stats) / n
    send_us = sum(_safe_float(c.get("sendDurationUs")) for c in core_stats) / n
    recv_us = sum(_safe_float(c.get("recvDurationUs")) for c in core_stats) / n
    infeed_us = (
        sum(_safe_float(c.get("hostInfeedDurationUs")) for c in core_stats) / n
    )
    outfeed_us = (
        sum(_safe_float(c.get("hostOutfeedDurationUs")) for c in core_stats) / n
    )

    total_ms = round(total_us / 1000.0, 4)
    comp_ms = round(comp_us / 1000.0, 4)
    crs_ms = round(crs_us / 1000.0, 4)
    send_ms = round(send_us / 1000.0, 4)
    recv_ms = round(recv_us / 1000.0, 4)
    comm_ms = round(crs_ms + send_ms + recv_ms, 4)
    infeed_ms = round(infeed_us / 1000.0, 4)
    outfeed_ms = round(outfeed_us / 1000.0, 4)

    comp_pct = round(comp_ms / total_ms * 100, 2) if total_ms > 0 else 0.0
    comm_pct = round(comm_ms / total_ms * 100, 2) if total_ms > 0 else 0.0
    infeed_pct = round(infeed_ms / total_ms * 100, 2) if total_ms > 0 else 0.0
    outfeed_pct = round(outfeed_ms / total_ms * 100, 2) if total_ms > 0 else 0.0

    b_list = [
        str(c.get("bottleneck")) for c in core_stats if c.get("bottleneck")
    ]
    if b_list:
      bottleneck = collections.Counter(b_list).most_common(1)[0][0]
    else:
      if comm_pct > comp_pct and comm_pct > infeed_pct:
        bottleneck = "Communication"
      elif infeed_pct > comp_pct:
        bottleneck = "Input / Infeed"
      else:
        bottleneck = "Compute"

    comm_breakdown = CommunicationBreakdown(
        all_reduce_ms=crs_ms,
        send_ms=send_ms,
        recv_ms=recv_ms,
    )

    steps.append(
        StepInfo(
            step_num=s_num,
            step_time_ms=total_ms,
            compute_time_ms=comp_ms,
            compute_percent=comp_pct,
            communication_time_ms=comm_ms,
            communication_percent=comm_pct,
            infeed_time_ms=infeed_ms,
            infeed_percent=infeed_pct,
            outfeed_time_ms=outfeed_ms,
            outfeed_percent=outfeed_pct,
            bottleneck=bottleneck,
            communication_breakdown_ms=comm_breakdown,
        )
    )

  return steps, None


def _parse_input_pipeline(
    raw_data: str,
    step_num: int | None = None,
) -> tuple[list[StepInfo], dict[str, Any] | None]:
  """Extracts per-step breakdowns from input_pipeline.json."""
  try:
    data = json.loads(raw_data)
  except json.JSONDecodeError:
    return [], None

  if not isinstance(data, list):
    return [], None

  step_section = None
  for section in data:
    if isinstance(section, dict) and "cols" in section and "rows" in section:
      cols = [col.get("id", "").lower() for col in section.get("cols", [])]
      if any("step" in c for c in cols) or any("infeed" in c for c in cols):
        step_section = section
        break

  if not step_section:
    return [], None

  cols = step_section.get("cols", [])
  col_map = {
      str(col.get("id")).lower(): idx
      for idx, col in enumerate(cols)
      if isinstance(col, dict) and col.get("id") is not None
  }
  step_col = col_map.get("stepnum", 0)
  comp_col = col_map.get("noninfeedtimems", 1)
  infeed_col = col_map.get("infeedtimems", 2)
  infeed_pct_col = col_map.get("infeedpercentaverage", 4)

  steps = []
  for row in step_section.get("rows", []):
    cells = row.get("c", [])
    if not cells:
      continue

    s_val = cells[step_col].get("v") if len(cells) > step_col else None
    if s_val is not None:
      try:
        s_num = int(s_val)
      except (ValueError, TypeError):
        s_num = s_val
    else:
      s_num = None

    if step_num is not None and s_num != step_num:
      continue

    comp_ms = (
        _safe_float(cells[comp_col].get("v")) if len(cells) > comp_col else 0.0
    )
    infeed_ms = (
        _safe_float(cells[infeed_col].get("v"))
        if len(cells) > infeed_col
        else 0.0
    )
    total_ms = round(comp_ms + infeed_ms, 4)

    if (
        len(cells) > infeed_pct_col
        and cells[infeed_pct_col].get("v") is not None
    ):
      infeed_pct = round(_safe_float(cells[infeed_pct_col].get("v")), 2)
    else:
      infeed_pct = round(infeed_ms / total_ms * 100, 2) if total_ms > 0 else 0.0

    comp_pct = round(comp_ms / total_ms * 100, 2) if total_ms > 0 else 0.0
    bottleneck = "Input / Infeed" if infeed_pct > 50.0 else "Compute"

    steps.append(
        StepInfo(
            step_num=s_num,
            step_time_ms=total_ms,
            compute_time_ms=round(comp_ms, 4),
            compute_percent=comp_pct,
            communication_time_ms=0.0,
            communication_percent=0.0,
            infeed_time_ms=round(infeed_ms, 4),
            infeed_percent=infeed_pct,
            outfeed_time_ms=0.0,
            outfeed_percent=0.0,
            bottleneck=bottleneck,
            communication_breakdown_ms=None,
        )
    )

  extra_props = step_section.get("p")
  return steps, extra_props if extra_props else None


def _parse_overview_page(
    raw_data: str,
) -> tuple[list[StepInfo], SummaryData | None]:
  """Fallback to parse aggregate overview statistics."""
  try:
    overview_data = json.loads(raw_data)
  except json.JSONDecodeError:
    return [], None

  if not isinstance(overview_data, list):
    return [], None

  all_p = {}
  for sec in overview_data:
    if isinstance(sec, dict):
      all_p.update(sec.get("p", {}))

  step_time = _safe_float(all_p.get("steptime_ms_average"))
  if step_time <= 0 and "stat_step_time" in all_p:
    step_time = _safe_float(
        str(all_p["stat_step_time"]).replace("ms", "").strip()
    )
  if step_time <= 0:
    return [], None

  infeed_avg = _safe_float(
      all_p.get("tc_infeed_ms_average", all_p.get("sc_infeed_ms_average"))
  )
  outfeed_avg = _safe_float(
      all_p.get("tc_outfeed_ms_average", all_p.get("sc_outfeed_ms_average"))
  )
  idle_avg = _safe_float(
      all_p.get("tc_idle_ms_average", all_p.get("sc_idle_ms_average"))
  )
  comp_avg = max(0.0, step_time - infeed_avg - outfeed_avg - idle_avg)
  infeed_pct = round(infeed_avg / step_time * 100, 2) if step_time > 0 else 0.0

  summary = SummaryData(
      total_steps=1,
      step_time_ms_average=round(step_time, 4),
      step_time_ms_min=round(step_time, 4),
      step_time_ms_max=round(step_time, 4),
      step_time_ms_stddev=round(
          _safe_float(all_p.get("steptime_ms_standard_deviation")), 4
      ),
      compute_time_ms_average=round(comp_avg, 4),
      compute_percent=(
          round(comp_avg / step_time * 100, 2) if step_time > 0 else 0.0
      ),
      communication_time_ms_average=0.0,
      communication_percent=0.0,
      infeed_time_ms_average=round(infeed_avg, 4),
      infeed_percent=infeed_pct,
      outfeed_time_ms_average=round(outfeed_avg, 4),
      outfeed_percent=(
          round(outfeed_avg / step_time * 100, 2) if step_time > 0 else 0.0
      ),
      primary_bottleneck="Input / Infeed" if infeed_pct > 50.0 else "Compute",
      note=(
          "Aggregated overview statistics (individual step breakdown not"
          " available)."
      ),
  )
  return [], summary


@decorators.cached(expire=86400)
def get_step_trace(
    session_id: str,
    *,
    step_num: int | None = None,
    limit: int = _DEFAULT_STEP_LIMIT,
    device_core: int | None = None,
    output_format: Literal["json", "markdown"] = "json",
    include_summary: bool = True,
) -> str:
  """Retrieves step execution breakdowns and timing data from an XProf session.

  Args:
      session_id: The unique XProf session ID.
      step_num: Optional step number to filter for.
      limit: Maximum steps in breakdown (default _DEFAULT_STEP_LIMIT).
      device_core: Optional specific core ID to filter by.
      output_format: Output format, either 'json' or 'markdown' (default
        'json').
      include_summary: Whether to include aggregate summary (default True).

  Returns:
      Formatted step execution and timing breakdown in requested format.
  """
  session_id = str(session_id)
  client = xprof_client.get_client()

  for tool_name, parser in (
      (
          "pod_viewer.json",
          lambda d: _parse_pod_viewer(d, step_num, device_core),
      ),
      ("input_pipeline.json", lambda d: _parse_input_pipeline(d, step_num)),
  ):
    try:
      result = client.fetch(
          tool_name=tool_name, session_id=session_id, format="json"
      )
    except _FETCH_EXCEPTIONS as e:
      logging.warning("Error fetching %s for %s: %s", tool_name, session_id, e)
      continue

    data = (
        result[1] if isinstance(result, tuple) and len(result) == 2 else result
    )
    if not data:
      continue
    if isinstance(data, bytes):
      data = data.decode("utf-8", errors="replace")

    steps, extra_props = parser(data)
    if steps:
      limited_steps = steps[:limit] if limit > 0 else steps
      summary_data = (
          _build_summary(steps, extra_props) if include_summary else None
      )
      if output_format == "markdown":
        return _format_markdown(limited_steps, summary_data)

      out: dict[str, Any] = {}
      if summary_data is not None:
        out["summary"] = {
            k: v
            for k, v in dataclasses.asdict(summary_data).items()
            if v is not None
        }
      out["step_breakdown"] = [_step_to_dict(s) for s in limited_steps]
      return json.dumps(out, indent=2)

  # Fallback to overview_page.json
  try:
    result = client.fetch(
        tool_name="overview_page.json", session_id=session_id, format="json"
    )
  except _FETCH_EXCEPTIONS as e:
    logging.warning(
        "Error fetching overview_page.json for %s: %s", session_id, e
    )
    result = None

  if result:
    data = (
        result[1] if isinstance(result, tuple) and len(result) == 2 else result
    )
    if data:
      if isinstance(data, bytes):
        data = data.decode("utf-8", errors="replace")
      _, summary_data = _parse_overview_page(data)
      if summary_data:
        if output_format == "markdown":
          return _format_markdown([], summary_data if include_summary else None)

        out = {}
        if include_summary:
          out["summary"] = {
              k: v
              for k, v in dataclasses.asdict(summary_data).items()
              if v is not None
          }
        out["step_breakdown"] = []
        return json.dumps(out, indent=2)

  return _format_error(
      f"No step trace data returned for session {session_id}", output_format
  )
