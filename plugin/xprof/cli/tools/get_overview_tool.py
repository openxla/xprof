"""Tool to fetch overview page data from XProf."""

from collections.abc import Iterable
import json
import logging
import math
import os
import pathlib
import re
import statistics
from typing import Any

from xprof.cli.internal import decorators
from xprof.cli.internal import trace_selection
from xprof.cli.internal.oss import xprof_client
from xprof.cli.tools import get_roofline_model_tool
from xprof.protobuf import op_stats_pb2

# Mirrors `kSparseCoreIndexStart` in
# `third_party/xprof/convert/op_stats_to_input_pipeline_analysis.cc` and
# `kMaxDevicesPerHost` in `third_party/xprof/convert/op_stats_combiner.h`.
_SPARSE_CORE_INDEX_START = 1_000_000
_MAX_DEVICES_PER_HOST = 1000

# Keys to extract from overview_page.json for performance summary.
# These keys supplement keys starting with stat_ or sc_.
PERFORMANCE_SUMMARY_KEYS = frozenset({
    "steptime_ms_average",
    "steptime_ms_standard_deviation",
    "active_steptime_ms_average",
    "tc_active_ms_average",
    "sc_active_ms_average",
    "tc_idle_ms_average",
    "tc_infeed_ms_average",
    "tc_outfeed_ms_average",
    "host_transfer_ms_average",
    "sc_step_time_ms_average",
    "sc_idle_ms_average",
    "sc_infeed_ms_average",
    "sc_outfeed_ms_average",
    "mxu_utilization_percent",
    "flop_rate_utilization_relative_to_roofline",
    "device_duty_cycle_percent",
    "memory_bw_utilization_relative_to_hw_limit",
    "program_goodput_percent",
    "host_tf_op_percent",
    "device_tf_op_percent",
    "host_op_time_eager_percent",
    "device_op_time_eager_percent",
    "device_idle_time_percent",
    "hbm_bw_utilization_percent",
    "host_idle_time_percent",
})

# Keys to extract from overview_page.json for run environment.
# These keys supplement keys starting with run_.
RUN_ENVIRONMENT_KEYS = frozenset({
    "is_training",
    "profile_start_time",
    "profile_duration_ms",
    "host_count",
    "task_count",
    "device_type",
    "device_core_count",
    "change_list",
    "build_time",
    "build_target",
})


def _populate_roofline_fallback(
    client: xprof_client.CachedXprofClient,
    session_id: str,
    performance_summary: dict[str, Any],
) -> None:
  """Plumbs roofline metrics into performance_summary if missing or 0.0%."""
  flop_roofline = performance_summary.get(
      "flop_rate_utilization_relative_to_roofline"
  )
  mem_bw = performance_summary.get("memory_bw_utilization_relative_to_hw_limit")
  hbm_bw = performance_summary.get("hbm_bw_utilization_percent")

  needs_flop = not flop_roofline or flop_roofline in (
      "0.0%",
      "0%",
      "0.0",
      0,
      0.0,
  )
  needs_mem = (not mem_bw or mem_bw in ("0.0%", "0%", "0.0", 0, 0.0)) and (
      not hbm_bw or hbm_bw in ("0.0%", "0%", "0.0", 0, 0.0)
  )

  try:
    result = client.fetch(
        tool_name="roofline_model.json",
        session_id=session_id,
    )
    if not result or (isinstance(result, tuple) and not result[1]):
      result = client.fetch(
          tool_name="roofline_model",
          session_id=session_id,
      )
    if isinstance(result, tuple) and len(result) == 2:
      _, data = result
    else:
      data = result

    if not data:
      return

    if isinstance(data, bytes):
      data = data.decode("utf-8", errors="replace")

    roofline_data = json.loads(data)
    if not isinstance(roofline_data, list) or not roofline_data:
      return

    table_data = roofline_data[0]
    cols = [col.get("id", "") for col in table_data.get("cols", [])]
    rows = table_data.get("rows", [])
    if not rows:
      return

    row_cells = rows[0].get("c", [])
    vals = [c.get("v") if isinstance(c, dict) else c for c in row_cells]
    prog_dict = dict(zip(cols, vals))

    def safe_float(val: Any) -> float | None:
      if val is None:
        return None
      try:
        return float(val)
      except (ValueError, TypeError):
        return None

    roofline_eff = safe_float(prog_dict.get("roofline_efficiency"))
    compute_eff = safe_float(prog_dict.get("compute_efficiency"))
    max_mem_bw = safe_float(prog_dict.get("max_mem_bw_utilization"))
    bound_by = prog_dict.get("bound_by")
    operational_intensity = safe_float(prog_dict.get("operational_intensity"))

    if needs_flop and roofline_eff is not None:
      roofline_str = f"{roofline_eff * 100.0:.2f}%"
      performance_summary["roofline_efficiency_percent"] = roofline_str
      performance_summary["flop_rate_utilization_relative_to_roofline"] = (
          roofline_str
      )

    if needs_flop and compute_eff is not None:
      performance_summary["compute_efficiency_percent"] = (
          f"{compute_eff * 100.0:.2f}%"
      )

    if needs_mem and max_mem_bw is not None:
      max_mem_bw_str = f"{max_mem_bw * 100.0:.2f}%"
      performance_summary["max_mem_bw_utilization_percent"] = max_mem_bw_str
      performance_summary["memory_bw_utilization_relative_to_hw_limit"] = (
          max_mem_bw_str
      )
      performance_summary["hbm_bw_utilization_percent"] = max_mem_bw_str

    if (needs_flop or needs_mem) and bound_by:
      performance_summary["bound_by"] = bound_by
    if (needs_flop or needs_mem) and operational_intensity is not None:
      performance_summary["operational_intensity_flop_per_byte"] = round(
          operational_intensity, 4
      )

    custom_call_share_pct = 0.0
    prog_cells = rows[0].get("c", []) if rows else []
    prog_vals = [
        cell.get("v") if isinstance(cell, dict) else cell for cell in prog_cells
    ]
    prog_dict = dict(zip(cols, prog_vals))
    prog_total = safe_float(prog_dict.get("total_self_time_percent"))
    percent_scale = (
        100.0 if (prog_total is None or prog_total <= 1.0 + 1e-6) else 1.0
    )
    seen_custom_call_keys = set()
    for row in rows[1:]:
      row_cells = row.get("c", [])
      row_vals = [
          cell.get("v") if isinstance(cell, dict) else cell
          for cell in row_cells
      ]
      row_dict = dict(zip(cols, row_vals))
      op_name = str(row_dict.get("operation") or row_dict.get("hlo_name", ""))
      op_category = str(
          row_dict.get("category") or row_dict.get("hlo_category", "")
      )
      op_key = (int(safe_float(row_dict.get("rank")) or 0), op_name)
      if op_key in seen_custom_call_keys:
        continue
      if op_name.startswith("custom-call") or op_category.lower() in (
          "custom-call",
          "custom_call",
      ):
        seen_custom_call_keys.add(op_key)
        val = safe_float(row_dict.get("total_self_time_percent"))
        if val is not None:
          custom_call_share_pct += val * percent_scale
    custom_call_share_pct = min(round(custom_call_share_pct, 2), 100.0)
    if custom_call_share_pct >= 10.0:
      performance_summary["custom_call_share_pct"] = custom_call_share_pct
      performance_summary["custom_call_warning"] = (
          get_roofline_model_tool.format_custom_call_warning(
              custom_call_share_pct
          )
      )
  except (KeyError, ValueError, TypeError, json.JSONDecodeError, OSError):
    logging.exception("Failed to fetch roofline fallback")


def _accumulate_step_host_maxes(
    steps: Iterable[Iterable[tuple[int, str, float]]],
) -> dict[tuple[int, str], list[float]]:
  """Accumulates the per-step maximum duration in ms for each host across steps.

  Args:
    steps: For each step, an iterable of `(core_id, hostname, duration_ms)`
      tuples across cores.

  Returns:
    Mapping from `(src_host_id, hostname)` to the list of per-step maximum
    durations in milliseconds.
  """
  by_group: dict[tuple[int, str], list[float]] = {}
  for step_records in steps:
    per_host_step_max: dict[tuple[int, str], float] = {}
    for core_id, raw_host, dur_ms in step_records:
      if (
          core_id >= _SPARSE_CORE_INDEX_START
          or not math.isfinite(dur_ms)
          or dur_ms <= 0
      ):
        continue
      src_host_id = max(core_id, 0) // _MAX_DEVICES_PER_HOST
      group_key = (src_host_id, raw_host)
      if dur_ms > per_host_step_max.get(group_key, 0.0):
        per_host_step_max[group_key] = dur_ms
    for group_key, dur_ms in per_host_step_max.items():
      by_group.setdefault(group_key, []).append(dur_ms)
  return by_group


def _format_per_host_summaries(
    by_group: dict[tuple[int, str], list[float]],
    local_stems: list[str],
) -> dict[str, dict[str, str]]:
  """Formats per-host step-time lists into `steptime_ms_*` summary dicts."""
  groups = sorted(by_group)
  if len(groups) < 2:
    return {}

  use_local_stems = (
      bool(local_stems)
      and len({g[0] for g in groups}) == len(groups)
      and all(0 <= g[0] < len(local_stems) for g in groups)
  )
  distinct_raw_hosts = len({g[1] for g in groups if g[1]}) == len(groups)

  per_host: dict[str, dict[str, str]] = {}
  for src_host_id, raw_host in groups:
    vals = by_group[(src_host_id, raw_host)]
    if not vals:
      continue
    if use_local_stems:
      label = local_stems[src_host_id]
    elif distinct_raw_hosts:
      label = raw_host
    elif raw_host:
      label = f"{raw_host}_{src_host_id}"
    else:
      label = f"host_{src_host_id}"
    avg_ms = statistics.mean(vals)
    std_ms = statistics.stdev(vals) if len(vals) > 1 else 0.0
    per_host[label] = {
        "steptime_ms_average": f"{avg_ms:.2f}",
        "steptime_ms_standard_deviation": f"{std_ms:.2f}",
        "steptime_ms_minimum": f"{min(vals):.2f}",
        "steptime_ms_maximum": f"{max(vals):.2f}",
    }
  return per_host if len(per_host) >= 2 else {}


def _extract_per_host_from_op_stats_file(
    client: xprof_client.CachedXprofClient,
    session_id: str,
    local_paths: list[str],
    local_stems: list[str],
) -> dict[str, dict[str, str]]:
  """Reads per-host step-time summaries from an owned `ALL_HOSTS.op_stats_v2.pb`."""
  if len(local_paths) < 2 or len(local_stems) < 2:
    return {}
  try:
    raw_bytes: bytes | None = None
    read_owned_fn = getattr(client, "read_owned_op_stats_bytes", None)
    if callable(read_owned_fn):
      candidate = read_owned_fn(session_id, xspace_paths=local_paths)
      if isinstance(candidate, bytes):
        raw_bytes = candidate
    if raw_bytes is None:
      get_owned_fn = getattr(client, "get_owned_op_stats_path", None)
      if callable(get_owned_fn):
        op_stats_path = get_owned_fn(session_id, xspace_paths=local_paths)
        if isinstance(op_stats_path, (str, os.PathLike)):
          op_stats_file = pathlib.Path(op_stats_path)
          if op_stats_file.is_file():
            raw_bytes = op_stats_file.read_bytes()
    if not raw_bytes:
      return {}
    op_stats = op_stats_pb2.OpStats()
    op_stats.ParseFromString(raw_bytes)
    env_hosts = op_stats.run_environment.host_count
    if env_hosts > 0 and env_hosts != len(local_stems):
      return {}
    core_details = op_stats.core_id_to_details

    def iter_op_stats_steps() -> Iterable[Iterable[tuple[int, str, float]]]:
      for seq in op_stats.step_db.step_sequence:
        yield (
            (
                int(core_id),
                str(core_details[core_id].hostname or "").strip()
                if core_id in core_details
                else "",
                float(step_info.duration_ps) / 1e9,
            )
            for core_id, step_info in seq.step_info_per_core.items()
            if step_info.duration_ps > 0
        )

    by_group = _accumulate_step_host_maxes(iter_op_stats_steps())
    if len(by_group) != len(local_stems) or {g[0] for g in by_group} != set(
        range(len(local_stems))
    ):
      return {}
    return _format_per_host_summaries(by_group, local_stems)
  except Exception:  # pylint: disable=broad-exception-caught
    logging.exception("Failed to read ALL_HOSTS.op_stats_v2.pb")
    return {}


def _extract_per_host_step_time_ms(
    client: xprof_client.CachedXprofClient,
    session_id: str,
    local_stems: list[str],
    local_paths: list[str] | None = None,
) -> dict[str, dict[str, str]]:
  """Computes per-host step-time summaries from OpStats or pod_viewer records.

  Matches C++ `ComputeStepTimeSummaryInMs` in
  `op_stats_to_input_pipeline_analysis`: for each step, takes the maximum
  step duration across all non-SparseCore cores belonging to each host, then
  computes average, sample standard deviation, minimum, and maximum across
  steps.

  Args:
    client: The XProf client.
    session_id: The XProf session ID.
    local_stems: Trace stems in `SessionSnapshot` merge order for local runs.
    local_paths: Selected local trace file paths for local runs.

  Returns:
    Mapping from host or trace stem to per-host `steptime_ms_*` strings, or an
    empty dict when fewer than 2 hosts have step records.
  """
  if local_paths:
    from_op_stats = _extract_per_host_from_op_stats_file(
        client, session_id, local_paths, local_stems
    )
    if from_op_stats:
      return from_op_stats

  try:
    result = client.fetch(
        tool_name="pod_viewer.json",
        session_id=session_id,
    )
    if not result or (isinstance(result, tuple) and not result[1]):
      result = client.fetch(
          tool_name="pod_viewer",
          session_id=session_id,
      )
    if isinstance(result, tuple) and len(result) == 2:
      _, data = result
    else:
      data = result
    if not data:
      return {}
    if isinstance(data, bytes):
      data = data.decode("utf-8", errors="replace")
    pod_data = json.loads(data)
    if not isinstance(pod_data, dict):
      return {}
    pod_map = pod_data.get("podStatsSequence", {}).get("podStatsMap", [])
    if not isinstance(pod_map, list) or not pod_map:
      return {}

    def iter_pod_viewer_steps() -> Iterable[list[tuple[int, str, float]]]:
      for entry in pod_map:
        if not isinstance(entry, dict):
          continue
        cores = entry.get("podStatsPerCore", {})
        if not isinstance(cores, dict):
          continue
        step_records: list[tuple[int, str, float]] = []
        for core_id_str, record in cores.items():
          if not isinstance(record, dict):
            continue
          try:
            core_id = int(core_id_str)
          except (ValueError, TypeError):
            core_id = 0
          raw_host = str(
              record.get("hostName") or record.get("host_name") or ""
          ).strip()
          try:
            dur_us = float(
                record.get(
                    "totalDurationUs", record.get("total_duration_us", 0)
                )
            )
          except (ValueError, TypeError):
            continue
          step_records.append((core_id, raw_host, dur_us / 1000.0))
        yield step_records

    by_group = _accumulate_step_host_maxes(iter_pod_viewer_steps())
    return _format_per_host_summaries(by_group, local_stems)
  except (KeyError, ValueError, TypeError, json.JSONDecodeError, OSError):
    logging.exception("Failed to extract per-host step time from pod_viewer")
    return {}
  except Exception:  # pylint: disable=broad-exception-caught
    logging.exception("Unexpected error fetching pod_viewer for %s", session_id)
    return {}


def _populate_multi_host_step_time(
    client: xprof_client.CachedXprofClient,
    session_id: str,
    performance_summary: dict[str, Any],
    run_environment: dict[str, Any],
) -> None:
  """Annotates multi-host step-time scope and per-host step-time breakdown."""
  try:
    host_count = int(run_environment.get("host_count", 1))
  except (ValueError, TypeError):
    host_count = 1

  is_local = False
  local_paths: list[str] = []
  local_stems: list[str] = []
  is_local_fn = getattr(client, "is_local_session", None)
  select_paths_fn = getattr(client, "select_paths", None)
  if callable(is_local_fn) and callable(select_paths_fn):
    try:
      is_local_res = is_local_fn(session_id)
      if isinstance(is_local_res, bool) and is_local_res:
        is_local = True
        raw_paths = list(select_paths_fn(session_id))
        if all(isinstance(p, (str, os.PathLike)) for p in raw_paths):
          local_paths = [str(p) for p in raw_paths]
          local_stems = [trace_selection.trace_stem(p) for p in raw_paths]
    except (OSError, ValueError, TypeError):
      local_paths = []
      local_stems = []

  if is_local:
    if len(local_stems) < 2:
      return
    num_hosts = len(local_stems)
  else:
    if host_count < 2:
      return
    num_hosts = host_count

  per_host = _extract_per_host_step_time_ms(
      client, session_id, local_stems, local_paths=local_paths
  )
  if per_host:
    performance_summary["per_host_step_time_ms"] = per_host
    num_hosts = max(num_hosts, len(per_host))

  if (
      "steptime_ms_average" not in performance_summary
      and "steptime_ms_standard_deviation" not in performance_summary
      and not per_host
  ):
    return

  performance_summary["step_time_scope"] = "per_step_max_across_hosts"
  performance_summary["step_time_warning"] = (
      trace_selection.format_step_time_warning(
          trace_selection.traits_for("get_overview").step_time_fields,
          num_hosts=num_hosts,
      )
  )


@decorators.cached(expire=86400)
def get_overview(
    session_id: str,
    include_command: bool = False,
    bypass_cache: bool = False,
) -> str:
  """Gets a comprehensive overview of the XProf session.

  **Use this** to get a unified view of the session's performance summary,
  run environment, and normalized metrics.
  Args:
      session_id: The unique XProf session ID.
      include_command: Whether to include the full command line in the run
        environment (can be noisy).
      bypass_cache: Whether to bypass cache and recompute metrics.

  Returns:
      A JSON-formatted string containing 'performance_summary',
      'run_environment', and 'normalized_metrics'.
  """
  session_id = str(session_id)
  client = xprof_client.get_client()
  try:
    result = client.fetch(
        tool_name="overview_page",
        session_id=session_id,
        format="json",
        bypass_cache=bypass_cache,
    )
    if not result or (isinstance(result, tuple) and not result[1]):
      result = client.fetch(
          tool_name="overview_page.json",
          session_id=session_id,
          format="json",
          bypass_cache=bypass_cache,
      )
    if isinstance(result, tuple) and len(result) == 2:
      _, data = result
    else:
      data = result
    if not data:
      raise FileNotFoundError("No overview data returned for the session")
    if isinstance(data, bytes):
      data = data.decode("utf-8", errors="replace")
    overview_data = json.loads(data)
    if not isinstance(overview_data, list) or not overview_data:
      raise ValueError("Unexpected overview page data format")
    # Aggregate p_dict from all sections
    all_p_dict = {}
    for section in overview_data:
      all_p_dict.update(section.get("p", {}))
    # Look for run environment table in overview_data sections to extract
    # hostname, bns and command.
    hostname, bns, command = None, None, None
    for section in overview_data:
      if "cols" in section and "rows" in section:
        cols = section.get("cols", [])
        rows = section.get("rows", [])
        col_map = {col.get("id"): i for i, col in enumerate(cols)}
        # If host_id is in columns, assume this is run environment table
        if "host_id" in col_map and rows and rows[0].get("c"):
          row_values = [c.get("v") for c in rows[0]["c"]]
          if (index := col_map.get("host_id")) is not None:
            hostname = row_values[index]
          if (index := col_map.get("bns_address")) is not None:
            bns = row_values[index]
          if (index := col_map.get("command_line")) is not None:
            command = row_values[index]
          break
    # Extract Performance Summary
    performance_summary = {}
    for key, val in all_p_dict.items():
      if (
          key.startswith("stat_")
          or key.startswith("sc_")
          or key in PERFORMANCE_SUMMARY_KEYS
      ):
        performance_summary[key] = val

    _populate_roofline_fallback(client, session_id, performance_summary)

    # Extract Run Environment
    run_environment = {}
    for key, val in all_p_dict.items():
      if key.startswith("run_") or key in RUN_ENVIRONMENT_KEYS:
        if key == "run_command" and not include_command:
          continue
        run_environment[key] = val
    if hostname:
      run_environment["hostname"] = hostname
    if bns:
      run_environment["bns"] = bns
    if include_command and command and "run_command" not in run_environment:
      run_environment["run_command"] = command
    if "device_type" in all_p_dict:
      run_environment["device_type"] = all_p_dict["device_type"]
    if command:
      match = re.search(
          r"--streamz_default_root_labels=\S*xmanager:int:(\d+)", command
      )
      if match:
        run_environment["xid"] = match.group(1)

    _populate_multi_host_step_time(
        client, session_id, performance_summary, run_environment
    )

    output = {
        "performance_summary": performance_summary,
        "run_environment": run_environment,
    }

    return json.dumps(output, indent=2)
  except (FileNotFoundError, ValueError):
    raise
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.exception("Error fetching overview data for session %s", session_id)
    raise RuntimeError(f"Error fetching overview data: {e}") from e
