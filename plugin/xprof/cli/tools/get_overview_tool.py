"""Tool to fetch overview page data from XProf."""

import json
import logging
import re
import traceback

from xprof.cli.internal import decorators
from xprof.cli.internal.oss import xprof_client

# Keys to extract from overview_page.json for performance summary.
# These keys supplement keys starting with stat_ or sc_.
PERFORMANCE_SUMMARY_KEYS = frozenset({
    "steptime_ms_average",
    "steptime_ms_standard_deviation",
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


@decorators.cached(expire=86400)
def get_overview(session_id: str, include_command: bool = False) -> str:
  """Gets a comprehensive overview of the XProf session.

  **Use this** to get a unified view of the session's performance summary,
  run environment, and normalized metrics.
  Args:
      session_id: The unique XProf session ID.
      include_command: Whether to include the full command line in the run
        environment (can be noisy).
  Returns:
      A JSON-formatted string containing 'performance_summary',
      'run_environment', and 'normalized_metrics'.
  """
  session_id = str(session_id)
  client = xprof_client.get_client()
  try:
    result = client.fetch(
        tool_name="overview_page.json",
        session_id=session_id,
        format="json",
    )
    if isinstance(result, tuple) and len(result) == 2:
      _, data = result
    else:
      data = result
    if not data:
      return json.dumps(
          dict(error="No overview data returned for the session"), indent=2
      )
    if isinstance(data, bytes):
      data = data.decode("utf-8", errors="replace")
    overview_data = json.loads(data)
    if not isinstance(overview_data, list) or not overview_data:
      return json.dumps(
          dict(error="Unexpected overview page data format"), indent=2
      )
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

    output = {
        "performance_summary": performance_summary,
        "run_environment": run_environment,
    }

    return json.dumps(output, indent=2)
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.exception("Error fetching overview data")
    return json.dumps(
        dict(
            error=f"Error fetching overview data: {e}",
            traceback=traceback.format_exc(),
        ),
        indent=2,
    )
