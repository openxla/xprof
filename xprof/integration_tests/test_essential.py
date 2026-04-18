"""Tests that the xprof server can start and runs AB tests."""

import json
import os
import re
import sys
import urllib.error
import urllib.request
from absl import logging
import pytest

sys.path.append(os.path.dirname(__file__))
import utils  # pylint: disable=g-import-not-at-top


CRITICAL_TOOLS = [
    "overview_page",
    "framework_op_stats",
    "graph_viewer",
    "hlo_stats",
    "input_pipeline_analyzer",
    "memory_profile",
    "memory_viewer",
    "op_profile",
    "roofline_model",
    "trace_viewer@",
]

HLO_TOOLS = [
    "memory_viewer",
    "graph_viewer",
]

IGNORE_MAP = {
    "memory_profile": [
        "memoryProfilePerAllocator.*.memoryProfileSnapshots",
    ]
}


def _sanitize_filename(name):
  """Replaces non-filesystem safe characters with underscores."""
  if not name:
    return ""
  # Replace anything that isn't alphanumeric, -, or _ with _
  return re.sub(r"[^a-zA-Z0-9_\-]", "_", name)


def test_server_root_healthy(server_url):
  """Smoke test: ensures the server root returns 200 OK."""
  with urllib.request.urlopen(server_url, timeout=5) as response:
    assert response.status == 200


def test_runs_listing(server_url):
  """Verifies the /runs endpoint matches the expected historical output."""
  url = f"{server_url}/data/plugin/profile/runs"
  data = utils.fetch_json(url)
  utils.verify_golden(data, "runs.json")


@pytest.mark.parametrize("run_name", utils.get_available_runs())
@pytest.mark.parametrize("tool_name", CRITICAL_TOOLS)
def test_tool_data_availability(server_url, run_name, tool_name):
  """Verifies that a critical tool loads correctly for ALL hosts in a given run."""
  tools_url = f"{server_url}/data/plugin/profile/run_tools?run={run_name}"
  tools = utils.fetch_json(tools_url)

  if tool_name not in tools:
    pytest.skip(f"Tool '{tool_name}' not applicable for run '{run_name}'")

  # 1. Fetch Hosts
  hosts_url = (
      f"{server_url}/data/plugin/profile/hosts?run={run_name}&tag={tool_name}"
  )
  hosts = utils.fetch_json(hosts_url)
  assert hosts, f"No hosts found for tool {tool_name} in run {run_name}"

  for host_entry in hosts:
    host = host_entry["hostname"]

    # 2. Determine Sub-Tests (Modules/HLOs)
    # If it's an HLO tool, we iterate over every module.
    # If not, we run once with module=None.
    modules_to_test = [None]

    if tool_name in HLO_TOOLS:
      hlos_url = (
          f"{server_url}/data/plugin/profile/module_list?run={run_name}"
          f"&tag={tool_name}&host={host}&graph_type=xla"
      )
      # The endpoint returns a text CSV list
      text_response = utils.fetch_text(hlos_url)
      if text_response:
        modules_to_test = text_response.split(",")
        logging.info(
            "Found %d HLOs for %s on %s",
            len(modules_to_test), tool_name, host
        )
      else:
        logging.warning("Expected HLOs for %s but found none.", tool_name)
        continue

    # 3. Run Tests
    for module in modules_to_test:
      logging.info("Verifying %s for host %s (Module: %s)",
                   tool_name, host, module)

      if tool_name == "memory_viewer":
        # memory_viewer requires specific memory_space arg usually
        data_url = (
            f"{server_url}/data/plugin/profile/data?run={run_name}"
            f"&tag={tool_name}&host={host}&module_name={module}&memory_space=0"
        )
      elif tool_name == "graph_viewer":
        # Preserving your specific graph_viewer parameters
        data_url = (
            f"{server_url}/data/plugin/profile/data?run={run_name}"
            f"&tag=op_profile&host={host}&op_profile_limit=50&use_xplane=1"
        )
        if module:
          data_url += f"&module_name={module}"
      else:
        # Standard tools
        data_url = f"{server_url}/data/plugin/profile/data?run={run_name}&tag={tool_name}&host={host}"

      # 4. Fetch and Verify
      try:
        data = utils.fetch_json(data_url)
      except (urllib.error.URLError, json.JSONDecodeError) as e:
        pytest.fail(
            f"Failed to fetch data for {tool_name} (Host: {host}, Module:"
            f" {module}): {e}"
        )

      # Construct unique filename
      # e.g., data_run1_memory_viewer_host1_hlo_module_123.json
      safe_host = _sanitize_filename(host)
      golden_filename = f"data_{run_name}_{tool_name}_{safe_host}"

      if module:
        safe_module = _sanitize_filename(module)
        golden_filename += f"_{safe_module}"

      golden_filename += ".json"

      tool_ignores = IGNORE_MAP.get(tool_name, [])
      utils.verify_golden(data, golden_filename, ignore_paths=tool_ignores)
