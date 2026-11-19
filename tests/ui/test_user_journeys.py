# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Declarative User Journey State Machine Test Engine for OpenXLA XProf."""

from collections.abc import Callable
import dataclasses
import enum
import json
import os
import pathlib
import re

from playwright.sync_api import expect
from playwright.sync_api import Page
import pytest

from tests.ui.conftest import BrowserErrors
from tests.ui.invariants import KNOWN_UNWRAPPED_ROW_SELECTORS
from tests.ui.journey_capture import settled_tool_url_pattern
from tests.ui.sxs_diff_engine import make_run_resolver
from tests.ui.sxs_diff_engine import resolve_scenario_runs
from tests.ui.ui_helpers import assert_component_geometry
from tests.ui.ui_helpers import assert_healthy
from tests.ui.ui_helpers import build_tool_url
from tests.ui.ui_helpers import expand_op_profile_row
from tests.ui.ui_helpers import filter_table_rows
from tests.ui.ui_helpers import plot_graph_node
from tests.ui.ui_helpers import select_category_filter
from tests.ui.ui_helpers import select_host
from tests.ui.ui_helpers import select_memory_id
from tests.ui.ui_helpers import select_module
from tests.ui.ui_helpers import select_op_profile_group_by
from tests.ui.ui_helpers import select_session
from tests.ui.ui_helpers import sort_table_column
from tests.ui.ui_helpers import step_history
from tests.ui.ui_helpers import switch_tool
from tests.ui.ui_helpers import URL_SETTLE_TIMEOUT_MS


class ActionType(str, enum.Enum):
  """Permitted declarative state transition action types."""

  GOTO = "goto"
  SWITCH_TOOL = "switch_tool"
  SELECT_SESSION = "select_session"
  SELECT_HOST = "select_host"
  SELECT_MODULE = "select_module"
  SELECT_OP_PROFILE_GROUP_BY = "select_op_profile_group_by"
  EXPAND_OP_PROFILE_ROW = "expand_op_profile_row"
  PLOT_GRAPH_NODE = "plot_graph_node"
  SELECT_MEMORY_ID = "select_memory_id"
  SELECT_CATEGORY_FILTER = "select_category_filter"
  FILTER_TABLE_ROWS = "filter_table_rows"
  SORT_TABLE_COLUMN = "sort_table_column"
  GO_BACK = "go_back"
  GO_FORWARD = "go_forward"


DEFAULT_CATALOG_DIR = pathlib.Path(__file__).resolve().parent / "journeys"
DEFAULT_CATALOG_FILE = "diagnostic_journeys.json"


@dataclasses.dataclass(frozen=True)
class JourneyStep:
  """Single state-transition step within a user journey."""

  action: ActionType
  target: str
  expected_selector: str
  expected_url_substring: str = ""


@dataclasses.dataclass(frozen=True)
class JourneyScenario:
  """Declarative definition of an end-to-end user diagnostic journey."""

  id: str
  fixture: str
  initial_tool: str
  steps: tuple[JourneyStep, ...]


def load_journey_scenarios(
    catalog_dir: pathlib.Path | None = None,
    filename: str = DEFAULT_CATALOG_FILE,
) -> list[JourneyScenario]:
  """Loads journey scenarios from the canonical JSON catalog file.

  Args:
    catalog_dir: Optional directory containing catalog files.
    filename: Name of the JSON catalog file.

  Returns:
    A list of validated JourneyScenario instances.

  Raises:
    FileNotFoundError: If the catalog file does not exist.
    ValueError: If the catalog is empty or malformed.
  """
  dir_path = catalog_dir or DEFAULT_CATALOG_DIR
  catalog_file = dir_path / os.environ.get("XPROF_JOURNEY_CATALOG", filename)

  if not catalog_file.is_file():
    raise FileNotFoundError(f"Journey catalog file not found: {catalog_file}")

  try:
    data = json.loads(catalog_file.read_text(encoding="utf-8"))
  except (json.JSONDecodeError, OSError) as err:
    raise ValueError(f"Failed to read catalog {catalog_file}: {err}") from err

  if not isinstance(data, list) or not data:
    raise ValueError(f"Catalog {catalog_file} must contain a non-empty list.")

  scenarios: list[JourneyScenario] = []
  for item in data:
    steps = tuple(
        JourneyStep(
            action=ActionType(step["action"]),
            target=step["target"],
            expected_selector=step["expected_selector"],
            expected_url_substring=step.get("expected_url_substring", ""),
        )
        for step in item.get("steps", [])
    )
    scenarios.append(
        JourneyScenario(
            id=item["id"],
            fixture=item["fixture"],
            initial_tool=item["initial_tool"],
            steps=steps,
        )
    )

  return scenarios


JOURNEY_SCENARIOS: list[JourneyScenario] = load_journey_scenarios()


def _resolve_run_name(logdir: str, run_name: str) -> str:
  """Maps a run name onto one in the logdir, keeping it when none matches."""
  try:
    return make_run_resolver(logdir)(run_name)
  except OSError:
    return run_name


def _wait_for_tool_data(page: Page, tool: str) -> None:
  """Waits until the router shows `tool` and the app stops loading its data.

  Leaving a tool before its data arrives can leave the app-wide loading flag
  set, and the main page then keeps every later tool collapsed. The URL check
  comes first because the flag only goes up once the router mounts the tool.

  Args:
    page: Page that was just pointed at `tool`.
    tool: Tag or display name of the tool.
  """
  expect(page).to_have_url(
      settled_tool_url_pattern(tool), timeout=URL_SETTLE_TIMEOUT_MS
  )
  expect(
      page.locator(
          "main-page > mat-sidenav-container > mat-sidenav-content"
          " > div.full-height"
      )
  ).to_be_attached(timeout=URL_SETTLE_TIMEOUT_MS)


def dispatch_action(
    page: Page, server_url: str, logdir: str, step: JourneyStep
) -> None:
  """Dispatches the UI navigation action corresponding to the journey step."""
  match step.action:
    case ActionType.SWITCH_TOOL:
      switch_tool(page, step.target)
      _wait_for_tool_data(page, step.target)
    case ActionType.SELECT_SESSION:
      run_name = _resolve_run_name(logdir, step.target)
      select_session(page, run_name)
    case ActionType.SELECT_HOST:
      select_host(page, step.target)
    case ActionType.SELECT_MODULE:
      select_module(page, step.target)
    case ActionType.SELECT_OP_PROFILE_GROUP_BY:
      select_op_profile_group_by(page, step.target)
    case ActionType.EXPAND_OP_PROFILE_ROW:
      expand_op_profile_row(page, step.target)
    case ActionType.PLOT_GRAPH_NODE:
      plot_graph_node(page, step.target)
    case ActionType.SELECT_MEMORY_ID:
      select_memory_id(page, step.target)
    case ActionType.SELECT_CATEGORY_FILTER:
      select_category_filter(page, step.target)
    case ActionType.FILTER_TABLE_ROWS:
      filter_table_rows(page, step.target)
    case ActionType.SORT_TABLE_COLUMN:
      sort_table_column(page, step.target)
    case ActionType.GO_BACK | ActionType.GO_FORWARD:
      step_history(page, step.target, step.action == ActionType.GO_FORWARD)
      _wait_for_tool_data(page, step.target)
    case ActionType.GOTO:
      parts = step.target.split("/", 1)
      run_name = _resolve_run_name(logdir, parts[0])
      tag = parts[1] if len(parts) > 1 else "overview_page"
      dest_path = os.path.join(logdir, run_name)
      dest_url = build_tool_url(server_url, dest_path, run_name, tag)
      page.goto(dest_url, wait_until="domcontentloaded")
      _wait_for_tool_data(page, tag)
    case _:
      raise ValueError(f"Unsupported journey action type: {step.action}")


@pytest.mark.parametrize("scenario", JOURNEY_SCENARIOS, ids=lambda s: s.id)
def test_user_journey_state_machine(
    page: Page,
    server_url: str,
    synthesized_logdir: str,
    open_tool: Callable[..., str],
    browser_errors: BrowserErrors,
    scenario: JourneyScenario,
) -> None:
  """Executes declarative user journeys with invariant sweeps."""
  # 1. Mount initial starting waypoint
  scenario = resolve_scenario_runs(
      scenario,
      make_run_resolver(synthesized_logdir),
      logdir=synthesized_logdir,
  )
  open_tool(scenario.fixture, scenario.initial_tool)
  _wait_for_tool_data(page, scenario.initial_tool)
  expect(page.locator("body")).to_be_visible()
  assert_healthy(page, context=f"initial load of {scenario.id}")

  # 2. Iterate through declarative state machine steps
  for idx, step in enumerate(scenario.steps, start=1):
    step_context = (
        f"step {idx}/{len(scenario.steps)} ({step.action} -> {step.target})"
    )
    dispatch_action(page, server_url, synthesized_logdir, step)
    if step.expected_url_substring:
      expect(page).to_have_url(
          re.compile(re.escape(step.expected_url_substring)),
          timeout=URL_SETTLE_TIMEOUT_MS,
      )
    assert_component_geometry(page, step.expected_selector, f"step {step}")
    overflow_allow = None
    if step.action == ActionType.GO_BACK and step.target == "Overview Page":
      # Upstream race: after popstate from a tool that closed the sidenav,
      # step-time-graph can draw before the sidenav reopens and never re-fits.
      overflow_allow = (
          *KNOWN_UNWRAPPED_ROW_SELECTORS,
          "overview-page step-time-graph",
      )
    assert_healthy(
        page,
        context=step_context,
        allowed_overflow_selectors=overflow_allow,
    )

  # 3. Verify clean console log state
  browser_errors.assert_clean(f"Scenario {scenario.id}")
