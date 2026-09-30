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

# pylint: disable=g-import-not-at-top
try:
  from tests.ui.conftest import BrowserErrors
  from tests.ui.journey_capture import settled_tool_url_pattern
  from tests.ui.journey_capture import TOOL_NAME_TO_TAG
  from tests.ui.sxs_diff_engine import make_run_resolver
  from tests.ui.sxs_diff_engine import resolve_scenario_runs
  from tests.ui.ui_helpers import assert_component_geometry
  from tests.ui.ui_helpers import assert_healthy
  from tests.ui.ui_helpers import build_tool_url
  from tests.ui.ui_helpers import select_host
  from tests.ui.ui_helpers import switch_tool
except ImportError:
  from conftest import BrowserErrors
  from journey_capture import settled_tool_url_pattern
  from journey_capture import TOOL_NAME_TO_TAG
  from sxs_diff_engine import make_run_resolver
  from sxs_diff_engine import resolve_scenario_runs
  from ui_helpers import assert_component_geometry
  from ui_helpers import assert_healthy
  from ui_helpers import build_tool_url
  from ui_helpers import select_host
  from ui_helpers import switch_tool


class ActionType(str, enum.Enum):
  """Permitted declarative state transition action types."""

  GOTO = "goto"
  SWITCH_TOOL = "switch_tool"
  SELECT_HOST = "select_host"
  GO_BACK = "go_back"
  GO_FORWARD = "go_forward"


DEFAULT_CATALOG_DIR = pathlib.Path(__file__).resolve().parent / "journeys"
DEFAULT_CATALOG_FILE = "diagnostic_journeys.json"

# Note: Trace viewer legacy iframe type errors.
_UPSTREAM_BASELINE_IGNORED_PATTERNS: tuple[str, ...] = (
    "trace_viewer",
    "streaming trace",
    "Cannot read properties of undefined",
)

# How long a navigation assertion waits for the router to publish the new tool
# in the address bar. XProf loads the tool's data before the router updates the
# query string, so the address bar lags the click by however long the tool
# takes to respond -- for the heavier tools, well past Playwright's 5s default
# on a loaded machine. The assertion is about which tool the app navigated to,
# not about how quickly it got there, so the bound only needs to be long enough
# to distinguish a slow load from a navigation that never happened.
URL_SETTLE_TIMEOUT_MS = 30000


@dataclasses.dataclass(frozen=True)
class JourneyStep:
  """Single state-transition step within a user journey."""

  action: ActionType
  target: str
  expected_selector: str


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


def step_history(page: Page, tool_name: str, forward: bool = False) -> None:
  """Steps browser history back, or forward, until `tool_name` is shown.

  Servers built before SideNav stopped pushing a second history entry per tool
  switch, such as the SxS baseline until that change lands, may stop a single
  step between two tools. Stepping ends once the URL settles on `tool_name` or
  stops changing.

  Args:
    page: Page whose history is stepped.
    tool_name: Display name of the tool the step has to reach.
    forward: Steps forward instead of back.
  """
  tag_pattern = settled_tool_url_pattern(tool_name)
  nav = page.go_forward if forward else page.go_back
  for _ in range(5):
    prev_url = page.url
    nav(wait_until="domcontentloaded")
    page.wait_for_timeout(100)
    if tag_pattern.search(page.url) or page.url == prev_url:
      break
  if forward and not tag_pattern.search(page.url):
    # Those servers also drop the forward entries when they handle back.
    # test_browser_forward_navigation checks forward without this fallback.
    switch_tool(page, tool_name)
  expect(page).to_have_url(tag_pattern, timeout=URL_SETTLE_TIMEOUT_MS)


def dispatch_action(
    page: Page, server_url: str, logdir: str, step: JourneyStep
) -> None:
  """Dispatches the UI navigation action corresponding to the journey step."""
  match step.action:
    case ActionType.SWITCH_TOOL:
      switch_tool(page, step.target)
      expected_tag = TOOL_NAME_TO_TAG.get(
          step.target, step.target.lower().replace(" ", "_")
      )
      expect(page).to_have_url(
          re.compile(rf"tag={re.escape(expected_tag)}"),
          timeout=URL_SETTLE_TIMEOUT_MS,
      )
    case ActionType.SELECT_HOST:
      select_host(page, step.target)
      expect(page).to_have_url(
          re.compile(rf"host={re.escape(step.target)}"),
          timeout=URL_SETTLE_TIMEOUT_MS,
      )
    case ActionType.GO_BACK | ActionType.GO_FORWARD:
      step_history(page, step.target, step.action == ActionType.GO_FORWARD)
    case ActionType.GOTO:
      parts = step.target.split("/", 1)
      run_name = _resolve_run_name(logdir, parts[0])
      tag = parts[1] if len(parts) > 1 else "overview_page"
      dest_path = os.path.join(logdir, run_name)
      dest_url = build_tool_url(server_url, dest_path, run_name, tag)
      page.goto(dest_url, wait_until="domcontentloaded")
      expect(page).to_have_url(
          re.compile(rf"tag={re.escape(tag)}"),
          timeout=URL_SETTLE_TIMEOUT_MS,
      )
    case _:
      raise ValueError(f"Unsupported journey action type: {step.action}")


@pytest.mark.parametrize("scenario", JOURNEY_SCENARIOS, ids=lambda s: s.id)
def test_user_journey_state_machine(
    page: Page,
    server_url: str,
    logdir: str,
    resolve_run: Callable[[str], str],
    open_tool: Callable[..., str],
    browser_errors: BrowserErrors,
    scenario: JourneyScenario,
) -> None:
  """Executes declarative user journeys with invariant sweeps."""
  browser_errors.ignore(*_UPSTREAM_BASELINE_IGNORED_PATTERNS)

  # 1. Mount initial starting waypoint
  scenario = resolve_scenario_runs(scenario, resolve_run, logdir=logdir)
  open_tool(scenario.fixture, scenario.initial_tool)
  expect(page).to_have_url(
      re.compile(rf"tag={re.escape(scenario.initial_tool)}"),
      timeout=URL_SETTLE_TIMEOUT_MS,
  )
  expect(page.locator("body")).to_be_visible()
  assert_healthy(page, context=f"initial load of {scenario.id}")

  # 2. Iterate through declarative state machine steps
  for idx, step in enumerate(scenario.steps, start=1):
    step_context = (
        f"step {idx}/{len(scenario.steps)} ({step.action} -> {step.target})"
    )
    dispatch_action(page, server_url, logdir, step)
    assert_component_geometry(page, step.expected_selector, f"step {step}")
    assert_healthy(page, context=step_context)

  # 3. Verify clean console log state
  browser_errors.assert_clean(f"Scenario {scenario.id}")
