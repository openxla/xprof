"""Declarative User Journey State Machine Test Engine for OpenXLA XProf."""

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
  from tests.ui.ui_helpers import assert_healthy
  from tests.ui.ui_helpers import build_tool_url
  from tests.ui.ui_helpers import select_host
  from tests.ui.ui_helpers import switch_tool
except ImportError:
  from conftest import BrowserErrors
  from journey_capture import settled_tool_url_pattern
  from journey_capture import TOOL_NAME_TO_TAG
  from sxs_diff_engine import make_run_resolver
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

_UPSTREAM_BASELINE_IGNORED_PATTERNS: tuple[str, ...] = (
    "trace_viewer",
    "streaming trace",
    "Cannot read properties of undefined",
    "split is not a function",
)

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
  """Resolves a run name against logdir, trying hyphen/underscore variants."""
  try:
    return make_run_resolver(logdir)(run_name)
  except (FileNotFoundError, OSError):
    return run_name


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
      tag_pattern = settled_tool_url_pattern(step.target)
      is_back = step.action == ActionType.GO_BACK
      nav = page.go_back if is_back else page.go_forward
      for _ in range(5):
        prev_url = page.url
        nav(wait_until="domcontentloaded")
        page.wait_for_timeout(100)
        if tag_pattern.search(page.url) or page.url == prev_url:
          break
      expect(page).to_have_url(tag_pattern, timeout=URL_SETTLE_TIMEOUT_MS)
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


def _assert_component_geometry(
    page: Page, selector: str, step: JourneyStep
) -> None:
  """Asserts that the component is mounted with positive geometry and rendered child content."""
  comp = page.locator(f":is({selector}):visible").first
  expect(comp).to_be_visible(timeout=20000)
  bbox = comp.bounding_box()
  assert (
      bbox is not None and bbox["width"] > 0 and bbox["height"] > 0
  ), f"Component {selector} collapsed at step {step}"
  child = comp.locator(
      "svg, canvas, table, mat-card, .mat-mdc-card, iframe, .table, :scope > *"
  ).first
  expect(child).to_be_visible(timeout=20000)
  child_bbox = child.bounding_box()
  assert (
      child_bbox is not None
      and child_bbox["width"] > 0
      and child_bbox["height"] > 0
  ), f"Component {selector} child content collapsed at step {step}"


@pytest.mark.parametrize("scenario", JOURNEY_SCENARIOS, ids=lambda s: s.id)
def test_user_journey_state_machine(
    page: Page,
    server_url: str,
    logdir: str,
    browser_errors: BrowserErrors,
    scenario: JourneyScenario,
) -> None:
  """Executes declarative user journeys with invariant sweeps."""
  browser_errors.ignore(*_UPSTREAM_BASELINE_IGNORED_PATTERNS)

  # 1. Mount initial starting waypoint
  fixture_name = _resolve_run_name(logdir, scenario.fixture)
  session_path = os.path.join(logdir, fixture_name)
  if not os.path.exists(session_path):
    pytest.skip(f"Fixture '{scenario.fixture}' not present in logdir")
  url = build_tool_url(
      server_url, session_path, fixture_name, scenario.initial_tool
  )
  page.goto(url, wait_until="domcontentloaded")
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
    _assert_component_geometry(page, step.expected_selector, step)
    assert_healthy(page, context=step_context)

  # 3. Verify clean console log state
  browser_errors.assert_clean(f"Scenario {scenario.id}")
