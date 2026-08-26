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
  from tests.ui.invariants import run_content_invariants
  from tests.ui.ui_helpers import build_tool_url
  from tests.ui.ui_helpers import select_host
  from tests.ui.ui_helpers import switch_tool
except ImportError:
  from conftest import BrowserErrors
  from invariants import run_content_invariants
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

# TODO(b/552235521): Trace viewer legacy iframe type errors.
_UPSTREAM_BASELINE_IGNORED_PATTERNS: tuple[str, ...] = (
    "trace_viewer",
    "streaming trace",
    "Cannot read properties of undefined",
    "split is not a function",
)

_TOOL_NAME_TO_TAG: dict[str, str] = {
    "Overview Page": "overview_page",
    "Framework Op Stats": "framework_op_stats",
    "Input Pipeline Analysis": "input_pipeline",
    "Memory Profile": "memory_profile",
    "Pod Viewer": "pod_viewer",
    "Op Profile": "op_profile",
    "HLO Op Profile": "op_profile",
    "Memory Viewer": "memory_viewer",
    "Graph Viewer": "graph_viewer",
    "HLO Op Stats": "hlo_stats",
    "Inference Profile": "inference_profile",
    "Roofline Model": "roofline_model",
    "Kernel Stats": "kernel_stats",
    "Trace Viewer": "trace_viewer",
    "Megascale Viewer": "megascale_stats",
    "Perf Counters": "perf_counters",
    "Utilization Viewer": "utilization_viewer",
}


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


def _dispatch_action(
    page: Page, server_url: str, logdir: str, step: JourneyStep
) -> None:
  """Dispatches the UI navigation action corresponding to the journey step."""
  match step.action:
    case ActionType.SWITCH_TOOL:
      switch_tool(page, step.target)
      expected_tag = _TOOL_NAME_TO_TAG.get(
          step.target, step.target.lower().replace(" ", "_")
      )
      expect(page).to_have_url(re.compile(rf"tag={re.escape(expected_tag)}"))
    case ActionType.SELECT_HOST:
      select_host(page, step.target)
      expect(page).to_have_url(re.compile(rf"host={re.escape(step.target)}"))
    case ActionType.GO_BACK:
      page.go_back(wait_until="domcontentloaded")
      expected_tag = _TOOL_NAME_TO_TAG.get(
          step.target, step.target.lower().replace(" ", "_")
      )
      expect(page).to_have_url(re.compile(rf"tag={re.escape(expected_tag)}"))
    case ActionType.GO_FORWARD:
      page.go_forward(wait_until="domcontentloaded")
      expected_tag = _TOOL_NAME_TO_TAG.get(
          step.target, step.target.lower().replace(" ", "_")
      )
      expect(page).to_have_url(re.compile(rf"tag={re.escape(expected_tag)}"))
    case ActionType.GOTO:
      parts = step.target.split("/", 1)
      run_name = parts[0]
      tag = parts[1] if len(parts) > 1 else "overview_page"
      dest_path = os.path.join(logdir, run_name)
      dest_url = build_tool_url(server_url, dest_path, run_name, tag)
      page.goto(dest_url, wait_until="domcontentloaded")
      expect(page).to_have_url(re.compile(rf"tag={re.escape(tag)}"))
    case _:
      raise ValueError(f"Unsupported journey action type: {step.action}")


def _assert_component_geometry(
    page: Page, selector: str, step: JourneyStep
) -> None:
  """Asserts that the component is mounted with positive geometry."""
  comp = page.locator(f":is({selector}):visible").first
  expect(comp).to_be_visible(timeout=20000)
  bbox = comp.bounding_box()
  assert (
      bbox is not None and bbox["width"] > 0 and bbox["height"] > 0
  ), f"Component {selector} collapsed at step {step}"


def _assert_content_invariants(page: Page, context_msg: str) -> None:
  """Sweeps DOM text for poison tokens (raw template variables, error dumps)."""
  violations = run_content_invariants(page.inner_text("body"))
  assert (
      not violations
  ), f"Poison tokens detected at {context_msg}: {violations}"


@pytest.mark.parametrize("scenario", JOURNEY_SCENARIOS, ids=lambda s: s.id)
def test_user_journey_state_machine(
    page: Page,
    server_url: str,
    logdir: str,
    browser_errors: BrowserErrors,
    scenario: JourneyScenario,
) -> None:
  """Executes declarative user journey state transitions with invariant sweeps."""
  browser_errors.ignore(*_UPSTREAM_BASELINE_IGNORED_PATTERNS)

  # 1. Mount initial starting waypoint
  session_path = os.path.join(logdir, scenario.fixture)
  if not os.path.exists(session_path):
    pytest.skip(f"Fixture '{scenario.fixture}' not present in logdir")
  url = build_tool_url(
      server_url, session_path, scenario.fixture, scenario.initial_tool
  )
  page.goto(url, wait_until="domcontentloaded")
  expect(page).to_have_url(
      re.compile(rf"tag={re.escape(scenario.initial_tool)}")
  )
  expect(page.locator("body")).to_be_visible()
  _assert_content_invariants(page, f"initial load of {scenario.id}")

  # 2. Iterate through declarative state machine steps
  for idx, step in enumerate(scenario.steps, start=1):
    step_context = (
        f"step {idx}/{len(scenario.steps)} ({step.action} -> {step.target})"
    )
    _dispatch_action(page, server_url, logdir, step)
    _assert_component_geometry(page, step.expected_selector, step)
    _assert_content_invariants(page, step_context)

  # 3. Verify clean console log state
  browser_errors.assert_clean(f"Scenario {scenario.id}")
