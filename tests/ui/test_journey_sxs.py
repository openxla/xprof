"""SxS diff engine coverage over real XProf user journey renders.

test_user_journeys.py drives the journeys and asserts on geometry, DOM content
and console errors, but discards what the page looked like. This module walks
the same journeys, captures each waypoint, and feeds the captures to the SxS
diff engine so the engine runs against the real product instead of synthetic
images.

Two renders of an unchanged build must be identical. That is what is asserted
here, and it is the precondition for ever gating a build on a visual delta:
until capture is provably stable, a non-zero diff cannot be attributed to a
code change.
"""

from collections.abc import Iterator
import functools
import os
import pathlib
import re
import tempfile

from playwright.sync_api import Browser
from playwright.sync_api import expect
import pytest

# pylint: disable=g-import-not-at-top
try:
  from tests.ui.journey_capture import capture_waypoint
  from tests.ui.journey_capture import NetworkRecorder
  from tests.ui.journey_capture import WaypointCapture
  from tests.ui.sxs_diff_engine import SxsDiffEngine
  from tests.ui.sxs_diff_engine import WaypointDiff
  from tests.ui.sxs_report_generator import generate_sxs_html_report
  from tests.ui.test_user_journeys import dispatch_action
  from tests.ui.test_user_journeys import JOURNEY_SCENARIOS
  from tests.ui.test_user_journeys import JourneyScenario
  from tests.ui.ui_helpers import build_tool_url
except ImportError:
  from journey_capture import capture_waypoint
  from journey_capture import NetworkRecorder
  from journey_capture import WaypointCapture
  from sxs_diff_engine import SxsDiffEngine
  from sxs_diff_engine import WaypointDiff
  from sxs_report_generator import generate_sxs_html_report
  from test_user_journeys import dispatch_action
  from test_user_journeys import JOURNEY_SCENARIOS
  from test_user_journeys import JourneyScenario
  from ui_helpers import build_tool_url

# Explicit override for the report location, used by CI to point the report at
# an artifact directory the build system will collect.
_REPORT_DIR_ENV = "XPROF_SXS_REPORT_DIR"
_REPORT_FILENAME = "sxs_report.html"

# Adjudications a reviewer has accepted, keyed by "journey:waypoint". Resolved
# next to this module the same way the journey catalog is, so a checkout and a
# runfiles tree both find it. The engine treats an absent file as "nothing
# approved", which is the correct default before anyone has adjudicated.
_APPROVED_MANIFEST_PATH = (
    pathlib.Path(__file__).resolve().parent / "approved_manifest.json"
)

# Verdicts that do not fail the run: byte-identical, or a delta a reviewer has
# already accepted in the manifest above.
_NON_DIVERGENT_VERDICTS = frozenset({"SAME", "APPROVED"})

# Pinned so a capture taken on a workstation is comparable to one taken in CI.
_VIEWPORT = {"width": 1280, "height": 900}


@functools.cache
def _report_path() -> str:
  """Returns the one path this process writes its SxS report to.

  Memoized because the fallback allocates a fresh temporary directory. Both the
  fixture that writes the report and the banner that cites it read the path
  from here, so the location a reviewer is handed is always the location that
  was written; resolving it independently at each site previously sent
  reviewers to a file that did not exist.

  Returns:
    Absolute path of the report file for this run.
  """
  # Writing straight into the undeclared outputs directory, rather than writing
  # elsewhere and copying, means the report a reviewer opens from the test
  # runner's artifact view is the same bytes the run produced.
  report_dir = (
      os.environ.get(_REPORT_DIR_ENV)
      or os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
      or tempfile.mkdtemp(prefix="xprof_sxs_")
  )
  return str(pathlib.Path(report_dir).resolve() / _REPORT_FILENAME)


@pytest.fixture(scope="session")
def sxs_collector() -> Iterator[list[WaypointDiff]]:
  """Collects waypoint diffs across journeys and renders one report."""
  collected: list[WaypointDiff] = []
  yield collected
  if not collected:
    return
  try:
    generate_sxs_html_report(collected, _report_path())
  except Exception as err:  # pylint: disable=broad-except
    # The report is diagnostic output for a run whose verdict is already
    # decided and already printed. A missing template or an unwritable
    # directory must not be allowed to convert a readable journey failure into
    # an opaque teardown error and take the failure detail down with it.
    print(f"\nSxS report generation failed, diffs stand as printed: {err!r}")
    return
  print(f"\nSxS report: {_report_path()}")


def _walk_journey(
    browser: Browser, server_url: str, logdir: str, scenario: JourneyScenario
) -> list[WaypointCapture]:
  """Drives one journey end to end in a cold context, capturing every waypoint.

  Each side of the comparison gets its own context. Angular injects component
  styles on first use, so a warm page reaches a waypoint with a different set
  of <style> elements than a cold one and the DOM compares unequal for reasons
  that have nothing to do with the build.

  Args:
    browser: Playwright browser used to open an isolated context.
    server_url: Base URL of the XProf server to drive.
    logdir: Directory holding the profile fixtures.
    scenario: Journey to walk.

  Returns:
    One capture per waypoint, in visit order.
  """
  context = browser.new_context(viewport=_VIEWPORT)
  # External webfonts resolve asynchronously and depend on external network
  # egress, introducing font metric differences that cause chart width flapping.
  # Abort external font requests so rendering uses deterministic local fonts.
  context.route(
      re.compile(r"https://fonts\.(?:googleapis|gstatic)\.com/.*"),
      lambda route: route.abort(),
  )
  # Disable transitions/animations and stabilize scrollbars so container layout
  # is instantaneous and deterministic, eliminating width flapping.
  context.add_init_script(
      "document.addEventListener('DOMContentLoaded', () => {"
      "  const style = document.createElement('style');"
      "  style.textContent = `"
      "    *, *::before, *::after {"
      "      transition-duration: 0s !important;"
      "      animation-duration: 0s !important;"
      "    }"
      "    html {"
      "      scrollbar-gutter: stable !important;"
      "      overflow-y: scroll !important;"
      "    }"
      "  `;"
      "  document.head.appendChild(style);"
      "});"
  )
  try:
    page = context.new_page()
    recorder = NetworkRecorder(page)
    captures: list[WaypointCapture] = []

    session_path = os.path.join(logdir, scenario.fixture)
    url = build_tool_url(
        server_url, session_path, scenario.fixture, scenario.initial_tool
    )
    page.goto(url, wait_until="domcontentloaded")
    expect(page).to_have_url(
        re.compile(rf"tag={re.escape(scenario.initial_tool)}")
    )
    captures.append(
        capture_waypoint(page, recorder, f"00_{scenario.initial_tool}")
    )

    for idx, step in enumerate(scenario.steps, start=1):
      dispatch_action(page, server_url, logdir, step)
      # Without this the capture can fire while the component is still
      # mounting, so the same waypoint renders differently on each walk.
      expect(
          page.locator(f":is({step.expected_selector}):visible").first
      ).to_be_visible(timeout=20000)
      captures.append(
          capture_waypoint(
              page, recorder, f"{idx:02d}_{step.action.value}_{step.target}"
          )
      )

    return captures
  finally:
    context.close()


def _describe_divergence(diff: WaypointDiff) -> str:
  """Summarizes which modalities diverged at a waypoint, and by how much."""
  parts: list[str] = []
  if diff.visual.dimension_mismatch:
    parts.append(f"geometry: {diff.visual.dimension_mismatch}")
  if diff.visual.diff_pixels:
    parts.append(
        f"visual: {diff.visual.diff_pixels} px"
        f" ({diff.visual.diff_ratio:.4%} of {diff.visual.total_pixels})"
    )
  if diff.dom.has_changes:
    parts.append(
        f"dom: +{diff.dom.added_lines}/-{diff.dom.deleted_lines} lines"
    )
  if diff.network.has_changes:
    parts.append(f"network: {'; '.join(diff.network.status_mismatches[:3])}")
  return f"{diff.waypoint_name} -> " + " | ".join(parts)


def _format_failure_banner(scenario_id: str, diverged: list[str]) -> str:
  """Builds the failure banner shown when two walks of one build disagree.

  Formatting only. The report itself is written once by the session fixture,
  so a per-failure regeneration here would rewrite a multi-megabyte file for
  every failing journey and would put an I/O failure on the path that reports
  the original failure.

  Args:
    scenario_id: Identifier of the journey that diverged.
    diverged: One human-readable summary per divergent waypoint.

  Returns:
    The banner text to attach to the assertion.
  """
  lines = [
      "",
      "=" * 80,
      "                    VISUAL / DOM / NETWORK DIFF DETECTED",
      "=" * 80,
      f"Journey: {scenario_id}",
      "",
      "Divergent Waypoints:",
  ]
  lines.extend(f"  \u2022 {d}" for d in diverged)

  lines.extend([
      "",
      "Interactive Diff Report:",
      "  Open 'sxs_report.html' from the Outputs of this test invocation to",
      "  inspect the swipe slider, DOM unified diff, and network waterfall.",
      f"  Report path: {_report_path()}",
      "",
      "How to Review and Take Action:",
      "  1. Open the report and identify which leg diverged. A DOM delta with",
      "     no visual delta points at capture normalization; a visual delta",
      "     with no DOM delta points at a rendering race in the component.",
      "  2. If the diff is UNINTENDED (product bug, flake, or a gap in the",
      "     normalization in journey_capture.py), fix the cause. Both sides of",
      "     this comparison are the same build walked twice, so a divergence",
      "     is not by itself evidence of an intended UI change.",
      "  3. If the diff is understood and accepted, open the Approval Portal",
      "     at the bottom of the report and copy the waypoint token(s) it",
      '     emits. Add them under "approved_diffs" in',
      '     tests/ui/approved_manifest.json, keyed by "journey:waypoint", and',
      "     commit that file with your CL. The engine reloads the manifest on",
      "     the next run and reports those waypoints as APPROVED instead of",
      "     failing the journey.",
      "",
      "  A token is a digest of the diff content, not a signature. It pins the",
      "  exact delta that was accepted: if the delta changes, the token stops",
      "  matching and the waypoint fails again.",
      "=" * 80,
  ])
  return "\n".join(lines)


# pylint: disable=redefined-outer-name
@pytest.mark.parametrize("scenario", JOURNEY_SCENARIOS, ids=lambda s: s.id)
def test_journey_capture_is_reproducible(
    browser: Browser,
    server_url: str,
    logdir: str,
    sxs_collector: list[WaypointDiff],
    scenario: JourneyScenario,
) -> None:
  """Two renders of one build must produce identical waypoints."""
  session_path = os.path.join(logdir, scenario.fixture)
  assert os.path.exists(
      session_path
  ), f"Fixture '{scenario.fixture}' not present in logdir {logdir}"

  baseline = _walk_journey(browser, server_url, logdir, scenario)
  candidate = _walk_journey(browser, server_url, logdir, scenario)
  assert len(baseline) == len(candidate), (
      f"Journey {scenario.id} produced {len(baseline)} waypoints on the first"
      f" walk and {len(candidate)} on the second"
  )

  engine = SxsDiffEngine(approved_manifest_path=str(_APPROVED_MANIFEST_PATH))
  diffs = [
      engine.evaluate_waypoint(
          journey_name=scenario.id,
          waypoint_name=before.name,
          img_a=before.png_bytes,
          img_b=after.png_bytes,
          html_a=before.html,
          html_b=after.html,
          requests_a=before.requests,
          requests_b=after.requests,
      )
      for before, after in zip(baseline, candidate)
  ]
  sxs_collector.extend(diffs)

  diverged = [
      _describe_divergence(d)
      for d in diffs
      if d.verdict not in _NON_DIVERGENT_VERDICTS
  ]
  # The banner is built as the assertion message rather than inside a branch,
  # so the cost is paid only on failure without hiding the assertion behind a
  # condition that could leave the test vacuously green.
  assert not diverged, _format_failure_banner(scenario.id, diverged)
