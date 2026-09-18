"""SxS diff engine coverage over real XProf user journey renders.

test_user_journeys.py drives the journeys and asserts on geometry, DOM content
and console errors, but discards what the page looked like. This module walks
the same journeys, captures each waypoint, and feeds the captures to the SxS
diff engine so the engine runs against the real product instead of synthetic
images.

Each journey is walked twice: once against the baseline build and once against
the candidate build, so a non-zero diff is attributable to the change under
review. Set XPROF_BASELINE_SERVER_URL to the baseline server to enable this.
When it is unset both walks target the same server, which asserts only that
capture is stable -- the precondition for gating a build on a visual delta.

Execution:
  - Live browser suite: `pytest tests/ui/test_journey_sxs.py` (invoked in CI by
    `kokoro/gcp_ubuntu/kokoro_build_testing.sh`).
  - Hermetic unit tests: `pytest tests/ui/test_journey_capture.py
    tests/ui/test_sxs_diff_engine.py`.
"""

from collections.abc import Callable
from collections.abc import Iterator
import getpass
import hashlib
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
  from tests.ui.sxs_diff_engine import resolve_scenario_runs
  from tests.ui.sxs_diff_engine import SxsDiffEngine
  from tests.ui.sxs_diff_engine import WaypointDiff
  from tests.ui.sxs_report_generator import generate_sxs_html_report
  from tests.ui.sxs_report_generator import publish_report_artifact
  from tests.ui.test_user_journeys import dispatch_action
  from tests.ui.test_user_journeys import JOURNEY_SCENARIOS
  from tests.ui.test_user_journeys import JourneyScenario
  from tests.ui.test_user_journeys import URL_SETTLE_TIMEOUT_MS
  from tests.ui.ui_helpers import build_tool_url
except ImportError:
  from journey_capture import capture_waypoint
  from journey_capture import NetworkRecorder
  from journey_capture import WaypointCapture
  from sxs_diff_engine import resolve_scenario_runs
  from sxs_diff_engine import SxsDiffEngine
  from sxs_diff_engine import WaypointDiff
  from sxs_report_generator import generate_sxs_html_report
  from sxs_report_generator import publish_report_artifact
  from test_user_journeys import dispatch_action
  from test_user_journeys import JOURNEY_SCENARIOS
  from test_user_journeys import JourneyScenario
  from test_user_journeys import URL_SETTLE_TIMEOUT_MS
  from ui_helpers import build_tool_url

# Directory the combined HTML report is written to. Kokoro points this at
# KOKORO_ARTIFACTS_DIR; under Bazel the report additionally lands in
# TEST_UNDECLARED_OUTPUTS_DIR for test runner output artifacts.
_REPORT_DIR_ENV = "XPROF_SXS_REPORT_DIR"
_REPORT_FILENAME = "sxs_report.html"

# Pinned so a capture taken on a workstation is comparable to one taken in CI.
_VIEWPORT = {"width": 1280, "height": 900}

_session_report_dir: str | None = None


def _get_report_filename() -> str:
  """Returns a worker-scoped report filename when running under pytest-xdist."""
  worker = os.environ.get("PYTEST_XDIST_WORKER")
  if worker:
    return f"sxs_report_{worker}.html"
  return _REPORT_FILENAME


def _get_report_dir() -> str:
  """Returns a stable directory for the HTML report across the test session."""
  global _session_report_dir
  for env_var in (
      _REPORT_DIR_ENV,
      "KOKORO_ARTIFACTS_DIR",
      "TEST_UNDECLARED_OUTPUTS_DIR",
  ):
    if env_dir := os.environ.get(env_var):
      return env_dir
  if _session_report_dir is None:
    _session_report_dir = tempfile.mkdtemp(prefix="xprof_sxs_")
  return _session_report_dir


def _clear_tools_cache(logdir: str) -> None:
  """Removes ephemeral XProf cache files from run directories and $TMPDIR.

  XProf's ToolsCache writes `.cached_tools.json` into `$TMPDIR/xprof_<uid>/` for
  `demo/plugins/profile` paths (profile_plugin.py:501-507) and writes derived
  `.SSTABLE`, `.hlo_proto.pb`, `ALL_HOSTS.op_stats.pb`, and `cache_version.txt`
  files into each run directory alongside `<host>.xplane.pb`. Clearing both
  locations before each walk guarantees that baseline and candidate walks see
  identical server cache state.

  Args:
    logdir: Profile log directory containing run subdirectories.
  """
  logdir_path = pathlib.Path(logdir)
  if logdir_path.is_dir():
    for run_dir in logdir_path.iterdir():
      if not run_dir.is_dir():
        continue
      for child in run_dir.iterdir():
        if child.is_file() and not child.name.endswith(".xplane.pb"):
          try:
            child.unlink()
          except OSError:
            pass
      for candidate_str in (str(run_dir), str(run_dir.resolve())):
        cache_key = hashlib.sha256(candidate_str.encode("utf-8")).hexdigest()[
            :16
        ]
        user_id = (
            os.getuid() if hasattr(os, "getuid") else getpass.getuser()
        )
        tmp_cache = (
            pathlib.Path(tempfile.gettempdir())
            / f"xprof_{user_id}"
            / f"xprof_{cache_key}_.cached_tools.json"
        )
        try:
          tmp_cache.unlink()
        except OSError:
          pass
  user_id = os.getuid() if hasattr(os, "getuid") else getpass.getuser()
  tmp_dir = pathlib.Path(tempfile.gettempdir()) / f"xprof_{user_id}"
  if tmp_dir.is_dir():
    for cache_file in tmp_dir.glob("xprof_*_.cached_tools.json"):
      try:
        cache_file.unlink()
      except OSError:
        pass


@pytest.fixture(scope="session")
def sxs_collector(logdir: str) -> Iterator[list[WaypointDiff]]:
  """Collects waypoint diffs across journeys and renders one report."""
  _clear_tools_cache(logdir)
  collected: list[WaypointDiff] = []
  try:
    yield collected
  finally:
    _clear_tools_cache(logdir)
    if collected:
      report_dir = _get_report_dir()
      filename = _get_report_filename()
      report_path = generate_sxs_html_report(
          collected, str(pathlib.Path(report_dir) / filename)
      )
      publish_report_artifact(
          report_path,
          os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR"),
          artifact_name=filename,
      )
      print(f"\nSxS report: {report_path}")


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
      "(() => {\n"
      "  const inject = () => {\n"
      "    const style = document.createElement('style');\n"
      "    style.textContent = `\n"
      "      *, *::before, *::after {\n"
      "        transition-duration: 0s !important;\n"
      "        animation-duration: 0s !important;\n"
      "      }\n"
      "      html {\n"
      "        scrollbar-gutter: stable !important;\n"
      "        overflow-y: scroll !important;\n"
      "      }\n"
      "    `;\n"
      "    (document.head || document.documentElement).appendChild(style);\n"
      "  };\n"
      "  if (document.readyState === 'loading') {\n"
      "    document.addEventListener('DOMContentLoaded', inject);\n"
      "  } else {\n"
      "    inject();\n"
      "  }\n"
      "})();"
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
        re.compile(rf"tag={re.escape(scenario.initial_tool)}"),
        timeout=URL_SETTLE_TIMEOUT_MS,
    )
    init_name = f"00_{scenario.initial_tool}"
    captures.append(capture_waypoint(page, recorder, init_name))
    print(f"  [{scenario.id}] Captured {init_name}", flush=True)

    for idx, step in enumerate(scenario.steps, start=1):
      dispatch_action(page, server_url, logdir, step)
      # Without this the capture can fire while the component is still
      # mounting, so the same waypoint renders differently on each walk.
      expect(
          page.locator(f":is({step.expected_selector}):visible").first
      ).to_be_visible(timeout=20000)
      step_name = f"{idx:02d}_{step.action.value}_{step.target}"
      captures.append(capture_waypoint(page, recorder, step_name))
      print(f"  [{scenario.id}] Captured {step_name}", flush=True)

    return captures
  finally:
    context.close()


def _describe_divergence(
    diff: WaypointDiff,
    before: WaypointCapture | None = None,
    after: WaypointCapture | None = None,
) -> str:
  """Summarizes which modalities diverged or failed to stabilize."""
  parts: list[str] = []
  unsettled: list[str] = []
  if before is not None and not before.stabilized:
    unsettled.extend(f"baseline:{r}" for r in before.unsettled_reasons)
  if after is not None and not after.stabilized:
    unsettled.extend(f"candidate:{r}" for r in after.unsettled_reasons)
  if unsettled:
    parts.append(f"capture did not stabilize ({', '.join(unsettled)})")
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


def _format_failure_banner(
    scenario_id: str,
    diverged: list[str],
    is_ab_comparison: bool = False,
) -> str:
  """Builds a structured failure banner referencing the session HTML report."""
  filename = _get_report_filename()
  report_path = str(pathlib.Path(_get_report_dir()) / filename)

  lines = [
      "",
      "=" * 80,
      "                    VISUAL / DOM / NETWORK DIFF DETECTED",
      "=" * 80,
      f"Journey: {scenario_id}",
      "",
      "Divergent Waypoints:",
  ]
  for d in diverged:
    lines.append(f"  • {d}")

  lines.extend([
      "",
      "Interactive Diff Report (Generated at Session Teardown):",
      f"  • Artifact Filename:     {filename}",
  ])
  if undeclared := os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR"):
    lines.append(f"  • Test Outputs Artifact: {undeclared}/{filename}")
  abs_report = os.path.abspath(report_path)
  lines.append(f"  • Report Path:           file://{abs_report}")

  if is_ab_comparison:
    lines.extend([
        "",
        "How to Review and Take Action (A/B Certification Gate):",
        f"  1. Open '{filename}' in your browser to inspect the swipe slider,",
        "     DOM unified diff, and network waterfall.",
        "  2. If the diff is an unintended regression, fix the UI/backend code",
        "     and re-run the test.",
        "  3. If the diff is an intentional UI update, copy the approval JSON",
        "     from the report into tests/ui/approved_manifest.json.",
        "=" * 80,
    ])
  else:
    lines.extend([
        "",
        "How to Review and Take Action (Single-Server Reproducibility Gate):",
        f"  1. Open '{filename}' after the test session finishes to inspect",
        "     the swipe slider, DOM unified diff, and network waterfall.",
        "  2. Because both walks targeted the same build, any diff indicates",
        "     non-determinism or an unsettled capture:",
        "     - Mask dynamic regions in journey_capture.MASK_SELECTORS.",
        "     - Normalize unstable attributes in _NORMALIZED_HTML_JS.",
        "     - Extend stabilization waits in capture_waypoint.",
        "=" * 80,
    ])
  return "\n".join(lines)


def _resolve_scenario_runs(
    scenario: JourneyScenario,
    resolve_run: Callable[[str], str],
    logdir: str | None = None,
) -> JourneyScenario:
  """Maps every run and host a scenario names onto ones in the logdir."""
  return resolve_scenario_runs(scenario, resolve_run, logdir=logdir)


# pylint: disable=redefined-outer-name
@pytest.mark.parametrize("scenario", JOURNEY_SCENARIOS, ids=lambda s: s.id)
def test_journey_capture_is_reproducible(
    browser: Browser,
    server_url: str,
    baseline_server_url: str,
    logdir: str,
    resolve_run: Callable[[str], str],
    sxs_collector: list[WaypointDiff],
    scenario: JourneyScenario,
) -> None:
  """Baseline and candidate builds must produce identical waypoints."""
  scenario = _resolve_scenario_runs(scenario, resolve_run, logdir=logdir)
  session_path = os.path.join(logdir, scenario.fixture)
  if not os.path.exists(session_path):
    pytest.skip(f"Fixture '{scenario.fixture}' not present in logdir {logdir}")

  _clear_tools_cache(logdir)
  baseline = _walk_journey(browser, baseline_server_url, logdir, scenario)
  _clear_tools_cache(logdir)
  candidate = _walk_journey(browser, server_url, logdir, scenario)
  assert len(baseline) == len(candidate), (
      f"Journey {scenario.id} produced {len(baseline)} waypoints on the"
      f" baseline build and {len(candidate)} on the candidate build"
  )

  is_ab_comparison = baseline_server_url.rstrip("/") != server_url.rstrip("/")
  # In single-server reproducibility runs (baseline_server_url == server_url),
  # pass an empty manifest path so approvals cannot silence non-determinism.
  engine = (
      SxsDiffEngine()
      if is_ab_comparison
      else SxsDiffEngine(approved_manifest_path="")
  )
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
      for before, after in zip(baseline, candidate, strict=True)
  ]
  sxs_collector.extend(diffs)

  diverged = [
      _describe_divergence(d, before, after)
      for d, before, after in zip(diffs, baseline, candidate, strict=True)
      if (d.verdict == "CHANGED" if is_ab_comparison else d.verdict != "SAME")
      or not before.stabilized
      or not after.stabilized
  ]
  banner = ""
  if diverged:
    banner = _format_failure_banner(
        scenario.id, diverged, is_ab_comparison=is_ab_comparison
    )
  assert not diverged, banner
