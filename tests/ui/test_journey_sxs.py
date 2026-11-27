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

from tests.ui.conftest import BrowserErrors
from tests.ui.conftest import dump_failure_artifacts_for_page
from tests.ui.journey_capture import block_external_fonts
from tests.ui.journey_capture import capture_waypoint
from tests.ui.journey_capture import NetworkRecorder
from tests.ui.journey_capture import settled_tool_url_pattern
from tests.ui.journey_capture import WaypointCapture
from tests.ui.sxs_diff_engine import make_run_resolver
from tests.ui.sxs_diff_engine import resolve_profile_logdir
from tests.ui.sxs_diff_engine import resolve_scenario_runs
from tests.ui.sxs_diff_engine import SxsDiffEngine
from tests.ui.sxs_diff_engine import waypoint_names
from tests.ui.sxs_diff_engine import WaypointDiff
from tests.ui.sxs_report_generator import generate_sxs_html_report
from tests.ui.sxs_report_generator import publish_report_artifact
from tests.ui.test_user_journeys import dispatch_action
from tests.ui.test_user_journeys import JOURNEY_SCENARIOS
from tests.ui.test_user_journeys import JourneyScenario
from tests.ui.ui_helpers import build_tool_url
from tests.ui.ui_helpers import URL_SETTLE_TIMEOUT_MS

pytestmark = pytest.mark.xdist_group("sxs")

# Directory the combined HTML report is written to. Kokoro points this at
# KOKORO_ARTIFACTS_DIR; under Bazel the report additionally lands in
# TEST_UNDECLARED_OUTPUTS_DIR for test runner output artifacts.
_REPORT_DIR_ENV = "XPROF_SXS_REPORT_DIR"
# Where CI publishes the report, shown in the failure banner instead of the
# local path when set.
_REPORT_URL_ENV = "XPROF_SXS_REPORT_URL"
_REPORT_FILENAME = "sxs_report.html"

# Pinned so a capture taken on a workstation is comparable to one taken in CI.
_VIEWPORT = {"width": 1280, "height": 900}

_session_report_dir: str | None = None


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
  """Removes ephemeral XProf cache files for `logdir` runs only.

  XProf's ToolsCache writes `.cached_tools.json` into `$TMPDIR/xprof_<uid>/` for
  `demo/plugins/profile` paths (profile_plugin.py:506-511) and writes derived
  `.SSTABLE`, `.hlo_proto.pb`, `ALL_HOSTS.op_stats.pb`, and `cache_version.txt`
  files into each run directory alongside `<host>.xplane.pb`. Clearing only
  files belonging to `logdir` guarantees identical server cache state without
  disturbing concurrent processes.

  Args:
    logdir: Profile log directory containing run subdirectories.
  """
  logdir_path = pathlib.Path(resolve_profile_logdir(logdir))
  cache_keys: set[str] = set()
  if logdir_path.is_dir():
    for run_dir in logdir_path.iterdir():
      if not run_dir.is_dir():
        continue
      for raw in {str(run_dir), str(run_dir.resolve())}:
        cache_keys.add(hashlib.sha256(raw.encode()).hexdigest()[:16])
      for child in run_dir.iterdir():
        if child.is_file() and not child.name.endswith(".xplane.pb"):
          try:
            child.unlink()
          except OSError:
            pass
  user_id = os.getuid() if hasattr(os, "getuid") else getpass.getuser()
  tmp_dir = pathlib.Path(tempfile.gettempdir()) / f"xprof_{user_id}"
  if tmp_dir.is_dir():
    for key in cache_keys:
      cache_file = tmp_dir / f"xprof_{key}_.cached_tools.json"
      try:
        cache_file.unlink()
      except OSError:
        pass


@pytest.fixture(scope="session")
def sxs_collector(
    synthesized_logdir: str,
    server_url: str,
    baseline_server_url: str,
) -> Iterator[list[WaypointDiff]]:
  """Collects waypoint diffs across journeys and renders one report."""
  _clear_tools_cache(synthesized_logdir)
  collected: list[WaypointDiff] = []
  is_ab_comparison = baseline_server_url.rstrip("/") != server_url.rstrip("/")
  try:
    yield collected
  finally:
    _clear_tools_cache(synthesized_logdir)
    report_dir = _get_report_dir()
    report_path = generate_sxs_html_report(
        collected,
        str(pathlib.Path(report_dir) / _REPORT_FILENAME),
        is_ab_comparison=is_ab_comparison,
    )
    publish_report_artifact(
        report_path,
        os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR"),
        artifact_name=_REPORT_FILENAME,
    )
    print(f"\nSxS report: {report_path}")


def _walk_journey(
    browser: Browser,
    server_url: str,
    logdir: str,
    scenario: JourneyScenario,
    names: list[str],
    *,
    assert_browser_clean: bool = True,
    artifact_label: str = "walk",
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
    scenario: Journey to walk, with runs and hosts already resolved onto ones
      present in `logdir`.
    names: Labels for the captures, from `sxs_diff_engine.waypoint_names`
      applied to the *declared* scenario. Passed in because `scenario` here
      carries run names already resolved against the local logdir.
    assert_browser_clean: Whether unhandled browser/HTTP errors fail the walk.
    artifact_label: Label distinguishing baseline vs candidate failure dumps.

  Returns:
    One capture per waypoint, in visit order.
  """
  context = browser.new_context(viewport=_VIEWPORT)
  # External webfonts resolve asynchronously and depend on external network
  # egress, introducing font metric differences that cause chart width flapping.
  # Fulfill external font requests locally with an empty 200 CSS response so
  # rendering uses deterministic local fonts without console errors.
  block_external_fonts(context)
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
  page = None
  tracker = BrowserErrors()
  try:
    page = context.new_page()
    tracker.attach(page)
    recorder = NetworkRecorder(page)
    captures: list[WaypointCapture] = []

    session_path = os.path.join(logdir, scenario.fixture)
    url = build_tool_url(
        server_url, session_path, scenario.fixture, scenario.initial_tool
    )
    page.goto(url, wait_until="domcontentloaded")
    expect(page).to_have_url(
        settled_tool_url_pattern(scenario.initial_tool),
        timeout=URL_SETTLE_TIMEOUT_MS,
    )
    init_name = names[0]
    captures.append(capture_waypoint(page, recorder, init_name))
    print(f"  [{scenario.id}] Captured {init_name}", flush=True)

    for idx, step in enumerate(scenario.steps, start=1):
      dispatch_action(page, server_url, logdir, step)
      if step.expected_url_substring:
        expect(page).to_have_url(
            re.compile(re.escape(step.expected_url_substring)),
            timeout=URL_SETTLE_TIMEOUT_MS,
        )
      # Without this the capture can fire while the component is still
      # mounting, so the same waypoint renders differently on each walk.
      expect(
          page.locator(f":is({step.expected_selector}):visible").first
      ).to_be_visible(timeout=20000)
      step_name = names[idx]
      captures.append(capture_waypoint(page, recorder, step_name))
      print(f"  [{scenario.id}] Captured {step_name}", flush=True)

    if assert_browser_clean:
      tracker.assert_clean(f"{scenario.id} ({artifact_label})")
    return captures
  except Exception:
    dump_failure_artifacts_for_page(
        page, f"sxs_{scenario.id}_{artifact_label}", tracker
    )
    raise
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
  if diff.visual.decode_error:
    parts.append(f"decode: {diff.visual.decode_error}")
  if diff.visual.mask_error:
    parts.append(f"mask: {diff.visual.mask_error}")
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
  filename = _REPORT_FILENAME
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
  report_link = os.environ.get(_REPORT_URL_ENV) or f"file://{abs_report}"
  lines.append(f"  • Report:                {report_link}")

  if is_ab_comparison:
    lines.extend([
        "",
        "How to Review and Take Action (A/B Certification Gate):",
        f"  1. Open '{filename}' in your browser to inspect the swipe slider,",
        "     DOM unified diff, and network waterfall.",
        "  2. If the diff is an unintended regression, fix the UI/backend code",
        "     and re-run the test.",
        "  3. If the diff is an intentional UI update, the change author must",
        "     approve it in the report's Author UI Diff Approval Portal,",
        "     replace tests/ui/approved_manifest.json with the generated file,",
        "     and link the report in the change description for code review.",
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


# pylint: disable=redefined-outer-name
@pytest.mark.parametrize("scenario", JOURNEY_SCENARIOS, ids=lambda s: s.id)
def test_journey_capture_is_reproducible(
    browser: Browser,
    server_url: str,
    baseline_server_url: str,
    synthesized_logdir: str,
    sxs_collector: list[WaypointDiff],
    scenario: JourneyScenario,
) -> None:
  """Baseline and candidate builds must produce identical waypoints."""
  # Names are taken before resolution so the approval keys they become do not
  # depend on which runs happen to exist in this environment's logdir.
  names = waypoint_names(scenario)
  scenario = resolve_scenario_runs(
      scenario,
      make_run_resolver(synthesized_logdir),
      logdir=synthesized_logdir,
  )
  session_path = os.path.join(synthesized_logdir, scenario.fixture)
  if not os.path.exists(session_path):
    pytest.skip(
        f"Fixture '{scenario.fixture}' not present in {synthesized_logdir}"
    )

  is_ab_comparison = baseline_server_url.rstrip("/") != server_url.rstrip("/")
  _clear_tools_cache(synthesized_logdir)
  baseline = _walk_journey(
      browser,
      baseline_server_url,
      synthesized_logdir,
      scenario,
      names,
      assert_browser_clean=not is_ab_comparison,
      artifact_label="baseline",
  )
  _clear_tools_cache(synthesized_logdir)
  candidate = _walk_journey(
      browser,
      server_url,
      synthesized_logdir,
      scenario,
      names,
      assert_browser_clean=True,
      artifact_label="candidate",
  )
  assert len(baseline) == len(candidate), (
      f"Journey {scenario.id} produced {len(baseline)} waypoints on the"
      f" baseline build and {len(candidate)} on the candidate build"
  )

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
          mask_boxes_a=before.mask_boxes,
          mask_boxes_b=after.mask_boxes,
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
