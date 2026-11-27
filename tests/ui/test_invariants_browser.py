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
"""Real-Chromium DOM/layout invariant and SxS report interaction tests."""

import io
import json
import os
import pathlib

from PIL import Image
from playwright.sync_api import expect
from playwright.sync_api import Page
from tests.ui.conftest import BrowserErrors
from tests.ui.invariants import check_no_horizontal_overflow
from tests.ui.invariants import check_no_sidenav_overlap
from tests.ui.invariants import run_cell_invariants
from tests.ui.invariants import run_dom_invariants
from tests.ui.sxs_report_generator import generate_sxs_html_report

# pylint: disable=g-import-not-at-top
try:
  from tests.ui import journey_capture
  from tests.ui import sxs_diff_engine
except ImportError:
  import journey_capture
  import sxs_diff_engine
# pylint: enable=g-import-not-at-top

SxsDiffEngine = sxs_diff_engine.SxsDiffEngine


def _png_bytes(size: tuple[int, int], color: tuple[int, int, int]) -> bytes:
  """Creates solid-color PNG bytes of `size`."""
  img = Image.new("RGB", size, color)
  buf = io.BytesIO()
  img.save(buf, format="PNG")
  return buf.getvalue()


def test_dom_and_cell_invariants_in_chromium(
    page: Page, browser_errors: BrowserErrors
) -> None:
  """Verifies DOM, MDC table/card, and cell invariants in real Chromium."""
  page.set_content("""
    <html><body>
      <div class="mat-mdc-card">Healthy Card</div>
      <svg width="120" height="60"><rect width="80" height="40"/></svg>
      <svg width="0" height="0" style="visibility: hidden"></svg>
      <div role="table" class="mat-mdc-table">
        <div role="row" class="mat-mdc-header-row">
          <div role="columnheader" class="mat-mdc-header-cell">Self %</div>
          <div role="columnheader" class="mat-mdc-header-cell">% Diff</div>
          <div role="columnheader" class="mat-mdc-header-cell">Time</div>
        </div>
        <div role="row" class="mat-mdc-row">
          <div role="cell" class="mat-mdc-cell">42.5%</div>
          <div role="cell" class="mat-mdc-cell">+240.0%</div>
          <div role="cell" class="mat-mdc-cell">12.4 ms</div>
        </div>
      </div>
    </body></html>
  """)
  assert not run_dom_invariants(page)
  assert not run_cell_invariants(page)

  page.set_content("""
    <html><body>
      <div class="mat-mdc-card">   </div>
      <svg style="width: 0px; height: 40px; display: block"></svg>
      <div class="mat-mdc-snack-bar-container">Backend unavailable</div>
      <main-page><mat-sidenav-content>
        <div class="hidden-content"></div>
      </mat-sidenav-content></main-page>
      <table>
        <thead><tr><th colspan="2">Group</th><th>Duration</th></tr></thead>
        <tbody><tr><td>op</td><td>115.0%</td><td>-3.5 ms</td></tr></tbody>
      </table>
    </body></html>
  """)
  dom_v = run_dom_invariants(page)
  assert any("collapsed geometry" in v for v in dom_v)
  assert any("unexpectedly empty" in v for v in dom_v)
  assert any("Backend unavailable" in v for v in dom_v)
  assert any("hidden-content" in v for v in dom_v)
  cell_v = run_cell_invariants(page)
  assert any("115.0%" in v for v in cell_v)
  assert any("-3.5" in v for v in cell_v)
  browser_errors.assert_clean("dom_and_cell_invariants")


def test_layout_overflow_and_sidenav_overlap_in_chromium(
    page: Page, browser_errors: BrowserErrors
) -> None:
  """Verifies content-pane overflow exemptions and sidenav overlap detection."""
  page.set_viewport_size({"width": 1280, "height": 900})
  page.set_content("""
    <html><body style="margin: 0">
      <mat-sidenav-container style="display: flex; width: 1280px">
        <mat-sidenav class="mat-drawer-side mat-drawer-opened"
            style="width: 260px; flex-shrink: 0">Nav</mat-sidenav>
        <mat-sidenav-content id="pane"
            style="width: 1020px; overflow-x: auto; display: block">
          <div style="overflow-x: auto; width: 100%">
            <table style="width: 1800px"><tr><td>Wide table</td></tr></table>
          </div>
        </mat-sidenav-content>
      </mat-sidenav-container>
    </body></html>
  """)
  assert not check_no_sidenav_overlap(page)
  assert not check_no_horizontal_overflow(page, check_content_pane=True)

  page.set_content("""
    <html><body style="margin: 0">
      <mat-sidenav-container
          style="position: relative; display: block; width: 1280px">
        <mat-sidenav class="mat-drawer-side mat-drawer-opened"
            style="position: absolute; left: 0; width: 300px; height: 400px">
          Nav
        </mat-sidenav>
        <mat-sidenav-content id="pane"
            style="display: block; margin-left: 100px; width: 900px;
                   overflow-x: auto">
          <div class="unwrapped-row" style="width: 1500px; height: 80px">
            Chart
          </div>
        </mat-sidenav-content>
      </mat-sidenav-container>
    </body></html>
  """)
  page.evaluate("document.getElementById('pane').scrollLeft = 400")
  expect(page.locator("#pane")).to_be_visible()
  assert any("overlaps" in v for v in check_no_sidenav_overlap(page))
  assert any(
      "mat-sidenav-content horizontal overflow" in v
      for v in check_no_horizontal_overflow(page, check_content_pane=True)
  )
  browser_errors.assert_clean("layout_overflow_and_overlap")


def test_sxs_report_interactive_tabs_modes_slider_and_approval(
    page: Page, tmp_path: pathlib.Path, browser_errors: BrowserErrors
) -> None:
  """Verifies generated SxS report tabs, slider aspect ratio, and approval."""
  engine = SxsDiffEngine(approved_manifest_path="")
  wp_same = engine.evaluate_waypoint(
      "triage",
      "00_overview",
      _png_bytes((200, 100), (80, 80, 80)),
      _png_bytes((200, 100), (80, 80, 80)),
      "<p>A</p>",
      "<p>A</p>",
      [],
      [],
  )
  wp_diff = engine.evaluate_waypoint(
      "triage",
      "01_hlo",
      _png_bytes((100, 200), (20, 20, 20)),
      _png_bytes((200, 100), (220, 40, 40)),
      "<p>A</p>",
      "<p>B</p>",
      [],
      [],
  )
  report_path = generate_sxs_html_report(
      [wp_same, wp_diff], str(tmp_path / "sxs_report.html")
  )
  page.goto(f"file://{os.path.abspath(report_path)}", wait_until="load")
  tabs = page.locator("button.tab")
  expect(tabs).to_have_count(2)
  tabs.nth(1).click()
  expect(tabs.nth(1)).to_have_class("tab active")

  page.locator("#btn-slider-wp_1").click()
  expect(page.locator("#view-slider-wp_1")).to_be_visible()
  widths = page.evaluate("""() => [
    document.getElementById('slider-base-img-wp_1').clientWidth,
    document.getElementById('slider-cand-img-wp_1').clientWidth,
  ]""")
  assert widths == [100, 200]

  page.locator("#btn-heatmap-wp_1").click()
  expect(page.locator("#view-heatmap-wp_1")).to_be_visible()

  page.locator("#rationale-input").fill("Intentional layout update")
  page.locator("button.btn-approve", has_text="Approve & Generate").click()
  expect(page.locator("#certified-banner")).to_be_visible()
  manifest = json.loads(page.locator("#manifest-code").inner_text())
  assert (
      manifest["approved_diffs"]["triage:01_hlo"]["diff_hash"]
      == wp_diff.diff_hash
  )
  browser_errors.assert_clean("sxs_report_interactive")


# pylint: disable=protected-access
def test_normalized_html_cdk_overlay_filtering_in_chromium(
    page: Page, browser_errors: BrowserErrors
) -> None:
  """Verifies _normalized_html strips inactive overlays and diffs dialogs."""
  page.set_viewport_size({"width": 1280, "height": 900})
  page.set_content("""
    <html><head><style>
      .css-hidden-pane { display: none; }
    </style></head><body>
      <main>Overview</main>
      <div class="cdk-overlay-container">
        <div class="cdk-overlay-backdrop"></div>
        <div class="cdk-global-overlay-wrapper">
          <div id="cdk-overlay-0" class="cdk-overlay-pane css-hidden-pane">
            <mat-dialog-container>Hidden by stylesheet</mat-dialog-container>
          </div>
          <div id="cdk-overlay-1" class="cdk-overlay-pane"
               style="visibility: hidden; width: 200px; height: 80px">
            <div>Hidden by visibility</div>
          </div>
          <div id="cdk-overlay-2" class="cdk-overlay-pane"
               style="opacity: 0; width: 200px; height: 80px">
            <div>Transparent</div>
          </div>
          <div id="cdk-overlay-3" class="cdk-overlay-pane mat-mdc-tooltip-panel"
               style="width: 120px; height: 32px">
            <mat-tooltip-component>Hover tip</mat-tooltip-component>
          </div>
          <div id="cdk-overlay-4" class="cdk-overlay-pane"></div>
        </div>
      </div>
    </body></html>
  """)
  inactive_html = journey_capture._normalized_html(page)
  assert "cdk-overlay-container" not in inactive_html
  assert "Hidden by stylesheet" not in inactive_html
  assert "Hover tip" not in inactive_html

  dialog_template = """
    <html><body>
      <main>Trace Viewer</main>
      <div class="cdk-overlay-container">
        <div class="cdk-overlay-backdrop cdk-overlay-dark-backdrop"></div>
        <div class="cdk-global-overlay-wrapper" style="top: {top}px">
          <div id="cdk-overlay-{ov}" class="cdk-overlay-pane"
               style="width: 600px; height: 320px; left: {left}px">
            <div class="cdk-focus-trap-anchor" aria-hidden="true"></div>
            <mat-dialog-container id="mat-mdc-dialog-{dlg}"
                aria-labelledby="mat-mdc-dialog-title-{dlg}" role="dialog">
              <h2 id="mat-mdc-dialog-title-{dlg}">{title}</h2>
              <button class="mdc-button {focus}">Close</button>
            </mat-dialog-container>
          </div>
        </div>
      </div>
    </body></html>
  """
  page.set_content(
      dialog_template.format(
          top=12,
          left=40,
          ov=0,
          dlg=0,
          title="Trace Viewer Settings",
          focus="cdk-focused cdk-program-focused",
      )
  )
  active_1 = journey_capture._normalized_html(page)
  page.set_content(
      dialog_template.format(
          top=18, left=44, ov=4, dlg=2, title="Trace Viewer Settings", focus=""
      )
  )
  active_2 = journey_capture._normalized_html(page)
  assert active_1 == active_2
  assert "cdk-overlay-container" in active_1
  assert 'id="cdk-overlay-N"' in active_1
  assert 'id="mat-mdc-dialog-N"' in active_1
  assert "Trace Viewer Settings" in active_1
  assert "cdk-focus-trap-anchor" not in active_1
  assert "cdk-focused" not in active_1

  engine = SxsDiffEngine()
  assert not engine.compute_dom_diff(active_1, active_2).has_changes
  page.set_content(
      dialog_template.format(
          top=18, left=44, ov=4, dlg=2, title="Profiler Settings", focus=""
      )
  )
  active_changed = journey_capture._normalized_html(page)
  assert engine.compute_dom_diff(active_1, active_changed).has_changes
  browser_errors.assert_clean("normalized_html_cdk_overlay_filtering")
