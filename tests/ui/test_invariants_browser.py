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

import base64
import importlib.util
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
from tests.ui.sxs_diff_engine import SxsDiffEngine
from tests.ui.sxs_report_generator import generate_sxs_html_report


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


def _find_static_font_path(filename: str) -> pathlib.Path | None:
  """Locates a bundled static font file in the wheel or source tree."""
  candidates: list[pathlib.Path] = []
  if static_dir := os.environ.get("XPROF_STATIC_DIR"):
    candidates.append(pathlib.Path(static_dir) / filename)
  spec = importlib.util.find_spec("xprof")
  if spec and spec.origin:
    candidates.append(pathlib.Path(spec.origin).parent / "static" / filename)
  roots: list[pathlib.Path] = []
  if repo_root := os.environ.get("XPROF_REPO_ROOT"):
    roots.append(pathlib.Path(repo_root))
  roots.extend(pathlib.Path(__file__).resolve().parents)
  for root in roots:
    candidates.append(
        root / "plugin" / "tensorboard_plugin_profile" / "static" / filename
    )
    candidates.append(root / "plugin" / "xprof" / "static" / filename)
  for candidate in candidates:
    if candidate.is_file():
      return candidate
  return None


def test_bundled_icon_fonts_render_ligatures_in_chromium(
    page: Page, browser_errors: BrowserErrors
) -> None:
  """Verifies bundled WOFF2 icon fonts shape ligatures and local fonts load."""
  mi_path = _find_static_font_path("materialicons.woff2")
  ms_path = _find_static_font_path("material_symbols_outlined.woff2")
  assert mi_path is not None, "materialicons.woff2 not found"
  assert ms_path is not None, "material_symbols_outlined.woff2 not found"

  mi_b64 = base64.b64encode(mi_path.read_bytes()).decode("ascii")
  ms_b64 = base64.b64encode(ms_path.read_bytes()).decode("ascii")
  page.set_content(f"""
    <html><head><style>
      @font-face {{
        font-family: 'Material Icons';
        font-style: normal;
        font-weight: 400;
        src: url(data:font/woff2;base64,{mi_b64}) format('woff2');
      }}
      @font-face {{
        font-family: 'Material Symbols Outlined';
        font-style: normal;
        font-weight: 100 700;
        src: url(data:font/woff2;base64,{ms_b64}) format('woff2');
      }}
      .mi {{
        font-family: 'Material Icons';
        font-size: 24px;
        line-height: 1;
        display: inline-block;
        white-space: nowrap;
        font-feature-settings: 'liga';
      }}
      .ms {{
        font-family: 'Material Symbols Outlined';
        font-size: 24px;
        line-height: 1;
        display: inline-block;
        white-space: nowrap;
        font-feature-settings: 'liga';
      }}
      .plain {{
        font-family: sans-serif;
        font-size: 24px;
        line-height: 1;
        display: inline-block;
        white-space: nowrap;
      }}
    </style></head><body>
      <span id="mi" class="mi">settings</span>
      <span id="ms" class="ms">settings</span>
      <span id="plain" class="plain">settings</span>
    </body></html>
  """)
  metrics = page.evaluate("""async () => {
    await document.fonts.ready;
    return {
      miLoaded: document.fonts.check('24px "Material Icons"'),
      msLoaded: document.fonts.check('24px "Material Symbols Outlined"'),
      miWidth: document.getElementById('mi').getBoundingClientRect().width,
      msWidth: document.getElementById('ms').getBoundingClientRect().width,
      plainWidth: document.getElementById('plain').getBoundingClientRect().width,
    };
  }""")
  assert metrics["miLoaded"], "Material Icons WOFF2 failed to load"
  assert metrics["msLoaded"], "Material Symbols Outlined WOFF2 failed to load"
  # Ligature glyphs render near 24px square, far narrower than raw "settings".
  assert 16 <= metrics["miWidth"] <= 32, f"Raw ligature? {metrics}"
  assert 16 <= metrics["msWidth"] <= 32, f"Raw ligature? {metrics}"
  assert metrics["miWidth"] < metrics["plainWidth"] * 0.5
  assert metrics["msWidth"] < metrics["plainWidth"] * 0.5

  if os.environ.get("XPROF_UI_REQUIRE_LOCAL_FONTS") == "1":
    required_families = [
        "Roboto",
        "Roboto Mono",
        "Google Sans",
        "Google Sans Flex",
        "Google Sans Code",
    ]
    missing = page.evaluate(
        """(families) => {
          const sample = 'AgQ0123456789@#%&';
          const render = (fontSpec) => {
            const canvas = document.createElement('canvas');
            canvas.width = 640;
            canvas.height = 80;
            const ctx = canvas.getContext('2d');
            ctx.fillStyle = '#000';
            ctx.font = fontSpec;
            ctx.fillText(sample, 10, 60);
            return canvas.toDataURL();
          };
          const serifBaseline = render('48px serif');
          return families.filter(
              (f) => render(`48px "${f}", serif`) === serifBaseline
          );
        }""",
        required_families,
    )
    assert not missing, f"Local UI test fonts did not load in Blink: {missing}"

  browser_errors.assert_clean("bundled_icon_fonts")
