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
"""End-to-end tests for data consistency, routing, and UI edge cases."""

from collections.abc import Callable
import os
import pathlib
import re
import urllib.parse

from playwright.sync_api import expect
from playwright.sync_api import Page
import pytest

# pylint: disable=g-import-not-at-top
try:
  from tests.ui.conftest import BrowserErrors
  from tests.ui.ui_helpers import assert_healthy
except ImportError:
  from conftest import BrowserErrors
  from ui_helpers import assert_healthy


@pytest.mark.parametrize(
    "run_name,host_name",
    [
        ("tpu-training", None),
        ("megascale-2host", "mega-node-0"),
    ],
)
def test_streaming_trace_viewer_keeps_backend_alive(
    page: Page,
    server_url: str,
    logdir: str,
    resolve_run: Callable[[str], str],
    open_tool: Callable[..., str],
    browser_errors: BrowserErrors,
    run_name: str,
    host_name: str | None,
) -> None:
  """Verifies streaming trace_viewer@ responds HTTP 200 without crashing."""
  resolved = resolve_run(run_name)
  session_path = os.path.join(logdir, resolved)
  target_host = host_name or min(
      trace_path.name.removesuffix(".xplane.pb")
      for trace_path in pathlib.Path(session_path).glob("*.xplane.pb")
  )
  open_tool(resolved, "trace_viewer@", host=target_host)
  expect(page.locator("mat-toolbar")).to_be_visible(timeout=20000)

  query = urllib.parse.urlencode({
      "run": resolved,
      "tag": "trace_viewer@",
      "host": target_host,
      "session_path": session_path,
  })
  data_resp = page.request.get(
      f"{server_url.rstrip('/')}/data/plugin/profile/data?{query}"
  )
  assert data_resp.status == 200
  assert len(data_resp.body()) > 64

  runs_resp = page.request.get(
      f"{server_url.rstrip('/')}/data/plugin/profile/runs"
  )
  assert runs_resp.status == 200
  browser_errors.assert_clean(f"trace_viewer@ on {resolved}")


def test_streaming_trace_viewer_gpu_multi_line_tracks(
    page: Page,
    server_url: str,
    logdir: str,
    open_tool: Callable[..., str],
    browser_errors: BrowserErrors,
) -> None:
  """Verifies multi-line async GPU streaming traces build LevelDB cleanly."""
  gpu_dir = os.path.join(logdir, "gpu-training")
  if not os.path.exists(gpu_dir):
    pytest.skip("gpu-training fixture not present in logdir")

  open_tool("gpu-training", "trace_viewer@", host="gpu-node-1")
  expect(page.locator("mat-toolbar")).to_be_visible(timeout=20000)

  query = urllib.parse.urlencode({
      "run": "gpu-training",
      "tag": "trace_viewer@",
      "host": "gpu-node-1",
      "session_path": gpu_dir,
  })
  resp = page.request.get(
      f"{server_url.rstrip('/')}/data/plugin/profile/data?{query}"
  )
  assert resp.status == 200
  assert len(resp.body()) > 1000
  alive = page.request.get(f"{server_url.rstrip('/')}/data/plugin/profile/runs")
  assert alive.status == 200
  browser_errors.assert_clean("gpu trace_viewer@")


@pytest.mark.parametrize("run_name", ["perf-utilization", "perf-v4"])
def test_perf_counters_renders_filters_and_hex_values(
    page: Page,
    open_tool: Callable[..., str],
    browser_errors: BrowserErrors,
    run_name: str,
) -> None:
  """Verifies Performance Counters renders filter controls and hex rows."""
  open_tool(run_name, "perf_counters")
  comp = page.locator("perf-counters")
  expect(comp).to_be_visible(timeout=20000)

  filters = comp.locator(
      "category-filter mat-form-field, string-filter mat-form-field"
  )
  expect(filters.first).to_be_visible(timeout=15000)
  assert filters.count() >= 4

  expect(comp).to_contain_text("tcs_count_cycles", timeout=15000)
  expect(comp).to_contain_text("0x1ab3f0", timeout=15000)
  assert_healthy(page, browser_errors, f"perf_counters_{run_name}")


def test_utilization_viewer_renders_filters_and_unit_charts(
    page: Page,
    open_tool: Callable[..., str],
    browser_errors: BrowserErrors,
) -> None:
  """Verifies Utilization Viewer mounts Host/Device/Sample filters and SVGs."""
  open_tool("perf-utilization", "utilization_viewer")
  comp = page.locator("utilization-viewer")
  expect(comp).to_be_visible(timeout=20000)

  filters = comp.locator("category-filter mat-form-field")
  expect(filters).to_have_count(3, timeout=15000)

  charts = comp.locator("svg")
  expect(charts.first).to_be_visible(timeout=15000)
  assert charts.count() >= 1
  assert_healthy(page, browser_errors, "utilization_viewer")


def test_kernel_utilization_routes_to_viewer_and_hides_raw_menu_tag(
    page: Page,
    open_tool: Callable[..., str],
    browser_errors: BrowserErrors,
) -> None:
  """Verifies kernel_utilization routes to UtilizationViewer, not EmptyPage."""
  open_tool("perf-utilization", "kernel_utilization")
  expect(page.locator("utilization-viewer")).to_be_visible(timeout=20000)
  expect(page.locator("empty-page")).to_have_count(0)

  tools_select = page.locator(
      "sidenav .item-container:has-text('Tools') mat-select"
  )
  expect(tools_select).to_be_visible(timeout=10000)
  tools_select.click()
  options = page.locator("mat-option")
  expect(options.first).to_be_visible(timeout=5000)
  texts = [text.strip() for text in options.all_inner_texts()]
  assert "Utilization Viewer" in texts
  assert "kernel_utilization" not in texts
  page.locator(".cdk-overlay-backdrop").first.click()
  expect(options).to_have_count(0)

  open_tool("perf-utilization", "overview_page", new_nav="true")
  drawer = page.locator("rack-navigation-drawer")
  expect(drawer).to_be_visible(timeout=15000)
  util_item = drawer.locator(
      "rack-navigation-item", has_text="Utilization Viewer"
  ).first
  expect(util_item).to_be_visible(timeout=10000)
  assert "kernel_utilization" not in drawer.inner_text()
  assert_healthy(page, browser_errors, "kernel_utilization_routing")


def test_megascale_stats_single_host_shows_stats_panel(
    page: Page,
    open_tool: Callable[..., str],
    browser_errors: BrowserErrors,
) -> None:
  """Verifies Megascale Stats shows and loads its stats table on one host."""
  open_tool("megascale-2host", "megascale_stats")
  expect(page.locator("megascale-stats")).to_be_visible(timeout=20000)
  assert_healthy(page, browser_errors, "megascale_2host")

  open_tool("megascale-1host", "megascale_stats")
  expect(
      page.locator("main-page mat-sidenav-content > div.full-height")
  ).to_be_visible(timeout=20000)
  expect(page.locator("megascale-stats")).to_be_visible(timeout=20000)
  expect(page.locator("megascale-stats mat-accordion")).to_be_visible(
      timeout=15000
  )
  header = page.locator("megascale-stats mat-expansion-panel-header").first
  header.click()
  expect(header).to_have_attribute("aria-expanded", "true", timeout=10000)
  expect(
      page.locator(
          "megascale-stats .google-visualization-table-table tbody tr"
      ).first
  ).to_be_visible(timeout=20000)
  assert_healthy(page, browser_errors, "megascale_1host")


def test_roofline_bound_by_tooltip_and_graph_viewer_links(
    page: Page,
    open_tool: Callable[..., str],
    browser_errors: BrowserErrors,
) -> None:
  """Verifies Graph Viewer op links and Roofline Bound By scatter tooltips."""
  open_tool("tpu-training", "hlo_stats")
  hlo_links = page.locator("hlo-stats a[href*='graph_viewer']")
  expect(hlo_links.first).to_be_attached(timeout=20000)

  open_tool("tpu-training", "roofline_model")
  roof_links = page.locator("roofline-model a[href*='graph_viewer']")
  expect(roof_links.first).to_be_attached(timeout=20000)

  scatter = page.locator(
      "operation-level-analysis chart[charttype='ScatterChart'] svg"
  ).first
  expect(scatter).to_be_visible(timeout=20000)
  expect(
      page.locator("main-page > mat-sidenav-container > .mat-drawer-animating")
  ).to_have_count(0, timeout=10000)
  scatter.scroll_into_view_if_needed()
  expect(scatter).to_be_in_viewport(timeout=5000)

  circles = scatter.locator("circle")
  expect(circles.first).to_be_visible(timeout=10000)
  target_pt = scatter.evaluate("""(svg) => {
    const svgRect = svg.getBoundingClientRect();
    for (const circle of svg.querySelectorAll('circle')) {
      const fill = (circle.getAttribute('fill') || '').toLowerCase();
      const rect = circle.getBoundingClientRect();
      const cx = rect.x + rect.width / 2;
      const cy = rect.y + rect.height / 2;
      if (
          fill && fill !== 'none' && fill !== '#ffffff' &&
          rect.width >= 4 && rect.x < svgRect.x + svgRect.width * 0.72 &&
          document.elementFromPoint(cx, cy) === circle
      ) {
        return {x: cx, y: cy};
      }
    }
    return null;
  }""")
  assert target_pt is not None
  page.mouse.click(target_pt["x"], target_pt["y"])
  tooltip = page.locator(".google-visualization-tooltip")
  expect(tooltip.first).to_contain_text("Bound By:", timeout=10000)
  assert_healthy(page, browser_errors, "roofline_tooltip_and_links")


def test_new_nav_drawer_groups_and_new_nav_persistence(
    page: Page,
    open_tool: Callable[..., str],
    browser_errors: BrowserErrors,
) -> None:
  """Verifies the ?new_nav=true drawer renders groups and keeps new_nav."""
  open_tool("tpu-training", "overview_page", new_nav="true")
  drawer = page.locator("rack-navigation-drawer")
  expect(drawer).to_be_visible(timeout=20000)
  expect(drawer.locator("rack-navigation-group").first).to_be_visible(
      timeout=10000
  )

  hlo_item = drawer.locator("rack-navigation-item", has_text="HLO Op Stats")
  expect(hlo_item.first).to_be_visible(timeout=10000)
  hlo_item.first.click()
  expect(page.locator("hlo-stats")).to_be_visible(timeout=20000)
  expect(page).to_have_url(re.compile(r"new_nav=true"), timeout=15000)
  assert_healthy(page, browser_errors, "new_nav_drawer")


def test_narrow_viewport_keeps_side_panel_closed_on_first_load(
    page: Page,
    server_url: str,
    logdir: str,
    resolve_run: Callable[[str], str],
    browser_errors: BrowserErrors,
) -> None:
  """Verifies a first load below 960px leaves the overlay side panel closed."""
  run = resolve_run("tpu-training")
  session_path = os.path.join(logdir, run)
  page.set_viewport_size({"width": 768, "height": 1024})
  query = urllib.parse.urlencode({"run": run, "session_path": session_path})
  page.goto(f"{server_url.rstrip('/')}/?{query}", wait_until="domcontentloaded")
  expect(page.locator("overview-page, overview-viewer").first).to_be_visible(
      timeout=20000
  )
  expect(
      page.locator("main-page > mat-sidenav-container > mat-sidenav")
  ).not_to_have_class(re.compile(r"\bmat-drawer-opened\b"), timeout=10000)
  assert_healthy(page, browser_errors, "narrow_viewport_first_load")


def test_ui_rendered_metrics_match_backend_data_json(
    page: Page,
    server_url: str,
    logdir: str,
    resolve_run: Callable[[str], str],
    open_tool: Callable[..., str],
    browser_errors: BrowserErrors,
) -> None:
  """Verifies UI text matches backend /data JSON without hardcoded constants."""
  tpu_run = resolve_run("tpu-training")
  tpu_path = os.path.join(logdir, tpu_run)
  tpu_host = min(
      trace_path.name.removesuffix(".xplane.pb")
      for trace_path in pathlib.Path(tpu_path).glob("*.xplane.pb")
  )

  # 1. Overview Page step-time average matches /data JSON property.
  ov_q = urllib.parse.urlencode({
      "run": tpu_run,
      "tag": "overview_page",
      "host": tpu_host,
      "session_path": tpu_path,
  })
  ov_json = page.request.get(
      f"{server_url.rstrip('/')}/data/plugin/profile/data?{ov_q}"
  ).json()
  perf_summary_tbl = next(
      table
      for table in ov_json
      if isinstance(table, dict)
      and "steptime_ms_average" in (table.get("p") or {})
  )
  step_avg = str(perf_summary_tbl["p"]["steptime_ms_average"])
  open_tool(tpu_run, "overview_page", host=tpu_host)
  expect(page.locator("overview-page, overview-viewer").first).to_contain_text(
      step_avg, timeout=20000
  )

  # 2. HLO Stats top-ranked op name matches row 0 of /data JSON.
  hlo_q = urllib.parse.urlencode({
      "run": tpu_run,
      "tag": "hlo_stats",
      "host": tpu_host,
      "session_path": tpu_path,
  })
  hlo_json = page.request.get(
      f"{server_url.rstrip('/')}/data/plugin/profile/data?{hlo_q}"
  ).json()
  hlo_tbl = hlo_json[0] if isinstance(hlo_json, list) else hlo_json
  cols = [col["id"] for col in hlo_tbl["cols"]]
  op_idx = cols.index("hlo_op_name")
  top_op = str(hlo_tbl["rows"][0]["c"][op_idx]["v"])
  open_tool(tpu_run, "hlo_stats", host=tpu_host)
  expect(page.locator("hlo-stats")).to_contain_text(top_op, timeout=20000)

  # 3. Performance Counters counter name and hex value match /data JSON.
  pc_path = os.path.join(logdir, "perf-utilization")
  pc_q = urllib.parse.urlencode({
      "run": "perf-utilization",
      "tag": "perf_counters",
      "host": "perf-node-1",
      "session_path": pc_path,
  })
  pc_json = page.request.get(
      f"{server_url.rstrip('/')}/data/plugin/profile/data?{pc_q}"
  ).json()
  pc_tbl = pc_json[0] if isinstance(pc_json, list) else pc_json
  pc_cols = [col["id"] for col in pc_tbl["cols"]]
  name_idx = pc_cols.index("Counter")
  val_idx = pc_cols.index("Value")
  counter_name = str(pc_tbl["rows"][0]["c"][name_idx]["v"])
  counter_hex = hex(int(pc_tbl["rows"][0]["c"][val_idx]["v"]))
  open_tool("perf-utilization", "perf_counters", host="perf-node-1")
  expect(page.locator("perf-counters")).to_contain_text(
      counter_name, timeout=20000
  )
  expect(page.locator("perf-counters")).to_contain_text(
      counter_hex, timeout=20000
  )
  assert_healthy(page, browser_errors, "ui_backend_json_consistency")
