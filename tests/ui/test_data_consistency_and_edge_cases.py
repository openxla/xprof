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
import json
import os
import pathlib
import re
from typing import Any
import urllib.parse

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import expect
from playwright.sync_api import Page
from playwright.sync_api import Route
import pytest

# pylint: disable=g-import-not-at-top
try:
  from tests.ui.conftest import BrowserErrors
  from tests.ui.ui_helpers import assert_healthy
except ImportError:
  from conftest import BrowserErrors
  from ui_helpers import assert_healthy

_RUN_TOOLS_ROUTE_RE = re.compile(
    r"^[^?#]*/data/plugin/profile/run_tools(?:[?#]|$)"
)


def _first_xplane_host(session_dir: str) -> str:
  """Returns the lexicographically first `.xplane.pb` host in `session_dir`."""
  return min(
      trace_path.name.removesuffix(".xplane.pb")
      for trace_path in pathlib.Path(session_dir).glob("*.xplane.pb")
  )


def _fetch_tool_data_json(
    page: Page,
    server_url: str,
    run: str,
    tag: str,
    host: str,
    session_path: str,
    min_body_bytes: int = 0,
) -> Any:
  """Fetches `/data/plugin/profile/data` JSON and asserts an HTTP 200 status."""
  query = urllib.parse.urlencode({
      "run": run,
      "tag": tag,
      "host": host,
      "session_path": session_path,
  })
  resp = page.request.get(
      f"{server_url.rstrip('/')}/data/plugin/profile/data?{query}"
  )
  assert resp.status == 200
  assert len(resp.body()) > min_body_bytes
  return resp.json()


def _assert_streaming_trace_viewer_alive(
    page: Page,
    server_url: str,
    open_tool: Callable[..., str],
    browser_errors: BrowserErrors,
    run: str,
    host: str,
    session_path: str,
    min_body_bytes: int = 64,
) -> None:
  """Opens `trace_viewer@` and verifies trace events and backend liveness."""
  open_tool(run, "trace_viewer@", host=host)
  trace_view = page.locator(
      "#filter-bar, .filter-bar, trace-viewer iframe, iframe"
  )
  expect(trace_view.first).to_be_visible(timeout=20000)

  payload = _fetch_tool_data_json(
      page,
      server_url,
      run,
      "trace_viewer@",
      host,
      session_path,
      min_body_bytes=min_body_bytes,
  )
  assert any(
      ev.get("ph") in ("X", "b", "e")
      for ev in payload.get("traceEvents", [])
  ), f"Expected non-metadata trace events in {run}"

  runs_resp = page.request.get(
      f"{server_url.rstrip('/')}/data/plugin/profile/runs"
  )
  assert runs_resp.status == 200
  browser_errors.assert_clean(f"trace_viewer@ on {run}")


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
    resolve_run_dir: Callable[[str], tuple[str, str]],
    open_tool: Callable[..., str],
    browser_errors: BrowserErrors,
    run_name: str,
    host_name: str | None,
) -> None:
  """Verifies streaming trace_viewer@ responds HTTP 200 without crashing."""
  resolved, session_path = resolve_run_dir(run_name)
  target_host = host_name or _first_xplane_host(session_path)
  _assert_streaming_trace_viewer_alive(
      page,
      server_url,
      open_tool,
      browser_errors,
      resolved,
      target_host,
      session_path,
  )


def test_streaming_trace_viewer_gpu_multi_line_tracks(
    page: Page,
    server_url: str,
    synthesized_logdir: str,
    open_tool: Callable[..., str],
    browser_errors: BrowserErrors,
) -> None:
  """Verifies multi-line async GPU streaming traces build LevelDB cleanly."""
  gpu_dir = os.path.join(synthesized_logdir, "gpu-training")
  if not os.path.isdir(gpu_dir):
    if os.environ.get("XPROF_STAGE_SYNTHETIC_RUNS") == "1":
      pytest.fail(
          f"Expected staged gpu-training fixture under {synthesized_logdir}"
      )
    pytest.skip("gpu-training fixture not present in logdir")

  _assert_streaming_trace_viewer_alive(
      page,
      server_url,
      open_tool,
      browser_errors,
      "gpu-training",
      "gpu-node-1",
      gpu_dir,
      min_body_bytes=1000,
  )


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
  expect(filters.nth(3)).to_be_visible(timeout=15000)
  expect(comp.locator("category-filter[column='Device']")).to_have_count(0)
  expect(comp).not_to_contain_text("Filter by undefined")

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

  charts = comp.locator(".bar-chart chart[charttype='BarChart'] svg")
  expect(charts.first).to_be_visible(timeout=15000)
  expect(charts.nth(1)).to_be_visible(timeout=15000)
  for idx in (0, 1):
    bbox = charts.nth(idx).bounding_box()
    assert bbox and bbox["width"] > 0 and bbox["height"] > 0, (
        f"BarChart[{idx}] collapsed: {bbox}"
    )
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
  expect(header).to_be_visible(timeout=20000)
  header.click()
  expect(header).to_have_attribute("aria-expanded", "true", timeout=10000)
  rows = page.locator(
      "megascale-stats .google-visualization-table-table tbody tr"
  )
  expect(rows.first).to_be_visible(timeout=20000)
  expect(page.locator("megascale-stats")).to_contain_text(
      "all_reduce_dcn_0", timeout=15000
  )
  assert_healthy(page, browser_errors, "megascale_1host")


def test_empty_memory_profile_renders_empty_state_cleanly(
    page: Page,
    open_tool: Callable[..., str],
    browser_errors: BrowserErrors,
) -> None:
  """Verifies Memory Profile renders its zero-allocator empty state cleanly."""
  open_tool("empty-memory", "memory_profile")
  comp = page.locator("memory-profile")
  expect(comp).to_be_visible(timeout=20000)
  expect(comp.locator(".error-message").first).to_contain_text(
      "There is no memory profile to display", timeout=15000
  )
  expect(comp.locator("memory-profile-summary")).to_have_count(0)
  assert_healthy(page, browser_errors, "empty_memory_profile")


def _enable_inference_profile_tool(route: Route) -> None:
  """Appends `inference_profile` to `run_tools` while disabled by default."""
  resp = route.fetch()
  tools = resp.json()
  if isinstance(tools, list) and "inference_profile" not in tools:
    tools = [*tools, "inference_profile"]
  route.fulfill(
      status=resp.status,
      content_type="application/json",
      body=json.dumps(tools),
  )


def test_inference_profile_renders_serving_tables_and_metrics(
    page: Page,
    open_tool: Callable[..., str],
    browser_errors: BrowserErrors,
) -> None:
  """Verifies Inference Profile renders request and batch tables on serving."""
  page.route(_RUN_TOOLS_ROUTE_RE, _enable_inference_profile_tool)
  try:
    open_tool("tpu-serving", "inference_profile", host="serving-node-1")
    comp = page.locator("inference-profile")
    expect(comp).to_be_visible(timeout=20000)
    expect(comp).to_contain_text("Request level analysis", timeout=15000)
    expect(comp).to_contain_text("Batch level analysis", timeout=15000)
    expect(
        comp.locator(".google-visualization-table-table tbody tr").first
    ).to_be_visible(timeout=20000)
    assert_healthy(page, browser_errors, "inference_profile_tpu_serving")
  finally:
    page.unroute(_RUN_TOOLS_ROUTE_RE)


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
  hlo_link = hlo_item.first.locator("a.nav-link")
  expect(hlo_link).to_have_attribute(
      "href", re.compile(r"hlo_stats\?.*new_nav=true")
  )
  hlo_item.first.click()
  expect(page.locator("hlo-stats")).to_be_visible(timeout=20000)
  expect(page).to_have_url(re.compile(r"new_nav=true"), timeout=15000)
  assert_healthy(page, browser_errors, "new_nav_drawer")


def test_narrow_viewport_keeps_side_panel_closed_on_first_load(
    page: Page,
    server_url: str,
    resolve_run_dir: Callable[[str], tuple[str, str]],
    browser_errors: BrowserErrors,
) -> None:
  """Verifies a first load below 960px leaves the overlay side panel closed."""
  run, session_path = resolve_run_dir("tpu-training")
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


def test_overview_page_step_time_matches_backend_json(
    page: Page,
    server_url: str,
    resolve_run_dir: Callable[[str], tuple[str, str]],
    open_tool: Callable[..., str],
    browser_errors: BrowserErrors,
) -> None:
  """Verifies Overview Page average step time matches backend /data JSON."""
  tpu_run, tpu_path = resolve_run_dir("tpu-training")
  tpu_host = _first_xplane_host(tpu_path)
  ov_json = _fetch_tool_data_json(
      page, server_url, tpu_run, "overview_page", tpu_host, tpu_path
  )
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
  assert_healthy(page, browser_errors, "overview_backend_json_consistency")


def test_hlo_stats_top_op_matches_backend_json(
    page: Page,
    server_url: str,
    resolve_run_dir: Callable[[str], tuple[str, str]],
    open_tool: Callable[..., str],
    browser_errors: BrowserErrors,
) -> None:
  """Verifies HLO Op Stats top-ranked op name matches backend /data JSON."""
  tpu_run, tpu_path = resolve_run_dir("tpu-training")
  tpu_host = _first_xplane_host(tpu_path)
  hlo_json = _fetch_tool_data_json(
      page, server_url, tpu_run, "hlo_stats", tpu_host, tpu_path
  )
  hlo_tbl = hlo_json[0] if isinstance(hlo_json, list) else hlo_json
  cols = [col["id"] for col in hlo_tbl["cols"]]
  op_idx = cols.index("hlo_op_name")
  top_op = str(hlo_tbl["rows"][0]["c"][op_idx]["v"])
  open_tool(tpu_run, "hlo_stats", host=tpu_host)
  expect(page.locator("hlo-stats")).to_contain_text(top_op, timeout=20000)
  assert_healthy(page, browser_errors, "hlo_stats_backend_json_consistency")


def test_perf_counters_values_match_backend_json(
    page: Page,
    server_url: str,
    resolve_run_dir: Callable[[str], tuple[str, str]],
    open_tool: Callable[..., str],
    browser_errors: BrowserErrors,
) -> None:
  """Verifies Performance Counters name and hex value match backend JSON."""
  pc_run, pc_path = resolve_run_dir("perf-utilization")
  pc_json = _fetch_tool_data_json(
      page, server_url, pc_run, "perf_counters", "perf-node-1", pc_path
  )
  pc_tbl = pc_json[0] if isinstance(pc_json, list) else pc_json
  pc_cols = [col["id"] for col in pc_tbl["cols"]]
  name_idx = pc_cols.index("Counter")
  val_idx = pc_cols.index("Value")
  counter_name = str(pc_tbl["rows"][0]["c"][name_idx]["v"])
  counter_hex = hex(int(pc_tbl["rows"][0]["c"][val_idx]["v"]))
  open_tool(pc_run, "perf_counters", host="perf-node-1")
  expect(page.locator("perf-counters")).to_contain_text(
      counter_name, timeout=20000
  )
  expect(page.locator("perf-counters")).to_contain_text(
      counter_hex, timeout=20000
  )
  assert_healthy(page, browser_errors, "perf_counters_backend_json_consistency")


@pytest.mark.parametrize("hold_loader", [False, True])
def test_framework_op_stats_cold_load_renders_pie_charts(
    page: Page,
    open_tool: Callable[..., str],
    browser_errors: BrowserErrors,
    hold_loader: bool,
) -> None:
  """Verifies Framework Op Stats renders pie charts when gcharts loads late.

  Without `hold_loader`, profile data arrives while the chart packages load.
  With it, `loader.js` is held too, so the app boots on its 2.5s fallback
  timer and profile data arrives before `google` is defined.
  """
  held_routes: list[Route] = []
  release_charts = False

  def delay_gcharts(route: Route) -> None:
    is_loader = route.request.url.endswith("/charts/loader.js")
    if release_charts or (is_loader and not hold_loader):
      route.continue_()
    else:
      held_routes.append(route)

  gcharts_pattern = re.compile(r"^https?://[^/]*gstatic\.(?:com|cn)/charts/")
  page.route(gcharts_pattern, delay_gcharts)
  try:
    with page.expect_response(
        lambda response: "/data/plugin/profile/data" in response.url
        and "tag=framework_op_stats" in response.url,
        timeout=25000,
    ):
      open_tool("tpu-training", "framework_op_stats")
    expect(page.locator("framework-op-stats stats-table")).to_be_visible(
        timeout=15000
    )
  finally:
    # Release held requests before unrouting. Unrouting resumes every request
    # that the handler still holds, and resuming one twice is an error.
    release_charts = True
    try:
      for held in tuple(held_routes):
        try:
          held.continue_()
        except PlaywrightError:
          # The held route may have already been aborted, fulfilled, or closed.
          pass
    finally:
      page.unroute(gcharts_pattern, delay_gcharts)

  pies = page.locator("framework-op-stats chart[charttype='PieChart'] svg")
  expect(pies.first).to_be_visible(timeout=20000)
  expect(pies.nth(1)).to_be_visible(timeout=20000)
  expect(pies.first.locator("path").first).to_be_visible(timeout=20000)
  expect(pies.nth(1).locator("path").first).to_be_visible(timeout=20000)
  assert_healthy(page, browser_errors, "framework_op_stats_cold_load")


def test_standalone_stack_trace_page_direct_load_and_download_hlo_run(
    page: Page,
    server_url: str,
    resolve_run_dir: Callable[[str], tuple[str, str]],
    open_tool: Callable[..., str],
    browser_errors: BrowserErrors,
) -> None:
  """Verifies direct /stack_trace_page load and DownloadHlo run parameter."""
  tpu_run, tpu_path = resolve_run_dir("tpu-training")
  cfg_body = (
      '{"hideCaptureProfileButton": false, "srcPathPrefix": "/workspace"}'
  )
  config_route = "**/data/plugin/profile/config*"
  page.route(
      config_route,
      lambda route: route.fulfill(
          status=200, content_type="application/json", body=cfg_body
      ),
  )
  try:
    query = urllib.parse.urlencode({
        "hlo_module": "jit_train_step",
        "hlo_op": "fusion.1",
        "session_path": tpu_path,
    })
    encoded_run = urllib.parse.quote(tpu_run, safe="")
    direct_url = (
        f"{server_url.rstrip('/')}/data/plugin/profile/"
        f"stack_trace_page/{encoded_run}?{query}"
    )
    page.goto(direct_url, wait_until="domcontentloaded")
    expect(
        page.locator("main-page mat-sidenav-content > div.full-height")
    ).to_be_visible(timeout=20000)
    expect(page.locator("stack-trace-page .container")).to_be_visible(
        timeout=20000
    )
    expect(page.locator("stack-trace-page .container")).to_contain_text(
        "fusion.1", timeout=15000
    )
    assert_healthy(page, browser_errors, "standalone_stack_trace_page")
  finally:
    page.unroute(config_route)

  open_tool("tpu-training", "graph_viewer")
  dl_btn = page.locator("graph-viewer download-hlo .save-button").first
  expect(dl_btn).to_be_visible(timeout=20000)
  dl_btn.click()
  pbtxt_item = page.locator(
      "button[mat-menu-item]", has_text="Download as .pbtxt"
  ).first
  expect(pbtxt_item).to_be_visible(timeout=10000)
  with page.expect_request(
      lambda req: "/data/plugin/profile/data" in req.url
      and "tag=graph_viewer" in req.url
      and "type=pbtxt" in req.url,
      timeout=15000,
  ) as dl_req_info:
    pbtxt_item.click()
  expect(pbtxt_item).to_be_hidden(timeout=5000)
  dl_query = urllib.parse.parse_qs(
      urllib.parse.urlsplit(dl_req_info.value.url).query
  )
  assert dl_query.get("run") == [tpu_run]
  assert_healthy(page, browser_errors, "download_hlo_query_run")


@pytest.mark.parametrize(
    "tag,selector",
    [
        ("utilization_viewer", "utilization-viewer"),
        ("perf_counters", "perf-counters"),
    ],
)
def test_utilization_and_perf_counters_csv_export_include_host(
    page: Page,
    server_url: str,
    open_tool: Callable[..., str],
    browser_errors: BrowserErrors,
    tag: str,
    selector: str,
) -> None:
  """Verifies Export as CSV includes host on Utilization and PerfCounters."""
  open_tool("perf-utilization", tag, host="perf-node-1")
  comp = page.locator(selector)
  expect(comp).to_be_visible(timeout=20000)
  page.evaluate(
      "() => {"
      "  delete document.body.dataset.lastCsvUrl;"
      "  const origOpen = window.open;"
      "  window.__restoreOpen = () => { window.open = origOpen; };"
      "  window.open = (url) => {"
      "    document.body.dataset.lastCsvUrl = String(url);"
      "    return null;"
      "  };"
      "}"
  )
  try:
    save_btn = comp.locator("export-as-csv .save-button").first
    expect(save_btn).to_be_visible(timeout=15000)
    save_btn.click()
    expect(page.locator("body")).to_have_attribute(
        "data-last-csv-url",
        re.compile(r"/data/plugin/profile/data_csv\?"),
        timeout=5000,
    )
    csv_url = page.locator("body").get_attribute("data-last-csv-url")
    assert csv_url is not None
    csv_query = urllib.parse.parse_qs(urllib.parse.urlsplit(csv_url).query)
    assert csv_query.get("host") == ["perf-node-1"]
    csv_response = page.request.get(
        urllib.parse.urljoin(server_url.rstrip("/") + "/", csv_url)
    )
    assert csv_response.status == 200 and len(csv_response.body()) > 0
  finally:
    page.evaluate(
        "() => {"
        "  if (window.__restoreOpen) window.__restoreOpen();"
        "  delete window.__restoreOpen;"
        "  delete document.body.dataset.lastCsvUrl;"
        "}"
    )
  assert_healthy(page, browser_errors, f"{tag}_csv_export_host")


@pytest.mark.parametrize(
    "path_template,expected_base",
    [
        (
            "/proxy/sub/data/plugin/profile/stack_trace_page/run1",
            "/proxy/sub/data/plugin/profile/",
        ),
        ("/proxy/data/plugin/profile", "/proxy/data/plugin/profile/"),
        ("/data/plugin/profile/", "/data/plugin/profile/"),
        ("//evil.example/data/plugin/profile/", "/data/plugin/profile/"),
        (
            "//{netloc}//evil.example/data/plugin/profile/",
            "/data/plugin/profile/",
        ),
    ],
)
def test_index_html_resolves_same_origin_base_href(
    page: Page,
    server_url: str,
    browser_errors: BrowserErrors,
    path_template: str,
    expected_base: str,
) -> None:
  """Verifies index.html rewrites <base href> only for same-origin prefixes."""
  index_response = page.request.get(
      f"{server_url.rstrip('/')}/data/plugin/profile/index.html"
  )
  assert index_response.status == 200
  index_html = index_response.text()
  server_parts = urllib.parse.urlsplit(server_url)
  request_path = path_template.format(netloc=server_parts.netloc)
  browser_errors.ignore("net::ERR_FAILED")

  def serve_index_only(route: Route) -> None:
    if route.request.resource_type == "document":
      route.fulfill(status=200, content_type="text/html", body=index_html)
    else:
      route.abort()

  page.route("**/*", serve_index_only)
  try:
    page.goto(
        f"{server_parts.scheme}://{server_parts.netloc}{request_path}",
        wait_until="domcontentloaded",
    )
    expect(page.locator("head > base")).to_have_attribute("href", expected_base)
  finally:
    page.unroute("**/*", serve_index_only)
  browser_errors.assert_clean("index_html_base_href")

