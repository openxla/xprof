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
"""Tests verifying functional and visual data rendering across core tools."""

from collections.abc import Callable
import contextlib
import json
import os
import pathlib
import re
import urllib.parse

from playwright.sync_api import expect
from playwright.sync_api import Page
from playwright.sync_api import Route
import pytest

from tests.ui.conftest import BrowserErrors
from tests.ui.invariants import check_no_horizontal_overflow
from tests.ui.ui_helpers import assert_component_geometry
from tests.ui.ui_helpers import assert_healthy
from tests.ui.ui_helpers import build_tool_url
from tests.ui.ui_helpers import enabling_run_tool
from tests.ui.ui_helpers import ensure_inference_serving_run
from tests.ui.ui_helpers import ensure_sidenav_open
from tests.ui.ui_helpers import expand_op_profile_row
from tests.ui.ui_helpers import select_category_filter
from tests.ui.ui_helpers import select_memory_id
from tests.ui.ui_helpers import select_module
from tests.ui.ui_helpers import select_session
from tests.ui.ui_helpers import switch_tool


def test_overview_page_deep_components(
    page: Page,
    open_tool: Callable[..., str],
    browser_errors: BrowserErrors,
) -> None:
  """Validates overview summary metrics, step-time chart, and host selector."""
  open_tool("tpu-training", "overview_page")

  overview = page.locator("overview-page, overview-viewer")
  expect(overview).to_be_visible(timeout=20000)

  # 1. Performance Summary metrics
  summary_card = overview.locator("mat-card:has-text('Performance Summary')")
  expect(summary_card).to_be_visible(timeout=10000)
  expect(summary_card).to_contain_text(
      re.compile(r"Average (Tensor Core )?Step Time")
  )
  expect(summary_card).to_contain_text(re.compile(r"(?<!-)\b\d+(\.\d+)?\s*ms"))
  expect(summary_card).to_contain_text("FLOPS Utilization")

  # 2. Step-time Graph geometry
  graph_comp = page.locator("step-time-graph")
  expect(graph_comp).to_be_visible(timeout=20000)
  bbox = graph_comp.bounding_box()
  assert bbox is not None, "Step-time graph bounding box is None"
  assert bbox["width"] >= 200, f"Graph width collapsed: {bbox['width']}px"
  assert bbox["height"] >= 100, f"Graph height collapsed: {bbox['height']}px"

  # 3. Host selector interaction
  ensure_sidenav_open(page)
  host_select = page.locator(
      "sidenav .item-container:has-text('Hosts') mat-select"
  )
  expect(host_select).to_be_visible(timeout=20000)
  host_select.click()
  options = page.locator("mat-option")
  expect(options.first).to_be_visible(timeout=5000)
  first_option_text = options.first.inner_text().strip()
  options.first.click()
  expect(host_select).to_contain_text(first_option_text)

  # 4. Invariant health verification
  assert_healthy(page, browser_errors, "Overview Page")


def test_memory_profile_table_rendering(
    page: Page,
    open_tool: Callable[..., str],
    browser_errors: BrowserErrors,
) -> None:
  """Verifies that the Memory Profile tool mounts its data tables."""
  open_tool("tpu-training", "memory_profile")

  mem_comp = page.locator("memory-viewer, memory-profile").first
  expect(mem_comp).to_be_visible(timeout=20000)

  table_container = mem_comp.locator(
      "memory-breakdown-table .table, memory-breakdown-table"
  ).first
  expect(table_container).to_be_visible(timeout=20000)

  rows = table_container.locator("table tbody tr, tr:has(td)")
  expect(rows.first).to_be_visible(timeout=10000)
  expect(rows.first.locator("td").first).to_be_visible(timeout=5000)
  assert bool(rows.first.inner_text().strip()), "Memory profile row is empty"
  assert_healthy(page, browser_errors, "Memory Profile")


def test_kernel_stats_rendering(
    page: Page,
    synthesized_logdir: str,
    open_tool: Callable[..., str],
    browser_errors: BrowserErrors,
) -> None:
  """Verifies that the GPU Kernel Stats table renders execution rows."""
  # Checked before resolving: on a single-run logdir the resolver would fall
  # back to that (TPU) run.
  if not os.path.exists(os.path.join(synthesized_logdir, "gpu-training")):
    pytest.skip("gpu-training fixture not present in logdir")
  open_tool("gpu-training", "kernel_stats")

  selector = "kernel-stats, kernel-stats-adapter"
  expect(page.locator(selector).first).to_be_visible(timeout=20000)
  expect(
      page.locator("main-page mat-sidenav-content > div.full-height")
  ).to_be_visible(timeout=20000)
  expect(page).to_have_url(re.compile(r"[?&]run=gpu-training\b"), timeout=10000)
  expect(page.locator("text=There is no GPU data to display")).to_be_hidden(
      timeout=10000
  )
  assert_component_geometry(page, selector, "kernel_stats")
  rows = page.locator(selector).first.locator(
      "table tr:has(td), table mat-row, mat-row"
  )
  expect(rows.first).to_be_visible(timeout=10000)
  assert rows.count() >= 1, "Expected kernel stats rows"
  assert_healthy(page, browser_errors, "kernel_stats")


def test_roofline_model_operation_piechart_and_filter_redraw(
    page: Page,
    open_tool: Callable[..., str],
    browser_errors: BrowserErrors,
) -> None:
  """Verifies Roofline Model Operation-Level PieChart draws and updates."""
  open_tool("tpu-training", "overview_page")
  expect(
      page.locator("overview-page mat-card, overview-viewer mat-card").first
  ).to_be_visible(timeout=20000)

  switch_tool(page, "Roofline Model")
  op_analysis = page.locator("operation-level-analysis")
  expect(op_analysis).to_be_visible(timeout=20000)

  scatter_svg = op_analysis.locator("chart[charttype='ScatterChart'] svg").first
  pie_svg = op_analysis.locator("chart[charttype='PieChart'] svg").first
  expect(scatter_svg).to_be_visible(timeout=20000)
  expect(pie_svg).to_be_visible(timeout=10000)
  expect(pie_svg).to_contain_text("Percentage of self time per HLO op category")

  pie_box = pie_svg.bounding_box()
  assert pie_box is not None, "Operation-level PieChart bounding box is None"
  assert (
      pie_box["width"] >= 200 and pie_box["height"] >= 200
  ), f"Operation-level PieChart collapsed: {pie_box}"

  select_category_filter(page, "convolution fusion")
  expect(pie_svg).to_be_visible(timeout=10000)
  expect(pie_svg).to_contain_text("convolution")
  expect(pie_svg).to_contain_text("100%")
  assert_healthy(page, browser_errors, "roofline_model_operation_piechart")


@pytest.mark.parametrize(
    "viewport_name,width,height",
    [("tablet", 768, 1024), ("mobile", 375, 812)],
)
@pytest.mark.parametrize("tag", ["overview_page", "roofline_model"])
def test_responsive_viewport_no_horizontal_clipping(
    page: Page,
    open_tool: Callable[..., str],
    browser_errors: BrowserErrors,
    viewport_name: str,
    width: int,
    height: int,
    tag: str,
) -> None:
  """Verifies Tablet and Mobile viewports do not clip main tool content."""
  page.set_viewport_size({"width": width, "height": height})
  open_tool("tpu-training", tag)

  tool_comp = page.locator(
      "overview-page, overview-viewer, roofline-model"
  ).first
  expect(tool_comp).to_be_visible(timeout=20000)
  expect(
      page.locator("main-page mat-sidenav-content > div.full-height")
  ).to_be_visible(timeout=20000)

  content_pane = page.locator("mat-sidenav-content").first
  expect(content_pane).to_be_visible(timeout=10000)
  pane_box = content_pane.bounding_box()
  assert pane_box is not None, f"mat-sidenav-content missing on {viewport_name}"
  assert pane_box["width"] >= width * 0.6, (
      f"{viewport_name} ({width}x{height}) {tag}: mat-sidenav-content width"
      f" {pane_box['width']:.0f}px is squeezed below 60% of viewport width"
  )

  overflow_px = page.evaluate("""() => {
        const pane = document.querySelector('mat-sidenav-content');
        if (!pane) return 0;
        return pane.scrollWidth - pane.clientWidth;
      }""")
  assert overflow_px <= 16, (
      f"{viewport_name} ({width}x{height}) {tag}: horizontal chart clipping"
      f" detected (overflow={overflow_px}px)"
  )
  assert_healthy(page, browser_errors, f"responsive_{viewport_name}_{tag}")


def test_trace_viewer_mounts_canvas(
    page: Page,
    server_url: str,
    open_tool: Callable[..., str],
    browser_errors: BrowserErrors,
) -> None:
  """Verifies Trace Viewer V2 starts without errors and shows a canvas."""
  # V2 needs its script from the server and a WebGPU adapter in the browser.
  # Both are checked before the app loads, so a skip leaves no console errors.
  v2_script = f"{server_url}/data/plugin/profile/trace_viewer_v2.js"
  if not page.request.get(v2_script).ok:
    pytest.skip("server does not serve Trace Viewer V2")
  page.goto(v2_script)
  expect(page).to_have_url(v2_script)
  if not page.evaluate("async () => !!(await navigator.gpu?.requestAdapter())"):
    pytest.skip("browser has no WebGPU adapter for Trace Viewer V2")
  open_tool("tpu-training", "trace_viewer", use_trace_viewer_v2="true")
  expect(page).to_have_url(
      re.compile(r"[?&]use_trace_viewer_v2=true(?:[&#]|$)"), timeout=15000
  )

  canvas = page.locator("canvas#canvas:visible")
  expect(canvas).to_be_visible(timeout=20000)
  bbox = canvas.bounding_box()
  assert bbox and bbox["width"] > 0 and bbox["height"] > 0
  assert_healthy(page, browser_errors, "trace_viewer_v2")


def test_trace_viewer_lifecycle_teardown(
    page: Page,
    open_tool: Callable[..., str],
    browser_errors: BrowserErrors,
) -> None:
  """Verifies navigating away from Trace Viewer cleanly tears down iframe."""
  open_tool("tpu-training", "trace_viewer")
  trace_el = page.locator(
      "iframe, .trace-viewer-container, #filter-bar"
  ).first
  expect(trace_el).to_be_visible(timeout=20000)
  bbox = trace_el.bounding_box()
  assert bbox and bbox["width"] > 0 and bbox["height"] > 0

  # Switch to Op Profile via UI dropdown
  switch_tool(page, "Op Profile")
  expect(page.locator("op-profile, op-profile-base").first).to_be_visible(
      timeout=20000
  )

  # Verify trace viewer iframe is destroyed
  expect(page.locator("trace-viewer iframe, iframe")).to_have_count(0)
  assert_healthy(page, browser_errors, "trace_viewer_lifecycle_teardown")


def test_tool_switching_cleanup(
    page: Page,
    open_tool: Callable[..., str],
    browser_errors: BrowserErrors,
) -> None:
  """Verifies switching tools cleans up previously mounted views via UI."""
  open_tool("tpu-training", "overview_page")
  expect(
      page.locator("overview-page mat-card, overview-viewer mat-card").first
  ).to_be_visible(timeout=20000)

  switch_tool(page, "Op Profile")
  expect(page.locator("op-profile, op-profile-base").first).to_be_visible(
      timeout=20000
  )
  expect(page.locator("overview-page, overview-viewer")).to_have_count(0)

  switch_tool(page, "Memory Viewer")
  expect(page.locator("memory-viewer, memory-profile").first).to_be_visible(
      timeout=20000
  )
  expect(page.locator("op-profile, op-profile-base")).to_have_count(0)
  assert_healthy(page, browser_errors, "tool_switching_cleanup")


def test_memory_viewer_shows_the_selected_module(
    page: Page,
    open_tool: Callable[..., str],
    browser_errors: BrowserErrors,
) -> None:
  """Verifies Memory Viewer reloads its summary for the selected module."""
  open_tool("tpu-training", "memory_viewer")
  summary = page.locator("memory-viewer-main")
  # Memory Viewer opens on the first module, not the one picked below.
  expect(summary).to_contain_text("Module Name: jit__where(", timeout=20000)

  select_module(page, "jit_train_step")
  expect(summary).to_contain_text("Module Name: jit_train_step(", timeout=20000)
  assert_healthy(page, browser_errors, "memory_viewer_module_selection")


def test_input_pipeline_analysis_deep_components(
    page: Page,
    open_tool: Callable[..., str],
    browser_errors: BrowserErrors,
) -> None:
  """Verifies Input Pipeline Analysis mounts summary, charts, and details."""
  open_tool("tpu-training", "input_pipeline_analyzer")

  pipeline_comp = page.locator("input-pipeline")
  expect(pipeline_comp).to_be_visible(timeout=20000)

  # 1. Summary conclusion
  summary = pipeline_comp.locator("analysis-summary")
  expect(summary).to_be_visible(timeout=10000)
  conclusion = summary.locator(".summary-conclusion")
  expect(conclusion).to_be_visible(timeout=10000)
  expect(conclusion).to_contain_text(re.compile(r"input-bound", re.IGNORECASE))

  # 2. Device-side analysis statistics and charts
  device_detail = pipeline_comp.locator("device-side-analysis-detail")
  expect(device_detail).to_be_visible(timeout=10000)
  expect(device_detail).to_contain_text(
      re.compile(r"Average:\s*\d+(\.\d+)?\s*ms")
  )
  area_chart = device_detail.locator("chart[charttype='AreaChart']")
  expect(area_chart).to_be_visible(timeout=20000)
  area_svg = area_chart.locator("svg").first
  expect(area_svg).to_be_visible(timeout=10000)
  bbox = area_svg.bounding_box()
  assert bbox is not None and bbox["width"] > 0 and bbox["height"] > 0

  # 3. Max infeed detail table and Host-side analysis detail
  max_infeed = pipeline_comp.locator("max-infeed-detail")
  expect(max_infeed).to_be_visible(timeout=10000)
  infeed_table = max_infeed.locator(".table, table")
  expect(infeed_table.first).to_be_visible(timeout=10000)
  expect(infeed_table.first.locator("td, th").first).to_be_visible(timeout=5000)

  host_detail = pipeline_comp.locator("host-side-analysis-detail")
  expect(host_detail).to_be_visible(timeout=10000)
  expect(host_detail).to_contain_text("Host-side analysis details")
  assert_healthy(page, browser_errors, "input_pipeline_deep_components")


_SYNTHETIC_HOST_OP_TABLES: tuple[dict[str, object], ...] = (
    {
        "p": {
            "num_host_op_tables": "1",
            "valid_host_ops": "IteratorGetNext",
            "hostnames": "tpu-worker-0",
            "values": "All-cores",
        },
        "cols": [],
        "rows": [],
    },
    {
        "p": {
            "hostop": "IteratorGetNext",
            "hostname": "tpu-worker-0",
            "value": "All-cores",
        },
        "cols": [
            {
                "id": "tpuStepNumber",
                "label": "TPU Step Number",
                "type": "number",
            },
            {
                "id": "core0",
                "label": "tpu-worker-0:0",
                "type": "number",
            },
        ],
        "rows": [
            {"c": [{"v": 1}, {"v": 2}]},
            {"c": [{"v": 2}, {"v": 3}]},
        ],
    },
)


def _is_host_op_table(tbl: object) -> bool:
  """Returns whether `tbl` is a host-op meta or step-analysis table."""
  props = tbl.get("p") if isinstance(tbl, dict) else None
  return isinstance(props, dict) and (
      "num_host_op_tables" in props or "hostop" in props
  )


def _fulfill_input_pipeline_route(
    route: Route, *, inject_host_op_tables: bool
) -> None:
  """Fulfills an Input Pipeline `/data` route with stripped or synthetic host-ops."""
  resp = route.fetch()
  payload = resp.json()
  if isinstance(payload, list):
    payload = [tbl for tbl in payload if not _is_host_op_table(tbl)]
    if inject_host_op_tables:
      payload.extend(_SYNTHETIC_HOST_OP_TABLES)
  route.fulfill(
      status=resp.status,
      content_type="application/json",
      body=json.dumps(payload),
  )


def _resolve_session_switch_runs(
    logdir: str,
    resolve_run: Callable[[str], str],
    scenario: str,
) -> tuple[str, str]:
  """Resolves `(tpu_run, target_run)` for a host-op session-switch scenario."""
  tpu_run = resolve_run("tpu-training")
  if scenario == "gpu":
    if tpu_run == "gpu-training" or not os.path.exists(
        os.path.join(logdir, "gpu-training")
    ):
      pytest.skip("Requires distinct TPU and gpu-training fixtures in logdir")
    return tpu_run, "gpu-training"

  for candidate in ("v6e-4-training", "multi-host"):
    if candidate != tpu_run and os.path.exists(os.path.join(logdir, candidate)):
      return tpu_run, candidate
  pytest.skip("No secondary TPU fixture present in logdir")


def _make_input_pipeline_route_matcher(
    runs: set[str],
) -> Callable[[str], bool]:
  """Returns a URL predicate matching Input Pipeline `/data` for `runs`."""

  def _matches(url: str) -> bool:
    parsed = urllib.parse.urlsplit(url)
    if not parsed.path.endswith("/data/plugin/profile/data"):
      return False
    params = urllib.parse.parse_qs(parsed.query)
    run = (params.get("run") or [""])[0]
    return run in runs and params.get("tag") == ["input_pipeline_analyzer"]

  return _matches


def _assert_input_pipeline_run_url(page: Page, run: str) -> None:
  """Asserts that the active URL targets `run` on Input Pipeline Analysis."""
  expect(page).to_have_url(
      re.compile(rf"[?&]run={re.escape(run)}\b"), timeout=15000
  )
  expect(page).to_have_url(
      re.compile(r"[?&]tag=input_pipeline_analyzer\b"), timeout=15000
  )


@pytest.mark.parametrize("scenario", ["zero_table_tpu", "gpu"])
def test_input_pipeline_host_op_tables_reset_on_session_switch(
    page: Page,
    server_url: str,
    logdir: str,
    resolve_run: Callable[[str], str],
    browser_errors: BrowserErrors,
    scenario: str,
) -> None:
  """Verifies Host-vs-TPU step analysis hides on 0-table or GPU runs."""
  tpu_run, target_run = _resolve_session_switch_runs(
      logdir, resolve_run, scenario
  )

  def _handle_host_op_route(route: Route) -> None:
    params = urllib.parse.parse_qs(
        urllib.parse.urlsplit(route.request.url).query
    )
    _fulfill_input_pipeline_route(
        route,
        inject_host_op_tables=(params.get("run") or [""])[0] == tpu_run,
    )

  intercepted_runs = (
      {tpu_run, target_run} if scenario == "zero_table_tpu" else {tpu_run}
  )
  route_matcher = _make_input_pipeline_route_matcher(intercepted_runs)
  page.route(route_matcher, _handle_host_op_route)
  try:
    # Keep `run_path=logdir` instead of `build_tool_url` (which sets
    # `session_path` and prevents multi-run `select_session` switching).
    query = urllib.parse.urlencode({
        "run_path": logdir,
        "run": tpu_run,
        "tag": "input_pipeline_analyzer",
    })
    page.goto(
        f"{server_url.rstrip('/')}/?{query}",
        wait_until="domcontentloaded",
    )
    host_op_div = page.locator("host-op > div")
    expect(host_op_div).to_be_visible(timeout=20000)
    expect(host_op_div).to_contain_text("Host-vs-TPU step analysis")

    select_session(page, target_run)
    _assert_input_pipeline_run_url(page, target_run)
    expect(page.locator("input-pipeline")).to_be_visible(timeout=15000)
    expect(page.locator("host-op")).to_be_attached(timeout=15000)
    expect(host_op_div).to_be_hidden(timeout=15000)

    select_session(page, tpu_run)
    _assert_input_pipeline_run_url(page, tpu_run)
    expect(host_op_div).to_be_visible(timeout=15000)
    expect(host_op_div).to_contain_text("Host-vs-TPU step analysis")

    assert_healthy(
        page,
        browser_errors,
        f"input_pipeline_host_op_session_switch_{scenario}",
    )
  finally:
    page.unroute(route_matcher, _handle_host_op_route)


def test_op_profile_interactive_toggles(
    page: Page,
    open_tool: Callable[..., str],
    browser_errors: BrowserErrors,
) -> None:
  """Verifies Op Profile interactive controls and slide toggles update state."""
  open_tool("tpu-training", "op_profile")

  op_profile = page.locator("op-profile, op-profile-base").first
  expect(op_profile).to_be_visible(timeout=20000)

  # 1. Exclude Idle slide toggle
  exclude_idle_control = op_profile.locator(".control:has-text('Exclude Idle')")
  expect(exclude_idle_control).to_be_visible(timeout=10000)
  switch_btn = exclude_idle_control.locator("button[role='switch']")
  expect(switch_btn).to_be_visible(timeout=10000)
  initial_state = switch_btn.get_attribute("aria-checked")
  expected_next = "false" if initial_state == "true" else "true"
  switch_btn.click()
  expect(switch_btn).to_have_attribute(
      "aria-checked", expected_next, timeout=5000
  )

  # 2. Invariant health verification
  assert_healthy(page, browser_errors, "op_profile_interactive_toggles")


def test_op_profile_serving_metadata_chips(
    page: Page,
    server_url: str,
    logdir: str,
    tmp_path: pathlib.Path,
    browser_errors: BrowserErrors,
) -> None:
  """Verifies Op Profile renders model and batch size chips on serving runs."""
  serving_dir = ensure_inference_serving_run(logdir, tmp_path)
  page.goto(
      build_tool_url(server_url, str(serving_dir), "tpu-serving", "op_profile"),
      wait_until="domcontentloaded",
  )

  op_profile = page.locator("op-profile, op-profile-base").first
  expect(op_profile).to_be_visible(timeout=20000)

  serving_badge = op_profile.locator("op-table-entry .serving-badge").first
  expect(serving_badge).to_be_visible(timeout=15000)
  expect(serving_badge).to_have_text("BS 8 \u2022 Model: ranking_model:1")

  expand_op_profile_row(page, "main")
  expect(page.locator("op-details .info-header-subheader")).to_have_text(
      "Program"
  )
  serving_entries = page.locator("op-details .serving-entry")
  expect(serving_entries).to_have_count(2, timeout=10000)
  expect(serving_entries.nth(0)).to_have_text("Model: /models/ranking_model:1")
  expect(serving_entries.nth(1)).to_have_text("Batch size: 8")
  assert_healthy(page, browser_errors, "op_profile_serving_metadata_chips")


def test_hlo_stats_chart_expansion(
    page: Page,
    open_tool: Callable[..., str],
    browser_errors: BrowserErrors,
) -> None:
  """Verifies HLO Op Stats Chart Section expansion renders pie charts."""
  open_tool("tpu-training", "hlo_stats")

  expect(
      page.locator("main-page mat-sidenav-content > div.full-height")
  ).to_be_visible(timeout=20000)
  hlo_comp = page.locator("hlo-stats").first
  expect(hlo_comp).to_be_visible(timeout=20000)

  # 1. Chart Section starts expanded (showChartSection = true); verify toggle
  # collapses and re-expands cleanly with rendered PieChart SVGs.
  header = hlo_comp.locator(
      "mat-expansion-panel-header:has-text('Chart Section')"
  )
  expect(header).to_be_visible(timeout=10000)
  expect(header).to_have_attribute("aria-expanded", "true", timeout=10000)
  header.click()
  expect(header).to_have_attribute("aria-expanded", "false", timeout=10000)
  header.click()
  expect(header).to_have_attribute("aria-expanded", "true", timeout=10000)

  pie_chart = hlo_comp.locator("chart[charttype='PieChart']").first
  expect(pie_chart).to_be_visible(timeout=20000)
  svg = pie_chart.locator("svg").first
  expect(svg).to_be_visible(timeout=15000)
  bbox = svg.bounding_box()
  assert bbox is not None and bbox["width"] > 0 and bbox["height"] > 0

  # 2. Invariant health verification
  assert_healthy(page, browser_errors, "hlo_stats_chart_expansion")


def test_framework_op_stats_idle_toggle_and_table_rendering(
    page: Page,
    open_tool: Callable[..., str],
    browser_errors: BrowserErrors,
) -> None:
  """Verifies Framework Op Stats charts, table, and idle toggle behavior."""
  open_tool("tpu-training", "framework_op_stats")

  framework_comp = page.locator("framework-op-stats").first
  expect(framework_comp).to_be_visible(timeout=20000)

  pie = framework_comp.locator("chart[charttype='PieChart']").first
  expect(pie).to_be_visible(timeout=20000)
  svg = pie.locator("svg").first
  expect(svg).to_be_visible(timeout=15000)
  stats_tbl = framework_comp.locator("stats-table").first
  expect(stats_tbl).to_be_visible(timeout=10000)
  rows = stats_tbl.locator("table tbody tr, tr:has(td)")
  expect(rows.first).to_be_visible(timeout=10000)

  # 1. Idle toggle menu interaction (verify active Yes state before reverting)
  idle_rows = stats_tbl.locator("tr:has-text('IDLE')")
  initial_idle_count = idle_rows.count()
  idle_btn = framework_comp.locator(".idle-button-container button").first
  expect(idle_btn).to_be_visible(timeout=10000)

  idle_btn.click()
  enabled_item = page.locator("button.mat-mdc-menu-item:not([disabled])").first
  expect(enabled_item).to_be_visible(timeout=5000)
  target_text = enabled_item.inner_text().strip()
  enabled_item.click()
  expect(idle_btn).to_contain_text(target_text, timeout=5000)
  expect(idle_rows).not_to_have_count(initial_idle_count, timeout=10000)
  expect(svg).to_be_visible(timeout=15000)
  toggled_bbox = svg.bounding_box()
  assert (
      toggled_bbox is not None
      and toggled_bbox["width"] > 0
      and toggled_bbox["height"] > 0
  )
  expect(rows.first).to_be_visible(timeout=10000)
  assert bool(rows.first.inner_text().strip())

  idle_btn.click()
  return_item = page.locator("button.mat-mdc-menu-item:not([disabled])").first
  expect(return_item).to_be_visible(timeout=5000)
  return_text = return_item.inner_text().strip()
  return_item.click()
  expect(idle_btn).to_contain_text(return_text, timeout=5000)
  expect(idle_rows).to_have_count(initial_idle_count, timeout=10000)

  # 2. Device and Host pie charts rendering
  expect(svg).to_be_visible(timeout=15000)
  bbox = svg.bounding_box()
  assert bbox is not None and bbox["width"] > 0 and bbox["height"] > 0

  # 3. Stats table rows rendering
  expect(stats_tbl).to_be_visible(timeout=10000)
  expect(rows.first).to_be_visible(timeout=10000)
  assert bool(rows.first.inner_text().strip())

  # 4. Invariant health verification
  assert_healthy(page, browser_errors, "framework_op_stats_idle_toggle")


def test_memory_profile_interactive_memory_id_and_timeline(
    page: Page,
    open_tool: Callable[..., str],
    browser_errors: BrowserErrors,
) -> None:
  """Verifies Memory Profile memory ID selector updates timeline and summary."""
  open_tool("tpu-training", "memory_profile")

  mem_comp = page.locator("memory-viewer, memory-profile").first
  expect(mem_comp).to_be_visible(timeout=20000)

  # 1. Summary card and Memory ID selector rendering
  mem_id_selector = page.locator("#memory-id-selector")
  expect(mem_id_selector).to_be_visible(timeout=10000)
  expect(mem_id_selector).to_contain_text("Memory ID", timeout=5000)

  summary = mem_comp.locator("memory-profile-summary").first
  expect(summary).to_be_visible(timeout=15000)
  expect(summary).to_contain_text("Memory Capacity", timeout=10000)
  expect(summary).to_contain_text("Lifetime Peak Heap Usage", timeout=5000)

  # 2. Timeline graph rendering
  timeline = mem_comp.locator("memory-timeline-graph").first
  expect(timeline).to_be_visible(timeout=15000)
  timeline_chart = timeline.locator("chart, svg, canvas").first
  expect(timeline_chart).to_be_visible(timeout=15000)
  bbox = timeline_chart.bounding_box()
  assert bbox is not None and bbox["width"] > 0 and bbox["height"] > 0

  # 3. Interactive non-default memory ID dropdown selection
  mem_select = page.locator("#memory-id-selector mat-select")
  expect(mem_select).to_be_visible(timeout=10000)
  initial_summary = summary.inner_text()
  select_memory_id(page, "1")
  expect(summary).not_to_have_text(initial_summary, timeout=10000)
  expect(summary).to_contain_text("Memory Capacity", timeout=10000)
  expect(timeline_chart).to_be_visible(timeout=15000)

  # 4. Breakdown table rendering
  breakdown = mem_comp.locator("memory-breakdown-table").first
  expect(breakdown).to_be_visible(timeout=10000)
  expect(breakdown).to_contain_text("Memory Breakdown Table", timeout=5000)

  # 5. Invariant health verification
  assert_healthy(page, browser_errors, "memory_profile_interactive_controls")


def test_roofline_model_program_analysis_scatter_and_table(
    page: Page,
    open_tool: Callable[..., str],
    browser_errors: BrowserErrors,
) -> None:
  """Verifies Roofline Model Program-Level scatter plot and summary table."""
  open_tool("tpu-training", "roofline_model")

  roofline = page.locator("roofline-model, .roofline-container").first
  expect(roofline).to_be_visible(timeout=20000)

  # 1. Program-level analysis scatter chart
  prog_analysis = roofline.locator("program-level-analysis")
  expect(prog_analysis).to_be_visible(timeout=20000)
  scatter_svg = (
      prog_analysis.locator("chart[charttype='ScatterChart'] svg").first
  )
  expect(scatter_svg).to_be_visible(timeout=20000)
  bbox = scatter_svg.bounding_box()
  assert bbox is not None and bbox["width"] > 0 and bbox["height"] > 0

  # 2. Program-level data table rendering
  table = prog_analysis.locator("table").first
  expect(table).to_be_visible(timeout=10000)
  expect(table.locator("td, th").first).to_be_visible(timeout=5000)

  # 3. Invariant health verification
  assert_healthy(page, browser_errors, "roofline_model_program_analysis")


def test_pod_viewer_step_breakdown_rendering(
    page: Page,
    open_tool: Callable[..., str],
    browser_errors: BrowserErrors,
) -> None:
  """Verifies Pod Viewer renders step breakdown bar chart and controls."""
  with enabling_run_tool(page, "pod_viewer"):
    open_tool("tpu-training", "pod_viewer")
    comp = page.locator("pod-viewer")
    expect(comp).to_be_visible(timeout=20000)
    expect(comp).to_contain_text("Pod Viewer", timeout=10000)
    expect(comp).to_contain_text("Step breakdown", timeout=10000)
    svg = comp.locator("stack-bar-chart svg").first
    expect(svg).to_be_visible(timeout=15000)
    bbox = svg.bounding_box()
    assert bbox is not None and bbox["width"] > 0 and bbox["height"] > 0
    assert_healthy(page, browser_errors, "pod_viewer")


def test_static_kernel_viewer_rail_and_empty_state(
    page: Page,
    server_url: str,
    open_tool: Callable[..., str],
    browser_errors: BrowserErrors,
) -> None:
  """Verifies Static Kernel Viewer mounts its HLO Ops rail and empty state."""
  # Static Kernel Viewer mounts the WebGPU timeline container even on an empty
  # kernel list. Probe WebGPU support on a non-app endpoint so headless hosts
  # without a GPU adapter (such as macOS VMs) forgive only the expected
  # missing-adapter log while still catching all other console/DOM errors.
  probe_url = f"{server_url}/data/plugin/profile/runs"
  page.goto(probe_url)
  expect(page).to_have_url(probe_url)
  has_webgpu = bool(
      page.evaluate("async () => !!(await navigator.gpu?.requestAdapter())")
  )
  if has_webgpu:
    webgpu_ctx = contextlib.nullcontext()
  else:
    webgpu_ctx = browser_errors.ignoring(
        "WebGPU cannot be initialized",
        "WebGPU is not supported",
        "Failed to initialize Trace Viewer V2 WASM module",
        url_pattern="kernel_viewer",
    )

  list_re = re.compile(
      r".*/data/plugin/profile/data.*[?&]tag=kernel_viewer\b.*"
      r"[?&]request_type=list\b.*"
  )

  def _fulfill_kernel_list(route: Route) -> None:
    route.fulfill(
        status=200,
        content_type="application/json",
        body='{"hlo_modules": [], "module_kernels": {}, "kernels": []}',
    )

  page.route(list_re, _fulfill_kernel_list)
  try:
    with webgpu_ctx:
      with enabling_run_tool(page, "kernel_viewer"):
        open_tool("tpu-training", "kernel_viewer")
        rail = page.locator("nav.kernel-rail")
        expect(rail).to_be_visible(timeout=20000)
        expect(rail).to_contain_text("HLO Ops", timeout=10000)
        expect(rail).to_contain_text(
            "No HLO ops in this profile", timeout=10000
        )
        assert_healthy(page, context="static_kernel_viewer")
    browser_errors.assert_clean("static_kernel_viewer")
  finally:
    page.unroute(list_re, _fulfill_kernel_list)


def test_megascale_perfetto_fetches_trace_and_handshakes_iframe(
    page: Page,
    open_tool: Callable[..., str],
    browser_errors: BrowserErrors,
) -> None:
  """Verifies Megascale Perfetto fetches trace bytes and posts to Perfetto."""
  stub_html = (
      "<!DOCTYPE html><html><body><script>"
      "window.addEventListener('message', (e) => {"
      "  if (e.data === 'PING') e.source.postMessage('PONG', '*');"
      "  else if (e.data && e.data.perfetto && e.data.perfetto.buffer) {"
      "    document.body.dataset.traceBytes ="
      " String(e.data.perfetto.buffer.byteLength);"
      "  }"
      "});"
      "</script>Perfetto Stub</body></html>"
  )
  perfetto_re = re.compile(r"^https://ui\.perfetto\.dev(?:/|\?|$).*")

  def _fulfill_perfetto(route: Route) -> None:
    route.fulfill(status=200, content_type="text/html", body=stub_html)

  page.route(perfetto_re, _fulfill_perfetto)
  try:
    with enabling_run_tool(page, "megascale_perfetto"):
      open_tool("megascale-2host", "megascale_perfetto", host="mega-node-0")
      iframe = page.locator("megascale-perfetto iframe#perfetto-page")
      expect(iframe).to_be_visible(timeout=20000)
      frame_body = page.frame_locator("iframe#perfetto-page").locator("body")
      expect(frame_body).to_have_attribute(
          "data-trace-bytes", re.compile(r"^[1-9]\d*$"), timeout=20000
      )
      assert_healthy(page, browser_errors, "megascale_perfetto")
  finally:
    page.unroute(perfetto_re, _fulfill_perfetto)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Upstream XProf flex `.row` in framework-op-stats and hlo-stats lacks"
        " flex-wrap and overflows mat-sidenav-content at 1280px desktop width."
        " (perf-counters only overflows in internal builds; Copybara strips its"
        " extra Host and Show Zero Values controls in OSS.)"
    ),
)
@pytest.mark.parametrize(
    "run_name,tag",
    [
        ("tpu-training", "framework_op_stats"),
        ("tpu-training", "hlo_stats"),
    ],
)
def test_desktop_chart_rows_fit_content_pane(
    page: Page,
    open_tool: Callable[..., str],
    run_name: str,
    tag: str,
) -> None:
  """Tracks 1280px desktop horizontal overflow in unwrapped flex rows."""
  page.set_viewport_size({"width": 1280, "height": 900})
  open_tool(run_name, tag)
  expect(page.locator("framework-op-stats, hlo-stats").first).to_be_visible(
      timeout=20000
  )
  expect(
      page.locator("chart svg, .google-visualization-table").first
  ).to_be_visible(timeout=20000)
  assert not check_no_horizontal_overflow(
      page, check_content_pane=True, allowed_overflow_selectors=()
  )
