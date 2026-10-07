"""Tests verifying functional and visual data rendering across core tools."""

from collections.abc import Callable
import os
import re

from playwright.sync_api import expect
from playwright.sync_api import Page
import pytest

# pylint: disable=g-import-not-at-top
try:
  from tests.ui.conftest import BrowserErrors
  from tests.ui.ui_helpers import assert_component_geometry
  from tests.ui.ui_helpers import assert_healthy
  from tests.ui.ui_helpers import ensure_sidenav_open
  from tests.ui.ui_helpers import select_category_filter
  from tests.ui.ui_helpers import select_module
  from tests.ui.ui_helpers import switch_tool
except ImportError:
  from conftest import BrowserErrors
  from ui_helpers import assert_component_geometry
  from ui_helpers import assert_healthy
  from ui_helpers import ensure_sidenav_open
  from ui_helpers import select_category_filter
  from ui_helpers import select_module
  from ui_helpers import switch_tool


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
    logdir: str,
    open_tool: Callable[..., str],
    browser_errors: BrowserErrors,
) -> None:
  """Verifies that the GPU Kernel Stats table renders execution rows."""
  # Checked before resolving: on a single-run logdir the resolver would fall
  # back to that (TPU) run.
  if not os.path.exists(os.path.join(logdir, "gpu-training")):
    pytest.skip("gpu-training fixture not present in logdir")
  open_tool("gpu-training", "kernel_stats")

  selector = "kernel-stats, kernel-stats-adapter"
  expect(page.locator(selector).first).to_be_visible(timeout=20000)
  expect(
      page.locator("main-page mat-sidenav-content > div.full-height")
  ).to_be_visible(timeout=20000)
  expect(page).to_have_url(
      re.compile(r"[?&]run=gpu-training\b"), timeout=10000
  )
  expect(
      page.locator("text=There is no GPU data to display")
  ).to_be_hidden(timeout=10000)
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
  expect(pie_svg).to_contain_text(
      "Percentage of self time per HLO op category"
  )

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

  overflow_px = page.evaluate(
      """() => {
        const pane = document.querySelector('mat-sidenav-content');
        if (!pane) return 0;
        return pane.scrollWidth - pane.clientWidth;
      }"""
  )
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
  page.add_init_script(
      "try { window.localStorage.setItem('use_trace_viewer_v2', 'true'); }"
      " catch (e) {}"
  )
  open_tool("tpu-training", "trace_viewer", use_trace_viewer_v2="true")

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
