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
  from tests.ui.ui_helpers import switch_tool
except ImportError:
  from conftest import BrowserErrors
  from ui_helpers import assert_component_geometry
  from ui_helpers import assert_healthy
  from ui_helpers import ensure_sidenav_open
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
  assert_component_geometry(page, selector, "kernel_stats")
  rows = page.locator(selector).first.locator(
      "table tr:has(td), table mat-row, mat-row"
  )
  expect(rows.first).to_be_visible(timeout=10000)
  assert rows.count() >= 1, "Expected kernel stats rows"
  assert_healthy(page, browser_errors, "kernel_stats")


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
