"""Tests verifying functional and visual data rendering across core tools."""

import os
import re

from playwright.sync_api import expect
from playwright.sync_api import Page
import pytest

# pylint: disable=g-import-not-at-top
try:
  from tests.ui.conftest import BrowserErrors
  from tests.ui.ui_helpers import assert_healthy
  from tests.ui.ui_helpers import build_tool_url
  from tests.ui.ui_helpers import ensure_sidenav_open
  from tests.ui.ui_helpers import switch_tool
except ImportError:
  from conftest import BrowserErrors
  from ui_helpers import assert_healthy
  from ui_helpers import build_tool_url
  from ui_helpers import ensure_sidenav_open
  from ui_helpers import switch_tool


def test_overview_page_deep_components(
    page: Page,
    server_url: str,
    logdir: str,
    browser_errors: BrowserErrors,
) -> None:
  """Validates overview summary metrics, step-time chart, and host selector."""
  session_path = os.path.join(logdir, "tpu-training")
  url = build_tool_url(
      server_url, session_path, "tpu-training", "overview_page"
  )
  page.goto(url, wait_until="domcontentloaded")

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
    server_url: str,
    logdir: str,
    browser_errors: BrowserErrors,
) -> None:
  """Verifies that the Memory Profile tool mounts its data tables."""
  session_path = os.path.join(logdir, "tpu-training")
  url = build_tool_url(
      server_url, session_path, "tpu-training", "memory_profile"
  )
  page.goto(url, wait_until="domcontentloaded")

  mem_comp = page.locator("memory-viewer, memory-profile").first
  expect(mem_comp).to_be_visible(timeout=20000)

  table_container = mem_comp.locator(
      "memory-breakdown-table .table, memory-breakdown-table"
  ).first
  expect(table_container).to_be_visible(timeout=20000)

  rows = table_container.locator("table tbody tr, tr:has(td)")
  expect(rows.first).to_be_visible(timeout=10000)
  expect(rows.first.locator("td").first).to_be_visible(timeout=5000)
  assert rows.count() >= 1, "Expected at least one memory profile table row"
  assert bool(rows.first.inner_text().strip()), "Memory profile row is empty"
  assert_healthy(page, browser_errors, "Memory Profile")


_MOUNTING_CHECKS: list[tuple[str, str, str, str]] = [
    (
        "tpu-training",
        "graph_viewer",
        "graph-viewer, iframe.graph-viewer-iframe, .graph-viewer-container",
        "iframe, svg, canvas, .graph-container",
    ),
    (
        "tpu-training",
        "hlo_stats",
        "hlo-stats",
        "google-chart, .google-visualization-table",
    ),
    (
        "tpu-training",
        "input_pipeline_analyzer",
        "input-pipeline",
        "text=Summary of input-pipeline analysis",
    ),
    (
        "tpu-training",
        "op_profile",
        "op-profile, op-profile-base",
        "text=jit_train_step",
    ),
]


@pytest.mark.parametrize("run,tag,container_sel,child_sel", _MOUNTING_CHECKS)
def test_tool_component_mounting(
    page: Page,
    server_url: str,
    logdir: str,
    browser_errors: BrowserErrors,
    run: str,
    tag: str,
    container_sel: str,
    child_sel: str,
) -> None:
  """Verifies domain-specific content rendering for profiling tools."""
  fixture_path = os.path.join(logdir, run)
  if not os.path.exists(fixture_path):
    pytest.skip(f"Fixture '{run}' not present in logdir")

  url = build_tool_url(server_url, fixture_path, run, tag)
  page.goto(url, wait_until="domcontentloaded")

  container = page.locator(container_sel).first
  expect(container).to_be_visible(timeout=20000)

  child = container.locator(child_sel).first
  expect(child).to_be_visible(timeout=20000)
  bbox = child.bounding_box()
  assert bbox and bbox["width"] > 0 and bbox["height"] > 0, (
      f"{tag} child {child_sel} collapsed: {bbox}"
  )
  assert_healthy(page, browser_errors, tag)


def test_kernel_stats_rendering(
    page: Page,
    server_url: str,
    logdir: str,
    browser_errors: BrowserErrors,
) -> None:
  """Verifies that the GPU Kernel Stats table renders execution rows."""
  fixture_path = os.path.join(logdir, "gpu-training")
  if not os.path.exists(fixture_path):
    pytest.skip("gpu-training fixture not present in logdir")

  url = build_tool_url(server_url, fixture_path, "gpu-training", "kernel_stats")
  page.goto(url, wait_until="domcontentloaded")

  ks_comp = page.locator("kernel-stats, kernel-stats-adapter").first
  expect(ks_comp).to_be_visible(timeout=20000)

  rows = ks_comp.locator("table tr:has(td), table mat-row, mat-row")
  expect(rows.first).to_be_visible(timeout=10000)
  assert rows.count() >= 1, "Expected kernel stats rows"
  assert_healthy(page, browser_errors, "kernel_stats")


def test_trace_viewer_mounts_canvas(
    page: Page,
    server_url: str,
    logdir: str,
    browser_errors: BrowserErrors,
) -> None:
  """Verifies Trace Viewer V2 mounts a non-zero canvas element."""
  session_path = os.path.join(logdir, "tpu-training")
  url = build_tool_url(
      server_url,
      session_path,
      "tpu-training",
      "trace_viewer",
      use_trace_viewer_v2="true",
  )
  page.goto(url, wait_until="domcontentloaded")

  trace_container = page.locator(
      "canvas#canvas, .trace-viewer-container canvas, canvas, iframe"
  ).first
  expect(trace_container).to_be_visible(timeout=20000)
  bbox = trace_container.bounding_box()
  assert bbox and bbox["width"] > 0 and bbox["height"] > 0
  browser_errors.assert_clean()


def test_trace_viewer_lifecycle_teardown(
    page: Page,
    server_url: str,
    logdir: str,
    browser_errors: BrowserErrors,
) -> None:
  """Verifies Trace Viewer tears down cleanly when switching tools."""
  session_path = os.path.join(logdir, "tpu-training")
  url = build_tool_url(
      server_url,
      session_path,
      "tpu-training",
      "trace_viewer",
      use_trace_viewer_v2="true",
  )
  page.goto(url, wait_until="domcontentloaded")

  trace_container = page.locator(
      "canvas#canvas, .trace-viewer-container canvas, canvas, iframe"
  ).first
  expect(trace_container).to_be_visible(timeout=20000)
  bbox = trace_container.bounding_box()
  assert bbox and bbox["width"] > 0 and bbox["height"] > 0

  switch_tool(page, "Op Profile")
  expect(page.locator("op-profile, op-profile-base").first).to_be_visible(
      timeout=20000
  )
  expect(page.locator("trace-viewer iframe, iframe")).to_have_count(0)
  browser_errors.assert_clean()


def test_tool_switching_cleanup(
    page: Page,
    server_url: str,
    logdir: str,
    browser_errors: BrowserErrors,
) -> None:
  """Verifies switching tools cleans up previously mounted views via UI."""
  session_path = os.path.join(logdir, "tpu-training")
  url_overview = build_tool_url(
      server_url, session_path, "tpu-training", "overview_page"
  )
  page.goto(url_overview, wait_until="domcontentloaded")
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
  browser_errors.assert_clean()
