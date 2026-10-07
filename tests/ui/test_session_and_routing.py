"""Tests for session discovery, deep linking, and URL routing."""

# pylint: disable=g-doc-args,g-doc-return-or-yield,g-short-docstring-punctuation

from collections.abc import Callable
import os
import pathlib
import re
import urllib.parse

# pylint: disable=g-import-not-at-top
try:
  from tests.ui.conftest import BrowserErrors
  from tests.ui.journey_capture import settled_tool_url_pattern
  from tests.ui.ui_helpers import assert_healthy
  from tests.ui.ui_helpers import plot_graph_node
  from tests.ui.ui_helpers import select_module
  from tests.ui.ui_helpers import select_session
  from tests.ui.ui_helpers import switch_tool
except ImportError:
  from conftest import BrowserErrors
  from journey_capture import settled_tool_url_pattern
  from ui_helpers import assert_healthy
  from ui_helpers import plot_graph_node
  from ui_helpers import select_module
  from ui_helpers import select_session
  from ui_helpers import switch_tool
from playwright.sync_api import expect
from playwright.sync_api import Page
import pytest


def test_deep_link_parameter_preservation(
    page: Page,
    logdir: str,
    resolve_run: Callable[[str], str],
    open_tool: Callable[..., str],
    browser_errors: BrowserErrors,
) -> None:
  """Verifies deep-linked URL parameters are preserved and reflected in UI."""
  run = resolve_run("tpu-training")
  target_host = min(
      trace.name.removesuffix(".xplane.pb")
      for trace in pathlib.Path(logdir, run).glob("*.xplane.pb")
  )
  open_tool(run, "overview_page", host=target_host)

  overview_comp = page.locator("overview-page, overview-viewer")
  expect(overview_comp).to_be_visible(timeout=20000)
  expect(
      page.locator("main-page mat-sidenav-content > div.full-height")
  ).to_be_visible(timeout=20000)
  expect(
      page.locator("sidenav .item-container:has-text('Sessions') mat-select")
  ).to_contain_text(run)
  expect(
      page.locator("sidenav .item-container:has-text('Tools') mat-select")
  ).to_contain_text("Overview Page")
  expect(
      page.locator("sidenav .item-container:has-text('Hosts') mat-select")
  ).to_contain_text(target_host)
  assert_healthy(page, browser_errors, "deep_link_parameter_preservation")


def test_graph_viewer_chip_click_preserves_run_query_param(
    page: Page,
    open_tool: Callable[..., str],
    browser_errors: BrowserErrors,
) -> None:
  """Verifies Graph Viewer chip navigation retains the run= query parameter."""
  run = open_tool("tpu-training", "graph_viewer")
  expect(page.locator("graph-viewer").first).to_be_visible(timeout=20000)

  plot_graph_node(page, "fusion.")
  expect(page).to_have_url(
      re.compile(rf"[?&]run={re.escape(run)}\b"), timeout=10000
  )
  expect(page).to_have_url(re.compile(r"[?&]tag=graph_viewer\b"), timeout=10000)
  assert_healthy(page, browser_errors, "graph_viewer_chip_url_preservation")


def test_memory_viewer_module_selection_persists_across_reload(
    page: Page,
    open_tool: Callable[..., str],
    browser_errors: BrowserErrors,
) -> None:
  """Verifies Memory Viewer keeps moduleName= in the URL and across reloads."""
  open_tool("tpu-training", "memory_viewer")
  summary = page.locator("memory-viewer-main")
  expect(summary).to_contain_text("Module Name: jit__where(", timeout=20000)

  select_module(page, "jit_train_step")
  expect(summary).to_contain_text("Module Name: jit_train_step(", timeout=20000)
  expect(summary).to_contain_text("12106.12", timeout=10000)
  expect(page).to_have_url(
      re.compile(r"[?&]moduleName=jit_train_step"), timeout=10000
  )

  page.reload(wait_until="domcontentloaded")
  expect(page).to_have_url(
      re.compile(r"[?&]moduleName=jit_train_step"), timeout=10000
  )
  expect(summary).to_contain_text("Module Name: jit_train_step(", timeout=20000)
  expect(summary).to_contain_text("12106.12", timeout=10000)
  expect(
      page.locator("sidenav .item-container:has-text('Hlo Modules') mat-select")
  ).to_contain_text("jit_train_step")
  assert_healthy(page, browser_errors, "memory_viewer_reload_persistence")


def test_cross_run_switch_from_exclusive_tool(
    page: Page,
    server_url: str,
    logdir: str,
    resolve_run: Callable[[str], str],
    browser_errors: BrowserErrors,
) -> None:
  """Verifies switching runs from GPU Kernel Stats to TPU does not error."""
  if not os.path.exists(os.path.join(logdir, "gpu-training")):
    pytest.skip("gpu-training fixture not present in logdir")
  tpu_run = resolve_run("tpu-training")

  query = urllib.parse.urlencode({
      "run_path": logdir,
      "run": "gpu-training",
      "tag": "kernel_stats",
  })
  page.goto(
      f"{server_url.rstrip('/')}/?{query}",
      wait_until="domcontentloaded",
  )
  expect(
      page.locator("kernel-stats, kernel-stats-adapter").first
  ).to_be_visible(timeout=20000)
  expect(
      page.locator("main-page mat-sidenav-content > div.full-height")
  ).to_be_visible(timeout=20000)
  expect(
      page.locator("sidenav .item-container:has-text('Sessions') mat-select")
  ).to_be_visible(timeout=20000)

  select_session(page, tpu_run)
  expect(page).to_have_url(
      settled_tool_url_pattern("Overview Page"), timeout=20000
  )
  expect(
      page.locator("overview-page mat-card, overview-viewer mat-card").first
  ).to_be_visible(timeout=20000)
  expect(
      page.locator("main-page mat-sidenav-content > div.full-height")
  ).to_be_visible(timeout=20000)
  expect(page.locator("diagnostics-view .callout.is-critical")).to_have_count(0)
  assert_healthy(page, browser_errors, "cross_run_switch_from_exclusive_tool")


def _navigate_three_tools_and_step_back(
    page: Page, open_tool: Callable[..., str]
) -> None:
  """Navigates Overview -> Roofline -> Memory Profile, then steps back once."""
  overview_card = page.locator(
      "overview-page mat-card, overview-viewer mat-card"
  ).first
  tools_select = page.locator(
      "sidenav .item-container:has-text('Tools') mat-select"
  )
  open_tool("tpu-training", "overview_page")
  expect(overview_card).to_be_visible(timeout=20000)

  switch_tool(page, "Roofline Model")
  expect(page.locator("roofline-model")).to_be_visible(timeout=20000)
  expect(page).to_have_url(
      settled_tool_url_pattern("Roofline Model"), timeout=20000
  )

  switch_tool(page, "Memory Profile")
  expect(page.locator("memory-viewer, memory-profile")).to_be_visible(
      timeout=20000
  )
  expect(page).to_have_url(
      settled_tool_url_pattern("Memory Profile"), timeout=20000
  )

  page.go_back(wait_until="domcontentloaded")
  expect(page).to_have_url(
      settled_tool_url_pattern("Roofline Model"), timeout=20000
  )
  expect(tools_select).to_contain_text("Roofline Model")
  expect(page.locator("roofline-model")).to_be_visible(timeout=20000)


def test_browser_back_navigation(
    page: Page,
    open_tool: Callable[..., str],
    browser_errors: BrowserErrors,
) -> None:
  """Verifies browser back steps restore previous tools without split-brain."""
  _navigate_three_tools_and_step_back(page, open_tool)

  page.go_back(wait_until="domcontentloaded")
  expect(page).to_have_url(
      settled_tool_url_pattern("Overview Page"), timeout=20000
  )
  expect(
      page.locator("sidenav .item-container:has-text('Tools') mat-select")
  ).to_contain_text("Overview Page")
  expect(
      page.locator("overview-page mat-card, overview-viewer mat-card").first
  ).to_be_visible(timeout=20000)
  expect(
      page.locator("main-page mat-sidenav-content > div.full-height")
  ).to_be_visible(timeout=20000)
  assert_healthy(page, browser_errors, "browser_back_navigation")


def test_browser_forward_navigation(
    page: Page,
    open_tool: Callable[..., str],
    browser_errors: BrowserErrors,
) -> None:
  """Verifies a single browser forward step restores the next tool cleanly."""
  _navigate_three_tools_and_step_back(page, open_tool)

  page.go_forward(wait_until="domcontentloaded")
  expect(page).to_have_url(
      settled_tool_url_pattern("Memory Profile"), timeout=20000
  )
  expect(
      page.locator("sidenav .item-container:has-text('Tools') mat-select")
  ).to_contain_text("Memory Profile")
  expect(page.locator("memory-viewer, memory-profile")).to_be_visible(
      timeout=20000
  )
  assert_healthy(page, browser_errors, "browser_forward_navigation")
