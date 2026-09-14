"""Parametrized tool rendering and navigation verification across runs."""

# pylint: disable=g-doc-args,g-doc-return-or-yield,g-short-docstring-punctuation

import os

# pylint: disable=g-import-not-at-top
try:
  from tests.ui.conftest import BrowserErrors
  from tests.ui.invariants import run_content_invariants
except ImportError:
  from conftest import BrowserErrors
  from invariants import run_content_invariants
from playwright.sync_api import expect
from playwright.sync_api import Page
import pytest

# Mapping of (run, tool_tag) to the corresponding DOM element selector that
# MUST mount and render with non-zero geometry when the tool loads.
ACTIVE_TOOL_SPECS: list[tuple[str, str, str]] = [
    # TPU-specific and shared tools
    ("tpu-training", "overview_page", "overview-page, overview-viewer"),
    ("tpu-training", "trace_viewer", "iframe, #filter-bar, .filter-bar"),
    (
        "tpu-training",
        "graph_viewer",
        "graph-viewer, iframe.graph-viewer-iframe, .graph-viewer-container",
    ),
    ("tpu-training", "op_profile", "op-profile, op-profile-base"),
    ("tpu-training", "input_pipeline_analyzer", "input-pipeline"),
    ("tpu-training", "memory_profile", "memory-viewer, memory-profile"),
    ("tpu-training", "memory_viewer", "memory-viewer, memory-profile"),
    ("tpu-training", "roofline_model", "roofline-model, .roofline-container"),
    ("tpu-training", "framework_op_stats", "framework-op-stats"),
    ("tpu-training", "hlo_stats", "hlo-stats"),
    # GPU-specific tools
    ("gpu-training", "kernel_stats", "kernel-stats, kernel-stats-adapter"),
]


def test_tool_navigation_shell(
    page: Page,
    server_url: str,
    logdir: str,
    browser_errors: BrowserErrors,
):
  """Verifies that navigation to the application root loads the shell."""
  session_path = os.path.join(logdir, "tpu-training")
  url = (
      f"{server_url}/?session_path={session_path}&run=tpu-training"
      "&tag=overview_page"
  )
  page.goto(url, wait_until="domcontentloaded")

  # Assert toolbar and header branding
  toolbar = page.locator("mat-toolbar")
  expect(toolbar).to_be_visible(timeout=20000)
  expect(toolbar).to_contain_text("XProf")

  # Assert sidebar session and tool selectors are initialized
  sidenav = page.locator("sidenav")
  expect(sidenav).to_be_visible(timeout=10000)
  expect(
      sidenav.locator(".item-container:has-text('Sessions')")
  ).to_be_visible()
  expect(sidenav.locator(".item-container:has-text('Tools')")).to_be_visible()

  browser_errors.assert_clean()


@pytest.mark.parametrize("run,tag,selector", ACTIVE_TOOL_SPECS)
def test_individual_tool_loads(
    page: Page,
    server_url: str,
    logdir: str,
    browser_errors: BrowserErrors,
    run: str,
    tag: str,
    selector: str,
):
  """Verifies that deep-linking to each tool mounts its specific component view."""
  session_path = os.path.join(logdir, run)
  url = f"{server_url}/?session_path={session_path}&run={run}&tag={tag}"
  page.goto(url, wait_until="domcontentloaded")

  # Assert the specific tool view mounted
  tool_view = page.locator(selector).first
  expect(tool_view).to_be_visible(timeout=20000)

  # Assert positive layout geometry
  bbox = tool_view.bounding_box()
  assert bbox is not None, f"Tool component '{tag}' has no bounding box"
  assert (
      bbox["width"] > 0 and bbox["height"] > 0
  ), f"Tool component '{tag}' collapsed: {bbox}"

  # Assert DOM invariant health (no poison tokens)
  violations = run_content_invariants(page.inner_text("body"))
  assert not violations, f"Poison tokens detected in {tag}: {violations}"
  browser_errors.assert_clean()


def test_unavailable_tool_fallback_redirection(
    page: Page,
    server_url: str,
    logdir: str,
    browser_errors: BrowserErrors,
):
  """Verifies deep-linking to an unsupported tool tag redirects to overview_page."""
  session_path = os.path.join(logdir, "tpu-training")

  # 'kernel_stats' is only for GPU runs; on TPU it should fallback
  url = (
      f"{server_url}/?session_path={session_path}&run=tpu-training"
      "&tag=kernel_stats"
  )
  page.goto(url, wait_until="domcontentloaded")

  # Must gracefully route to overview page without crashing the shell
  overview_view = page.locator("overview-page, overview-viewer").first
  expect(overview_view).to_be_visible(timeout=20000)
  browser_errors.assert_clean()


def test_tool_dropdown_selection(
    page: Page,
    server_url: str,
    logdir: str,
    browser_errors: BrowserErrors,
):
  """Verifies selecting a tool from the sidebar dropdown loads that tool view."""
  session_path = os.path.join(logdir, "tpu-training")
  url = (
      f"{server_url}/?session_path={session_path}&run=tpu-training"
      "&tag=overview_page"
  )
  page.goto(url, wait_until="domcontentloaded")
  expect(
      page.locator("overview-page mat-card, overview-viewer mat-card").first
  ).to_be_visible(timeout=20000)

  # Open the Tools dropdown and select Op Profile
  tool_dropdown = page.locator(
      "sidenav .item-container:has-text('Tools') mat-select"
  )
  tool_dropdown.click()
  page.locator("mat-option").filter(has_text="Op Profile").first.click()

  # Verify Op Profile mounts
  expect(page.locator("op-profile, op-profile-base").first).to_be_visible(
      timeout=20000
  )
  browser_errors.assert_clean()
