"""Parametrized tool rendering and navigation verification across runs."""

# pylint: disable=g-doc-args,g-doc-return-or-yield,g-short-docstring-punctuation

from collections.abc import Callable
import re

# pylint: disable=g-import-not-at-top
try:
  from tests.ui.conftest import BrowserErrors
  from tests.ui.ui_helpers import assert_component_geometry
  from tests.ui.ui_helpers import assert_healthy
  from tests.ui.ui_helpers import switch_tool
except ImportError:
  from conftest import BrowserErrors
  from ui_helpers import assert_component_geometry
  from ui_helpers import assert_healthy
  from ui_helpers import switch_tool
from playwright.sync_api import expect
from playwright.sync_api import Page
import pytest

# Mapping of (run, tool_tag) to the corresponding DOM element selector that
# MUST mount and render with non-zero geometry when the tool loads. GPU-only
# Kernel Stats is left to test_kernel_stats_rendering, which skips unless the
# logdir has a gpu-training run; neither the demo nor the Kokoro logdir does.
ACTIVE_TOOL_SPECS: list[tuple[str, str, str]] = [
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
]

# Tool content that MUST render inside the component, besides the drawn
# visualization that assert_component_geometry requires. ":scope" (the
# component itself) for tools without such content.
TOOL_CONTENT_SELECTORS: dict[str, str] = {
    "graph_viewer": "iframe, svg, canvas",
    "op_profile": "text=jit_train_step",
    "input_pipeline_analyzer": "text=Summary of input-pipeline analysis",
    "hlo_stats": "google-chart, .google-visualization-table",
}


def test_tool_navigation_shell(
    page: Page,
    open_tool: Callable[..., str],
    browser_errors: BrowserErrors,
):
  """Verifies that navigation to the application root loads the shell."""
  open_tool("tpu-training", "overview_page")

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
    open_tool: Callable[..., str],
    browser_errors: BrowserErrors,
    run: str,
    tag: str,
    selector: str,
):
  """Verifies that deep-linking to each tool mounts its component view."""
  open_tool(run, tag)

  # Assert the specific tool view and its child visualization rendered
  assert_component_geometry(page, selector, tag)

  # Assert the tool's own content rendered inside the component
  content = TOOL_CONTENT_SELECTORS.get(tag, ":scope")
  element = page.locator(f":is({selector}):visible").first.locator(content)
  expect(element.first).to_be_visible(timeout=20000)
  bbox = element.first.bounding_box()
  assert bbox and bbox["width"] > 0 and bbox["height"] > 0, (
      f"{tag} content {content} collapsed: {bbox}"
  )

  # Assert DOM invariant health (non-empty body and no poison tokens)
  assert_healthy(page, browser_errors, tag)


def test_unavailable_tool_fallback_redirection(
    page: Page,
    open_tool: Callable[..., str],
    browser_errors: BrowserErrors,
):
  """Verifies deep-linking to an unsupported tool redirects to overview_page."""
  # 'kernel_stats' is only for GPU runs; on TPU it should fallback
  open_tool("tpu-training", "kernel_stats")

  # Must gracefully route to overview page without crashing the shell
  overview_view = page.locator("overview-page, overview-viewer").first
  expect(overview_view).to_be_visible(timeout=20000)
  assert_healthy(page, browser_errors, "unavailable_tool_fallback")


def test_tool_dropdown_selection(
    page: Page,
    open_tool: Callable[..., str],
    browser_errors: BrowserErrors,
):
  """Verifies selecting a tool from the sidebar dropdown loads that tool view."""
  open_tool("tpu-training", "overview_page")
  expect(
      page.locator("overview-page mat-card, overview-viewer mat-card").first
  ).to_be_visible(timeout=20000)

  switch_tool(page, "Op Profile")
  expect(page).to_have_url(re.compile(r"tag=op_profile"), timeout=20000)
  expect(page.locator("op-profile, op-profile-base").first).to_be_visible(
      timeout=20000
  )
  assert_healthy(page, browser_errors, "Op Profile")
