"""Tests for session discovery, deep linking, and URL routing."""

# pylint: disable=g-doc-args,g-doc-return-or-yield,g-short-docstring-punctuation

from collections.abc import Callable
import pathlib
import re

# pylint: disable=g-import-not-at-top
try:
  from tests.ui.conftest import BrowserErrors
  from tests.ui.journey_capture import settled_tool_url_pattern
  from tests.ui.test_user_journeys import step_history
  from tests.ui.ui_helpers import assert_healthy
  from tests.ui.ui_helpers import switch_tool
except ImportError:
  from conftest import BrowserErrors
  from journey_capture import settled_tool_url_pattern
  from test_user_journeys import step_history
  from ui_helpers import assert_healthy
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
):
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
      page.locator("sidenav .item-container:has-text('Sessions') mat-select")
  ).to_contain_text(run)
  expect(
      page.locator("sidenav .item-container:has-text('Tools') mat-select")
  ).to_contain_text("Overview Page")
  expect(
      page.locator("sidenav .item-container:has-text('Hosts') mat-select")
  ).to_contain_text(target_host)
  assert_healthy(page, browser_errors, "deep_link_parameter_preservation")


def _back_from_memory_profile_to_overview(
    page: Page, open_tool: Callable[..., str]
) -> None:
  """Opens Overview Page, switches to Memory Profile, then steps back."""
  overview_card = page.locator(
      "overview-page mat-card, overview-viewer mat-card"
  ).first
  open_tool("tpu-training", "overview_page")
  expect(overview_card).to_be_visible(timeout=20000)

  switch_tool(page, "Memory Profile")
  expect(page.locator("memory-viewer, memory-profile")).to_be_visible(
      timeout=20000
  )
  expect(page).to_have_url(re.compile(r"tag=memory_profile"), timeout=20000)

  step_history(page, "Overview Page")
  expect(overview_card).to_be_visible(timeout=20000)


def test_browser_back_navigation(
    page: Page,
    open_tool: Callable[..., str],
    browser_errors: BrowserErrors,
):
  """Verifies browser back navigation restores the previous tool view."""
  _back_from_memory_profile_to_overview(page, open_tool)
  assert_healthy(page, browser_errors, "browser_back_navigation")


@pytest.mark.xfail(
    strict=True,
    reason=(
        "SideNav.navigateTools() calls history.pushState() while it handles"
        " popstate, which drops the forward history entries"
    ),
)
def test_browser_forward_navigation(
    page: Page,
    open_tool: Callable[..., str],
    browser_errors: BrowserErrors,
):
  """Verifies browser forward navigation restores the next tool view."""
  _back_from_memory_profile_to_overview(page, open_tool)

  page.go_forward(wait_until="domcontentloaded")
  expect(page).to_have_url(
      settled_tool_url_pattern("Memory Profile"), timeout=20000
  )
  expect(page.locator("memory-viewer, memory-profile")).to_be_visible(
      timeout=20000
  )
  assert_healthy(page, browser_errors, "browser_forward_navigation")
