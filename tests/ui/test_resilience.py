"""Tests for application resilience, network fault injection, and concurrency."""

from collections.abc import Callable
from collections.abc import Iterator
import contextlib
import re
import tempfile

from playwright.sync_api import expect
from playwright.sync_api import Page

# pylint: disable=g-import-not-at-top
try:
  from tests.ui.conftest import BrowserErrors
  from tests.ui.ui_helpers import assert_component_geometry
  from tests.ui.ui_helpers import assert_healthy
  from tests.ui.ui_helpers import build_tool_url
  from tests.ui.ui_helpers import switch_tool
except ImportError:
  from conftest import BrowserErrors
  from ui_helpers import assert_component_geometry
  from ui_helpers import assert_healthy
  from ui_helpers import build_tool_url
  from ui_helpers import switch_tool


@contextlib.contextmanager
def _mock_api_status(page: Page, status: int) -> Iterator[None]:
  """Intercepts profile data requests and fulfills with a simulated HTTP error."""
  route_pattern = re.compile(r".*/data/plugin/profile/data.*")
  page.route(
      route_pattern,
      lambda r: r.fulfill(
          status=status,
          content_type="application/json",
          body=f'{{"error": "{status}"}}',
      ),
  )
  try:
    yield
  finally:
    page.unroute(route_pattern)


@contextlib.contextmanager
def _hold_overview_data_request(page: Page) -> Iterator[None]:
  """Holds overview_page data requests in flight until the block exits."""
  route_pattern = re.compile(
      r".*/data/plugin/profile/data.*[?&]tag=overview_page\b.*"
  )
  held_routes = []

  def _handle_route(route) -> None:
    held_routes.append(route)

  page.route(route_pattern, _handle_route)
  try:
    yield
  finally:
    for route in held_routes:
      try:
        route.abort()
      except Exception:  # pylint: disable=broad-exception-caught
        pass
    page.unroute(route_pattern)


def test_api_403_forbidden_resilience(
    page: Page,
    open_tool: Callable[..., str],
    browser_errors: BrowserErrors,
) -> None:
  """Verifies application shell stays interactive during 403 responses."""
  browser_errors.ignore("403")
  with _mock_api_status(page, 403):
    with page.expect_response(
        re.compile(r".*/data/plugin/profile/data.*")
    ) as response_info:
      open_tool("tpu-training", "overview_page")
    assert response_info.value.status == 403
    toolbar = page.locator("mat-toolbar")
    expect(toolbar).to_be_visible(timeout=20000)
    expect(toolbar).to_contain_text("XProf")
    switch_tool(page, "Memory Profile")
    expect(toolbar).to_be_visible(timeout=5000)
    expect(page.locator("mat-sidenav-container")).to_be_visible(timeout=5000)
  browser_errors.assert_clean()


def test_api_500_backend_recovery(
    page: Page,
    open_tool: Callable[..., str],
    browser_errors: BrowserErrors,
) -> None:
  """Verifies that the application recovers cleanly after a 500 error."""
  browser_errors.ignore("500")
  with _mock_api_status(page, 500):
    with page.expect_response(
        re.compile(r".*/data/plugin/profile/data.*")
    ) as response_info:
      open_tool("tpu-training", "overview_page")
    assert response_info.value.status == 500
    expect(page.locator("mat-toolbar")).to_be_visible(timeout=20000)

  # After unrouting 500, navigating to another tool recovers successfully
  switch_tool(page, "Memory Profile")
  expect(page.locator("memory-viewer, memory-profile")).to_be_visible(
      timeout=20000
  )
  browser_errors.assert_clean()


def test_empty_session_directory_clean_fallback(
    page: Page,
    server_url: str,
    browser_errors: BrowserErrors,
) -> None:
  """Verifies that an empty session directory displays the empty-state view."""
  with tempfile.TemporaryDirectory() as empty_dir:
    url = build_tool_url(server_url, empty_dir, "empty", "overview_page")
    page.goto(url, wait_until="domcontentloaded")
    expect(page.locator("text='No profile data was found.'")).to_be_visible(
        timeout=20000
    )
    expect(
        page.locator("button:has-text('CAPTURE PROFILE')").first
    ).to_be_visible()
    assert_healthy(page, browser_errors, "empty session")


def test_rapid_tool_switching_concurrency(
    page: Page,
    open_tool: Callable[..., str],
    browser_errors: BrowserErrors,
) -> None:
  """Verifies rapid client-side SPA tool switches do not crash the app."""
  open_tool("tpu-training", "overview_page")
  expect(page.locator("overview-page, overview-viewer")).to_be_visible(
      timeout=20000
  )

  for tool_name in ["Op Profile", "Memory Viewer", "Overview Page"]:
    switch_tool(page, tool_name)

  expect(page.locator("overview-page, overview-viewer")).to_be_visible(
      timeout=20000
  )
  expect(
      page.locator("main-page mat-sidenav-content > div.full-height")
  ).to_be_visible(timeout=20000)
  assert_healthy(page, browser_errors, "rapid tool switching")


def test_leaving_overview_page_mid_fetch_clears_loading_state(
    page: Page,
    open_tool: Callable[..., str],
    browser_errors: BrowserErrors,
) -> None:
  """Verifies leaving Overview Page mid-fetch does not lock subsequent tools."""
  with _hold_overview_data_request(page):
    open_tool("tpu-training", "overview_page")
    # Wait for the overview fetch itself, not the first-load "Navigating" text.
    expect(
        page.locator(".loading-message", has_text="Loading overview data")
    ).to_be_visible(timeout=10000)
    switch_tool(page, "Memory Profile")

  expect(page).to_have_url(re.compile(r"tag=memory_profile"), timeout=20000)
  expect(page.locator(".loading-message")).to_have_count(0, timeout=10000)
  expect(
      page.locator("main-page mat-sidenav-content > div.hidden-content")
  ).to_have_count(0, timeout=10000)
  assert_component_geometry(
      page, "memory-viewer, memory-profile", "leaving_overview_page_mid_fetch"
  )
  assert_healthy(page, browser_errors, "leaving_overview_page_mid_fetch")
