"""Tests for application resilience, network fault injection, and concurrency."""

from collections.abc import Iterator
import contextlib
import os
import re
import tempfile

from playwright.sync_api import expect
from playwright.sync_api import Page

# pylint: disable=g-import-not-at-top
try:
  from tests.ui.conftest import BrowserErrors
  from tests.ui.invariants import run_content_invariants
  from tests.ui.ui_helpers import build_tool_url
  from tests.ui.ui_helpers import switch_tool
except ImportError:
  from conftest import BrowserErrors
  from invariants import run_content_invariants
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


def test_api_403_forbidden_resilience(
    page: Page,
    server_url: str,
    logdir: str,
    browser_errors: BrowserErrors,
) -> None:
  """Verifies application shell stays interactive during 403 responses."""
  browser_errors.ignore("403")
  session_path = os.path.join(logdir, "tpu-training")
  url = build_tool_url(
      server_url, session_path, "tpu-training", "overview_page"
  )
  with _mock_api_status(page, 403):
    with page.expect_response(
        re.compile(r".*/data/plugin/profile/data.*")
    ) as response_info:
      page.goto(url, wait_until="domcontentloaded")
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
    server_url: str,
    logdir: str,
    browser_errors: BrowserErrors,
) -> None:
  """Verifies that the application recovers cleanly after a 500 error."""
  browser_errors.ignore("500")
  session_path = os.path.join(logdir, "tpu-training")
  url = build_tool_url(
      server_url, session_path, "tpu-training", "overview_page"
  )
  with _mock_api_status(page, 500):
    with page.expect_response(
        re.compile(r".*/data/plugin/profile/data.*")
    ) as response_info:
      page.goto(url, wait_until="domcontentloaded")
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
    violations = run_content_invariants(page.inner_text("body"))
    assert not violations, f"Poison tokens detected: {violations}"
    browser_errors.assert_clean()


def test_rapid_tool_switching_concurrency(
    page: Page,
    server_url: str,
    logdir: str,
    browser_errors: BrowserErrors,
) -> None:
  """Verifies rapid client-side SPA tool switches do not crash the app."""
  session_path = os.path.join(logdir, "tpu-training")
  url = build_tool_url(
      server_url, session_path, "tpu-training", "overview_page"
  )
  page.goto(url, wait_until="domcontentloaded")
  expect(page.locator("overview-page, overview-viewer")).to_be_visible(
      timeout=20000
  )

  for tool_name in ["Op Profile", "Memory Viewer", "Overview Page"]:
    switch_tool(page, tool_name)

  expect(page.locator("overview-page, overview-viewer")).to_be_visible(
      timeout=20000
  )
  violations = run_content_invariants(page.inner_text("body"))
  assert not violations, f"Poison tokens detected: {violations}"
  browser_errors.assert_clean()
