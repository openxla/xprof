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
"""Application resilience, network fault injection, and concurrency tests."""

from collections.abc import Callable
from collections.abc import Iterator
import contextlib
import re
import tempfile

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import expect
from playwright.sync_api import Page
from playwright.sync_api import Route

from tests.ui.conftest import BrowserErrors
from tests.ui.ui_helpers import assert_component_geometry
from tests.ui.ui_helpers import assert_healthy
from tests.ui.ui_helpers import build_tool_url
from tests.ui.ui_helpers import switch_tool

_DATA_ROUTE_RE = re.compile(r".*/data/plugin/profile/data.*")


@contextlib.contextmanager
def _mock_api_status(page: Page, status: int, body: str = "") -> Iterator[None]:
  """Intercepts profile data requests and fulfills with a simulated response."""
  payload = body or f'{{"error": "{status}"}}'

  def _handle(route: Route) -> None:
    route.fulfill(status=status, content_type="application/json", body=payload)

  page.route(_DATA_ROUTE_RE, _handle)
  try:
    yield
  finally:
    page.unroute(_DATA_ROUTE_RE, _handle)


@contextlib.contextmanager
def _hold_data_request(
    page: Page, browser_errors: BrowserErrors, tag: str
) -> Iterator[None]:
  """Holds `/data` requests for `tag` in flight until the block exits."""
  route_pattern = re.compile(
      rf".*/data/plugin/profile/data.*[?&]tag={re.escape(tag)}\b.*"
  )
  held_routes: list[Route] = []

  def _handle_route(route: Route) -> None:
    held_routes.append(route)

  page.route(route_pattern, _handle_route)
  with browser_errors.ignoring("net::ERR_FAILED", url_pattern="/data/plugin/"):
    try:
      yield
    finally:
      for route in held_routes:
        try:
          route.abort()
        except PlaywrightError:
          pass
      page.unroute(route_pattern, _handle_route)


def test_api_403_forbidden_resilience(
    page: Page,
    open_tool: Callable[..., str],
    browser_errors: BrowserErrors,
) -> None:
  """Verifies application shell stays interactive during 403 responses."""
  with (
      browser_errors.ignoring("403", url_pattern="/data/plugin/profile/"),
      _mock_api_status(page, 403),
  ):
    with page.expect_response(_DATA_ROUTE_RE) as response_info:
      open_tool("tpu-training", "overview_page")
    assert response_info.value.status == 403
    toolbar = page.locator("mat-toolbar")
    expect(toolbar).to_be_visible(timeout=20000)
    expect(toolbar).to_contain_text("XProf")
    switch_tool(page, "Memory Profile")
    expect(toolbar).to_be_visible(timeout=5000)
    expect(page.locator("mat-sidenav-container")).to_be_visible(timeout=5000)
  browser_errors.assert_clean()


def test_api_500_and_corrupt_json_backend_recovery(
    page: Page,
    open_tool: Callable[..., str],
    browser_errors: BrowserErrors,
) -> None:
  """Verifies recovery after HTTP 500 and malformed JSON `/data` responses."""
  with (
      browser_errors.ignoring("500", url_pattern="/data/plugin/profile/"),
      _mock_api_status(page, 500),
  ):
    with page.expect_response(_DATA_ROUTE_RE) as response_info:
      open_tool("tpu-training", "overview_page")
    assert response_info.value.status == 500
    expect(page.locator("mat-toolbar")).to_be_visible(timeout=20000)

  with (
      browser_errors.ignoring(
          "SyntaxError", "JSON", url_pattern="/data/plugin/profile/"
      ),
      _mock_api_status(page, 200, body="{corrupt_json"),
  ):
    with page.expect_response(_DATA_ROUTE_RE):
      open_tool("tpu-training", "hlo_stats")
    expect(page.locator("mat-toolbar")).to_be_visible(timeout=20000)

  # After unrouting faults, navigating to another tool recovers successfully
  switch_tool(page, "Memory Profile")
  expect(page.locator("memory-viewer, memory-profile")).to_be_visible(
      timeout=20000
  )
  assert_healthy(page, browser_errors, "api_500_and_corrupt_recovery")


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


def test_leaving_tool_mid_fetch_clears_loading_state(
    page: Page,
    open_tool: Callable[..., str],
    browser_errors: BrowserErrors,
) -> None:
  """Verifies leaving Overview or Op Profile mid-fetch unlocks later tools."""
  with _hold_data_request(page, browser_errors, "overview_page"):
    open_tool("tpu-training", "overview_page")
    # Wait for the overview fetch itself, not the first-load "Navigating" text.
    expect(
        page.locator(".loading-message", has_text="Loading overview data")
    ).to_be_visible(timeout=10000)
    switch_tool(page, "Memory Profile")

  expect(page.locator(".loading-message")).to_have_count(0, timeout=10000)
  expect(
      page.locator("main-page mat-sidenav-content > div.hidden-content")
  ).to_have_count(0, timeout=10000)
  assert_component_geometry(
      page, "memory-viewer, memory-profile", "leaving_overview_page_mid_fetch"
  )

  with _hold_data_request(page, browser_errors, "op_profile"):
    switch_tool(page, "Op Profile")
    expect(page.locator("mat-toolbar")).to_be_visible(timeout=10000)
    switch_tool(page, "Overview Page")

  expect(page.locator(".loading-message")).to_have_count(0, timeout=10000)
  assert_component_geometry(
      page, "overview-page, overview-viewer", "leaving_op_profile_mid_fetch"
  )
  assert_healthy(page, browser_errors, "leaving_tool_mid_fetch")
