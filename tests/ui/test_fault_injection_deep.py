"""Tests for network fault injection, API errors (403, 500), and client.

resilience.
"""

# pylint: disable=g-doc-args,g-doc-return-or-yield,g-short-docstring-punctuation

import os
import re
from conftest import BrowserErrors
from playwright.sync_api import expect
from playwright.sync_api import Page


def test_api_403_forbidden_resilience(
    page: Page,
    server_url: str,
    logdir: str,
    browser_errors: BrowserErrors,
):
  """Verifies that the application remains interactive during 403.

  Forbidden responses.
  """
  browser_errors.ignore("403")
  session_path = os.path.join(logdir, "tpu-training")
  route_pattern = re.compile(r".*/data/plugin/profile/data.*")

  def handle_403(route):
    route.fulfill(
        status=403,
        content_type="application/json",
        body='{"error": "AccessDenied", "message": "Authentication required."}',
    )

  page.route(route_pattern, handle_403)
  try:
    url = f"{server_url}/?session_path={session_path}"
    page.goto(url, wait_until="domcontentloaded")

    toolbar = page.locator("mat-toolbar")
    expect(toolbar).to_be_visible(timeout=20000)
    expect(toolbar).to_contain_text("XProf")

    tools_dropdown = page.locator("mat-select").nth(1)
    expect(tools_dropdown).to_be_visible()
    tools_dropdown.click()
    expect(page.locator("mat-option").first).to_be_visible(timeout=5000)
  finally:
    page.unroute(route_pattern)


def test_api_500_backend_recovery(
    page: Page,
    server_url: str,
    logdir: str,
    browser_errors: BrowserErrors,
):
  """Verifies recovery after 500 Internal Server Error."""
  browser_errors.ignore("500")
  session_path = os.path.join(logdir, "tpu-training")
  route_pattern = re.compile(r".*/data/plugin/profile/data.*")
  should_fail = [True]

  def handle_data(route):
    if should_fail[0]:
      route.fulfill(
          status=500,
          content_type="application/json",
          body='{"error": "InternalServerError"}',
      )
    else:
      route.continue_()

  page.route(route_pattern, handle_data)
  try:
    url = f"{server_url}/?session_path={session_path}"
    page.goto(url, wait_until="domcontentloaded")

    toolbar = page.locator("mat-toolbar")
    expect(toolbar).to_be_visible(timeout=20000)

    # Recover
    should_fail[0] = False
    tools_dropdown = page.locator("mat-select").nth(1)
    tools_dropdown.click()
    page.locator("mat-option:has-text('Memory Profile')").click()

    expect(page.locator("memory-viewer, memory-profile")).to_be_visible(
        timeout=20000
    )
  finally:
    page.unroute(route_pattern)
