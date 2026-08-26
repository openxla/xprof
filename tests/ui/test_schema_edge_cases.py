"""Tests for schema boundary conditions and empty session directories."""

# pylint: disable=g-doc-args,g-doc-return-or-yield,g-short-docstring-punctuation

import tempfile
from conftest import BrowserErrors
from playwright.sync_api import expect
from playwright.sync_api import Page


def test_empty_session_directory_clean_fallback(
    page: Page,
    server_url: str,
    browser_errors: BrowserErrors,
):
  """Verifies that an empty directory displays the clean empty-state view."""
  with tempfile.TemporaryDirectory() as empty_dir:
    url = f"{server_url}/?session_path={empty_dir}"
    page.goto(url, wait_until="domcontentloaded")

    empty_heading = page.locator("text='No profile data was found.'")
    expect(empty_heading).to_be_visible(timeout=20000)

    capture_btn = page.locator("button:has-text('CAPTURE PROFILE')")
    expect(capture_btn).to_be_visible()

    browser_errors.assert_clean()
