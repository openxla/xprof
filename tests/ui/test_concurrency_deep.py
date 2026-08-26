"""Tests for concurrent user actions and tool switching."""

# pylint: disable=g-doc-args,g-doc-return-or-yield,g-short-docstring-punctuation

import os
from conftest import BrowserErrors
from playwright.sync_api import expect
from playwright.sync_api import Page


def test_rapid_tool_navigation_stability(
    page: Page,
    server_url: str,
    logdir: str,
    browser_errors: BrowserErrors,
):
  """Verifies that rapid consecutive tool switches do not crash the app.

  shell.
  """
  session_path = os.path.join(logdir, "tpu-training")
  tools = ["memory_profile", "graph_viewer", "overview_page"]

  for tool_tag in tools:
    url_tool = (
        f"{server_url}/?session_path={session_path}&run=tpu-training"
        f"&tag={tool_tag}"
    )
    page.goto(url_tool, wait_until="domcontentloaded")
    page.wait_for_timeout(300)

  expect(page.locator("overview-page, overview-viewer")).to_be_visible(
      timeout=20000
  )
  browser_errors.assert_clean()
