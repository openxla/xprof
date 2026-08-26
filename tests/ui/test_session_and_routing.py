"""Tests for session discovery, deep linking, and URL routing."""

# pylint: disable=g-doc-args,g-doc-return-or-yield,g-short-docstring-punctuation

import os
from conftest import BrowserErrors
from playwright.sync_api import expect
from playwright.sync_api import Page


def test_deep_link_parameter_preservation(
    page: Page,
    server_url: str,
    logdir: str,
    browser_errors: BrowserErrors,
):
  """Verifies that deep-linked URL parameters are preserved and reflected.

  in UI.
  """
  session_path = os.path.join(logdir, "tpu-training")
  target_host = "gke-tpu-b309f56b-rq5s"
  deep_link = (
      f"{server_url}/?session_path={session_path}&run=tpu-training"
      f"&tag=overview_page&host={target_host}"
  )
  page.goto(deep_link, wait_until="domcontentloaded")

  overview_comp = page.locator("overview-page, overview-viewer")
  expect(overview_comp).to_be_visible(timeout=20000)

  selects = page.locator("mat-select")
  expect(selects.nth(0)).to_contain_text("tpu-training")
  expect(selects.nth(1)).to_contain_text("Overview Page")

  browser_errors.assert_clean()


def test_browser_back_forward_history_navigation(
    page: Page,
    server_url: str,
    logdir: str,
    browser_errors: BrowserErrors,
):
  """Verifies browser Back and Forward navigation restores previous tool.

  views.
  """
  session_path = os.path.join(logdir, "tpu-training")

  # 1. Overview Page
  url_overview = (
      f"{server_url}/?session_path={session_path}&run=tpu-training"
      "&tag=overview_page"
  )
  page.goto(url_overview, wait_until="domcontentloaded")
  expect(
      page.locator("overview-page mat-card, overview-viewer mat-card").first
  ).to_be_visible(timeout=20000)

  # 2. Switch to Memory Profile via UI
  tools_dropdown = page.locator("mat-select").nth(1)
  tools_dropdown.click()
  page.locator("mat-option:has-text('Memory Profile')").click()
  expect(page.locator("memory-viewer, memory-profile")).to_be_visible(
      timeout=20000
  )

  # 3. Back
  page.go_back(wait_until="domcontentloaded")
  expect(
      page.locator("overview-page mat-card, overview-viewer mat-card").first
  ).to_be_visible(timeout=20000)

  # 4. Forward
  page.go_forward(wait_until="domcontentloaded")
  expect(page.locator("memory-viewer, memory-profile")).to_be_visible(
      timeout=20000
  )

  browser_errors.assert_clean()
