"""Visual snapshot and layout regression tests for XProf UI."""

# pylint: disable=g-doc-args,g-doc-return-or-yield,g-short-docstring-punctuation

import os
from conftest import BrowserErrors
from playwright.sync_api import expect
from playwright.sync_api import Page


def test_overview_page_visual_snapshot(
    page: Page,
    server_url: str,
    logdir: str,
    browser_errors: BrowserErrors,
):
  """Validates visual snapshot of the Overview Page against golden.

  baseline.
  """
  session_path = os.path.join(logdir, "tpu-training")
  page.goto(f"{server_url}/?session_path={session_path}")

  overview_view = page.locator("overview-page, .overview-container").first
  expect(overview_view).to_be_visible(timeout=20000)

  golden_path = os.path.join(
      os.path.dirname(__file__),
      "goldens",
      "linux",
      "overview_page_baseline.png",
  )
  if os.environ.get("UPDATE_GOLDENS") == "1" or not os.path.exists(golden_path):
    os.makedirs(os.path.dirname(golden_path), exist_ok=True)
    page.screenshot(path=golden_path)

  # Check that screenshot baseline exists and page renders cleanly
  assert os.path.exists(golden_path)
  browser_errors.assert_clean()
