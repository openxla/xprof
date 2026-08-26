"""Tests for the Overview Page components and performance summary metrics."""

# pylint: disable=g-doc-args,g-doc-return-or-yield,g-short-docstring-punctuation

import os
import re
from conftest import BrowserErrors
from playwright.sync_api import expect
from playwright.sync_api import Page


def test_performance_summary_metrics(
    page: Page,
    server_url: str,
    logdir: str,
    browser_errors: BrowserErrors,
):
  """Verifies that the Performance Summary table renders execution metrics."""
  session_path = os.path.join(logdir, "tpu-training")
  url = (
      f"{server_url}/?session_path={session_path}&run=tpu-training"
      "&tag=overview_page"
  )
  page.goto(url, wait_until="domcontentloaded")

  overview = page.locator("overview-page, overview-viewer")
  expect(overview).to_be_visible(timeout=20000)

  summary_card = overview.locator("mat-card:has-text('Performance Summary')")
  expect(summary_card).to_be_visible(timeout=10000)
  expect(summary_card).to_contain_text("Average Step Time")

  step_time_text = summary_card.inner_text()
  assert re.search(
      r"\d+(\.\d+)?\s*ms", step_time_text
  ), f"Average Step Time missing millisecond value: {step_time_text}"
  browser_errors.assert_clean()


def test_step_time_graph_geometry(
    page: Page,
    server_url: str,
    logdir: str,
    browser_errors: BrowserErrors,
):
  """Verifies that the Step-time Graph component renders with non-zero.

  dimensions.
  """
  session_path = os.path.join(logdir, "tpu-training")
  url = (
      f"{server_url}/?session_path={session_path}&run=tpu-training"
      "&tag=overview_page"
  )
  page.goto(url, wait_until="domcontentloaded")

  graph_comp = page.locator("step-time-graph")
  expect(graph_comp).to_be_visible(timeout=20000)

  bbox = graph_comp.bounding_box()
  assert bbox is not None, "Step-time graph bounding box is None"
  assert bbox["width"] >= 200, f"Graph width collapsed: {bbox['width']}px"
  assert bbox["height"] >= 100, f"Graph height collapsed: {bbox['height']}px"

  browser_errors.assert_clean()


def test_host_selector_interaction(
    page: Page,
    server_url: str,
    logdir: str,
    browser_errors: BrowserErrors,
):
  """Verifies that selecting hosts in the navbar dropdown updates the view."""
  session_path = os.path.join(logdir, "tpu-training")
  url = (
      f"{server_url}/?session_path={session_path}&run=tpu-training"
      "&tag=overview_page"
  )
  page.goto(url, wait_until="domcontentloaded")

  host_select = page.locator("mat-select").nth(2)
  expect(host_select).to_be_visible(timeout=20000)

  host_select.click()
  options = page.locator("mat-option")
  expect(options.first).to_be_visible(timeout=5000)

  first_option_text = options.first.inner_text().strip()
  options.first.click()

  expect(host_select).to_contain_text(first_option_text)
  browser_errors.assert_clean()
