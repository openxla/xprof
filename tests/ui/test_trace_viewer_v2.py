"""Tests for Trace Viewer component and navigation lifecycle."""

# pylint: disable=g-doc-args,g-doc-return-or-yield,g-short-docstring-punctuation

import os
from conftest import BrowserErrors
from playwright.sync_api import expect
from playwright.sync_api import Page


def test_trace_viewer_mounts_canvas(
    page: Page,
    server_url: str,
    logdir: str,
    browser_errors: BrowserErrors,
):
  """Verifies that Trace Viewer mounts its container/iframe with positive.

  dimensions.
  """
  session_path = os.path.join(logdir, "tpu-training")
  host = "gke-tpu-b309f56b-rq5s"
  url = (
      f"{server_url}/?session_path={session_path}&run=tpu-training"
      f"&tag=trace_viewer&host={host}"
  )
  page.goto(url, wait_until="domcontentloaded")

  trace_element = page.locator(
      "iframe, .trace-viewer-container, #filter-bar, canvas"
  ).first
  expect(trace_element).to_be_visible(timeout=20000)

  bbox = trace_element.bounding_box()
  assert bbox is not None, "Trace viewer element bounding box is None"
  assert (
      bbox["width"] >= 300
  ), f"Trace viewer element width collapsed: {bbox['width']}px"
  assert (
      bbox["height"] >= 200
  ), f"Trace viewer element height collapsed: {bbox['height']}px"

  browser_errors.assert_clean()


def test_trace_viewer_lifecycle_teardown(
    page: Page,
    server_url: str,
    logdir: str,
    browser_errors: BrowserErrors,
):
  """Verifies navigating away from Trace Viewer unmounts iframe cleanly."""
  session_path = os.path.join(logdir, "tpu-training")
  host = "gke-tpu-b309f56b-rq5s"

  url_trace = (
      f"{server_url}/?session_path={session_path}&run=tpu-training"
      f"&tag=trace_viewer&host={host}"
  )
  page.goto(url_trace, wait_until="domcontentloaded")
  expect(page.locator("iframe, #filter-bar, .filter-bar").first).to_be_visible(
      timeout=20000
  )

  url_overview = (
      f"{server_url}/?session_path={session_path}&run=tpu-training"
      "&tag=overview_page"
  )
  page.goto(url_overview, wait_until="domcontentloaded")
  expect(page.locator("overview-page, overview-viewer")).to_be_visible(
      timeout=20000
  )
  expect(page.locator("iframe")).to_have_count(0)

  browser_errors.assert_clean()
