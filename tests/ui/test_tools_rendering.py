"""Tests for visualization components: Memory Profile and Graph Viewer."""

# pylint: disable=g-doc-args,g-doc-return-or-yield,g-short-docstring-punctuation

import os
from conftest import BrowserErrors
from playwright.sync_api import expect
from playwright.sync_api import Page


def test_memory_profile_table_rendering(
    page: Page,
    server_url: str,
    logdir: str,
    browser_errors: BrowserErrors,
):
  """Verifies that the Memory Profile tool renders the allocation table."""
  session_path = os.path.join(logdir, "tpu-training")
  url = (
      f"{server_url}/?session_path={session_path}&run=tpu-training"
      "&tag=memory_profile"
  )
  page.goto(url, wait_until="domcontentloaded")

  memory_viewer = page.locator("memory-viewer, memory-profile")
  expect(memory_viewer).to_be_visible(timeout=20000)

  table_headers = memory_viewer.locator(
      "table th, table .mat-header-cell, button:has-text('Sort column')"
  )
  expect(table_headers.first).to_be_visible(timeout=10000)

  rows = memory_viewer.locator("table tr:has(td), table mat-row")
  expect(rows.first).to_be_visible(timeout=10000)
  assert rows.count() >= 1, "Memory breakdown table has zero data rows."

  browser_errors.assert_clean()


def test_graph_viewer_rendering(
    page: Page,
    server_url: str,
    logdir: str,
    browser_errors: BrowserErrors,
):
  """Verifies that the Graph Viewer tool mounts with valid dimensions."""
  session_path = os.path.join(logdir, "tpu-training")
  url = (
      f"{server_url}/?session_path={session_path}&run=tpu-training"
      "&tag=graph_viewer"
  )
  page.goto(url, wait_until="domcontentloaded")

  graph_viewer = page.locator(
      "graph-viewer, iframe.graph-viewer-iframe, .graph-viewer-container"
  ).first
  expect(graph_viewer).to_be_visible(timeout=20000)

  bbox = graph_viewer.bounding_box()
  assert bbox is not None, "Graph viewer bounding box is None"
  assert (
      bbox["width"] >= 200
  ), f"Graph viewer width collapsed: {bbox['width']}px"
  assert (
      bbox["height"] >= 200
  ), f"Graph viewer height collapsed: {bbox['height']}px"

  browser_errors.assert_clean()


def test_tool_switching_cleanup(
    page: Page,
    server_url: str,
    logdir: str,
    browser_errors: BrowserErrors,
):
  """Verifies that switching between tools cleans up previously mounted.

  component views.
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

  # 2. Switch to Memory Profile
  url_memory = (
      f"{server_url}/?session_path={session_path}&run=tpu-training"
      "&tag=memory_profile"
  )
  page.goto(url_memory, wait_until="domcontentloaded")
  expect(page.locator("memory-viewer, memory-profile")).to_be_visible(
      timeout=20000
  )
  expect(page.locator("overview-page, overview-viewer")).to_have_count(0)

  # 3. Switch to Graph Viewer
  url_graph = (
      f"{server_url}/?session_path={session_path}&run=tpu-training"
      "&tag=graph_viewer"
  )
  page.goto(url_graph, wait_until="domcontentloaded")
  expect(
      page.locator(
          "graph-viewer, iframe.graph-viewer-iframe, .graph-viewer-container"
      ).first
  ).to_be_visible(timeout=20000)
  expect(page.locator("memory-viewer, memory-profile")).to_have_count(0)

  browser_errors.assert_clean()
