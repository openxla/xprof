"""Parametrized tool rendering and navigation verification across runs."""

# pylint: disable=g-doc-args,g-doc-return-or-yield,g-short-docstring-punctuation

import os
from typing import List
from conftest import BrowserErrors
from playwright.sync_api import expect
from playwright.sync_api import Page
import pytest

PRIMARY_TOOLS: List[str] = [
    "overview_page",
    "trace_viewer",
    "framework_op_stats",
    "hlo_op_stats",
    "hlo_op_profile",
    "op_profile",
    "input_pipeline_analyzer",
    "kernel_stats",
    "memory_profile",
    "memory_viewer",
    "graph_viewer",
    "roofline_model",
    "pod_viewer",
    "perf_counters",
    "utilization_viewer",
    "megascale_stats",
    "inference_profile",
]


def test_tool_navigation_shell(
    page: Page,
    server_url: str,
    logdir: str,
    browser_errors: BrowserErrors,
):
  """Verifies that navigation to the application loads the shell."""
  session_path = os.path.join(logdir, "tpu-training")
  url = (
      f"{server_url}/?session_path={session_path}&run=tpu-training"
      "&tag=overview_page"
  )
  page.goto(url, wait_until="domcontentloaded")

  toolbar = page.locator("mat-toolbar")
  expect(toolbar).to_be_visible(timeout=20000)
  expect(toolbar).to_contain_text("XProf")
  browser_errors.assert_clean()


@pytest.mark.parametrize("tag", PRIMARY_TOOLS)
def test_individual_tool_loads(
    page: Page,
    server_url: str,
    logdir: str,
    browser_errors: BrowserErrors,
    tag: str,
):
  """Verifies that deep-linking directly to a tool loads the application.

  shell cleanly.
  """
  browser_errors.ignore("EmptyError")
  session_path = os.path.join(logdir, "tpu-training")

  url = f"{server_url}/?session_path={session_path}&run=tpu-training&tag={tag}"
  page.goto(url, wait_until="domcontentloaded")

  toolbar = page.locator("mat-toolbar")
  expect(toolbar).to_be_visible(timeout=20000)
  browser_errors.assert_clean()
