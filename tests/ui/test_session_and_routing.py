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
"""Tests for session discovery, deep linking, and URL routing."""

# pylint: disable=g-doc-args,g-doc-return-or-yield,g-short-docstring-punctuation

from collections.abc import Callable
import os
import pathlib
import re
import urllib.parse

from playwright.sync_api import expect
from playwright.sync_api import Page
import pytest

from tests.ui.conftest import BrowserErrors
from tests.ui.journey_capture import settled_tool_url_pattern
from tests.ui.ui_helpers import assert_healthy
from tests.ui.ui_helpers import plot_graph_node
from tests.ui.ui_helpers import select_host
from tests.ui.ui_helpers import select_hosts
from tests.ui.ui_helpers import select_module
from tests.ui.ui_helpers import select_session
from tests.ui.ui_helpers import step_history
from tests.ui.ui_helpers import switch_tool


def test_deep_link_parameter_preservation(
    page: Page,
    open_tool: Callable[..., str],
    browser_errors: BrowserErrors,
) -> None:
  """Verifies deep-linked non-default host and tool parameters are preserved."""
  open_tool("multi-host", "hlo_stats", host="worker-1")

  expect(page.locator("hlo-stats").first).to_be_visible(timeout=20000)
  expect(
      page.locator("main-page mat-sidenav-content > div.full-height")
  ).to_be_visible(timeout=20000)
  expect(page).to_have_url(
      re.compile(r"[?&]host=worker-1(?:[&#]|$)"), timeout=10000
  )
  expect(
      page.locator("sidenav .item-container:has-text('Sessions') mat-select")
  ).to_contain_text("multi-host")
  expect(
      page.locator("sidenav .item-container:has-text('Tools') mat-select")
  ).to_contain_text("HLO Op Stats")
  expect(
      page.locator("sidenav .item-container:has-text('Hosts') mat-select")
  ).to_contain_text("worker-1")
  assert_healthy(page, browser_errors, "deep_link_parameter_preservation")


def test_multi_host_trace_viewer_selection_and_reload(
    page: Page,
    open_tool: Callable[..., str],
    browser_errors: BrowserErrors,
) -> None:
  """Verifies Trace Viewer multi-host selection updates URL across reloads."""
  open_tool("multi-host", "trace_viewer", host="worker-0")
  expect(page.locator("iframe, #filter-bar, .filter-bar").first).to_be_visible(
      timeout=20000
  )

  select_hosts(page, "worker-0", "worker-1")
  page.reload(wait_until="domcontentloaded")
  for host_name in ("worker-0", "worker-1"):
    expect(page).to_have_url(
        re.compile(
            rf"[?&]hosts=(?:[^&#]*(?:,|%2[Cc]))?"
            rf"{re.escape(host_name)}(?:,|%2[Cc]|[&#]|$)"
        ),
        timeout=20000,
    )
  expect(
      page.locator("sidenav .item-container:has-text('Hosts')")
  ).to_contain_text("Hosts (2)", timeout=15000)
  assert_healthy(page, browser_errors, "multi_host_trace_viewer_selection")


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Upstream SideNav.onTagSelectionChange (sidenav.ts:731) clears"
        " selectedHostInternal='' on tool switch, resetting worker-1 to"
        " ALL_HOSTS."
    ),
)
def test_host_selection_persists_across_tool_switches(
    page: Page,
    open_tool: Callable[..., str],
    browser_errors: BrowserErrors,
) -> None:
  """Verifies selecting worker-1 persists across SPA tool switches."""
  open_tool("multi-host", "hlo_stats", host="worker-0")
  expect(page.locator("hlo-stats").first).to_be_visible(timeout=20000)

  select_host(page, "worker-1")
  switch_tool(page, "Framework Op Stats")
  expect(page.locator("framework-op-stats").first).to_be_visible(timeout=20000)
  expect(page).to_have_url(
      re.compile(r"[?&]host=worker-1(?:[&#]|$)"), timeout=15000
  )
  expect(
      page.locator("sidenav .item-container:has-text('Hosts') mat-select")
  ).to_contain_text("worker-1")
  assert_healthy(
      page, browser_errors, "host_selection_persists_across_tool_switches"
  )


def test_graph_viewer_chip_click_preserves_run_query_param(
    page: Page,
    open_tool: Callable[..., str],
    browser_errors: BrowserErrors,
) -> None:
  """Verifies Graph Viewer chip navigation retains the run= query parameter."""
  run = open_tool("tpu-training", "graph_viewer")
  expect(page.locator("graph-viewer").first).to_be_visible(timeout=20000)

  plot_graph_node(page, "fusion.")
  expect(page).to_have_url(
      re.compile(rf"[?&]run={re.escape(run)}\b"), timeout=10000
  )
  expect(page).to_have_url(re.compile(r"[?&]tag=graph_viewer\b"), timeout=10000)
  assert_healthy(page, browser_errors, "graph_viewer_chip_url_preservation")


def test_memory_viewer_module_selection_persists_across_reload(
    page: Page,
    open_tool: Callable[..., str],
    browser_errors: BrowserErrors,
) -> None:
  """Verifies Memory Viewer keeps moduleName= in the URL and across reloads."""
  open_tool("tpu-training", "memory_viewer")
  summary = page.locator("memory-viewer-main")
  expect(summary).to_contain_text("Module Name: jit__where(", timeout=20000)

  select_module(page, "jit_train_step")
  expect(summary).to_contain_text("Module Name: jit_train_step(", timeout=20000)
  expect(summary).to_contain_text("12105.90", timeout=10000)
  expect(page).to_have_url(
      re.compile(r"[?&]moduleName=jit_train_step"), timeout=10000
  )

  page.reload(wait_until="domcontentloaded")
  expect(page).to_have_url(
      re.compile(r"[?&]moduleName=jit_train_step"), timeout=10000
  )
  expect(summary).to_contain_text("Module Name: jit_train_step(", timeout=20000)
  expect(summary).to_contain_text("12105.90", timeout=10000)
  expect(
      page.locator("sidenav .item-container:has-text('Hlo Modules') mat-select")
  ).to_contain_text("jit_train_step")
  assert_healthy(page, browser_errors, "memory_viewer_reload_persistence")


def test_cross_run_switch_from_exclusive_tool(
    page: Page,
    server_url: str,
    synthesized_logdir: str,
    resolve_run: Callable[[str], str],
    browser_errors: BrowserErrors,
) -> None:
  """Verifies switching runs from GPU Kernel Stats to TPU does not error."""
  if not os.path.exists(os.path.join(synthesized_logdir, "gpu-training")):
    pytest.skip("gpu-training fixture not present in logdir")
  tpu_run = resolve_run("tpu-training")

  query = urllib.parse.urlencode({
      "run_path": synthesized_logdir,
      "run": "gpu-training",
      "tag": "kernel_stats",
  })
  page.goto(
      f"{server_url.rstrip('/')}/?{query}",
      wait_until="domcontentloaded",
  )
  expect(
      page.locator("kernel-stats, kernel-stats-adapter").first
  ).to_be_visible(timeout=20000)
  expect(
      page.locator("main-page mat-sidenav-content > div.full-height")
  ).to_be_visible(timeout=20000)
  expect(
      page.locator("sidenav .item-container:has-text('Sessions') mat-select")
  ).to_be_visible(timeout=20000)

  select_session(page, tpu_run)
  expect(page).to_have_url(
      settled_tool_url_pattern("Overview Page"), timeout=20000
  )
  expect(
      page.locator("overview-page mat-card, overview-viewer mat-card").first
  ).to_be_visible(timeout=20000)
  expect(
      page.locator("main-page mat-sidenav-content > div.full-height")
  ).to_be_visible(timeout=20000)
  expect(page.locator("diagnostics-view .callout.is-critical")).to_have_count(0)
  assert_healthy(page, browser_errors, "cross_run_switch_from_exclusive_tool")


def _navigate_three_tools_and_step_back(
    page: Page,
    open_tool: Callable[..., str],
    middle_tool: str = "Roofline Model",
    middle_selector: str = "roofline-model",
) -> None:
  """Navigates Overview -> middle_tool -> Memory Profile, then steps back."""
  overview_card = page.locator(
      "overview-page mat-card, overview-viewer mat-card"
  ).first
  middle_view = page.locator(middle_selector).first
  tools_select = page.locator(
      "sidenav .item-container:has-text('Tools') mat-select"
  )
  open_tool("tpu-training", "overview_page")
  expect(overview_card).to_be_visible(timeout=20000)

  switch_tool(page, middle_tool)
  expect(middle_view).to_be_visible(timeout=20000)

  switch_tool(page, "Memory Profile")
  expect(page.locator("memory-viewer, memory-profile")).to_be_visible(
      timeout=20000
  )

  step_history(page, middle_tool)
  expect(tools_select).to_contain_text(middle_tool)
  expect(middle_view).to_be_visible(timeout=20000)


def test_browser_back_navigation(
    page: Page,
    open_tool: Callable[..., str],
    browser_errors: BrowserErrors,
) -> None:
  """Verifies browser back steps restore previous tools without split-brain."""
  _navigate_three_tools_and_step_back(page, open_tool)

  step_history(page, "Overview Page")
  expect(
      page.locator("sidenav .item-container:has-text('Tools') mat-select")
  ).to_contain_text("Overview Page")
  expect(
      page.locator("overview-page mat-card, overview-viewer mat-card").first
  ).to_be_visible(timeout=20000)
  assert_healthy(page, browser_errors, "browser_back_navigation")


def test_browser_forward_navigation(
    page: Page,
    open_tool: Callable[..., str],
    browser_errors: BrowserErrors,
) -> None:
  """Verifies a single browser forward step restores the next tool cleanly."""
  _navigate_three_tools_and_step_back(page, open_tool)

  step_history(page, "Memory Profile", forward=True)
  expect(
      page.locator("sidenav .item-container:has-text('Tools') mat-select")
  ).to_contain_text("Memory Profile")
  expect(page.locator("memory-viewer, memory-profile")).to_be_visible(
      timeout=20000
  )
  assert_healthy(page, browser_errors, "browser_forward_navigation")


def test_trace_viewer_navigation_preserves_browser_history(
    page: Page,
    open_tool: Callable[..., str],
    browser_errors: BrowserErrors,
) -> None:
  """Verifies Trace Viewer navigation preserves browser history on back."""
  _navigate_three_tools_and_step_back(
      page,
      open_tool,
      middle_tool="Trace Viewer",
      middle_selector="trace-viewer, iframe, #filter-bar, .filter-bar",
  )
  assert_healthy(page, browser_errors, "trace_viewer_browser_back_navigation")


def test_overview_page_roofline_model_link_navigates_single_and_multi_host(
    page: Page,
    logdir: str,
    resolve_run: Callable[[str], str],
    open_tool: Callable[..., str],
    browser_errors: BrowserErrors,
) -> None:
  """Verifies Overview Page roofline_model link on 1- and N-host runs."""
  run = resolve_run("tpu-training")
  run_dir = pathlib.Path(logdir, run)
  traces = list(run_dir.glob("*.xplane.pb"))
  assert traces, f"No *.xplane.pb traces found in {run_dir}"
  target_host = min(trace.name.removesuffix(".xplane.pb") for trace in traces)
  encoded_run = re.escape(urllib.parse.quote_plus(run))
  encoded_host = re.escape(urllib.parse.quote_plus(target_host))
  open_tool(run, "overview_page", host=target_host)

  roofline_link = page.locator(
      "performance-summary a", has_text="roofline_model"
  ).first
  expect(roofline_link).to_be_visible(timeout=20000)
  expect(roofline_link).to_have_attribute(
      "href",
      re.compile(
          r"^index\.html(?=\?)"
          rf"(?=.*[?&]run={encoded_run}(?:[&#]|$))"
          r"(?=.*[?&]tag=roofline_model(?:[&#]|$))"
          rf"(?=.*[?&]host={encoded_host}(?:[&#]|$))"
          r"(?!.*[?&]host=ALL_HOSTS(?:[&#]|$))"
      ),
  )

  roofline_link.click()
  expect(page).to_have_url(
      settled_tool_url_pattern("Roofline Model"), timeout=20000
  )
  expect(page).to_have_url(
      re.compile(rf"[?&]host={encoded_host}(?:[&#]|$)"), timeout=10000
  )
  expect(page.locator("roofline-model")).to_be_visible(timeout=20000)
  expect(page.locator("empty-page")).to_have_count(0)
  expect(
      page.locator("sidenav .item-container:has-text('Tools') mat-select")
  ).to_contain_text("Roofline Model")
  expect(
      page.locator("sidenav .item-container:has-text('Hosts') mat-select")
  ).to_contain_text(target_host)
  assert_healthy(
      page, browser_errors, "overview_page_roofline_model_link_single_host"
  )

  mh_run = open_tool("multi-host", "overview_page", host="ALL_HOSTS")
  encoded_mh_run = re.escape(urllib.parse.quote_plus(mh_run))
  mh_link = page.locator(
      "performance-summary a", has_text="roofline_model"
  ).first
  expect(mh_link).to_be_visible(timeout=20000)
  expect(mh_link).to_have_attribute(
      "href",
      re.compile(
          r"^index\.html(?=\?)"
          rf"(?=.*[?&]run={encoded_mh_run}(?:[&#]|$))"
          r"(?=.*[?&]tag=roofline_model(?:[&#]|$))"
          r"(?!.*[?&]host=)"
      ),
  )

  mh_link.click()
  expect(page).to_have_url(
      settled_tool_url_pattern("Roofline Model"), timeout=20000
  )
  expect(page).not_to_have_url(re.compile(r"[?&]host=ALL_HOSTS(?:[&#]|$)"))
  expect(page.locator("roofline-model")).to_be_visible(timeout=20000)
  expect(page.locator("empty-page")).to_have_count(0)
  expect(
      page.locator("sidenav .item-container:has-text('Tools') mat-select")
  ).to_contain_text("Roofline Model")
  expect(
      page.locator("sidenav .item-container:has-text('Hosts') mat-select")
  ).not_to_contain_text("ALL_HOSTS")
  assert_healthy(
      page, browser_errors, "overview_page_roofline_model_link_multi_host"
  )
