"""Shared Playwright UI interaction helpers for XProf frontend tests."""

import logging
import pathlib
import re
import urllib.parse
from playwright import sync_api

# pylint: disable=g-import-not-at-top
try:
  from tests.ui.invariants import run_content_invariants
except ImportError:
  from invariants import run_content_invariants


def assert_healthy(
    page: sync_api.Page,
    browser_errors: object = None,
    context: str = "",
) -> None:
  """Asserts that page renders non-empty body, no poison tokens, and clean logs."""
  text = page.inner_text("body")
  ctx = f" at {context}" if context else ""
  assert text.strip(), f"Empty page body rendered{ctx}"
  violations = run_content_invariants(text)
  assert not violations, f"Poison tokens detected{ctx}: {violations}"
  if browser_errors is not None and hasattr(browser_errors, "assert_clean"):
    browser_errors.assert_clean(context)


def build_tool_url(
    server_url: str,
    session_path: str,
    run: str,
    tag: str,
    **extra_params: str,
) -> str:
  """Constructs a normalized, URL-encoded XProf tool URL."""
  base = server_url.rstrip("/")
  params = {
      "session_path": pathlib.Path(session_path).as_posix(),
      "run": run,
      "tag": tag,
      **extra_params,
  }
  return f"{base}/?{urllib.parse.urlencode(params)}"


def ensure_sidenav_open(page: sync_api.Page) -> None:
  """Ensures the navigation drawer is open and ready for user interactions."""
  drawer = page.locator("mat-sidenav:has(sidenav)")
  sync_api.expect(drawer).to_be_attached(timeout=10000)
  drawer_classes = drawer.get_attribute("class") or ""
  if "mat-drawer-opened" not in drawer_classes:
    toggle_btn = page.locator("button.sidenav-toggle-button")
    toggle_btn.click()
  sync_api.expect(
      page.locator("mat-sidenav.mat-drawer-opened:has(sidenav)")
  ).to_be_visible(timeout=5000)


def _select_sidenav_dropdown_option(
    page: sync_api.Page, label: str, option_text: str
) -> None:
  """Opens sidenav and clicks an option within the specified dropdown."""
  ensure_sidenav_open(page)
  dropdown = page.locator(
      f"sidenav .item-container:has-text('{label}') mat-select"
  )
  sync_api.expect(dropdown).to_be_visible(timeout=5000)
  dropdown.click()
  aliases = [option_text]
  if option_text in ("Op Profile", "HLO Op Profile"):
    aliases = ["Op Profile", "HLO Op Profile"]
  pattern = re.compile(
      rf"^\s*(?:{'|'.join(re.escape(a) for a in aliases)})\s*$"
  )
  option = page.locator("mat-option").filter(has_text=pattern).first
  sync_api.expect(option).to_be_visible(timeout=5000)
  option.click()
  sync_api.expect(page.locator(".cdk-overlay-pane")).to_have_count(
      0, timeout=5000
  )
  try:
    page.mouse.move(0, 0)
  except sync_api.Error as err:
    logging.debug("Ignored mouse reset error after dropdown close: %s", err)


def switch_tool(page: sync_api.Page, tool_name: str) -> None:
  """Opens the navigation drawer and switches tools via dropdown."""
  _select_sidenav_dropdown_option(page, "Tools", tool_name)


def select_host(page: sync_api.Page, host_name: str) -> None:
  """Opens the navigation drawer and selects a worker host from dropdown."""
  _select_sidenav_dropdown_option(page, "Hosts", host_name)
