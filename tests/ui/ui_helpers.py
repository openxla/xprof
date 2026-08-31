"""Shared Playwright UI interaction helpers for XProf frontend tests."""

import pathlib
import urllib.parse
from playwright.sync_api import expect
from playwright.sync_api import Page


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


def ensure_sidenav_open(page: Page) -> None:
  """Ensures the navigation drawer is open and ready for user interactions."""
  drawer = page.locator("mat-sidenav:has(sidenav)")
  expect(drawer).to_be_attached(timeout=10000)
  if not drawer.evaluate("el => el.classList.contains('mat-drawer-opened')"):
    page.locator("button.sidenav-toggle-button").click()
    expect(
        page.locator("mat-sidenav.mat-drawer-opened:has(sidenav)")
    ).to_be_visible(timeout=5000)


def switch_tool(page: Page, tool_name: str) -> None:
  """Opens the navigation drawer and switches tools via dropdown."""
  ensure_sidenav_open(page)
  dropdown = page.locator(
      "sidenav .item-container:has-text('Tools') mat-select"
  )
  expect(dropdown).to_be_visible(timeout=5000)
  dropdown.click()
  option = page.locator("mat-option").filter(has_text=tool_name).first
  expect(option).to_be_visible(timeout=5000)
  option.click()
  expect(page.locator(".cdk-overlay-pane")).to_have_count(0, timeout=5000)
