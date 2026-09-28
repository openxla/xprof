"""Shared Playwright UI interaction helpers for XProf frontend tests."""

import logging
import pathlib
import re
import urllib.parse

# pylint: disable=g-import-not-at-top
try:
  from google3.third_party.xprof.tests.ui import invariants
except ImportError:
  try:
    from tests.ui import invariants  # pyrefly: ignore[missing-import]
  except ImportError:
    import invariants  # pyrefly: ignore[missing-import]
# pylint: enable=g-import-not-at-top

# invariants owns the optional Playwright import and its hermetic stub.
sync_api = invariants.sync_api

# Genuine rendered visualization and data-bearing elements inside XProf tools.
# Deliberately excludes wrappers that exist before anything is drawn (e.g.
# :scope > *, div, the <chart> host) and the decorative svgs Angular Material
# marks focusable="false" or aria-hidden="true" (select arrows, icons,
# spinners, switches), so child geometry checks cannot pass on page chrome.
# An iframe counts once it is visible; its document is not inspected, so an
# iframe that has not drawn anything yet (Graph Viewer's graph) still counts.
VISUALIZATION_SELECTOR = (
    "svg:not([focusable='false'], [aria-hidden='true']), canvas, table,"
    " mat-card, .mat-mdc-card, iframe, op-table-entry .row"
)


def assert_component_geometry(
    page: sync_api.Page,
    selector: str,
    context: str = "",
) -> None:
  """Asserts that a component and its visualization have positive geometry."""
  ctx = f" at {context}" if context else ""
  comp = page.locator(f":is({selector}):visible").first
  sync_api.expect(comp).to_be_visible(timeout=20000)
  bbox = comp.bounding_box()
  assert (
      bbox is not None and bbox["width"] > 0 and bbox["height"] > 0
  ), f"Component {selector!r} collapsed{ctx}: {bbox}"

  vis_selector = (
      f":is({selector}):is({VISUALIZATION_SELECTOR}):visible, "
      f":is({selector}) :is({VISUALIZATION_SELECTOR}):visible"
  )
  child = page.locator(vis_selector).first
  sync_api.expect(child).to_be_visible(timeout=20000)
  child_bbox = child.bounding_box()
  assert (
      child_bbox is not None
      and child_bbox["width"] > 0
      and child_bbox["height"] > 0
  ), f"Component {selector!r} child visualization collapsed{ctx}: {child_bbox}"


def assert_healthy(
    page: sync_api.Page,
    browser_errors: object = None,
    context: str = "",
) -> None:
  """Asserts a non-empty body, clean page invariants and clean browser logs."""
  ctx = f" at {context}" if context else ""
  body = page.locator("body")
  sync_api.expect(body).to_contain_text(re.compile(r"\S"), timeout=20000)
  text = body.inner_text()
  assert text.strip(), f"Empty page body rendered{ctx}"
  violations = invariants.run_content_invariants(text)
  assert not violations, f"Poison tokens detected{ctx}: {violations}"
  invariants.assert_page_invariants(page)
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
  except invariants.PlaywrightError as err:
    logging.debug("Ignored mouse reset error after dropdown close: %s", err)


def switch_tool(page: sync_api.Page, tool_name: str) -> None:
  """Opens the navigation drawer and switches tools via dropdown."""
  _select_sidenav_dropdown_option(page, "Tools", tool_name)


def select_host(page: sync_api.Page, host_name: str) -> None:
  """Opens the navigation drawer and selects a worker host from dropdown."""
  _select_sidenav_dropdown_option(page, "Hosts", host_name)
