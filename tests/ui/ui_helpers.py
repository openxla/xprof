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


def _select_option(
    page: sync_api.Page, dropdown: sync_api.Locator, *labels: str
) -> None:
  """Picks the option of a mat-select whose text is one of `labels`.

  An option may end in a parenthesized number, such as an HLO module's program
  id, which differs between profiles.

  Args:
    page: Page holding the dropdown.
    dropdown: The mat-select to open.
    *labels: Accepted option texts.
  """
  pattern = re.compile(
      rf"^\s*(?:{'|'.join(re.escape(label) for label in labels)})"
      r"(?:\(\d+\))?\s*$"
  )
  sync_api.expect(dropdown).to_be_visible(timeout=10000)
  dropdown.click()
  option = page.locator("mat-option").filter(has_text=pattern).first
  sync_api.expect(option).to_be_visible(timeout=5000)
  option.click()
  try:
    page.mouse.move(0, 0)
  except invariants.PlaywrightError as err:
    logging.debug("Ignored mouse reset error after dropdown close: %s", err)
  sync_api.expect(page.locator("mat-option")).to_have_count(0, timeout=5000)
  sync_api.expect(dropdown).to_have_text(pattern, timeout=5000)


def _select_sidenav_dropdown_option(
    page: sync_api.Page, label: str, option_text: str
) -> None:
  """Opens sidenav and clicks an option within the specified dropdown."""
  ensure_sidenav_open(page)
  aliases = [option_text]
  if option_text in ("Op Profile", "HLO Op Profile"):
    aliases = ["Op Profile", "HLO Op Profile"]
  _select_option(
      page,
      page.locator(f"sidenav .item-container:has-text('{label}') mat-select"),
      *aliases,
  )


def switch_tool(page: sync_api.Page, tool_name: str) -> None:
  """Opens the navigation drawer and switches tools via dropdown."""
  _select_sidenav_dropdown_option(page, "Tools", tool_name)


def select_host(page: sync_api.Page, host_name: str) -> None:
  """Opens the navigation drawer and selects a worker host from dropdown."""
  _select_sidenav_dropdown_option(page, "Hosts", host_name)


def select_module(page: sync_api.Page, module_name: str) -> None:
  """Opens the navigation drawer and selects an HLO module from dropdown."""
  _select_sidenav_dropdown_option(page, "Hlo Modules", module_name)


def select_op_profile_group_by(page: sync_api.Page, group_by: str) -> None:
  """Selects an Op Profile grouping: Program, Category or Provenance."""
  _select_option(
      page,
      page.locator(
          "op-profile-base mat-form-field:has(mat-label:text-is('Group by'))"
          " mat-select"
      ),
      group_by,
  )


def expand_op_profile_row(page: sync_api.Page, row_text: str) -> None:
  """Expands the first Op Profile row containing `row_text` and selects it."""
  row = (
      page.locator("op-table-entry .row:visible")
      .filter(has_text=row_text)
      .first
  )
  sync_api.expect(row).to_be_visible(timeout=10000)
  row.click()
  page.mouse.move(0, 0)
  # Both triangles stay in the DOM; only the rendered one is the state.
  sync_api.expect(row.locator(".disclosure")).to_have_text(
      "\u25bc", use_inner_text=True, timeout=5000
  )
  sync_api.expect(
      page.locator("op-details .info-header-title")
  ).to_contain_text(row_text, timeout=10000)


def plot_graph_node(page: sync_api.Page, node_text: str) -> None:
  """Clicks a Graph Viewer quick-option chip to plot its HLO graph."""
  chip = (
      page.locator("graph-viewer mat-chip-option:visible")
      .filter(has_text=node_text)
      .first
  )
  sync_api.expect(chip).to_be_visible(timeout=15000)
  chip.click()
  sync_api.expect(page).to_have_url(
      re.compile(rf"[?&]node_name=[^&#]*{re.escape(node_text)}"),
      timeout=20000,
  )
  graph_node = (
      page.frame_locator("iframe#graph-html").locator("svg g.node").first
  )
  sync_api.expect(graph_node).to_be_visible(timeout=20000)
  page.mouse.move(0, 0)


def select_memory_id(page: sync_api.Page, memory_id: str) -> None:
  """Selects a memory ID once Memory Profile has loaded its data."""
  sync_api.expect(page.locator("memory-breakdown-table")).to_be_visible(
      timeout=20000
  )
  _select_option(
      page, page.locator("#memory-id-selector mat-select"), memory_id
  )


def select_category_filter(page: sync_api.Page, category: str) -> None:
  """Selects an operation category in HLO Op Stats or Roofline Model."""
  _select_option(
      page,
      page.locator(
          "category-filter:has(mat-label:has-text('Category')) mat-select"
      ),
      category,
  )


def filter_table_rows(page: sync_api.Page, text: str) -> None:
  """Filters the tool's op table by `text` and checks that only matches stay.

  The table must first show a row without `text`, so a filter that has
  nothing to hide cannot pass.

  Args:
    page: Page showing HLO Op Stats, Framework Op Stats or Memory Profile.
    text: Text to filter by, matched case-insensitively.
  """
  table = page.locator(
      "hlo-stats:visible, stats-table:visible, memory-breakdown-table:visible"
  ).first
  rows = table.locator(".google-visualization-table-table tbody tr")
  others = rows.filter(has_not_text=re.compile(re.escape(text), re.IGNORECASE))
  sync_api.expect(others.first).to_be_visible(timeout=20000)
  filter_input = table.locator(
      "string-filter[column='hlo_op_expression'] input,"
      " mat-form-field:has(mat-label:text-is('Operation Type')) input,"
      " input[placeholder='Operation']"
  )
  filter_input.fill(text)
  # Blurring commits the value where the table filters on change, not input.
  filter_input.blur()
  sync_api.expect(others).to_have_count(0, timeout=10000)
  sync_api.expect(rows.first).to_be_visible()
  page.mouse.move(0, 0)


def sort_table_column(page: sync_api.Page, column_name: str) -> None:
  """Clicks a Google Visualization table header to sort by `column_name`."""
  header = (
      page.locator(".google-visualization-table-table thead th:visible")
      .filter(has_text=column_name)
      .first
  )
  sorted_class = re.compile(r"\bsort-(?:ascending|descending)\b")
  sync_api.expect(header).to_be_visible(timeout=10000)
  sync_api.expect(header).not_to_have_class(sorted_class)
  header.click()
  sync_api.expect(header).to_have_class(sorted_class, timeout=5000)
  page.mouse.move(0, 0)
