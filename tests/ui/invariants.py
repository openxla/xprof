"""Generic, tool-agnostic invariants for surfacing data-pipeline bugs."""

# pylint: disable=g-doc-args,g-doc-return-or-yield,g-short-docstring-punctuation

import re
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import Locator
from playwright.sync_api import Page

POISON_PATTERNS: dict[str, str] = {
    "NaN": r"\bNaN\b",
    "undefined": r"\bundefined\b",
    "[object Object]": r"\[object Object\]",
    "Infinity": r"-?\bInfinity\b",
    "null": r"(?<![\"'])\bnull\b(?![\"'])",
    "(null)": r"\(\s*null\s*\)",
    "INVALID": r"\bINVALID\b",
}

# Values a broken chart pipeline writes into SVG geometry. "-Infinity" is
# omitted because the substring match below already catches it via "Infinity".
_NON_FINITE_TOKENS = ("NaN", "Infinity", "undefined", "null")
_SVG_GEOMETRY_ATTRS = (
    "x",
    "y",
    "width",
    "height",
    "cx",
    "cy",
    "r",
    "transform",
    "d",
    "points",
    "viewBox",
)

# Header words that mark a column as a comparison against a baseline.
_DIFF_HEADER_TERMS = ("diff", "delta", "vs", "change", "improvement")


def check_poison_tokens(text: str) -> list[str]:
  """Flags values that indicate a broken adapter or missing proto field."""
  return [
      f"Rendered poison token {name!r}"
      for name, pat in POISON_PATTERNS.items()
      if re.search(pat, text)
  ]


def check_percentages(
    text: str, lo: float = 0.0, hi: float = 100.0
) -> list[str]:
  """Flags percentage values falling outside the valid range.

  Percentages outside [0, 100] on standard profile metrics usually indicate
  unnormalized rates or broken aggregation arithmetic.
  """
  return [
      f"Percentage {val}% outside [{lo}, {hi}]"
      for val in re.findall(r"(-?\d+(?:\.\d+)?)\s*%", text)
      if float(val) < lo or float(val) > hi
  ]


def check_durations_non_negative(text: str) -> list[str]:
  """Flags negative wall-clock durations.

  Negative wall-clock durations are always a bug in a profiler and indicate
  uncalibrated hardware timestamps or clock drift.
  """
  return [
      f"Negative duration {val}"
      for val in re.findall(r"(-\d+(?:\.\d+)?)\s*(?:ms|us|µs|ns|s)\b", text)
      if float(val) < 0.0
  ]


def check_no_layout_collapse(
    page: Page, selector: str, min_size: int = 8
) -> list[str]:
  """Flags elements that are visible yet occupy essentially no space."""
  violations = []
  for el in page.locator(selector).all():
    box = el.bounding_box()
    if box and (
        (0 < box["width"] < min_size) or (0 < box["height"] < min_size)
    ):
      violations.append(
          f"{selector} layout collapse:"
          f" {box['width']:.0f}x{box['height']:.0f}px"
      )
  return violations


def check_table_has_data_rows(page: Page, min_rows: int = 1) -> list[str]:
  """Flags data tables with headers but no data rows."""
  violations = []
  for i, table in enumerate(
      page.locator(
          "table:has(th, .mat-header-cell), mat-table:has(th, .mat-header-cell)"
      ).all()
  ):
    rows = table.locator("tr:has(td), mat-row, tr[mat-row]").count()
    if rows < min_rows:
      violations.append(
          f"Table[{i}] has headers but {rows} data row(s), expected >="
          f" {min_rows}"
      )
  return violations


def check_svg_geometry_deep(page: Page) -> list[str]:
  """Flags SVG geometry attributes that hold a non-finite value."""
  violations = []
  # Locators pierce Shadow DOM but not frame boundaries, so each frame has to
  # be queried on its own.
  for frame in page.frames:
    if frame.is_detached():
      continue
    name = frame.name or "main"
    frame_detached = False
    for attr in _SVG_GEOMETRY_ATTRS:
      if frame_detached or frame.is_detached():
        break
      for token in _NON_FINITE_TOKENS:
        try:
          elements = frame.locator(
              f"svg[{attr}*='{token}'], svg *[{attr}*='{token}']"
          ).all()
        except PlaywrightError:
          # Frame navigated or detached while querying locators.
          frame_detached = True
          break
        for element in elements:
          try:
            value = element.get_attribute(attr)
            if value is not None and token in value:
              violations.append(
                  f"Frame {name!r} SVG element [{attr}={value!r}] contains"
                  f" forbidden token {token!r}"
              )
          except PlaywrightError:
            # Element detached mid-inspection.
            continue
  return violations


def _diff_column_indices(table: Locator) -> set[int]:
  """Returns indices of columns whose header marks them as a comparison."""
  try:
    headers = table.locator(
        "thead tr:last-child th, mat-header-row:last-of-type mat-header-cell"
    ).all_inner_texts()
    if not headers:
      headers = table.locator("tr:first-child th").all_inner_texts()
  except PlaywrightError:
    return set()
  return {
      index
      for index, header in enumerate(headers)
      if any(term in header.lower() for term in _DIFF_HEADER_TERMS)
  }


def run_content_invariants(text: str) -> list[str]:
  """Runs text invariants safe to execute against the whole page."""
  return check_poison_tokens(text)


def run_cell_invariants(page: Page, max_cells: int = 4000) -> list[str]:
  """Runs numeric invariants scoped to table cells, skipping comparison columns.

  A column that reports a delta against a baseline legitimately holds negative
  durations and percentages far above 100, so the range checks are applied only
  to the remaining columns. Cells are read one row at a time so that a ragged
  row or a colspan cannot shift the column index of every row after it, and so
  that each row costs a single browser round trip. Poison tokens are not
  checked here because `run_content_invariants` already covers the whole page.
  """
  violations = []
  remaining = max_cells
  try:
    tables = page.locator("table, mat-table").all()
  except PlaywrightError:
    return violations
  for table in tables:
    if remaining <= 0:
      break
    diff_columns = _diff_column_indices(table)
    try:
      rows = table.locator("tbody tr, mat-row, tr[mat-row]").all()
    except PlaywrightError:
      continue
    for row in rows:
      if remaining <= 0:
        break
      try:
        texts = row.locator(
            "td, th, mat-cell, [mat-cell]"
        ).all_inner_texts()
      except PlaywrightError:
        continue
      texts = texts[:remaining]
      remaining -= len(texts)
      for column, text in enumerate(texts):
        if column in diff_columns:
          continue
        violations.extend(check_percentages(text))
        violations.extend(check_durations_non_negative(text))
  return violations


def run_dom_invariants(
    page: Page, collapse_selectors: list[str] | None = None
) -> list[str]:
  """Runs all DOM-shape invariants against the page."""
  violations = []
  for sel in collapse_selectors or []:
    violations.extend(check_no_layout_collapse(page, sel))
  violations.extend(check_table_has_data_rows(page))
  return violations


def format_violations(tool: str, violations: list[str], limit: int = 25) -> str:
  """Formats a violation list into a readable diagnostic message."""
  head = (
      f"{len(violations)} invariant violation(s) while rendering tool {tool!r}:"
  )
  shown = violations[:limit]
  tail = (
      f"\n  ... and {len(violations) - limit} more"
      if len(violations) > limit
      else ""
  )
  return head + "\n  - " + "\n  - ".join(shown) + tail


def assert_page_invariants(
    page: Page,
    collapse_selectors: list[str] | None = None,
    max_cells: int = 4000,
) -> None:
  """Asserts all content, DOM, SVG, and cell invariants in a single call."""
  violations = []
  text = page.inner_text("body")
  if text:
    violations.extend(run_content_invariants(text))
  violations.extend(run_dom_invariants(page, collapse_selectors))
  violations.extend(check_svg_geometry_deep(page))
  violations.extend(run_cell_invariants(page, max_cells))
  assert not violations, format_violations(page.url, violations)
