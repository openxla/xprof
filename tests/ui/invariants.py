"""Generic, tool-agnostic invariants for surfacing data-pipeline bugs."""

import re
from typing import List, Optional
from playwright.sync_api import Page

POISON_PATTERNS = {
    "NaN": r"\bNaN\b",
    "undefined": r"\bundefined\b",
    "[object Object]": r"\[object Object\]",
    "Infinity": r"-?\bInfinity\b",
    "null": r"(?<![\"'])\bnull\b(?![\"'])",
    "(null)": r"\(\s*null\s*\)",
}


def check_poison_tokens(text: str) -> List[str]:
  """Flags values that indicate a broken adapter or missing proto field."""
  return [
      f"Rendered poison token {name!r}"
      for name, pat in POISON_PATTERNS.items()
      if re.search(pat, text)
  ]


def check_percentages(
    text: str, lo: float = 0.0, hi: float = 100.0
) -> List[str]:
  """Percentages outside [0, 100] mean a bad ratio or double-scaling bug."""
  return [
      f"Percentage {val}% outside [{lo}, {hi}]"
      for val in re.findall(r"(-?\d+(?:\.\d+)?)\s*%", text)
      if float(val) < lo or float(val) > hi
  ]


def check_durations_non_negative(text: str) -> List[str]:
  """Negative wall-clock durations are always a bug in a profiler."""
  return [
      f"Negative duration {val}"
      for val in re.findall(r"(-\d+(?:\.\d+)?)\s*(?:ms|us|µs|ns|s)\b", text)
  ]


def check_no_layout_collapse(
    page: Page, selector: str, min_size: int = 8
) -> List[str]:
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


def check_table_has_data_rows(page: Page, min_rows: int = 1) -> List[str]:
  """Flags data tables with headers but no data rows."""
  violations = []
  for i, table in enumerate(
      page.locator("table:has(th, .mat-header-cell)").all()
  ):
    rows = table.locator("tr:has(td), mat-row, tr[mat-row]").count()
    if rows < min_rows:
      violations.append(
          f"Table[{i}] has headers but {rows} data row(s), expected >="
          f" {min_rows}"
      )
  return violations


def run_content_invariants(text: str) -> List[str]:
  """Text invariants safe to run against the whole page."""
  return check_poison_tokens(text)


def run_cell_invariants(page: Page, max_cells: int = 4000) -> List[str]:
  """Numeric invariants scoped to table cells evaluated via browser DOM."""
  texts = page.locator("td, .mat-cell, [mat-cell]").all_inner_texts()[
      :max_cells
  ]
  combined = " ".join(
      t for t in texts if "%" in t or re.search(r"\b(?:ms|us|µs|ns|s)\b", t)
  )
  return check_percentages(combined) + check_durations_non_negative(combined)


def run_dom_invariants(
    page: Page, collapse_selectors: Optional[List[str]] = None
) -> List[str]:
  """All DOM-shape invariants."""
  violations = []
  for sel in collapse_selectors or []:
    violations.extend(check_no_layout_collapse(page, sel))
  violations.extend(check_table_has_data_rows(page))
  return violations


def format_violations(tool: str, violations: List[str], limit: int = 25) -> str:
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
    collapse_selectors: Optional[List[str]] = None,
    max_cells: int = 4000,
):
  """Asserts all content, DOM, and cell invariants in a single call."""
  violations = []
  text = page.inner_text("body")
  if text:
    violations.extend(run_content_invariants(text))
  violations.extend(run_dom_invariants(page, collapse_selectors))
  violations.extend(run_cell_invariants(page, max_cells))
  assert not violations, format_violations(page.url, violations)
