"""Generic, tool-agnostic invariants for surfacing data-pipeline bugs."""

# pylint: disable=g-doc-args,g-doc-return-or-yield,g-short-docstring-punctuation

import re
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
  """Flags percentage values falling outside the valid range."""
  return [
      f"Percentage {val}% outside [{lo}, {hi}]"
      for val in re.findall(r"(-?\d+(?:\.\d+)?)\s*%", text)
      if float(val) < lo or float(val) > hi
  ]


def check_durations_non_negative(text: str) -> list[str]:
  """Flags negative wall-clock durations."""
  return [
      f"Negative duration {val}"
      for val in re.findall(r"(-\d+(?:\.\d+)?)\s*(?:ms|us|µs|ns|s)\b", text)
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
  """Detects non-finite attributes in SVG elements across page frames in pure Python."""
  violations = []
  tokens = ("NaN", "Infinity", "-Infinity", "undefined", "null")
  attributes = ("y", "height", "width", "x", "transform", "cx", "cy", "r")

  # Playwright locators penetrate Shadow DOM by default
  frames = page.frames if hasattr(page, "frames") and page.frames else [page]
  for frame in frames:
    for attr in attributes:
      for token in tokens:
        loc = frame.locator(f"svg [{attr}*='{token}']")
        count = loc.count()
        if count > 0:
          for el in loc.all():
            val = el.get_attribute(attr) or ""
            frame_name = getattr(frame, "name", "") or "main"
            violations.append(
                f"Frame '{frame_name}' SVG element [{attr}='{val}'] contains"
                f" forbidden token '{token}'"
            )
  return violations


def run_cell_invariants_optimized(
    page: Page, max_cells: int = 2000
) -> list[str]:
  """Evaluates table cells with header awareness and duration bounds in pure Python."""
  violations = []
  tables = page.locator("table, mat-table").all()
  for table in tables:
    # Identify diff/delta comparison columns from leaf headers
    header_cells = table.locator(
        "thead tr:last-child th, mat-header-row:last-of-type mat-header-cell"
    ).all()
    diff_cols: set[int] = set()
    for idx, th in enumerate(header_cells):
      txt = th.inner_text().lower()
      if any(
          term in txt
          for term in ("diff", "delta", "vs", "change", "improvement")
      ):
        diff_cols.add(idx)

    cells = table.locator("tbody td, mat-row mat-cell").all()[:max_cells]
    col_count = len(header_cells) or 1
    for idx, cell in enumerate(cells):
      text = cell.inner_text().strip()
      if not text:
        continue
      col_idx = idx % col_count
      is_diff = col_idx in diff_cols

      violations.extend(check_poison_tokens(text))
      if not is_diff:
        violations.extend(check_durations_non_negative(text))
        violations.extend(check_percentages(text, lo=0.0, hi=1000.0))

  return violations


def run_content_invariants(text: str) -> list[str]:
  """Text invariants safe to run against the whole page."""
  return check_poison_tokens(text)


def run_cell_invariants(page: Page, max_cells: int = 4000) -> list[str]:
  """Numeric invariants scoped to table cells evaluated via browser DOM."""
  texts = page.locator("td, .mat-cell, [mat-cell]").all_inner_texts()[
      :max_cells
  ]
  violations = []
  for t in texts:
    if "%" in t or re.search(r"\b(?:ms|us|µs|ns|s)\b", t):
      violations.extend(check_percentages(t))
      violations.extend(check_durations_non_negative(t))
  return violations


def run_dom_invariants(
    page: Page, collapse_selectors: list[str] | None = None
) -> list[str]:
  """All DOM-shape invariants."""
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
  violations.extend(run_cell_invariants_optimized(page, max_cells))
  assert not violations, format_violations(page.url, violations)
