"""Generic, tool-agnostic invariants for surfacing data-pipeline bugs."""

# pylint: disable=g-doc-args,g-doc-return-or-yield,g-short-docstring-punctuation

import importlib
import re

try:
  _pw_sync = importlib.import_module("playwright.sync_api")
  PlaywrightError = _pw_sync.Error
  Locator = _pw_sync.Locator
  Page = _pw_sync.Page
except ImportError:

  class PlaywrightError(Exception):
    """Fallback error when playwright is not available in hermetic env."""

  class _DynamicStub:
    """Fallback stub when playwright is not available in hermetic env."""

    def __getattr__(self, name: str):
      return _DynamicStub()

    def __call__(self, *args, **kwargs):
      return _DynamicStub()

    def __iter__(self):
      return iter(())

  Locator = _DynamicStub
  Page = _DynamicStub

POISON_PATTERNS: dict[str, str] = {
    "NaN": r"\bNaN\b",
    "undefined": r"\bundefined\b",
    "[object Object]": r"\[object Object\]",
    "Infinity": r"-?\bInfinity\b",
    "null": r"(?<![\"'])\bnull\b(?![\"'])",
    "(null)": r"\(\s*null\s*\)",
    "INVALID": r"\bINVALID\b",
}

_NON_FINITE_TOKENS = ("NaN", "Infinity", "undefined", "null")
_SVG_GEOMETRY_ATTRS = (
    "x", "y", "width", "height", "cx", "cy", "r", "transform", "d",
    "points", "viewBox",
)

_DIFF_HEADER_RE = re.compile(
    r"(?:^|[^a-zA-Z0-9])(?:diff|delta|vs\.?|change|improvement)"
    r"(?:$|[^a-zA-Z0-9])",
    re.IGNORECASE,
)
_PERCENT_RE = re.compile(
    r"(?<![a-zA-Z0-9_.+-])(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*%"
)
_DURATION_RE = re.compile(
    r"(?<![a-zA-Z0-9_.+-])(-\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*"
    r"(?:ms|us|µs|ns|s)\b"
)


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
  violations = []
  for val in _PERCENT_RE.findall(text):
    try:
      num = float(val)
      if num < lo or num > hi:
        violations.append(f"Percentage {val}% outside [{lo}, {hi}]")
    except ValueError:
      continue
  return violations


def check_durations_non_negative(text: str) -> list[str]:
  """Flags negative wall-clock durations."""
  violations = []
  for val in _DURATION_RE.findall(text):
    try:
      if float(val) < 0.0:
        violations.append(f"Negative duration {val}")
    except ValueError:
      continue
  return violations


def check_no_layout_collapse(
    page: Page, selector: str, min_size: int = 8
) -> list[str]:
  """Flags elements that are visible yet occupy essentially no space."""
  violations = []
  try:
    for el in page.locator(selector).all():
      box = el.bounding_box()
      if box and (
          (0 < box["width"] < min_size) or (0 < box["height"] < min_size)
      ):
        violations.append(
            f"{selector} layout collapse:"
            f" {box['width']:.0f}x{box['height']:.0f}px"
        )
  except PlaywrightError:
    pass
  return violations


def check_table_has_data_rows(page: Page, min_rows: int = 1) -> list[str]:
  """Flags data tables with headers but no data rows."""
  violations = []
  try:
    tables = page.locator(
        "table:has(th, .mat-header-cell), mat-table:has(th, .mat-header-cell)"
    ).all()
    for i, table in enumerate(tables):
      rows = table.locator("tr:has(td), mat-row, tr[mat-row]").count()
      if rows < min_rows:
        violations.append(
            f"Table[{i}] has headers but {rows} data row(s), expected >="
            f" {min_rows}"
        )
  except PlaywrightError:
    pass
  return violations


def _build_svg_non_finite_selector() -> str:
  selectors = []
  for attr in _SVG_GEOMETRY_ATTRS:
    for tok in _NON_FINITE_TOKENS:
      selectors.append(f"svg[{attr}*='{tok}'], svg *[{attr}*='{tok}']")
  return ", ".join(selectors)


_SVG_NON_FINITE_SELECTOR = _build_svg_non_finite_selector()


def check_svg_geometry_deep(page: Page) -> list[str]:
  """Flags SVG geometry attributes that hold a non-finite value."""
  violations = []
  for frame in getattr(page, "frames", []):
    if hasattr(frame, "is_detached") and frame.is_detached():
      continue
    name = getattr(frame, "name", "") or "main"
    try:
      elements = frame.locator(_SVG_NON_FINITE_SELECTOR).all()
    except PlaywrightError:
      continue
    for el in elements:
      for attr in _SVG_GEOMETRY_ATTRS:
        try:
          val = el.get_attribute(attr)
        except PlaywrightError:
          break
        if val and any(tok in val for tok in _NON_FINITE_TOKENS):
          for tok in _NON_FINITE_TOKENS:
            if tok in val:
              violations.append(
                  f"Frame {name!r} SVG element [{attr}={val!r}] contains"
                  f" forbidden token {tok!r}"
              )
  return violations


def check_positive_rendered_content(page: Page) -> list[str]:
  """Verifies rendered charts have positive geometry and cards have text."""
  violations = []
  try:
    for i, chart in enumerate(page.locator("svg, canvas").all()):
      try:
        box = chart.bounding_box()
        if box and (box["width"] <= 0 or box["height"] <= 0):
          violations.append(
              f"Chart[{i}] collapsed geometry: {box['width']}x{box['height']}"
          )
      except PlaywrightError:
        continue
    for i, card in enumerate(
        page.locator("mat-card, .dashboard-card, .metric-card").all()
    ):
      try:
        if not (card.inner_text() or "").strip():
          violations.append(f"Card[{i}] is unexpectedly empty")
      except PlaywrightError:
        continue
  except PlaywrightError:
    pass
  return violations


def _diff_column_indices(table: Locator) -> set[int]:
  """Returns indices of columns whose header marks them as a comparison."""
  diff_indices: set[int] = set()
  try:
    locators = table.locator(
        "thead tr:last-child :is(th, .mat-header-cell, [mat-header-cell],"
        " .mat-mdc-header-cell), mat-header-row:last-of-type"
        " :is(mat-header-cell, .mat-header-cell, [mat-header-cell],"
        " .mat-mdc-header-cell)"
    ).all()
    if not locators:
      locators = table.locator(
          "tr:first-child :is(th, .mat-header-cell, [mat-header-cell],"
          " .mat-mdc-header-cell)"
      ).all()
    col = 0
    for th in locators:
      try:
        span = max(1, int(th.get_attribute("colspan") or "1"))
      except ValueError:
        span = 1
      if _DIFF_HEADER_RE.search(th.inner_text()):
        for offset in range(span):
          diff_indices.add(col + offset)
      col += span
  except PlaywrightError:
    return set()
  return diff_indices


def run_content_invariants(text: str) -> list[str]:
  """Runs text invariants safe to execute against the whole page."""
  return check_poison_tokens(text)


def run_cell_invariants(page: Page, max_cells: int = 4000) -> list[str]:
  """Runs numeric invariants scoped to table cells, skipping diff columns."""
  violations, remaining = [], max_cells
  try:
    tables = page.locator("table, mat-table").all()
  except PlaywrightError:
    return violations
  for table in tables:
    if remaining <= 0:
      break
    diff_cols = _diff_column_indices(table)
    try:
      rows = table.locator(
          "tr:has(td), mat-row, .mat-row, tr[mat-row], .mat-mdc-row"
      ).all()
    except PlaywrightError:
      continue
    for row in rows:
      if remaining <= 0:
        break
      try:
        cells = row.locator(
            "td, th, mat-cell, [mat-cell], .mat-cell, .mat-mdc-cell"
        ).all()
      except PlaywrightError:
        continue
      col = 0
      for cell in cells:
        if remaining <= 0:
          break
        remaining -= 1
        try:
          span = max(1, int(cell.get_attribute("colspan") or "1"))
          text = cell.inner_text()
        except (PlaywrightError, ValueError):
          col += 1
          continue
        is_diff = any((col + offset) in diff_cols for offset in range(span))
        col += span
        if not is_diff:
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
  violations.extend(check_positive_rendered_content(page))
  return violations


def format_violations(
    tool: str, violations: list[str], limit: int = 25
) -> str:
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
