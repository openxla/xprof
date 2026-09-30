"""Generic, tool-agnostic invariants for surfacing data-pipeline bugs."""

# pylint: disable=g-doc-args,g-doc-return-or-yield,g-short-docstring-punctuation

import importlib
import re

try:
  sync_api = importlib.import_module("playwright.sync_api")
  PlaywrightError = sync_api.Error
except ImportError:

  class PlaywrightError(Exception):
    """Fallback error when playwright is absent in hermetic unit tests."""

  class _DynamicStub:
    """Fallback stub when playwright is absent in hermetic unit tests."""

    def __getattr__(self, name: str):
      del name
      return _DynamicStub()

    def __call__(self, *args, **kwargs):
      del args, kwargs
      return _DynamicStub()

  _DynamicStub.Page = _DynamicStub
  sync_api = _DynamicStub()

Page = sync_api.Page

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
_DIFF_HEADER_RE = re.compile(
    r"(?:^|[^a-zA-Z0-9])(?:diff|delta|vs\.?|change|improvement)"
    r"(?:$|[^a-zA-Z0-9])",
    re.IGNORECASE,
)
# Roofline Model columns that divide a measured rate by the hardware limit.
# They can exceed 100%: the backend notes it for asynchronous copies, and custom
# calls such as Pallas kernels report their own cost estimates, which can
# overstate the work.
_HW_LIMIT_HEADER_RE = re.compile(
    r"roofline efficiency|flop rate / peak|max memory bw utilization",
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
  """Flags percentage values falling outside the valid range.

  Percentages outside [0, 100] on standard profile metrics usually indicate
  unnormalized rates or broken aggregation arithmetic.
  """
  violations = []
  for val in _PERCENT_RE.findall(text):
    try:
      num = float(val)
    except ValueError:
      continue
    if num < lo or num > hi:
      violations.append(f"Percentage {val}% outside [{lo}, {hi}]")
  return violations


def check_durations_non_negative(text: str) -> list[str]:
  """Flags negative wall-clock durations.

  Negative wall-clock durations are always a bug in a profiler and indicate
  uncalibrated hardware timestamps or clock drift.
  """
  violations = []
  for val in _DURATION_RE.findall(text):
    try:
      num = float(val)
    except ValueError:
      continue
    if num < 0.0:
      violations.append(f"Negative duration {val}")
  return violations


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
  # Excludes the off-screen copy of each chart's data that Google Charts keeps
  # for screen readers (a table inside an aria-labelled div): an empty chart,
  # such as outside compilation on a JAX profile, leaves it with headers only.
  for i, table in enumerate(
      page.locator(
          "table:has(th, .mat-header-cell):not([aria-label] > table),"
          " mat-table:has(th, .mat-header-cell)"
      ).all()
  ):
    rows = table.locator("tr:has(td), mat-row, tr[mat-row]").count()
    if rows < min_rows:
      violations.append(
          f"Table[{i}] has headers but {rows} data row(s), expected >="
          f" {min_rows}"
      )
  return violations


def _build_svg_non_finite_selector() -> str:
  """Builds union CSS selector for SVG elements with non-finite attributes."""
  selectors = []
  for attr in _SVG_GEOMETRY_ATTRS:
    for token in _NON_FINITE_TOKENS:
      selectors.append(f"svg[{attr}*='{token}'], svg *[{attr}*='{token}']")
  return ", ".join(selectors)


_SVG_NON_FINITE_SELECTOR = _build_svg_non_finite_selector()

# Collects the text of every non-comparison table cell in a single round trip.
# Reading the cells through locators costs two round trips per cell, which is
# minutes on the larger tool tables. Like a locator, it also searches open
# shadow roots, which is where the Trace Viewer page renders its help table.
_EXTRACT_NON_DIFF_CELLS_JS = """
([maxCells, diffPattern]) => {
  const diffRe = new RegExp(diffPattern, 'i');
  const out = [];
  const roots = [document];
  for (let i = 0; i < roots.length; i++) {
    for (const el of roots[i].querySelectorAll('*')) {
      if (el.shadowRoot) roots.push(el.shadowRoot);
    }
  }
  const tables = roots.flatMap(r => [...r.querySelectorAll('table, mat-table')]);
  for (const table of tables) {
    if (out.length >= maxCells) break;
    let headers = table.querySelectorAll(
      'thead tr:last-child :is(th, .mat-header-cell, [mat-header-cell], .mat-mdc-header-cell), ' +
      'mat-header-row:last-of-type :is(mat-header-cell, .mat-header-cell, [mat-header-cell], .mat-mdc-header-cell)'
    );
    if (!headers.length) {
      headers = table.querySelectorAll(
        'tr:first-child :is(th, .mat-header-cell, [mat-header-cell], .mat-mdc-header-cell)'
      );
    }
    const diffCols = new Set();
    let col = 0;
    for (const th of headers) {
      // colSpan is 1 when colspan is absent or invalid, and at most 1000.
      const span = th.colSpan || 1;
      if (diffRe.test(th.innerText || '')) {
        for (let k = 0; k < span; k++) diffCols.add(col + k);
      }
      col += span;
    }
    const rows = table.querySelectorAll('tr, mat-row, .mat-row, [mat-row], .mat-mdc-row');
    for (const row of rows) {
      if (out.length >= maxCells) break;
      if (row.tagName === 'TR' && !row.querySelector('td') && !row.hasAttribute('mat-row')) continue;
      const cells = row.querySelectorAll('td, th, mat-cell, [mat-cell], .mat-cell, .mat-mdc-cell');
      let c = 0;
      for (const cell of cells) {
        if (out.length >= maxCells) break;
        const span = cell.colSpan || 1;
        let isDiff = false;
        for (let k = 0; k < span; k++) {
          if (diffCols.has(c + k)) { isDiff = true; break; }
        }
        c += span;
        if (!isDiff) out.push(cell.innerText || '');
      }
    }
  }
  return out;
}
"""


def check_svg_geometry_deep(page: Page) -> list[str]:
  """Flags SVG geometry attributes that hold a non-finite value."""
  violations = []
  # Locators pierce Shadow DOM but not frame boundaries, so each frame has to
  # be queried on its own.
  for frame in page.frames:
    if frame.is_detached():
      continue
    name = frame.name or "main"
    try:
      elements = frame.locator(_SVG_NON_FINITE_SELECTOR).all()
    except PlaywrightError:
      continue
    for element in elements:
      for attr in _SVG_GEOMETRY_ATTRS:
        try:
          value = element.get_attribute(attr)
        except PlaywrightError:
          break
        if value is None:
          continue
        for token in _NON_FINITE_TOKENS:
          if token in value:
            violations.append(
                f"Frame {name!r} SVG element [{attr}={value!r}] contains"
                f" forbidden token {token!r}"
            )
  return violations


def run_content_invariants(text: str) -> list[str]:
  """Runs text invariants safe to execute against the whole page."""
  return check_poison_tokens(text)


def run_cell_invariants(page: Page, max_cells: int = 4000) -> list[str]:
  """Runs numeric invariants scoped to table cells, skipping unbounded columns.

  A column that reports a delta against a baseline legitimately holds negative
  durations and percentages far above 100, and a Roofline Model ratio to the
  hardware limit can exceed 100%, so the range checks are applied only to the
  remaining columns. Cells are read one row at a time so that a ragged row or a
  colspan cannot shift the column index of every row after it. Poison tokens
  are not checked here because `run_content_invariants` already covers the
  whole page.
  """
  violations = []
  for text in page.evaluate(
      _EXTRACT_NON_DIFF_CELLS_JS,
      [max_cells, f"{_DIFF_HEADER_RE.pattern}|{_HW_LIMIT_HEADER_RE.pattern}"],
  ):
    violations.extend(check_percentages(text))
    violations.extend(check_durations_non_negative(text))
  return violations


def check_positive_rendered_content(page: Page) -> list[str]:
  """Verifies rendered charts have positive geometry and cards have text."""
  violations = []
  for i, chart in enumerate(page.locator("svg, canvas").all()):
    try:
      box = chart.bounding_box()
    except PlaywrightError:
      continue
    if box and (box["width"] <= 0 or box["height"] <= 0):
      violations.append(
          f"Chart[{i}] collapsed geometry: {box['width']}x{box['height']}"
      )
  for i, card in enumerate(
      page.locator("mat-card, .dashboard-card, .metric-card").all()
  ):
    try:
      text = card.inner_text()
    except PlaywrightError:
      continue
    if not text.strip():
      violations.append(f"Card[{i}] is unexpectedly empty")
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
