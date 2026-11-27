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
"""Generic, tool-agnostic invariants for surfacing data-pipeline bugs."""

# pylint: disable=g-doc-args,g-doc-return-or-yield,g-short-docstring-punctuation

from collections.abc import Sequence
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
          "table:has(th, .mat-header-cell, .mat-mdc-header-cell)"
          ":not([aria-label] > table),"
          " mat-table:has(th, .mat-header-cell, .mat-mdc-header-cell)"
      ).all()
  ):
    rows = table.locator(
        "tr:has(td), mat-row, tr[mat-row], .mat-mdc-row"
    ).count()
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
  for i, chart in enumerate(
      page.locator(
          "svg:not([focusable='false'], [aria-hidden='true']), canvas"
      ).all()
  ):
    try:
      box = chart.bounding_box()
      if (
          box
          and (box["width"] <= 0 or box["height"] <= 0)
          and chart.evaluate("el => getComputedStyle(el).visibility")
          != "hidden"
      ):
        violations.append(
            f"Chart[{i}] collapsed geometry: {box['width']}x{box['height']}"
        )
    except PlaywrightError:
      continue
  for i, card in enumerate(
      page.locator(
          "mat-card, .mat-mdc-card, .dashboard-card, .metric-card"
      ).all()
  ):
    try:
      text = card.inner_text()
    except PlaywrightError:
      continue
    if not text.strip():
      violations.append(f"Card[{i}] is unexpectedly empty")
  return violations


def check_no_error_banners(page: Page) -> list[str]:
  """Flags visible diagnostics, snackbar, or Google Visualization errors."""
  violations = []
  selector = (
      "diagnostics-view .callout.is-critical:visible,"
      " mat-snack-bar-container:visible, .mat-mdc-snack-bar-container:visible,"
      " [id^='google-visualization-errors-all']:visible"
  )
  for i, banner in enumerate(page.locator(selector).all()):
    try:
      text = banner.inner_text().strip()
    except PlaywrightError:
      continue
    violations.append(f"Visible error banner[{i}]: {text!r}")
  return violations


_CHECK_SIDENAV_OVERLAP_JS = """
() => {
  const out = [];
  const containers = document.querySelectorAll('mat-sidenav-container');
  for (const container of containers) {
    const sidenav = container.querySelector(
        ':scope > mat-sidenav.mat-drawer-side.mat-drawer-opened' +
        ':not(.mat-drawer-animating)'
    );
    const content = container.querySelector(':scope > mat-sidenav-content');
    if (!sidenav || !content) continue;
    const sRect = sidenav.getBoundingClientRect();
    const cRect = content.getBoundingClientRect();
    if (sRect.width <= 0 || cRect.width <= 0) continue;
    const overlap = sRect.right - cRect.left;
    if (overlap > 2) {
      out.push(
          'Open side drawer overlaps mat-sidenav-content by ' +
          `${overlap.toFixed(0)}px`
      );
    }
  }
  return out;
}
"""

# Upstream 1280px unwrapped flex chart/filter rows. `framework-op-stats` is
# tracked by a strict xfail test in
# `test_tools_rendering.py::test_desktop_chart_rows_fit_content_pane`.
# `perf-counters` overflows at 1280px only in internal builds (Copybara strips
# the extra `Host` and `Show Zero Values` controls in OSS).
KNOWN_UNWRAPPED_ROW_SELECTORS: tuple[str, ...] = (
    "framework-op-stats .row",
    "perf-counters .row",
)

_SCROLLABLE_TABLE_SELECTORS: tuple[str, ...] = (
    "table",
    "mat-table",
    ".google-visualization-table",
)

_CHECK_HORIZONTAL_OVERFLOW_JS = """
([maxOverflow, checkContentPane, allowedSelector]) => {
  const out = [];
  const doc = document.documentElement;
  const vw = doc.clientWidth || window.innerWidth;
  const docOverflow = doc.scrollWidth - vw;
  if (docOverflow > maxOverflow) {
    out.push(
        `Document horizontal overflow: ${docOverflow.toFixed(0)}px > ` +
        `${maxOverflow}px`
    );
  }
  if (checkContentPane) {
    const pane = document.querySelector('mat-sidenav-content');
    if (pane && pane.scrollWidth - pane.clientWidth > maxOverflow) {
      const paneLeft = pane.getBoundingClientRect().left;
      const rightEdge = paneLeft + pane.clientLeft + pane.clientWidth;
      const scrollLeft = pane.scrollLeft || 0;
      const overflowing = [...pane.querySelectorAll('*')].filter(el => {
        const r = el.getBoundingClientRect();
        if (r.width <= 0 || r.height <= 0) return false;
        if (r.right + scrollLeft - rightEdge <= maxOverflow) return false;
        for (let p = el.parentElement; p && p !== pane; p = p.parentElement) {
          if (getComputedStyle(p).overflowX !== 'visible') {
            const pr = p.getBoundingClientRect();
            const pRight = pr.left + p.clientLeft + p.clientWidth;
            if (pRight + scrollLeft - rightEdge <= maxOverflow) return false;
          }
        }
        return true;
      });
      const leaves = overflowing.filter(
          el => !overflowing.some(o => o !== el && el.contains(o))
      );
      if (
          !leaves.length ||
          !allowedSelector ||
          !leaves.every(el => Boolean(el.closest(allowedSelector)))
      ) {
        const delta = pane.scrollWidth - pane.clientWidth;
        out.push(
            'mat-sidenav-content horizontal overflow: ' +
            `${delta.toFixed(0)}px > ${maxOverflow}px`
        );
      }
    }
  }
  return out;
}
"""


def _is_transient_eval_error(err: Exception) -> bool:
  """Returns whether Playwright eval failed due to navigation.

  Args:
    err: Exception raised during `page.evaluate`.

  Returns:
    True when the message indicates frame detachment or navigation.
  """
  msg = str(err).lower()
  return any(
      token in msg
      for token in (
          "execution context was destroyed",
          "frame was detached",
          "navigat",
      )
  )


def _eval_layout_check(
    page: Page,
    js_fn: str,
    args: object = None,
    settle_timeout_ms: int = 600,
) -> list[str]:
  """Evaluates a layout-check JS function with a brief settle retry."""

  def _run() -> list[str]:
    return list(page.evaluate(js_fn, args))

  try:
    if not _run():
      return []
    try:
      page.wait_for_function(
          f"a => ({js_fn})(a).length === 0",
          arg=args,
          timeout=settle_timeout_ms,
      )
      return []
    except PlaywrightError:
      return _run()
  except PlaywrightError as err:
    if _is_transient_eval_error(err):
      return []
    raise


def check_no_sidenav_overlap(page: Page) -> list[str]:
  """Flags an open side-mode navigation drawer overlapping main tool content.

  Args:
    page: Playwright page under test.

  Returns:
    List of overlap violation descriptions, empty when clear or navigating.
  """
  return _eval_layout_check(page, _CHECK_SIDENAV_OVERLAP_JS)


def check_no_horizontal_overflow(
    page: Page,
    max_overflow_px: int = 16,
    check_content_pane: bool = True,
    allowed_overflow_selectors: Sequence[str] = (),
) -> list[str]:
  """Flags horizontal overflow beyond the viewport or sidenav content pane.

  Args:
    page: Playwright page under test.
    max_overflow_px: Allowed pixel tolerance before reporting overflow.
    check_content_pane: Whether to also inspect `mat-sidenav-content` overflow.
    allowed_overflow_selectors: Selectors whose overflowing descendants are
      tolerated in addition to scrollable tables.

  Returns:
    List of overflow violation descriptions, empty when within bounds.
  """
  allowed = ", ".join(
      (*_SCROLLABLE_TABLE_SELECTORS, *allowed_overflow_selectors)
  )
  args = [max_overflow_px, check_content_pane, allowed]
  return _eval_layout_check(page, _CHECK_HORIZONTAL_OVERFLOW_JS, args=args)


def check_no_stuck_loading_lock(page: Page) -> list[str]:
  """Flags a router-outlet stuck inside a 0px hidden-content loading wrapper."""
  violations = []
  for _ in page.locator(
      "main-page mat-sidenav-content > div.hidden-content"
  ).all():
    violations.append(
        "Main page router-outlet is locked inside div.hidden-content (0px"
        " height)"
    )
  return violations


def run_dom_invariants(
    page: Page,
    collapse_selectors: list[str] | None = None,
    allowed_overflow_selectors: Sequence[str] | None = None,
) -> list[str]:
  """Runs all DOM-shape invariants against the page."""
  overflow_allow = (
      KNOWN_UNWRAPPED_ROW_SELECTORS
      if allowed_overflow_selectors is None
      else allowed_overflow_selectors
  )
  violations = []
  for sel in collapse_selectors or []:
    violations.extend(check_no_layout_collapse(page, sel))
  violations.extend(check_table_has_data_rows(page))
  violations.extend(check_positive_rendered_content(page))
  violations.extend(check_no_error_banners(page))
  violations.extend(check_no_stuck_loading_lock(page))
  violations.extend(check_no_sidenav_overlap(page))
  violations.extend(
      check_no_horizontal_overflow(
          page,
          check_content_pane=True,
          allowed_overflow_selectors=overflow_allow,
      )
  )
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
    allowed_overflow_selectors: Sequence[str] | None = None,
) -> None:
  """Asserts all content, DOM, SVG, and cell invariants in a single call."""
  violations = []
  text = page.inner_text("body")
  if text:
    violations.extend(run_content_invariants(text))
  violations.extend(
      run_dom_invariants(page, collapse_selectors, allowed_overflow_selectors)
  )
  violations.extend(check_svg_geometry_deep(page))
  violations.extend(run_cell_invariants(page, max_cells))
  assert not violations, format_violations(page.url, violations)
