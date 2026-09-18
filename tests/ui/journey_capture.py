"""Stabilized screenshot, DOM, and network capture for XProf UI journeys.

The SxS diff engine compares two renders of the same waypoint. Anything that
varies between two runs of an unchanged build -- animation frames, the text
caret, the server port -- becomes a false divergence, so every capture here is
normalized before it reaches the engine.
"""

from __future__ import annotations

import dataclasses
import importlib
import logging
import re
import time
import typing
import urllib.parse

sync_api: typing.Any = None
try:
  sync_api = importlib.import_module("playwright.sync_api")
  _PlaywrightError = sync_api.Error
  _PlaywrightTimeoutError = sync_api.TimeoutError
except ImportError:  # pragma: no cover - hermetic unit test fallback
  _PlaywrightError = RuntimeError
  _PlaywrightTimeoutError = TimeoutError

# Playwright waits this long for the network to fall idle before a capture.
# Exceeding it is not an error: XProf tools poll, so some pages never reach
# idle, and the quiescence wait below is what actually stabilizes those.
# Bound to 1000ms so polling tools do not incur multi-second idle stalls.
_NETWORK_IDLE_TIMEOUT_MS: int = 1000

# In-flight API requests (/data/plugin/profile/*) must hold at zero for this
# window before NetworkRecorder drains, preventing responses from crossing a
# waypoint boundary.
_API_QUIESCENCE_WINDOW_MS: int = 250
_API_QUIESCENCE_TIMEOUT_MS: int = 5000
_API_POLL_MS: int = 50

# The DOM is considered settled once two consecutive samples this far apart are
# identical. Google Charts redraws when its container resizes, and the redraw
# lands after the network is already quiet, so a capture taken on a fixed delay
# catches a chart mid-layout and records different gridline geometry on each
# walk.
_QUIESCENCE_POLL_MS: int = 250
_QUIESCENCE_TIMEOUT_MS: int = 8000

# Regions that legitimately differ between two runs of the same build. Masking
# paints them a flat color in both screenshots and clears their text in the
# serialized DOM so the engine sees them as equal across runs.
MASK_SELECTORS: tuple[str, ...] = (
    # Absolute log directory, which embeds a per-run temporary path.
    "sidenav .session-path",
    # Trace Viewer rotates tutorial tips on a 3s timer while loading.
    ".tutorial",
)

# Flat fill painted over masked regions. Chosen to be absent from the XProf
# palette so a mask is obvious when a reviewer opens the report.
_MASK_COLOR = "#ff00ff"

_SESSION_PATH_RE = re.compile(r"(?<=\bsession_path=)[^&\"'\s<>]*")
_LOCALHOST_ORIGIN_RE = re.compile(r"https?://(?:localhost|127\.0\.0\.1):[0-9]+")


@dataclasses.dataclass(frozen=True)
class WaypointCapture:
  """Multi-modal snapshot of a single journey waypoint."""

  name: str
  png_bytes: bytes
  html: str
  requests: list[dict[str, object]]
  stabilized: bool = True
  unsettled_reasons: tuple[str, ...] = ()


def _relative_url(url: str) -> str:
  """Strips scheme/authority and masks per-run session_path parameters.

  Control and candidate run on different ports and may serve distinct temporary
  logdir paths, so unnormalized URLs would differ for every request and report
  the whole waterfall as diverged.

  Args:
    url: Absolute or relative request URL.

  Returns:
    The normalized path and query components, without scheme, host, or port.
  """
  parts = urllib.parse.urlsplit(url)
  query = _SESSION_PATH_RE.sub("<masked>", parts.query)
  return urllib.parse.urlunsplit(("", "", parts.path, query, ""))


class NetworkRecorder:
  """Records completed XProf API responses issued by a page."""

  def __init__(self, page: typing.Any):
    self._entries: list[dict[str, object]] = []
    self._inflight: int = 0
    page.on("request", self._on_request)
    page.on("requestfinished", self._on_request_done)
    page.on("requestfailed", self._on_request_done)
    page.on("response", self._on_response)

  def _is_tracked_api_url(self, url: str) -> bool:
    path = urllib.parse.urlsplit(url).path
    return "/data/plugin/profile/" in path

  def _on_request(self, request: typing.Any) -> None:
    if self._is_tracked_api_url(getattr(request, "url", "")):
      self._inflight += 1

  def _on_request_done(self, request: typing.Any) -> None:
    if self._is_tracked_api_url(getattr(request, "url", "")):
      self._inflight = max(0, self._inflight - 1)

  def _on_response(self, response: typing.Any) -> None:
    if not self._is_tracked_api_url(response.url):
      return
    self._entries.append({
        "method": response.request.method,
        "url": _relative_url(response.url),
        "status": response.status,
    })

  def wait_for_quiescence(
      self,
      page: typing.Any,
      window_ms: int = _API_QUIESCENCE_WINDOW_MS,
      timeout_ms: int = _API_QUIESCENCE_TIMEOUT_MS,
  ) -> bool:
    """Waits until in-flight API requests remain at zero for window_ms."""
    if not hasattr(page, "wait_for_timeout"):
      return True
    deadline = time.monotonic() + timeout_ms / 1000
    quiet_since: float | None = None
    while time.monotonic() < deadline:
      if self._inflight == 0:
        now = time.monotonic()
        if quiet_since is None:
          quiet_since = now
        if (now - quiet_since) * 1000 >= window_ms:
          return True
      else:
        quiet_since = None
      page.wait_for_timeout(_API_POLL_MS)
    logging.warning(
        "API network did not quiesce within %dms (%d requests in flight).",
        timeout_ms,
        self._inflight,
    )
    return False

  def drain(self) -> list[dict[str, object]]:
    """Returns deduplicated API responses recorded so far and clears buffer."""
    recorded = self._entries
    self._entries = []
    deduped: list[dict[str, object]] = []
    seen_gets: set[tuple[str, str, object]] = set()
    for entry in recorded:
      method = str(entry.get("method") or "GET").upper()
      sig = (method, str(entry.get("url") or ""), entry.get("status"))
      if method == "GET":
        if sig in seen_gets:
          continue
        seen_gets.add(sig)
      deduped.append(entry)
    return deduped


# Serializes the document with every element's attributes in sorted order.
#
# Google Charts emits attributes in a different order between renders, so two
# structurally identical SVGs serialize to different strings. Sorting makes the
# comparison order-insensitive without weakening it: an added, removed or
# changed attribute still shows up.
#
# The reordering runs on a detached clone. Rewriting attributes on the live
# document would invalidate layout and could trigger a chart redraw between the
# DOM snapshot and the screenshot.
_NORMALIZED_HTML_JS = """(maskSelectors) => {
  const clone = document.documentElement.cloneNode(true);
  for (const s of clone.querySelectorAll('style, link[rel="stylesheet"]')) {
    s.remove();
  }
  for (const el of clone.querySelectorAll(
      '.cdk-live-announcer-element, .cdk-describedby-message-container, ' +
      '.cdk-overlay-container, div[style*="display: none"], ' +
      'div[style*="display:none"]'
  )) {
    el.remove();
  }
  if (Array.isArray(maskSelectors) && maskSelectors.length) {
    for (const sel of maskSelectors) {
      for (const el of clone.querySelectorAll(sel)) {
        el.textContent = '';
      }
    }
  }
  const stack = [clone];
  while (stack.length) {
    const el = stack.pop();
    const attrs = Array.from(el.attributes)
        .map(a => [a.name, a.value])
        .sort((x, y) => x[0] < y[0] ? -1 : x[0] > y[0] ? 1 : 0);
    for (const [name] of attrs) el.removeAttribute(name);
    for (const [name, value] of attrs) {
      const cleanedVal = value
          .replace(/https?:\\/\\/(?:localhost|127\\.0\\.0\\.1):[0-9]+/g, '')
          .replace(/(?<=\\bsession_path=)[^&"'\\s<>]*/g, '<masked>');
      el.setAttribute(name, cleanedVal);
    }
    for (const child of el.children) stack.push(child);
  }
  return clone.outerHTML;
}"""


def _normalized_html(page: typing.Any) -> str:
  """Returns the document serialized with attributes in a stable order."""
  raw = page.evaluate(_NORMALIZED_HTML_JS, list(MASK_SELECTORS))
  return _SESSION_PATH_RE.sub("<masked>", _LOCALHOST_ORIGIN_RE.sub("", raw))


def _wait_for_dom_quiescence(page: typing.Any) -> tuple[str, bool]:
  """Polls until the serialized DOM stops changing, returning (html, settled).

  Args:
    page: Page to sample.

  Returns:
    A tuple of (last_serialized_html, is_settled).
  """
  try:
    page.evaluate("document.fonts.ready")
  except _PlaywrightError:
    pass
  deadline = time.monotonic() + _QUIESCENCE_TIMEOUT_MS / 1000
  previous = _normalized_html(page)
  stable_count = 0
  while time.monotonic() < deadline:
    page.wait_for_timeout(_QUIESCENCE_POLL_MS)
    current = _normalized_html(page)
    if current == previous:
      stable_count += 1
      if stable_count >= 2:
        return current, True
    else:
      stable_count = 0
    previous = current
  logging.warning(
      "DOM did not quiesce within %dms; capture may be unstable.",
      _QUIESCENCE_TIMEOUT_MS,
  )
  return previous, False


def capture_waypoint(
    page: typing.Any, recorder: NetworkRecorder, name: str
) -> WaypointCapture:
  """Captures a stabilized screenshot, serialized DOM, and network log.

  Args:
    page: Page already navigated to the waypoint under capture.
    recorder: Recorder attached to the same page.
    name: Waypoint identifier used in the report.

  Returns:
    The captured waypoint state.
  """
  unsettled: list[str] = []

  # 1. Wait for in-flight tool data and layout network requests to settle.
  try:
    page.wait_for_load_state("networkidle", timeout=_NETWORK_IDLE_TIMEOUT_MS)
  except _PlaywrightTimeoutError:
    # Polling tools never reach idle. Quiescence below still applies.
    pass

  # 2. Move mouse away and clear focus so transient tooltips and focus outlines
  # do not produce visual or DOM divergences between runs.
  try:
    page.mouse.move(0, 0)
    page.evaluate(
        "() => { if (document.activeElement && document.activeElement.blur)"
        " document.activeElement.blur(); }"
    )
  except _PlaywrightError:
    pass

  # 3. Wait for all visible progress bars, spinners, and loading banners.
  try:
    sync_api.expect(
        page.locator(
            ":is(.mat-mdc-progress-spinner, mat-spinner, .loading-spinner,"
            " mat-progress-bar, .mat-mdc-progress-bar,"
            " .loading-message):visible"
        )
    ).to_have_count(0, timeout=15000)
  except (_PlaywrightError, AssertionError):
    logging.warning(
        "Loading indicators remained visible after 15000ms on waypoint '%s'.",
        name,
    )
    unsettled.append("loading_indicators_timeout")

  # 4. Wait for visible charts to finish drawing their SVGs and stabilize.
  try:
    charts = page.locator(":is(chart, google-chart, step-time-graph)")
    for chart in charts.all():
      try:
        is_active = chart.evaluate("el => el.offsetParent !== null")
        if not is_active:
          continue
        if chart.locator("svg").count() > 0:
          chart.locator("svg :is(path, rect, line, circle, g)").first.wait_for(
              state="visible", timeout=2000
          )
        elif (
            chart.locator(":is(table, .google-visualization-table)").count() > 0
        ):
          pass
        else:
          try:
            chart.locator(
                ":is(svg, table, .google-visualization-table)"
            ).first.wait_for(state="visible", timeout=1500)
            if chart.locator("svg").count() > 0:
              chart.locator(
                  "svg :is(path, rect, line, circle, g)"
              ).first.wait_for(state="visible", timeout=2000)
          except (_PlaywrightError, AssertionError):
            pass
      except (_PlaywrightError, AssertionError):
        pass
  except _PlaywrightError:
    pass

  # Eliminate SVG chart dimension flutter and timing races across animation
  # frames before sampling DOM and screenshot.
  try:
    page.evaluate("""() => new Promise(resolve => {
      let timerId = null;
      let rafId = null;
      let lastSizes = '';
      let stableCount = 0;
      function cleanup() {
        if (timerId !== null) clearTimeout(timerId);
        if (rafId !== null) cancelAnimationFrame(rafId);
      }
      function done() {
        cleanup();
        resolve();
      }
      function check() {
        const svgs = Array.from(document.querySelectorAll('svg'));
        if (svgs.length === 0) {
          return done();
        }
        const sizes = svgs.map(s => {
          const r = s.getBoundingClientRect();
          return `${Math.round(r.width)}x${Math.round(r.height)}`;
        }).join(';');
        if (sizes === lastSizes) {
          stableCount++;
          if (stableCount >= 2) return done();
        } else {
          stableCount = 0;
          lastSizes = sizes;
        }
        rafId = requestAnimationFrame(check);
      }
      timerId = setTimeout(done, 3000);
      rafId = requestAnimationFrame(check);
    })""")
  except _PlaywrightError:
    pass

  # 5. Stabilize serialized DOM across consecutive quiescence polls.
  html, dom_settled = _wait_for_dom_quiescence(page)
  if not dom_settled:
    unsettled.append("dom_quiescence_timeout")

  if not recorder.wait_for_quiescence(page):
    unsettled.append("network_quiescence_timeout")

  png_bytes = page.screenshot(
      full_page=True,
      animations="disabled",
      caret="hide",
      scale="css",
      mask=[page.locator(selector) for selector in MASK_SELECTORS],
      mask_color=_MASK_COLOR,
  )
  return WaypointCapture(
      name=name,
      png_bytes=png_bytes,
      html=html,
      requests=recorder.drain(),
      stabilized=not unsettled,
      unsettled_reasons=tuple(unsettled),
  )
