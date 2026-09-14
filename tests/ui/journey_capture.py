"""Stabilized screenshot, DOM, and network capture for XProf UI journeys.

The SxS diff engine compares two renders of the same waypoint. Anything that
varies between two runs of an unchanged build -- animation frames, the text
caret, the server port -- becomes a false divergence, so every capture here is
normalized before it reaches the engine.
"""

import dataclasses
import time
import urllib.parse

from playwright import sync_api

# Playwright waits this long for the network to fall idle before a capture.
# Exceeding it is not an error: XProf tools poll, so some pages never reach
# idle, and the quiescence wait below is what actually stabilizes those.
_NETWORK_IDLE_TIMEOUT_MS = 5000

# The DOM is considered settled once two consecutive samples this far apart are
# identical. Google Charts redraws when its container resizes, and the redraw
# lands after the network is already quiet, so a capture taken on a fixed delay
# catches a chart mid-layout and records different gridline geometry on each
# walk.
_QUIESCENCE_POLL_MS = 250
_QUIESCENCE_TIMEOUT_MS = 8000

# Regions that legitimately differ between two runs of the same build. Masking
# paints them a flat color in both captures so the engine sees them as equal
# instead of reporting a diff at every waypoint.
MASK_SELECTORS: tuple[str, ...] = (
    # Absolute log directory, which embeds a per-run temporary path.
    "sidenav .session-path",
)

# Flat fill painted over masked regions. Chosen to be absent from the XProf
# palette so a mask is obvious when a reviewer opens the report.
_MASK_COLOR = "#ff00ff"


@dataclasses.dataclass(frozen=True)
class WaypointCapture:
  """Multi-modal snapshot of a single journey waypoint."""

  name: str
  png_bytes: bytes
  html: str
  requests: list[dict[str, object]]


def _relative_url(url: str) -> str:
  """Strips scheme and authority, keeping the path and query.

  Control and candidate run on different ports, so absolute URLs would differ
  for every request and report the whole waterfall as diverged.

  Args:
    url: Absolute or relative request URL.

  Returns:
    The path and query components, without scheme, host, or port.
  """
  parts = urllib.parse.urlsplit(url)
  return urllib.parse.urlunsplit(("", "", parts.path, parts.query, ""))


class NetworkRecorder:
  """Records completed responses issued by a page."""

  def __init__(self, page: sync_api.Page):
    self._entries: list[dict[str, object]] = []
    page.on("response", self._on_response)

  def _on_response(self, response: sync_api.Response) -> None:
    self._entries.append({
        "method": response.request.method,
        "url": _relative_url(response.url),
        "status": response.status,
    })

  def drain(self) -> list[dict[str, object]]:
    """Returns the responses recorded so far and clears the buffer."""
    recorded = self._entries
    self._entries = []
    return recorded


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
_NORMALIZED_HTML_JS = """() => {
  const clone = document.documentElement.cloneNode(true);
  const stack = [clone];
  while (stack.length) {
    const el = stack.pop();
    const attrs = Array.from(el.attributes)
        .map(a => [a.name, a.value])
        .sort((x, y) => x[0] < y[0] ? -1 : x[0] > y[0] ? 1 : 0);
    for (const [name] of attrs) el.removeAttribute(name);
    for (const [name, value] of attrs) el.setAttribute(name, value);
    for (const child of el.children) stack.push(child);
  }
  return clone.outerHTML;
}"""


def _normalized_html(page: sync_api.Page) -> str:
  """Returns the document serialized with attributes in a stable order."""
  return page.evaluate(_NORMALIZED_HTML_JS)


def _wait_for_dom_quiescence(page: sync_api.Page) -> str:
  """Polls until the serialized DOM stops changing, and returns it.

  Args:
    page: Page to sample.

  Returns:
    The last serialization observed. On timeout this is the most recent
    sample rather than a settled one; the reproducibility assertion in the
    caller is what surfaces that, since an unsettled page will not match on
    the next walk.
  """
  try:
    page.evaluate("document.fonts.ready")
  except sync_api.Error:
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
        return current
    else:
      stable_count = 0
    previous = current
  return previous


def capture_waypoint(
    page: sync_api.Page, recorder: NetworkRecorder, name: str
) -> WaypointCapture:
  """Captures a stabilized screenshot, serialized DOM, and network log.

  Args:
    page: Page already navigated to the waypoint under capture.
    recorder: Recorder attached to the same page.
    name: Waypoint identifier used in the report.

  Returns:
    The captured waypoint state.
  """
  # 1. Wait for in-flight tool data and layout network requests to settle.
  try:
    page.wait_for_load_state("networkidle", timeout=_NETWORK_IDLE_TIMEOUT_MS)
  except sync_api.TimeoutError:
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
  except sync_api.Error:
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
  except (sync_api.Error, AssertionError):
    pass

  # 4. Wait for all visible charts to finish drawing their SVGs.
  try:
    for chart in page.locator("chart:visible, google-chart:visible").all():
      try:
        chart.locator("svg").first.wait_for(state="visible", timeout=10000)
      except sync_api.Error:
        pass
  except sync_api.Error:
    pass

  # 5. Stabilize serialized DOM across consecutive quiescence polls.
  html = _wait_for_dom_quiescence(page)

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
  )
