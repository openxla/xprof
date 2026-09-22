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

# Angular reports itself stable once change detection, timers, and zone-tracked
# requests have all drained. Host Angular components reach this state well
# within the timeout (Roofline Model is slowest at ~4.7s). Tools hosted inside
# an iframe (Trace Viewer, Graph Viewer) or running a background tutorial timer
# (Trace Viewer's interval(3000)) are short-circuited once the host shell and
# iframe mount, while _ANGULAR_BOOTSTRAP_TIMEOUT_MS waits for Angular to
# register window.getAllAngularTestabilities() right after page.goto().
_ANGULAR_STABLE_POLL_MS: int = 100
_ANGULAR_BOOTSTRAP_TIMEOUT_MS: int = 2000
_ANGULAR_STABLE_TIMEOUT_MS: int = 15000

# Trace Viewer runs in a nested document that neither Angular's testability
# registry nor the serialized DOM can observe, yet the screenshot renders it.
# Its document reports "complete" roughly 1.4s before it paints and holds
# perfectly still in between, so readiness has to be judged on painted canvases
# rather than on the document being loaded or merely unchanging. Measured from
# the moment Angular reports the host page stable, canvases appear after about
# 2s and the trace text settles about 1s after that. The timeout is a hang
# guard set an order of magnitude above that; a healthy capture never nears it,
# and a slow machine is allowed to take its time rather than record a blank
# trace.
_NESTED_DOC_POLL_MS = 250
_NESTED_DOC_TIMEOUT_MS = 30000
_NESTED_DOC_STABLE_SAMPLES = 3

# Google Charts draws from its own loader callbacks rather than from Angular's
# zone, so a chart element can still be empty after Angular reports the page
# stable. The Roofline Model pie chart does exactly that under load: one walk
# serialized a drawn chart and the next serialized a bare <chart> element,
# roughly 7KB of missing DOM. The wait is taken once for the whole page rather
# than per chart, so a page full of charts costs one timeout rather than one
# each.
_CHART_RENDER_POLL_MS = 100
_CHART_RENDER_SETTLE_MS = 1500
_CHART_RENDER_TIMEOUT_MS = 25000

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
    // Google Charts and Angular Material number generated elements from
    // page-global counters, so async render order shifts the suffixes across
    // walks without changing the structure or content of the page. For
    // two-counter tokens like mat-tab-label-<groupId>-<tabIndex>, normalize
    // only the page-global group counter while preserving the semantic index.
    for (const attr of [
      'id', 'for', 'aria-labelledby', 'aria-controls', 'aria-owns',
      'aria-describedby',
    ]) {
      const val = el.getAttribute(attr);
      if (!val) continue;
      const norm = val
          .replace(
              /\\b(google-visualization-errors(?:-all)?-)[0-9]+\\b/g, '$1N'
          )
          .replace(
              /\\b((?:mat|cdk)-[a-z0-9-]+?-)[0-9]+(-[0-9]+)\\b/g, '$1N$2'
          )
          .replace(
              /\\b((?:mat|cdk)-(?![a-z0-9-]+-N-[0-9]+\\b)[a-z0-9-]+-)[0-9]+\\b/g,
              '$1N'
          );
      if (norm !== val) el.setAttribute(attr, norm);
    }
    for (const child of el.children) stack.push(child);
  }
  return clone.outerHTML;
}"""

_GVIS_ID_RE = re.compile(r"\b(google-visualization-errors(?:-all)?-)[0-9]+\b")
_MAT_TWO_COUNTER_ID_RE = re.compile(
    r"\b((?:mat|cdk)-[a-z0-9-]+?-)[0-9]+(-[0-9]+)\b"
)
_MAT_SINGLE_COUNTER_ID_RE = re.compile(
    r"\b((?:mat|cdk)-(?![a-z0-9-]+-N-[0-9]+\b)[a-z0-9-]+-)[0-9]+\b"
)


def normalize_generated_attr_ids(value: str) -> str:
  """Normalizes page-global counters in Material/CDK and Google Charts IDs."""
  norm = _GVIS_ID_RE.sub(r"\1N", value)
  norm = _MAT_TWO_COUNTER_ID_RE.sub(r"\1N\2", norm)
  return _MAT_SINGLE_COUNTER_ID_RE.sub(r"\1N", norm)


def _normalized_html(page: typing.Any) -> str:
  """Returns the document serialized with attributes in a stable order."""
  raw = page.evaluate(_NORMALIZED_HTML_JS, list(MASK_SELECTORS))
  return _SESSION_PATH_RE.sub("<masked>", _LOCALHOST_ORIGIN_RE.sub("", raw))


# Reports whether every Angular testability on the page has drained its pending
# work, or null when Angular has not yet finished bootstrapping. When an
# iframe-hosted tool (Trace Viewer, Graph Viewer) has mounted its visible
# iframe and no host loading indicator is active, host stability is treated as
# reached so Trace Viewer's in-zone interval(3000) tutorial rotation does not
# block until the 15s timeout.
#
# Sampling is driven from Python rather than from a timer inside this snippet,
# because a timer scheduled here would itself be a pending task in whichever
# zone it lands in and could keep the page from ever reporting stable.
_ANGULAR_IS_STABLE_JS = """() => {
  if (typeof window.getAllAngularTestabilities !== 'function') return null;
  const testabilities = window.getAllAngularTestabilities();
  if (testabilities.length === 0) return null;
  if (testabilities.every(t => t.isStable())) return true;
  const hasVisibleSourcedIframe = Array.from(document.querySelectorAll('iframe')).some(
      f => {
        const src = (f.getAttribute('src') || '').trim();
        return f.offsetParent !== null &&
               f.getBoundingClientRect().width > 0 &&
               Boolean(src) &&
               src.toLowerCase() !== 'about:blank';
      }
  );
  if (hasVisibleSourcedIframe) {
    const hasActiveSpinner = document.querySelector(
        '.mat-mdc-progress-spinner, mat-spinner, .loading-spinner, ' +
        'mat-progress-bar, .mat-mdc-progress-bar, .loading-message'
    );
    if (!hasActiveSpinner) return true;
  }
  return false;
}"""


def _wait_for_angular_stable(page: typing.Any) -> bool:
  """Waits until Angular has no pending change detection, timers, or requests.

  A settled DOM is not sufficient on its own. A tool request finishes, and
  Angular then spends over a second building its chart data before painting
  it, during which the DOM is static and the network is silent. Quiescence
  settles inside that window and captures an empty chart shell, so the same
  waypoint records a populated table on one walk and an empty one on the next.
  Angular reports stable only after that work completes.

  Args:
    page: Page to sample.

  Returns:
    True if Angular reported stable (or the page is non-Angular after the
    bootstrap window), False if the stability timeout expired.
  """
  start = time.monotonic()
  bootstrap_deadline = start + _ANGULAR_BOOTSTRAP_TIMEOUT_MS / 1000
  deadline = start + _ANGULAR_STABLE_TIMEOUT_MS / 1000
  stable_count = 0
  while time.monotonic() < deadline:
    try:
      is_stable = page.evaluate(_ANGULAR_IS_STABLE_JS)
    except _PlaywrightError:
      stable_count = 0
      page.wait_for_timeout(_ANGULAR_STABLE_POLL_MS)
      continue
    if is_stable is None:
      stable_count = 0
      if time.monotonic() >= bootstrap_deadline:
        return True
      page.wait_for_timeout(_ANGULAR_STABLE_POLL_MS)
      continue
    if is_stable:
      # Two consecutive samples, so a momentary lull between two asynchronous
      # stages of a tool load is not mistaken for the end of the load.
      stable_count += 1
      if stable_count >= 2:
        return True
    else:
      stable_count = 0
    page.wait_for_timeout(_ANGULAR_STABLE_POLL_MS)
  logging.warning(
      "Angular testability did not stabilize within %dms.",
      _ANGULAR_STABLE_TIMEOUT_MS,
  )
  return False


# Counts visible iframes that are expected to load a nested document. Unplotted
# Graph Viewer pages mount an empty <iframe id="graph-html"> without a src
# attribute that stays at about:blank until a node is queried, and Trace Viewer
# V2 hides the legacy <iframe #tvIframe> with [hidden]="useTraceViewerV2".
_ACTIVE_IFRAME_COUNT_JS = """() => {
  const url = window.location.href || '';
  const hasGraphTarget = /[?&](node_name|opName|symbol_id)=[^&]+/.test(url);
  const isTraceViewer = /trace_viewer/.test(url);
  let active = 0;
  for (const el of document.querySelectorAll('iframe')) {
    if (el.offsetParent === null || el.getBoundingClientRect().width <= 0) {
      continue;
    }
    const src = (el.getAttribute('src') || '').trim();
    if (src && src.toLowerCase() !== 'about:blank') {
      active++;
      continue;
    }
    try {
      const href = el.contentWindow?.location?.href || 'about:blank';
      if (
          href !== 'about:blank' ||
          isTraceViewer ||
          (el.id === 'graph-html' && hasGraphTarget)
      ) {
        active++;
      }
    } catch (e) {
      active++;
    }
  }
  return active;
}"""


# Reports a nested document's paint progress as
# "readyState|bodyChildren|canvasSizes|textLength". Canvas sizes are the
# load-bearing field: Trace Viewer reaches "complete" with an empty body and
# only later attaches the canvases it draws the trace into.
_NESTED_DOC_STATE_JS = """() => {
  const canvases = Array.from(document.querySelectorAll('canvas'))
      .map(c => `${c.width}x${c.height}`)
      .filter(s => s !== '0x0')
      .join(',');
  const body = document.body;
  const children = body ? body.children.length : 0;
  const text = body && body.innerText ? body.innerText.length : 0;
  return `${document.readyState}|${children}|${canvases}|${text}`;
}"""


def _nested_doc_is_painted(
    needs_canvas: bool, state: str, frame_url: str = ""
) -> bool:
  """Judges whether a nested document has finished drawing.

  Args:
    needs_canvas: Whether the document is expected to paint canvases.
    state: Signature produced by the state script.
    frame_url: Optional URL of the nested frame.

  Returns:
    True once the document is loaded, has non-empty body children, and, where
    canvases are expected, has attached at least one with a non-zero size.
  """
  if frame_url.strip().lower() == "about:blank":
    return False
  fields = state.split("|")
  if len(fields) != 4 or fields[0] != "complete":
    return False
  try:
    body_children = int(fields[1])
  except ValueError:
    return False
  if body_children <= 0:
    return False
  return bool(fields[2]) if needs_canvas else True


def _wait_for_nested_documents(page: typing.Any) -> bool:
  """Waits until every nested document on the page has painted and settled.

  The screenshot renders nested documents, but none of the other waits can
  see into one: Angular's registry tracks only the host application, and the
  serialized DOM stops at the frame element. Trace Viewer exploits both gaps.
  It reports "complete" while still blank, stays byte-for-byte identical for
  over a second, and only then paints, so a wait that settles on stillness
  captures an empty trace on one walk and a drawn one on the next.

  Args:
    page: Page to sample.

  Returns:
    True if all nested documents painted and settled (or none exist), False if
    the timeout expired.
  """
  if not hasattr(page, "frames"):
    return True
  deadline = time.monotonic() + _NESTED_DOC_TIMEOUT_MS / 1000
  previous = None
  stable_count = 0
  while time.monotonic() < deadline:
    try:
      active_count = page.evaluate(_ACTIVE_IFRAME_COUNT_JS)
      has_active_int = isinstance(active_count, int) and not isinstance(
          active_count, bool
      )
      if has_active_int and active_count == 0:
        return True
      frames = [frame for frame in page.frames if frame.parent_frame]
      if not frames:
        # The frame element can exist before its document attaches, so an
        # empty list means either "no nested documents" or "not yet". Only
        # the former is a reason to stop waiting.
        if page.locator("iframe").count() == 0:
          return True
        page.wait_for_timeout(_NESTED_DOC_POLL_MS)
        continue
      # Trace Viewer is the only nested document that draws to a canvas, and
      # it is identifiable from either its own URL or the hosting tool's.
      needs_canvas = "trace_viewer" in getattr(page, "url", "")
      states = []
      painted_count = 0
      for frame in frames:
        state = frame.evaluate(_NESTED_DOC_STATE_JS)
        states.append(state)
        frame_url = getattr(frame, "url", "")
        if _nested_doc_is_painted(
            needs_canvas or "trace_viewer" in frame_url,
            state,
            frame_url,
        ):
          painted_count += 1
      required_painted = active_count if has_active_int else len(frames)
      painted = painted_count >= required_painted
      current = "\n".join(states)
    except _PlaywrightError:
      # A frame navigating or detaching mid-sample invalidates the reading
      # rather than the page; take the next sample instead of settling.
      previous = None
      stable_count = 0
      page.wait_for_timeout(_NESTED_DOC_POLL_MS)
      continue
    if painted and current == previous:
      stable_count += 1
      if stable_count >= _NESTED_DOC_STABLE_SAMPLES:
        return True
    else:
      stable_count = 0
    previous = current
    page.wait_for_timeout(_NESTED_DOC_POLL_MS)
  logging.warning(
      "Nested documents did not settle within %dms.",
      _NESTED_DOC_TIMEOUT_MS,
  )
  return False


# Counts visible chart elements that have drawn versus those still blank.
# Google Charts renders into the element as an SVG, or as a table for the
# tabular variants, so an element with neither is still pending.
_CHART_COUNTS_JS = """() => {
  const seen = new WeakSet();
  function findDataProvider(root, depth) {
    if (!root || typeof root !== 'object' || depth < 0 || seen.has(root)) {
      return null;
    }
    seen.add(root);
    if (root.update && typeof root.update.emit === 'function') return root;
    const vals = Array.isArray(root) ? root : Object.values(root);
    for (const v of vals) {
      if (v && typeof v === 'object' && !(v instanceof Node) && !(v instanceof Window)) {
        const found = findDataProvider(v, depth - 1);
        if (found) return found;
      }
    }
    return null;
  }
  const els = document.querySelectorAll(
      'chart, google-chart, step-time-graph');
  let drawn = 0;
  let pending = 0;
  for (const el of els) {
    if (el.offsetParent === null) continue;
    if (el.querySelector('svg, table, .google-visualization-table')) {
      drawn++;
    } else {
      const dp = findDataProvider(el.__ngContext__, 3);
      if (dp) {
        const proto = Object.getPrototypeOf(dp);
        if (proto && proto !== Object.prototype && !proto.__xprofHooked) {
          proto.__xprofHooked = true;
          const sym = Symbol('dataTable');
          Object.defineProperty(proto, 'dataTable', {
            configurable: true,
            enumerable: true,
            get() { return this[sym]; },
            set(val) {
              this[sym] = val;
              if (val && this.update && typeof this.update.emit === 'function') {
                Promise.resolve().then(() => {
                  try { this.update.emit(); } catch (e) {}
                });
              }
            },
          });
        }
        if (dp.dataTable && !el.__xprofEmitted) {
          el.__xprofEmitted = true;
          try { dp.update.emit(); } catch (e) {}
        }
      }
      pending++;
    }
  }
  return [drawn, pending];
}"""


def _wait_for_charts_to_render(page: typing.Any) -> bool:
  """Waits until charts on the page have drawn and any blank count settles.

  Angular reporting stable does not imply the charts have drawn, because
  Google Charts loads its script modules and draws from its own callbacks.
  Capturing in that window serializes bare chart elements, which differ from
  the drawn charts the other walk captured by thousands of characters of DOM.

  Two rules govern when the wait can finish:
  1. Once at least one chart has drawn (drawn > 0), any remaining blank charts
     are either finishing in the same draw batch or bound to an empty category
     that will never draw, so the wait finishes as soon as the (drawn, pending)
     pair holds steady for `required` samples.
  2. When a profile run has zero data for a tool's charts (`drawn == 0` and
     `pending >= 1` after Angular and network quiescence), the wait settles
     once `(0, pending)` holds steady for `required * 2` samples instead of
     hanging until timeout.

  Args:
    page: Page to sample.

  Returns:
    True once charts have drawn and settled, False if the timeout expired.
  """
  required = max(1, _CHART_RENDER_SETTLE_MS // _CHART_RENDER_POLL_MS)
  deadline = time.monotonic() + _CHART_RENDER_TIMEOUT_MS / 1000
  previous = None
  stable_count = 0
  while time.monotonic() < deadline:
    try:
      counts = page.evaluate(_CHART_COUNTS_JS)
    except _PlaywrightError:
      previous = None
      stable_count = 0
      page.wait_for_timeout(_CHART_RENDER_POLL_MS)
      continue
    if not isinstance(counts, list) or len(counts) != 2:
      return True
    drawn, pending = counts[0], counts[1]
    if not pending:
      return True
    if counts == previous:
      stable_count += 1
      target_samples = required if drawn > 0 else required * 2
      if stable_count >= target_samples:
        return True
    else:
      stable_count = 0
    previous = counts
    page.wait_for_timeout(_CHART_RENDER_POLL_MS)
  logging.warning(
      "Charts did not finish rendering within %dms.",
      _CHART_RENDER_TIMEOUT_MS,
  )
  return False


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


_DEFAULT_TOOL_NAME_TO_TAG: dict[str, str] = {
    "Overview Page": "overview_page",
    "Input Pipeline Analysis": "input_pipeline",
    "Kernel Stats": "kernel_stats",
    "Trace Viewer": "trace_viewer",
    "Memory Profile": "memory_profile",
    "Pod Viewer": "pod_viewer",
    "Graph Viewer": "graph_viewer",
    "HLO Op Profile": "op_profile",
    "Memory Viewer": "memory_viewer",
    "Framework Op Stats": "framework_op_stats",
    "Megascale Stats": "megascale_stats",
    "Roofline Model": "roofline_model",
    "HLO Op Stats": "hlo_stats",
}


def settled_tool_url_pattern(tool_name: str) -> re.Pattern[str]:
  """Builds a URL regex requiring both pathname and tag query param to match."""
  expected_tag = _DEFAULT_TOOL_NAME_TO_TAG.get(
      tool_name, tool_name.lower().replace(" ", "_")
  )
  return re.compile(
      rf"/{re.escape(expected_tag)}(?:_analyzer)?(?:@|%40|[/?#]|$).*"
      rf"tag={re.escape(expected_tag)}(?:_analyzer)?(?:@|%40|[&#]|$)"
  )


def capture_waypoint(
    page: typing.Any,
    recorder: NetworkRecorder,
    name: str,
) -> WaypointCapture:
  """Captures a settled waypoint's visual, DOM, geometry, and network state.

  Args:
    page: Playwright page positioned at the waypoint.
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
  if sync_api is not None:
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

  # 4. Wait for Angular to finish rendering the tool's data. This is what
  # makes the capture deterministic; the waits around it only smooth layout.
  if not _wait_for_angular_stable(page):
    unsettled.append("angular_stable_timeout")

  # 5. Wait for nested documents, which the screenshot renders but no other
  # wait here can observe, to finish painting.
  if not _wait_for_nested_documents(page):
    unsettled.append("nested_documents_timeout")

  # 6. Wait for visible charts to finish drawing their SVGs and stabilize.
  # The page-wide wait comes first, so a chart that has not started drawing is
  # given time before the per-chart waits below refine what it drew.
  if not _wait_for_charts_to_render(page):
    unsettled.append("charts_render_timeout")
  try:
    charts = page.locator(":is(chart, google-chart, step-time-graph)")
    for chart in charts.all():
      try:
        is_active = chart.evaluate("el => el.offsetParent !== null")
        if not is_active:
          continue
        # Ensure rendered SVG vector shapes are visible without penalizing
        # permanently empty chart categories that _wait_for_charts_to_render
        # already settled.
        if chart.locator("svg").count() > 0:
          chart.locator("svg :is(path, rect, line, circle, g)").first.wait_for(
              state="visible", timeout=2000
          )
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

  # 7. Stabilize serialized DOM across consecutive quiescence polls.
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
