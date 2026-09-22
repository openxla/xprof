"""Unit tests for journey_capture stabilization, normalization, and recording."""

import collections.abc
import dataclasses
import unittest

# pylint: disable=g-import-not-at-top
try:
  from google3.third_party.xprof.tests.ui import journey_capture
except ImportError:
  try:
    from tests.ui import journey_capture  # type: ignore[no-redef]
  except ImportError:
    import journey_capture  # type: ignore[no-redef]


@dataclasses.dataclass(frozen=True)
class _FakeRequest:
  method: str
  url: str


@dataclasses.dataclass(frozen=True)
class _FakeResponse:
  request: _FakeRequest
  url: str
  status: int


class _FakePage:
  """Minimal Playwright Page double for hermetic unit testing."""

  def __init__(
      self,
      evaluate_results: collections.abc.Sequence[object] | None = None,
      default_result: object = "drift",
  ):
    self.handlers: dict[str, list[object]] = {}
    self._evaluate_results = list(evaluate_results or [])
    self._default_result = default_result
    self.waits: list[int] = []

  def on(self, event: str, handler: object) -> None:
    self.handlers.setdefault(event, []).append(handler)

  def emit(self, event: str, payload: object) -> None:
    for handler in self.handlers.get(event, []):
      if callable(handler):
        handler(payload)

  def evaluate(self, script: str, arg: object = None) -> object:
    del script, arg
    if not self._evaluate_results:
      if self._default_result == "drift":
        return f"<html>drift-{len(self.waits)}</html>"
      return self._default_result
    result = self._evaluate_results.pop(0)
    if isinstance(result, Exception):
      raise result
    return result

  def wait_for_timeout(self, ms: int) -> None:
    self.waits.append(ms)


# pylint: disable=protected-access
class JourneyCaptureTest(unittest.TestCase):
  """Verifies URL masking, API network recording, and DOM quiescence."""

  def test_relative_url_strips_origin_and_masks_session_path(self):
    """Verifies scheme/port removal and session_path query masking."""
    raw = (
        "http://localhost:40397/data/plugin/profile/data"
        "?run=tpu_training&session_path=%2Ftmp%2Frun_123&tag=overview_page"
    )
    self.assertEqual(
        journey_capture._relative_url(raw),
        "/data/plugin/profile/data"
        "?run=tpu_training&session_path=<masked>&tag=overview_page",
    )

  def test_network_recorder_filters_non_api_and_deduplicates_gets(self):
    """Verifies static assets are ignored and duplicate GET polls collapse."""
    page = _FakePage()
    recorder = journey_capture.NetworkRecorder(page)

    # Static bundle should be ignored.
    page.emit(
        "response",
        _FakeResponse(
            request=_FakeRequest("GET", "http://localhost:8791/zone.js"),
            url="http://localhost:8791/zone.js",
            status=200,
        ),
    )
    # Duplicate GET /data/plugin/profile/config should collapse to one entry.
    for _ in range(2):
      page.emit(
          "response",
          _FakeResponse(
              request=_FakeRequest(
                  "GET", "http://localhost:8791/data/plugin/profile/config"
              ),
              url="http://localhost:8791/data/plugin/profile/config",
              status=200,
          ),
      )

    drained = recorder.drain()
    self.assertEqual(
        drained,
        [{
            "method": "GET",
            "url": "/data/plugin/profile/config",
            "status": 200,
        }],
    )

  def test_wait_for_dom_quiescence_reports_settled_and_timeout(self):
    """Verifies (html, True) on two matching polls and (html, False) on drift."""
    settled_page = _FakePage([
        "ready",
        "<html>A</html>",
        "<html>B</html>",
        "<html>B</html>",
        "<html>B</html>",
    ])
    html, settled = journey_capture._wait_for_dom_quiescence(settled_page)
    self.assertTrue(settled)
    self.assertEqual(html, "<html>B</html>")

    drifting_samples = ["ready"] + [f"<html>{i}</html>" for i in range(100)]
    drifting_page = _FakePage(drifting_samples)
    orig_timeout = journey_capture._QUIESCENCE_TIMEOUT_MS
    journey_capture._QUIESCENCE_TIMEOUT_MS = 10
    try:
      _, settled_timeout = journey_capture._wait_for_dom_quiescence(
          drifting_page
      )
      self.assertFalse(settled_timeout)
    finally:
      journey_capture._QUIESCENCE_TIMEOUT_MS = orig_timeout

  def test_normalize_generated_attr_ids_preserves_semantic_tab_index(self):
    """Verifies two-counter Material IDs preserve tab index."""
    self.assertEqual(
        journey_capture.normalize_generated_attr_ids("mat-tab-label-0-1"),
        "mat-tab-label-N-1",
    )
    self.assertEqual(
        journey_capture.normalize_generated_attr_ids("mat-tab-label-0-2"),
        "mat-tab-label-N-2",
    )
    self.assertEqual(
        journey_capture.normalize_generated_attr_ids(
            "cdk-describedby-message-ng-1-14"
        ),
        "cdk-describedby-message-ng-N-14",
    )
    self.assertEqual(
        journey_capture.normalize_generated_attr_ids("mat-select-4"),
        "mat-select-N",
    )
    self.assertEqual(
        journey_capture.normalize_generated_attr_ids(
            "google-visualization-errors-all-7"
        ),
        "google-visualization-errors-all-N",
    )

  def test_wait_for_angular_stable_handles_bootstrap_errors_and_timeouts(self):
    """Verifies bootstrap grace window, error retry, and timeout handling."""
    # 1. Instant settle after two consecutive True samples.
    instant_page = _FakePage([True, True])
    self.assertTrue(journey_capture._wait_for_angular_stable(instant_page))

    # 2. Bootstrap delay (None -> None -> False -> True -> True) waits for
    # Angular rather than exiting early on the first None.
    bootstrap_page = _FakePage([None, None, False, True, True])
    self.assertTrue(journey_capture._wait_for_angular_stable(bootstrap_page))
    self.assertGreaterEqual(len(bootstrap_page.waits), 3)

    # 3. Transient _PlaywrightError during navigation is retried until settled.
    retry_page = _FakePage(
        [RuntimeError("context destroyed"), False, True, True]
    )
    self.assertTrue(journey_capture._wait_for_angular_stable(retry_page))

    # 4. Non-Angular page (None past _ANGULAR_BOOTSTRAP_TIMEOUT_MS) returns
    # True.
    orig_boot = journey_capture._ANGULAR_BOOTSTRAP_TIMEOUT_MS
    orig_stable = journey_capture._ANGULAR_STABLE_TIMEOUT_MS
    journey_capture._ANGULAR_BOOTSTRAP_TIMEOUT_MS = 5
    journey_capture._ANGULAR_STABLE_TIMEOUT_MS = 50
    try:
      non_ng_page = _FakePage(default_result=None)
      self.assertTrue(journey_capture._wait_for_angular_stable(non_ng_page))
    finally:
      journey_capture._ANGULAR_BOOTSTRAP_TIMEOUT_MS = orig_boot
      journey_capture._ANGULAR_STABLE_TIMEOUT_MS = orig_stable

    # 5. Unstable Angular page past _ANGULAR_STABLE_TIMEOUT_MS returns False.
    journey_capture._ANGULAR_STABLE_TIMEOUT_MS = 5
    try:
      hung_page = _FakePage(default_result=False)
      self.assertFalse(journey_capture._wait_for_angular_stable(hung_page))
    finally:
      journey_capture._ANGULAR_STABLE_TIMEOUT_MS = orig_stable

    # 6. Placeholder about:blank iframes are excluded from iframe bypass.
    self.assertIn("about:blank", journey_capture._ANGULAR_IS_STABLE_JS)

  def test_nested_doc_is_painted_rejects_empty_body_and_about_blank(self):
    """Verifies empty body children and about:blank are rejected."""
    self.assertFalse(
        journey_capture._nested_doc_is_painted(
            needs_canvas=False, state="complete|0||0"
        )
    )
    self.assertFalse(
        journey_capture._nested_doc_is_painted(
            needs_canvas=False, state="complete|2||10", frame_url="about:blank"
        )
    )
    self.assertTrue(
        journey_capture._nested_doc_is_painted(
            needs_canvas=False,
            state="complete|2||10",
            frame_url="http://localhost:8791/graph_viewer",
        )
    )
    self.assertFalse(
        journey_capture._nested_doc_is_painted(
            needs_canvas=True,
            state="complete|2||10",
            frame_url="http://localhost:8791/trace_viewer",
        )
    )
    self.assertTrue(
        journey_capture._nested_doc_is_painted(
            needs_canvas=True,
            state="complete|2|800x600|10",
            frame_url="http://localhost:8791/trace_viewer",
        )
    )

  def test_wait_for_nested_documents_skips_idle_iframe_and_settles_active_trace(
      self,
  ):
    """Verifies idle frame skipping, active frame settling, and error recovery."""

    class _FakeFrame:
      """Fake Playwright child frame returning scripted nested-doc states."""

      def __init__(
          self,
          states: list[tuple[str, str | Exception]],
      ):
        self.parent_frame = object()
        self._states = list(states)
        self.url = states[0][0] if states else "about:blank"

      def evaluate(self, script: str) -> str:
        del script
        if len(self._states) > 1:
          self.url, state = self._states.pop(0)
        else:
          self.url, state = self._states[0]
        if isinstance(state, Exception):
          raise state
        return state

    class _FakeNestedPage:
      """Fake Playwright page with an active iframe count and child frame."""

      def __init__(
          self,
          active_count: int | list[int | Exception],
          frame: _FakeFrame,
      ):
        self.url = "http://localhost:8791/#profile/graph_viewer"
        self.frames = [frame]
        self._active_counts = (
            list(active_count)
            if isinstance(active_count, list)
            else [active_count]
        )
        self.waits: list[int] = []

      def evaluate(self, script: str) -> int:
        del script
        if len(self._active_counts) > 1:
          val = self._active_counts.pop(0)
        else:
          val = self._active_counts[0]
        if isinstance(val, Exception):
          raise val
        return val

      def wait_for_timeout(self, ms: int) -> None:
        self.waits.append(ms)

    # 1. Idle unplotted Graph Viewer iframe (active_count == 0) returns True
    # immediately even though its child frame is at about:blank.
    idle_frame = _FakeFrame([("about:blank", "complete|0||0")])
    idle_page = _FakeNestedPage(active_count=0, frame=idle_frame)
    self.assertTrue(journey_capture._wait_for_nested_documents(idle_page))
    self.assertEqual(idle_page.waits, [])

    # 2. Active Trace Viewer iframe transitions from about:blank to painted
    # canvas and settles across consecutive samples.
    active_frame = _FakeFrame([
        ("about:blank", "complete|0||0"),
        ("http://localhost:8791/trace_viewer", "complete|2|800x600|42"),
    ])
    active_page = _FakeNestedPage(active_count=1, frame=active_frame)
    active_page.url = "http://localhost:8791/#profile/trace_viewer"
    self.assertTrue(journey_capture._wait_for_nested_documents(active_page))
    self.assertGreaterEqual(
        len(active_page.waits), journey_capture._NESTED_DOC_STABLE_SAMPLES
    )

    # 3. Transient _PlaywrightError on first evaluate call recovers and settles.
    error_frame = _FakeFrame([
        (
            "http://localhost:8791/trace_viewer",
            journey_capture._PlaywrightError("Frame detached mid-sample"),
        ),
        ("http://localhost:8791/trace_viewer", "complete|2|800x600|42"),
    ])
    error_page = _FakeNestedPage(active_count=1, frame=error_frame)
    error_page.url = "http://localhost:8791/#profile/trace_viewer"
    self.assertTrue(journey_capture._wait_for_nested_documents(error_page))
    self.assertEqual(
        len(error_page.waits), journey_capture._NESTED_DOC_STABLE_SAMPLES + 1
    )

    # 4. Mid-sequence error invalidates prior matching samples and resets
    # previous and stable_count.
    flaky_frame = _FakeFrame([
        ("http://localhost:8791/trace_viewer", "complete|2|800x600|42"),
        ("http://localhost:8791/trace_viewer", "complete|2|800x600|42"),
        (
            "http://localhost:8791/trace_viewer",
            journey_capture._PlaywrightError("Frame detached mid-sample"),
        ),
        ("http://localhost:8791/trace_viewer", "complete|2|800x600|42"),
    ])
    flaky_page = _FakeNestedPage(active_count=1, frame=flaky_frame)
    flaky_page.url = "http://localhost:8791/#profile/trace_viewer"
    self.assertTrue(journey_capture._wait_for_nested_documents(flaky_page))
    self.assertEqual(
        len(flaky_page.waits), journey_capture._NESTED_DOC_STABLE_SAMPLES + 3
    )

    # 5. Page evaluate error (e.g. navigation destroying context) recovers
    # cleanly.
    page_error_frame = _FakeFrame([
        ("http://localhost:8791/trace_viewer", "complete|2|800x600|42"),
    ])
    page_error_page = _FakeNestedPage(
        active_count=[
            journey_capture._PlaywrightError("Context destroyed"),
            1,
        ],
        frame=page_error_frame,
    )
    page_error_page.url = "http://localhost:8791/#profile/trace_viewer"
    self.assertTrue(journey_capture._wait_for_nested_documents(page_error_page))
    self.assertEqual(
        len(page_error_page.waits),
        journey_capture._NESTED_DOC_STABLE_SAMPLES + 1,
    )

    # 6. Page without frames attribute returns True immediately.
    self.assertTrue(journey_capture._wait_for_nested_documents(object()))

  def test_wait_for_charts_to_render_settles_when_drawn_and_pending_stabilize(
      self,
  ):
    """Verifies chart wait holds on [0, pending] and settles once drawn > 0 holds steady."""

    class _FakeChartPage:
      """Fake Playwright page returning a sequence of [drawn, pending] counts."""

      def __init__(self, counts_sequence: list[list[int]]):
        self._seq = list(counts_sequence)
        self.waits: list[int] = []

      def evaluate(self, script: str) -> list[int]:
        del script
        if len(self._seq) > 1:
          return self._seq.pop(0)
        return self._seq[0]

      def wait_for_timeout(self, ms: int) -> None:
        self.waits.append(ms)

    page = _FakeChartPage([[0, 3], [2, 1]])
    self.assertTrue(journey_capture._wait_for_charts_to_render(page))
    self.assertGreaterEqual(len(page.waits), 2)

    zero_data_page = _FakeChartPage([[0, 2]])
    self.assertTrue(journey_capture._wait_for_charts_to_render(zero_data_page))
    self.assertGreaterEqual(len(zero_data_page.waits), 4)
    js = journey_capture._CHART_COUNTS_JS
    self.assertIn("proto !== Object.prototype", js)
    self.assertNotIn("Object.defineProperty(Object.prototype", js)

  def test_settled_tool_url_pattern_rejects_mismatched_pathname_and_tag(self):
    """Verifies settled_tool_url_pattern rejects premature updateUrlHistory URLs."""
    mem_pattern = journey_capture.settled_tool_url_pattern("Memory Profile")
    self.assertIsNone(
        mem_pattern.search(
            "http://localhost:8791/overview_page?run=r1&tag=memory_profile"
        )
    )
    self.assertIsNone(
        mem_pattern.search(
            "http://localhost:8791/memory_profile?run=r1&tag=overview_page"
        )
    )
    self.assertIsNotNone(
        mem_pattern.search(
            "http://localhost:8791/memory_profile?run=r1&tag=memory_profile"
        )
    )
    inp_pattern = journey_capture.settled_tool_url_pattern(
        "Input Pipeline Analysis"
    )
    self.assertIsNotNone(
        inp_pattern.search(
            "http://localhost:8791/input_pipeline_analyzer?run=r1&tag=input_pipeline_analyzer@"
        )
    )


if __name__ == "__main__":
  unittest.main()

