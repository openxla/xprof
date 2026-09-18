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


if __name__ == "__main__":
  unittest.main()
