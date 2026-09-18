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
      self, evaluate_results: collections.abc.Sequence[object] | None = None
  ):
    self.handlers: dict[str, list[object]] = {}
    self._evaluate_results = list(evaluate_results or [])
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
      return f"<html>drift-{len(self.waits)}</html>"
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


if __name__ == "__main__":
  unittest.main()
