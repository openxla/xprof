"""Tests for http.logging_middleware."""

from __future__ import annotations

import io
import json
import unittest

from xprof.profile_plugin.http.logging_middleware import logging_wrapper


class LoggingMiddlewareTest(unittest.TestCase):

  def test_logs_json_with_status_and_latency(self):
    def app(environ, start_response):
      start_response('404 Not Found', [('Content-Type', 'text/plain')])
      return [b'nope']

    buf = io.StringIO()
    times = iter([100.0, 100.5])
    wrapped = logging_wrapper(app, clock=lambda: next(times), writer=buf)
    list(
        wrapped(
            {
                'REQUEST_METHOD': 'GET',
                'SCRIPT_NAME': '',
                'PATH_INFO': '/data',
                'QUERY_STRING': 'run=r1',
            },
            lambda *a, **k: None,
        )
    )
    line = buf.getvalue().strip()
    entry = json.loads(line)
    self.assertEqual(entry['httpRequest']['requestMethod'], 'GET')
    self.assertEqual(entry['httpRequest']['requestUrl'], '/data?run=r1')
    self.assertEqual(entry['httpRequest']['status'], 404)
    self.assertEqual(entry['httpRequest']['latency'], '0.500000000s')

  def test_malformed_status_defaults_log_path(self):
    def app(environ, start_response):
      start_response('OK', [])  # not "200 OK"
      return [b'']

    buf = io.StringIO()
    wrapped = logging_wrapper(app, clock=lambda: 1.0, writer=buf)
    # second clock call in finally — use constant then bump
    times = [1.0]

    def clock():
      times[0] += 0.1
      return times[0]

    wrapped = logging_wrapper(app, clock=clock, writer=buf)
    list(wrapped({'REQUEST_METHOD': 'POST', 'PATH_INFO': '/x'}, lambda *a, **k: None))
    entry = json.loads(buf.getvalue().strip())
    # parse fails → stays default 200
    self.assertEqual(entry['httpRequest']['status'], 200)


if __name__ == '__main__':
  unittest.main()
