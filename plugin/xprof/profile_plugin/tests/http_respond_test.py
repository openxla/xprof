"""Unit tests for HTTP respond helpers."""

from __future__ import annotations

import gzip
import json
import unittest

from xprof.profile_plugin.http.request_params import (
    generate_csv_filename,
    get_bool_arg,
)
from xprof.profile_plugin.http.respond import respond


class FakeRequest:
  def __init__(self, args):
    self.args = args


class RespondTest(unittest.TestCase):

  def test_json_gzip(self):
    resp = respond({'a': 1, 'b': 2}, 'application/json')
    self.assertEqual(resp.status_code, 200)
    self.assertEqual(resp.headers.get('Content-Encoding'), 'gzip')
    body = gzip.decompress(resp.get_data())
    self.assertEqual(json.loads(body), {'a': 1, 'b': 2})
    self.assertIn('Content-Security-Policy', resp.headers)

  def test_plain_no_auto_gzip_when_encoding_set(self):
    resp = respond('hello', 'text/plain', content_encoding='identity')
    # When content_encoding is truthy, body is not gzip-compressed by respond().
    self.assertEqual(resp.get_data(), b'hello')
    self.assertEqual(resp.headers.get('Content-Encoding'), 'identity')

  def test_get_bool_arg(self):
    self.assertTrue(get_bool_arg({'x': 'true'}, 'x', False))
    self.assertFalse(get_bool_arg({'x': 'false'}, 'x', True))
    self.assertTrue(get_bool_arg({}, 'x', True))
    self.assertFalse(get_bool_arg({}, 'x', False))

  def test_csv_filename_sanitized(self):
    req = FakeRequest({'tag': 'overview/page', 'run': 'run 1', 'host': 'a-b-host0'})
    name = generate_csv_filename(req)
    self.assertEqual(name, 'overview_page_run_1_host0.csv')


if __name__ == '__main__':
  unittest.main()
