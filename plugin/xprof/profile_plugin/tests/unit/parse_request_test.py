"""Tests for HTTP → ToolRequest parsing."""

from __future__ import annotations

import unittest

from xprof.profile_plugin.http.parse_request import tool_request_from_args


class ParseRequestTest(unittest.TestCase):

  def test_basic_fields(self):
    req = tool_request_from_args({
        'run': 'session_a',
        'tag': 'overview_page',
        'host': 'ALL_HOSTS',
        'use_saved_result': 'true',
    })
    self.assertEqual(req.run, 'session_a')
    self.assertEqual(req.tool, 'overview_page')
    self.assertEqual(req.host, 'ALL_HOSTS')
    self.assertEqual(req.hosts, ())
    self.assertTrue(req.use_saved_result)

  def test_hosts_csv(self):
    req = tool_request_from_args({
        'run': 'r',
        'tag': 'trace_viewer@',
        'hosts': 'h0,h1',
    })
    self.assertEqual(req.hosts, ('h0', 'h1'))

  def test_use_saved_result_default_true(self):
    req = tool_request_from_args({'run': 'r', 'tag': 'overview_page'})
    self.assertTrue(req.use_saved_result)

  def test_raw_args_preserves_strings(self):
    req = tool_request_from_args({
        'run': 'r',
        'tag': 'trace_viewer@',
        'resolution': '8000',
    })
    self.assertEqual(req.raw_args.get('resolution'), '8000')


if __name__ == '__main__':
  unittest.main()
