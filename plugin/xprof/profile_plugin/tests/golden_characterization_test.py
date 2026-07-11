# Copyright 2026 The XProf Authors. All Rights Reserved.
"""Golden characterization tests for profile plugin HTTP/domain contracts.

These tests lock stable, path-normalized snapshots for:
  - runs listing
  - run_tools ordering (with mocked tool_names)
  - hosts for ALL_HOSTS_ONLY vs SUPPORTED tools
  - convert call shapes (tool, basenames, hosts, use_saved_result, options)
  - /data HTTP status + gzip body
  - /config required keys

Update fixtures under tests/fixtures/golden/ only when intentional behavior
changes. Prefer basenames over absolute paths so fixtures stay portable.
"""

from __future__ import annotations

import gzip
import json
import os
import tempfile
import unittest
from unittest import mock

from werkzeug import wrappers
from werkzeug.test import EnvironBuilder

from xprof import profile_plugin
from xprof.profile_plugin.tests.fixtures import golden_loader
from xprof.profile_plugin.tests.fixtures.logdir_factory import (
    basenames_from_paths,
    create_two_host_session,
)
from xprof.profile_plugin.tests.fixtures.plugin_factory import make_plugin


def _invoke_route(app, path: str, query: dict) -> tuple[str, list, bytes]:
  """Call a Werkzeug app; return (status, headers, body)."""
  env = EnvironBuilder(path=path, query_string=query).get_environ()
  captured: dict = {}

  def start_response(status, headers, exc_info=None):
    captured['status'] = status
    captured['headers'] = headers

  body = b''.join(app(env, start_response))
  return captured['status'], captured.get('headers', []), body


def _header_value(headers, name: str) -> str | None:
  name_l = name.lower()
  for k, v in headers:
    if k.lower() == name_l:
      return v
  return None


def _decode_body(headers, body: bytes) -> bytes:
  enc = _header_value(headers, 'Content-Encoding')
  if enc and 'gzip' in enc.lower():
    return gzip.decompress(body)
  return body


class GoldenCharacterizationTest(unittest.TestCase):
  """Characterization suite against committed golden JSON fixtures."""

  def setUp(self):
    super().setUp()
    self._td = tempfile.TemporaryDirectory()
    self.logdir = self._td.name
    self.session = 'session_a'
    self.run_dir = create_two_host_session(self.logdir, self.session)
    self.calls: list[tuple] = []

    def recording_convert(paths, tool, params):
      self.calls.append(
          (
              basenames_from_paths(paths),
              tool,
              {
                  'hosts': list(params.get('hosts', [])),
                  'use_saved_result': params.get('use_saved_result'),
                  'trace_viewer_options': params.get('trace_viewer_options'),
              },
          )
      )
      return (
          json.dumps({'ok': True, 'tool': tool}, sort_keys=True),
          'application/json',
      )

    self.plugin = make_plugin(self.logdir, xspace_fn=recording_convert)

  def tearDown(self):
    self._td.cleanup()
    super().tearDown()

  def test_golden_runs(self):
    golden = golden_loader.load_golden('runs')
    runs = self.plugin.runs_imp()
    self.assertEqual(runs, golden['runs'])

  def test_golden_run_tools(self):
    golden = golden_loader.load_golden('run_tools')
    # Patch catalog path used by generate_tools_of_run (convert may be stubbed).
    with mock.patch(
        'xprof.profile_plugin.services.runs.get_active_tools',
        return_value=list(golden['tools']),
    ):
      tools = self.plugin.run_tools_imp(self.session)
    self.assertEqual(tools[0], golden['first'])
    self.assertEqual(
        'trace_viewer@' in tools, golden['has_trace_viewer_at']
    )
    self.assertEqual(
        'trace_viewer' in tools, golden['has_plain_trace_viewer']
    )
    for expected in golden['tools']:
      self.assertIn(expected, tools)

  def test_golden_hosts_overview(self):
    golden = golden_loader.load_golden('hosts_overview')
    hosts = self.plugin.host_impl(self.session, 'overview_page')
    self.assertEqual(hosts, golden['hosts'])

  def test_golden_hosts_kernel_stats(self):
    golden = golden_loader.load_golden('hosts_kernel_stats')
    hosts = self.plugin.host_impl(self.session, 'kernel_stats')
    names = sorted(h['hostname'] for h in hosts)
    self.assertEqual(names, golden['hostnames'])

  def test_golden_convert_overview(self):
    golden = golden_loader.load_golden('convert_overview')
    self.calls.clear()
    req = wrappers.Request({})
    req.args = {
        'run': self.session,
        'tag': 'overview_page',
        'host': 'ALL_HOSTS',
        'use_saved_result': 'true',
    }
    data, content_type, _ = self.plugin.data_impl(req)
    self.assertEqual(content_type, golden['content_type'])
    self.assertEqual(json.loads(data), golden['response_json'])
    self.assertEqual(len(self.calls), 1)
    basenames, tool, meta = self.calls[0]
    self.assertEqual(tool, golden['tool'])
    self.assertEqual(basenames, golden['asset_basenames'])
    self.assertEqual(sorted(meta['hosts']), sorted(golden['params_hosts']))
    self.assertEqual(meta['use_saved_result'], golden['use_saved_result'])
    self.assertTrue(
        os.path.exists(
            os.path.join(self.run_dir, profile_plugin.CACHE_VERSION_FILE)
        )
    )

  def test_golden_convert_trace_single_host(self):
    golden = golden_loader.load_golden('convert_trace_single_host')
    self.calls.clear()
    req = wrappers.Request({})
    req.args = {
        'run': self.session,
        'tag': 'trace_viewer@',
        'host': 'host0',
        'use_saved_result': 'true',
        'resolution': '8000',
    }
    data, content_type, _ = self.plugin.data_impl(req)
    self.assertEqual(content_type, golden['content_type'])
    self.assertEqual(json.loads(data), golden['response_json'])
    self.assertEqual(len(self.calls), 1)
    basenames, tool, meta = self.calls[0]
    self.assertEqual(tool, golden['tool'])
    self.assertEqual(basenames, golden['asset_basenames'])
    self.assertEqual(meta['hosts'], golden['params_hosts'])
    self.assertEqual(meta['use_saved_result'], golden['use_saved_result'])
    tv = meta['trace_viewer_options'] or {}
    for key, expected in golden['trace_viewer_options'].items():
      self.assertEqual(tv.get(key), expected, msg=f'trace option {key}')

  def test_golden_http_data_ok(self):
    golden = golden_loader.load_golden('http_data_ok')
    app = self.plugin.get_plugin_apps()[profile_plugin.DATA_ROUTE]
    status, headers, body = _invoke_route(
        app,
        '/data',
        {
            'run': self.session,
            'tag': 'overview_page',
            'host': 'ALL_HOSTS',
        },
    )
    self.assertTrue(status.startswith(golden['status_prefix']), status)
    self.assertEqual(
        _header_value(headers, 'Content-Encoding'), golden['content_encoding']
    )
    decoded = _decode_body(headers, body)
    self.assertEqual(json.loads(decoded), golden['body_json'])

  def test_golden_http_data_404(self):
    golden = golden_loader.load_golden('http_data_404')

    def empty_convert(paths, tool, params):
      return None, 'application/json'

    plugin = make_plugin(self.logdir, xspace_fn=empty_convert)
    app = plugin.get_plugin_apps()[profile_plugin.DATA_ROUTE]
    status, headers, body = _invoke_route(
        app,
        '/data',
        {
            'run': self.session,
            'tag': 'overview_page',
            'host': 'ALL_HOSTS',
        },
    )
    self.assertTrue(status.startswith(golden['status_prefix']), status)
    decoded = _decode_body(headers, body).decode('utf-8')
    self.assertEqual(decoded, golden['body_text'])

  def test_golden_config(self):
    golden = golden_loader.load_golden('config')
    app = self.plugin.get_plugin_apps()[profile_plugin.CONFIG_ROUTE]
    status, headers, body = _invoke_route(app, '/config', {})
    self.assertTrue(status.startswith('200'), status)
    payload = json.loads(_decode_body(headers, body))
    for key in golden['required_keys']:
      self.assertIn(key, payload)


if __name__ == '__main__':
  unittest.main()
