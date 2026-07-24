"""Integration-style tests for ProfilePlugin with mocked conversion."""

from __future__ import annotations

import concurrent.futures
import gzip
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from etils import epath
from werkzeug import test as werkzeug_test
from werkzeug import wrappers

from xprof import profile_plugin
from xprof.profile_plugin.plugin import ProfilePlugin
from xprof.standalone.tensorboard_shim import base_plugin
from xprof.standalone.tensorboard_shim import data_provider
from xprof.standalone.tensorboard_shim import plugin_asset_util
from xprof.standalone.tensorboard_shim import plugin_event_multiplexer


class _FakeFlags:
  def __init__(self, logdir, master_tpu_unsecure_channel=''):
    self.logdir = logdir
    self.master_tpu_unsecure_channel = master_tpu_unsecure_channel


def _make_plugin(logdir, xspace_fn=None, version='9.9.9'):
  multiplexer = plugin_event_multiplexer.EventMultiplexer()
  multiplexer.AddRunsFromDirectory(logdir)
  context = base_plugin.TBContext(
      logdir=logdir,
      multiplexer=multiplexer,
      data_provider=data_provider.MultiplexerDataProvider(multiplexer, logdir),
      flags=_FakeFlags(logdir),
  )
  return ProfilePlugin(
      context,
      epath_module=epath,
      xspace_to_tool_data_fn=xspace_fn
      or (lambda paths, tool, params: (json.dumps({'ok': True, 'tool': tool}), 'application/json')),
      version_module=SimpleNamespace(__version__=version),
      cache_generation_executor=concurrent.futures.ThreadPoolExecutor(max_workers=1),
  )


class ProfilePluginIntegrationTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self._td = tempfile.TemporaryDirectory()
    self.logdir = self._td.name
    plugin_dir = plugin_asset_util.PluginDirectory(
        self.logdir, profile_plugin.PLUGIN_NAME
    )
    self.session = 'session_a'
    self.run_dir = os.path.join(plugin_dir, self.session)
    os.makedirs(self.run_dir)
    # Two hosts
    for host in ('host0', 'host1'):
      path = os.path.join(self.run_dir, f'{host}.xplane.pb')
      with open(path, 'wb') as f:
        f.write(b'fake')

  def tearDown(self):
    self._td.cleanup()
    super().tearDown()

  def test_public_exports(self):
    self.assertTrue(hasattr(profile_plugin, 'ProfilePlugin'))
    self.assertTrue(hasattr(profile_plugin, 'ToolsCache'))
    self.assertTrue(hasattr(profile_plugin, 'make_filename'))
    self.assertTrue(hasattr(profile_plugin, 'XPLANE_TOOLS'))
    self.assertEqual(
        profile_plugin.make_filename('h', 'overview_page'), 'h.xplane.pb'
    )

  def test_generate_runs_and_runs_imp(self):
    plugin = _make_plugin(self.logdir)
    runs = list(plugin.generate_runs())
    self.assertIn(self.session, runs)
    self.assertEqual(plugin.runs_imp(), sorted(runs, reverse=True))

  def test_session_path_override(self):
    plugin = _make_plugin(self.logdir)
    req = wrappers.Request({})
    req.args = {'session_path': self.run_dir}
    mapping = plugin._session_dir_by_run_name_from_request(req)
    self.assertEqual(mapping, {self.session: self.run_dir})
    self.assertEqual(plugin.runs_imp(req), [self.session])

  @mock.patch(
      'xprof.profile_plugin.tools.catalog.convert',
      create=True,
  )
  def test_run_tools_uses_xspace_tool_names(self, _unused):
    # Patch the convert function used via lazy import path.
    with mock.patch(
        'xprof.convert.raw_to_tool_data.xspace_to_tool_names',
        return_value=['overview_page', 'trace_viewer@', 'kernel_stats'],
    ):
      plugin = _make_plugin(self.logdir)
      tools = plugin.run_tools_imp(self.session)
    self.assertEqual(tools[0], 'overview_page')
    self.assertIn('trace_viewer@', tools)
    self.assertNotIn('trace_viewer', tools)  # overridden by @

  def test_hosts_all_hosts_only_tool(self):
    plugin = _make_plugin(self.logdir)
    hosts = plugin.host_impl(self.session, 'overview_page')
    self.assertEqual(hosts, [{'hostname': 'ALL_HOSTS'}])

  def test_hosts_supported_tool(self):
    plugin = _make_plugin(self.logdir)
    hosts = plugin.host_impl(self.session, 'kernel_stats')
    names = {h['hostname'] for h in hosts}
    self.assertEqual(names, {'host0', 'host1', 'ALL_HOSTS'})

  def test_data_impl_calls_converter_with_hosts(self):
    calls = []

    def fake_convert(paths, tool, params):
      calls.append((list(map(str, paths)), tool, dict(params)))
      return json.dumps({'tool': tool}), 'application/json'

    plugin = _make_plugin(self.logdir, xspace_fn=fake_convert)
    req = wrappers.Request({})
    req.args = {
        'run': self.session,
        'tag': 'overview_page',
        'host': 'ALL_HOSTS',
        'use_saved_result': 'true',
    }
    data, content_type, _ = plugin.data_impl(req)
    self.assertEqual(content_type, 'application/json')
    self.assertIsNotNone(data)
    self.assertEqual(len(calls), 1)
    paths, tool, params = calls[0]
    self.assertEqual(tool, 'overview_page')
    self.assertEqual(len(paths), 2)
    self.assertEqual(set(params['hosts']), {'host0', 'host1'})
    # Missing cache version forces recompute path write
    self.assertFalse(params['use_saved_result'])
    self.assertTrue(
        os.path.exists(os.path.join(self.run_dir, profile_plugin.CACHE_VERSION_FILE))
    )

  def test_data_route_404_and_gzip(self):
    def fake_convert(paths, tool, params):
      return None, 'application/json'

    plugin = _make_plugin(self.logdir, xspace_fn=fake_convert)
    client = werkzeug_test.Client(wrappers.Request.application(plugin.data_route))
    # Actually need full app map
    apps = plugin.get_plugin_apps()
    app = apps[profile_plugin.DATA_ROUTE]
    # Use werkzeug client with the data route app
    environ_base = {
        'QUERY_STRING': f'run={self.session}&tag=overview_page&host=ALL_HOSTS',
    }
    # Build via wrappers
    from werkzeug.test import EnvironBuilder
    builder = EnvironBuilder(
        path='/data',
        query_string={
            'run': self.session,
            'tag': 'overview_page',
            'host': 'ALL_HOSTS',
        },
    )
    env = builder.get_environ()

    def start_response(status, headers, exc_info=None):
      start_response.status = status
      start_response.headers = headers

    body = b''.join(app(env, start_response))
    self.assertTrue(start_response.status.startswith('404'))

    # Success path
    def ok_convert(paths, tool, params):
      return '{"x":1}', 'application/json'

    plugin2 = _make_plugin(self.logdir, xspace_fn=ok_convert)
    app2 = plugin2.get_plugin_apps()[profile_plugin.DATA_ROUTE]
    body2 = b''.join(app2(env, start_response))
    self.assertTrue(start_response.status.startswith('200'))
    self.assertEqual(json.loads(gzip.decompress(body2)), {'x': 1})

  def test_config_route(self):
    plugin = _make_plugin(self.logdir)
    app = plugin.get_plugin_apps()[profile_plugin.CONFIG_ROUTE]
    from werkzeug.test import EnvironBuilder
    env = EnvironBuilder(path='/config').get_environ()
    status_holder = {}

    def start_response(status, headers, exc_info=None):
      status_holder['status'] = status
      status_holder['headers'] = headers

    body = b''.join(app(env, start_response))
    data = json.loads(gzip.decompress(body))
    self.assertIn('hideCaptureProfileButton', data)
    self.assertIn('srcPathPrefix', data)

  def test_generate_cache_accepts_post(self):
    plugin = _make_plugin(self.logdir)
    with mock.patch(
        'xprof.convert.raw_to_tool_data.xspace_to_tool_names',
        return_value=['overview_page', 'trace_viewer@'],
    ):
      app = plugin.get_plugin_apps()[profile_plugin.GENERATE_CACHE_ROUTE]
      from werkzeug.test import EnvironBuilder
      env = EnvironBuilder(
          method='POST',
          path='/generate_cache',
          query_string={'session_path': self.run_dir, 'tools': 'overview_page'},
      ).get_environ()
      status_holder = {}

      def start_response(status, headers, exc_info=None):
        status_holder['status'] = status

      body = b''.join(app(env, start_response))
      self.assertTrue(
          status_holder['status'].startswith('202'), status_holder['status']
      )
      payload = json.loads(gzip.decompress(body))
      self.assertEqual(payload['status'], 'ACCEPTED')

  def test_static_index_readable(self):
    plugin = _make_plugin(self.logdir)
    # static files exist under plugin/xprof/static
    contents = plugin._read_static_file_impl('index.html')
    self.assertIsInstance(contents, (bytes, bytearray))
    self.assertTrue(len(contents) > 0)


if __name__ == '__main__':
  unittest.main()
