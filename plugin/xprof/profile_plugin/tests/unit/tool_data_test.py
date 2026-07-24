"""Unit tests for ToolDataService convert orchestration."""

from __future__ import annotations

import os
import tempfile
import threading
import types
import unittest
from unittest import mock

from etils import epath

from xprof.profile_plugin.constants import CACHE_VERSION_FILE
from xprof.profile_plugin.models import ToolRequest
from xprof.profile_plugin.services.hosts import HostSelector
from xprof.profile_plugin.services.sessions import SessionResolver
from xprof.profile_plugin.services.tool_data import ToolDataService


class _FakeConvert:
  def __init__(self, calls: list):
    self._calls = calls

  def xspace_to_tool_data(self, paths, tool, params):
    self._calls.append((tool, list(params.get('hosts', [])), params, list(paths)))
    return '{"ok":true}', 'application/json'

  def xspace_to_tool_names(self, paths):
    return []

  def json_to_csv_string(self, data):
    return ''


class _FakeFs:
  def __init__(self, basenames: list[str] | None = None):
    self._basenames = basenames

  def get_xplane_basenames(self, dir_path: str) -> list[str]:
    if self._basenames is not None:
      return list(self._basenames)
    # Discover real xplane basenames under dir_path.
    try:
      names = [
          n
          for n in os.listdir(dir_path)
          if n.endswith('.xplane.pb') or n.endswith('.xplane.riegeli')
      ]
    except OSError:
      return []
    return names

  def dir_has_xplane_files(self, path: str) -> bool:
    return bool(self.get_xplane_basenames(path))

  def get_session_paths(self, path: str) -> dict[str, str]:
    return {}


class _FsFactory:
  def __init__(self, fs: _FakeFs | None = None):
    self._fs = fs or _FakeFs()

  def get(self, path: str) -> _FakeFs:
    return self._fs


class ToolDataServiceTest(unittest.TestCase):

  def setUp(self):
    self._td = tempfile.TemporaryDirectory()
    self.root = self._td.name
    self.session = os.path.join(self.root, 'my_session')
    os.makedirs(self.session)
    for host in ('host0', 'host1'):
      with open(os.path.join(self.session, f'{host}.xplane.pb'), 'wb') as f:
        f.write(b'x')
    self.version = types.SimpleNamespace(__version__='9.9.9')
    self.calls: list = []
    self.sessions = SessionResolver(
        epath_module=epath,
        fs_factory=lambda p: _FakeFs(),
    )
    self.service = ToolDataService(
        convert=_FakeConvert(self.calls),
        sessions=self.sessions,
        hosts=HostSelector(),
        version=self.version,
        epath_module=epath,
        fs_factory=_FsFactory(),
    )

  def tearDown(self):
    self._td.cleanup()

  def _req(
      self,
      tool: str = 'overview_page',
      *,
      host: str | None = 'host0',
      hosts: tuple[str, ...] = (),
      use_saved_result: bool = True,
      raw_args: dict | None = None,
  ) -> ToolRequest:
    args = dict(raw_args or {})
    args.setdefault('tag', tool)
    args.setdefault('run', 'my_session')
    if host is not None:
      args.setdefault('host', host)
    return ToolRequest(
        run='my_session',
        tool=tool,
        host=host,
        hosts=hosts,
        use_saved_result=use_saved_result,
        raw_args=args,
    )

  def test_passes_hosts_and_tool_to_convert(self):
    req = self._req('overview_page', host='host0')
    result = self.service.get_tool_data(
        req, session_path=self.session
    )
    self.assertIsNotNone(result.data)
    self.assertIn('ok', result.data)
    self.assertEqual(result.content_type, 'application/json')
    self.assertEqual(len(self.calls), 1)
    tool, hosts, params, paths = self.calls[0]
    self.assertEqual(tool, 'overview_page')
    self.assertEqual(hosts, ['host0'])
    self.assertEqual(params['host'], 'host0')
    self.assertEqual(len(paths), 1)
    self.assertTrue(str(paths[0]).endswith('host0.xplane.pb'))

  def test_all_hosts_passes_both(self):
    from xprof.profile_plugin.constants import ALL_HOSTS

    req = self._req('overview_page', host=ALL_HOSTS)
    result = self.service.get_tool_data(req, session_path=self.session)
    self.assertIn('ok', result.data)
    _, hosts, _, paths = self.calls[0]
    self.assertEqual(set(hosts), {'host0', 'host1'})
    self.assertEqual(len(paths), 2)

  def test_unknown_tool_returns_none(self):
    req = self._req('not_a_real_tool', host='host0')
    result = self.service.get_tool_data(req, session_path=self.session)
    self.assertIsNone(result.data)
    self.assertEqual(result.content_type, 'application/json')
    self.assertEqual(self.calls, [])

  def test_counter_names_only_bypasses_convert(self):
    import sys

    fake = types.ModuleType('xprof.convert.counter_extractor')
    fake.get_all_counters = lambda device_type: ['c1', 'c2']
    convert_mod = sys.modules.get(
        'xprof.convert', types.ModuleType('xprof.convert')
    )
    convert_mod.counter_extractor = fake
    with mock.patch.dict(
        sys.modules,
        {
            'xprof.convert': convert_mod,
            'xprof.convert.counter_extractor': fake,
        },
    ):
      req = self._req(
          'perf_counters',
          host=None,
          raw_args={'names_only': '1', 'device_type': 'tpu'},
      )
      result = self.service.get_tool_data(req, session_path=self.session)
    self.assertEqual(self.calls, [])
    self.assertEqual(result.content_type, 'application/json')
    self.assertIn('c1', result.data)

  def test_writes_cache_version_when_not_using_saved(self):
    # No cache version file → should_use_saved_result forces False.
    req = self._req('overview_page', host='host0', use_saved_result=True)
    result = self.service.get_tool_data(req, session_path=self.session)
    self.assertIn('ok', result.data)
    version_path = os.path.join(self.session, CACHE_VERSION_FILE)
    self.assertTrue(os.path.isfile(version_path))
    with open(version_path) as f:
      self.assertEqual(f.read(), '9.9.9')
    # use_saved_result in convert params should be False after policy.
    _, _, params, _ = self.calls[0]
    self.assertFalse(params['use_saved_result'])

  def test_skips_cache_version_write_when_saved_ok(self):
    version_path = os.path.join(self.session, CACHE_VERSION_FILE)
    with open(version_path, 'w') as f:
      f.write('9.9.9')
    req = self._req('overview_page', host='host0', use_saved_result=True)
    self.service.get_tool_data(req, session_path=self.session)
    _, _, params, _ = self.calls[0]
    self.assertTrue(params['use_saved_result'])
    # File still exists with same content (not rewritten needed, but ok if same).
    with open(version_path) as f:
      self.assertEqual(f.read(), '9.9.9')

  def test_missing_host_raises(self):
    req = self._req('kernel_stats', host='nope')
    with self.assertRaises(FileNotFoundError):
      self.service.get_tool_data(req, session_path=self.session)

  def test_convert_attribute_error_wrapped(self):
    class BoomConvert(_FakeConvert):
      def xspace_to_tool_data(self, paths, tool, params):
        raise AttributeError('no convert')

    self.service = ToolDataService(
        convert=BoomConvert(self.calls),
        sessions=self.sessions,
        hosts=HostSelector(),
        version=self.version,
        epath_module=epath,
        fs_factory=_FsFactory(),
    )
    req = self._req('overview_page', host='host0')
    with self.assertRaises(AttributeError) as ctx:
      self.service.get_tool_data(req, session_path=self.session)
    self.assertIn('Error generating analysis results', str(ctx.exception))


if __name__ == '__main__':
  unittest.main()
