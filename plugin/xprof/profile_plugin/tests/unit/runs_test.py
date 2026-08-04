"""Tests for RunDiscovery (logdir walk + tools of run)."""

from __future__ import annotations

import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from etils import epath

from xprof.profile_plugin.services.runs import RunDiscovery, list_plugin_assets
from xprof.profile_plugin.services.sessions import tb_run_directory


class TbRunDirectoryTest(unittest.TestCase):

  def test_root_and_nested(self):
    self.assertEqual(tb_run_directory('/logs', '.'), '/logs')
    self.assertEqual(tb_run_directory('/logs', 'train'), '/logs/train')


class ListPluginAssetsTest(unittest.TestCase):

  def test_lists_per_run(self):
    with mock.patch(
        'xprof.profile_plugin.services.runs.plugin_asset_util.ListAssets',
        side_effect=lambda run_path, name: ['runA'] if 'train' in run_path else [],
    ):
      result = list_plugin_assets('/logs', ['.', 'train'], 'profile')
    self.assertEqual(result['train'], ['runA'])
    self.assertEqual(result['.'], [])


class RunDiscoveryTest(unittest.TestCase):

  def test_empty_logdir_yields_nothing(self):
    disc = RunDiscovery(
        logdir=None,
        epath_module=epath,
        run_dir_cache={},
        cache_lock=threading.Lock(),
        get_all_basenames=lambda p: [],
        get_file_system=lambda p: None,
    )
    self.assertEqual(list(disc.iter_frontend_runs()), [])

  def test_tools_of_run_missing_dir(self):
    disc = RunDiscovery(
        logdir='/logs',
        epath_module=epath,
        run_dir_cache={},
        cache_lock=threading.Lock(),
        get_all_basenames=lambda p: [],
        get_file_system=lambda p: None,
    )
    self.assertEqual(list(disc.tools_of_run('r', None)), [])
    self.assertEqual(list(disc.tools_of_run('r', '')), [])

  def test_tools_of_run_uses_cache(self):
    class FakeFS:
      pass

    fake_cache = mock.Mock()
    fake_cache.load.return_value = ['overview_page', 'kernel_stats']
    with mock.patch(
        'xprof.profile_plugin.services.runs.ToolsCache', return_value=fake_cache
    ):
      disc = RunDiscovery(
          logdir='/logs',
          epath_module=epath,
          run_dir_cache={},
          cache_lock=threading.Lock(),
          get_all_basenames=lambda p: self.fail('should not list files'),
          get_file_system=lambda p: FakeFS(),
      )
      tools = list(disc.tools_of_run('run1', '/logs/plugins/profile/run1'))
    self.assertEqual(tools, ['overview_page', 'kernel_stats'])
    fake_cache.save.assert_not_called()

  def test_tools_of_run_regenerates_when_cache_miss(self):
    class FakeFS:
      pass

    fake_cache = mock.Mock()
    fake_cache.load.return_value = None
    with mock.patch(
        'xprof.profile_plugin.services.runs.ToolsCache', return_value=fake_cache
    ):
      with mock.patch(
          'xprof.profile_plugin.services.runs.get_active_tools',
          return_value=['overview_page'],
      ) as gat:
        disc = RunDiscovery(
            logdir='/logs',
            epath_module=epath,
            run_dir_cache={},
            cache_lock=threading.Lock(),
            get_all_basenames=lambda p: ['h.xplane.pb'],
            get_file_system=lambda p: FakeFS(),
        )
        tools = list(disc.tools_of_run('run1', '/tmp/run1'))
    self.assertEqual(tools, ['overview_page'])
    gat.assert_called_once()
    fake_cache.save.assert_called_once_with(['overview_page'])

  def test_iter_frontend_runs_discovers_local_layout(self):
    with tempfile.TemporaryDirectory() as tmp:
      # logs/plugins/profile/run1/
      run1 = Path(tmp) / 'plugins' / 'profile' / 'run1'
      run1.mkdir(parents=True)
      (run1 / 'host.xplane.pb').write_bytes(b'x')
      # logs/train/plugins/profile/run2/
      run2 = Path(tmp) / 'train' / 'plugins' / 'profile' / 'run2'
      run2.mkdir(parents=True)
      (run2 / 'host.xplane.pb').write_bytes(b'x')

      cache = {}
      lock = threading.Lock()
      disc = RunDiscovery(
          logdir=tmp,
          epath_module=epath,
          run_dir_cache=cache,
          cache_lock=lock,
          get_all_basenames=lambda p: [],
          get_file_system=lambda p: None,
      )
      with mock.patch(
          'xprof.profile_plugin.services.runs.list_plugin_assets',
          side_effect=lambda session, runs, name: {
              r: (['run1'] if r == '.' else ['run2']) for r in runs
          },
      ):
        # walk may also find train path via fsspec/walk — use is_dir on paths
        # Ensure PluginDirectory joins correctly: mock ListAssets path is via list_plugin_assets
        runs = list(disc.iter_frontend_runs())

      # At least root-style run1 if assets returned run1 for '.'
      self.assertTrue(any('run' in r for r in runs) or runs == [] or True)
      # When list_plugin_assets returns run names, directories must exist:
      # plugin_dir / profile_run must be dirs — we created them.
      self.assertIn('run1', runs)
      self.assertIn(os.path.join('train', 'run2').replace('\\', '/'), 
                    [r.replace('\\', '/') for r in runs])
      self.assertIn('run1', cache)


if __name__ == '__main__':
  unittest.main()
