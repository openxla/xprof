"""Unit tests for ToolsCache (moved package location)."""

from __future__ import annotations

import json
import os
import unittest

from etils import epath

from xprof import profile_io
from xprof.profile_plugin.cache.tools_cache import ToolsCache
from xprof.profile_plugin.tools.registry import TOOLS


class ToolsCacheUnitTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.temp_dir = self.create_temp_dir()
    self.profile_run_dir = epath.Path(self.temp_dir)
    self.cache = ToolsCache(
        self.profile_run_dir,
        profile_io.get_file_system(str(self.profile_run_dir)),
    )

  def create_temp_dir(self):
    import tempfile
    self._td = tempfile.TemporaryDirectory()
    return self._td.name

  def tearDown(self):
    self._td.cleanup()
    super().tearDown()

  def _create_xplane(self, host='host1', content='x'):
    path = self.profile_run_dir / f'{host}.{TOOLS["xplane"]}'
    with path.open('w') as f:
      f.write(content)
    return path

  def test_save_load_roundtrip(self):
    self._create_xplane('h1')
    self._create_xplane('h2')
    tools = ['overview_page', 'trace_viewer@']
    self.cache.save(tools)
    self.assertEqual(self.cache.load(), tools)

  def test_load_missing(self):
    self.assertIsNone(self.cache.load())

  def test_version_mismatch_invalidates(self):
    self._create_xplane()
    self.cache.save(['overview_page'])
    cache_file = self.profile_run_dir / ToolsCache.CACHE_FILE_NAME
    with cache_file.open('r') as f:
      data = json.load(f)
    data['version'] = ToolsCache.CACHE_VERSION - 1
    with cache_file.open('w') as f:
      json.dump(data, f)
    self.assertIsNone(self.cache.load())
    self.assertFalse(cache_file.exists())

  def test_file_change_invalidates(self):
    f1 = self._create_xplane('h1')
    self.cache.save(['overview_page'])
    st = os.stat(str(f1))
    os.utime(str(f1), ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    self.assertIsNone(self.cache.load())


if __name__ == '__main__':
  unittest.main()
