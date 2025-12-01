"""Tests for ToolsCache class."""

import json
import os

from etils import epath

from absl.testing import absltest
from xprof import profile_plugin

_TOOLS_1 = ('tool1',)
_TOOLS_2 = ('tool1', 'tool2')
_XPLANE_FILE_1 = f'host1.{profile_plugin.TOOLS["xplane"]}'
_XPLANE_FILE_2 = f'host2.{profile_plugin.TOOLS["xplane"]}'
_CORRUPTED_CACHE_CONTENT = 'invalid json'


class ToolsCacheTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.temp_dir = self.create_tempdir().full_path
    self.profile_run_dir = epath.Path(self.temp_dir)
    self.cache = profile_plugin.ToolsCache(self.profile_run_dir)
    self.cache_file = (
        self.profile_run_dir / profile_plugin.ToolsCache.CACHE_FILE_NAME
    )

  def _create_xplane_file(self, filename, content=''):
    file_path = self.profile_run_dir / filename
    with file_path.open('w') as f:
      f.write(content)
    return file_path

  def test_initialization(self):
    self.assertEqual(self.cache._profile_run_dir, self.profile_run_dir)
    self.assertEqual(self.cache._cache_file, self.cache_file)
    self.assertFalse(self.cache_file.exists())

  def test_save_and_load(self):
    self._create_xplane_file(_XPLANE_FILE_1)
    self._create_xplane_file(_XPLANE_FILE_2)

    self.cache.save(_TOOLS_2)
    self.assertTrue(self.cache_file.exists())

    loaded_tools = self.cache.load()
    self.assertEqual(loaded_tools, list(_TOOLS_2))

  def test_load_no_cache_file(self):
    self.assertIsNone(self.cache.load())

  def test_load_version_mismatch(self):
    self.cache.save(_TOOLS_1)
    with self.cache_file.open('r') as f:
      cache_data = json.load(f)
    cache_data['version'] = self.cache.CACHE_VERSION - 1
    with self.cache_file.open('w') as f:
      json.dump(cache_data, f)

    self.assertIsNone(self.cache.load())
    self.assertFalse(self.cache_file.exists())

  def test_load_file_mtime_changed(self):
    file1 = self._create_xplane_file(_XPLANE_FILE_1)
    self.cache.save(_TOOLS_1)

    # Manually change mtime to ensure the test is deterministic.
    st = os.stat(str(file1))
    # Add one second to the modification time to ensure it's a significant
    # change that will be picked up by filesystems with low precision.
    os.utime(str(file1), ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))

    self.assertIsNone(self.cache.load())

  def test_load_file_added(self):
    self._create_xplane_file(_XPLANE_FILE_1)
    self.cache.save(_TOOLS_1)
    self._create_xplane_file(_XPLANE_FILE_2)
    self.assertIsNone(self.cache.load())

  def test_load_file_removed(self):
    file1 = self._create_xplane_file(_XPLANE_FILE_1)
    self._create_xplane_file(_XPLANE_FILE_2)
    self.cache.save(_TOOLS_1)
    file1.unlink()
    self.assertIsNone(self.cache.load())

  def test_empty_directory(self):
    self.cache.save(_TOOLS_1)
    self.assertEqual(self.cache.load(), list(_TOOLS_1))

  def test_invalidate(self):
    self.cache.save(_TOOLS_1)
    self.assertTrue(self.cache_file.exists())
    self.cache.invalidate()
    self.assertFalse(self.cache_file.exists())

  def test_corrupted_cache_file(self):
    with self.cache_file.open('w') as f:
      f.write(_CORRUPTED_CACHE_CONTENT)
    self.assertIsNone(self.cache.load())
    self.assertFalse(self.cache_file.exists())

  def test_get_states_failed_on_save(self):
    self._create_xplane_file(_XPLANE_FILE_1)
    # Make directory unreadable to cause an OSError during file globbing.
    os.chmod(self.profile_run_dir, 0o000)
    try:
      self.cache.save(_TOOLS_1)
    finally:
      # Restore permissions so other tests and cleanup can succeed.
      os.chmod(self.profile_run_dir, 0o700)
    self.assertFalse(self.cache_file.exists())

  def test_save_load_non_existent_directory(self):
    non_existent_dir = epath.Path(self.temp_dir) / 'non_existent'
    cache = profile_plugin.ToolsCache(non_existent_dir)
    cache_file = non_existent_dir / profile_plugin.ToolsCache.CACHE_FILE_NAME

    # Test save and load on non-existent dir.
    cache.save(_TOOLS_1)
    self.assertFalse(cache_file.exists())
    self.assertIsNone(cache.load())

  def test_save_load_after_creating_directory(self):
    was_non_existent_dir = epath.Path(self.temp_dir) / 'was_non_existent'
    cache = profile_plugin.ToolsCache(was_non_existent_dir)
    cache_file = (
        was_non_existent_dir / profile_plugin.ToolsCache.CACHE_FILE_NAME
    )

    # dir does not exist initially.
    self.assertFalse(cache_file.exists())
    self.assertIsNone(cache.load())

    # Now create directory and save.
    was_non_existent_dir.mkdir()
    cache.save(_TOOLS_1)
    self.assertTrue(cache_file.exists())

    # Load should work now.
    self.assertEqual(cache.load(), list(_TOOLS_1))


if __name__ == '__main__':
  absltest.main()
