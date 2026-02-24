# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for the GCS and Local file system abstraction in profile_io.py."""

import os
from typing import Any, Iterable, Optional
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized

from xprof import profile_io


class MockIterator:
  """Mock for GCS iterator from list_blobs."""

  def __init__(
      self, items: Iterable[Any], prefixes: Optional[list[str]] = None
  ):
    """Initializes the instance."""
    self.items = items
    self.prefixes = prefixes or []

  def __iter__(self):
    """Returns an iterator for the items."""
    return iter(self.items)


class GetFileSystemTest(parameterized.TestCase):
  """Tests for the get_file_system factory function."""

  def setUp(self):
    """Initializes the test environment."""
    super().setUp()
    self.mock_storage = self.enter_context(
        mock.patch.object(profile_io, 'storage', autospec=True)
    )

  @parameterized.named_parameters(
      ('gcs', 'gs://my-bucket/path', profile_io.GcsFileSystem),
      ('local', '/my/local/path', profile_io.LocalFileSystem),
  )
  def test_get_file_system(self, path, expected_fs_type):
    """Checks that paths return the correct FileSystem type."""
    fs = profile_io.get_file_system(path)
    self.assertIsInstance(fs, expected_fs_type)


class LocalFileSystemTest(absltest.TestCase):
  """Tests for LocalFileSystem."""

  def setUp(self):
    """Initializes the test environment."""
    super().setUp()
    self.temp_dir = self.create_tempdir()
    self.fs = profile_io.LocalFileSystem()

  def test_get_xplane_basenames(self):
    """Checks that it returns only .xplane.pb basenames."""
    self.temp_dir.create_file('1.xplane.pb', content='test')
    self.temp_dir.create_file('2.txt', content='test2')
    basenames = self.fs.get_xplane_basenames(self.temp_dir.full_path)
    self.assertEqual(basenames, ['1.xplane.pb'])

  def test_dir_has_xplane_files(self):
    """Checks for the presence of .xplane.pb files."""
    self.assertFalse(self.fs.dir_has_xplane_files(self.temp_dir.full_path))
    self.temp_dir.create_file('1.xplane.pb', content='test')
    self.assertTrue(self.fs.dir_has_xplane_files(self.temp_dir.full_path))

  def test_get_all_basenames(self):
    """Checks that it returns all basenames in the directory."""
    self.temp_dir.create_file('1.xplane.pb', content='test')
    self.temp_dir.create_file('2.txt', content='test2')
    basenames = self.fs.get_all_basenames(self.temp_dir.full_path)
    self.assertCountEqual(basenames, ['1.xplane.pb', '2.txt'])

  def test_get_session_paths(self):
    """Checks that session paths are correctly detected."""
    session1 = self.temp_dir.mkdir('session1')
    session1.create_file('1.xplane.pb', content='test')
    self.temp_dir.mkdir('session2')  # Empty session
    sessions = self.fs.get_session_paths(self.temp_dir.full_path)
    self.assertLen(sessions, 1)
    self.assertIn('session1', sessions)

  def test_read_write_json(self):
    """Checks reading and writing JSON files."""
    json_path = os.path.join(self.temp_dir.full_path, 'data.json')
    data = {'key': 'value'}
    self.fs.write_json(json_path, data)
    read_data = self.fs.read_json(json_path)
    self.assertEqual(read_data, data)

  def test_read_write_text(self):
    """Checks reading and writing text files."""
    text_path = os.path.join(self.temp_dir.full_path, 'data.txt')
    data = 'hello world'
    self.fs.write_text(text_path, data)
    read_data = self.fs.read_text(text_path)
    self.assertEqual(read_data, data)

  def test_delete_file(self):
    """Checks deleting a file."""
    file_path = self.temp_dir.create_file('delete_me.txt').full_path
    self.assertTrue(os.path.exists(file_path))
    self.fs.delete_file(file_path)
    self.assertFalse(os.path.exists(file_path))


class GcsFileSystemTest(absltest.TestCase):
  """Tests for GcsFileSystem."""

  def setUp(self):
    """Initializes the test environment."""
    super().setUp()
    self.mock_storage = self.enter_context(
        mock.patch.object(profile_io, 'storage', autospec=True)
    )
    self.mock_gcs_exceptions = self.enter_context(
        mock.patch.object(profile_io, 'gcs_exceptions', autospec=True)
    )
    self.mock_client = self.mock_storage.Client.return_value
    self.fs = profile_io.GcsFileSystem()

  def test_get_xplane_basenames(self):
    """Checks that it returns only .xplane.pb basenames from GCS."""
    mock_blob1 = mock.MagicMock()
    mock_blob1.name = 'path/to/1.xplane.pb'
    mock_blob2 = mock.MagicMock()
    mock_blob2.name = 'path/to/2.txt'

    self.mock_client.list_blobs.return_value = MockIterator(
        [mock_blob1, mock_blob2]
    )

    basenames = self.fs.get_xplane_basenames('gs://bucket/path')
    self.assertEqual(basenames, ['1.xplane.pb'])

  def test_read_json(self):
    """Checks reading JSON from GCS."""
    blob_mock = self.mock_storage.Blob.from_string.return_value
    blob_mock.download_as_bytes.return_value = b'{"hello": "world"}'
    data = self.fs.read_json('gs://bucket/path/data.json')
    self.assertEqual(data, {'hello': 'world'})


if __name__ == '__main__':
  absltest.main()
