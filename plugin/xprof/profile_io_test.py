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

import json
import os
from typing import Any, Iterable, Optional
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized

from xprof import profile_io


class MockNotFoundError(Exception):
  pass


class MockGoogleAPICallError(Exception):
  pass


class MockIterator:
  """Mock for GCS iterator from list_blobs."""

  def __init__(
      self, items: Iterable[Any], prefixes: Optional[list[str]] = None
  ):
    self.items = items
    self.prefixes = prefixes or []

  def __iter__(self):
    return iter(self.items)


class GetFileSystemTest(parameterized.TestCase):
  """Tests for the get_file_system factory function."""

  def setUp(self):
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

  def test_get_file_system_gcs_no_storage(self):
    """Checks that gs:// paths return LocalFileSystem if storage is not available."""
    with mock.patch.object(profile_io, 'storage', None):
      fs = profile_io.get_file_system('gs://my-bucket/path')
      self.assertIsInstance(fs, profile_io.LocalFileSystem)


class LocalFileSystemTest(absltest.TestCase):
  """Tests for LocalFileSystem."""

  def setUp(self):
    super().setUp()
    self.temp_dir = self.create_tempdir()
    self.fs = profile_io.LocalFileSystem()

  def test_get_xplane_basenames(self):
    """Checks that it returns only .xplane.pb and .xplane.riegeli basenames."""
    self.temp_dir.create_file('1.xplane.pb', content='test')
    self.temp_dir.create_file('2.txt', content='test2')
    self.temp_dir.create_file('3.xplane.riegeli', content='test3')
    basenames = self.fs.get_xplane_basenames(self.temp_dir.full_path)
    self.assertCountEqual(basenames, ['1.xplane.pb', '3.xplane.riegeli'])

  def test_get_xplane_basenames_oserror(self):
    """Checks that OSError in glob is handled gracefully and returns empty list."""
    mock_epath = mock.MagicMock()
    mock_path = mock_epath.Path.return_value
    mock_path.glob.side_effect = OSError('permission denied')
    fs = profile_io.LocalFileSystem(epath_module=mock_epath)
    self.assertEqual(fs.get_xplane_basenames('/some/path'), [])

  def test_dir_has_xplane_files(self):
    """Checks for the presence of .xplane.pb files."""
    self.assertFalse(self.fs.dir_has_xplane_files(self.temp_dir.full_path))
    self.temp_dir.create_file('1.xplane.pb', content='test')
    self.assertTrue(self.fs.dir_has_xplane_files(self.temp_dir.full_path))

  def test_dir_has_xplane_files_oserror(self):
    """Checks that OSError in dir_has_xplane_files is handled gracefully."""
    mock_epath = mock.MagicMock()
    mock_path = mock_epath.Path.return_value
    mock_path.glob.side_effect = OSError('permission denied')
    fs = profile_io.LocalFileSystem(epath_module=mock_epath)
    self.assertFalse(fs.dir_has_xplane_files('/some/path'))

  def test_get_all_basenames(self):
    """Checks that it returns all basenames in the directory."""
    self.temp_dir.create_file('1.xplane.pb', content='test')
    self.temp_dir.create_file('2.txt', content='test2')
    basenames = self.fs.get_all_basenames(self.temp_dir.full_path)
    self.assertCountEqual(basenames, ['1.xplane.pb', '2.txt'])

  def test_get_all_basenames_oserror(self):
    """Checks that OSError in get_all_basenames is handled gracefully."""
    mock_epath = mock.MagicMock()
    mock_path = mock_epath.Path.return_value
    mock_path.iterdir.side_effect = OSError('permission denied')
    fs = profile_io.LocalFileSystem(epath_module=mock_epath)
    self.assertEqual(fs.get_all_basenames('/some/path'), [])

  def test_get_session_paths(self):
    """Checks that session paths are correctly detected."""
    session1 = self.temp_dir.mkdir('session1')
    session1.create_file('1.xplane.pb', content='test')
    self.temp_dir.mkdir('session2')  # Empty session
    sessions = self.fs.get_session_paths(self.temp_dir.full_path)
    self.assertLen(sessions, 1)
    self.assertIn('session1', sessions)

  def test_get_session_paths_oserror(self):
    """Checks that OSError in get_session_paths is handled gracefully."""
    mock_epath = mock.MagicMock()
    mock_path = mock_epath.Path.return_value
    mock_path.iterdir.side_effect = OSError('permission denied')
    fs = profile_io.LocalFileSystem(epath_module=mock_epath)
    self.assertEqual(fs.get_session_paths('/some/path'), {})

  def test_read_write_json(self):
    """Checks reading and writing JSON files."""
    json_path = os.path.join(self.temp_dir.full_path, 'data.json')
    data = {'key': 'value'}
    self.fs.write_json(json_path, data)
    read_data = self.fs.read_json(json_path)
    self.assertEqual(read_data, data)

  def test_read_json_decode_error(self):
    """Checks that decode error triggers deletion and returns None."""
    json_path = os.path.join(self.temp_dir.full_path, 'bad_data.json')
    self.fs.write_text(json_path, 'invalid json')
    self.assertIsNone(self.fs.read_json(json_path))
    self.assertFalse(os.path.exists(json_path))

  def test_read_json_oserror(self):
    """Checks that OSError in read_json deletes file and returns None."""
    mock_epath = mock.MagicMock()
    mock_path = mock_epath.Path.return_value
    mock_open = mock_path.open
    mock_open.side_effect = OSError('permission denied')
    fs = profile_io.LocalFileSystem(epath_module=mock_epath)
    self.assertIsNone(fs.read_json('/some/file.json'))
    mock_path.unlink.assert_called_once()

  def test_write_json_oserror(self):
    """Checks that OSError in write_json is handled without crash."""
    mock_epath = mock.MagicMock()
    mock_path = mock_epath.Path.return_value
    mock_open = mock_path.open
    mock_open.side_effect = OSError('permission denied')
    fs = profile_io.LocalFileSystem(epath_module=mock_epath)
    fs.write_json('/some/file.json', {})  # No exception raised

  def test_read_write_text(self):
    """Checks reading and writing text files."""
    text_path = os.path.join(self.temp_dir.full_path, 'data.txt')
    data = 'hello world'
    self.fs.write_text(text_path, data)
    read_data = self.fs.read_text(text_path)
    self.assertEqual(read_data, data)

  def test_read_text_not_found(self):
    """Checks that read_text returns None when file is missing."""
    self.assertIsNone(
        self.fs.read_text(os.path.join(self.temp_dir.full_path, 'missing.txt'))
    )

  def test_read_text_oserror(self):
    """Checks that OSError in read_text is handled and returns None."""
    mock_epath = mock.MagicMock()
    mock_path = mock_epath.Path.return_value
    mock_open = mock_path.open
    mock_open.side_effect = OSError('permission denied')
    fs = profile_io.LocalFileSystem(epath_module=mock_epath)
    self.assertIsNone(fs.read_text('/some/file.txt'))

  def test_write_text_oserror(self):
    """Checks that OSError in write_text is handled without crash."""
    mock_epath = mock.MagicMock()
    mock_path = mock_epath.Path.return_value
    mock_open = mock_path.open
    mock_open.side_effect = OSError('permission denied')
    fs = profile_io.LocalFileSystem(epath_module=mock_epath)
    fs.write_text('/some/file.txt', 'data')  # No exception raised

  def test_delete_file(self):
    """Checks deleting a file."""
    file_path = self.temp_dir.create_file('delete_me.txt').full_path
    self.assertTrue(os.path.exists(file_path))
    self.fs.delete_file(file_path)
    self.assertFalse(os.path.exists(file_path))

  def test_delete_file_not_found(self):
    """Checks that deleting a missing file handles FileNotFoundError."""
    self.fs.delete_file(
        os.path.join(self.temp_dir.full_path, 'missing_file.txt')
    )

  def test_delete_file_oserror(self):
    """Checks that OSError in delete_file is handled without crash."""
    mock_epath = mock.MagicMock()
    mock_path = mock_epath.Path.return_value
    mock_path.unlink.side_effect = OSError('permission denied')
    fs = profile_io.LocalFileSystem(epath_module=mock_epath)
    fs.delete_file('/some/file.txt')  # No exception raised

  def test_get_xplane_file_states_success(self):
    """Checks getting file states with mtime-size identifiers."""
    self.temp_dir.create_file('1.xplane.pb', content='test')
    self.temp_dir.create_file('2.txt', content='test2')
    self.temp_dir.create_file('3.xplane.riegeli', content='test3')
    states = self.fs.get_xplane_file_states(self.temp_dir.full_path)
    self.assertIsNotNone(states)
    self.assertIn('1.xplane.pb', states)
    self.assertIn('3.xplane.riegeli', states)
    self.assertNotIn('2.txt', states)

  def test_get_xplane_file_states_oserror_stat(self):
    """Checks handling OSError inside stat logic."""
    self.temp_dir.create_file('1.xplane.pb', content='test')
    with mock.patch('os.stat', side_effect=OSError('permission denied')):
      self.assertIsNone(self.fs.get_xplane_file_states(self.temp_dir.full_path))

  def test_get_xplane_file_states_not_found(self):
    """Checks handling FileNotFoundError inside stat logic."""
    self.temp_dir.create_file('1.xplane.pb', content='test')
    with mock.patch('os.stat', side_effect=FileNotFoundError):
      self.assertIsNone(self.fs.get_xplane_file_states(self.temp_dir.full_path))

  def test_get_xplane_file_states_oserror_glob(self):
    """Checks handling OSError inside glob logic."""
    mock_epath = mock.MagicMock()
    mock_path = mock_epath.Path.return_value
    mock_path.glob.side_effect = OSError('permission denied')
    fs = profile_io.LocalFileSystem(epath_module=mock_epath)
    self.assertIsNone(fs.get_xplane_file_states('/some/path'))


class GcsFileSystemTest(absltest.TestCase):
  """Tests for GcsFileSystem."""

  def setUp(self):
    super().setUp()
    self.mock_storage = self.enter_context(
        mock.patch.object(profile_io, 'storage', autospec=True)
    )
    self.mock_gcs_exceptions = self.enter_context(
        mock.patch.object(profile_io, 'gcs_exceptions', autospec=True)
    )
    # Override exception classes with real ones so `except` clauses work.
    self.mock_gcs_exceptions.NotFound = MockNotFoundError
    self.mock_gcs_exceptions.GoogleAPICallError = MockGoogleAPICallError
    self.mock_client = mock.MagicMock()
    self.fs = profile_io.GcsFileSystem(storage_client=self.mock_client)

  def test_get_xplane_basenames(self):
    """Checks that it returns only .xplane.pb and .xplane.riegeli basenames from GCS."""
    mock_blob1 = mock.MagicMock()
    mock_blob1.name = 'path/to/1.xplane.pb'
    mock_blob2 = mock.MagicMock()
    mock_blob2.name = 'path/to/2.txt'
    mock_blob3 = mock.MagicMock()
    mock_blob3.name = 'path/to/3.xplane.riegeli'

    self.mock_client.list_blobs.return_value = MockIterator(
        [mock_blob1, mock_blob2, mock_blob3]
    )

    basenames = self.fs.get_xplane_basenames('gs://bucket/path')
    self.assertCountEqual(basenames, ['1.xplane.pb', '3.xplane.riegeli'])

  def test_get_xplane_basenames_exception(self):
    """Checks exception handling in get_xplane_basenames."""
    self.mock_client.list_blobs.side_effect = Exception('network error')
    self.assertEqual(self.fs.get_xplane_basenames('gs://bucket/path'), [])

  def test_get_xplane_file_states_success(self):
    """Checks getting file states with md5 hash identification."""
    mock_blob1 = mock.MagicMock()
    mock_blob1.name = 'path/to/1.xplane.pb'
    mock_blob1.md5_hash = 'abcdef'
    mock_blob2 = mock.MagicMock()
    mock_blob2.name = 'path/to/2.txt'  # should be skipped

    self.mock_client.list_blobs.return_value = MockIterator(
        [mock_blob1, mock_blob2]
    )

    states = self.fs.get_xplane_file_states('gs://bucket/path')
    self.assertEqual(states, {'1.xplane.pb': 'abcdef'})

  def test_get_xplane_file_states_missing_hash(self):
    """Checks handling missing md5 in GCS blobs."""
    mock_blob = mock.MagicMock()
    mock_blob.name = 'path/to/1.xplane.pb'
    mock_blob.md5_hash = None

    self.mock_client.list_blobs.return_value = MockIterator([mock_blob])

    self.assertIsNone(self.fs.get_xplane_file_states('gs://bucket/path'))

  def test_get_xplane_file_states_exception(self):
    """Checks Exception handling in get_xplane_file_states."""
    self.mock_client.list_blobs.side_effect = Exception('network error')
    self.assertIsNone(self.fs.get_xplane_file_states('gs://bucket/path'))

  def test_dir_has_xplane_files(self):
    """Checks checking dir_has_xplane_files on GCS."""
    mock_blob = mock.MagicMock()
    mock_blob.name = 'path/to/1.xplane.pb'
    self.mock_client.list_blobs.return_value = MockIterator([mock_blob])

    self.assertTrue(self.fs.dir_has_xplane_files('gs://bucket/path'))

    mock_blob.name = 'path/to/2.txt'
    self.mock_client.list_blobs.return_value = MockIterator([mock_blob])
    self.assertFalse(self.fs.dir_has_xplane_files('gs://bucket/path'))

  def test_dir_has_xplane_files_exception(self):
    """Checks exception handling in dir_has_xplane_files."""
    self.mock_client.list_blobs.side_effect = Exception('network error')
    self.assertFalse(self.fs.dir_has_xplane_files('gs://bucket/path'))

  def test_get_all_basenames(self):
    """Checks getting all basenames from GCS."""
    mock_blob = mock.MagicMock()
    mock_blob.name = 'path/to/1.txt'
    self.mock_client.list_blobs.return_value = MockIterator([mock_blob])

    self.assertCountEqual(
        self.fs.get_all_basenames('gs://bucket/path'), ['1.txt']
    )

  def test_get_all_basenames_exception(self):
    """Checks Exception handling in get_all_basenames."""
    self.mock_client.list_blobs.side_effect = Exception('network error')
    self.assertEqual(self.fs.get_all_basenames('gs://bucket/path'), [])

  def test_get_session_paths(self):
    """Checks getting session paths on GCS."""
    mock_blob1 = mock.MagicMock()
    mock_blob1.name = 'path/session1/1.xplane.pb'
    self.mock_client.list_blobs.return_value = MockIterator(
        [mock_blob1], prefixes=['session1/', 'session2/']
    )

    with mock.patch.object(self.fs, 'dir_has_xplane_files') as mock_has_files:
      mock_has_files.side_effect = lambda path: 'session1' in path
      paths = self.fs.get_session_paths('gs://bucket/path')
      self.assertEqual(paths, {'session1': 'gs://bucket/session1/'})

  def test_get_session_paths_no_bucket(self):
    """Checks handling missing bucket_name inside get_session_paths."""
    self.mock_client.list_blobs.return_value = MockIterator([], prefixes=[])
    paths = self.fs.get_session_paths('gs://')
    self.assertEqual(paths, {})

  def test_get_session_paths_exception(self):
    """Checks Exception handling in get_session_paths."""
    self.mock_client.list_blobs.side_effect = Exception('network error')
    self.assertEqual(self.fs.get_session_paths('gs://bucket/path'), {})

  def test_read_json(self):
    """Checks reading JSON from GCS."""
    blob_mock = self.mock_storage.Blob.from_string.return_value
    blob_mock.download_as_bytes.return_value = b'{"hello": "world"}'
    data = self.fs.read_json('gs://bucket/path/data.json')
    self.assertEqual(data, {'hello': 'world'})

  def test_read_json_not_found(self):
    """Checks reading JSON when file not found on GCS."""
    blob_mock = self.mock_storage.Blob.from_string.return_value
    blob_mock.download_as_bytes.side_effect = self.mock_gcs_exceptions.NotFound(
        'not found'
    )
    self.assertIsNone(self.fs.read_json('gs://bucket/path/missing.json'))

  def test_read_json_decode_error(self):
    """Checks reading JSON on GCS handles JSONDecodeError by deleting."""
    blob_mock = self.mock_storage.Blob.from_string.return_value
    blob_mock.download_as_bytes.side_effect = json.JSONDecodeError(
        'msg', 'doc', 0
    )
    with mock.patch.object(self.fs, 'delete_file') as mock_delete:
      self.assertIsNone(self.fs.read_json('gs://bucket/path/bad.json'))
      mock_delete.assert_called_once_with('gs://bucket/path/bad.json')

  def test_read_json_api_call_error(self):
    """Checks reading JSON on GCS handles GoogleAPICallError by deleting."""
    blob_mock = self.mock_storage.Blob.from_string.return_value
    blob_mock.download_as_bytes.side_effect = (
        self.mock_gcs_exceptions.GoogleAPICallError('err')
    )
    with mock.patch.object(self.fs, 'delete_file') as mock_delete:
      self.assertIsNone(self.fs.read_json('gs://bucket/path/bad.json'))
      mock_delete.assert_called_once_with('gs://bucket/path/bad.json')

  def test_write_json_success(self):
    """Checks writing JSON to GCS."""
    blob_mock = self.mock_storage.Blob.from_string.return_value
    self.fs.write_json('gs://bucket/path/data.json', {'key': 'val'})
    blob_mock.upload_from_string.assert_called_once()

  def test_write_json_exception(self):
    """Checks exception handling in write_json for GCS."""
    blob_mock = self.mock_storage.Blob.from_string.return_value
    blob_mock.upload_from_string.side_effect = (
        self.mock_gcs_exceptions.GoogleAPICallError('err')
    )
    with self.assertLogs(
        'tensorboard.plugins.profile', level='ERROR'
    ):
      self.fs.write_json('gs://bucket/path/data.json', {'key': 'val'})

  def test_delete_file_success(self):
    """Checks deleting file on GCS."""
    blob_mock = self.mock_storage.Blob.from_string.return_value
    self.fs.delete_file('gs://bucket/path/file.txt')
    blob_mock.delete.assert_called_once()

  def test_delete_file_not_found(self):
    """Checks deleting GCS file handles NotFound exception."""
    blob_mock = self.mock_storage.Blob.from_string.return_value
    blob_mock.delete.side_effect = self.mock_gcs_exceptions.NotFound(
        'not found'
    )
    self.fs.delete_file('gs://bucket/path/file.txt')

  def test_delete_file_api_error(self):
    """Checks deleting GCS file handles GoogleAPICallError exception."""
    blob_mock = self.mock_storage.Blob.from_string.return_value
    blob_mock.delete.side_effect = self.mock_gcs_exceptions.GoogleAPICallError(
        'err'
    )
    with self.assertLogs(
        'tensorboard.plugins.profile', level='ERROR'
    ):
      self.fs.delete_file('gs://bucket/path/file.txt')

  def test_read_text_success(self):
    """Checks reading text file from GCS."""
    blob_mock = self.mock_storage.Blob.from_string.return_value
    blob_mock.download_as_bytes.return_value = b'text content'
    self.assertEqual(
        self.fs.read_text('gs://bucket/path/file.txt'), 'text content'
    )

  def test_read_text_not_found(self):
    """Checks reading text handles GCS NotFound."""
    blob_mock = self.mock_storage.Blob.from_string.return_value
    blob_mock.download_as_bytes.side_effect = self.mock_gcs_exceptions.NotFound(
        'not found'
    )
    self.assertIsNone(self.fs.read_text('gs://bucket/path/file.txt'))

  def test_read_text_api_error(self):
    """Checks reading text handles GCS GoogleAPICallError."""
    blob_mock = self.mock_storage.Blob.from_string.return_value
    blob_mock.download_as_bytes.side_effect = (
        self.mock_gcs_exceptions.GoogleAPICallError('err')
    )
    self.assertIsNone(self.fs.read_text('gs://bucket/path/file.txt'))

  def test_write_text_success(self):
    """Checks writing text file to GCS."""
    blob_mock = self.mock_storage.Blob.from_string.return_value
    self.fs.write_text('gs://bucket/path/file.txt', 'my text')
    blob_mock.upload_from_string.assert_called_once_with(
        'my text', content_type='text/plain'
    )

  def test_write_text_api_error(self):
    """Checks writing text handles GCS GoogleAPICallError."""
    blob_mock = self.mock_storage.Blob.from_string.return_value
    blob_mock.upload_from_string.side_effect = (
        self.mock_gcs_exceptions.GoogleAPICallError('err')
    )
    with self.assertLogs(
        'tensorboard.plugins.profile', level='ERROR'
    ):
      self.fs.write_text('gs://bucket/path/file.txt', 'my text')


class GetStorageClientTest(absltest.TestCase):
  """Tests for the get_storage_client module-level function."""

  def setUp(self):
    super().setUp()
    self.addCleanup(profile_io.get_storage_client.cache_clear)
    profile_io.get_storage_client.cache_clear()

  def test_get_storage_client_with_auth(self):
    """Checks get_storage_client when google_auth is present."""
    mock_auth = mock.MagicMock()
    mock_requests = mock.MagicMock()
    mock_storage = mock.MagicMock()

    with mock.patch.object(
        profile_io, 'google_auth', mock_auth
    ), mock.patch.object(
        profile_io, 'google_auth_requests', mock_requests
    ), mock.patch.object(
        profile_io, 'storage', mock_storage
    ):
      mock_auth.default.return_value = ('my-creds', 'my-project')

      client = profile_io.get_storage_client()

      mock_auth.default.assert_called_once()
      mock_requests.AuthorizedSession.assert_called_once_with('my-creds')
      mock_storage.Client.assert_called_once_with(
          project='my-project',
          credentials='my-creds',
          _http=mock_requests.AuthorizedSession.return_value,
      )
      self.assertEqual(client, mock_storage.Client.return_value)

  def test_get_storage_client_no_auth(self):
    """Checks get_storage_client when google_auth is None."""
    mock_storage = mock.MagicMock()
    with mock.patch.object(profile_io, 'google_auth', None), mock.patch.object(
        profile_io, 'storage', mock_storage
    ):
      client = profile_io.get_storage_client()
      mock_storage.Client.assert_called_once_with()
      self.assertEqual(client, mock_storage.Client.return_value)


class GcsFileSystemSoftImportTest(absltest.TestCase):
  """Tests GcsFileSystem when GCS optional libraries are not imported."""

  def setUp(self):
    super().setUp()
    self.enter_context(mock.patch.object(profile_io, 'storage', None))
    self.enter_context(
        mock.patch.object(profile_io, 'gcs_exceptions', None)
    )

  def test_init_runtime_error(self):
    with self.assertRaises(RuntimeError):
      profile_io.GcsFileSystem()

  def test_no_op_methods(self):
    fs = profile_io.GcsFileSystem(storage_client=mock.MagicMock())
    self.assertIsNone(fs.read_json('gs://b/file.json'))
    self.assertIsNone(fs.read_text('gs://b/file.txt'))

    fs.write_json('gs://b/file.json', {})
    fs.write_text('gs://b/file.txt', 'data')
    fs.delete_file('gs://b/file.txt')


if __name__ == '__main__':
  absltest.main()
