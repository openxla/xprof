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
"""File system abstraction for the TensorBoard profile plugin."""

import abc
from collections.abc import Mapping
from collections.abc import Sequence
import functools
import json
import logging
import os
from typing import Any

from etils import epath

logger = logging.getLogger('tensorboard.plugins.profile')

try:
  # pylint: disable=g-import-not-at-top
  from google.cloud import exceptions as gcs_exceptions  # type: ignore
  from google.cloud import storage  # type: ignore
except ImportError:
  logger.warning(
      'Google Cloud Storage not found. GCS paths will not be supported.'
  )
  gcs_exceptions = None
  storage = None


@functools.lru_cache(maxsize=None)
def get_storage_client():
  """Returns a memoized storage client instance."""
  if storage is None:
    raise RuntimeError(
        'Google Cloud Storage libraries not found. gs:// paths are not'
        ' supported.'
    )
  return storage.Client()


def _list_gcs_dir(storage_client, gcs_path: str):
  """Lists blobs and sub-directories in a GCS path."""
  gcs_path_no_prefix = gcs_path.removeprefix('gs://')
  bucket_name, _, prefix = gcs_path_no_prefix.partition('/')
  if prefix and not prefix.endswith('/'):
    prefix += '/'
  iterator = storage_client.list_blobs(
      bucket_name, prefix=prefix, delimiter='/'
  )
  return list(iterator), iterator.prefixes, bucket_name


class ProfileFileSystem(abc.ABC):
  """Abstract base class for file system operations."""

  def __init__(self, epath_module: Any = epath):
    """Initializes the instance."""
    self._epath = epath_module

  @abc.abstractmethod
  def get_xplane_file_states(self, dir_path: str) -> Mapping[str, str] | None:
    """Gets the current state of XPlane files in a directory."""

  @abc.abstractmethod
  def get_xplane_basenames(self, dir_path: str) -> Sequence[str]:
    """Returns a list of .xplane.pb base filenames in the given path."""

  @abc.abstractmethod
  def dir_has_xplane_files(self, dir_path: str) -> bool:
    """Checks if the directory contains any .xplane.pb files."""

  @abc.abstractmethod
  def get_all_basenames(self, dir_path: str) -> Sequence[str]:
    """Returns a list of all base filenames in the given path."""

  @abc.abstractmethod
  def get_session_paths(self, dir_path: str) -> Mapping[str, str]:
    """Returns a map of session names to absolute paths."""

  @abc.abstractmethod
  def read_json(self, file_path: str) -> Mapping[str, Any] | None:
    """Reads a JSON file."""

  @abc.abstractmethod
  def write_json(self, file_path: str, data: Mapping[str, Any]) -> None:
    """Writes a JSON file."""

  @abc.abstractmethod
  def delete_file(self, file_path: str) -> None:
    """Deletes a file."""

  @abc.abstractmethod
  def read_text(self, file_path: str) -> str | None:
    """Reads a text file."""

  @abc.abstractmethod
  def write_text(self, file_path: str, data: str) -> None:
    """Writes a text file."""


class GcsFileSystem(ProfileFileSystem):
  """GCS implementation of ProfileFileSystem."""

  def __init__(self, epath_module: Any = epath, storage_client=None):
    """Initializes the instance."""
    if storage_client is None and storage is None:
      raise RuntimeError(
          'Google Cloud Storage libraries not found. gs:// paths are not'
          ' supported.'
      )
    super().__init__(epath_module)
    self._storage_client = (
        storage_client if storage_client is not None else get_storage_client()
    )

  def get_xplane_file_states(self, dir_path: str) -> dict[str, str] | None:
    file_identifiers = {}
    try:
      blobs, _, _ = _list_gcs_dir(self._storage_client, dir_path)
      for blob in blobs:
        if not blob.name.endswith('.xplane.pb'):
          continue
        md5_hash = blob.md5_hash
        if not isinstance(md5_hash, str):
          logger.warning(
              'Could not find a valid md5_hash for gs://%s/%s,'
              ' cache will be invalidated.',
              blob.bucket.name,
              blob.name,
          )
          return None
        file_identifiers[os.path.basename(blob.name)] = md5_hash
      return file_identifiers
    except RuntimeError:
      return None

  def get_xplane_basenames(self, dir_path: str) -> list[str]:
    try:
      blobs, _, _ = _list_gcs_dir(self._storage_client, dir_path)
      return [
          os.path.basename(blob.name)
          for blob in blobs
          if blob.name.endswith('.xplane.pb')
      ]
    except RuntimeError:
      return []

  def dir_has_xplane_files(self, dir_path: str) -> bool:
    try:
      blobs, _, _ = _list_gcs_dir(self._storage_client, dir_path)
      return any(blob.name.endswith('.xplane.pb') for blob in blobs)
    except RuntimeError:
      return False

  def get_all_basenames(self, dir_path: str) -> list[str]:
    try:
      blobs, _, _ = _list_gcs_dir(self._storage_client, dir_path)
      return [os.path.basename(blob.name) for blob in blobs]
    except RuntimeError:
      return []

  def get_session_paths(self, dir_path: str) -> dict[str, str]:
    path_by_session_name = {}
    try:
      _, prefixes, bucket_name = _list_gcs_dir(self._storage_client, dir_path)
      if not bucket_name:
        return path_by_session_name
      for subdir_prefix in prefixes:
        session_path_str = f'gs://{bucket_name}/{subdir_prefix}'
        if not self.dir_has_xplane_files(session_path_str):
          continue
        session_name = self._epath.Path(subdir_prefix).name
        path_by_session_name[session_name] = session_path_str
    except RuntimeError:
      pass
    return path_by_session_name

  def read_json(self, file_path: str) -> dict[str, Any] | None:
    if storage is None or gcs_exceptions is None:
      return None
    try:
      blob = storage.Blob.from_string(file_path, client=self._storage_client)
      return json.loads(blob.download_as_bytes())
    except gcs_exceptions.NotFound:
      logger.info('File not found on GCS: %s', file_path)
      return None
    except (json.JSONDecodeError, gcs_exceptions.GoogleAPICallError) as e:
      logger.exception(
          'Error reading or decoding GCS file %s: %r, invalidating.',
          file_path,
          e,
      )
      self.delete_file(file_path)
      return None

  def write_json(self, file_path: str, data: Mapping[str, Any]) -> None:
    if storage is None or gcs_exceptions is None:
      return
    try:
      blob = storage.Blob.from_string(file_path, client=self._storage_client)
      blob.upload_from_string(
          json.dumps(data, sort_keys=True, indent=2),
          content_type='application/json',
      )
      logger.info('File saved to GCS: %s', file_path)
    except (TypeError, gcs_exceptions.GoogleAPICallError) as e:
      logger.exception('Error writing GCS file %s: %r', file_path, e)

  def delete_file(self, file_path: str) -> None:
    if storage is None or gcs_exceptions is None:
      return
    try:
      blob = storage.Blob.from_string(file_path, client=self._storage_client)
      blob.delete()
      logger.info('File deleted from GCS: %s', file_path)
    except gcs_exceptions.NotFound:
      pass
    except gcs_exceptions.GoogleAPICallError as e:
      logger.exception('Error deleting GCS file %s: %r', file_path, e)

  def read_text(self, file_path: str) -> str | None:
    if storage is None or gcs_exceptions is None:
      return None
    try:
      blob = storage.Blob.from_string(file_path, client=self._storage_client)
      return blob.download_as_bytes().decode('utf-8')
    except gcs_exceptions.NotFound:
      return None
    except gcs_exceptions.GoogleAPICallError as e:
      logger.warning(
          'Error reading GCS file %s: %r', file_path, e, exc_info=True
      )
      return None

  def write_text(self, file_path: str, data: str) -> None:
    if storage is None or gcs_exceptions is None:
      return
    try:
      blob = storage.Blob.from_string(file_path, client=self._storage_client)
      blob.upload_from_string(data, content_type='text/plain')
    except gcs_exceptions.GoogleAPICallError as e:
      logger.error('Error writing GCS file %s: %r', file_path, e, exc_info=True)


def _get_local_file_identifier(file_path_str: str) -> str | None:
  """Gets a string identifier for a local file based on mtime and size."""
  try:
    stat_result = os.stat(file_path_str)
    return f'{int(stat_result.st_mtime)}-{stat_result.st_size}'
  except FileNotFoundError:
    logger.warning('Local file not found: %s', file_path_str)
    return None
  except OSError as e:
    logger.error(
        'OSError getting stat for local file %s: %r',
        file_path_str,
        e,
        exc_info=True,
    )
    return None


class LocalFileSystem(ProfileFileSystem):
  """Local implementation of ProfileFileSystem."""

  def get_xplane_file_states(self, dir_path: str) -> dict[str, str] | None:
    path = self._epath.Path(dir_path)
    file_identifiers = {}
    try:
      for xplane_file in path.glob('*.xplane.pb'):
        file_id = _get_local_file_identifier(str(xplane_file))
        if file_id is None:
          logger.warning(
              'Could not get identifier for %s, cache will be invalidated.',
              xplane_file,
          )
          return None
        file_identifiers[xplane_file.name] = file_id
      return file_identifiers
    except OSError as e:
      logger.warning(
          'Could not glob files in %s: %r',
          dir_path,
          e,
          exc_info=True,
      )
      return None

  def get_xplane_basenames(self, dir_path: str) -> list[str]:
    path = self._epath.Path(dir_path)
    try:
      return [f.name for f in path.glob('*.xplane.pb')]
    except OSError as e:
      logger.warning(
          'Cannot read asset directory: %s, OpError %s',
          dir_path,
          e,
          exc_info=True,
      )
      return []

  def dir_has_xplane_files(self, dir_path: str) -> bool:
    path = self._epath.Path(dir_path)
    try:
      return any(path.glob('*.xplane.pb'))
    except OSError:
      return False

  def get_all_basenames(self, dir_path: str) -> list[str]:
    path = self._epath.Path(dir_path)
    try:
      return [f.name for f in path.iterdir()]
    except OSError as e:
      logger.warning(
          'Cannot read asset directory: %s, Error %r',
          dir_path,
          e,
          exc_info=True,
      )
      return []

  def get_session_paths(self, dir_path: str) -> dict[str, str]:
    path_by_session_name = {}
    path = self._epath.Path(dir_path)
    try:
      for session in path.iterdir():
        if self.dir_has_xplane_files(str(session)):
          path_by_session_name[session.name] = str(session)
    except OSError as e:
      logger.warning(
          'Cannot read asset directory: %s, Error %r',
          dir_path,
          e,
          exc_info=True,
      )
    return path_by_session_name

  def read_json(self, file_path: str) -> dict[str, Any] | None:
    path = self._epath.Path(file_path)
    try:
      with path.open('r') as f:
        return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
      logger.exception(
          'Error reading or decoding file %s: %r, invalidating.',
          file_path,
          e,
      )
      self.delete_file(file_path)
      return None

  def write_json(self, file_path: str, data: Mapping[str, Any]) -> None:
    path = self._epath.Path(file_path)
    try:
      with path.open('w') as f:
        json.dump(data, f, sort_keys=True, indent=2)
      logger.info('File saved: %s', file_path)
    except (OSError, TypeError) as e:
      logger.exception('Error writing file %s: %r', file_path, e)

  def delete_file(self, file_path: str) -> None:
    path = self._epath.Path(file_path)
    try:
      path.unlink()
      logger.info('File deleted: %s', file_path)
    except FileNotFoundError:
      pass
    except OSError as e:
      logger.exception('Error deleting file %s: %r', file_path, e)

  def read_text(self, file_path: str) -> str | None:
    path = self._epath.Path(file_path)
    try:
      with path.open('r') as f:
        return f.read()
    except FileNotFoundError:
      return None
    except OSError as e:
      logger.warning(
          'Cannot read text file %s: %r', file_path, e, exc_info=True
      )
      return None

  def write_text(self, file_path: str, data: str) -> None:
    path = self._epath.Path(file_path)
    try:
      with path.open('w') as f:
        f.write(data)
    except OSError as e:
      logger.warning(
          'Cannot write text file to %s: %r', file_path, e, exc_info=True
      )


def get_file_system(path: str, epath_module: Any = epath) -> ProfileFileSystem:
  """Returns a file system abstracting remote/local file operations."""
  if storage is not None and path.startswith('gs://'):
    return GcsFileSystem(epath_module, storage_client=get_storage_client())
  return LocalFileSystem(epath_module)
