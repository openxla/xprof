# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Cache for available tool lists based on XPlane file states."""

from __future__ import annotations

from collections.abc import Sequence

from etils import epath

from xprof import profile_io
from xprof.profile_plugin.logging_config import logger


class ToolsCache:
  """A cache for tool lists based on file content hashes or mtimes.

  Attributes:
    CACHE_FILE_NAME: The name of the cache file.
    CACHE_VERSION: The version of the cache format.
  """

  CACHE_FILE_NAME = '.cached_tools.json'
  CACHE_VERSION = 1

  def __init__(
      self, profile_run_dir: epath.Path, fs: profile_io.ProfileFileSystem
  ):
    """Initializes the ToolsCache.

    Args:
      profile_run_dir: The directory containing the profile run data.
      fs: The file system object to use for file operations.
    """
    self._profile_run_dir = profile_run_dir
    self._cache_file = self._profile_run_dir / self.CACHE_FILE_NAME
    self._fs = fs
    logger.info('ToolsCache initialized for %s', self._cache_file)

  def load(self) -> list[str] | None:
    """Loads the cached list of tools if the cache is valid.

    The cache is valid if the cache file exists, the version matches, and
    the file states (hashes/mtimes) of the XPlane files have not changed.

    Returns:
      A list of tool names if the cache is valid, otherwise None.
    """
    cached_data = self._fs.read_json(str(self._cache_file))
    if cached_data is None:
      return None

    if cached_data.get('version') != self.CACHE_VERSION:
      logger.info(
          'ToolsCache invalid: version mismatch, expected %s, got %s.'
          ' Invalidating %s',
          self.CACHE_VERSION,
          cached_data.get('version'),
          self._cache_file,
      )
      self.invalidate()
      return None

    current_files = self._fs.get_xplane_file_states(str(self._profile_run_dir))
    if current_files is None:
      logger.info(
          'ToolsCache invalid: could not determine current file states.'
          ' Invalidating %s',
          self._cache_file,
      )
      self.invalidate()
      return None

    if cached_data.get('files') != current_files:
      logger.info(
          'ToolsCache invalid: file states differ. Invalidating %s',
          self._cache_file,
      )
      self.invalidate()
      return None

    logger.info('ToolsCache hit: %s', self._cache_file)
    return cached_data.get('tools')

  def save(self, tools: Sequence[str]) -> None:
    """Saves the list of tools and the current file states to the cache file.

    Args:
      tools: The list of tool names to cache.
    """
    current_files_for_cache = self._fs.get_xplane_file_states(
        str(self._profile_run_dir)
    )
    if current_files_for_cache is None:
      logger.warning(
          'ToolsCache not saved: could not get file states %s', self._cache_file
      )
      return

    new_cache_data = {
        'version': self.CACHE_VERSION,
        'files': current_files_for_cache,
        'tools': tools,
    }
    self._fs.write_json(str(self._cache_file), new_cache_data)

  def invalidate(self) -> None:
    """Deletes the cache file, forcing regeneration on the next load."""
    self._fs.delete_file(str(self._cache_file))
