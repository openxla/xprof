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
"""Policy for tool-result caching via cache_version.txt."""

from __future__ import annotations

import os
from typing import Any

from xprof.profile_plugin.constants import CACHE_VERSION_FILE
from xprof.profile_plugin.logging_config import logger


def should_use_saved_result(
    run_dir: str,
    requested: bool,
    version_module: Any,
    epath_module: Any,
) -> bool:
  """Returns whether converters may use on-disk saved tool results.

  If the cache version file is missing, older than the plugin version, or
  unreadable, forces a recompute by returning False.
  """
  use_saved_result = requested
  try:
    with epath_module.Path(os.path.join(run_dir, CACHE_VERSION_FILE)).open(
        'r'
    ) as f:
      cache_version = f.read().strip()
      if cache_version < version_module.__version__:
        use_saved_result = False
  except FileNotFoundError:
    logger.info('Cache version file not found, invalidating cache.')
    use_saved_result = False
  except OSError:
    logger.warning('Cannot read cache version file', exc_info=True)
    use_saved_result = False
  return use_saved_result


def write_cache_version_file(
    run_dir: str, version_module: Any, epath_module: Any
) -> None:
  """Writes the current plugin version to the cache version file."""
  try:
    with epath_module.Path(os.path.join(run_dir, CACHE_VERSION_FILE)).open(
        'w'
    ) as f:
      f.write(version_module.__version__)
  except OSError:
    logger.warning(
        'Cannot write cache version file to %s', run_dir, exc_info=True
    )
