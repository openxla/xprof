# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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
"""A standalone version of Tensorboard's context and utils."""

import collections
import logging
import os
import pathlib
import re
from typing import Iterator
import urllib.parse

from xprof.standalone import plugin_asset_util

logger = logging.getLogger("tensorboard")


def IsCloudPath(path: str) -> bool:
  """Checks whether a given path is Cloud filesystem path."""
  parsed_url = urllib.parse.urlparse(path)
  return parsed_url.scheme and parsed_url.scheme not in ["file", ""]


_ESCAPE_GLOB_CHARACTERS_REGEX = re.compile("([*?[])")


def _EscapeGlobCharacters(path: str) -> str:
  drive, path = os.path.splitdrive(path)
  return "%s%s" % (drive, _ESCAPE_GLOB_CHARACTERS_REGEX.sub(r"[\1]", path))


def ListRecursivelyViaGlobbing(
    top: str,
) -> Iterator[tuple[str, str]]:
  """Recursively lists all files with the directory.

  Based off of
  tensorboard/backend/event_processing/io_wrapper.py:ListRecursivelyViaGlobbing

  Args:
    top: The top-level directory to traverse.

  Yields:
    (dir_name, file_paths) tuples.
  """
  fs = plugin_asset_util.get_fs_protocol(top)
  current_glob_string = _EscapeGlobCharacters(top)
  level = 0

  while True:
    logger.info("GlobAndListFiles: Starting to glob level %d", level)
    glob = fs.glob(current_glob_string + "/*")
    logger.info(
        "GlobAndListFiles: %d files glob-ed at level %d", len(glob), level
    )

    if not glob:
      return

    pairs = collections.defaultdict(list)
    for file_path in glob:
      pairs[os.path.dirname(file_path)].append(file_path)
    for dir_name, file_paths in pairs.items():
      yield (dir_name, tuple(file_paths))

    if len(pairs) == 1:
      current_glob_string = list(pairs.keys())[0]

    current_glob_string = os.path.join(current_glob_string, "*")
    level += 1


def ListRecursivelyViaWalking(
    top: str,
) -> Iterator[tuple[str, str]]:
  for dir_path, _, filenames in plugin_asset_util.walk_with_fsspec(top):
    yield (
        dir_path,
        tuple(os.path.join(dir_path, filename) for filename in filenames),
    )


def IsTensorFlowEventsFile(path: str):
  if not path:
    raise ValueError("Path must be a nonempty string")
  return "tfevents" in os.path.basename(path)


class Run:

  def __init__(self, path):
    self.run_name = path


class EventMultiplexer:
  """https://github.com/tensorflow/tensorboard/blob/a7178f4f622a786463d23ef645e0f16f6ea7a1cb/tensorboard/data/provider.py#L26."""

  def __init__(self, logdir=""):
    self._paths = {}
    if logdir:
      self.AddRunsFromDirectory(logdir)

  def list_runs(self, *_, **__):  # pylint: disable=invalid-name,unused-argument
    return self._paths.values()

  def AddRun(self, path, name=None):
    name = name or path
    self._paths[name] = Run(name)
    return self

  def AddRunsFromDirectory(self, path: str, name: str = None):
    path = str(pathlib.Path(path).expanduser())
    logger.info("Starting AddRunsFromDirectory: %s", path)
    for subdir in GetLogdirSubdirectories(path):
      logger.info("Adding run from directory %s", subdir)
      rpath = os.path.relpath(subdir, path)
      subname = os.path.join(name, rpath) if name else rpath
      self.AddRun(subdir, name=subname)
    logger.info("Done with AddRunsFromDirectory: %s", path)
    return self

  def Reload(self):
    pass

DataProvider = EventMultiplexer


def GetLogdirSubdirectories(
    path: str,
) -> Iterator[str]:
  """Obtains all subdirectories with events files.

  Based off of tensorboard/backend/event_processing/io_wrapper.py.

  Args:
    path: The top-level directory to traverse.

  Returns:
    Subdirectories containing event files.
  """
  fs = plugin_asset_util.get_fs_protocol(path)
  if not fs.exists(path):
    return ()

  if not fs.isdir(path):
    raise ValueError(
        "GetLogdirSubdirectories: path exists and is not a directory, %s" % path
    )

  if IsCloudPath(path):
    logger.info(
        "GetLogdirSubdirectories: Starting to list directories via glob-ing."
    )
    traversal_method = ListRecursivelyViaGlobbing
  else:
    logger.info(
        "GetLogdirSubdirectories: Starting to list directories via walking."
    )
    traversal_method = ListRecursivelyViaWalking

  return (
      subdir
      for (subdir, files) in traversal_method(path)
      if any(IsTensorFlowEventsFile(f) for f in files)
  )


class MultiplexerDataProvider:

  def __init__(self, multiplexer, logdir):
    self.multiplexer = multiplexer
    self.logdir = logdir
