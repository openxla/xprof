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
"""Utility functions for plugin assets."""

import os
import urllib.parse
import fsspec

_PLUGINS_DIR = "plugins"


def get_fs_protocol(path):
  string_path = str(path)
  parsed_url = urllib.parse.urlparse(string_path)
  protocol = parsed_url.scheme or "file"
  return fsspec.filesystem(protocol)


def PluginDirectory(logdir, plugin_name):  # pylint: disable=invalid-name
  return os.path.join(logdir, _PLUGINS_DIR, plugin_name)


def walk_with_fsspec(top):
  """Mimics os.walk using fsspec."""
  try:
    parsed_url = urllib.parse.urlparse(top)
    protocol = parsed_url.scheme or "file"
    fs = fsspec.filesystem(protocol)

    # Use a queue for breadth-first traversal
    queue = [top]

    while queue:
      current_dir = queue.pop(0)
      dirnames = []
      filenames = []

      try:
        items = fs.listdir(current_dir)
      except FileNotFoundError:
        continue  # skip if the directory is not found.

      for item in items:
        full_path = os.path.join(current_dir, item["name"])
        if item["type"] == "directory":
          dirnames.append(item["name"])
          queue.append(full_path)
        else:
          filenames.append(item["name"])

      yield (current_dir, dirnames, filenames)

  except OSError as e:
    print(f"An error occurred: {e}")
    return


def iterate_directory_with_fsspec(plugin_dir):
  """Replaces upath.UPath(plugin_dir).iterdir() with fsspec."""
  try:
    fs = get_fs_protocol(plugin_dir)
    for item in fs.listdir(plugin_dir):
      full_path = os.path.join(
          plugin_dir, item["name"]
      )  # construct the full path
      yield full_path
  except FileNotFoundError:
    # Handle cases where the directory doesn't exist.
    print(f"Directory not found: {plugin_dir}")
    return


def ListAssets(logdir, plugin_name):  # pylint: disable=invalid-name
  plugin_dir = PluginDirectory(logdir, plugin_name)
  try:
    # Strip trailing slashes, which listdir() includes for some filesystems.
    return [
        x.rstrip("/")[len(plugin_dir) + 1 :]
        for x in iterate_directory_with_fsspec(plugin_dir)
    ]
  except FileNotFoundError:
    return []
