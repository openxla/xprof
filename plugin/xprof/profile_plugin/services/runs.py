# Copyright 2026 The XProf Authors. All Rights Reserved.
"""Logdir walk: discover frontend run names and per-run tool lists.

Ported from ProfilePlugin.generate_runs / generate_tools_of_run so discovery
can be unit-tested without Werkzeug or the full plugin class.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable, Iterator, MutableMapping, Sequence
from typing import Any

import six

from xprof.profile_plugin.cache.tools_cache import ToolsCache
from xprof.profile_plugin.constants import PLUGIN_NAME, TB_NAME
from xprof.profile_plugin.logging_config import logger
from xprof.profile_plugin.services.sessions import tb_run_directory
from xprof.profile_plugin.tools.catalog import get_active_tools
from xprof.standalone.tensorboard_shim import plugin_asset_util


def list_plugin_assets(
    session_dir: str, runs: Sequence[str], plugin_name: str
) -> dict[str, Sequence[str]]:
  """List profile plugin assets for each TensorBoard run name."""
  result = {}
  for run in runs:
    run_path = tb_run_directory(session_dir, run)
    assets = plugin_asset_util.ListAssets(run_path, plugin_name)
    result[run] = assets
  return result


class RunDiscovery:
  """Discovers profile "frontend runs" under a TensorBoard logdir."""

  def __init__(
      self,
      logdir: str | None,
      epath_module: Any,
      run_dir_cache: MutableMapping[str, str],
      cache_lock: threading.Lock,
      get_all_basenames: Callable[[str], Sequence[str]],
      get_file_system: Callable[[str], Any],
  ):
    self.logdir = logdir
    self._epath = epath_module
    self._run_dir_cache = run_dir_cache
    self._cache_lock = cache_lock
    self._get_all_basenames = get_all_basenames
    self._get_file_system = get_file_system

  def iter_frontend_runs(self) -> Iterator[str]:
    """Yield frontend run names (see ProfilePlugin.generate_runs docstring)."""
    if not self.logdir:
      return

    logdir_path = self._epath.Path(self.logdir)
    schemeless_logdir = str(logdir_path)
    if '://' in schemeless_logdir:
      schemeless_logdir = schemeless_logdir.split('://', 1)[1]
    tb_runs = {'.'}

    if logdir_path.is_dir():
      try:
        import etils.epath.backend  # pylint: disable=g-import-not-at-top

        fs = etils.epath.backend.fsspec_backend.fs(self.logdir)
        for path_str in fs.glob(os.path.join(self.logdir, '**', PLUGIN_NAME)):
          path = self._epath.Path(path_str)
          if fs.isdir(path) and path.parent.name == TB_NAME:
            tb_run_dir = path.parent.parent
            tb_run = tb_run_dir.relative_to(schemeless_logdir)
            tb_runs.add(str(tb_run))
      except ValueError:
        # gcsfs not available, fall back to legacy path walk.
        for cur_dir, _, _ in logdir_path.walk():
          if cur_dir.name == PLUGIN_NAME and cur_dir.parent.name == TB_NAME:
            tb_run_dir = cur_dir.parent.parent
            tb_run = tb_run_dir.relative_to(logdir_path)
            tb_runs.add(str(tb_run))
    tb_run_names_to_dirs = {
        run: tb_run_directory(self.logdir, run) for run in tb_runs
    }
    plugin_assets = list_plugin_assets(
        self.logdir, list(tb_run_names_to_dirs), PLUGIN_NAME
    )
    visited_runs = set()
    for tb_run_name, profile_runs in six.iteritems(plugin_assets):
      tb_run_dir = tb_run_names_to_dirs[tb_run_name]
      tb_plugin_dir = plugin_asset_util.PluginDirectory(tb_run_dir, PLUGIN_NAME)

      for profile_run in profile_runs:
        profile_run = profile_run.rstrip(os.sep)
        if tb_run_name == '.':
          frontend_run = profile_run
        else:
          frontend_run = str(self._epath.Path(tb_run_name) / profile_run)
        profile_run_dir = str(self._epath.Path(tb_plugin_dir) / profile_run)
        if self._epath.Path(profile_run_dir).is_dir():
          with self._cache_lock:
            self._run_dir_cache[frontend_run] = profile_run_dir
          if frontend_run not in visited_runs:
            visited_runs.add(frontend_run)
            yield frontend_run

  def tools_of_run(self, run: str, run_dir: str | None) -> Iterator[str]:
    """Yield tool names available for a profile run directory."""
    if not run_dir:
      logger.warning('Cannot find asset directory for: %s', run)
      return
    profile_run_dir = self._epath.Path(run_dir)
    fs = self._get_file_system(run_dir)
    cache = ToolsCache(profile_run_dir, fs)

    cached_tools = cache.load()
    if cached_tools is not None:
      for tool in cached_tools:
        yield tool
      return

    tools = []
    all_filenames = self._get_all_basenames(run_dir)
    if all_filenames:
      tools = get_active_tools(all_filenames, str(profile_run_dir))
      cache.save(tools)

    for tool in tools:
      yield tool
