# Copyright 2026 The XProf Authors. All Rights Reserved.
"""Session/run path resolution (session_path > run_path > logdir).

Ported from ProfilePlugin._session_dir_by_run_name_from_request and _run_dir.
Logdir walk / generate_runs lives in services.runs.RunDiscovery.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable, Mapping, MutableMapping
from typing import Any

from xprof.profile_plugin.constants import PLUGIN_NAME
from xprof.standalone.tensorboard_shim import plugin_asset_util


def tb_run_directory(session_dir: str, run: str) -> str:
  """Returns the TensorBoard run directory for a TensorBoard run name.

  For the root run '.' this is the bare session_dir path; for all other runs
  this is session_dir joined with the run name.
  """
  return session_dir if run == '.' else os.path.join(session_dir, run)


class SessionResolver:
  """Resolves frontend run names to profile session directories."""

  def __init__(
      self,
      epath_module: Any,
      fs_factory: Callable[[str], Any] | None = None,
  ):
    self._epath = epath_module
    self._fs_factory = fs_factory

  def _fs(self, path: str) -> Any:
    if self._fs_factory is not None:
      return self._fs_factory(path)
    from xprof import profile_io  # pylint: disable=g-import-not-at-top

    return profile_io.get_file_system(path, self._epath)

  def run_map_from_params(
      self, session_path: str | None, run_path: str | None
  ) -> dict[str, str] | None:
    """Build run-name → session-dir map from URL-style params.

    Precedence: session_path > run_path > None (caller falls back to logdir).
    """
    if session_path:
      if self._fs(session_path).dir_has_xplane_files(session_path):
        run_name = self._epath.Path(session_path).name
        return {run_name: session_path}
      return {}
    if run_path:
      return dict(self._fs(run_path).get_session_paths(run_path))
    return None

  def resolve_run_dir(
      self,
      run: str,
      run_map: Mapping[str, str] | None,
      logdir: str | None,
      run_dir_cache: MutableMapping[str, str],
      cache_lock: threading.Lock,
  ) -> str:
    """Map a frontend run name to a profile run directory.

    When run_map is provided (from session_path/run_path), looks up run there.
    Otherwise checks run_dir_cache, then derives the path from logdir layout
    (tb_run/plugins/profile/profile_run), matching ProfilePlugin._run_dir.

    Raises:
      ValueError: run missing from a non-None run_map.
      RuntimeError: no logdir or TB run directory does not exist.
    """
    if run_map is not None:
      if run in run_map:
        return run_map[run]
      raise ValueError(f'Run {run} not found in run map: {run_map}')

    with cache_lock:
      if run in run_dir_cache:
        return run_dir_cache[run]

    if not logdir:
      raise RuntimeError(
          'No matching run directory for run %s. Logdir is empty.' % run
      )
    tb_run_name, profile_run_name = os.path.split(run.rstrip(os.sep))
    if not tb_run_name:
      tb_run_name = '.'
    tb_run_dir = tb_run_directory(logdir, tb_run_name)
    if not self._epath.Path(tb_run_dir).is_dir():
      raise RuntimeError('No matching run directory for run %s' % run)
    plugin_directory = plugin_asset_util.PluginDirectory(
        tb_run_dir, PLUGIN_NAME
    )
    return os.path.join(plugin_directory, profile_run_name)
