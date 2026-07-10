# Copyright 2026 The XProf Authors. All Rights Reserved.
"""Build fixed profile logdirs for characterization tests."""

from __future__ import annotations

import os
from typing import Sequence

from xprof import profile_plugin
from xprof.standalone.tensorboard_shim import plugin_asset_util


def create_two_host_session(
    logdir: str,
    session: str = 'session_a',
    hosts: Sequence[str] = ('host0', 'host1'),
) -> str:
  """Create ``logdir/plugins/profile/<session>/<host>.xplane.pb`` layout.

  Returns:
    Absolute path to the session directory.
  """
  plugin_dir = plugin_asset_util.PluginDirectory(
      logdir, profile_plugin.PLUGIN_NAME
  )
  run_dir = os.path.join(plugin_dir, session)
  os.makedirs(run_dir, exist_ok=True)
  for host in hosts:
    path = os.path.join(run_dir, f'{host}.xplane.pb')
    with open(path, 'wb') as f:
      f.write(b'fake-xplane')
  return run_dir


def basenames_from_paths(paths) -> list[str]:
  """Normalize convert path args to stable basenames (sorted)."""
  return sorted(os.path.basename(str(p)) for p in paths)
