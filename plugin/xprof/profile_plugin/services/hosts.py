# Copyright 2026 The XProf Authors. All Rights Reserved.
"""Host selection for multi-host XPlane profile sessions.

Ported from ProfilePlugin._get_valid_hosts.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any

from xprof.profile_plugin.constants import ALL_HOSTS
from xprof.profile_plugin.models import HostSelection
from xprof.profile_plugin.tools.filenames import parse_filename
from xprof.profile_plugin.tools.registry import (
    XPLANE_TOOLS_ALL_HOSTS_ONLY,
    supports_multi_host_selection,
)

logger = logging.getLogger('tensorboard.plugins.profile')


class HostSelector:
  """Selects hosts and corresponding XPlane asset paths for a tool request."""

  def select(
      self,
      run_dir: str,
      tool: str,
      host: str | None,
      hosts_param: str | None,
      xplane_basenames: Sequence[str],
      path_join: Callable[..., Any],
  ) -> HostSelection:
    """Validate host parameters and resolve asset paths.

    Args:
      run_dir: Profile session (run) directory.
      tool: Requested tool name.
      host: Single host parameter (may be ALL_HOSTS).
      hosts_param: Comma-separated multi-host list (trace_viewer*).
      xplane_basenames: Basename listing of XPlane files in run_dir.
      path_join: Joins path segments into an asset path object (e.g. epath.Path).

    Returns:
      HostSelection with selected host names and asset paths.

    Raises:
      FileNotFoundError: Missing xplanes or host not found when required.
    """
    asset_paths: list[Any] = []
    selected_hosts: list[str] = []
    all_xplane_files: dict[str, Any] = {}

    for basename in xplane_basenames:
      host_name, _ = parse_filename(basename)
      if host_name:
        all_xplane_files[host_name] = path_join(run_dir, basename)

    if not all_xplane_files:
      logger.warning(
          'no xplane files found for run: %s, tool: %s', run_dir, tool
      )
      raise FileNotFoundError(
          'No xplane file found for run: %s, tool: %s' % (run_dir, tool)
      )

    if hosts_param and supports_multi_host_selection(tool):
      selected_hosts = hosts_param.split(',')
      for selected_host in selected_hosts:
        if selected_host in all_xplane_files:
          asset_paths.append(all_xplane_files[selected_host])
        else:
          raise FileNotFoundError(
              'No xplane file found for host: %s in run: %s'
              % (selected_host, run_dir)
          )
    elif host == ALL_HOSTS:
      asset_paths = list(all_xplane_files.values())
      selected_hosts = list(all_xplane_files.keys())
    elif host and host in all_xplane_files:
      selected_hosts = [host]
      asset_paths = [all_xplane_files[host]]
    elif host:
      logger.warning(
          'No xplane file found for host: %s in run: %s', host, run_dir
      )
      # Preserve original plugin behavior: compare host against the tool set.
      if host not in XPLANE_TOOLS_ALL_HOSTS_ONLY:
        raise FileNotFoundError(
            'No xplane file found for host: %s in run: %s' % (host, run_dir)
        )
    # Request that does not specify host or hosts: use all hosts.
    elif not host and not hosts_param:
      selected_hosts = list(all_xplane_files.keys())
      asset_paths = list(all_xplane_files.values())

    if not asset_paths:
      logger.warning(
          'No matching asset paths found for run %s, tool %s, host(s) %s / %s',
          run_dir,
          tool,
          hosts_param,
          host,
      )
      if not host and tool not in XPLANE_TOOLS_ALL_HOSTS_ONLY:
        raise FileNotFoundError(
            'Host must be specified for tool %s in run %s' % (tool, run_dir)
        )

    return HostSelection(
        selected_hosts=tuple(selected_hosts),
        asset_paths=tuple(asset_paths),
    )
