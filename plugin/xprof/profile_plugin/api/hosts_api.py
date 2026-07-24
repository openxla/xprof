# Copyright 2026 The XProf Authors. All Rights Reserved.
"""Frontend host listing for a run+tool (/hosts)."""

from __future__ import annotations

import os
from typing import Any

from werkzeug import wrappers

from xprof.profile_plugin.api.types import HostMetadata
from xprof.profile_plugin.http.respond import respond
from xprof.profile_plugin.logging_config import logger
from xprof.profile_plugin.tools.filenames import hosts_from_xplane_filenames


class HostsApiMixin:
  """HTTP handlers for this frontend API group (mixed into ProfilePlugin)."""

  def _run_host_impl(
      self, run: str, run_dir: str, tool: str
  ) -> list[HostMetadata]:
    if not run_dir:
      logger.warning('Cannot find asset directory for: %s', run)
      return []
    return [
        {'hostname': host}
        for host in hosts_from_xplane_filenames(
            self._get_xplane_basenames(run_dir), tool
        )
    ]


  def host_impl(
      self, run: str, tool: str, request: wrappers.Request | None = None
  ) -> list[HostMetadata]:
    """Returns hosts metadata for the run and tool in log directory.

    In the plugin log directory, each directory contains profile data for a
    single run (identified by the directory name), and files in the run
    directory contains data for different tools and hosts. The file that
    contains profile for a specific tool "x" will have extension TOOLS["x"].

    Example:
      log/
        run1/
          plugins/
            profile/
              host1.trace
              host2.trace
              module1.hlo_proto.pb
              module2.hlo_proto.pb
        run2/
          plugins/
            profile/
              host1.trace
              host2.trace

    Args:
      run: the frontend run name, e.g., 'run1' or 'run2' for the example above.
      tool: the requested tool, e.g., 'trace_viewer' for the example above.
      request: Optional; werkzeug request used for grabbing ctx and experiment
        id for other host implementations

    Returns:
      A list of host names, e.g.:
        host_impl(run1, trace_viewer) --> [{"hostname": "host1"}, {"hostname":
        "host2"}]
        host_impl(run1, memory_viewer) --> [{"hostname": "module1"},
        {"hostname":
        "module2"}]
    """
    run_dir = self._run_dir(run, request)
    return self._run_host_impl(run, run_dir, tool)


  @wrappers.Request.application
  def hosts_route(self, request: wrappers.Request) -> wrappers.Response:
    # pytype: enable=wrong-arg-types
    run = request.args.get('run')
    tool = request.args.get('tag')
    hosts = self.host_impl(run, tool, request)
    return respond(hosts, 'application/json')


  def _get_valid_hosts(
      self, run_dir: str, run: str, tool: str, hosts_param: str, host: str
  ) -> tuple[list[str], list[Any]]:
    """Retrieves and validates the hosts and asset paths for a run and tool.

    Thin wrapper around HostSelector; keeps list/list return type for callers.

    Args:
      run_dir: The run directory.
      run: The frontend run name (unused; retained for call-site compatibility).
      tool: The requested tool.
      hosts_param: Comma-separated list of selected hosts.
      host: The single host parameter.

    Returns:
      A tuple containing (selected_hosts, asset_paths).

    Raises:
      FileNotFoundError: If a required xplane file for the specified host(s)
        is not found.
    """
    del run  # Retained for callers; HostSelector messages use run_dir.
    basenames = self._get_xplane_basenames(run_dir)
    selection = self._hosts.select(
        run_dir=run_dir,
        tool=tool,
        host=host,
        hosts_param=hosts_param,
        xplane_basenames=basenames,
        path_join=lambda *p: self._epath.Path(os.path.join(*p)),
    )
    return list(selection.selected_hosts), list(selection.asset_paths)


