# Copyright 2026 The XProf Authors. All Rights Reserved.
"""Tool data orchestration: session resolve → options → hosts → convert."""

from __future__ import annotations

import os
import threading
from collections.abc import MutableMapping
from typing import Any

from xprof.profile_plugin.cache.result_cache_policy import (
    should_use_saved_result,
    write_cache_version_file,
)
from xprof.profile_plugin.deps import ConvertPort, FileSystemFactory, VersionProvider
from xprof.profile_plugin.logging_config import logger
from xprof.profile_plugin.models import ToolRequest, ToolResult
from xprof.profile_plugin.services.counter_names import try_counter_names_only
from xprof.profile_plugin.services.hosts import HostSelector
from xprof.profile_plugin.services.sessions import SessionResolver
from xprof.profile_plugin.tools.options import build_tool_params
from xprof.profile_plugin.tools.registry import TOOLS, use_xplane


class ToolDataService:
  """Orchestrates XPlane conversion for a ToolRequest (data_impl algorithm)."""

  def __init__(
      self,
      convert: ConvertPort,
      sessions: SessionResolver,
      hosts: HostSelector,
      version: VersionProvider,
      epath_module: Any,
      fs_factory: FileSystemFactory,
  ):
    self._convert = convert
    self._sessions = sessions
    self._hosts = hosts
    self._version = version
    self._epath = epath_module
    self._fs_factory = fs_factory

  def get_tool_data(
      self,
      req: ToolRequest,
      *,
      session_path: str | None = None,
      run_path: str | None = None,
      logdir: str | None = None,
      run_dir_cache: MutableMapping[str, str] | None = None,
      cache_lock: threading.Lock | None = None,
  ) -> ToolResult:
    """Resolve session, hosts, options and convert to tool payload.

    Matches ProfilePlugin.data_impl behavior (without HTTP).

    Raises:
      ValueError: Counter names_only missing device_type, or convert failure.
      FileNotFoundError: Missing host/xplane assets or convert path errors.
      AttributeError: Convert unavailable / conversion attribute errors.
      RuntimeError: Session resolution failure when logdir is missing.
    """
    names_only = try_counter_names_only(req)
    if names_only is not None:
      return names_only

    content_type = 'application/json'
    if run_dir_cache is None:
      run_dir_cache = {}
    if cache_lock is None:
      cache_lock = threading.Lock()

    run_map = self._sessions.run_map_from_params(session_path, run_path)
    run_dir = self._sessions.resolve_run_dir(
        req.run,
        run_map,
        logdir,
        run_dir_cache,
        cache_lock,
    )

    use_saved = should_use_saved_result(
        run_dir, req.use_saved_result, self._version, self._epath
    )
    params = build_tool_params(req, use_saved_result=use_saved)

    tool = req.tool
    if tool not in TOOLS and not use_xplane(tool):
      return ToolResult(data=None, content_type=content_type)

    if not use_xplane(tool):
      logger.info('%s does not use xplane', tool)
      return ToolResult(data=None, content_type=content_type)

    hosts_param = ','.join(req.hosts) if req.hosts else None
    basenames = self._fs_factory.get(run_dir).get_xplane_basenames(run_dir)
    selection = self._hosts.select(
        run_dir=run_dir,
        tool=tool,
        host=req.host,
        hosts_param=hosts_param,
        xplane_basenames=basenames,
        path_join=lambda *p: self._epath.Path(os.path.join(*p)),
    )
    if not selection.asset_paths:
      return ToolResult(data=None, content_type=content_type)

    params['hosts'] = list(selection.selected_hosts)
    try:
      data, content_type = self._convert.xspace_to_tool_data(
          list(selection.asset_paths),
          tool,
          params,
      )
    except AttributeError as e:
      logger.warning('Error generating analysis results due to %r', e)
      raise AttributeError(
          'Error generating analysis results due to %r' % e
      ) from e
    except ValueError as e:
      logger.warning('XPlane convert to tool data failed as %r', e)
      raise
    except FileNotFoundError as e:
      logger.warning('XPlane convert to tool data failed as %r', e)
      raise

    if not use_saved:
      write_cache_version_file(run_dir, self._version, self._epath)

    return ToolResult(
        data=data,
        content_type=content_type,
        content_encoding=None,
    )
