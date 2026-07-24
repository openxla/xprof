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
"""ProfilePlugin façade: construct services and register frontend routes.

HTTP handlers live in ``xprof.profile_plugin.api.*`` (one module per UI concern).
Domain logic lives in ``services/`` and ``tools/`` (no Werkzeug).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections.abc import Callable, Iterator, Mapping, Sequence
import concurrent.futures
import functools
import threading
from typing import Any

from etils import epath
from werkzeug import wrappers

from xprof import profile_io
from xprof import version
from xprof.profile_plugin.api import (
    CacheApiMixin,
    CaptureApiMixin,
    DataApiMixin,
    HostsApiMixin,
    RunsApiMixin,
    StaticApiMixin,
)
from xprof.profile_plugin.api.types import HostMetadata  # re-export for tests
from xprof.profile_plugin.constants import (
    BASE_ROUTE,
    BUNDLE_JS_ROUTE,
    CAPTURE_ROUTE,
    CONFIG_ROUTE,
    DATA_CSV_ROUTE,
    DATA_ROUTE,
    GENERATE_CACHE_ROUTE,
    HLO_MODULE_LIST_ROUTE,
    HOSTS_ROUTE,
    INDEX_HTML_ROUTE,
    INDEX_JS_ROUTE,
    LOCAL_ROUTE,
    MATERIALICONS_WOFF2_ROUTE,
    PLUGIN_NAME,
    RUNS_ROUTE,
    RUN_TOOLS_ROUTE,
    STYLES_CSS_ROUTE,
    TRACE_VIEWER_INDEX_HTML_ROUTE,
    TRACE_VIEWER_INDEX_JS_ROUTE,
    TRACE_VIEWER_V2_JS_ROUTE,
    TRACE_VIEWER_V2_WASM_ROUTE,
    VERSION_ROUTE,
    ZONE_JS_ROUTE,
)
from xprof.profile_plugin.http.logging_middleware import logging_wrapper
from xprof.profile_plugin.http.respond import version_route
from xprof.profile_plugin.lazy_imports import load_convert_module, load_pywrap_module
from xprof.profile_plugin.services.hosts import HostSelector
from xprof.profile_plugin.services.runs import RunDiscovery
from xprof.profile_plugin.services.sessions import SessionResolver
from xprof.profile_plugin.services.tool_data import ToolDataService
from xprof.profile_plugin.tensorflow_bridge import create_tf_profiler
from xprof.standalone.tensorboard_shim import base_plugin

# Back-compat aliases used by older patches / call sites.
_load_convert_module = load_convert_module
_load_pywrap_module = load_pywrap_module


class _ConvertAdapter:
  """Adapts the injectable ``xspace_to_tool_data`` callable to ConvertPort."""

  def __init__(
      self,
      get_fn: Callable[
          [],
          Callable[
              [Sequence[epath.Path], str, dict[str, Any]],
              tuple[bytes | str | None, str],
          ],
      ],
  ):
    self._get_fn = get_fn

  def xspace_to_tool_data(
      self,
      paths: Sequence[Any],
      tool: str,
      params: Mapping[str, Any],
  ) -> tuple[bytes | str | None, str]:
    return self._get_fn()(paths, tool, params)


class _FsFactoryAdapter:
  """Adapts ``_profile_fs`` to FileSystemFactory."""

  def __init__(self, get_fs: Callable[[str], Any]):
    self._get_fs = get_fs

  def get(self, path: str) -> Any:
    return self._get_fs(path)


class ProfilePlugin(
    StaticApiMixin,
    RunsApiMixin,
    HostsApiMixin,
    DataApiMixin,
    CaptureApiMixin,
    CacheApiMixin,
    base_plugin.TBPlugin,
):
  """Profile Plugin for TensorBoard / XProf UI.

  Construction and route table live here. Per-API handlers are mixins under
  ``api/`` so each frontend surface (static, runs, hosts, data, capture, cache)
  has a single home.
  """

  plugin_name = PLUGIN_NAME


  def __init__(
      self,
      context,
      *,
      epath_module: Any = epath,
      xspace_to_tool_data_fn: Callable[
          [Sequence[epath.Path], str, dict[str, Any]],
          tuple[bytes | str | None, str],
      ]
      | None = None,
      version_module: Any = version,
      cache_generation_executor: concurrent.futures.Executor | None = None,
  ):
    """Constructs a profiler plugin for TensorBoard.

    This plugin adds handlers for performance-related frontends.
    Args:
      context: A base_plugin.TBContext instance.
      epath_module: The epath module to use, can be injected for testing.
      xspace_to_tool_data_fn: Function to convert xspace to tool data.
      version_module: The version module to use, can be injected for testing.
      cache_generation_executor: A `concurrent.futures.Executor` instance for
        async cache generation. If None, a default executor is created.
    """
    self.logdir = context.logdir
    self.data_provider = context.data_provider
    self.master_tpu_unsecure_channel = context.flags.master_tpu_unsecure_channel
    self.hide_capture_profile_button = getattr(
        context, 'hide_capture_profile_button', False
    )
    self.enable_tab_name_label = getattr(
        context, 'enable_tab_name_label', False
    )
    self.src_prefix = getattr(context, 'src_prefix', '')
    self._epath = epath_module
    # May be None until first conversion; resolved lazily via _get_xspace_fn().
    self._xspace_to_tool_data = xspace_to_tool_data_fn
    self._version = version_module
    self._sessions = SessionResolver(
        self._epath, fs_factory=lambda p: self._profile_fs(p)
    )
    self._hosts = HostSelector()
    self._tool_data = ToolDataService(
        convert=_ConvertAdapter(self._get_xspace_fn),
        sessions=self._sessions,
        hosts=self._hosts,
        version=self._version,
        epath_module=self._epath,
        fs_factory=_FsFactoryAdapter(self._profile_fs),
    )

    # Whether the plugin is active. This is an expensive computation, so we
    # compute this asynchronously and cache positive results indefinitely.
    self._is_active = False
    # Lock to ensure at most one thread computes _is_active at a time.
    self._is_active_lock = threading.Lock()
    # Lock to protect access to _run_to_profile_run_dir.
    self._run_dir_cache_lock = threading.Lock()
    # Cache to map profile run name to corresponding tensorboard dir name
    self._run_to_profile_run_dir = {}
    self._run_discovery = RunDiscovery(
        logdir=self.logdir,
        epath_module=self._epath,
        run_dir_cache=self._run_to_profile_run_dir,
        cache_lock=self._run_dir_cache_lock,
        get_all_basenames=lambda p: self._profile_fs(p).get_all_basenames(p),
        get_file_system=self._profile_fs,
    )
    self._tf_profiler = create_tf_profiler()
    # Limit to 1 worker to prevent potential Out-of-Memory (OOM) errors.
    # Cache generation, especially for tools like trace viewer, can be
    # memory-intensive when processing large XPlane files. Running multiple
    # cache generation tasks in parallel increases the risk of excessive
    # memory consumption.
    self._cache_generation_pool = cache_generation_executor or (
        concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix='XprofCacheGen'
        )
    )


  def _get_xspace_fn(self):
    """Resolve the XSpace→tool conversion callable (default loaded lazily)."""
    if self._xspace_to_tool_data is None:
      self._xspace_to_tool_data = _load_convert_module().xspace_to_tool_data
    return self._xspace_to_tool_data


  def _profile_fs(self, path_str: str):
    """Return a ProfileFileSystem for ``path_str`` (local disk or GCS)."""
    return profile_io.get_file_system(path_str, self._epath)


  def _get_xplane_basenames(self, path_str: str) -> Sequence[str]:
    """List ``*.xplane.pb`` / ``*.xplane.riegeli`` basenames under a session dir.

    Kept as a method so tests can simulate I/O failures via patch.
    """
    return self._profile_fs(path_str).get_xplane_basenames(path_str)


  def is_active(self) -> bool:
    """Whether this plugin is active and has any profile data to show.

    Returns:
      Whether any run has profile data.
    """
    if not self._is_active:
      self._is_active = any(self.generate_runs())
    return self._is_active


  def get_plugin_apps(
      self,
  ) -> dict[str, Callable[[wrappers.Request], wrappers.Response]]:
    return {
        route: logging_wrapper(app)
        for route, app in {
            BASE_ROUTE: self.default_handler,
            INDEX_JS_ROUTE: self.static_file_route,
            INDEX_HTML_ROUTE: self.static_file_route,
            BUNDLE_JS_ROUTE: self.static_file_route,
            STYLES_CSS_ROUTE: self.static_file_route,
            MATERIALICONS_WOFF2_ROUTE: self.static_file_route,
            TRACE_VIEWER_INDEX_HTML_ROUTE: self.static_file_route,
            TRACE_VIEWER_INDEX_JS_ROUTE: self.static_file_route,
            TRACE_VIEWER_V2_JS_ROUTE: self.static_file_route,
            TRACE_VIEWER_V2_WASM_ROUTE: self.static_file_route,
            ZONE_JS_ROUTE: self.static_file_route,
            RUNS_ROUTE: self.runs_route,
            RUN_TOOLS_ROUTE: self.run_tools_route,
            HOSTS_ROUTE: self.hosts_route,
            DATA_ROUTE: self.data_route,
            DATA_CSV_ROUTE: self.data_csv_route,
            VERSION_ROUTE: version_route,
            HLO_MODULE_LIST_ROUTE: self.hlo_module_list_route,
            CAPTURE_ROUTE: self.capture_route,
            LOCAL_ROUTE: self.default_handler,
            CONFIG_ROUTE: self.config_route,
            GENERATE_CACHE_ROUTE: self.generate_cache_route,
        }.items()
    }  # pytype: disable=bad-return-type

