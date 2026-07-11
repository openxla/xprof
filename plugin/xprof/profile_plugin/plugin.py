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
"""The TensorBoard plugin for performance profiling.

This module defines the ProfilePlugin façade. Supporting logic lives in
submodules under `xprof.profile_plugin` (tools, http, cache).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
import functools
import concurrent.futures
import os
import threading
from typing import Any, TypedDict

from etils import epath
from werkzeug import wrappers

from xprof import profile_io
from xprof import version
from xprof.profile_plugin.cache.result_cache_policy import write_cache_version_file
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
from xprof.profile_plugin.http.parse_request import tool_request_from_args
from xprof.profile_plugin.http.request_params import generate_csv_filename
from xprof.profile_plugin.http.respond import respond, version_route
from xprof.profile_plugin.logging_config import logger
from xprof.profile_plugin.services.hosts import HostSelector
from xprof.profile_plugin.services.runs import RunDiscovery
from xprof.profile_plugin.services.sessions import SessionResolver
from xprof.profile_plugin.services.tool_data import ToolDataService
from xprof.profile_plugin.tensorflow_bridge import create_tf_profiler
from xprof.profile_plugin.tools.filenames import (
    hosts_from_xplane_filenames,
    parse_filename,
)
from xprof.profile_plugin.tools.registry import (
    DEFAULT_CACHE_TOOLS,
    XPLANE_TOOLS_SET,
)
from xprof.standalone.tensorboard_shim import base_plugin


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


@functools.lru_cache(maxsize=1)
def _load_convert_module():
  """Import convert helpers once (requires generated protos / pywrap)."""
  from xprof.convert import raw_to_tool_data as convert  # pylint: disable=g-import-not-at-top
  return convert


@functools.lru_cache(maxsize=1)
def _load_pywrap_module():
  """Import the native profiler extension once."""
  from xprof.convert import _pywrap_profiler_plugin as pywrap  # pylint: disable=g-import-not-at-top
  return pywrap


HostMetadata = TypedDict('HostMetadata', {'hostname': str})

class ProfilePlugin(base_plugin.TBPlugin):
  """Profile Plugin for TensorBoard."""

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

  # pytype: disable=wrong-arg-types
  @wrappers.Request.application
  def default_handler(self, _: wrappers.Request) -> wrappers.Response:
    contents = self._read_static_file_impl('index.html')
    return respond(contents, 'text/html')

  # pytype: disable=wrong-arg-types
  @wrappers.Request.application
  def config_route(self, _: wrappers.Request) -> wrappers.Response:
    # pytype: enable=wrong-arg-types
    """Returns UI configuration details."""
    config_data = {
        'hideCaptureProfileButton': self.hide_capture_profile_button,
        'srcPathPrefix': self.src_prefix,
        'enableTabNameLabel': self.enable_tab_name_label,
    }
    logger.info('config_route: %s', config_data)
    return respond(config_data, 'application/json')

  def frontend_metadata(self):
    return base_plugin.FrontendMetadata(es_module_path='/index.js')

  def _read_static_file_impl(self, filename: str) -> bytes:
    """Reads contents from a filename.

    Args:
      filename (str): Name of the file.

    Returns:
      Contents of the file.
    Raises:
      IOError: File could not be read or found.
    """
    filepath = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static', filename)

    try:
      with open(filepath, 'rb') as infile:
        contents = infile.read()
    except IOError as io_error:
      raise io_error
    return contents

  # pytype: disable=wrong-arg-types
  @wrappers.Request.application
  def static_file_route(self, request: wrappers.Request) -> wrappers.Response:
    """Handles static files."""
    # pytype: enable=wrong-arg-types
    filename = os.path.basename(request.path)
    extension = os.path.splitext(filename)[1]
    if extension == '.html':
      mimetype = 'text/html'
    elif extension == '.css':
      mimetype = 'text/css'
    elif extension == '.js':
      mimetype = 'application/javascript'
    elif extension == '.wasm':
      mimetype = 'application/wasm'
    else:
      mimetype = 'application/octet-stream'
    try:
      contents = self._read_static_file_impl(filename)
    except IOError:
      return respond('Fail to read the files.', 'text/plain', code=404)
    return respond(contents, mimetype)

  # pytype: disable=wrong-arg-types
  @wrappers.Request.application
  def runs_route(self, request: wrappers.Request) -> wrappers.Response:
    # pytype: enable=wrong-arg-types
    runs = self.runs_imp(request)
    return respond(runs, 'application/json')

  def _session_dir_by_run_name_from_request(
      self, request: wrappers.Request | None = None
  ) -> Mapping[str, str] | None:
    """Returns a map of run names to session directories from the request.

    Args:
      request: Optional; werkzeug request used for grabbing session_path and
        run_path arguments.
    """
    session_path_arg = request.args.get('session_path') if request else None
    run_path_arg = (
        request.args.get('run_path')
        if request and not session_path_arg
        else None
    )
    return self._sessions.run_map_from_params(session_path_arg, run_path_arg)

  def _run_dir(
      self, run: str, request: wrappers.Request | None = None
  ) -> str | None:
    """Helper that maps a frontend run name to a profile "run" directory.

    The frontend run name consists of the TensorBoard run name (aka the relative
    path from the logdir root to the directory containing the data) path-joined
    to the Profile plugin's "run" concept (which is a subdirectory of the
    plugins/profile directory representing an individual run of the tool), with
    the special case that TensorBoard run is the logdir root (which is the run
    named '.') then only the Profile plugin "run" name is used, for backwards
    compatibility.

    Args:
      run: the frontend run name, as described above, e.g. train/run1.
      request: Optional; werkzeug request used for grabbing session_path and
        run_path arguments.

    Returns:
      The resolved directory path, e.g. /logdir/train/plugins/profile/run1.

    Raises:
      ValueError: If the run is not found in the run map.
      RuntimeError: If the run directory is not found.
    """
    session_dir_by_run_name = self._session_dir_by_run_name_from_request(
        request
    )
    return self._sessions.resolve_run_dir(
        run,
        session_dir_by_run_name,
        self.logdir,
        self._run_to_profile_run_dir,
        self._run_dir_cache_lock,
    )

  def runs_imp(self, request: wrappers.Request | None = None) -> list[str]:
    """Returns a list all runs for the profile plugin.

    Args:
      request: Optional; werkzeug request used for grabbing ctx and experiment
        id for other host implementations
    """
    session_dir_by_run_name = self._session_dir_by_run_name_from_request(
        request
    )
    if session_dir_by_run_name is not None:
      runs = session_dir_by_run_name.keys()
    else:
      runs = self.generate_runs()
    return sorted(runs, reverse=True)

  # pytype: disable=wrong-arg-types
  @wrappers.Request.application
  def run_tools_route(self, request: wrappers.Request) -> wrappers.Response:
    # pytype: enable=wrong-arg-types
    run = request.args.get('run')
    run_tools = self.run_tools_imp(run, request)
    return respond(run_tools, 'application/json')

  def run_tools_imp(
      self, run, request: wrappers.Request | None = None
  ) -> list[str]:
    """Returns a list of tools given a single run.

    Args:
      run: the frontend run name, item is list returned by runs_imp
      request: Optional; werkzeug request used for grabbing ctx and experiment
        id for other host implementations
    """
    run_dir = self._run_dir(run, request)
    return list(self.generate_tools_of_run(run, run_dir))

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

  # pytype: disable=wrong-arg-types
  @wrappers.Request.application
  def hosts_route(self, request: wrappers.Request) -> wrappers.Response:
    # pytype: enable=wrong-arg-types
    run = request.args.get('run')
    tool = request.args.get('tag')
    hosts = self.host_impl(run, tool, request)
    return respond(hosts, 'application/json')

  # pytype: disable=wrong-arg-types
  @wrappers.Request.application
  def hlo_module_list_route(
      self, request: wrappers.Request
  ) -> wrappers.Response:
    module_names_str = self.hlo_module_list_impl(request)
    return respond(module_names_str, 'text/plain')

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

  def _write_cache_version_file(self, run_dir: str) -> None:
    """Persist the plugin version next to a session for result-cache invalidation.

    Thin delegate so tests can still patch this method; ToolDataService writes
    via ``result_cache_policy.write_cache_version_file`` directly.
    """
    write_cache_version_file(run_dir, self._version, self._epath)

  def data_impl(
      self, request: wrappers.Request
  ) -> tuple[bytes | str | None, str, str | None]:
    """Retrieves and processes the tool data for a run and a host.

    Args:
      request: XMLHttpRequest

    Returns:
      A string that can be served to the frontend tool or None if tool,
        run or host is invalid.

    Raises:
      FileNotFoundError: If a required xplane file for the specified host(s)
        is not found.
      IOError: If there is an error reading asset directories.
      AttributeError: If there is an error during xplane to tool data conversion
      ValueError: If xplane conversion fails due to invalid data.
      RuntimeError: If session resolution fails when logdir is missing.
    """
    req = tool_request_from_args(request.args)
    result = self._tool_data.get_tool_data(
        req,
        session_path=request.args.get('session_path'),
        run_path=request.args.get('run_path'),
        logdir=self.logdir,
        run_dir_cache=self._run_to_profile_run_dir,
        cache_lock=self._run_dir_cache_lock,
    )
    return result.data, result.content_type, result.content_encoding

  def hlo_module_list_impl(self, request: wrappers.Request) -> str:
    """Returns a string of HLO module names concatenated by comma for the given run."""
    run = request.args.get('run')
    run_dir = self._run_dir(run, request)
    if not run_dir:
      logger.warning('Cannot find asset directory for: %s', run)
      return ''
    try:
      all_basenames = self._profile_fs(run_dir).get_all_basenames(run_dir)
      module_list = [
          name
          for f in all_basenames
          if f.endswith('.hlo_proto.pb') and (name := parse_filename(f)[0])
      ]

      if not module_list:
        xplane_basenames = self._get_xplane_basenames(run_dir)
        xplane_filenames = [os.path.join(run_dir, f) for f in xplane_basenames]
        if xplane_filenames:
          try:
            # This triggers ConvertMultiXSpaceToHloProto in the C++ backend.
            _load_convert_module().xspace_to_tool_names(xplane_filenames)
          except AttributeError:
            logger.warning(
                'XPlane converters are available after Tensorflow 2.4'
            )

          all_basenames = self._profile_fs(run_dir).get_all_basenames(run_dir)
          module_list = [
              name
              for f in all_basenames
              if f.endswith('.hlo_proto.pb') and (name := parse_filename(f)[0])
          ]

      return ','.join(module_list)
    except OSError as e:
      logger.warning('Cannot read asset directory: %s, OpError %r', run_dir, e)
      return ''

  # pytype: disable=wrong-arg-types
  @wrappers.Request.application
  def data_route(self, request: wrappers.Request) -> wrappers.Response:
    """Handlers for data."""
    # pytype: enable=wrong-arg-types
    # params
    #   request: XMLHTTPRequest.
    try:
      data, content_type, content_encoding = self.data_impl(request)
      if data is None:
        return respond('No Data', 'text/plain', code=404)
      return respond(data, content_type, content_encoding=content_encoding)
    # Data fetch error handler
    except TimeoutError as e:
      return respond(str(e), 'text/plain', code=500)
    except AttributeError as e:
      return respond(str(e), 'text/plain', code=500)
    except ValueError as e:
      return respond(str(e), 'text/plain', code=500)
    except FileNotFoundError as e:
      return respond(str(e), 'text/plain', code=500)
    except IOError as e:
      return respond(str(e), 'text/plain', code=500)

  # pytype: disable=wrong-arg-types
  @wrappers.Request.application
  # pytype: enable=wrong-arg-types
  def data_csv_route(self, request: wrappers.Request) -> wrappers.Response:
    """Retrieves tool data and converts it to CSV before responding."""
    try:
      data, content_type, _ = self.data_impl(request)

      if data is None:
        return respond('No Data Found', 'text/plain', code=404)

      if content_type != 'application/json':
        return respond(
            'CSV format not supported for this tool type',
            'text/plain',
            code=400,
        )

      csv_data = _load_convert_module().json_to_csv_string(data)
      filename = generate_csv_filename(request)

      return respond(
          csv_data,
          'text/csv',
          content_encoding=None,
          extra_headers={
              'Content-Disposition': f'attachment; filename="{filename}"'
          },
      )

    except (
        TimeoutError,
        AttributeError,
        ValueError,
        FileNotFoundError,
        IOError,
        TypeError,
    ) as e:
      logger.exception('CSV conversion error')
      return respond(str(e), 'text/plain', code=500)

  # pytype: disable=wrong-arg-types
  @wrappers.Request.application
  def capture_route(self, request: wrappers.Request) -> wrappers.Response:
    # pytype: enable=wrong-arg-types
    return self.capture_route_impl(request)

  def capture_route_impl(self, request: wrappers.Request) -> wrappers.Response:
    """Runs the client trace for capturing profiling information."""
    service_addr = request.args.get('service_addr')
    duration = int(request.args.get('duration', '1000'))
    is_tpu_name = request.args.get('is_tpu_name') == 'true'
    worker_list = request.args.get('worker_list')
    num_tracing_attempts = int(request.args.get('num_retry', '0')) + 1
    options = {
        'host_tracer_level': int(request.args.get('host_tracer_level', '2')),
        'device_tracer_level': int(
            request.args.get('device_tracer_level', '1')
        ),
        'python_tracer_level': int(
            request.args.get('python_tracer_level', '0')
        ),
        'delay_ms': int(request.args.get('delay', '0')),
    }

    if is_tpu_name:
      if self._tf_profiler is None:
        return respond(
            {
                'error': (
                    'TensorFlow is not installed, but is required to use TPU'
                    ' names.'
                )
            },
            'application/json',
            code=500,
        )
      try:
        # Delegate to the helper class for all TF-related logic.
        service_addr, worker_list, master_ip = (
            self._tf_profiler.resolve_tpu_name(service_addr, worker_list or '')
        )
        self.master_tpu_unsecure_channel = master_ip
      except (RuntimeError, ValueError) as err:
        return respond({'error': str(err)}, 'application/json', code=500)

    if not self.logdir:
      return respond(
          {'error': 'logdir is not set, abort capturing.'},
          'application/json',
          code=500,
      )
    try:
      # The core trace call remains, now with cleanly resolved parameters.
      _load_pywrap_module().trace(
          service_addr.removeprefix('grpc://'),
          str(self.logdir),
          worker_list,
          True,
          duration,
          num_tracing_attempts,
          options,
      )
      return respond(
          {'result': 'Capture profile successfully. Please refresh.'},
          'application/json',
      )
    except Exception as e:  # pylint: disable=broad-except
      return respond({'error': str(e)}, 'application/json', code=500)

  def generate_runs(self) -> Iterator[str]:
    """Yield frontend run names under logdir (delegates to RunDiscovery)."""
    yield from self._run_discovery.iter_frontend_runs()

  def generate_tools_of_run(self, run: str, run_dir: str) -> Iterator[str]:
    """Yield tools for a run (delegates to RunDiscovery)."""
    yield from self._run_discovery.tools_of_run(run, run_dir)

  # pytype: disable=wrong-arg-types
  @wrappers.Request.application
  def generate_cache_route(
      self, request: wrappers.Request
  ) -> wrappers.Response:
    # pytype: enable=wrong-arg-types
    """Generates tool data cache in the background."""
    return self._generate_cache_impl(request)

  def _generate_cache_impl(
      self, request: wrappers.Request
  ) -> wrappers.Response:
    """Generates tool data cache in the background.

    Args:
      request: The Werkzeug request object. `request.args` may contain the
        following parameters: - session_path: The path to the session directory
        containing XPlane files. (Required) - tools: An optional comma-separated
        list of tool names to generate cache for. If not provided, defaults to
        `DEFAULT_CACHE_TOOLS`.

    Returns:
      A JSON response indicating whether the task was accepted.
    """
    logger.info('Received generate_cache request.')
    if request.method != 'POST':
      return respond('Method Not Allowed', 'text/plain', code=405)

    params = request.args
    session_path = params.get('session_path')
    logger.info('generate_cache called with params: %s', params)

    if not session_path:
      return respond('Missing "session_path" parameter', 'text/plain', code=400)

    try:
      path = self._epath.Path(session_path)
      xplane_basenames = self._get_xplane_basenames(session_path)
      asset_paths = sorted(
          str(path / basename) for basename in xplane_basenames
      )
      if not asset_paths:
        return respond(
            'No XPlane files found in session_path', 'text/plain', code=404
        )
      logger.info(
          'Found %d *.xplane.pb files in %s.', len(asset_paths), session_path
      )
    except OSError as e:
      logger.exception('Error listing files in session_path: %s', session_path)
      return respond(
          f'Error listing files in session_path: {e!r}', 'text/plain', code=500
      )

    runs = self.runs_imp(request)
    if len(runs) != 1:
      # When 'session_path' is provided, runs_imp should return exactly one run
      # corresponding to the session_path's name.
      return respond(
          'Expected exactly one run for the provided session_path, but found'
          ' %d: %s. Please ensure session_path points to a valid profile run'
          ' directory.' % (len(runs), runs),
          'text/plain',
          code=400,
      )
    run_name = runs[0]
    logger.info(
        'Querying available tools for run %s via run_tools_imp.',
        run_name,
    )
    available_run_tools = set(self.run_tools_imp(run_name, request))
    logger.info(
        'Discovered tools for cache generation: %s for run %s',
        available_run_tools,
        run_name,
    )

    tools_str = params.get('tools')
    requested_tools = (
        set(t.strip() for t in tools_str.split(',') if t.strip())
        if tools_str
        else set(DEFAULT_CACHE_TOOLS)
    )
    if tools_str:
      logger.info('Request tools for cache generation: %s', requested_tools)
    else:
      logger.info(
          'No tools specified in request, using default tools: %s',
          DEFAULT_CACHE_TOOLS,
      )

    available_xplane_tools = available_run_tools.intersection(XPLANE_TOOLS_SET)

    filtered_tools = requested_tools.intersection(available_xplane_tools)

    skipped_tools = requested_tools.difference(filtered_tools)
    for tool in skipped_tools:
      if tool not in available_run_tools:
        logger.info(
            'Tool %s was requested for caching but is not available for run %s,'
            ' skipping.',
            tool,
            run_name,
        )
      else:
        logger.warning(
            'Tool %s is available for run %s but not in XPLANE_TOOLS_SET,'
            ' skipping cache generation.',
            tool,
            run_name,
        )

    if not filtered_tools:
      return respond(
          'No valid XPlane tools found or specified for caching in run %s.'
          % run_name,
          'text/plain',
          code=400,
      )

    logger.info(
        'Filtered tools for cache generation: %s for session %s',
        filtered_tools,
        session_path,
    )

    try:
      logger.info(
          'Submitting cache generation task to thread pool for session %s...',
          session_path,
      )
      self._cache_generation_pool.submit(
          self._generate_cache_task,
          asset_paths=asset_paths,
          tool_list=sorted(list(filtered_tools)),
          params=params,
          session_path=session_path,
      )
    except RuntimeError as e:
      logger.exception(
          'Failed to schedule cache generation task for session_path: %s',
          session_path,
      )
      return respond(f'Failed to schedule task: {e!r}', 'text/plain', code=500)
    else:
      return respond(
          {'status': 'ACCEPTED', 'message': 'Cache generation started'},
          'application/json',
          code=202,
      )

  def _generate_cache_task(
      self,
      *,
      asset_paths: Sequence[str],
      tool_list: Sequence[str],
      params: Mapping[str, Any],
      session_path: str,
  ) -> None:
    """Generates and caches tool data from XPlane files in a background thread.

    Args:
      asset_paths: A list of paths to the XPlane files.
      tool_list: A list of tool names for which to generate cache.
      params: Additional parameters from the request.
      session_path: The path to the session directory.
    """
    logger.info(
        'Background cache generation task started for tools: %s', tool_list
    )
    logger.info('Writing cache version file to %s', session_path)
    self._write_cache_version_file(session_path)

    filenames = [os.path.basename(p) for p in asset_paths]

    base_tool_params = dict(params)

    for tool in tool_list:
      try:
        logger.info('Generating cache for tool %s...', tool)
        tool_params = base_tool_params.copy()
        tool_params['hosts'] = hosts_from_xplane_filenames(filenames, tool)
        self._get_xspace_fn()(
            [self._epath.Path(p) for p in asset_paths], tool, tool_params
        )
        logger.info(
            'Successfully generated cache for tool %s for %d files.',
            tool,
            len(asset_paths),
        )
      # Catch all exceptions to prevent the background thread from crashing.
      # This ensures that even if one tool fails to generate, other tools
      # can still be processed. The error is logged for debugging.
      except (AttributeError, ValueError, OSError):
        logger.exception(
            'Background cache generation failed for tool %s in session %s',
            tool,
            session_path,
        )
      except Exception:  # pylint: disable=broad-except
        logger.exception(
            'Unexpected error during background cache generation for tool %s'
            ' in session %s',
            tool,
            session_path,
        )
