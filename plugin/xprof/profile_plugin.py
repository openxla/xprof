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
"""The TensorBoard plugin for performance profiling."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections.abc import Callable, Iterator, Mapping
import gzip
import json
import logging
import os
import re
import sys
import threading
from typing import Any, Dict, List, Optional, Sequence, TypedDict

from etils import epath
import etils.epath.backend
from fsspec import core
import six
from werkzeug import wrappers

from xprof import version
from xprof.convert import raw_to_tool_data as convert
from xprof.standalone.tensorboard_shim import base_plugin
from xprof.standalone.tensorboard_shim import plugin_asset_util
from xprof.convert import _pywrap_profiler_plugin

logger = logging.getLogger('tensorboard.plugins.profile')
logger.setLevel(logging.INFO)
if not logger.handlers:
  handler = logging.StreamHandler(sys.stderr)
  formatter = logging.Formatter(
      '%(levelname)s %(asctime)s [%(filename)s:%(lineno)d] %(message)s'
  )
  handler.setFormatter(formatter)
  logger.addHandler(handler)
  logger.propagate = False

try:
  import tensorflow.compat.v2 as tf  # pylint: disable=g-import-not-at-top # pytype: disable=import-error

  tf.enable_v2_behavior()
except ImportError:
  logger.info(
      'Disabling some remote capture features as tensorflow is not available'
  )
  tf = None


# The prefix of routes provided by this plugin.
TB_NAME = 'plugins'
PLUGIN_NAME = 'profile'

BASE_ROUTE = '/'
INDEX_JS_ROUTE = '/index.js'
INDEX_HTML_ROUTE = '/index.html'
BUNDLE_JS_ROUTE = '/bundle.js'
STYLES_CSS_ROUTE = '/styles.css'
MATERIALICONS_WOFF2_ROUTE = '/materialicons.woff2'
TRACE_VIEWER_INDEX_HTML_ROUTE = '/trace_viewer_index.html'
TRACE_VIEWER_INDEX_JS_ROUTE = '/trace_viewer_index.js'
ZONE_JS_ROUTE = '/zone.js'
DATA_ROUTE = '/data'
RUNS_ROUTE = '/runs'
RUN_TOOLS_ROUTE = '/run_tools'
HOSTS_ROUTE = '/hosts'
HLO_MODULE_LIST_ROUTE = '/module_list'
CAPTURE_ROUTE = '/capture_profile'
LOCAL_ROUTE = '/local'
CONFIG_ROUTE = '/config'
CACHE_VERSION_FILE = 'cache_version.txt'

# Suffixes of "^, #, @" symbols represent different input data formats for the
# same tool.
# 1) '^': data generate from XPlane.
# 2) '#': data is in gzip format.
# 3) '@': data generate from proto, or tracetable for streaming trace viewer.
# 4) no suffix: data is in json format, ready to feed to frontend.
TOOLS = {
    'xplane': 'xplane.pb',
    'hlo_proto': 'hlo_proto.pb',
}

ALL_HOSTS = 'ALL_HOSTS'

HostMetadata = TypedDict('HostMetadata', {'hostname': str})

_EXTENSION_TO_TOOL = {extension: tool for tool, extension in TOOLS.items()}

_FILENAME_RE = re.compile(r'(?:(.*)\.)?(' +
                          '|'.join(TOOLS.values()).replace('.', r'\.') + r')')


# Tools that can be generated from xplane end with ^.
XPLANE_TOOLS = [
    'trace_viewer',  # non-streaming before TF 2.13
    'trace_viewer@',  # streaming since TF 2.14
    'overview_page',
    'input_pipeline_analyzer',
    'framework_op_stats',
    'kernel_stats',
    'memory_profile',
    'pod_viewer',
    'op_profile',
    'hlo_stats',
    'roofline_model',
    'inference_profile',
    'memory_viewer',
    'graph_viewer',
    'megascale_stats',
]

# XPlane generated tools that support all host mode.
XPLANE_TOOLS_ALL_HOSTS_SUPPORTED = frozenset([
    'input_pipeline_analyzer',
    'framework_op_stats',
    'kernel_stats',
    'overview_page',
    'pod_viewer',
    'megascale_stats',
])

# XPlane generated tools that only support all host mode.
XPLANE_TOOLS_ALL_HOSTS_ONLY = frozenset(
    ['overview_page', 'pod_viewer'])

# Rate limiter constants, the GCS quota defined below
# https://cloud.google.com/storage/quotas#rate-quotas.
# currently set to 1000 request per minute.
# TODO(kcai): The assumption on the average number of subdirs is not
# always true. If this is not sufficient, we can consider a token-based
# approach that counts the number of subdirs after calling iterdir.
MAX_GCS_REQUESTS = 1000
LIMIT_WINDOW_SECONDS = 60
AVERAGE_SUBDIR_NUMBER = 10


def use_xplane(tool: str) -> bool:
  return tool in XPLANE_TOOLS


# HLO generated tools.
HLO_TOOLS = frozenset(['graph_viewer', 'memory_viewer'])


def use_hlo(tool: str) -> bool:
  return tool in HLO_TOOLS


def make_filename(host: str, tool: str) -> str:
  """Returns the name of the file containing data for the given host and tool.

  Args:
    host: Name of the host that produced the profile data, e.g., 'localhost'.
    tool: Name of the tool, e.g., 'trace_viewer'.

  Returns:
    The host name concatenated with the tool-specific extension, e.g.,
    'localhost.trace'.
  """
  filename = str(host) + '.' if host else ''
  if use_hlo(tool):
    tool = 'hlo_proto'
  elif use_xplane(tool):
    tool = 'xplane'
  return filename + TOOLS[tool]


def _parse_filename(filename: str) -> tuple[Optional[str], Optional[str]]:
  """Returns the host and tool encoded in a filename in the run directory.

  Args:
    filename: Name of a file in the run directory. The name might encode a host
      and tool, e.g., 'host.tracetable', 'host.domain.op_profile.json', or just
      a tool, e.g., 'trace', 'tensorflow_stats.pb'.

  Returns:
    A tuple (host, tool) containing the names of the host and tool, e.g.,
    ('localhost', 'trace_viewer'). Either of the tuple's components can be None.
  """
  m = _FILENAME_RE.fullmatch(filename)
  if m is None:
    return filename, None
  return m.group(1), _EXTENSION_TO_TOOL[m.group(2)]


def _get_hosts(filenames: list[str]) -> set[str]:
  """Parses a list of filenames and returns the set of hosts.

  Args:
    filenames: A list of filenames (just basenames, no directory).

  Returns:
    A set of host names encoded in the filenames.
  """
  hosts = set()
  for name in filenames:
    host, _ = _parse_filename(name)
    if host:
      hosts.add(host)
  return hosts


def _get_tools(filenames: list[str], profile_run_dir: str) -> set[str]:
  """Parses a list of filenames and returns the set of tools.

  If xplane is present in the repository, add tools that can be generated by
  xplane if we don't have a file for the tool.

  Args:
    filenames: A list of filenames.
    profile_run_dir: The run directory of the profile.

  Returns:
    A set of tool names encoded in the filenames.
  """
  tools = set()
  found = set()
  xplane_filenames = []
  for name in filenames:
    _, tool = _parse_filename(name)
    if tool == 'xplane':
      xplane_filenames.append(os.path.join(profile_run_dir, name))
      continue
    elif tool == 'hlo_proto':
      continue
    elif tool:
      tools.add(tool)
      if tool[-1] in ('@'):
        found.add(tool[:-1])
      else:
        found.add(tool)
  # profile_run_dir might be empty, like in cloud AI use case.
  if not profile_run_dir:
    if xplane_filenames:
      for item in XPLANE_TOOLS:
        if item[:-1] not in found:
          tools.add(item)
  else:
    try:
      if xplane_filenames:
        return set(convert.xspace_to_tool_names(xplane_filenames))
    except AttributeError:
      logger.warning('XPlane converters are available after Tensorflow 2.4')
  return tools


def respond(
    body: Any,
    content_type: str,
    code: int = 200,
    content_encoding: Optional[tuple[str, str]] = None,
) -> wrappers.Response:
  """Create a Werkzeug response, handling JSON serialization and CSP.

  Args:
    body: For JSON responses, a JSON-serializable object; otherwise, a raw
      `bytes` string or Unicode `str` (which will be encoded as UTF-8).
    content_type: Response content-type (`str`); use `application/json` to
      automatically serialize structures.
    code: HTTP status code (`int`).
    content_encoding: Response Content-Encoding header ('str'); e.g. 'gzip'. If
      the content type is not set, The data would be compressed and the content
      encoding would be set to gzip.

  Returns:
    A `werkzeug.wrappers.Response` object.
  """
  if content_type == 'application/json' and isinstance(
      body, (dict, list, set, tuple)):
    body = json.dumps(body, sort_keys=True)
  if not isinstance(body, bytes):
    body = body.encode('utf-8')
  csp_parts = {
      'default-src': ["'self'"],
      'script-src': [
          "'self'",
          "'unsafe-eval'",
          "'unsafe-inline'",
          'https://www.gstatic.com',
      ],
      'object-src': ["'none'"],
      'style-src': [
          "'self'",
          "'unsafe-inline'",
          'https://fonts.googleapis.com',
          'https://www.gstatic.com',
      ],
      'font-src': [
          "'self'",
          'https://fonts.googleapis.com',
          'https://fonts.gstatic.com',
          'data:',
      ],
      'connect-src': [
          "'self'",
          'data:',
          'www.gstatic.com',
      ],
      'img-src': [
          "'self'",
          'blob:',
          'data:',
      ],
      'script-src-elem': [
          "'self'",
          "'unsafe-inline'",
          # Remember to restrict on integrity when importing from jsdelivr
          # Whitelist this domain to support hlo_graph_dumper html format
          'https://cdn.jsdelivr.net/npm/',
          'https://www.gstatic.com',
      ],
  }
  csp = ';'.join((' '.join([k] + v) for (k, v) in csp_parts.items()))
  headers = [
      ('Content-Security-Policy', csp),
      ('X-Content-Type-Options', 'nosniff'),
  ]
  if content_encoding:
    headers.append(('Content-Encoding', content_encoding))
  else:
    headers.append(('Content-Encoding', 'gzip'))
    body = gzip.compress(body)
  return wrappers.Response(
      body, content_type=content_type, status=code, headers=headers
  )


def _plugin_assets(
    session_dir: str, runs: list[str], plugin_name: str
) -> dict[str, list[str]]:
  result = {}
  for run in runs:
    run_path = _tb_run_directory(session_dir, run)
    assets = plugin_asset_util.ListAssets(run_path, plugin_name)
    result[run] = assets
  return result


def _tb_run_directory(session_dir: str, run: str) -> str:
  """Returns the TensorBoard run directory for a TensorBoard run name.

  This helper returns the TensorBoard-level run directory (the one that would)
  contain tfevents files) for a given TensorBoard run name (aka the relative
  path from the session_dir root to this directory). For the root run '.'
  this is the bare session_dir path; for all other runs this is the
  session_dir joined with the run name.

  Args:
    session_dir: the TensorBoard log directory root path
    run: the TensorBoard run name, e.g. '.' or 'train'

  Returns:
    The TensorBoard run directory path, e.g. my/session_dir or
    my/session_dir/train.
  """
  return session_dir if run == '.' else os.path.join(session_dir, run)


def filenames_to_hosts(filenames: list[str], tool: str) -> list[str]:
  """Convert a list of filenames to a list of host names given a tool.

  Args:
    filenames: A list of filenames.
    tool: A string representing the profiling tool.

  Returns:
    A list of hostnames.
  """
  hosts = _get_hosts(filenames)
  if len(hosts) > 1:
    if tool in XPLANE_TOOLS_ALL_HOSTS_ONLY:
      hosts = [ALL_HOSTS]
    elif tool in XPLANE_TOOLS_ALL_HOSTS_SUPPORTED:
      hosts.add(ALL_HOSTS)
  return sorted(hosts)


def _get_bool_arg(
    args: Mapping[str, Any], arg_name: str, default: bool
) -> bool:
  """Gets a boolean argument from a request.

  Args:
    args: The werkzeug request arguments.
    arg_name: The name of the argument.
    default: The default value if the argument is not present.

  Returns:
    The boolean value of the argument.
  """
  arg_str = args.get(arg_name)
  if arg_str is None:
    return default
  return arg_str.lower() == 'true'


class ToolsCache:
  """Caches the list of tools for a profile run based on file content hashes or mtimes.

  Attributes:
    CACHE_FILE_NAME: The name of the cache file.
    CACHE_VERSION: The version of the cache format.
  """

  CACHE_FILE_NAME = '.cached_tools.json'
  CACHE_VERSION = 1

  def __init__(self, profile_run_dir: epath.Path):
    """Initializes the ToolsCache.

    Args:
      profile_run_dir: The directory containing the profile run data.
    """
    self._profile_run_dir = profile_run_dir
    self._cache_file = self._profile_run_dir / self.CACHE_FILE_NAME
    logger.info('ToolsCache initialized for %s', self._cache_file)

  def _get_local_file_identifier(self, file_path_str: str) -> Optional[str]:
    """Gets a string identifier for a local file.

    The identifier is a combination of the file's last modification time (mtime)
    and size, in the format "{mtime}-{size}".

    Args:
      file_path_str: The absolute path to the local file.

    Returns:
      A string identifier, or None if the file is not found or an error occurs.
    """
    try:
      stat_result = os.stat(file_path_str)
      return f'{int(stat_result.st_mtime)}-{stat_result.st_size}'
    except FileNotFoundError:
      logger.warning('Local file not found: %s', file_path_str)
      return None
    except OSError as e:
      logger.error(
          'OSError getting stat for local file %s: %r', file_path_str, e
      )
      return None

  def _get_gcs_file_hash(self, file_path_str: str) -> Optional[str]:
    """Gets the MD5 hash for a GCS file.

    Args:
      file_path_str: The GCS path (e.g., "gs://bucket/object").

    Returns:
      The MD5 hash string, or None if the file is not found or an error occurs.
    """
    try:
      fs = core.get_fs_token_paths(file_path_str)[0]
      info = fs.info(file_path_str)
      md5_hash = info.get('md5Hash')

      if not isinstance(md5_hash, str):
        logger.warning(
            'Could not find a valid md5Hash string in info for %s: %s',
            file_path_str,
            info,
        )
        return None

      return md5_hash

    except FileNotFoundError:
      logger.warning('GCS path not found: %s', file_path_str)
      return None
    except IndexError:
      logger.error('Could not get filesystem for GCS path: %s', file_path_str)
      return None
    except Exception as e:  # pylint: disable=broad-exception-caught
      logger.exception(
          'Unexpected error getting hash for GCS path %s: %r', file_path_str, e
      )
      return None

  def get_file_identifier(self, file_path_str: str) -> Optional[str]:
    """Gets a string identifier for a file.

    For GCS files, this is the MD5 hash.
    For local files, this is a string combining mtime and size.

    Args:
      file_path_str: The full path to the file (local or GCS).

    Returns:
      A string identifier, or None if an error occurs.
    """
    if file_path_str.startswith('gs://'):
      return self._get_gcs_file_hash(file_path_str)
    else:
      return self._get_local_file_identifier(file_path_str)

  def _get_current_xplane_file_states(self) -> Optional[Dict[str, str]]:
    """Gets the current state of XPlane files in the profile run directory.

    Returns:
      A dictionary mapping filename to a string identifier (hash or mtime-size),
      or None if any file state cannot be determined.
    """
    try:
      file_identifiers = {}
      for xplane_file in self._profile_run_dir.glob(f"*.{TOOLS['xplane']}"):
        file_id = self.get_file_identifier(str(xplane_file))
        if file_id is None:
          logger.warning(
              'Could not get identifier for %s, cache will be invalidated.',
              xplane_file,
          )
          return None
        file_identifiers[xplane_file.name] = file_id
      return file_identifiers
    except OSError as e:
      logger.warning('Could not glob files in %s: %r', self._profile_run_dir, e)
      return None

  def load(self) -> Optional[List[str]]:
    """Loads the cached list of tools if the cache is valid.

    The cache is valid if the cache file exists, the version matches, and
    the file states (hashes/mtimes) of the XPlane files have not changed.

    Returns:
      A list of tool names if the cache is valid, otherwise None.
    """
    try:
      with self._cache_file.open('r') as f:
        cached_data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
      logger.warning(
          'Error reading or decoding cache file %s: %r, invalidating.',
          self._cache_file,
          e,
      )
      self.invalidate()
      return None

    if cached_data.get('version') != self.CACHE_VERSION:
      logger.info(
          'ToolsCache invalid: version mismatch, expected %s, got %s.'
          ' Invalidating %s',
          self.CACHE_VERSION,
          cached_data.get('version'),
          self._cache_file,
      )
      self.invalidate()
      return None

    current_files = self._get_current_xplane_file_states()
    if current_files is None:
      logger.info(
          'ToolsCache invalid: could not determine current file states.'
          ' Invalidating %s',
          self._cache_file,
      )
      self.invalidate()
      return None

    if cached_data.get('files') != current_files:
      logger.info(
          'ToolsCache invalid: file states differ. Invalidating %s',
          self._cache_file,
      )
      self.invalidate()
      return None

    logger.info('ToolsCache hit: %s', self._cache_file)
    return cached_data.get('tools')

  def save(self, tools: Sequence[str]) -> None:
    """Saves the list of tools and the current file states to the cache file.

    Args:
      tools: The list of tool names to cache.
    """
    current_files_for_cache = self._get_current_xplane_file_states()
    if current_files_for_cache is None:
      logger.warning(
          'ToolsCache not saved: could not get file states %s', self._cache_file
      )
      return

    new_cache_data = {
        'version': self.CACHE_VERSION,
        'files': current_files_for_cache,
        'tools': tools,
    }
    try:
      with self._cache_file.open('w') as f:
        json.dump(new_cache_data, f, sort_keys=True, indent=2)
      logger.info('ToolsCache saved: %s', self._cache_file)
    except (OSError, TypeError) as e:
      logger.error('Error writing cache file %s: %r', self._cache_file, e)

  def invalidate(self) -> None:
    """Deletes the cache file, forcing regeneration on the next load."""
    try:
      self._cache_file.unlink()
      logger.info('ToolsCache invalidated: %s', self._cache_file)
    except FileNotFoundError:
      pass
    except OSError as e:
      logger.error('Error removing cache file %s: %r', self._cache_file, e)


class _TfProfiler:
  """A helper class to encapsulate all TensorFlow-dependent profiler logic."""

  def __init__(self, tf_module):
    if not tf_module:
      raise ImportError('TensorFlow module is not available.')
    self.tf = tf_module

  def _get_worker_list(self, cluster_resolver) -> str:
    """Parses TPU workers list from the cluster resolver."""
    cluster_spec = cluster_resolver.cluster_spec()
    task_indices = cluster_spec.task_indices('worker')
    worker_list = [
        cluster_spec.task_address('worker', i).replace(':8470', ':8466')
        for i in task_indices
    ]
    return ','.join(worker_list)

  def resolve_tpu_name(
      self, tpu_name: str, worker_list: str
  ) -> tuple[str, str, str]:
    """Resolves a TPU name to its master IP, service address, and worker list.

    Args:
      tpu_name: The name of the TPU to resolve.
      worker_list: A comma-separated list of worker addresses.

    Returns:
      A tuple containing (service_addr, worker_list, master_ip).
    """
    try:
      resolver = self.tf.distribute.cluster_resolver.TPUClusterResolver(
          tpu_name
      )
      master_grpc_addr = resolver.get_master()
    except RuntimeError as err:
      # Propagate error to be handled by the caller.
      raise RuntimeError(
          f'Error initializing TPUClusterResolver: {err}'
      ) from err
    except (ValueError, TypeError) as e:
      # Handle cases where the TPU name is invalid.
      raise ValueError(f'No TPU found with the name: {tpu_name}') from e

    if not worker_list:
      worker_list = self._get_worker_list(resolver)

    # TPU cluster resolver always returns port 8470. Replace it with 8466
    # on which profiler service is running.
    master_ip = master_grpc_addr.replace('grpc://', '').replace(':8470', '')
    service_addr = f'{master_ip}:8466'
    return service_addr, worker_list, master_ip


class ProfilePlugin(base_plugin.TBPlugin):
  """Profile Plugin for TensorBoard."""
  plugin_name = PLUGIN_NAME

  def __init__(self, context):
    """Constructs a profiler plugin for TensorBoard.

    This plugin adds handlers for performance-related frontends.
    Args:
      context: A base_plugin.TBContext instance.
    """
    self.logdir = context.logdir
    self.data_provider = context.data_provider
    self.master_tpu_unsecure_channel = context.flags.master_tpu_unsecure_channel
    self.hide_capture_profile_button = getattr(
        context, 'hide_capture_profile_button', False
    )

    # Whether the plugin is active. This is an expensive computation, so we
    # compute this asynchronously and cache positive results indefinitely.
    self._is_active = False
    # Lock to ensure at most one thread computes _is_active at a time.
    self._is_active_lock = threading.Lock()
    # Cache to map profile run name to corresponding tensorboard dir name
    self._run_to_profile_run_dir = {}
    self._tf_profiler = _TfProfiler(tf) if tf else None

  def is_active(self) -> bool:
    """Whether this plugin is active and has any profile data to show.

    Returns:
      Whether any run has profile data.
    """
    if not self._is_active:
      self._is_active = any(self.generate_runs())
    return self._is_active

  def _does_tool_support_multi_hosts_processing(self, tool: str) -> bool:
    """Returns true if the tool supports multi-hosts processing."""
    return tool == 'trace_viewer@' or tool == 'trace_viewer'

  def get_plugin_apps(
      self,
  ) -> dict[str, Callable[[wrappers.Request], wrappers.Response]]:
    return {
        BASE_ROUTE: self.default_handler,
        INDEX_JS_ROUTE: self.static_file_route,
        INDEX_HTML_ROUTE: self.static_file_route,
        BUNDLE_JS_ROUTE: self.static_file_route,
        STYLES_CSS_ROUTE: self.static_file_route,
        MATERIALICONS_WOFF2_ROUTE: self.static_file_route,
        TRACE_VIEWER_INDEX_HTML_ROUTE: self.static_file_route,
        TRACE_VIEWER_INDEX_JS_ROUTE: self.static_file_route,
        ZONE_JS_ROUTE: self.static_file_route,
        RUNS_ROUTE: self.runs_route,
        RUN_TOOLS_ROUTE: self.run_tools_route,
        HOSTS_ROUTE: self.hosts_route,
        DATA_ROUTE: self.data_route,
        HLO_MODULE_LIST_ROUTE: self.hlo_module_list_route,
        CAPTURE_ROUTE: self.capture_route,
        LOCAL_ROUTE: self.default_handler,
        CONFIG_ROUTE: self.config_route,
    }

  # pytype: disable=wrong-arg-types
  @wrappers.Request.application
  def default_handler(self, _: wrappers.Request) -> wrappers.Response:
    contents = self._read_static_file_impl('index.html')
    return respond(contents, 'text/html')

  # pytype: disable=wrong-arg-types
  @wrappers.Request.application
  def config_route(self, request: wrappers.Request) -> wrappers.Response:
    # pytype: enable=wrong-arg-types
    """Returns UI configuration details."""
    logger.info('config_route: %s', self.logdir)
    config_data = {
        'hideCaptureProfileButton': self.hide_capture_profile_button,
    }
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
    filepath = os.path.join(os.path.dirname(__file__), 'static', filename)

    try:
      with open(filepath, 'rb') as infile:
        contents = infile.read()
    except IOError as io_error:
      raise io_error
    return contents

  # pytype: disable=wrong-arg-types
  @wrappers.Request.application
  def static_file_route(self, request: wrappers.Request) -> wrappers.Response:
    # pytype: enable=wrong-arg-types
    filename = os.path.basename(request.path)
    extention = os.path.splitext(filename)[1]
    if extention == '.html':
      mimetype = 'text/html'
    elif extention == '.css':
      mimetype = 'text/css'
    elif extention == '.js':
      mimetype = 'application/javascript'
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

  def _run_map_from_request(
      self, request: Optional[wrappers.Request] = None
  ) -> Optional[dict[str, str]]:
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
    run_map = None
    if session_path_arg:
      session_path = epath.Path(session_path_arg)
      run_name = session_path.name
      run_map = {}
      if session_path.is_dir() and any(session_path.glob('*.xplane.pb')):
        run_map[run_name] = str(session_path)
    elif run_path_arg:
      run_path = epath.Path(run_path_arg)
      run_map = {}
      for session in run_path.iterdir():
        if session.is_dir() and any(session.glob('*.xplane.pb')):
          run_map[session.name] = str(session)
    return run_map

  def _run_dir(
      self, run: str, request: Optional[wrappers.Request] = None
  ) -> Optional[str]:
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
    run_map = self._run_map_from_request(request)
    if run_map is not None:
      if run in run_map:
        return run_map[run]
      else:
        raise ValueError(f'Run {run} not found in run map: {run_map}')

    if run in self._run_to_profile_run_dir:
      return self._run_to_profile_run_dir[run]

    if not self.logdir:
      raise RuntimeError(
          'No matching run directory for run %s. Logdir is empty.' % run
      )
    tb_run_name, profile_run_name = os.path.split(run.rstrip(os.sep))
    if not tb_run_name:
      tb_run_name = '.'
    tb_run_directory = _tb_run_directory(self.logdir, tb_run_name)
    if not epath.Path(tb_run_directory).is_dir():
      raise RuntimeError('No matching run directory for run %s' % run)
    plugin_directory = plugin_asset_util.PluginDirectory(
        tb_run_directory, PLUGIN_NAME
    )
    return os.path.join(plugin_directory, profile_run_name)

  def runs_imp(self, request: Optional[wrappers.Request] = None) -> list[str]:
    """Returns a list all runs for the profile plugin.

    Args:
      request: Optional; werkzeug request used for grabbing ctx and experiment
        id for other host implementations
    """
    run_map = self._run_map_from_request(request)
    if run_map is not None:
      runs = run_map.keys()
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
      self, run, request: Optional[wrappers.Request] = None
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
  ) -> List[HostMetadata]:
    if not run_dir:
      logger.warning('Cannot find asset directory for: %s', run)
      return []
    tool_pattern = '*.xplane.pb'
    filenames = []
    try:
      path = epath.Path(run_dir)
      filenames = path.glob(tool_pattern)
    except OSError as e:
      logger.warning('Cannot read asset directory: %s, OpError %s', run_dir, e)
    filenames = [os.fspath(os.path.basename(f)) for f in filenames]

    return [{'hostname': host} for host in filenames_to_hosts(filenames, tool)]

  def host_impl(
      self, run: str, tool: str, request: Optional[wrappers.Request] = None
  ) -> List[HostMetadata]:
    """Returns available hosts and their metadata for the run and tool in the log directory.

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
  ) -> tuple[List[str], List[epath.Path]]:
    """Retrieves and validates the hosts and asset paths for a run and tool.

    Args:
      run_dir: The run directory.
      run: The frontend run name.
      tool: The requested tool.
      hosts_param: Comma-separated list of selected hosts.
      host: The single host parameter.

    Returns:
      A tuple containing (selected_hosts, asset_paths).

    Raises:
      FileNotFoundError: If a required xplane file for the specified host(s)
        is not found.
      IOError: If there is an error reading asset directories.
    """
    asset_paths = []
    selected_hosts = []
    all_xplane_files = {}  # Map host to path

    # Find all available xplane files for the run and map them by host.
    file_pattern = make_filename('*', 'xplane')
    try:
      path = epath.Path(run_dir)
      for xplane_path in path.glob(file_pattern):
        host_name, _ = _parse_filename(xplane_path.name)
        if host_name:
          all_xplane_files[host_name] = xplane_path
    except OSError as e:
      logger.warning('Cannot read asset directory: %s, OpError %r', run_dir, e)
      raise IOError(
          'Cannot read asset directory: %s, OpError %r' % (run_dir, e)
      ) from e

    if hosts_param and self._does_tool_support_multi_hosts_processing(tool):
      selected_hosts = hosts_param.split(',')
      for selected_host in selected_hosts:
        if selected_host in all_xplane_files:
          asset_paths.append(all_xplane_files[selected_host])
        else:
          raise FileNotFoundError(
              'No xplane file found for host: %s in run: %s'
              % (selected_host, run)
          )
    elif host == ALL_HOSTS:
      asset_paths = list(all_xplane_files.values())
      selected_hosts = list(all_xplane_files.keys())
    elif host and host in all_xplane_files:
      selected_hosts = [host]
      asset_paths = [all_xplane_files[host]]
    elif host:
      logger.warning('No xplane file found for host: %s in run: %s', host, run)
      if host not in XPLANE_TOOLS_ALL_HOSTS_ONLY:
        raise FileNotFoundError(
            'No xplane file found for host: %s in run: %s' % (host, run)
        )
    elif not host and not hosts_param and len(all_xplane_files) == 1:
      selected_hosts = list(all_xplane_files.keys())
      asset_paths = list(all_xplane_files.values())

    if not asset_paths:
      logger.warning(
          'No matching asset paths found for run %s, tool %s, host(s) %s / %s',
          run,
          tool,
          hosts_param,
          host,
      )
      if not host and tool not in XPLANE_TOOLS_ALL_HOSTS_ONLY:
        raise FileNotFoundError(
            'Host must be specified for tool %s in run %s' % (tool, run)
        )

    return selected_hosts, asset_paths

  def data_impl(
      self, request: wrappers.Request
  ) -> tuple[Optional[str], str, Optional[str]]:
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
    """
    run = request.args.get('run')
    tool = request.args.get('tag')
    hosts_param = request.args.get('hosts')
    host = request.args.get('host')
    module_name = request.args.get('module_name')
    tqx = request.args.get('tqx')
    use_saved_result = _get_bool_arg(request.args, 'use_saved_result', True)
    full_dma = _get_bool_arg(request.args, 'full_dma', False)
    run_dir = self._run_dir(run, request)

    # Check if the cache file exists and if the cache file version is less
    # than the current plugin version, clear the cache.
    try:
      if epath.Path(os.path.join(run_dir, CACHE_VERSION_FILE)).exists():
        with epath.Path(os.path.join(run_dir, CACHE_VERSION_FILE)).open(
            'r'
        ) as f:
          cache_version = f.read().strip()
          if cache_version < version.__version__:
            use_saved_result = False
      else:
        use_saved_result = False
    except OSError as e:
      logger.warning('Cannot read cache version file: %r', e)
      use_saved_result = False

    graph_viewer_options = self._get_graph_viewer_options(request)
    # Host param is used by HLO tools to identify the module.
    params = {
        'graph_viewer_options': graph_viewer_options,
        'tqx': tqx,
        'host': host,
        'module_name': module_name,
        'use_saved_result': use_saved_result,
    }
    if request.args.get('group_by'):
      params['group_by'] = request.args.get('group_by')
    content_type = 'application/json'

    if tool not in TOOLS and not use_xplane(tool):
      return None, content_type, None
    if tool == 'memory_viewer' and request.args.get(
        'view_memory_allocation_timeline'
    ):
      params['view_memory_allocation_timeline'] = True

    params['memory_space'] = request.args.get('memory_space', '0')

    if tool == 'trace_viewer@':
      options = {}
      options['resolution'] = request.args.get('resolution', 8000)
      options['full_dma'] = full_dma
      if request.args.get('start_time_ms') is not None:
        options['start_time_ms'] = request.args.get('start_time_ms')
      if request.args.get('end_time_ms') is not None:
        options['end_time_ms'] = request.args.get('end_time_ms')
      if request.args.get('event_name') is not None:
        options['event_name'] = request.args.get('event_name')
      if request.args.get('duration_ms') is not None:
        options['duration_ms'] = request.args.get('duration_ms')
      if request.args.get('unique_id') is not None:
        options['unique_id'] = request.args.get('unique_id')
      if request.args.get('search_prefix') is not None:
        options['search_prefix'] = request.args.get('search_prefix')
      params['trace_viewer_options'] = options

    _, content_encoding = None, None
    if use_xplane(tool):
      selected_hosts, asset_paths = self._get_valid_hosts(
          run_dir, run, tool, hosts_param, host
      )
      if not asset_paths:
        return None, content_type, None

      params['hosts'] = selected_hosts
      try:
        data, content_type = convert.xspace_to_tool_data(
            asset_paths, tool, params)
      except AttributeError as e:
        logger.warning('Error generating analysis results due to %r', e)
        raise AttributeError(
            'Error generating analysis results due to %r' % e
        ) from e
      except ValueError as e:
        logger.warning('XPlane convert to tool data failed as %r', e)
        raise e
      except FileNotFoundError as e:
        logger.warning('XPlane convert to tool data failed as %r', e)
        raise e

      # Write cache version file if use_saved_result is False.
      if not use_saved_result:
        try:
          with epath.Path(os.path.join(run_dir, CACHE_VERSION_FILE)).open(
              'w'
          ) as f:
            f.write(version.__version__)
        except OSError as e:
          logger.warning('Cannot write cache version file: %r', e)

      return data, content_type, content_encoding

    logger.info('%s does not use xplane', tool)
    return None, content_type, None

  def hlo_module_list_impl(
      self, request: wrappers.Request
  ) -> str:
    """Returns a string of HLO module names concatenated by comma for the given run."""
    run = request.args.get('run')
    run_dir = self._run_dir(run, request)
    module_list = []
    if not run_dir:
      logger.warning('Cannot find asset directory for: %s', run)
      return ''
    tool_pattern = '*.hlo_proto.pb'
    filenames = []
    try:
      path = epath.Path(run_dir)
      filenames = path.glob(tool_pattern)
    except OSError as e:
      logger.warning('Cannot read asset directory: %s, OpError %r', run_dir, e)
    filenames = [os.fspath(os.path.basename(f)) for f in filenames]
    for filename in filenames:
      module_name, _ = _parse_filename(filename)
      if module_name:
        module_list.append(module_name)
    module_names_str = ','.join(module_list)
    return module_names_str

  # pytype: disable=wrong-arg-types
  @wrappers.Request.application
  def data_route(self, request: wrappers.Request) -> wrappers.Response:
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
      if not self._tf_profiler:
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
      _pywrap_profiler_plugin.trace(
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

  def _get_graph_viewer_options(
      self, request: wrappers.Request
  ) -> dict[str, Any]:
    node_name = request.args.get('node_name')
    module_name = request.args.get('module_name')
    graph_width_str = request.args.get('graph_width') or ''
    graph_width = int(graph_width_str) if graph_width_str.isdigit() else 3
    show_metadata = int(request.args.get('show_metadata') == 'true')
    merge_fusion = int(request.args.get('merge_fusion') == 'true')
    return {
        'node_name': node_name,
        'module_name': module_name,
        'graph_width': graph_width,
        'show_metadata': show_metadata,
        'merge_fusion': merge_fusion,
        'format': request.args.get('format'),
        'type': request.args.get('type')
    }

  def generate_runs(self) -> Iterator[str]:
    """Generator for a list of runs.

    The "run name" here is a "frontend run name" - see _tb_run_directory() for
    the definition of a "frontend run name" and how it maps to a directory of
    profile data for a specific profile "run". The profile plugin concept of
    "run" is different from the normal TensorBoard run; each run in this case
    represents a single instance of profile data collection, more similar to a
    "step" of data in typical TensorBoard semantics. These runs reside in
    subdirectories of the plugins/profile directory within any regular
    TensorBoard run directory or within the session_dir root directory
    itself (even if it contains no tfevents file and would thus not be
    considered a normal TensorBoard run, for backwards compatibility).

    `generate_runs` will get all runs first, and get tools list from
    `generate_tools_of_run` for a single run due to expensive processing for
    xspace data to parse the tools.
    Example:
      logs/
        plugins/
          profile/
            run1/
              hostA.trace
        train/
          events.out.tfevents.foo
          plugins/
            profile/
              run1/
                hostA.trace
                hostB.trace
              run2/
                hostA.trace
        validation/
          events.out.tfevents.foo
          plugins/
            profile/
              run1/
                hostA.trace
        new_job/
          tensorboard/
            plugins/
              profile/
                run1/
                  hostA.xplane.pb
    Yields:
    A sequence of string that are "frontend run names".
    For the above example, this would be:
        "run1", "train/run1", "train/run2", "validation/run1",
        "new_job/tensorboard/run1"
    """
    if not self.logdir:
      return

    # Ensure that we check the root logdir and all subdirectories.
    # Note that we check if logdir is a directory to handle case where
    # it's actually a multipart directory spec, which this plugin does not
    # support.
    #
    # This change still enforce the requirement that the subdirectories must
    # end with plugins/profile directory, as enforced by TensorBoard.
    logdir_path = epath.Path(self.logdir)
    schemeless_logdir = str(logdir_path)
    if '://' in schemeless_logdir:
      schemeless_logdir = schemeless_logdir.split('://', 1)[1]
    tb_runs = {'.'}

    if logdir_path.is_dir():
      try:
        fs = etils.epath.backend.fsspec_backend.fs(self.logdir)
        for path_str in fs.glob(os.path.join(self.logdir, '**', PLUGIN_NAME)):
          path = epath.Path(path_str)
          if fs.isdir(path) and path.parent.name == TB_NAME:
            tb_run_dir = path.parent.parent
            tb_run = tb_run_dir.relative_to(schemeless_logdir)
            tb_runs.add(str(tb_run))
      except ValueError:
        # gcsfs not available, fall back to legacy path walk.
        for cur_dir, _, _ in logdir_path.walk():
          if (cur_dir.name == PLUGIN_NAME and cur_dir.parent.name == TB_NAME):
            tb_run_dir = cur_dir.parent.parent
            tb_run = tb_run_dir.relative_to(logdir_path)
            tb_runs.add(str(tb_run))
    tb_run_names_to_dirs = {
        run: _tb_run_directory(self.logdir, run) for run in tb_runs
    }
    plugin_assets = _plugin_assets(
        self.logdir, list(tb_run_names_to_dirs), PLUGIN_NAME
    )
    visited_runs = set()
    for tb_run_name, profile_runs in six.iteritems(plugin_assets):
      tb_run_dir = tb_run_names_to_dirs[tb_run_name]
      tb_plugin_dir = plugin_asset_util.PluginDirectory(tb_run_dir, PLUGIN_NAME)

      for profile_run in profile_runs:
        # Remove trailing separator; some filesystem implementations emit this.
        profile_run = profile_run.rstrip(os.sep)
        if tb_run_name == '.':
          frontend_run = profile_run
        else:
          frontend_run = str(epath.Path(tb_run_name) / profile_run)
        profile_run_dir = str(epath.Path(tb_plugin_dir) / profile_run)
        if epath.Path(profile_run_dir).is_dir():
          self._run_to_profile_run_dir[frontend_run] = profile_run_dir
          if frontend_run not in visited_runs:
            visited_runs.add(frontend_run)
            yield frontend_run

  def generate_tools_of_run(self, run: str, run_dir: str) -> Iterator[str]:
    """Generate a list of tools given a certain run."""
    if not run_dir:
      logger.warning('Cannot find asset directory for: %s', run)
      return
    profile_run_dir = epath.Path(run_dir)
    cache = ToolsCache(profile_run_dir)

    cached_tools = cache.load()

    if cached_tools is not None:
      for tool in cached_tools:
        yield tool
      return

    # Cache is invalid or doesn't exist, regenerate
    tools = []
    try:
      all_filenames = [f.name for f in profile_run_dir.iterdir()]
    except OSError as e:
      logger.warning(
          'Cannot read asset directory: %s, Error %r', profile_run_dir, e
      )
      return tools

    if all_filenames:
      tools = self._get_active_tools(all_filenames, str(profile_run_dir))
      cache.save(tools)

    for tool in tools:
      yield tool

  def _get_active_tools(self, filenames, profile_run_dir=''):
    """Get a list of tools available given the filenames created by profiler.

    Args:
      filenames: List of strings that represent filenames
      profile_run_dir: The run directory of the profile.

    Returns:
      A list of strings representing the available tools
    """
    tool_sort_order = [
        'overview_page',
        'trace_viewer',
        'trace_viewer@',
        'graph_viewer',
        'op_profile',
        'hlo_op_profile',
        'input_pipeline_analyzer',
        'input_pipeline',
        'kernel_stats',
        'memory_profile',
        'memory_viewer',
        'roofline_model',
        'perf_counters',
        'pod_viewer',
        'framework_op_stats',
        'tensorflow_stats',  # Legacy name for framework_op_stats
        'hlo_op_stats',
        'hlo_stats',  # Legacy name for hlo_op_stats
        'inference_profile',
        'megascale_stats',
    ]
    tools = _get_tools(filenames, profile_run_dir)
    if 'trace_viewer@' in tools:
      # streaming trace viewer always override normal trace viewer.
      # the trailing '@' is to inform tf-profile-dashboard.html and
      # tf-trace-viewer.html that stream trace viewer should be used.
      tools.discard('trace_viewer')

    sorted_tools = [t for t in tool_sort_order if t in tools]
    remaining_tools = tools.difference(sorted_tools)
    sorted_tools.extend(sorted(remaining_tools))

    return sorted_tools
