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
"""Utilities to start up a standalone webserver."""

import collections
import dataclasses
import logging
import socket
import sys
from typing import Optional

from cheroot import wsgi
from etils import epath

from xprof import profile_plugin_loader
from xprof.standalone import base_plugin
from xprof.standalone import plugin_event_multiplexer
from xprof.convert import _pywrap_profiler_plugin

logger = logging.getLogger("tensorboard.plugins.profile")
logger.setLevel(logging.INFO)
if not logger.handlers:
  log_handler = logging.StreamHandler(sys.stderr)
  formatter = logging.Formatter(
      "%(levelname)s %(asctime)s [%(filename)s:%(lineno)d] %(message)s"
  )
  log_handler.setFormatter(formatter)
  logger.addHandler(log_handler)
  logger.propagate = False

DataProvider = plugin_event_multiplexer.DataProvider
TBContext = base_plugin.TBContext
ProfilePluginLoader = profile_plugin_loader.ProfilePluginLoader


_DEFAULT_GRPC_PORT = 50051


@dataclasses.dataclass(frozen=True)
class ServerConfig:
  """Configuration parameters for launching the XProf server.

  This dataclass holds all the settings required to initialize and run the XProf
  profiling server, including network ports, log locations, and feature flags.
  """

  logdir: Optional[str]
  port: int
  grpc_port: int
  worker_service_address: str
  hide_capture_profile_button: bool
  src_prefix: Optional[str]
  max_concurrent_worker_requests: int
  enable_tab_name_label: bool = False


def make_wsgi_app(plugin):
  """Create a WSGI application for the standalone server."""

  apps = plugin.get_plugin_apps()

  prefix = "/data/plugin/profile"

  def application(environ, start_response):
    path = environ["PATH_INFO"]
    if path.startswith(prefix):
      path = path[len(prefix) :]
    if path != "/" and path.endswith("/"):
      path = path[:-1]
    handler = apps.get(path, plugin.default_handler)
    return handler(environ, start_response)

  return application


def run_server(plugin, host, port):
  """Starts a webserver for the standalone server."""

  app = make_wsgi_app(plugin)

  server = wsgi.Server((host, port), app)

  try:
    logger.info("XProf at http://localhost:%d/ (Press CTRL+C to quit)", port)
    server.start()
  except KeyboardInterrupt:
    server.stop()


def _has_ipv6():
    """Returns whether IPv6 is supported or unsupported"""
    try:
        socket.socket(socket.AF_INET6).close()
        return True
    except IOError as e:
        return False


def _get_wildcard_address(port) -> str:
  """Returns a wildcard address for the port in question.

  This will attempt to follow the best practice of calling
  getaddrinfo() with a null host and AI_PASSIVE to request a
  server-side socket wildcard address. If that succeeds, this
  returns the first IPv6 address found, or if none, then returns
  the first IPv4 address. If that fails, then this returns the
  hardcoded address "::" if socket.has_ipv6 is True, else
  "0.0.0.0".

  Args:
    port: The port number.

  Returns:
    The wildcard address.
  """
  has_ipv6 = _has_ipv6()
  fallback_address = "::" if has_ipv6 else "0.0.0.0"
  if hasattr(socket, "AI_PASSIVE"):
    try:
      addrinfos = socket.getaddrinfo(
          None,
          port,
          socket.AF_UNSPEC,
          socket.SOCK_STREAM,
          socket.IPPROTO_TCP,
          socket.AI_PASSIVE,
      )
    except socket.gaierror:
      return fallback_address
    addrs_by_family = collections.defaultdict(list)
    for family, _, _, _, sockaddr in addrinfos:
      # Format of the "sockaddr" socket address varies by address family,
      # but [0] is always the IP address portion.
      addrs_by_family[family].append(sockaddr[0])
    if has_ipv6 and hasattr(socket, "AF_INET6") and addrs_by_family[socket.AF_INET6]:
      return addrs_by_family[socket.AF_INET6][0]
    if hasattr(socket, "AF_INET") and addrs_by_family[socket.AF_INET]:
      return addrs_by_family[socket.AF_INET][0]
  return fallback_address


def _launch_server(
    config: ServerConfig,
):
  """Initializes and launches the main XProf server.

  This function sets up the necessary components for the XProf server based on
  the provided configuration. It starts the gRPC worker service if distributed
  processing is enabled, creates the TensorBoard context, loads the profile
  plugin, and finally starts the web server to handle HTTP requests.

  Args:
    config: The ServerConfig object containing all server settings.
  """
  _pywrap_profiler_plugin.initialize_stubs(config.worker_service_address)
  _pywrap_profiler_plugin.start_grpc_server(
      config.grpc_port, config.max_concurrent_worker_requests
  )

  context = TBContext(
      config.logdir, DataProvider(config.logdir), TBContext.Flags(False)
  )
  context.hide_capture_profile_button = config.hide_capture_profile_button
  context.enable_tab_name_label = config.enable_tab_name_label
  context.src_prefix = config.src_prefix
  loader = ProfilePluginLoader()
  plugin = loader.load(context)
  run_server(plugin, _get_wildcard_address(config.port), config.port)


def start_server(
    logdir: str | None = None,
    port: int = 8791,
    hide_capture_profile_button: bool = False,
    enable_tab_name_label: bool = False,
    worker_service_address: str | None = None,
    grpc_port: int = 50051,
    src_prefix: str | None = None,
    max_concurrent_worker_requests: int = 1,
    default_logdir: str | None = None,
):
  """Starts the XProf web server."""
  target_logdir = logdir if logdir is not None else default_logdir
  resolved_logdir = get_abs_path(target_logdir) if target_logdir else None

  if worker_service_address is None:
    worker_service_address = f"0.0.0.0:{grpc_port}"

  config = ServerConfig(
      logdir=resolved_logdir,
      port=port,
      grpc_port=grpc_port,
      worker_service_address=worker_service_address,
      hide_capture_profile_button=hide_capture_profile_button,
      enable_tab_name_label=enable_tab_name_label,
      src_prefix=src_prefix,
      max_concurrent_worker_requests=max_concurrent_worker_requests,
  )

  if resolved_logdir and not epath.Path(resolved_logdir).exists():
    raise ValueError(
        f"Log directory '{resolved_logdir}' does not exist or is not a"
        " directory."
    )

  if config.port == config.grpc_port:
    raise ValueError(
        "The main server port (--port) and the gRPC port (--grpc_port)"
        " must be different."
    )

  _launch_server(config)


def get_abs_path(logdir: str) -> str:
  """Gets the absolute path for a given log directory string.

  This function correctly handles both Google Cloud Storage (GCS) paths and
  local filesystem paths.

  - GCS paths (e.g., "gs://bucket/log") are returned as is.
  - Local filesystem paths (e.g., "~/logs", "log", ".") are made absolute.

  Args:
      logdir: The path string.

  Returns:
      The corresponding absolute path as a string.
  """
  if logdir.startswith("gs://"):
    return logdir

  return str(epath.Path(logdir).expanduser().resolve())
