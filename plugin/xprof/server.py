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

import argparse
import collections
import dataclasses
import socket
import sys
from typing import Optional

from cheroot import wsgi
from etils import epath

from xprof import profile_plugin_loader
from xprof.standalone import base_plugin
from xprof.standalone import plugin_event_multiplexer
from xprof.convert import _pywrap_profiler_plugin

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
    print(f"XProf at http://localhost:{port}/ (Press CTRL+C to quit)")
    server.start()
  except KeyboardInterrupt:
    server.stop()


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
  fallback_address = "::" if socket.has_ipv6 else "0.0.0.0"
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
    if hasattr(socket, "AF_INET6") and addrs_by_family[socket.AF_INET6]:
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
  _pywrap_profiler_plugin.start_grpc_server(config.grpc_port)

  context = TBContext(
      config.logdir, DataProvider(config.logdir), TBContext.Flags(False)
  )
  context.hide_capture_profile_button = config.hide_capture_profile_button
  context.src_prefix = config.src_prefix
  loader = ProfilePluginLoader()
  plugin = loader.load(context)
  run_server(plugin, _get_wildcard_address(config.port), config.port)


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


def _create_argument_parser() -> argparse.ArgumentParser:
  """Creates and configures the argument parser for the XProf server CLI.

  This function sets up argparse to handle command-line flags for specifying
  the log directory, server port, and other operational modes.

  Returns:
    The configured argument parser.
  """
  parser = argparse.ArgumentParser(
      prog="xprof",
      description="Launch the XProf profiling server.",
      formatter_class=argparse.RawDescriptionHelpFormatter,
      epilog=(
          "Examples:\n"
          "\txprof ~/jax/profile-logs -p 8080\n"
          "\txprof --logdir ~/jax/profile-logs -p 8080"
      ),
  )

  logdir_group = parser.add_mutually_exclusive_group(required=False)

  logdir_group.add_argument(
      "-l",
      "--logdir",
      dest="logdir_opt",
      metavar="<logdir>",
      type=str,
      help="The directory where profile files will be stored.",
  )

  logdir_group.add_argument(
      "logdir_pos",
      nargs="?",
      metavar="logdir",
      type=str,
      default=None,
      help="Positional argument for the profile log directory.",
  )

  parser.add_argument(
      "-p",
      "--port",
      metavar="<port>",
      type=int,
      default=8791,
      help="The port number for the server (default: %(default)s).",
  )

  parser.add_argument(
      "-hcpb",
      "--hide_capture_profile_button",
      action="store_true",
      default=False,
      help="Hides the 'Capture Profile' button in the UI.",
  )

  parser.add_argument(
      "-wsa",
      "--worker_service_address",
      type=str,
      default=None,
      help=(
          "A comma-separated list of worker service addresses (IPs or FQDNs)"
          " with their gRPC ports, used in distributed profiling. Example:"
          " 'worker-a.project.internal:50051,worker-b.project.internal:50051'."
          " If not provided, it will use 0.0.0.0 with the gRPC port."
      ),
  )

  parser.add_argument(
      "-gp",
      "--grpc_port",
      type=int,
      default=_DEFAULT_GRPC_PORT,
      help=(
          "The port for the gRPC server, which runs alongside the main HTTP"
          " server for distributed profiling. This must be different from the"
          " main server port (--port)."
      ),
  )

  parser.add_argument(
      "-spp",
      "--src_prefix",
      type=str,
      default=None,
      help="The path prefix for the source code being profiled.",
  )
  return parser


def main() -> int:
  """Parses command-line arguments and launches the XProf server.

  This is the main entry point for the XProf server application. It parses
  command-line arguments, creates a ServerConfig, and then launches the
  server.

  Returns:
    An exit code, 0 for success and non-zero for errors.
  """
  parser = _create_argument_parser()
  try:
    args = parser.parse_args()
  except SystemExit as e:
    return e.code

  logdir = (
      get_abs_path(args.logdir_opt or args.logdir_pos)
      if args.logdir_opt or args.logdir_pos
      else None
  )

  worker_service_address = args.worker_service_address
  if worker_service_address is None:
    worker_service_address = f"0.0.0.0:{args.grpc_port}"

  config = ServerConfig(
      logdir=logdir,
      port=args.port,
      grpc_port=args.grpc_port,
      worker_service_address=worker_service_address,
      hide_capture_profile_button=args.hide_capture_profile_button,
      src_prefix=args.src_prefix,
  )

  print("Attempting to start XProf server:")
  print(f"  Log Directory: {logdir}")
  print(f"  Port: {config.port}")
  print(f"  Worker Service Address: {config.worker_service_address}")
  print(f"  Hide Capture Button: {config.hide_capture_profile_button}")

  if logdir and not epath.Path(logdir).exists():
    print(
        f"Error: Log directory '{logdir}' does not exist or is not a"
        " directory.",
        file=sys.stderr,
    )
    return 1

  if config.port == config.grpc_port:
    print(
        "Error: The main server port (--port) and the gRPC port (--grpc_port)"
        " must be different.",
        file=sys.stderr,
    )
    return 1

  _launch_server(
      config,
  )
  return 0
