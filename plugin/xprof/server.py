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

  def _not_found_handler(environ, start_response):
    del environ  # Unused.
    start_response("404 Not Found", [("Content-Type", "text/plain")])
    return [b"Not Found"]

  def application(environ, start_response):
    path = environ["PATH_INFO"]
    if path.startswith(prefix):
      path = path[len(prefix) :]
    if path != "/" and path.endswith("/"):
      path = path[:-1]
    if path in apps:
      handler = apps[path]
    elif path in ("", "/"):
      handler = plugin.default_handler
    else:
      handler = _not_found_handler
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


# CLI subcommands introduced in xprof-nightly >= 2.24.0. Stable xprof wheels
# (<= 2.23.1) shipped zero CLI subcommands and routed the `xprof` console script
# to the TensorBoard-style server, whose argument parser rejects these tokens
# with an opaque "unrecognized arguments" error. `main` intercepts them and
# prints an actionable upgrade message instead. This mirrors the public (OSS)
# tools registered in `xprof.cli.xprof_cli.cli_main`.
_CLI_SUBCOMMANDS = frozenset({
    "aggregate_xplane_events",
    "check_host_boundness",
    "compute_utilization",
    "detect_layout_mismatch_copies",
    "detect_unfused_reshapes",
    "detect_unfused_updates",
    "detect_unnecessary_convert_dynamic_scale",
    "detect_unnecessary_convert_reduce",
    "get_avg_step_time",
    "get_device_information",
    "get_graph_viewer",
    "get_hlo_module_content",
    "get_hlo_neighborhood",
    "get_hlo_op_profile",
    "get_hlo_stats",
    "get_hlo_text",
    "get_hosts",
    "get_kernel_stats",
    "get_kernel_utilization",
    "get_kpi_metrics",
    "get_llo_analysis",
    "get_llo_debug_string",
    "get_memory_profile",
    "get_overview",
    "get_peak_allocations",
    "get_profile_summary",
    "get_roofline_model",
    "get_top_hlo_ops",
    "get_utilization_viewer",
    "get_xspace_proto",
    "list_hlo_modules",
    "list_xplane_events",
    "upload_trace",
    "verify_numerical_parity",
})


def cli_subcommand_required_message(subcommand: str) -> str:
  """Returns the upgrade message shown when a CLI subcommand hits the server.

  Args:
    subcommand: The CLI subcommand the user attempted to run (e.g.
      `get_overview`).

  Returns:
    An actionable, single-line error message directing the user to install
    xprof-nightly (or upgrade to xprof >= 2.24.0).
  """
  return (
      f"Error: 'xprof {subcommand}' CLI tools require xprof-nightly >= 2.24.0."
      " Please run 'pip install -U xprof-nightly' or upgrade to xprof >="
      " 2.24.0. For TensorBoard server usage, run 'xprof --help'."
  )


def main(argv: Optional[list[str]] = None) -> int:
  """Console entry point for the standalone XProf server.

  Intercepts CLI subcommands (introduced in xprof-nightly >= 2.24.0) that would
  otherwise fall through to the server argument parser and fail with an opaque
  "unrecognized arguments" error, printing an actionable upgrade message
  instead. Any other arguments are parsed as standard server flags and used to
  launch the web server.

  Args:
    argv: Command-line arguments excluding the program name. Defaults to
      `sys.argv[1:]`.

  Returns:
    A process exit code: 0 on a normal server launch, 2 when a CLI subcommand is
    intercepted.
  """
  argv = sys.argv[1:] if argv is None else list(argv)

  if argv and argv[0] in _CLI_SUBCOMMANDS:
    sys.stderr.write(cli_subcommand_required_message(argv[0]) + "\n")
    return 2

  parser = argparse.ArgumentParser(
      prog="xprof", description="Start the standalone XProf web server."
  )
  parser.add_argument(
      "--logdir", default=None, help="Directory containing profile data."
  )
  parser.add_argument(
      "--port", type=int, default=8791, help="Port for the web server."
  )
  parser.add_argument(
      "--grpc_port",
      type=int,
      default=_DEFAULT_GRPC_PORT,
      help="Port for the gRPC worker service.",
  )
  parser.add_argument(
      "--worker_service_address",
      default=None,
      help="Address of the gRPC worker service.",
  )
  parser.add_argument(
      "--src_prefix", default=None, help="Source path prefix for the server."
  )
  parser.add_argument(
      "--max_concurrent_worker_requests",
      type=int,
      default=1,
      help="Maximum number of concurrent worker requests.",
  )
  parser.add_argument(
      "--hide_capture_profile_button",
      action="store_true",
      help="Hide the capture-profile button in the UI.",
  )
  parser.add_argument(
      "--enable_tab_name_label",
      action="store_true",
      help="Enable the tab-name label in the UI.",
  )
  args = parser.parse_args(argv)

  start_server(
      logdir=args.logdir,
      port=args.port,
      hide_capture_profile_button=args.hide_capture_profile_button,
      enable_tab_name_label=args.enable_tab_name_label,
      worker_service_address=args.worker_service_address,
      grpc_port=args.grpc_port,
      src_prefix=args.src_prefix,
      max_concurrent_worker_requests=args.max_concurrent_worker_requests,
  )
  return 0


if __name__ == "__main__":
  sys.exit(main())
