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
from collections.abc import Callable, Iterable, Mapping, Sequence
import dataclasses
import ipaddress
import logging
import os
import sys
from typing import Any, Optional
import urllib.parse

from cheroot import wsgi
from etils import epath

from xprof import profile_plugin
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


_DEFAULT_HOST = "127.0.0.1"
_HOST_ENV_VAR = "XPROF_HOST"
_DEFAULT_GRPC_PORT = 50051


def get_default_host() -> str:
  """Returns the default server bind host from XPROF_HOST or loopback."""
  return (os.environ.get(_HOST_ENV_VAR) or "").strip() or _DEFAULT_HOST


def _extract_host_name(http_host: str) -> str:
  """Extracts the canonical lowercase hostname from a host or Host header."""
  if isinstance(http_host, bool) or not isinstance(http_host, str):
    return ""
  normalized = profile_plugin.normalize_authority(http_host)
  return normalized[0] if normalized is not None else ""


def _is_loopback_host(http_host: str) -> bool:
  """Returns True if http_host resolves to localhost or a loopback IP."""
  hostname = _extract_host_name(http_host)
  return bool(hostname) and profile_plugin.is_loopback_host(hostname)


def _normalize_bind_host(host: str) -> str:
  """Validates a server bind host and unwraps bracketed IPv6 literals."""
  if isinstance(host, bool) or not isinstance(host, str):
    raise ValueError("The --host flag requires a non-empty string value.")
  candidate = host.strip()
  if (
      not candidate
      or any(ch in candidate for ch in "/?#@\\,")
      or any(ch.isspace() for ch in candidate)
  ):
    raise ValueError(f"Invalid host address: {host!r}.")
  if "[" in candidate or "]" in candidate:
    if not (candidate.startswith("[") and candidate.endswith("]")):
      raise ValueError(f"Invalid host address: {host!r}.")
    inner = candidate[1:-1]
    try:
      ipv6 = ipaddress.IPv6Address(inner)
    except ValueError as err:
      raise ValueError(f"Invalid IPv6 host address: {host!r}.") from err
    return str(ipv6).lower()
  if ":" in candidate:
    try:
      ipaddress.IPv6Address(candidate)
    except ValueError as err:
      raise ValueError(f"Invalid host address: {host!r}.") from err
    return candidate
  try:
    parsed = urllib.parse.urlsplit(f"http://{candidate}")
    _ = parsed.port
  except ValueError as err:
    raise ValueError(f"Invalid host address: {host!r}.") from err
  if not (parsed.hostname or "").rstrip("."):
    raise ValueError(f"Invalid host address: {host!r}.")
  return candidate


def _plain_response(
    environ: Mapping[str, Any],
    start_response: Callable[..., Any],
    status: str,
    body: bytes,
) -> list[bytes]:
  """Emits a plain-text WSGI error response, omitting body bytes on HEAD."""
  headers = [
      ("Content-Type", "text/plain"),
      ("Content-Length", str(len(body))),
      ("X-Content-Type-Options", "nosniff"),
      ("Vary", profile_plugin.CROSS_ORIGIN_VARY_HEADER),
  ]
  start_response(status, headers)
  if (environ.get("REQUEST_METHOD") or "GET").upper() == "HEAD":
    return []
  return [body]


def _wrap_start_response_with_headers(
    start_response: Callable[..., Any],
    extra_headers: Sequence[tuple[str, str]],
) -> Callable[..., Any]:
  """Wraps WSGI start_response to append headers not already present."""

  def _wrapped_start_response(
      status: str,
      headers: Sequence[tuple[str, str]],
      exc_info: Any | None = None,
  ) -> Any:
    existing = {name.lower() for name, _ in headers}
    merged_headers = list(headers)
    for name, value in extra_headers:
      if name.lower() not in existing:
        merged_headers.append((name, value))
        existing.add(name.lower())
    if exc_info is not None:
      return start_response(status, merged_headers, exc_info)
    return start_response(status, merged_headers)

  return _wrapped_start_response


def _is_top_level_ui_navigation(environ: Mapping[str, Any]) -> bool:
  """Returns True if environ represents a top-level GET/HEAD UI navigation."""
  if (environ.get("REQUEST_METHOD") or "GET").upper() not in ("GET", "HEAD"):
    return False
  mode = (environ.get("HTTP_SEC_FETCH_MODE") or "").strip().lower()
  dest = (environ.get("HTTP_SEC_FETCH_DEST") or "").strip().lower()
  if mode or dest:
    return mode == "navigate" and dest == "document"
  return (
      not (environ.get("HTTP_SEC_FETCH_SITE") or "").strip()
      and not (environ.get("HTTP_ORIGIN") or "").strip()
  )


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
  host: str = _DEFAULT_HOST


def make_wsgi_app(plugin, host: str = _DEFAULT_HOST):
  """Create a WSGI application for the standalone server."""

  apps = plugin.get_plugin_apps()

  prefix = "/data/plugin/profile"
  enforce_loopback_host = _is_loopback_host(host)
  parameterized_spa_prefixes = (
      "/kernel_viewer/",
      "/megascale_perfetto/",
      "/roofline_model/",
      "/stack_trace_page/",
  )

  def _not_found_handler(environ, start_response):
    return _plain_response(
        environ, start_response, "404 Not Found", b"Not Found"
    )

  def application(
      environ: Mapping[str, Any], start_response: Callable[..., Any]
  ) -> Iterable[bytes]:
    if enforce_loopback_host:
      raw_http_host = environ.get("HTTP_HOST") or ""
      if "HTTP_HOST" in environ and not _is_loopback_host(raw_http_host):
        return _plain_response(
            environ, start_response, "403 Forbidden", b"Forbidden"
        )
    path = environ["PATH_INFO"]
    under_prefix = path == prefix or path.startswith(prefix + "/")
    if under_prefix:
      path = path[len(prefix) :]
    if path != "/" and path.endswith("/"):
      path = path[:-1]
    is_ui_shell = False
    is_parameterized_spa = under_prefix and path.startswith(
        parameterized_spa_prefixes
    )
    is_root_stack_trace = path == "/stack_trace_page" or path.startswith(
        "/stack_trace_page/"
    )
    if path in apps:
      handler = apps[path]
      is_ui_shell = path in (
          profile_plugin.BASE_ROUTE,
          profile_plugin.LOCAL_ROUTE,
          profile_plugin.INDEX_HTML_ROUTE,
      )
    elif (
        path in ("", "/")
        or is_parameterized_spa
        or is_root_stack_trace
        or (under_prefix and "." not in path.rsplit("/", 1)[-1])
    ):
      # The root, or a frontend route under the prefix such as /overview_page,
      # which the browser requests when a tool page is reloaded. The frontend
      # reads the route from the URL.
      handler = plugin.default_handler
      is_ui_shell = True
    else:
      handler = _not_found_handler
    if not (
        is_ui_shell and _is_top_level_ui_navigation(environ)
    ) and profile_plugin.is_cross_origin_request(environ):
      return _plain_response(
          environ, start_response, "403 Forbidden", b"Forbidden"
      )
    extra_headers = [("Vary", profile_plugin.CROSS_ORIGIN_VARY_HEADER)]
    if is_ui_shell:
      extra_headers.insert(0, ("X-Frame-Options", "SAMEORIGIN"))
    response_starter = _wrap_start_response_with_headers(
        start_response, extra_headers
    )
    return handler(environ, response_starter)

  return application


def run_server(plugin, host, port):
  """Starts a webserver for the standalone server."""

  app = make_wsgi_app(plugin, host=host)

  server = wsgi.Server((host, port), app)

  try:
    logger.info("XProf at http://localhost:%d/ (Press CTRL+C to quit)", port)
    server.start()
  except KeyboardInterrupt:
    server.stop()


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
  _pywrap_profiler_plugin.start_grpc_server(
      config.grpc_port, config.max_concurrent_worker_requests
  )
  _pywrap_profiler_plugin.initialize_stubs(config.worker_service_address)

  context = TBContext(
      config.logdir, DataProvider(config.logdir), TBContext.Flags(False)
  )
  context.hide_capture_profile_button = config.hide_capture_profile_button
  context.enable_tab_name_label = config.enable_tab_name_label
  context.src_prefix = config.src_prefix
  loader = ProfilePluginLoader()
  plugin = loader.load(context)
  run_server(plugin, config.host, config.port)


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
    host: str = _DEFAULT_HOST,
):
  """Starts the XProf web server."""
  normalized_host = _normalize_bind_host(host)
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
      host=normalized_host,
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


# CLI subcommands introduced in xprof 2.24.0. Older xprof wheels
# (<= 2.23.1) shipped zero CLI subcommands and routed the `xprof` console script
# to the TensorBoard-style server, whose argument parser rejects these tokens
# with an opaque "unrecognized arguments" error. `main` intercepts them and
# prints an actionable upgrade message instead. This mirrors the public (OSS)
# tools registered in `xprof.cli.xprof_cli.cli_main`.
_CLI_SUBCOMMANDS = frozenset({
    "aggregate_xplane_events",
    "analyze_async_overlap",
    "check_host_boundness",
    "check_kernel_profiling",
    "compute_utilization",
    "detect_layout_mismatch_copies",
    "detect_unfused_reshapes",
    "detect_unfused_updates",
    "detect_unnecessary_convert_dynamic_scale",
    "detect_unnecessary_convert_reduce",
    "diff_sessions",
    "events_db_info",
    "events_db_top_ops",
    "export_json",
    "get_device_information",
    "get_device_time_attribution",
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
    "get_llo_dump_analysis",
    "get_llo_static_analysis",
    "get_memory_profile",
    "get_overview",
    "get_peak_allocations",
    "get_profile_summary",
    "get_roofline_model",
    "get_top_hlo_ops",
    "get_utilization_viewer",
    "get_xspace_proto",
    "inspect_capture",
    "install_skills",
    "list_hlo_modules",
    "list_xplane_events",
    "list_xplane_lines",
    "query_hlo_graph",
    "upload_trace",
})


def cli_subcommand_required_message(subcommand: str) -> str:
  """Returns the upgrade message shown when a CLI subcommand hits the server.

  Args:
    subcommand: The CLI subcommand the user attempted to run (e.g.
      `get_overview`).

  Returns:
    An actionable, single-line error message directing the user to upgrade to
    xprof >= 2.24.0.
  """
  return (
      f"Error: 'xprof {subcommand}' CLI tools require xprof >= 2.24.0."
      " Please run 'pip install -U xprof'. For TensorBoard server usage, run"
      " 'xprof --help'."
  )


def main(argv: Optional[list[str]] = None) -> int:
  """Console entry point for the standalone XProf server.

  Intercepts CLI subcommands (introduced in xprof 2.24.0) that would
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
      "--host",
      default=get_default_host(),
      help="Host address to bind the web server to.",
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
      host=args.host,
  )
  return 0


if __name__ == "__main__":
  sys.exit(main())
