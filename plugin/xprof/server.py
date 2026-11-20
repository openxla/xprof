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
import re
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
_ALLOWED_HOSTS_ENV_VAR = "XPROF_ALLOWED_HOSTS"
_DEFAULT_GRPC_PORT = 50051
_INVALID_REQUEST_HOST_RE = re.compile(
    r"(?:\.\.|%|:(?:0+)$|^[^\[\]]*:.*:|^\d+(?:\.\d+){3}\.(?::|$))"
)
_VALID_DNS_HOSTNAME_RE = re.compile(r"^[a-z0-9_-]+(?:\.[a-z0-9_-]+)*$")


def get_default_host() -> str:
  """Returns the default server bind host from XPROF_HOST or loopback."""
  return (os.environ.get(_HOST_ENV_VAR) or "").strip() or _DEFAULT_HOST


def get_default_allowed_hosts() -> str | None:
  """Returns default allowed Host entries from XPROF_ALLOWED_HOSTS, if set."""
  return (os.environ.get(_ALLOWED_HOSTS_ENV_VAR) or "").strip() or None


def _extract_host_name(http_host: str) -> str:
  """Extracts the canonical lowercase hostname from a host or Host header."""
  if isinstance(http_host, bool) or not isinstance(http_host, str):
    return ""
  normalized = profile_plugin.normalize_authority(http_host)
  return normalized[0] if normalized is not None else ""


def _extract_request_host_name(raw_http_host: Any) -> str:
  """Extracts a canonical HTTP Host header hostname, rejecting malformed forms."""
  if not isinstance(raw_http_host, str):
    return ""
  candidate = raw_http_host.strip()
  if _INVALID_REQUEST_HOST_RE.search(candidate):
    return ""
  hostname = _extract_host_name(candidate)
  if not hostname:
    return ""
  ip = _parse_ip(hostname)
  if ip is not None:
    return "" if _is_unspecified_ip(ip) else hostname
  if not _VALID_DNS_HOSTNAME_RE.fullmatch(hostname):
    return ""
  return hostname


def _is_loopback_host(http_host: str) -> bool:
  """Returns True if http_host resolves to localhost or a loopback IP."""
  hostname = _extract_host_name(http_host)
  return bool(hostname) and profile_plugin.is_loopback_host(hostname)


def _parse_ip(
    hostname: str,
) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
  """Parses hostname as an IPv4/IPv6 literal, or returns None."""
  try:
    return ipaddress.ip_address(hostname)
  except ValueError:
    return None


def _is_unspecified_ip(
    ip: ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> bool:
  """Returns True if ip is an unspecified/wildcard IP (0.0.0.0 or ::)."""
  mapped = getattr(ip, "ipv4_mapped", None)
  return ip.is_unspecified or bool(mapped and mapped.is_unspecified)


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


def _normalize_allowed_host_entry(entry: str) -> str:
  """Validates and normalizes a single Host allowlist entry."""
  if isinstance(entry, bool) or not isinstance(entry, str):
    raise ValueError(
        "The --allowed_hosts flag requires non-empty host strings."
    )
  candidate = entry.strip()
  if not candidate:
    raise ValueError(f"Invalid allowed host entry: {entry!r}.")
  if candidate == "*":
    return "*"
  is_wildcard = candidate.startswith("*.")
  raw_host = candidate[2:] if is_wildcard else candidate
  if (
      not raw_host
      or "*" in raw_host
      or any(ch.isspace() for ch in raw_host)
  ):
    raise ValueError(f"Invalid allowed host entry: {entry!r}.")
  if is_wildcard and ("[" in raw_host or "]" in raw_host or ":" in raw_host):
    raise ValueError(f"Invalid allowed host entry: {entry!r}.")
  try:
    normalized_bind = _normalize_bind_host(raw_host)
  except ValueError as err:
    raise ValueError(f"Invalid allowed host entry: {entry!r}.") from err
  canonical = _extract_host_name(normalized_bind)
  if (
      not canonical
      or normalized_bind.startswith(".")
      or ".." in normalized_bind
  ):
    raise ValueError(f"Invalid allowed host entry: {entry!r}.")
  ip = _parse_ip(canonical)
  if is_wildcard:
    if ip is not None:
      raise ValueError(f"Wildcard allowed host cannot be an IP: {entry!r}.")
    if not _VALID_DNS_HOSTNAME_RE.fullmatch(canonical):
      raise ValueError(f"Invalid allowed host entry: {entry!r}.")
    return f"*.{canonical}"
  if ip is not None:
    if _is_unspecified_ip(ip) or (
        ip.version == 4 and raw_host.endswith(".")
    ):
      raise ValueError(f"Invalid allowed host entry: {entry!r}.")
    return canonical
  if not _VALID_DNS_HOSTNAME_RE.fullmatch(canonical):
    raise ValueError(f"Invalid allowed host entry: {entry!r}.")
  return canonical


def _normalize_allowed_hosts(
    allowed_hosts: str | Sequence[str] | None,
) -> tuple[str, ...]:
  """Validates and normalizes --allowed_hosts / XPROF_ALLOWED_HOSTS values."""
  if allowed_hosts is None:
    return ()
  if isinstance(allowed_hosts, bool):
    raise ValueError(
        "The --allowed_hosts flag requires a comma-separated host string or"
        " sequence of host strings."
    )
  if isinstance(allowed_hosts, str):
    if not allowed_hosts.strip():
      raise ValueError(
          "The --allowed_hosts flag requires a non-empty host string."
      )
    raw_entries = allowed_hosts.split(",")
  elif isinstance(allowed_hosts, Sequence) and not isinstance(
      allowed_hosts, (bytes, bytearray)
  ):
    raw_entries = []
    for item in allowed_hosts:
      if isinstance(item, bool) or not isinstance(item, str):
        raise ValueError(
            "The --allowed_hosts flag requires non-empty host strings."
        )
      raw_entries.extend(item.split(","))
  else:
    raise ValueError(
        "The --allowed_hosts flag requires a comma-separated host string or"
        " sequence of host strings."
    )
  normalized: list[str] = []
  seen: set[str] = set()
  for entry in raw_entries:
    canonical = _normalize_allowed_host_entry(entry)
    if canonical not in seen:
      seen.add(canonical)
      normalized.append(canonical)
  return tuple(normalized)


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
  allowed_hosts: tuple[str, ...] = ()


def make_wsgi_app(
    plugin,
    host: str = _DEFAULT_HOST,
    allowed_hosts: str | Sequence[str] | None = None,
):
  """Create a WSGI application for the standalone server."""

  apps = plugin.get_plugin_apps()

  prefix = "/data/plugin/profile"
  is_loopback_bind = _is_loopback_host(host)
  bound_hostname = _extract_host_name(host)
  normalized_allowed_hosts = _normalize_allowed_hosts(allowed_hosts)
  allow_any_host = "*" in normalized_allowed_hosts
  exact_hosts: set[str] = set()
  wildcard_suffix_list: list[str] = []
  for entry in normalized_allowed_hosts:
    if entry.startswith("*."):
      wildcard_suffix_list.append(entry[1:])
    elif entry != "*":
      exact_hosts.add(entry)
  if (
      bound_hostname
      and _parse_ip(bound_hostname) is None
      and _VALID_DNS_HOSTNAME_RE.fullmatch(bound_hostname)
  ):
    exact_hosts.add(bound_hostname)
  exact_allowed_hosts = frozenset(exact_hosts)
  wildcard_suffixes = tuple(wildcard_suffix_list)
  parameterized_spa_prefixes = (
      "/kernel_viewer/",
      "/megascale_perfetto/",
      "/roofline_model/",
      "/stack_trace_page/",
  )

  def _is_allowed_host(raw_http_host: Any) -> bool:
    """Returns True if HTTP_HOST is permitted for this server configuration."""
    hostname = _extract_request_host_name(raw_http_host)
    if not hostname:
      return False
    if allow_any_host:
      return True
    if profile_plugin.is_loopback_host(hostname):
      return True
    if not is_loopback_bind and _parse_ip(hostname) is not None:
      # Non-unspecified IP literals cannot be DNS-rebound by a browser
      # (unspecified IPs are already rejected by _extract_request_host_name).
      return True
    if hostname in exact_allowed_hosts:
      return True
    return any(
        len(hostname) > len(suffix) and hostname.endswith(suffix)
        for suffix in wildcard_suffixes
    )

  def _not_found_handler(environ, start_response):
    return _plain_response(
        environ, start_response, "404 Not Found", b"Not Found"
    )

  def application(
      environ: Mapping[str, Any], start_response: Callable[..., Any]
  ) -> Iterable[bytes]:
    if "HTTP_HOST" in environ:
      if not _is_allowed_host(environ.get("HTTP_HOST")):
        return _plain_response(
            environ, start_response, "403 Forbidden", b"Forbidden"
        )
    elif (environ.get("SERVER_PROTOCOL") or "").upper() == "HTTP/1.1":
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


def run_server(
    plugin,
    host,
    port,
    allowed_hosts: str | Sequence[str] | None = None,
):
  """Starts a webserver for the standalone server."""

  app = make_wsgi_app(plugin, host=host, allowed_hosts=allowed_hosts)

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
  run_server(
      plugin, config.host, config.port, allowed_hosts=config.allowed_hosts
  )


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
    allowed_hosts: str | Sequence[str] | None = None,
):
  """Starts the XProf web server."""
  normalized_host = _normalize_bind_host(host)
  normalized_allowed_hosts = _normalize_allowed_hosts(allowed_hosts)
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
      allowed_hosts=normalized_allowed_hosts,
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
      "--allowed_hosts",
      default=get_default_allowed_hosts(),
      help=(
          "Comma-separated Host header hostnames to allow (supports"
          " '*.example.com' or '*' to disable hostname filtering). Defaults to"
          " XPROF_ALLOWED_HOSTS if set."
      ),
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
      allowed_hosts=args.allowed_hosts,
  )
  return 0


if __name__ == "__main__":
  sys.exit(main())
