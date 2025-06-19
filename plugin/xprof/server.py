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
import socket
import sys

from cheroot import wsgi
from etils import epath
from xprof.profile_plugin_loader import ProfilePluginLoader
from xprof.standalone.base_plugin import TBContext
from xprof.standalone.plugin_event_multiplexer import DataProvider


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


def launch_server(logdir, port):
  context = TBContext(logdir, DataProvider(logdir), TBContext.Flags(False))
  loader = ProfilePluginLoader()
  plugin = loader.load(context)
  run_server(plugin, _get_wildcard_address(port), port)


def main() -> int:
  """Parses command-line arguments and launches the XProf server."""
  parser = argparse.ArgumentParser(
      prog="xprof",
      description="Launch the XProf profiling server.",
      formatter_class=argparse.RawDescriptionHelpFormatter,
      epilog="Examples:\n"
      "\txprof ~/jax/profile-logs -p 8080\n"
      "\txprof --logdir ~/jax/profile-logs -p 8080",
  )

  logdir_group = parser.add_mutually_exclusive_group(required=True)

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

  try:
    args = parser.parse_args()
  except SystemExit as e:
    return e.code

  logdir = args.logdir_opt or args.logdir_pos
  port = args.port

  print("Attempting to start XProf server:")
  print(f"  Log Directory: {logdir}")
  print(f"  Port: {port}")

  if not epath.Path(logdir).exists():
    print(
        f"Error: Log directory '{logdir}' does not exist or is not a"
        " directory.",
        file=sys.stderr,
    )
    return 1

  launch_server(logdir, port)
  return 0
