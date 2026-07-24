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
"""Filename encoding/decoding for profile run directories."""

from __future__ import annotations

import re
from collections.abc import Sequence

from xprof.profile_plugin.constants import ALL_HOSTS
from xprof.profile_plugin.tools.registry import (
    TOOLS,
    XPLANE_TOOLS_ALL_HOSTS_ONLY,
    XPLANE_TOOLS_ALL_HOSTS_SUPPORTED,
    use_hlo,
    use_xplane,
)

_EXTENSION_TO_TOOL = {extension: tool for tool, extension in TOOLS.items()}
_EXTENSION_TO_TOOL['xplane.riegeli'] = 'xplane'

_FILENAME_RE = re.compile(
    r"""
    (?:            # Start optional non-capturing group for the host.
      (.*)         #   Capture group 1: The host name.
      \.           #   A literal dot.
    )?             # End optional non-capturing group.
    (              # Start capture group 2: The tool extension.
    """
    + '|'.join(re.escape(v) for v in _EXTENSION_TO_TOOL.keys())
    + r"""
    )              # End capture group 2.
    """,
    re.VERBOSE,
)


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


def parse_filename(filename: str) -> tuple[str | None, str | None]:
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


def get_hosts(filenames: Sequence[str]) -> set[str]:
  """Parses a sequence of filenames and returns the set of hosts.

  Args:
    filenames: A sequence of filenames (just basenames, no directory).

  Returns:
    A set of host names encoded in the filenames.
  """
  hosts = set()
  for name in filenames:
    host, _ = parse_filename(name)
    if host:
      hosts.add(host)
  return hosts


def hosts_from_xplane_filenames(
    filenames: Sequence[str], tool: str
) -> Sequence[str]:
  """Converts a sequence of filenames to a list of host names given a tool.

  Args:
    filenames: A sequence of filenames.
    tool: A string representing the profiling tool.

  Returns:
    A list of hostnames.
  """
  hosts = get_hosts(filenames)
  if len(hosts) > 1:
    if tool in XPLANE_TOOLS_ALL_HOSTS_ONLY:
      hosts = [ALL_HOSTS]
    elif tool in XPLANE_TOOLS_ALL_HOSTS_SUPPORTED:
      hosts.add(ALL_HOSTS)
  return sorted(hosts)
