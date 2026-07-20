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
"""Tool taxonomy and host-mode policies for the profile plugin."""

from __future__ import annotations

from collections.abc import Iterable

# On-disk raw profile formats (extension without host prefix).
# Historical tool-name suffixes (handled by the convert layer / frontend):
#   '^'  generated from XPlane
#   '#'  gzip payload
#   '@'  streaming / proto-backed (e.g. trace_viewer@)
#   none JSON ready for the UI
TOOLS = {
    'xplane': 'xplane.pb',
    'hlo_proto': 'hlo_proto.pb',
}

# Tools whose data is produced from XPlane session files.
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
    'perf_counters',
    'utilization_viewer',
    'smart_suggestion',
]

XPLANE_TOOLS_SET = frozenset(XPLANE_TOOLS)
DEFAULT_CACHE_TOOLS = ('overview_page', 'trace_viewer@')

# Tools that can aggregate across hosts (UI may offer ALL_HOSTS).
XPLANE_TOOLS_ALL_HOSTS_SUPPORTED = frozenset([
    'input_pipeline_analyzer',
    'framework_op_stats',
    'kernel_stats',
    'overview_page',
    'pod_viewer',
    'megascale_stats',
])

# Tools that only expose the ALL_HOSTS aggregate when multiple hosts exist.
XPLANE_TOOLS_ALL_HOSTS_ONLY = frozenset(
    ['overview_page', 'pod_viewer', 'smart_suggestion']
)

# Tools that can also be driven from standalone HLO protos.
HLO_TOOLS = frozenset(['graph_viewer', 'memory_viewer'])

# Preferred UI ordering for available tools.
TOOL_SORT_ORDER = [
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

_MULTI_HOST_SELECTION_TOOLS = frozenset(['trace_viewer', 'trace_viewer@'])


def use_xplane(tool: str) -> bool:
  """Return whether ``tool`` is produced from XPlane session files."""
  return tool in XPLANE_TOOLS_SET


def use_hlo(tool: str) -> bool:
  """Return whether ``tool`` can be produced from standalone HLO protos."""
  return tool in HLO_TOOLS


def supports_multi_host_selection(tool: str) -> bool:
  """Return whether the client may pass ``hosts=h1,h2,...`` for this tool."""
  return tool in _MULTI_HOST_SELECTION_TOOLS


def sort_tools(tools: Iterable[str]) -> list[str]:
  """Sort tool names by UI preference, then alphabetically for the rest."""
  tool_set = set(tools)
  ordered = [name for name in TOOL_SORT_ORDER if name in tool_set]
  remaining = sorted(tool_set.difference(ordered))
  ordered.extend(remaining)
  return ordered
