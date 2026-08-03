# Copyright 2026 The XProf Authors. All Rights Reserved.
"""Map tool names to family option builders."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from xprof.profile_plugin.models import ToolRequest
from xprof.profile_plugin.tools.options.default import build_default_family
from xprof.profile_plugin.tools.options.graph import build_graph_family
from xprof.profile_plugin.tools.options.memory import build_memory_family
from xprof.profile_plugin.tools.options.trace import build_trace_family

FamilyBuilder = Callable[[ToolRequest], Mapping[str, Any]]

_FAMILY_BUILDERS: dict[str, FamilyBuilder] = {
    'trace_viewer': build_trace_family,
    'trace_viewer@': build_trace_family,
    'memory_viewer': build_memory_family,
    'graph_viewer': build_graph_family,
}


def get_family_builder(tool: str) -> FamilyBuilder:
  """Return the option builder for ``tool``, or the default builder."""
  return _FAMILY_BUILDERS.get(tool, build_default_family)
