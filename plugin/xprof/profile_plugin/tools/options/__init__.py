# Copyright 2026 The XProf Authors. All Rights Reserved.
"""Tool option builders: query args → convert option dicts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from xprof.profile_plugin.models import ToolRequest
from xprof.profile_plugin.tools.options.base import build_common_params
from xprof.profile_plugin.tools.options.graph import build_graph_viewer_options
from xprof.profile_plugin.tools.options.registry import get_family_builder

__all__ = [
    'build_graph_viewer_options',
    'build_tool_params',
]


def build_tool_params(
    req: ToolRequest,
    *,
    use_saved_result: bool | None = None,
    graph_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
  """Merge common convert fields with the tool-family option builder.

  Args:
    req: Normalized tool request (``raw_args`` holds residual query keys).
    use_saved_result: Override for cache policy result; defaults to
      ``req.use_saved_result`` when omitted.
    graph_options: Prebuilt ``graph_viewer_options``; when omitted, built from
      ``req.raw_args``.

  Returns:
    Dict suitable for ``ConvertPort.xspace_to_tool_data(..., options)``.
  """
  if use_saved_result is None:
    use_saved_result = req.use_saved_result
  if graph_options is None:
    graph_options = build_graph_viewer_options(req.raw_args)
  params = build_common_params(
      req,
      use_saved_result=use_saved_result,
      graph_options=graph_options,
  )
  family = get_family_builder(req.tool)
  params.update(family(req))
  return params
