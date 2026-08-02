# Copyright 2026 The XProf Authors. All Rights Reserved.
"""Common tool option helpers (Werkzeug-free)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from xprof.profile_plugin.models import ToolRequest


def get_bool_arg(args: Mapping[str, Any], arg_name: str, default: bool) -> bool:
  """Parse a boolean query argument (``true``/``false`` case-insensitive)."""
  arg_str = args.get(arg_name)
  if arg_str is None:
    return default
  return str(arg_str).lower() == 'true'


def build_common_params(
    req: ToolRequest,
    *,
    use_saved_result: bool,
    graph_options: Mapping[str, Any],
) -> dict[str, Any]:
  """Build convert params shared by all tool families.

  Mirrors the common block in ``ProfilePlugin.data_impl``.
  """
  args = req.raw_args
  params: dict[str, Any] = {
      'graph_viewer_options': dict(graph_options),
      'tqx': args.get('tqx'),
      'perfetto': get_bool_arg(args, 'perfetto', False),
      'host': req.host,
      'module_name': args.get('module_name'),
      'program_id': args.get('program_id'),
      'use_saved_result': use_saved_result,
  }
  if args.get('group_by'):
    params['group_by'] = args.get('group_by')
  if args.get('refresh_suggestion'):
    params['refresh_suggestion'] = args.get('refresh_suggestion')
  # Always present (same as data_impl); default '0'.
  params['memory_space'] = args.get('memory_space', '0')
  return params
