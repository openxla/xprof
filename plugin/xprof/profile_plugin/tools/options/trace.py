# Copyright 2026 The XProf Authors. All Rights Reserved.
"""Trace viewer option builder."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from xprof.profile_plugin.models import ToolRequest
from xprof.profile_plugin.tools.options.base import get_bool_arg


def build_trace_viewer_options(args: Mapping[str, Any]) -> dict[str, Any]:
  """Pack ``trace_viewer_options`` from query args.

  When ``event_name`` is set, ``format`` is forced to ``json`` (event detail
  fetch must not use the compressed protobuf path).
  """
  options: dict[str, Any] = {
      'resolution': args.get('resolution', 8000),
      'full_dma': get_bool_arg(args, 'full_dma', False),
      'enable_legacy_dcn': get_bool_arg(args, 'enable_legacy_dcn', False),
  }
  if args.get('start_time_ms') is not None:
    options['start_time_ms'] = args.get('start_time_ms')
  if args.get('end_time_ms') is not None:
    options['end_time_ms'] = args.get('end_time_ms')
  event_name = args.get('event_name')
  format_arg = args.get('format')
  if event_name is not None:
    options['event_name'] = event_name
    # Event selection (details) always uses JSON, not compressed protobuf.
    options['format'] = 'json'
  elif format_arg is not None:
    options['format'] = format_arg
  if args.get('duration_ms') is not None:
    options['duration_ms'] = args.get('duration_ms')
  if args.get('unique_id') is not None:
    options['unique_id'] = args.get('unique_id')
  if args.get('search_prefix') is not None:
    options['search_prefix'] = args.get('search_prefix')
  if args.get('search_metadata') is not None:
    options['search_metadata'] = get_bool_arg(args, 'search_metadata', False)
  return options


def build_trace_family(req: ToolRequest) -> dict[str, Any]:
  """Family extras for trace_viewer / trace_viewer@."""
  return {'trace_viewer_options': build_trace_viewer_options(req.raw_args)}
