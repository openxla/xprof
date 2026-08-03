# Copyright 2026 The XProf Authors. All Rights Reserved.
"""Parse HTTP query args into domain ToolRequest (edge adapter)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from xprof.profile_plugin.http.request_params import get_bool_arg
from xprof.profile_plugin.models import ToolRequest


def tool_request_from_args(args: Mapping[str, Any]) -> ToolRequest:
  """Build ToolRequest from a Werkzeug-like args mapping."""
  hosts_param = args.get('hosts') or ''
  if isinstance(hosts_param, str) and hosts_param.strip():
    hosts = tuple(h.strip() for h in hosts_param.split(',') if h.strip())
  else:
    hosts = ()
  # Normalize raw_args to str values for option builders.
  raw_args = {
      str(k): '' if v is None else str(v) for k, v in args.items()
  }
  return ToolRequest(
      run=str(args.get('run') or ''),
      tool=str(args.get('tag') or ''),
      host=(str(args['host']) if args.get('host') is not None else None),
      hosts=hosts,
      use_saved_result=get_bool_arg(args, 'use_saved_result', True),
      raw_args=raw_args,
  )
