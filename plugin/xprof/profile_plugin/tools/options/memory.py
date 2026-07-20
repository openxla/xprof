# Copyright 2026 The XProf Authors. All Rights Reserved.
"""Memory viewer option builder."""

from __future__ import annotations

from typing import Any

from xprof.profile_plugin.models import ToolRequest


def build_memory_family(req: ToolRequest) -> dict[str, Any]:
  """Family extras for memory_viewer."""
  extras: dict[str, Any] = {}
  if req.raw_args.get('view_memory_allocation_timeline'):
    extras['view_memory_allocation_timeline'] = True
  return extras
