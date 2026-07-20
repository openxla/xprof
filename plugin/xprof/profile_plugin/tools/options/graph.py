# Copyright 2026 The XProf Authors. All Rights Reserved.
"""Graph viewer option builder."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from xprof.profile_plugin.models import ToolRequest


def build_graph_viewer_options(args: Mapping[str, Any]) -> dict[str, Any]:
  """Pack ``graph_viewer_options`` from query args.

  Ported from ``ProfilePlugin._get_graph_viewer_options``.
  """
  node_name = args.get('node_name')
  module_name = args.get('module_name')
  graph_width_str = args.get('graph_width') or ''
  graph_width = int(graph_width_str) if str(graph_width_str).isdigit() else 3
  show_metadata = int(str(args.get('show_metadata')) == 'true')
  merge_fusion = int(str(args.get('merge_fusion')) == 'true')
  program_id = args.get('program_id')
  return {
      'node_name': node_name,
      'module_name': module_name,
      'program_id': program_id,
      'graph_width': graph_width,
      'show_metadata': show_metadata,
      'merge_fusion': merge_fusion,
      'format': args.get('format'),
      'type': args.get('type'),
  }


def build_graph_family(req: ToolRequest) -> dict[str, Any]:
  """Family extras for graph_viewer (options live in common params)."""
  del req  # graph_viewer_options is always set in common params
  return {}
