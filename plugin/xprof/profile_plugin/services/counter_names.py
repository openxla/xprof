# Copyright 2026 The XProf Authors. All Rights Reserved.
"""perf_counters names_only handler (outside XPlane convert path)."""

from __future__ import annotations

import json
import logging

from xprof.profile_plugin.models import ToolRequest, ToolResult

logger = logging.getLogger('tensorboard.plugins.profile')


def counter_names_result(device_type: str) -> ToolResult:
  """Return JSON list of counter names for ``device_type``."""
  try:
    from xprof.convert import counter_extractor  # pylint: disable=g-import-not-at-top

    names = counter_extractor.get_all_counters(device_type)
  except FileNotFoundError:
    logger.warning(
        'Failed to get counter names for device type: %s', device_type
    )
    names = []
  return ToolResult(data=json.dumps(names), content_type='application/json')


def try_counter_names_only(req: ToolRequest) -> ToolResult | None:
  """Handle ``perf_counters`` + ``names_only=1``, else return None.

  Raises:
    ValueError: If names_only is set but ``device_type`` is missing.
  """
  if req.tool != 'perf_counters':
    return None
  if req.raw_args.get('names_only') != '1':
    return None
  device_type = req.raw_args.get('device_type')
  if not device_type:
    raise ValueError(
        'device_type is required for perf_counters with names_only'
    )
  return counter_names_result(device_type)
