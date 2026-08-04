# Copyright 2026 The XProf Authors. All Rights Reserved.
"""Default option builder (common fields only)."""

from __future__ import annotations

from typing import Any

from xprof.profile_plugin.models import ToolRequest


def build_default_family(req: ToolRequest) -> dict[str, Any]:
  """No family-specific extras; common params cover the convert dict."""
  del req
  return {}
