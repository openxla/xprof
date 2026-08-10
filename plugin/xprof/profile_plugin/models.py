# Copyright 2026 The XProf Authors. All Rights Reserved.
"""Domain models for the profile plugin (Werkzeug-free)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SessionRef:
  """Resolved profile session directory for a frontend run name."""

  frontend_run: str
  directory: str


@dataclass(frozen=True)
class ToolRequest:
  """Normalized tool request after HTTP parsing."""

  run: str
  tool: str
  host: str | None
  hosts: tuple[str, ...]
  use_saved_result: bool
  raw_args: Mapping[str, str]


@dataclass(frozen=True)
class HostSelection:
  """Hosts and corresponding XPlane asset paths for a tool request."""

  selected_hosts: tuple[str, ...]
  asset_paths: tuple[Any, ...]  # epath.Path-like; avoid hard epath dep here


@dataclass(frozen=True)
class ToolResult:
  """Payload returned to the HTTP layer."""

  data: bytes | str | None
  content_type: str
  content_encoding: str | None = None
