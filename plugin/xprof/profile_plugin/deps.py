# Copyright 2026 The XProf Authors. All Rights Reserved.
"""Injectable ports for convert and filesystem boundaries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ConvertPort(Protocol):
  def xspace_to_tool_data(
      self,
      xspace_paths: Sequence[Any],
      tool: str,
      params: Mapping[str, Any],
  ) -> tuple[bytes | str | None, str]:
    ...

  def xspace_to_tool_names(self, xspace_paths: Sequence[str]) -> Sequence[str]:
    ...

  def json_to_csv_string(self, data: Any) -> str:
    ...


@runtime_checkable
class FileSystemFactory(Protocol):
  def get(self, path: str) -> Any:
    """Return a ProfileFileSystem for path."""
    ...


@runtime_checkable
class VersionProvider(Protocol):
  __version__: str
