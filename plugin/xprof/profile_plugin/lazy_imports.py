# Copyright 2026 The XProf Authors. All Rights Reserved.
"""Lazy loaders for C++ convert/pywrap (import only when a route needs them)."""

from __future__ import annotations

import functools


@functools.lru_cache(maxsize=1)
def load_convert_module():
  """Import convert helpers once (requires generated protos / pywrap)."""
  from xprof.convert import raw_to_tool_data as convert  # pylint: disable=g-import-not-at-top
  return convert


@functools.lru_cache(maxsize=1)
def load_pywrap_module():
  """Import the native profiler extension once."""
  from xprof.convert import _pywrap_profiler_plugin as pywrap  # pylint: disable=g-import-not-at-top
  return pywrap
