#!/usr/bin/env python3
"""Run package unit/integration tests with lightweight stubs for native deps."""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path


def _install_stubs() -> None:
  """Install stubs for C++ extension and generated protos if missing."""
  # protobuf generated module
  if 'xprof.protobuf.trace_events_old_pb2' not in sys.modules:
    try:
      from xprof.protobuf import trace_events_old_pb2  # noqa: F401
    except ImportError:
      pb = types.ModuleType('xprof.protobuf.trace_events_old_pb2')

      class Trace:
        def ParseFromString(self, _data):
          return None

      pb.Trace = Trace
      sys.modules['xprof.protobuf.trace_events_old_pb2'] = pb

  # native pywrap extension
  mod_name = 'xprof.convert._pywrap_profiler_plugin'
  if mod_name not in sys.modules:
    try:
      from xprof.convert import _pywrap_profiler_plugin  # noqa: F401
    except ImportError:
      wrap = types.ModuleType(mod_name)

      def _fail(*_a, **_k):
        return b'', False

      wrap.xspace_to_tools_data = _fail
      wrap.xspace_to_tools_data_from_byte_string = _fail
      wrap.trace = lambda *a, **k: None
      sys.modules[mod_name] = wrap


def main() -> int:
  plugin_root = Path(__file__).resolve().parents[3]  # .../plugin
  if str(plugin_root) not in sys.path:
    sys.path.insert(0, str(plugin_root))

  _install_stubs()

  loader = unittest.TestLoader()
  suite = loader.discover(
      start_dir=str(Path(__file__).resolve().parent),
      pattern='*_test.py',
  )
  result = unittest.TextTestRunner(verbosity=2).run(suite)
  return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
  sys.exit(main())
