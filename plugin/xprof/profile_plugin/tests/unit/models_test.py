"""Unit tests for profile_plugin.models."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from xprof.profile_plugin.models import (
    HostSelection,
    SessionRef,
    ToolRequest,
    ToolResult,
)


class ModelsTest(unittest.TestCase):

  def test_session_ref_frozen(self):
    ref = SessionRef(frontend_run='s1', directory='/tmp/s1')
    self.assertEqual(ref.frontend_run, 's1')
    with self.assertRaises(Exception):
      ref.frontend_run = 'other'  # type: ignore[misc]

  def test_tool_request_hosts_default_empty(self):
    req = ToolRequest(
        run='r',
        tool='overview_page',
        host=None,
        hosts=(),
        use_saved_result=True,
        raw_args={},
    )
    self.assertEqual(req.hosts, ())
    self.assertTrue(req.use_saved_result)

  def test_host_selection_tuple_paths(self):
    paths = (SimpleNamespace(name='a'),)
    sel = HostSelection(selected_hosts=('h0',), asset_paths=paths)
    self.assertEqual(sel.selected_hosts, ('h0',))
    self.assertEqual(len(sel.asset_paths), 1)

  def test_tool_result_defaults(self):
    result = ToolResult(data='{}', content_type='application/json')
    self.assertIsNone(result.content_encoding)


if __name__ == '__main__':
  unittest.main()
