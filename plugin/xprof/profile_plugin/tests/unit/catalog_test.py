"""Tests for tools.catalog without C++ convert."""

from __future__ import annotations

import unittest
from unittest import mock

from xprof.profile_plugin.tools import catalog
from xprof.profile_plugin.tools.filenames import make_filename


class CatalogTest(unittest.TestCase):

  def test_xplane_without_run_dir_adds_synthetic_tools(self):
    tools = catalog.get_tools_from_filenames(
        [make_filename("host", "xplane")], ""
    )
    self.assertIn("overview_page", tools)
    self.assertIn("trace_viewer@", tools)

  def test_get_tools_skips_hlo_proto(self):
    tools = catalog.get_tools_from_filenames(
        [make_filename("mod", "hlo_proto")], "/tmp/run"
    )
    self.assertEqual(tools, set())

  def test_get_active_tools_prefers_streaming_trace_viewer(self):
    with mock.patch.object(
        catalog,
        "get_tools_from_filenames",
        return_value={"trace_viewer", "trace_viewer@", "overview_page"},
    ):
      tools = catalog.get_active_tools(["ignored"], "/tmp/run")
    self.assertNotIn("trace_viewer", tools)
    self.assertIn("trace_viewer@", tools)
    self.assertEqual(tools[0], "overview_page")

  def test_xplane_convert_failure_returns_empty(self):
    """When convert is present but broken, catalog logs and returns empty."""
    fake = mock.MagicMock()
    fake.xspace_to_tool_names.side_effect = AttributeError("no convert")
    # Inject as the module object returned by `from xprof.convert import raw_to_tool_data`
    import types
    convert_pkg = types.ModuleType("xprof.convert")
    with mock.patch.dict(
        "sys.modules",
        {
            "xprof.convert": convert_pkg,
            "xprof.convert.raw_to_tool_data": fake,
        },
    ):
      tools = catalog.get_tools_from_filenames(
          [make_filename("h", "xplane")], "/tmp/session"
      )
    self.assertEqual(tools, set())


if __name__ == "__main__":
  unittest.main()
