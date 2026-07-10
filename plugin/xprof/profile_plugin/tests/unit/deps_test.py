"""Smoke tests for deps protocols."""

from __future__ import annotations

import unittest

from xprof.profile_plugin.deps import ConvertPort, FileSystemFactory, VersionProvider


class _FakeConvert:
  def xspace_to_tool_data(self, xspace_paths, tool, params):
    return '{}', 'application/json'

  def xspace_to_tool_names(self, xspace_paths):
    return ['overview_page']

  def json_to_csv_string(self, data):
    return 'a,b\n'


class DepsTest(unittest.TestCase):

  def test_fake_convert_is_convert_port(self):
    self.assertIsInstance(_FakeConvert(), ConvertPort)


if __name__ == '__main__':
  unittest.main()
