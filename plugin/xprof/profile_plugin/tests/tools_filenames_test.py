"""Unit tests for filename encoding and host selection policies."""

from __future__ import annotations

import unittest

from xprof.profile_plugin.constants import ALL_HOSTS
from xprof.profile_plugin.tools.filenames import (
    get_hosts,
    hosts_from_xplane_filenames,
    make_filename,
    parse_filename,
)
from xprof.profile_plugin.tools.registry import (
    HLO_TOOLS,
    TOOLS,
    XPLANE_TOOLS,
    sort_tools,
    use_hlo,
    use_xplane,
)


class FilenamesTest(unittest.TestCase):

  def test_make_filename_xplane_tool(self):
    self.assertEqual(
        make_filename('host0', 'overview_page'), 'host0.xplane.pb'
    )
    self.assertEqual(
        make_filename('host0', 'trace_viewer@'), 'host0.xplane.pb'
    )

  def test_make_filename_hlo_tool(self):
    self.assertEqual(
        make_filename('module1', 'memory_viewer'), 'module1.hlo_proto.pb'
    )
    self.assertEqual(
        make_filename('module1', 'graph_viewer'), 'module1.hlo_proto.pb'
    )

  def test_make_filename_empty_host(self):
    self.assertEqual(make_filename('', 'xplane'), 'xplane.pb')
    self.assertEqual(make_filename(None, 'xplane'), 'xplane.pb')  # type: ignore

  def test_parse_filename_xplane(self):
    host, tool = parse_filename('worker-0.xplane.pb')
    self.assertEqual(host, 'worker-0')
    self.assertEqual(tool, 'xplane')

  def test_parse_filename_riegeli(self):
    host, tool = parse_filename('host1.xplane.riegeli')
    self.assertEqual(host, 'host1')
    self.assertEqual(tool, 'xplane')

  def test_parse_filename_hlo(self):
    host, tool = parse_filename('mod.a.b.hlo_proto.pb')
    self.assertEqual(host, 'mod.a.b')
    self.assertEqual(tool, 'hlo_proto')

  def test_parse_filename_unknown(self):
    host, tool = parse_filename('notes.txt')
    self.assertEqual(host, 'notes.txt')
    self.assertIsNone(tool)

  def test_get_hosts(self):
    # Unknown basenames are returned as the "host" component with tool=None
    # (legacy parse_filename behavior), so they appear in the host set.
    hosts = get_hosts(
        ['h1.xplane.pb', 'h2.xplane.pb', 'notes.txt', 'm.hlo_proto.pb']
    )
    self.assertEqual(hosts, {'h1', 'h2', 'notes.txt', 'm'})

  def test_hosts_from_xplane_all_hosts_only(self):
    names = ['a.xplane.pb', 'b.xplane.pb']
    hosts = hosts_from_xplane_filenames(names, 'overview_page')
    self.assertEqual(list(hosts), [ALL_HOSTS])

  def test_hosts_from_xplane_all_hosts_supported(self):
    names = ['a.xplane.pb', 'b.xplane.pb']
    hosts = hosts_from_xplane_filenames(names, 'kernel_stats')
    self.assertEqual(set(hosts), {'a', 'b', ALL_HOSTS})
    self.assertEqual(list(hosts), sorted(hosts))

  def test_hosts_from_xplane_single_host(self):
    names = ['only.xplane.pb']
    hosts = hosts_from_xplane_filenames(names, 'overview_page')
    self.assertEqual(list(hosts), ['only'])

  def test_registry_consistency(self):
    self.assertTrue(use_xplane('overview_page'))
    self.assertTrue(use_hlo('graph_viewer'))
    self.assertFalse(use_xplane('not_a_tool'))
    self.assertIn('xplane.pb', TOOLS.values())
    self.assertTrue(set(HLO_TOOLS).issubset(set(XPLANE_TOOLS)))

  def test_sort_tools_prefers_overview(self):
    ordered = sort_tools({'kernel_stats', 'overview_page', 'zzz_custom'})
    self.assertEqual(ordered[0], 'overview_page')
    self.assertIn('zzz_custom', ordered)
    self.assertEqual(ordered[-1], 'zzz_custom')


if __name__ == '__main__':
  unittest.main()
