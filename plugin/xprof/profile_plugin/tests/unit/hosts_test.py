"""Unit tests for HostSelector."""

from __future__ import annotations

import os
import unittest

from xprof.profile_plugin.constants import ALL_HOSTS
from xprof.profile_plugin.services.hosts import HostSelector


def _join(*parts):
  return os.path.join(*parts)


class HostSelectorTest(unittest.TestCase):

  def setUp(self):
    self.selector = HostSelector()
    self.run_dir = '/runs/session'
    self.basenames = ('host0.xplane.pb', 'host1.xplane.pb')

  def test_all_hosts_returns_all_assets(self):
    sel = self.selector.select(
        run_dir=self.run_dir,
        tool='overview_page',
        host=ALL_HOSTS,
        hosts_param=None,
        xplane_basenames=self.basenames,
        path_join=_join,
    )
    self.assertEqual(set(sel.selected_hosts), {'host0', 'host1'})
    self.assertEqual(len(sel.asset_paths), 2)
    self.assertEqual(
        set(sel.asset_paths),
        {
            os.path.join(self.run_dir, 'host0.xplane.pb'),
            os.path.join(self.run_dir, 'host1.xplane.pb'),
        },
    )

  def test_single_host(self):
    sel = self.selector.select(
        run_dir=self.run_dir,
        tool='kernel_stats',
        host='host0',
        hosts_param=None,
        xplane_basenames=self.basenames,
        path_join=_join,
    )
    self.assertEqual(sel.selected_hosts, ('host0',))
    self.assertEqual(
        sel.asset_paths, (os.path.join(self.run_dir, 'host0.xplane.pb'),)
    )

  def test_hosts_csv_for_trace_viewer_at(self):
    sel = self.selector.select(
        run_dir=self.run_dir,
        tool='trace_viewer@',
        host=None,
        hosts_param='host0,host1',
        xplane_basenames=self.basenames,
        path_join=_join,
    )
    self.assertEqual(sel.selected_hosts, ('host0', 'host1'))
    self.assertEqual(
        list(sel.asset_paths),
        [
            os.path.join(self.run_dir, 'host0.xplane.pb'),
            os.path.join(self.run_dir, 'host1.xplane.pb'),
        ],
    )

  def test_missing_host_raises(self):
    with self.assertRaises(FileNotFoundError) as ctx:
      self.selector.select(
          run_dir=self.run_dir,
          tool='kernel_stats',
          host='missing_host',
          hosts_param=None,
          xplane_basenames=self.basenames,
          path_join=_join,
      )
    self.assertIn('missing_host', str(ctx.exception))

  def test_missing_host_in_csv_raises(self):
    with self.assertRaises(FileNotFoundError) as ctx:
      self.selector.select(
          run_dir=self.run_dir,
          tool='trace_viewer@',
          host=None,
          hosts_param='host0,nope',
          xplane_basenames=self.basenames,
          path_join=_join,
      )
    self.assertIn('nope', str(ctx.exception))

  def test_no_xplanes_raises(self):
    with self.assertRaises(FileNotFoundError):
      self.selector.select(
          run_dir=self.run_dir,
          tool='overview_page',
          host=ALL_HOSTS,
          hosts_param=None,
          xplane_basenames=(),
          path_join=_join,
      )


if __name__ == '__main__':
  unittest.main()
