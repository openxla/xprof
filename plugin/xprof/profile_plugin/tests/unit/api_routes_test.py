"""Smoke: every frontend API group is registered on ProfilePlugin."""

from __future__ import annotations

import unittest
from unittest import mock

from xprof.profile_plugin.constants import (
    CAPTURE_ROUTE,
    CONFIG_ROUTE,
    DATA_ROUTE,
    GENERATE_CACHE_ROUTE,
    HOSTS_ROUTE,
    RUNS_ROUTE,
    RUN_TOOLS_ROUTE,
)
from xprof.profile_plugin.plugin import ProfilePlugin


class ApiRoutesTest(unittest.TestCase):

  def test_get_plugin_apps_includes_each_frontend_group(self):
    ctx = mock.Mock()
    ctx.logdir = None
    ctx.data_provider = None
    ctx.flags = mock.Mock(master_tpu_unsecure_channel=None)
    plugin = ProfilePlugin(ctx)
    apps = plugin.get_plugin_apps()
    for route in (
        RUNS_ROUTE,
        RUN_TOOLS_ROUTE,
        HOSTS_ROUTE,
        DATA_ROUTE,
        CAPTURE_ROUTE,
        GENERATE_CACHE_ROUTE,
        CONFIG_ROUTE,
    ):
      self.assertIn(route, apps, msg=f'missing route {route}')
      self.assertTrue(callable(apps[route]))


if __name__ == '__main__':
  unittest.main()
