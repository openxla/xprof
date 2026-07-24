# Copyright 2026 The XProf Authors. All Rights Reserved.
"""Service spies: /data and /runs still delegate to domain services."""

from __future__ import annotations

import tempfile
import unittest
from unittest import mock

from xprof.profile_plugin.constants import DATA_ROUTE, RUNS_ROUTE
from xprof.profile_plugin.models import ToolRequest, ToolResult
from xprof.profile_plugin.tests.fixtures.logdir_factory import (
    create_two_host_session,
)
from xprof.profile_plugin.tests.fixtures.plugin_factory import make_plugin
from xprof.profile_plugin.tests.http_contract.wsgi_client import (
    call_route,
    json_body,
)


class ServiceSpiesTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self._td = tempfile.TemporaryDirectory()
    self.logdir = self._td.name
    self.session = 'session_a'
    create_two_host_session(self.logdir, session=self.session)
    self.plugin = make_plugin(self.logdir)

  def tearDown(self):
    self._td.cleanup()
    super().tearDown()

  def test_data_route_calls_tool_data_service_once(self):
    sentinel = ToolResult(
        data='{"spy": true}',
        content_type='application/json',
    )
    with mock.patch.object(
        self.plugin._tool_data,
        'get_tool_data',
        return_value=sentinel,
    ) as spy:
      result = call_route(
          self.plugin,
          DATA_ROUTE,
          query={
              'run': self.session,
              'tag': 'overview_page',
              'host': 'ALL_HOSTS',
          },
      )
    self.assertEqual(result.status_code, 200)
    self.assertEqual(json_body(result), {'spy': True})
    spy.assert_called_once()
    req = spy.call_args.args[0]
    self.assertIsInstance(req, ToolRequest)
    self.assertEqual(req.run, self.session)
    self.assertEqual(req.tool, 'overview_page')

  def test_runs_route_uses_run_discovery_sorted_reverse(self):
    fixed = ['run_b', 'run_a', 'run_c']
    with mock.patch.object(
        self.plugin._run_discovery,
        'iter_frontend_runs',
        return_value=iter(fixed),
    ) as spy:
      result = call_route(self.plugin, RUNS_ROUTE)
    self.assertEqual(result.status_code, 200)
    self.assertEqual(json_body(result), sorted(fixed, reverse=True))
    spy.assert_called_once()


if __name__ == '__main__':
  unittest.main()
