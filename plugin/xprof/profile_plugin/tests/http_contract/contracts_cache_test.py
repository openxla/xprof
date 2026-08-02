# Copyright 2026 The XProf Authors. All Rights Reserved.
"""HTTP contracts: /generate_cache GET 405 + POST 202."""

from __future__ import annotations

import tempfile
import unittest
from unittest import mock

from xprof.profile_plugin.constants import GENERATE_CACHE_ROUTE
from xprof.profile_plugin.tests.fixtures.logdir_factory import (
    create_two_host_session,
)
from xprof.profile_plugin.tests.fixtures.plugin_factory import make_plugin
from xprof.profile_plugin.tests.http_contract.wsgi_client import (
    call_route,
    json_body,
    text_body,
)


class ContractsCacheTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self._td = tempfile.TemporaryDirectory()
    self.logdir = self._td.name
    self.session = 'session_a'
    self.run_dir = create_two_host_session(self.logdir, session=self.session)
    self.plugin = make_plugin(self.logdir)

  def tearDown(self):
    self._td.cleanup()
    super().tearDown()

  def test_generate_cache_get_method_not_allowed(self):
    result = call_route(self.plugin, GENERATE_CACHE_ROUTE, method='GET')
    self.assertEqual(result.status_code, 405)
    self.assertEqual(text_body(result), 'Method Not Allowed')

  def test_generate_cache_post_accepted(self):
    # Patch run_tools_imp so tool discovery succeeds without importing native
    # convert/protobuf (raw_to_tool_data is unavailable outside run_all_tests
    # stubs). CacheApi calls run_tools_imp after session_path validation.
    with mock.patch.object(
        self.plugin,
        'run_tools_imp',
        return_value=['overview_page', 'trace_viewer@'],
    ):
      result = call_route(
          self.plugin,
          GENERATE_CACHE_ROUTE,
          method='POST',
          query={
              'session_path': self.run_dir,
              'tools': 'overview_page',
          },
      )
    self.assertEqual(result.status_code, 202, msg=text_body(result))
    payload = json_body(result)
    self.assertEqual(payload['status'], 'ACCEPTED')
    self.assertIn('message', payload)


if __name__ == '__main__':
  unittest.main()
