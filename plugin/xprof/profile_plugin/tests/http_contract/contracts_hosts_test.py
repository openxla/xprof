# Copyright 2026 The XProf Authors. All Rights Reserved.
"""HTTP contracts: /hosts."""

from __future__ import annotations

import tempfile
import unittest

from xprof.profile_plugin.constants import ALL_HOSTS, HOSTS_ROUTE
from xprof.profile_plugin.tests.fixtures.logdir_factory import (
    create_two_host_session,
)
from xprof.profile_plugin.tests.fixtures.plugin_factory import make_plugin
from xprof.profile_plugin.tests.http_contract.wsgi_client import (
    call_route,
    json_body,
)


class ContractsHostsTest(unittest.TestCase):

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

  def test_hosts_overview_includes_all_hosts(self):
    result = call_route(
        self.plugin,
        HOSTS_ROUTE,
        query={'run': self.session, 'tag': 'overview_page'},
    )
    self.assertEqual(result.status_code, 200)
    hosts = json_body(result)
    self.assertIsInstance(hosts, list)
    self.assertIn({'hostname': ALL_HOSTS}, hosts)

  def test_hosts_missing_run_returns_empty_list(self):
    """Missing run_dir logs a warning and returns [] (200), no crash."""
    result = call_route(
        self.plugin,
        HOSTS_ROUTE,
        query={'run': 'no_such_run', 'tag': 'overview_page'},
    )
    self.assertEqual(result.status_code, 200)
    self.assertEqual(json_body(result), [])


if __name__ == '__main__':
  unittest.main()
