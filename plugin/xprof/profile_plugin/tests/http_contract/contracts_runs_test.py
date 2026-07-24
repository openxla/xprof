# Copyright 2026 The XProf Authors. All Rights Reserved.
"""HTTP contracts: /runs."""

from __future__ import annotations

import tempfile
import unittest

from xprof.profile_plugin.constants import RUNS_ROUTE
from xprof.profile_plugin.tests.fixtures.logdir_factory import (
    create_two_host_session,
)
from xprof.profile_plugin.tests.fixtures.plugin_factory import make_plugin
from xprof.profile_plugin.tests.http_contract.wsgi_client import (
    call_route,
    json_body,
)


class ContractsRunsTest(unittest.TestCase):

  def test_runs_contains_session(self):
    with tempfile.TemporaryDirectory() as logdir:
      session = 'session_a'
      create_two_host_session(logdir, session=session)
      plugin = make_plugin(logdir)
      result = call_route(plugin, RUNS_ROUTE)
      self.assertEqual(result.status_code, 200)
      runs = json_body(result)
      self.assertIsInstance(runs, list)
      self.assertIn(session, runs)

  def test_runs_empty_logdir(self):
    with tempfile.TemporaryDirectory() as logdir:
      plugin = make_plugin(logdir)
      result = call_route(plugin, RUNS_ROUTE)
      self.assertEqual(result.status_code, 200)
      self.assertEqual(json_body(result), [])


if __name__ == '__main__':
  unittest.main()
