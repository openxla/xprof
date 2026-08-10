# Copyright 2026 The XProf Authors. All Rights Reserved.
"""HTTP contracts: /data happy + 404 No Data."""

from __future__ import annotations

import json
import tempfile
import unittest

from xprof.profile_plugin.constants import DATA_ROUTE
from xprof.profile_plugin.tests.fixtures.logdir_factory import (
    create_two_host_session,
)
from xprof.profile_plugin.tests.fixtures.plugin_factory import make_plugin
from xprof.profile_plugin.tests.http_contract.wsgi_client import (
    call_route,
    header_value,
    json_body,
    text_body,
)


class ContractsDataTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self._td = tempfile.TemporaryDirectory()
    self.logdir = self._td.name
    self.session = 'session_a'
    create_two_host_session(self.logdir, session=self.session)

  def tearDown(self):
    self._td.cleanup()
    super().tearDown()

  def _query(self):
    return {
        'run': self.session,
        'tag': 'overview_page',
        'host': 'ALL_HOSTS',
    }

  def test_data_success_gzip_json(self):
    def ok_convert(paths, tool, params):
      return json.dumps({'ok': True, 'tool': tool}, sort_keys=True), 'application/json'

    plugin = make_plugin(self.logdir, xspace_fn=ok_convert)
    result = call_route(plugin, DATA_ROUTE, query=self._query())
    self.assertEqual(result.status_code, 200)
    self.assertEqual(header_value(result, 'Content-Encoding'), 'gzip')
    self.assertEqual(json_body(result), {'ok': True, 'tool': 'overview_page'})

  def test_data_missing_404_no_data(self):
    def empty_convert(paths, tool, params):
      return None, 'application/json'

    plugin = make_plugin(self.logdir, xspace_fn=empty_convert)
    result = call_route(plugin, DATA_ROUTE, query=self._query())
    self.assertEqual(result.status_code, 404)
    self.assertEqual(text_body(result), 'No Data')


if __name__ == '__main__':
  unittest.main()
