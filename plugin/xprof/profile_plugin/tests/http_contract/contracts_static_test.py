# Copyright 2026 The XProf Authors. All Rights Reserved.
"""HTTP contracts: static assets + /config."""

from __future__ import annotations

import tempfile
import unittest

from xprof.profile_plugin.constants import CONFIG_ROUTE, INDEX_JS_ROUTE
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


class ContractsStaticTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self._td = tempfile.TemporaryDirectory()
    create_two_host_session(self._td.name)
    self.plugin = make_plugin(self._td.name)

  def tearDown(self):
    self._td.cleanup()
    super().tearDown()

  def test_config_happy_json_keys(self):
    result = call_route(self.plugin, CONFIG_ROUTE)
    self.assertEqual(result.status_code, 200)
    self.assertEqual(header_value(result, 'Content-Encoding'), 'gzip')
    payload = json_body(result)
    self.assertIn('hideCaptureProfileButton', payload)
    self.assertIn('srcPathPrefix', payload)
    self.assertIn('enableTabNameLabel', payload)
    self.assertIsInstance(payload['hideCaptureProfileButton'], bool)
    self.assertIsInstance(payload['srcPathPrefix'], str)

  def test_static_missing_file_404(self):
    # Same handler as index.js; path basename is the file looked up under static/.
    result = call_route(
        self.plugin,
        INDEX_JS_ROUTE,
        path='/this_file_does_not_exist_xyz.js',
    )
    self.assertEqual(result.status_code, 404)
    self.assertEqual(text_body(result), 'Fail to read the files.')


if __name__ == '__main__':
  unittest.main()
