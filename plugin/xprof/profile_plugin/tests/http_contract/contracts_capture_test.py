# Copyright 2026 The XProf Authors. All Rights Reserved.
"""HTTP contracts: /capture_profile error path (no TF)."""

from __future__ import annotations

import tempfile
import unittest

from xprof.profile_plugin.constants import CAPTURE_ROUTE
from xprof.profile_plugin.tests.fixtures.logdir_factory import (
    create_two_host_session,
)
from xprof.profile_plugin.tests.fixtures.plugin_factory import make_plugin
from xprof.profile_plugin.tests.http_contract.wsgi_client import (
    call_route,
    json_body,
)


class ContractsCaptureTest(unittest.TestCase):

  def test_tpu_capture_without_tf_returns_500_json_error(self):
    with tempfile.TemporaryDirectory() as logdir:
      create_two_host_session(logdir)
      plugin = make_plugin(logdir)
      # Force error contract even when TensorFlow is installed in the env.
      plugin._tf_profiler = None
      result = call_route(
          plugin,
          CAPTURE_ROUTE,
          query={
              'service_addr': 'my-tpu',
              'is_tpu_name': 'true',
              'duration': '1000',
          },
      )
      self.assertEqual(result.status_code, 500)
      payload = json_body(result)
      self.assertIn('error', payload)
      self.assertIn('TensorFlow', payload['error'])
      self.assertIn('not installed', payload['error'].lower())


if __name__ == '__main__':
  unittest.main()
