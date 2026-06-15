"""Unit tests for OSS LocalXprofClient logdir configuration."""

import sys
import unittest
from unittest import mock

# Mock xprof.convert so it can be imported in google3 test environment
sys.modules['xprof'] = mock.MagicMock()
sys.modules['xprof.convert'] = mock.MagicMock()

# pylint: disable=g-import-not-at-top
from xprof.cli.internal.oss import xprof_client


class OssXprofClientTest(unittest.TestCase):

  def test_set_logdir_none(self):
    client = xprof_client.LocalXprofClient()
    client.set_logdir(None)
    self.assertIsNone(client.logdir)

  def test_set_logdir_str(self):
    client = xprof_client.LocalXprofClient()
    client.set_logdir('/tmp/test')
    self.assertIsNotNone(client.logdir)


if __name__ == '__main__':
  unittest.main()
