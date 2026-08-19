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

  def test_get_run_dir_with_direct_dir(self):
    client = xprof_client.LocalXprofClient()
    run_dir = client.get_run_dir('/tmp')
    self.assertEqual(str(run_dir), '/tmp')

  def test_get_xspace_paths_single_file(self):
    client = xprof_client.LocalXprofClient()
    with mock.patch('pathlib.Path.is_file', return_value=True):
      paths = client.get_xspace_paths('/tmp/test.xplane.pb')
      self.assertEqual(paths, ['/tmp/test.xplane.pb'])


if __name__ == '__main__':
  unittest.main()
