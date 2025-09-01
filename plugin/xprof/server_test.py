"""Tests for the XProf server."""

import argparse
import os
from unittest import mock

from absl.testing import parameterized
from etils import epath

from google3.testing.pybase import googletest
from xprof import server


class ServerTest(googletest.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_launch_server = self.enter_context(
        mock.patch.object(server, 'launch_server', autospec=True)
    )
    self.mock_path = self.enter_context(
        mock.patch.object(epath, 'Path', autospec=True)
    )
    self.mock_parse_args = self.enter_context(
        mock.patch.object(argparse.ArgumentParser, 'parse_args', autospec=True)
    )
    self.mock_path_exists_return = True

    def side_effect(path):
      # Mock the epath.Path(...).expanduser().resolve() chain.
      mock_instance = self.mock_path.return_value
      expanded_path = os.path.expanduser(path)
      absolute_path = os.path.abspath(expanded_path)

      mock_instance.expanduser.return_value.resolve.return_value = absolute_path
      mock_instance.exists.return_value = self.mock_path_exists_return
      return mock_instance

    self.mock_path.side_effect = side_effect

  @parameterized.named_parameters(
      ('gcs', 'gs://bucket/log', 'gs://bucket/log'),
      ('absolute', '/tmp/log', '/tmp/log'),
      ('home', '~/log', os.path.expanduser('~/log')),
      ('relative', 'relative/path', os.path.abspath('relative/path')),
  )
  def test_get_abs_path(self, logdir, expected_path):
    # Act
    actual = server.get_abs_path(logdir)
    # Assert
    self.assertEqual(actual, expected_path)

  @parameterized.named_parameters(
      (
          'no_logdir',
          {
              'logdir_opt': None,
              'logdir_pos': None,
              'port': 1234,
              'hide_capture_profile_button': False,
          },
          True,
          0,
          (None, 1234, server.FeatureConfig(hide_capture_profile_button=False)),
          True,
      ),
      (
          'with_logdir_opt',
          {
              'logdir_opt': '/tmp/log',
              'logdir_pos': None,
              'port': 5678,
              'hide_capture_profile_button': False,
          },
          True,
          0,
          (
              '/tmp/log',
              5678,
              server.FeatureConfig(hide_capture_profile_button=False),
          ),
          True,
      ),
      (
          'with_logdir_pos',
          {
              'logdir_opt': None,
              'logdir_pos': '/tmp/log',
              'port': 9012,
              'hide_capture_profile_button': False,
          },
          True,
          0,
          (
              '/tmp/log',
              9012,
              server.FeatureConfig(hide_capture_profile_button=False),
          ),
          True,
      ),
      (
          'logdir_not_exists',
          {
              'logdir_opt': '/tmp/log',
              'logdir_pos': None,
              'port': 3456,
              'hide_capture_profile_button': False,
          },
          False,
          1,
          None,
          False,
      ),
  )
  def test_main(
      self,
      mock_args_dict,
      path_exists,
      expected_result,
      launch_server_args,
      should_launch_server,
  ):
    # Arrange
    mock_args = argparse.Namespace(**mock_args_dict)
    self.mock_parse_args.return_value = mock_args
    self.mock_path_exists_return = path_exists

    # Act
    result = server.main()

    # Assert
    self.assertEqual(result, expected_result)
    if should_launch_server:
      self.mock_launch_server.assert_called_once_with(*launch_server_args)
    else:
      self.mock_launch_server.assert_not_called()


if __name__ == '__main__':
  googletest.main()
