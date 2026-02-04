"""Tests for the XProf server."""

import argparse
import dataclasses
import os
import types
from unittest import mock

from absl.testing import parameterized
from etils import epath

from absl.testing import absltest
from xprof import server


DEFAULT_MOCK_ARGS = types.MappingProxyType({
    'logdir_opt': None,
    'logdir_pos': None,
    'port': 1234,
    'grpc_port': 50051,
    'worker_service_address': '0.0.0.0:50051',
    'hide_capture_profile_button': False,
    'src_prefix': '',
    'num_cqs': 1,
    'min_pollers': 1,
    'max_pollers': 1,
})

DEFAULT_SERVER_CONFIG = server.ServerConfig(
    logdir=None,
    port=1234,
    grpc_port=50051,
    worker_service_address='0.0.0.0:50051',
    hide_capture_profile_button=False,
    src_prefix='',
    num_cqs=1,
    min_pollers=1,
    max_pollers=1,
)


class ServerTest(absltest.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_launch_server = self.enter_context(
        mock.patch.object(server, '_launch_server', autospec=True)
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
          {'logdir_opt': None, 'logdir_pos': None},
          True,
          0,
          DEFAULT_SERVER_CONFIG,
          True,
      ),
      (
          'with_logdir_opt',
          {'logdir_opt': '/tmp/log', 'logdir_pos': None, 'port': 5678},
          True,
          0,
          dataclasses.replace(
              DEFAULT_SERVER_CONFIG, logdir='/tmp/log', port=5678
          ),
          True,
      ),
      (
          'with_logdir_pos',
          {'logdir_opt': None, 'logdir_pos': '/tmp/log', 'port': 9012},
          True,
          0,
          dataclasses.replace(
              DEFAULT_SERVER_CONFIG, logdir='/tmp/log', port=9012
          ),
          True,
      ),
      (
          'logdir_not_exists',
          {'logdir_opt': '/tmp/log', 'logdir_pos': None, 'port': 3456},
          False,
          1,
          None,
          False,
      ),
      (
          'hide_capture_button_enabled',
          {'hide_capture_profile_button': True},
          True,
          0,
          dataclasses.replace(
              DEFAULT_SERVER_CONFIG, hide_capture_profile_button=True
          ),
          True,
      ),
      (
          'all_features_enabled',
          {
              'logdir_opt': '/tmp/log',
              'hide_capture_profile_button': True,
          },
          True,
          0,
          dataclasses.replace(
              DEFAULT_SERVER_CONFIG,
              logdir='/tmp/log',
              hide_capture_profile_button=True,
          ),
          True,
      ),
      (
          'grpc_tuning',
          {
              'logdir_opt': None,
              'logdir_pos': None,
              'port': 6006,
              'grpc_port': 6007,
              'worker_service_address': '0.0.0.0:6007',
              'hide_capture_profile_button': False,
              'src_prefix': '',
              'num_cqs': 2,
              'min_pollers': 2,
              'max_pollers': 4,
          },
          True,
          0,
          dataclasses.replace(
              DEFAULT_SERVER_CONFIG,
              port=6006,
              grpc_port=6007,
              worker_service_address='0.0.0.0:6007',
              num_cqs=2,
              min_pollers=2,
              max_pollers=4,
          ),
          True,
      ),
  )
  def test_main(
      self,
      mock_args_dict,
      path_exists,
      expected_result,
      expected_config,
      should_launch_server,
  ):
    # Arrange
    args = {**DEFAULT_MOCK_ARGS, **mock_args_dict}
    mock_args = argparse.Namespace(**args)
    self.mock_parse_args.return_value = mock_args
    self.mock_path_exists_return = path_exists

    # Act
    result = server.main()

    # Assert
    self.assertEqual(result, expected_result)
    if should_launch_server:
      self.mock_launch_server.assert_called_once_with(expected_config)
    else:
      self.mock_launch_server.assert_not_called()


if __name__ == '__main__':
  absltest.main()
