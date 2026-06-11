"""Tests for the XProf server."""

import os
from unittest import mock

from absl.testing import parameterized
from etils import epath

from absl.testing import absltest
from xprof import server


class ServerTest(absltest.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_launch_server = self.enter_context(
        mock.patch.object(server, '_launch_server', autospec=True)
    )
    self.mock_path = self.enter_context(
        mock.patch.object(epath, 'Path', autospec=True)
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
              'logdir': None,
              'port': 1234,
              'grpc_port': 50051,
              'worker_service_address': '0.0.0.0:50051',
              'hide_capture_profile_button': False,
              'src_prefix': '',
              'max_concurrent_worker_requests': 1,
              'enable_tab_name_label': False,
          },
          server.ServerConfig(
              logdir=None,
              port=1234,
              grpc_port=50051,
              worker_service_address='0.0.0.0:50051',
              hide_capture_profile_button=False,
              src_prefix='',
              max_concurrent_worker_requests=1,
          ),
      ),
      (
          'with_logdir',
          {
              'logdir': '/tmp/log',
              'port': 5678,
              'grpc_port': 50051,
              'worker_service_address': '0.0.0.0:50051',
              'hide_capture_profile_button': False,
              'src_prefix': '',
              'max_concurrent_worker_requests': 1,
              'enable_tab_name_label': False,
          },
          server.ServerConfig(
              logdir='/tmp/log',
              port=5678,
              grpc_port=50051,
              worker_service_address='0.0.0.0:50051',
              hide_capture_profile_button=False,
              src_prefix='',
              max_concurrent_worker_requests=1,
          ),
      ),
      (
          'hide_capture_button_enabled',
          {
              'logdir': None,
              'port': 1234,
              'grpc_port': 50051,
              'worker_service_address': '0.0.0.0:50051',
              'hide_capture_profile_button': True,
              'src_prefix': '',
              'max_concurrent_worker_requests': 1,
              'enable_tab_name_label': False,
          },
          server.ServerConfig(
              logdir=None,
              port=1234,
              grpc_port=50051,
              worker_service_address='0.0.0.0:50051',
              hide_capture_profile_button=True,
              src_prefix='',
              max_concurrent_worker_requests=1,
          ),
      ),
  )
  def test_start_server(self, mock_args_dict, expected_config):
    # Arrange
    self.mock_path_exists_return = True

    # Act
    server.start_server(**mock_args_dict)

    # Assert
    self.mock_launch_server.assert_called_once_with(expected_config)

  @parameterized.named_parameters(
      (
          'port_collision',
          {
              'logdir': None,
              'port': 50051,
              'grpc_port': 50051,
              'worker_service_address': '0.0.0.0:50051',
              'hide_capture_profile_button': False,
              'src_prefix': '',
              'max_concurrent_worker_requests': 1,
              'enable_tab_name_label': False,
          },
          True,
          'The main server port',
      ),
      (
          'logdir_not_exists',
          {
              'logdir': '/tmp/log',
              'port': 3456,
              'grpc_port': 50051,
              'worker_service_address': '0.0.0.0:50051',
              'hide_capture_profile_button': False,
              'src_prefix': '',
              'max_concurrent_worker_requests': 1,
              'enable_tab_name_label': False,
          },
          False,
          'Log directory',
      ),
  )
  def test_start_server_errors(
      self, mock_args_dict, path_exists, expected_error_regex
  ):
    # Arrange
    self.mock_path_exists_return = path_exists

    # Act & Assert
    with self.assertRaisesRegex(ValueError, expected_error_regex):
      server.start_server(**mock_args_dict)
    self.mock_launch_server.assert_not_called()


if __name__ == '__main__':
  absltest.main()
