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

  def test_make_wsgi_app_routes(self):
    """Verifies WSGI app routing, prefix stripping, and 404 handling."""
    # Arrange
    mock_plugin = mock.MagicMock()
    mock_plugin.default_handler.return_value = [b'default']
    mock_app_handler = mock.MagicMock(return_value=[b'app_response'])
    mock_plugin.get_plugin_apps.return_value = {
        '/index.html': mock_app_handler,
        '/data': mock_app_handler,
    }
    app = server.make_wsgi_app(mock_plugin)
    start_response = mock.MagicMock()

    # Act: root paths route to default_handler
    response_root = app({'PATH_INFO': ''}, start_response)
    response_slash = app({'PATH_INFO': '/'}, start_response)

    # Assert
    self.assertEqual(response_root, [b'default'])
    self.assertEqual(response_slash, [b'default'])
    self.assertEqual(mock_plugin.default_handler.call_count, 2)

    # Act: path with prefix routes to mapped app handler
    mock_app_handler.reset_mock()
    response_data = app(
        {'PATH_INFO': '/data/plugin/profile/data'}, start_response
    )

    # Assert
    mock_app_handler.assert_called_once()
    self.assertEqual(response_data, [b'app_response'])

    # Act: unmapped route returns HTTP 404
    start_response.reset_mock()
    response_404 = app({'PATH_INFO': '/unmapped_route'}, start_response)

    # Assert
    start_response.assert_called_once_with(
        '404 Not Found', [('Content-Type', 'text/plain')]
    )
    self.assertEqual(response_404, [b'Not Found'])


if __name__ == '__main__':
  absltest.main()
