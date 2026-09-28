"""Tests for the XProf server."""

import contextlib
import io
import os
from unittest import mock

from absl.testing import parameterized
from etils import epath

from absl.testing import absltest
from xprof import server


class ServerTest(parameterized.TestCase):

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

  @parameterized.named_parameters(
      ('get_overview', 'get_overview'),
      ('get_roofline_model', 'get_roofline_model'),
      ('check_host_boundness', 'check_host_boundness'),
      ('get_top_hlo_ops', 'get_top_hlo_ops'),
  )
  def test_main_intercepts_cli_subcommand(self, subcommand):
    # Act
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr):
      exit_code = server.main([subcommand, '/path/to/trace'])

    # Assert: an actionable upgrade message is printed and the server is not
    # launched.
    self.assertEqual(exit_code, 2)
    message = stderr.getvalue()
    self.assertIn('xprof-nightly >= 2.24.0', message)
    self.assertIn(subcommand, message)
    self.mock_launch_server.assert_not_called()

  def test_main_launches_server(self):
    # Arrange
    self.mock_path_exists_return = True

    # Act
    exit_code = server.main(['--logdir', '/tmp/log', '--port', '1234'])

    # Assert
    self.assertEqual(exit_code, 0)
    self.mock_launch_server.assert_called_once()

  def test_cli_subcommand_required_message(self):
    message = server.cli_subcommand_required_message('get_overview')
    self.assertEqual(
        message,
        "Error: 'xprof get_overview' CLI tools require xprof-nightly >= 2.24.0."
        " Please run 'pip install -U xprof-nightly' or upgrade to xprof >="
        " 2.24.0. For TensorBoard server usage, run 'xprof --help'.",
    )


def _respond_with(body):
  """Returns a WSGI app that answers every request with 200 and `body`."""

  def app(environ, start_response):
    del environ  # Unused.
    start_response('200 OK', [('Content-Type', 'text/plain')])
    return [body]

  return app


class MakeWsgiAppTest(parameterized.TestCase):

  def _get(self, path):
    """Returns the status and body the standalone app serves for `path`."""
    plugin = mock.Mock()
    plugin.get_plugin_apps.return_value = {'/runs': _respond_with(b'runs')}
    plugin.default_handler = _respond_with(b'index')
    statuses = []
    body = server.make_wsgi_app(plugin)(
        {'PATH_INFO': path}, lambda status, headers: statuses.append(status)
    )
    return statuses, b''.join(body)

  @parameterized.parameters(
      ('/', b'index'),
      ('/data/plugin/profile/', b'index'),
      ('/data/plugin/profile/overview_page', b'index'),
      ('/data/plugin/profile/kernel_viewer/session_1', b'index'),
      ('/data/plugin/profile/runs', b'runs'),
      ('/data/plugin/profile/runs/', b'runs'),
  )
  def test_serves_plugin_and_frontend_routes(self, path, expected_body):
    self.assertEqual(self._get(path), (['200 OK'], expected_body))

  @parameterized.parameters('/data/plugin/profile/missing.js', '/favicon.ico')
  def test_unknown_file_is_not_found(self, path):
    self.assertEqual(self._get(path), (['404 Not Found'], b'Not Found'))


if __name__ == '__main__':
  absltest.main()
