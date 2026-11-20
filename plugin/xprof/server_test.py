"""Tests for the XProf server."""

import contextlib
import gzip
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
      ('unset', {}, '127.0.0.1'),
      ('empty', {'XPROF_HOST': ''}, '127.0.0.1'),
      ('whitespace', {'XPROF_HOST': '  \t '}, '127.0.0.1'),
      ('custom_host', {'XPROF_HOST': '0.0.0.0'}, '0.0.0.0'),
      ('custom_host_with_whitespace', {'XPROF_HOST': '  [::1]  '}, '[::1]'),
  )
  def test_get_default_host(self, env_overrides, expected_host):
    with mock.patch.dict(os.environ, {}, clear=False):
      os.environ.pop('XPROF_HOST', None)
      os.environ.update(env_overrides)
      self.assertEqual(server.get_default_host(), expected_host)

  @parameterized.named_parameters(
      ('unset', {}, None),
      ('empty', {'XPROF_ALLOWED_HOSTS': ''}, None),
      ('whitespace', {'XPROF_ALLOWED_HOSTS': '  \t '}, None),
      (
          'custom_hosts',
          {'XPROF_ALLOWED_HOSTS': ' xprof.example.com,*.internal.example '},
          'xprof.example.com,*.internal.example',
      ),
      ('wildcard_all', {'XPROF_ALLOWED_HOSTS': '*'}, '*'),
  )
  def test_get_default_allowed_hosts(self, env_overrides, expected_hosts):
    with mock.patch.dict(os.environ, {}, clear=False):
      os.environ.pop('XPROF_ALLOWED_HOSTS', None)
      os.environ.update(env_overrides)
      self.assertEqual(server.get_default_allowed_hosts(), expected_hosts)

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
              host='127.0.0.1',
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
              host='127.0.0.1',
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
              host='127.0.0.1',
          ),
      ),
      (
          'custom_host',
          {
              'logdir': None,
              'port': 1234,
              'grpc_port': 50051,
              'worker_service_address': '0.0.0.0:50051',
              'hide_capture_profile_button': False,
              'src_prefix': '',
              'max_concurrent_worker_requests': 1,
              'enable_tab_name_label': False,
              'host': '0.0.0.0',
          },
          server.ServerConfig(
              logdir=None,
              port=1234,
              grpc_port=50051,
              worker_service_address='0.0.0.0:50051',
              hide_capture_profile_button=False,
              src_prefix='',
              max_concurrent_worker_requests=1,
              host='0.0.0.0',
          ),
      ),
      (
          'bracketed_ipv6_loopback',
          {
              'logdir': None,
              'port': 1234,
              'grpc_port': 50051,
              'worker_service_address': '0.0.0.0:50051',
              'hide_capture_profile_button': False,
              'src_prefix': '',
              'max_concurrent_worker_requests': 1,
              'enable_tab_name_label': False,
              'host': '[::1]',
          },
          server.ServerConfig(
              logdir=None,
              port=1234,
              grpc_port=50051,
              worker_service_address='0.0.0.0:50051',
              hide_capture_profile_button=False,
              src_prefix='',
              max_concurrent_worker_requests=1,
              host='::1',
          ),
      ),
      (
          'bracketed_ipv6_wildcard',
          {
              'logdir': None,
              'port': 1234,
              'grpc_port': 50051,
              'worker_service_address': '0.0.0.0:50051',
              'hide_capture_profile_button': False,
              'src_prefix': '',
              'max_concurrent_worker_requests': 1,
              'enable_tab_name_label': False,
              'host': '[::]',
          },
          server.ServerConfig(
              logdir=None,
              port=1234,
              grpc_port=50051,
              worker_service_address='0.0.0.0:50051',
              hide_capture_profile_button=False,
              src_prefix='',
              max_concurrent_worker_requests=1,
              host='::',
          ),
      ),
      (
          'allowed_hosts_comma_string_and_dedup',
          {
              'logdir': None,
              'port': 1234,
              'grpc_port': 50051,
              'worker_service_address': '0.0.0.0:50051',
              'hide_capture_profile_button': False,
              'src_prefix': '',
              'max_concurrent_worker_requests': 1,
              'enable_tab_name_label': False,
              'host': '0.0.0.0',
              'allowed_hosts': (
                  'My-Proxy.Example.com., *.Cluster.Local., [0:0:0:0:0:0:0:1],'
                  ' my-proxy.example.com'
              ),
          },
          server.ServerConfig(
              logdir=None,
              port=1234,
              grpc_port=50051,
              worker_service_address='0.0.0.0:50051',
              hide_capture_profile_button=False,
              src_prefix='',
              max_concurrent_worker_requests=1,
              host='0.0.0.0',
              allowed_hosts=(
                  'my-proxy.example.com',
                  '*.cluster.local',
                  '::1',
              ),
          ),
      ),
      (
          'allowed_hosts_sequence',
          {
              'logdir': None,
              'port': 1234,
              'grpc_port': 50051,
              'worker_service_address': '0.0.0.0:50051',
              'hide_capture_profile_button': False,
              'src_prefix': '',
              'max_concurrent_worker_requests': 1,
              'enable_tab_name_label': False,
              'allowed_hosts': ('a.example.com', '*.b.example.com', '*'),
          },
          server.ServerConfig(
              logdir=None,
              port=1234,
              grpc_port=50051,
              worker_service_address='0.0.0.0:50051',
              hide_capture_profile_button=False,
              src_prefix='',
              max_concurrent_worker_requests=1,
              host='127.0.0.1',
              allowed_hosts=('a.example.com', '*.b.example.com', '*'),
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
      (
          'empty_host',
          {
              'logdir': None,
              'port': 3456,
              'grpc_port': 50051,
              'host': '',
          },
          True,
          'Invalid host address',
      ),
      (
          'whitespace_host',
          {
              'logdir': None,
              'port': 3456,
              'grpc_port': 50051,
              'host': '   ',
          },
          True,
          'Invalid host address',
      ),
      (
          'bool_host',
          {
              'logdir': None,
              'port': 3456,
              'grpc_port': 50051,
              'host': True,
          },
          True,
          'non-empty string',
      ),
      (
          'malformed_ipv6_bracket',
          {
              'logdir': None,
              'port': 3456,
              'grpc_port': 50051,
              'host': '[::1',
          },
          True,
          'Invalid host address',
      ),
      (
          'host_with_port',
          {
              'logdir': None,
              'port': 3456,
              'grpc_port': 50051,
              'host': '127.0.0.1:8791',
          },
          True,
          'Invalid host address',
      ),
      (
          'host_with_slash',
          {
              'logdir': None,
              'port': 3456,
              'grpc_port': 50051,
              'host': '127.0.0.1/path',
          },
          True,
          'Invalid host address',
      ),
      (
          'host_with_space',
          {
              'logdir': None,
              'port': 3456,
              'grpc_port': 50051,
              'host': '127.0.0.1 0.0.0.0',
          },
          True,
          'Invalid host address',
      ),
      (
          'invalid_allowed_hosts',
          {
              'logdir': None,
              'port': 3456,
              'grpc_port': 50051,
              'allowed_hosts': 'foo*.example.com',
          },
          True,
          'Invalid allowed host entry',
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
      ('empty', '', 'non-empty host string'),
      ('whitespace', '   ', 'non-empty host string'),
      ('bool', True, '--allowed_hosts'),
      ('int', 123, '--allowed_hosts'),
      ('trailing_comma', 'example.com,', 'Invalid allowed host entry'),
      ('partial_wildcard', 'foo*.example.com', 'Invalid allowed host entry'),
      ('bare_wildcard_dot', '*.', 'Invalid allowed host entry'),
      ('wildcard_ip', '*.127.0.0.1', 'Wildcard allowed host cannot be an IP'),
      (
          'wildcard_leading_space',
          '*. example.com',
          'Invalid allowed host entry',
      ),
      ('port_included', 'example.com:8791', 'Invalid allowed host entry'),
      ('unspecified_ip', '0.0.0.0', 'Invalid allowed host entry'),
      ('ipv4_trailing_dot', '127.0.0.1.', 'Invalid allowed host entry'),
      ('leading_dot', '.example.com', 'Invalid allowed host entry'),
      ('double_dot', 'a..example.com', 'Invalid allowed host entry'),
      ('non_dns_chars', '<script>', 'Invalid allowed host entry'),
      ('semicolon_host', 'a;b.example.com', 'Invalid allowed host entry'),
      ('wildcard_non_dns_chars', '*.<script>', 'Invalid allowed host entry'),
  )
  def test_normalize_allowed_hosts_errors(
      self, allowed_hosts, expected_error_regex
  ):
    with self.assertRaisesRegex(ValueError, expected_error_regex):
      server._normalize_allowed_hosts(allowed_hosts)

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
    self.assertIn('xprof >= 2.24.0', message)
    self.assertIn(subcommand, message)
    self.mock_launch_server.assert_not_called()

  def test_main_launches_server(self):
    # Arrange
    self.mock_path_exists_return = True

    # Act
    with mock.patch.dict(os.environ, {'XPROF_HOST': ''}, clear=False):
      exit_code = server.main(['--logdir', '/tmp/log', '--port', '1234'])

    # Assert
    self.assertEqual(exit_code, 0)
    self.mock_launch_server.assert_called_once()
    self.assertEqual(self.mock_launch_server.call_args[0][0].host, '127.0.0.1')

  def test_main_launches_server_custom_host(self):
    self.mock_path_exists_return = True

    exit_code = server.main(
        ['--logdir', '/tmp/log', '--port', '1234', '--host', '0.0.0.0']
    )

    self.assertEqual(exit_code, 0)
    self.mock_launch_server.assert_called_once()
    self.assertEqual(self.mock_launch_server.call_args[0][0].host, '0.0.0.0')

  def test_main_respects_xprof_host_env_and_cli_override(self):
    self.mock_path_exists_return = True

    with mock.patch.dict(os.environ, {'XPROF_HOST': '0.0.0.0'}, clear=False):
      exit_code = server.main(['--logdir', '/tmp/log', '--port', '1234'])
    self.assertEqual(exit_code, 0)
    self.assertEqual(self.mock_launch_server.call_args[0][0].host, '0.0.0.0')

    self.mock_launch_server.reset_mock()
    with mock.patch.dict(os.environ, {'XPROF_HOST': '   '}, clear=False):
      exit_code = server.main(['--logdir', '/tmp/log', '--port', '1234'])
    self.assertEqual(exit_code, 0)
    self.assertEqual(self.mock_launch_server.call_args[0][0].host, '127.0.0.1')

    self.mock_launch_server.reset_mock()
    with mock.patch.dict(os.environ, {'XPROF_HOST': '0.0.0.0'}, clear=False):
      exit_code = server.main(
          ['--logdir', '/tmp/log', '--port', '1234', '--host', '127.0.0.1']
      )
    self.assertEqual(exit_code, 0)
    self.assertEqual(self.mock_launch_server.call_args[0][0].host, '127.0.0.1')

  def test_main_respects_xprof_allowed_hosts_env_and_cli_override(self):
    self.mock_path_exists_return = True

    with mock.patch.dict(
        os.environ,
        {'XPROF_ALLOWED_HOSTS': 'proxy.example.com,*.svc.cluster.local'},
        clear=False,
    ):
      exit_code = server.main(['--logdir', '/tmp/log', '--port', '1234'])
    self.assertEqual(exit_code, 0)
    self.assertEqual(
        self.mock_launch_server.call_args[0][0].allowed_hosts,
        ('proxy.example.com', '*.svc.cluster.local'),
    )

    self.mock_launch_server.reset_mock()
    with mock.patch.dict(
        os.environ, {'XPROF_ALLOWED_HOSTS': 'proxy.example.com'}, clear=False
    ):
      exit_code = server.main([
          '--logdir',
          '/tmp/log',
          '--port',
          '1234',
          '--allowed_hosts',
          'override.example.com',
      ])
    self.assertEqual(exit_code, 0)
    self.assertEqual(
        self.mock_launch_server.call_args[0][0].allowed_hosts,
        ('override.example.com',),
    )

  def test_cli_subcommand_required_message(self):
    message = server.cli_subcommand_required_message('get_overview')
    self.assertEqual(
        message,
        "Error: 'xprof get_overview' CLI tools require xprof >= 2.24.0."
        " Please run 'pip install -U xprof'. For TensorBoard server usage, run"
        " 'xprof --help'.",
    )


class LaunchServerTest(absltest.TestCase):

  def test_launch_server_starts_grpc_before_initializing_stubs(self):
    """Ensures worker gRPC server is listening before stubs pre-warm."""
    call_order = []
    with (
        mock.patch.object(
            server._pywrap_profiler_plugin,
            'start_grpc_server',
            autospec=True,
            side_effect=lambda port, max_concurrent_requests: call_order.append(
                ('start_grpc', (port, max_concurrent_requests))
            ),
        ),
        mock.patch.object(
            server._pywrap_profiler_plugin,
            'initialize_stubs',
            autospec=True,
            side_effect=lambda worker_service_addresses: call_order.append(
                ('init_stubs', worker_service_addresses)
            ),
        ),
        mock.patch.object(server, 'DataProvider', autospec=True),
        mock.patch.object(server, 'TBContext', autospec=True),
        mock.patch.object(server, 'ProfilePluginLoader', autospec=True),
        mock.patch.object(server, 'run_server', autospec=True),
    ):
      config = server.ServerConfig(
          logdir='/tmp/log',
          port=8791,
          grpc_port=50051,
          worker_service_address='0.0.0.0:50051',
          hide_capture_profile_button=False,
          src_prefix=None,
          max_concurrent_worker_requests=4,
      )
      server._launch_server(config)

    self.assertEqual(
        call_order,
        [
            ('start_grpc', (50051, 4)),
            ('init_stubs', '0.0.0.0:50051'),
        ],
    )

  def test_launch_server_binds_to_configured_host_defaulting_to_loopback(self):
    with (
        mock.patch.object(
            server._pywrap_profiler_plugin,
            'start_grpc_server',
            autospec=True,
        ),
        mock.patch.object(
            server._pywrap_profiler_plugin,
            'initialize_stubs',
            autospec=True,
        ),
        mock.patch.object(server, 'DataProvider', autospec=True),
        mock.patch.object(server, 'TBContext', autospec=True),
        mock.patch.object(server, 'ProfilePluginLoader', autospec=True),
        mock.patch.object(
            server, 'run_server', autospec=True
        ) as mock_run_server,
    ):
      default_config = server.ServerConfig(
          logdir='/tmp/log',
          port=8791,
          grpc_port=50051,
          worker_service_address='0.0.0.0:50051',
          hide_capture_profile_button=False,
          src_prefix=None,
          max_concurrent_worker_requests=1,
      )
      server._launch_server(default_config)
      mock_run_server.assert_called_once_with(
          mock.ANY, '127.0.0.1', 8791, allowed_hosts=()
      )

      mock_run_server.reset_mock()
      custom_config = server.ServerConfig(
          logdir='/tmp/log',
          port=8791,
          grpc_port=50051,
          worker_service_address='0.0.0.0:50051',
          hide_capture_profile_button=False,
          src_prefix=None,
          max_concurrent_worker_requests=1,
          host='localhost',
          allowed_hosts=('proxy.example.com',),
      )
      server._launch_server(custom_config)
      mock_run_server.assert_called_once_with(
          mock.ANY,
          'localhost',
          8791,
          allowed_hosts=('proxy.example.com',),
      )


def _respond_with(body):
  """Returns a WSGI app that answers every request with 200 and `body`."""

  def app(environ, start_response):
    del environ  # Unused.
    start_response('200 OK', [('Content-Type', 'text/plain')])
    return [body]

  return app


class MakeWsgiAppTest(parameterized.TestCase):

  def _request(
      self, path, extra_environ=None, host='127.0.0.1', allowed_hosts=None
  ):
    """Returns statuses, headers dict, and body served for `path`."""
    plugin = mock.Mock()
    plugin.get_plugin_apps.return_value = {'/runs': _respond_with(b'runs')}
    plugin.default_handler = _respond_with(b'index')
    statuses = []
    captured_headers = []
    environ = {'PATH_INFO': path}
    if extra_environ:
      environ.update(extra_environ)

    def start_response(status, headers):
      statuses.append(status)
      captured_headers.extend(headers)

    body = server.make_wsgi_app(
        plugin, host=host, allowed_hosts=allowed_hosts
    )(environ, start_response)
    return statuses, dict(captured_headers), b''.join(body)

  def _get(
      self, path, extra_environ=None, host='127.0.0.1', allowed_hosts=None
  ):
    """Returns the status and body the standalone app serves for `path`."""
    statuses, _, body = self._request(
        path,
        extra_environ=extra_environ,
        host=host,
        allowed_hosts=allowed_hosts,
    )
    return statuses, body

  @parameterized.parameters(
      ('/', b'index', True),
      ('/data/plugin/profile/', b'index', True),
      ('/data/plugin/profile/overview_page', b'index', True),
      ('/data/plugin/profile/kernel_viewer/session_1', b'index', True),
      ('/data/plugin/profile/kernel_viewer/run_v1.0', b'index', True),
      ('/data/plugin/profile/megascale_perfetto/run_v1.0', b'index', True),
      ('/data/plugin/profile/roofline_model/train/run_v1.0', b'index', True),
      ('/data/plugin/profile/roofline_model/.', b'index', True),
      ('/data/plugin/profile/stack_trace_page/run_v1.0', b'index', True),
      ('/stack_trace_page/run_v1.0', b'index', True),
      ('/data/plugin/profile/runs', b'runs', False),
      ('/data/plugin/profile/runs/', b'runs', False),
  )
  def test_serves_plugin_and_frontend_routes(
      self, path, expected_body, expect_xfo
  ):
    statuses, headers, body = self._request(path)
    self.assertEqual((statuses, body), (['200 OK'], expected_body))
    self.assertEqual(
        headers.get('Vary'),
        'Origin, Sec-Fetch-Site, Sec-Fetch-Mode, Sec-Fetch-Dest',
    )
    if expect_xfo:
      self.assertEqual(headers.get('X-Frame-Options'), 'SAMEORIGIN')
    else:
      self.assertNotIn('X-Frame-Options', headers)

  @parameterized.parameters(
      '/data/plugin/profile/missing.js',
      '/favicon.ico',
      '/overview_page',
      '/data/plugin/profilez/overview_page',
  )
  def test_unknown_path_is_not_found(self, path):
    statuses, headers, body = self._request(path)
    self.assertEqual((statuses, body), (['404 Not Found'], b'Not Found'))
    self.assertEqual(headers.get('X-Content-Type-Options'), 'nosniff')
    self.assertEqual(
        headers.get('Vary'),
        'Origin, Sec-Fetch-Site, Sec-Fetch-Mode, Sec-Fetch-Dest',
    )

  def test_head_error_responses_omit_body_and_include_security_headers(self):
    not_found_statuses, not_found_headers, not_found_body = self._request(
        '/favicon.ico', extra_environ={'REQUEST_METHOD': 'HEAD'}
    )
    self.assertEqual(not_found_statuses, ['404 Not Found'])
    self.assertEqual(not_found_body, b'')
    self.assertEqual(not_found_headers.get('Content-Length'), '9')
    self.assertEqual(not_found_headers.get('X-Content-Type-Options'), 'nosniff')

    host_statuses, host_headers, host_body = self._request(
        '/',
        extra_environ={
            'REQUEST_METHOD': 'HEAD',
            'HTTP_HOST': 'attacker.com:8791',
        },
    )
    self.assertEqual(host_statuses, ['403 Forbidden'])
    self.assertEqual(host_body, b'')
    self.assertEqual(host_headers.get('Content-Length'), '9')
    self.assertEqual(host_headers.get('X-Content-Type-Options'), 'nosniff')

    cors_statuses, cors_headers, cors_body = self._request(
        '/data/plugin/profile/runs',
        extra_environ={
            'REQUEST_METHOD': 'HEAD',
            'HTTP_SEC_FETCH_SITE': 'cross-site',
        },
    )
    self.assertEqual(cors_statuses, ['403 Forbidden'])
    self.assertEqual(cors_body, b'')
    self.assertEqual(cors_headers.get('Content-Length'), '9')
    self.assertEqual(cors_headers.get('X-Content-Type-Options'), 'nosniff')
    self.assertEqual(
        cors_headers.get('Vary'),
        'Origin, Sec-Fetch-Site, Sec-Fetch-Mode, Sec-Fetch-Dest',
    )

  @parameterized.named_parameters(
      ('sec_fetch_site_cross_site', {'HTTP_SEC_FETCH_SITE': 'cross-site'}),
      ('sec_fetch_site_same_site', {'HTTP_SEC_FETCH_SITE': 'same-site'}),
      (
          'cross_origin_header',
          {
              'HTTP_HOST': 'localhost:8791',
              'HTTP_ORIGIN': 'https://evil.example.com',
          },
      ),
      (
          'different_port_origin',
          {
              'HTTP_HOST': 'localhost:8791',
              'HTTP_ORIGIN': 'http://localhost:9999',
          },
      ),
      ('null_origin', {'HTTP_ORIGIN': 'null'}),
      (
          'cross_origin_referer',
          {
              'HTTP_HOST': 'localhost:8791',
              'HTTP_REFERER': 'https://evil.example.com/attack.html',
          },
      ),
      (
          'server_name_port_mismatch',
          {
              'SERVER_NAME': 'localhost',
              'SERVER_PORT': '8791',
              'HTTP_ORIGIN': 'http://evil.example.com:8791',
          },
      ),
      (
          'no_host_non_loopback_origin',
          {
              'HTTP_ORIGIN': 'http://evil.example.com',
          },
      ),
      (
          'malformed_ipv6_origin',
          {
              'HTTP_ORIGIN': 'http://[::1',
          },
      ),
      (
          'malformed_ipv6_referer',
          {
              'HTTP_REFERER': 'http://[not-an-ip]',
          },
      ),
  )
  def test_rejects_cross_origin_requests(self, extra_environ):
    plugin = mock.Mock()
    runs_handler = mock.Mock()
    plugin.get_plugin_apps.return_value = {'/runs': runs_handler}
    statuses = []
    captured_headers = []
    environ = {'PATH_INFO': '/data/plugin/profile/runs', **extra_environ}
    body = server.make_wsgi_app(plugin)(
        environ,
        lambda status, headers: (
            statuses.append(status),
            captured_headers.extend(headers),
        ),
    )
    headers_dict = dict(captured_headers)
    self.assertEqual(
        (statuses, b''.join(body)), (['403 Forbidden'], b'Forbidden')
    )
    self.assertEqual(headers_dict.get('X-Content-Type-Options'), 'nosniff')
    self.assertEqual(
        headers_dict.get('Vary'),
        'Origin, Sec-Fetch-Site, Sec-Fetch-Mode, Sec-Fetch-Dest',
    )
    runs_handler.assert_not_called()

  @parameterized.named_parameters(
      (
          'explicit_host_port',
          {
              'HTTP_HOST': 'localhost:8791',
              'HTTP_SEC_FETCH_SITE': 'same-origin',
              'HTTP_ORIGIN': 'http://localhost:8791',
              'HTTP_REFERER': 'http://localhost:8791/overview_page',
          },
      ),
      (
          'trailing_dot_localhost',
          {
              'HTTP_HOST': 'localhost.:8791',
              'HTTP_ORIGIN': 'http://localhost.:8791',
          },
      ),
      (
          'trailing_dot_host_with_normal_origin',
          {
              'HTTP_HOST': 'localhost.:8791',
              'HTTP_ORIGIN': 'http://localhost:8791',
          },
      ),
      (
          'loopback_127_0_0_1',
          {
              'HTTP_HOST': '127.0.0.1:8791',
              'HTTP_ORIGIN': 'http://127.0.0.1:8791',
          },
      ),
      (
          'loopback_127_0_0_2',
          {
              'HTTP_HOST': '127.0.0.2:8791',
              'HTTP_ORIGIN': 'http://127.0.0.2:8791',
          },
      ),
      (
          'ipv6_loopback_host',
          {
              'HTTP_HOST': '[::1]:8791',
              'HTTP_ORIGIN': 'http://[::1]:8791',
          },
      ),
      (
          'expanded_ipv6_loopback_host',
          {
              'HTTP_HOST': '[0:0:0:0:0:0:0:1]:8791',
              'HTTP_ORIGIN': 'http://[::1]:8791',
          },
      ),
      (
          'ipv4_mapped_ipv6_loopback_host',
          {
              'HTTP_HOST': '[::ffff:127.0.0.1]:8791',
              'HTTP_ORIGIN': 'http://[::ffff:127.0.0.1]:8791',
          },
      ),
      (
          'host_rewriting_reverse_proxy_with_sec_fetch_same_origin',
          {
              'HTTP_HOST': '127.0.0.1:8791',
              'HTTP_SEC_FETCH_SITE': 'same-origin',
              'HTTP_ORIGIN': 'https://proxy.example.com',
              'HTTP_REFERER': 'https://proxy.example.com/overview_page',
          },
      ),
      (
          'default_http_port_80',
          {
              'HTTP_HOST': 'localhost',
              'HTTP_ORIGIN': 'http://localhost:80',
          },
      ),
      (
          'default_https_port_443',
          {
              'HTTP_HOST': 'localhost:443',
              'HTTP_ORIGIN': 'https://localhost',
          },
      ),
      (
          'server_name_and_port_fallback',
          {
              'SERVER_NAME': 'localhost',
              'SERVER_PORT': '8791',
              'HTTP_ORIGIN': 'http://localhost:8791',
          },
      ),
      (
          'ipv6_server_name_and_port_fallback',
          {
              'SERVER_NAME': '::1',
              'SERVER_PORT': '8791',
              'HTTP_ORIGIN': 'http://[::1]:8791',
          },
      ),
      (
          'cheroot_banner_server_name_loopback_fallback',
          {
              'SERVER_NAME': 'Cheroot/11.1.2',
              'SERVER_PORT': '8791',
              'HTTP_ORIGIN': 'http://127.0.0.1:8791',
          },
      ),
      (
          'loopback_origin_without_host',
          {
              'HTTP_ORIGIN': 'http://127.0.0.1:8791',
          },
      ),
  )
  def test_allows_same_origin_requests(self, extra_environ):
    self.assertEqual(
        self._get('/data/plugin/profile/runs', extra_environ),
        (['200 OK'], b'runs'),
    )

  @parameterized.named_parameters(
      (
          'root_with_session_path',
          '/',
          'session_path=/tmp/profile',
          {
              'REQUEST_METHOD': 'GET',
              'HTTP_SEC_FETCH_SITE': 'cross-site',
              'HTTP_SEC_FETCH_MODE': 'navigate',
              'HTTP_SEC_FETCH_DEST': 'document',
              'HTTP_REFERER': 'https://example.org/',
          },
      ),
      (
          'head_root_navigation',
          '/',
          '',
          {
              'REQUEST_METHOD': 'HEAD',
              'HTTP_SEC_FETCH_SITE': 'cross-site',
              'HTTP_SEC_FETCH_MODE': 'navigate',
              'HTTP_SEC_FETCH_DEST': 'document',
          },
      ),
      (
          'local_route',
          '/local',
          '',
          {
              'REQUEST_METHOD': 'GET',
              'HTTP_SEC_FETCH_SITE': 'cross-site',
              'HTTP_SEC_FETCH_MODE': 'navigate',
              'HTTP_SEC_FETCH_DEST': 'document',
              'HTTP_REFERER': 'https://example.org/',
          },
      ),
      (
          'index_html_route',
          '/data/plugin/profile/index.html',
          '',
          {
              'REQUEST_METHOD': 'GET',
              'HTTP_SEC_FETCH_SITE': 'cross-site',
              'HTTP_SEC_FETCH_MODE': 'navigate',
              'HTTP_SEC_FETCH_DEST': 'document',
          },
      ),
      (
          'prefixed_frontend_route',
          '/data/plugin/profile/overview_page',
          'run=run1',
          {
              'REQUEST_METHOD': 'GET',
              'HTTP_SEC_FETCH_SITE': 'same-site',
              'HTTP_SEC_FETCH_MODE': 'navigate',
              'HTTP_SEC_FETCH_DEST': 'document',
              'HTTP_REFERER': 'https://sub.example.com/',
          },
      ),
      (
          'legacy_navigation_without_fetch_metadata',
          '/',
          'session_path=/tmp/profile',
          {
              'REQUEST_METHOD': 'GET',
              'HTTP_REFERER': 'https://example.org/',
          },
      ),
  )
  def test_allows_cross_site_top_level_ui_navigation_with_real_plugin(
      self, path, query_string, extra_environ
  ):
    plugin = server.ProfilePluginLoader().load(
        server.TBContext(None, mock.MagicMock(), server.TBContext.Flags(False))
    )
    with mock.patch.object(
        plugin, '_read_static_file_impl', return_value=b'<html>index</html>'
    ):
      app = server.make_wsgi_app(plugin)
      statuses = []
      captured_headers = []
      environ = {
          'PATH_INFO': path,
          'QUERY_STRING': query_string,
          **extra_environ,
      }
      body = b''.join(
          app(
              environ,
              lambda status, headers, exc_info=None: (
                  statuses.append(status),
                  captured_headers.extend(headers),
              ),
          )
      )
    headers_dict = dict(captured_headers)
    self.assertEqual(statuses, ['200 OK'])
    self.assertEqual(headers_dict.get('X-Frame-Options'), 'SAMEORIGIN')
    self.assertEqual(
        headers_dict.get('Vary'),
        'Origin, Sec-Fetch-Site, Sec-Fetch-Mode, Sec-Fetch-Dest',
    )
    if extra_environ.get('REQUEST_METHOD') == 'HEAD':
      self.assertEqual(body, b'')
    else:
      self.assertEqual(gzip.decompress(body), b'<html>index</html>')

  @parameterized.named_parameters(
      (
          'iframe_dest_on_root',
          '/',
          {
              'REQUEST_METHOD': 'GET',
              'HTTP_SEC_FETCH_SITE': 'cross-site',
              'HTTP_SEC_FETCH_MODE': 'navigate',
              'HTTP_SEC_FETCH_DEST': 'iframe',
              'HTTP_REFERER': 'https://example.org/',
          },
      ),
      (
          'partial_fetch_metadata_missing_dest',
          '/',
          {
              'REQUEST_METHOD': 'GET',
              'HTTP_SEC_FETCH_SITE': 'cross-site',
              'HTTP_SEC_FETCH_MODE': 'navigate',
              'HTTP_REFERER': 'https://example.org/',
          },
      ),
      (
          'script_dest_on_root',
          '/',
          {
              'REQUEST_METHOD': 'GET',
              'HTTP_SEC_FETCH_SITE': 'cross-site',
              'HTTP_SEC_FETCH_MODE': 'no-cors',
              'HTTP_SEC_FETCH_DEST': 'script',
          },
      ),
      (
          'post_on_root',
          '/',
          {
              'REQUEST_METHOD': 'POST',
              'HTTP_SEC_FETCH_SITE': 'cross-site',
              'HTTP_SEC_FETCH_MODE': 'navigate',
              'HTTP_SEC_FETCH_DEST': 'document',
              'HTTP_REFERER': 'https://example.org/',
          },
      ),
      (
          'navigate_on_api_route',
          '/data/plugin/profile/runs',
          {
              'REQUEST_METHOD': 'GET',
              'HTTP_SEC_FETCH_SITE': 'cross-site',
              'HTTP_SEC_FETCH_MODE': 'navigate',
              'HTTP_SEC_FETCH_DEST': 'document',
              'HTTP_REFERER': 'https://example.org/',
          },
      ),
  )
  def test_rejects_non_top_level_or_api_cross_site_navigation_with_real_plugin(
      self, path, extra_environ
  ):
    plugin = server.ProfilePluginLoader().load(
        server.TBContext(None, mock.MagicMock(), server.TBContext.Flags(False))
    )
    with mock.patch.object(
        plugin, '_read_static_file_impl', return_value=b'<html>index</html>'
    ):
      app = server.make_wsgi_app(plugin)
      statuses = []
      environ = {'PATH_INFO': path, **extra_environ}
      body = b''.join(
          app(
              environ,
              lambda status, headers, exc_info=None: statuses.append(status),
          )
      )
    self.assertEqual((statuses, body), (['403 Forbidden'], b'Forbidden'))

  @parameterized.named_parameters(
      ('external_host_on_ipv4_loopback', '127.0.0.1', 'attacker.com:8791'),
      ('external_host_on_127_0_0_2', '127.0.0.2', 'attacker.com:8791'),
      ('external_host_on_ipv6_loopback', '::1', 'attacker.com:8791'),
      (
          'external_host_on_expanded_ipv6_loopback',
          '[0:0:0:0:0:0:0:1]',
          'attacker.com:8791',
      ),
      (
          'external_host_on_ipv4_mapped_ipv6_loopback',
          '::ffff:127.0.0.1',
          'attacker.com:8791',
      ),
      ('external_host_on_localhost', 'localhost', 'attacker.com:8791'),
      ('external_host_on_localhost_dot', 'localhost.', 'attacker.com:8791'),
      ('empty_http_host', '127.0.0.1', ''),
      ('whitespace_http_host', '127.0.0.1', '   '),
      (
          'comma_joined_http_host_with_space',
          '127.0.0.1',
          '127.0.0.1, attacker.com',
      ),
      (
          'comma_joined_http_host_no_space',
          '127.0.0.1',
          '127.0.0.1,attacker.com',
      ),
      ('malformed_ipv6_http_host', '127.0.0.1', '[::1'),
      (
          'unbracketed_expanded_ipv6_with_port',
          '127.0.0.1',
          '0:0:0:0:0:0:0:1:8791',
      ),
      (
          'unbracketed_ipv4_mapped_ipv6_with_port',
          '127.0.0.1',
          '::ffff:127.0.0.1:8791',
      ),
      ('bracketed_ipv4_with_port', '127.0.0.1', '[127.0.0.1]:8791'),
      ('bracketed_ipv4_without_port', '127.0.0.1', '[127.0.0.1]'),
      ('userinfo_http_host', '127.0.0.1', 'evil.com@127.0.0.1:8791'),
      ('slash_in_http_host', '127.0.0.1', '127.0.0.1/evil'),
      ('query_in_http_host', '127.0.0.1', '127.0.0.1?evil=1'),
      ('fragment_in_http_host', '127.0.0.1', '127.0.0.1#evil'),
      ('non_numeric_port_in_http_host', '127.0.0.1', '127.0.0.1:notaport'),
      ('bool_true_http_host', '127.0.0.1', True),
      ('bool_false_http_host', '127.0.0.1', False),
      ('int_http_host', '127.0.0.1', 8791),
  )
  def test_dns_rebinding_and_malformed_host_rejected_before_ui_exemption(
      self, bind_host, http_host
  ):
    environ = {
        'PATH_INFO': '/',
        'REQUEST_METHOD': 'GET',
        'HTTP_HOST': http_host,
        'HTTP_SEC_FETCH_SITE': 'same-origin',
        'HTTP_SEC_FETCH_MODE': 'navigate',
        'HTTP_SEC_FETCH_DEST': 'document',
    }
    self.assertEqual(
        self._get('/', extra_environ=environ, host=bind_host),
        (['403 Forbidden'], b'Forbidden'),
    )

  @parameterized.named_parameters(
      (
          'attacker_domain_on_ipv4_wildcard',
          '0.0.0.0',
          'evil.rebind.attacker.example:8791',
      ),
      (
          'attacker_domain_on_ipv6_wildcard',
          '::',
          'evil.rebind.attacker.example:8791',
      ),
      (
          'attacker_domain_on_explicit_lan_ip',
          '10.0.0.5',
          'attacker.com:8791',
      ),
      (
          'attacker_domain_on_explicit_hostname',
          'mybox.internal',
          'attacker.com:8791',
      ),
      ('localhost_prefix_spoof_on_wildcard', '0.0.0.0', 'localhost.evil:8791'),
      ('ip_prefix_spoof_on_wildcard', '0.0.0.0', '127.0.0.1.nip.io:8791'),
      ('evil_localhost_subdomain', '0.0.0.0', 'evil.localhost:8791'),
      ('unspecified_ipv4_on_wildcard', '0.0.0.0', '0.0.0.0:8791'),
      ('unspecified_ipv6_on_wildcard', '::', '[::]:8791'),
      (
          'ipv4_mapped_unspecified_on_wildcard',
          '::',
          '[::ffff:0.0.0.0]:8791',
      ),
      ('octal_ip_on_wildcard', '0.0.0.0', '0177.0.0.1:8791'),
      ('hex_ip_on_wildcard', '0.0.0.0', '0x7f000001:8791'),
      ('decimal_ip_on_wildcard', '0.0.0.0', '2130706433:8791'),
      ('short_ip_on_wildcard', '0.0.0.0', '127.1:8791'),
      ('non_loopback_ip_on_loopback_bind', '127.0.0.1', '10.0.0.5:8791'),
  )
  def test_dns_rebinding_host_rejected_even_on_non_loopback_bind(
      self, bind_host, http_host
  ):
    environ = {
        'HTTP_HOST': http_host,
        'HTTP_ORIGIN': f'http://{http_host}',
        'HTTP_REFERER': f'http://{http_host}/',
        'HTTP_SEC_FETCH_SITE': 'same-origin',
    }
    self.assertEqual(
        self._get(
            '/data/plugin/profile/runs', extra_environ=environ, host=bind_host
        ),
        (['403 Forbidden'], b'Forbidden'),
    )

  @parameterized.named_parameters(
      ('localhost_on_wildcard', '0.0.0.0', 'localhost:8791', None),
      ('ipv4_loopback_on_wildcard', '0.0.0.0', '127.0.0.1:8791', None),
      ('ipv6_loopback_on_wildcard', '::', '[::1]:8791', None),
      ('ipv4_literal_on_wildcard', '0.0.0.0', '10.0.0.5:8791', None),
      ('ipv6_literal_on_wildcard', '::', '[2001:db8::1]:8791', None),
      (
          'bound_hostname_on_explicit_bind',
          'mybox.internal',
          'mybox.internal:8791',
          None,
      ),
      (
          'exact_allowed_host_on_loopback_bind',
          '127.0.0.1',
          'my-proxy.example.com:8791',
          'my-proxy.example.com',
      ),
      (
          'exact_allowed_host_case_and_trailing_dot',
          '0.0.0.0',
          'Xprof-Agg-Service.Default.svc.cluster.local.:80',
          'xprof-agg-service.default.svc.cluster.local',
      ),
      (
          'wildcard_subdomain_allowed_host',
          '0.0.0.0',
          'xprof.proxy.example.com:443',
          '*.proxy.example.com',
      ),
      (
          'nested_wildcard_subdomain_allowed_host',
          '0.0.0.0',
          'a.b.proxy.example.com:443',
          '*.proxy.example.com',
      ),
      (
          'star_allowed_host_permits_valid_domain',
          '0.0.0.0',
          'custom.example.org:8791',
          '*',
      ),
  )
  def test_allows_permitted_host_headers_for_bind_and_allowlist(
      self, bind_host, http_host, allowed_hosts
  ):
    environ = {
        'HTTP_HOST': http_host,
        'HTTP_ORIGIN': f'http://{http_host}',
        'HTTP_SEC_FETCH_SITE': 'same-origin',
    }
    self.assertEqual(
        self._get(
            '/data/plugin/profile/runs',
            extra_environ=environ,
            host=bind_host,
            allowed_hosts=allowed_hosts,
        ),
        (['200 OK'], b'runs'),
    )

  @parameterized.named_parameters(
      (
          'wildcard_does_not_match_apex_domain',
          '0.0.0.0',
          'proxy.example.com:443',
          '*.proxy.example.com',
      ),
      (
          'wildcard_enforces_dot_boundary',
          '0.0.0.0',
          'evil-proxy.example.com:443',
          '*.proxy.example.com',
      ),
      (
          'wildcard_does_not_match_suffix_spoof',
          '0.0.0.0',
          'a.proxy.example.com.evil.com:443',
          '*.proxy.example.com',
      ),
      (
          'wildcard_rejects_leading_dot',
          '0.0.0.0',
          '.proxy.example.com:443',
          '*.proxy.example.com',
      ),
      (
          'wildcard_rejects_literal_wildcard',
          '0.0.0.0',
          '*.proxy.example.com:443',
          '*.proxy.example.com',
      ),
      (
          'wildcard_rejects_double_dot_label',
          '0.0.0.0',
          'a..proxy.example.com:443',
          '*.proxy.example.com',
      ),
      (
          'wildcard_rejects_multiple_trailing_dots',
          '0.0.0.0',
          'a.proxy.example.com..:443',
          '*.proxy.example.com',
      ),
      (
          'wildcard_rejects_bare_star',
          '0.0.0.0',
          '*',
          '*.proxy.example.com',
      ),
      (
          'unlisted_host_rejected_with_allowlist',
          '0.0.0.0',
          'other.example.com:8791',
          'allowed.example.com',
      ),
      (
          'star_still_rejects_malformed_host',
          '0.0.0.0',
          'evil.com@127.0.0.1:8791',
          '*',
      ),
      (
          'star_still_rejects_empty_host',
          '0.0.0.0',
          '',
          '*',
      ),
      (
          'star_still_rejects_leading_dot',
          '0.0.0.0',
          '.proxy.example.com:443',
          '*',
      ),
      (
          'star_still_rejects_literal_wildcard',
          '0.0.0.0',
          '*.proxy.example.com:443',
          '*',
      ),
      (
          'star_still_rejects_double_dot_label',
          '0.0.0.0',
          'a..proxy.example.com:443',
          '*',
      ),
      (
          'star_still_rejects_multiple_trailing_dots',
          '0.0.0.0',
          'proxy.example.com..:443',
          '*',
      ),
      (
          'star_still_rejects_bare_star',
          '0.0.0.0',
          '*',
          '*',
      ),
      (
          'star_still_rejects_wildcard_dot',
          '0.0.0.0',
          '*.',
          '*',
      ),
      (
          'star_still_rejects_unspecified_ipv4',
          '0.0.0.0',
          '0.0.0.0:8791',
          '*',
      ),
      (
          'star_still_rejects_unspecified_ipv6',
          '::',
          '[::]:8791',
          '*',
      ),
      (
          'star_still_rejects_bare_ipv6',
          '0.0.0.0',
          '::1',
          '*',
      ),
      (
          'star_still_rejects_bare_ipv6_with_port_ambiguity',
          '0.0.0.0',
          '::1:8791',
          '*',
      ),
      (
          'star_still_rejects_ipv6_zone_id',
          '0.0.0.0',
          '[fe80::1%25eth0]:8791',
          '*',
      ),
      (
          'star_still_rejects_ipv4_trailing_dot',
          '0.0.0.0',
          '127.0.0.1.:8791',
          '*',
      ),
      (
          'star_still_rejects_port_zero',
          '0.0.0.0',
          'localhost:0',
          '*',
      ),
      (
          'star_still_rejects_non_dns_chars',
          '0.0.0.0',
          '<script>:8791',
          '*',
      ),
      (
          'star_still_rejects_semicolon',
          '0.0.0.0',
          'a;b.example.com:8791',
          '*',
      ),
  )
  def test_rejects_disallowed_or_malformed_host_with_allowlist(
      self, bind_host, http_host, allowed_hosts
  ):
    environ = {
        'HTTP_HOST': http_host,
        'HTTP_SEC_FETCH_SITE': 'same-origin',
    }
    self.assertEqual(
        self._get(
            '/data/plugin/profile/runs',
            extra_environ=environ,
            host=bind_host,
            allowed_hosts=allowed_hosts,
        ),
        (['403 Forbidden'], b'Forbidden'),
    )

  def test_rejects_http_1_1_missing_host_header(self):
    self.assertEqual(
        self._get(
            '/data/plugin/profile/runs',
            extra_environ={'SERVER_PROTOCOL': 'HTTP/1.1'},
        ),
        (['403 Forbidden'], b'Forbidden'),
    )


if __name__ == '__main__':
  absltest.main()
