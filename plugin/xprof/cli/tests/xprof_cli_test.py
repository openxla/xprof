import inspect
import json
import pathlib
import sys
from typing import Any
import unittest
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from xprof.cli import xprof_cli

try:
  from google3.net.rpc.python import pywraprpc  # pylint: disable=g-import-not-at-top
except ImportError:
  pywraprpc = None


def _trace_only_tool(session_id: str):
  """A tool that is not marked as accepting a compiler dump directory."""
  return {'session_id': session_id}


class XProfCliTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.cli: Any = xprof_cli.XProfCli

  @mock.patch.object(xprof_cli.XProfCli, 'get_hlo_module_content')
  def test_get_hlo_module_content(self, mock_get_content):
    self.cli.get_hlo_module_content(
        'session_123', fmt='text', module_name=None, max_lines=2000
    )
    mock_get_content.assert_called_with(
        'session_123', fmt='text', module_name=None, max_lines=2000
    )

  @mock.patch.object(xprof_cli.XProfCli, 'get_hlo_neighborhood')
  def test_get_hlo_neighborhood(self, mock_get_neighborhood):
    self.cli.get_hlo_neighborhood('session_123', 'instr_name', 2, None)
    mock_get_neighborhood.assert_called_with(
        'session_123', 'instr_name', 2, None
    )

  @mock.patch.object(xprof_cli.XProfCli, 'get_hlo_neighborhood')
  def test_get_hlo_neighborhood_with_op_name(self, mock_get_neighborhood):
    self.cli.get_hlo_neighborhood('session_123', op_name='instr_name')
    mock_get_neighborhood.assert_called_with(
        'session_123', op_name='instr_name'
    )

  @mock.patch.object(xprof_cli.XProfCli, 'get_hlo_text')
  def test_get_hlo_text(self, mock_get_hlo_text):
    self.cli.get_hlo_text('session_123', 'path', 'module_name', 'op_name')
    mock_get_hlo_text.assert_called_with(
        'session_123', 'path', 'module_name', 'op_name'
    )

  @mock.patch.object(xprof_cli.XProfCli, 'list_hlo_modules')
  def test_list_hlo_modules(self, mock_list_modules):
    self.cli.list_hlo_modules('session_123')
    mock_list_modules.assert_called_with('session_123')

  @mock.patch.object(xprof_cli.XProfCli, 'get_hlo_op_profile')
  def test_get_hlo_op_profile(self, mock_get_op_profile):
    self.cli.get_hlo_op_profile('session_123', 15)
    mock_get_op_profile.assert_called_with('session_123', 15)

  @mock.patch.object(xprof_cli.XProfCli, 'list_xplane_events')
  def test_list_xplane_events(self, mock_list_events):
    self.cli.list_xplane_events('session_123', '.*', '.*', None, None, 100, 0)
    mock_list_events.assert_called_with(
        'session_123', '.*', '.*', None, None, 100, 0
    )

  @mock.patch.object(xprof_cli.XProfCli, 'aggregate_xplane_events')
  def test_aggregate_xplane_events(self, mock_agg_events):
    self.cli.aggregate_xplane_events('session_123', '.*', '.*')
    mock_agg_events.assert_called_with('session_123', '.*', '.*')

  @mock.patch.object(xprof_cli.XProfCli, 'get_xspace_proto')
  def test_get_xspace_proto(self, mock_get_xspace):
    self.cli.get_xspace_proto('session_123')
    mock_get_xspace.assert_called_with('session_123')

  @mock.patch.object(xprof_cli.XProfCli, 'get_overview')
  def test_get_overview(self, mock_get_overview):
    mock_get_overview.return_value = {'status': 'success'}
    result = self.cli.get_overview('session_123')
    mock_get_overview.assert_called_with('session_123')
    self.assertEqual(result, {'status': 'success'})

  @mock.patch.object(xprof_cli.XProfCli, 'get_profile_summary')
  def test_get_profile_summary(self, mock_get_summary):
    self.cli.get_profile_summary('session_123')
    mock_get_summary.assert_called_with('session_123')

  @mock.patch.object(xprof_cli.XProfCli, 'get_hosts')
  def test_get_hosts(self, mock_get_hosts):
    self.cli.get_hosts('session_123')
    mock_get_hosts.assert_called_with('session_123')

  @mock.patch.object(xprof_cli.XProfCli, 'get_roofline_model')
  def test_get_roofline_model(self, mock_get_roofline):
    self.cli.get_roofline_model('session_123')
    mock_get_roofline.assert_called_with('session_123')

  @mock.patch.object(xprof_cli.XProfCli, 'get_kpi_metrics')
  def test_get_kpi_metrics(self, mock_get_kpi):
    self.cli.get_kpi_metrics('session_123')
    mock_get_kpi.assert_called_with('session_123')

  @mock.patch.object(
      xprof_cli.XProfCli, 'get_kernel_utilization', autospec=True
  )
  def test_get_kernel_utilization(self, mock_get_kernel_util):
    self.cli.get_kernel_utilization('session_123', kernel_name='matmul')
    mock_get_kernel_util.assert_called_with('session_123', kernel_name='matmul')

  @mock.patch.object(xprof_cli.XProfCli, 'upload_trace', autospec=True)
  def test_upload_trace(self, mock_upload):
    self.cli.upload_trace('/path/to/trace.xplane.pb')
    mock_upload.assert_called_with('/path/to/trace.xplane.pb')

  @mock.patch.object(xprof_cli.fire, 'Fire')
  def test_main(self, mock_fire):
    xprof_cli.main([])
    mock_fire.assert_called_once_with(mock.ANY, command=None, name='xprof')
    self.assertIsInstance(mock_fire.call_args[0][0], xprof_cli.XProfCli)

  def test_all_tool_modules_registered_in_cli_main(self):
    """Ensures every *_tool.py file in cli/tools/ is registered in cli_main."""
    cli_module_dir = pathlib.Path(xprof_cli.__file__).parent
    tools_dir = cli_module_dir / 'tools'
    tool_files = [
        f
        for f in tools_dir.rglob('*_tool.py')
        if f.name != '__init__.py'
        and not f.name.startswith('test_')
        and 'google' not in f.parts
    ]

    cli_dict = xprof_cli.cli_main()

    for tool_file in tool_files:
      tool_name = tool_file.stem
      if tool_name.endswith('_tool'):
        tool_name = tool_name[:-5]
      self.assertIn(
          tool_name,
          cli_dict,
          msg=(
              f"Tool '{tool_name}' from '{tool_file.name}' is missing"
              ' registration in cli_main()!'
          ),
      )

  @mock.patch.object(xprof_cli, '_is_oss', return_value=True)
  def test_wrap_with_logdir_preserves_valid_signature_in_oss(self, _):
    """Ensures _wrap_with_logdir creates valid inspect signatures in OSS."""
    # Test on all real registered tools.
    for tool_name, tool_func in xprof_cli.cli_main().items():
      wrapped = xprof_cli._wrap_with_logdir(tool_func)
      self.assertTrue(callable(wrapped), msg=f'Failed wrapping {tool_name}')
      sig = inspect.signature(wrapped)
      self.assertIn('logdir', sig.parameters)

    # Test on a synthetic function with kwargs to prevent invalid parameter
    # ordering.
    def sample_func_with_kwargs(session_id: str, limit: int = 10, **kwargs):
      del session_id, limit, kwargs
      return 'ok'

    wrapped_sample = xprof_cli._wrap_with_logdir(sample_func_with_kwargs)
    sig_sample = inspect.signature(wrapped_sample)
    params = list(sig_sample.parameters.values())
    self.assertEqual(params[-1].kind, inspect.Parameter.VAR_KEYWORD)
    self.assertIn('logdir', sig_sample.parameters)
    self.assertIn('bypass_cache', sig_sample.parameters)
    self.assertEqual(
        sig_sample.parameters['logdir'].kind, inspect.Parameter.KEYWORD_ONLY
    )
    self.assertEqual(
        sig_sample.parameters['bypass_cache'].kind,
        inspect.Parameter.KEYWORD_ONLY,
    )

  @mock.patch.object(
      xprof_cli.fire,
      'Fire',
      side_effect=xprof_cli.fire.core.FireError('Invalid flag'),
  )
  @mock.patch.object(sys, 'stdout')
  @mock.patch.object(sys, 'stderr')
  def test_main_fire_usage_error_exit_2(self, mock_stderr, mock_stdout, _):
    with self.assertRaises(SystemExit) as cm:
      xprof_cli.main(['xprof', '--unknown'])
    self.assertEqual(cm.exception.code, 2)
    mock_stdout.write.assert_called()
    payload = json.loads(mock_stdout.write.call_args_list[0][0][0])
    self.assertEqual(payload['status'], 'ERROR')
    self.assertEqual(payload['reason'], 'USAGE_ERROR')
    self.assertNotIn('traceback', payload)
    mock_stderr.write.assert_called()
    self.assertIn('USAGE_ERROR', mock_stderr.write.call_args[0][0])

  @mock.patch.object(
      xprof_cli.fire, 'Fire', side_effect=FileNotFoundError('Trace not found')
  )
  @mock.patch.object(sys, 'stdout')
  @mock.patch.object(sys, 'stderr')
  def test_main_file_not_found_exit_3(self, mock_stderr, mock_stdout, _):
    with self.assertRaises(SystemExit) as cm:
      xprof_cli.main(['xprof', 'get_overview', 'non_existent_dir'])
    self.assertEqual(cm.exception.code, 3)
    mock_stdout.write.assert_called()
    payload = json.loads(mock_stdout.write.call_args_list[0][0][0])
    self.assertEqual(payload['status'], 'ERROR')
    self.assertEqual(payload['reason'], 'PATH_ERROR')
    self.assertNotIn('traceback', payload)
    mock_stderr.write.assert_called()
    self.assertIn('PATH_ERROR', mock_stderr.write.call_args[0][0])

  @mock.patch.object(
      xprof_cli.fire,
      'Fire',
      side_effect=PermissionError('Permission denied'),
  )
  @mock.patch.object(sys, 'stdout')
  @mock.patch.object(sys, 'stderr')
  def test_main_permission_error_exit_3(self, mock_stderr, mock_stdout, _):
    with self.assertRaises(SystemExit) as cm:
      xprof_cli.main(['xprof', 'upload_trace', 'trace.xplane.pb'])
    self.assertEqual(cm.exception.code, 3)
    mock_stdout.write.assert_called()
    payload = json.loads(mock_stdout.write.call_args_list[0][0][0])
    self.assertEqual(payload['status'], 'ERROR')
    self.assertEqual(payload['reason'], 'PATH_ERROR')
    self.assertNotIn('traceback', payload)
    mock_stderr.write.assert_called()
    self.assertIn('PATH_ERROR', mock_stderr.write.call_args[0][0])

  @mock.patch.object(
      xprof_cli.fire,
      'Fire',
      side_effect=OSError(28, 'No space left on device'),
  )
  @mock.patch.object(sys, 'stdout')
  @mock.patch.object(sys, 'stderr')
  def test_main_os_error_disk_full_exit_3(self, mock_stderr, mock_stdout, _):
    with self.assertRaises(SystemExit) as cm:
      xprof_cli.main(['xprof', 'upload_trace', 'trace.xplane.pb'])
    self.assertEqual(cm.exception.code, 3)
    mock_stdout.write.assert_called()
    payload = json.loads(mock_stdout.write.call_args_list[0][0][0])
    self.assertEqual(payload['status'], 'ERROR')
    self.assertEqual(payload['reason'], 'PATH_ERROR')
    self.assertNotIn('traceback', payload)
    mock_stderr.write.assert_called()
    self.assertIn('PATH_ERROR', mock_stderr.write.call_args[0][0])

  @mock.patch.object(
      xprof_cli.fire, 'Fire', side_effect=IsADirectoryError('Is a directory')
  )
  @mock.patch.object(sys, 'stdout')
  @mock.patch.object(sys, 'stderr')
  def test_main_is_a_directory_exit_3(self, mock_stderr, mock_stdout, _):
    with self.assertRaises(SystemExit) as cm:
      xprof_cli.main(['xprof', 'get_kernel_utilization', '/tmp/some_dir'])
    self.assertEqual(cm.exception.code, 3)
    mock_stdout.write.assert_called()
    payload = json.loads(mock_stdout.write.call_args_list[0][0][0])
    self.assertEqual(payload['status'], 'ERROR')
    self.assertEqual(payload['reason'], 'PATH_ERROR')
    mock_stderr.write.assert_called()
    self.assertIn('PATH_ERROR', mock_stderr.write.call_args[0][0])

  @mock.patch.object(
      xprof_cli.fire, 'Fire', side_effect=ValueError('Corrupt data')
  )
  @mock.patch.object(sys, 'stdout')
  @mock.patch.object(sys, 'stderr')
  def test_main_value_error_exit_4(self, mock_stderr, mock_stdout, _):
    with self.assertRaises(SystemExit) as cm:
      xprof_cli.main(['xprof', 'get_overview', 'corrupt_dir'])
    self.assertEqual(cm.exception.code, 4)
    mock_stdout.write.assert_called()
    payload = json.loads(mock_stdout.write.call_args_list[0][0][0])
    self.assertEqual(payload['status'], 'ERROR')
    self.assertEqual(payload['reason'], 'INVALID_VALUE')
    self.assertNotIn('traceback', payload)
    mock_stderr.write.assert_called()
    self.assertIn('INVALID_VALUE', mock_stderr.write.call_args[0][0])

  @mock.patch.object(
      xprof_cli.fire, 'Fire', side_effect=RuntimeError('Unexpected failure')
  )
  @mock.patch.object(sys, 'stdout')
  @mock.patch.object(sys, 'stderr')
  def test_main_internal_error_exit_1(self, mock_stderr, mock_stdout, _):
    with self.assertRaises(SystemExit) as cm:
      xprof_cli.main(['xprof', 'get_overview', 'broken_dir'])
    self.assertEqual(cm.exception.code, 1)
    mock_stdout.write.assert_called()
    payload = json.loads(mock_stdout.write.call_args_list[0][0][0])
    self.assertEqual(payload['status'], 'ERROR')
    self.assertEqual(payload['reason'], 'INTERNAL_ERROR')
    expected_bug_target = 'https://github.com/openxla/xprof/issues'
    self.assertIn(expected_bug_target, payload['error'])
    self.assertIn('traceback', payload)
    self.assertIn('RuntimeError: Unexpected failure', payload['traceback'])
    mock_stderr.write.assert_called()
    stderr_output = ''.join(
        call[0][0] for call in mock_stderr.write.call_args_list
    )
    self.assertIn('INTERNAL_ERROR', stderr_output)
    self.assertIn('RuntimeError: Unexpected failure', stderr_output)

  @unittest.skipIf(pywraprpc is None, 'pywraprpc not available in OSS')
  @mock.patch.object(sys, 'stdout')
  @mock.patch.object(sys, 'stderr')
  def test_main_rpc_error_exit_5(self, mock_stderr, mock_stdout):
    if pywraprpc is None:
      raise unittest.SkipTest('pywraprpc not available in OSS')
    rpc = pywraprpc.RPC()
    rpc.set_status(pywraprpc.RPC.UNREACHABLE)
    rpc_exc = pywraprpc.RPCException(rpc)

    with mock.patch.object(xprof_cli.fire, 'Fire', side_effect=rpc_exc):
      with self.assertRaises(SystemExit) as cm:
        xprof_cli.main(['xprof', 'get_overview', 'session_123'])
      self.assertEqual(cm.exception.code, 5)
      payload = json.loads(mock_stdout.write.call_args_list[0][0][0])
      self.assertEqual(payload['status'], 'ERROR')
      self.assertEqual(payload['reason'], 'RPC_ERROR')
      self.assertNotIn('traceback', payload)
      mock_stderr.write.assert_called()
      stderr_output = ''.join(
          call[0][0] for call in mock_stderr.write.call_args_list
      )
      self.assertIn('RPC_ERROR', stderr_output)

  @unittest.skipIf(pywraprpc is None, 'pywraprpc not available in OSS')
  @mock.patch.object(sys, 'stdout')
  @mock.patch.object(sys, 'stderr')
  def test_main_rpc_error_not_found_exit_3(self, mock_stderr, mock_stdout):
    if pywraprpc is None:
      raise unittest.SkipTest('pywraprpc not available in OSS')
    rpc = pywraprpc.RPC()
    pywraprpc.SetApplicationError(5, 'Session session_123 not found', rpc)
    rpc_exc = pywraprpc.RPCException(rpc)

    with mock.patch.object(xprof_cli.fire, 'Fire', side_effect=rpc_exc):
      with self.assertRaises(SystemExit) as cm:
        xprof_cli.main(['xprof', 'get_overview', 'session_123'])
      self.assertEqual(cm.exception.code, 3)
      payload = json.loads(mock_stdout.write.call_args_list[0][0][0])
      self.assertEqual(payload['status'], 'ERROR')
      self.assertEqual(payload['reason'], 'PATH_ERROR')
      self.assertNotIn('traceback', payload)
      mock_stderr.write.assert_called()
      stderr_output = ''.join(
          call[0][0] for call in mock_stderr.write.call_args_list
      )
      self.assertIn('PATH_ERROR', stderr_output)

  @unittest.skipIf(pywraprpc is None, 'pywraprpc not available in OSS')
  @mock.patch.object(sys, 'stdout')
  @mock.patch.object(sys, 'stderr')
  def test_main_rpc_error_chained_cause_exit_5(self, mock_stderr, mock_stdout):
    if pywraprpc is None:
      raise unittest.SkipTest('pywraprpc not available in OSS')
    rpc = pywraprpc.RPC()
    rpc.set_status(pywraprpc.RPC.UNREACHABLE)
    rpc_exc = pywraprpc.RPCException(rpc)
    chained_exc = RuntimeError('Tool failed')
    chained_exc.__cause__ = rpc_exc

    with mock.patch.object(xprof_cli.fire, 'Fire', side_effect=chained_exc):
      with self.assertRaises(SystemExit) as cm:
        xprof_cli.main(['xprof', 'get_overview', 'session_123'])
      self.assertEqual(cm.exception.code, 5)
      payload = json.loads(mock_stdout.write.call_args_list[0][0][0])
      self.assertEqual(payload['status'], 'ERROR')
      self.assertEqual(payload['reason'], 'RPC_ERROR')
      mock_stderr.write.assert_called()
      stderr_output = ''.join(
          call[0][0] for call in mock_stderr.write.call_args_list
      )
      self.assertIn('RPC_ERROR', stderr_output)

  @unittest.skipIf(pywraprpc is None, 'pywraprpc not available in OSS')
  @mock.patch.object(sys, 'stdout')
  @mock.patch.object(sys, 'stderr')
  def test_main_rpc_error_chained_context_not_found_exit_3(
      self, mock_stderr, mock_stdout
  ):
    if pywraprpc is None:
      raise unittest.SkipTest('pywraprpc not available in OSS')
    rpc = pywraprpc.RPC()
    pywraprpc.SetApplicationError(5, 'Session not found', rpc)
    rpc_exc = pywraprpc.RPCException(rpc)
    chained_exc = RuntimeError('Tool failed in context')
    chained_exc.__context__ = rpc_exc

    with mock.patch.object(xprof_cli.fire, 'Fire', side_effect=chained_exc):
      with self.assertRaises(SystemExit) as cm:
        xprof_cli.main(['xprof', 'get_overview', 'session_123'])
      self.assertEqual(cm.exception.code, 3)
      payload = json.loads(mock_stdout.write.call_args_list[0][0][0])
      self.assertEqual(payload['status'], 'ERROR')
      self.assertEqual(payload['reason'], 'PATH_ERROR')
      mock_stderr.write.assert_called()
      stderr_output = ''.join(
          call[0][0] for call in mock_stderr.write.call_args_list
      )
      self.assertIn('PATH_ERROR', stderr_output)

  def test_get_rpc_status_code_ignores_application_error_zero(self):
    fake_status = mock.Mock()
    fake_status.CanonicalCode.return_value = 5
    fake_rpc = mock.Mock(application_error=0, util_status=fake_status)
    self.assertEqual(xprof_cli._get_rpc_status_code(fake_rpc), 5)

  @mock.patch.object(
      xprof_cli.XProfCli,
      'upload_trace',
      side_effect=ValueError("Unsupported file format 'trace.txt'"),
  )
  @mock.patch.object(sys, 'stdout')
  @mock.patch.object(sys, 'stderr')
  def test_main_upload_trace_value_error_exit_4(
      self, mock_stderr, mock_stdout, _
  ):
    with self.assertRaises(SystemExit) as cm:
      xprof_cli.main(['xprof', 'upload_trace', 'trace.txt'])
    self.assertEqual(cm.exception.code, 4)
    payload = json.loads(mock_stdout.write.call_args_list[0][0][0])
    self.assertEqual(payload['status'], 'ERROR')
    self.assertEqual(payload['reason'], 'INVALID_VALUE')
    mock_stderr.write.assert_called()

  @mock.patch.object(
      xprof_cli.XProfCli,
      'upload_trace',
      side_effect=FileNotFoundError('Source trace file does not exist.'),
  )
  @mock.patch.object(sys, 'stdout')
  @mock.patch.object(sys, 'stderr')
  def test_main_upload_trace_file_not_found_exit_3(
      self, mock_stderr, mock_stdout, _
  ):
    with self.assertRaises(SystemExit) as cm:
      xprof_cli.main(['xprof', 'upload_trace', 'missing.xplane.pb'])
    self.assertEqual(cm.exception.code, 3)
    payload = json.loads(mock_stdout.write.call_args_list[0][0][0])
    self.assertEqual(payload['status'], 'ERROR')
    self.assertEqual(payload['reason'], 'PATH_ERROR')
    mock_stderr.write.assert_called()

  @mock.patch.object(
      xprof_cli.XProfCli,
      'upload_trace',
      side_effect=RuntimeError(
          'Failed to upload trace: Unexpected internal state'
      ),
  )
  @mock.patch.object(sys, 'stdout')
  @mock.patch.object(sys, 'stderr')
  def test_main_upload_trace_runtime_error_exit_1(
      self, mock_stderr, mock_stdout, _
  ):
    with self.assertRaises(SystemExit) as cm:
      xprof_cli.main(['xprof', 'upload_trace', 'trace.xplane.pb'])
    self.assertEqual(cm.exception.code, 1)
    payload = json.loads(mock_stdout.write.call_args_list[0][0][0])
    self.assertEqual(payload['status'], 'ERROR')
    self.assertEqual(payload['reason'], 'INTERNAL_ERROR')
    mock_stderr.write.assert_called()

  def test_empty_session_id_raises_value_error(self):
    """Ensures empty session_id is rejected with ValueError."""

    def dummy_tool(session_id: str):
      return session_id

    wrapped = xprof_cli._wrap_with_logdir(dummy_tool)
    with self.assertRaises(ValueError):
      wrapped('')
    with self.assertRaises(ValueError):
      wrapped(session_id='')

  def test_preprocess_argv_quotes_timestamp_tokens(self):
    """Ensures timestamp tokens with underscores are quoted for PEP 515."""
    raw_args = [
        'get_kernel_stats',
        '2026_08_24_06_33_12',
        '--logdir=/tmp/trace',
        '--session_id=2026_08_28_05_52_00',
    ]
    processed = xprof_cli._preprocess_argv(raw_args)
    self.assertEqual(
        processed,
        [
            'get_kernel_stats',
            '"2026_08_24_06_33_12"',
            '--logdir=/tmp/trace',
            '--session_id="2026_08_28_05_52_00"',
        ],
    )

  def test_preprocess_argv_preserves_numeric_and_standard_flags(self):
    """Ensures standard numeric flags without underscores are not quoted."""
    raw_args = ['list_xplane_events', 'sess1', '--limit=10', '-k=5']
    processed = xprof_cli._preprocess_argv(raw_args)
    self.assertEqual(
        processed, ['list_xplane_events', 'sess1', '--limit=10', '-k=5']
    )

  @parameterized.named_parameters(
      (
          'session_dir_equals',
          ['get_overview', '--session_dir=/tmp/trace'],
          ['get_overview', '/tmp/trace'],
      ),
      (
          'session_path_equals_with_trailing_flags',
          ['get_top_hlo_ops', '--session_path=/tmp/trace', '--limit=10'],
          ['get_top_hlo_ops', '/tmp/trace', '--limit=10'],
      ),
      (
          'source_space_separated',
          ['get_overview', '--source', '/tmp/trace'],
          ['get_overview', '/tmp/trace'],
      ),
      (
          'alias_before_subcommand',
          ['--session_dir=/tmp/trace', 'get_overview'],
          ['get_overview', '/tmp/trace'],
      ),
  )
  def test_d25_cli_argument_aliases(self, raw_argv, expected):
    """b/555254723: session-dir aliases normalize to the first positional."""
    self.assertEqual(xprof_cli._preprocess_argv(raw_argv), expected)

  @mock.patch.object(xprof_cli.fire, 'Fire', autospec=True, spec_set=True)
  def test_d25_cli_argument_aliases_reach_fire(self, mock_fire):
    """b/555254723: aliased invocations reach Fire without usage errors."""
    xprof_cli.main(['xprof', 'get_overview', '--session_dir=/tmp/trace'])
    mock_fire.assert_called_once_with(
        mock.ANY, command=['get_overview', '/tmp/trace'], name='xprof'
    )

  @mock.patch.object(xprof_cli.fire, 'Fire', autospec=True, spec_set=True)
  def test_d25_cli_argument_aliases_console_script_argv_none(self, mock_fire):
    """b/555254723: console_scripts entry point (argv=None) preprocesses sys.argv."""
    with mock.patch.object(
        sys, 'argv', ['xprof', 'get_overview', '--session_dir=/tmp/trace']
    ):
      xprof_cli.main(None)
    mock_fire.assert_called_once_with(
        mock.ANY, command=['get_overview', '/tmp/trace'], name='xprof'
    )

  def test_wrap_with_logdir_coerces_int_to_str(self):
    """Ensures int session_id and source parameters are coerced to string."""

    def dummy_tool(source: str, limit: int = 10):
      return {'source': source, 'limit': limit}

    wrapped = xprof_cli._wrap_with_logdir(dummy_tool)
    res = wrapped(20260824063312, limit=5)
    self.assertEqual(res['source'], '20260824063312')
    self.assertEqual(res['limit'], 5)

  def _make_compiler_dump_dir(self):
    """Creates a --xla_jf_dump_to style dir holding only text artifacts."""
    dump_dir = self.create_tempdir()
    dump_dir.create_file('mod-register-pressure.txt', content='Peak VREG: 64\n')
    dump_dir.create_file(
        'mod-per-bundle-utilization.txt', content='Bundle 0: MXU 100%\n'
    )
    return dump_dir.full_path

  def test_get_llo_dump_analysis_cli_accepts_compiler_dump_dir(self):
    """Compiler dump dirs hold no XPlane protos, so the CLI must not reject."""
    dump_dir = self._make_compiler_dump_dir()

    raw = self.cli.get_llo_dump_analysis(dump_dir, mode='register_pressure')

    self.assertEqual(json.loads(raw)['mode'], 'register_pressure')

  def test_get_llo_static_analysis_cli_accepts_compiler_dump_dir(self):
    """The `get_llo_static_analysis` alias behaves like `get_llo_dump_analysis`."""
    dump_dir = self._make_compiler_dump_dir()

    raw = self.cli.get_llo_static_analysis(dump_dir, mode='register_pressure')

    self.assertEqual(json.loads(raw)['mode'], 'register_pressure')

  def test_get_llo_dump_analysis_cli_rejects_empty_dir(self):
    """A directory with neither traces nor dump artifacts is still an error."""
    empty_dir = self.create_tempdir().full_path

    with self.assertRaisesRegex(FileNotFoundError, 'DATA_ABSENT'):
      self.cli.get_llo_dump_analysis(empty_dir, mode='register_pressure')

  def test_wrap_with_logdir_rejects_dump_dir_for_unmarked_tools(self):
    """Tools without the marker keep requiring an XPlane or XSpace file."""
    dump_dir = self._make_compiler_dump_dir()

    wrapped = xprof_cli._wrap_with_logdir(_trace_only_tool)

    with self.assertRaisesRegex(FileNotFoundError, 'DATA_ABSENT'):
      wrapped(dump_dir)


class MultiTraceSelectionTest(absltest.TestCase):
  """Tests capture reporting and host selection for multi-trace runs."""

  def setUp(self):
    super().setUp()
    self.run_dir = pathlib.Path(self.create_tempdir().full_path) / 'run'
    self.run_dir.mkdir()
    self.rank0 = self.run_dir / 'rank0_node0.xplane.pb'
    self.rank0.write_bytes(b'rank0')
    self.rank2 = self.run_dir / 'rank2_node0.xplane.pb'
    self.rank2.write_bytes(b'rank2')
    self.calls: list[dict[str, Any]] = []

  def _tool(self, name: str, native_host: bool = False, result: Any = None):
    """Builds a fake tool named `name` that records its arguments."""
    calls = self.calls
    payload = result if result is not None else json.dumps({'value': 1})

    if native_host:

      def tool(session_id: str, host: str = '') -> Any:
        calls.append({'session_id': session_id, 'host': host})
        return payload

    else:

      def tool(session_id: str) -> Any:
        calls.append({'session_id': session_id})
        return payload

    tool.__name__ = name
    return xprof_cli._wrap_with_logdir(tool)

  def test_signature_exposes_host(self):
    sig = inspect.signature(self._tool('get_overview'))
    self.assertEqual(
        sig.parameters['host'].kind, inspect.Parameter.KEYWORD_ONLY
    )

  def test_directory_attaches_capture_with_sum_warning(self):
    out = json.loads(self._tool('get_kernel_stats')(str(self.run_dir)))
    self.assertEqual(out['value'], 1)
    capture = out['capture']
    self.assertEqual(capture['files_used'], ['rank0_node0', 'rank2_node0'])
    self.assertTrue(capture['combined'])
    self.assertLen(capture['warnings'], 1)
    self.assertIn('summed', capture['warnings'][0])

  def test_file_attaches_capture_without_warning(self):
    out = json.loads(self._tool('get_overview')(str(self.rank0)))
    capture = out['capture']
    self.assertEqual(capture['files_used'], ['rank0_node0'])
    self.assertEqual(capture['files_available'], ['rank0_node0', 'rank2_node0'])
    self.assertEqual(capture['warnings'], [])

  def test_host_routes_directory_to_one_file(self):
    out = json.loads(
        self._tool('get_kernel_stats')(str(self.run_dir), host='rank0_node0')
    )
    self.assertEqual(self.calls, [{'session_id': str(self.rank0)}])
    self.assertEqual(out['capture']['files_used'], ['rank0_node0'])
    self.assertEqual(out['capture']['warnings'], [])

  def test_unknown_host_raises(self):
    with self.assertRaisesRegex(ValueError, r'(?s)nope.*rank0_node0'):
      self._tool('get_kernel_stats')(str(self.run_dir), host='nope')
    self.assertEqual(self.calls, [])

  def test_single_host_tool_rejects_directory(self):
    with self.assertRaisesRegex(
        ValueError,
        r'(?s)get_memory_profile needs exactly one trace.*rank0_node0'
        r'.*rank2_node0.*--host',
    ):
      self._tool('get_memory_profile')(str(self.run_dir))
    self.assertEqual(self.calls, [])

  def test_single_host_tool_accepts_host(self):
    self._tool('get_memory_profile')(str(self.run_dir), host='rank2_node0')
    self.assertEqual(self.calls, [{'session_id': str(self.rank2)}])

  def test_native_host_is_normalized(self):
    self._tool('get_llo_analysis', native_host=True)(
        str(self.run_dir), host='rank2_node0'
    )
    self.assertEqual(
        self.calls, [{'session_id': str(self.rank2), 'host': 'rank2_node0'}]
    )

  @mock.patch.object(sys, 'stderr')
  def test_non_json_result_writes_capture_to_stderr(self, mock_stderr):
    res = self._tool('get_hlo_text', result='HloModule m')(str(self.rank0))
    self.assertEqual(res, 'HloModule m')
    written = ''.join(call[0][0] for call in mock_stderr.write.call_args_list)
    self.assertIn('xprof-capture: ', written)
    self.assertIn('rank0_node0', written)

  @mock.patch.object(sys, 'stderr')
  def test_source_and_session_id_alias_signature(self, mock_stderr):
    calls = self.calls

    def get_kernel_stats(
        source: Any = None, session_id: str | None = None, *, limit: int = 10
    ) -> str:
      calls.append({'source': source, 'session_id': session_id})
      del limit
      return json.dumps([{'kernel': 'k'}])

    wrapped = xprof_cli._wrap_with_logdir(get_kernel_stats)
    res = wrapped(str(self.run_dir), host='rank0_node0')
    self.assertEqual(json.loads(res), [{'kernel': 'k'}])
    self.assertEqual(
        self.calls, [{'source': str(self.rank0), 'session_id': None}]
    )
    written = ''.join(call[0][0] for call in mock_stderr.write.call_args_list)
    self.assertIn('"files_used": ["rank0_node0"]', written)

  def test_host_rejected_for_remote_session(self):
    client = xprof_cli.xprof_client.get_client()
    with mock.patch.object(
        type(client), 'is_local_session', return_value=False
    ):
      with self.assertRaisesRegex(ValueError, 'only for local trace paths'):
        self._tool('get_overview')('remote-session-1', host='h1')
      res = self._tool('get_overview')('remote-session-1')
    self.assertNotIn('capture', json.loads(res))

  def _fire(self, wrapped: Any, *command: str) -> Any:
    """Runs `wrapped` through Fire, as the CLI does."""
    with mock.patch.object(sys, 'stdout'):
      return xprof_cli.fire.Fire(wrapped, command=list(command))

  def test_fire_native_host_bogus_is_rejected(self):
    wrapped = self._tool('get_llo_analysis', native_host=True)
    with self.assertRaisesRegex(ValueError, r'(?s)bogus.*rank0_node0'):
      self._fire(wrapped, str(self.run_dir), '--host=bogus')
    self.assertEqual(self.calls, [])

  def test_fire_native_host_conflicting_file_is_rejected(self):
    wrapped = self._tool('get_llo_analysis', native_host=True)
    with self.assertRaisesRegex(ValueError, 'Conflicting selection'):
      self._fire(wrapped, str(self.rank0), '--host=rank2_node0')
    self.assertEqual(self.calls, [])

  def test_fire_native_host_selects_file(self):
    wrapped = self._tool('get_llo_analysis', native_host=True)
    self._fire(wrapped, str(self.run_dir), '--host=rank2_node0')
    self.assertEqual(
        self.calls, [{'session_id': str(self.rank2), 'host': 'rank2_node0'}]
    )

  def test_fire_omitted_native_host_uses_tool_default(self):
    wrapped = self._tool('get_llo_analysis', native_host=True)
    self._fire(wrapped, str(self.rank0))
    self.assertEqual(self.calls, [{'session_id': str(self.rank0), 'host': ''}])

  def test_fire_empty_host_is_rejected(self):
    for native in (True, False):
      wrapped = self._tool('get_llo_analysis', native_host=native)
      with self.assertRaises(SystemExit):
        with mock.patch.object(sys, 'stderr'):
          self._fire(wrapped, str(self.rank0), '--host=')
    self.assertEqual(self.calls, [])

  @mock.patch.object(sys, 'stderr')
  def test_fire_hosts_list_selects_files(self, mock_stderr):
    del mock_stderr
    calls = self.calls

    def get_perf_counters(
        session_id: str, hosts: list[str] | None = None
    ) -> str:
      calls.append({'session_id': session_id, 'hosts': hosts})
      return json.dumps({'rows': []})

    wrapped = xprof_cli._wrap_with_logdir(get_perf_counters)
    res = self._fire(wrapped, str(self.run_dir), '--hosts=[rank2_node0]')
    self.assertEqual(json.loads(res)['capture']['files_used'], ['rank2_node0'])
    self.assertEqual(calls[0]['hosts'], ['rank2_node0'])

  def test_hosts_unparsed_list_string_is_accepted(self):
    # Fire leaves `--hosts=[a-b,c]` as a string when names contain hyphens.
    client = xprof_cli.xprof_client.get_client()
    paths = client.select_paths(
        str(self.run_dir), hosts='[rank2_node0, "rank0_node0"]'
    )
    self.assertEqual(
        [pathlib.Path(p).name for p in paths],
        [self.rank0.name, self.rank2.name],
    )

  def test_capture_is_first_key(self):
    out = json.loads(self._tool('get_overview')(str(self.rank0)))
    self.assertEqual(next(iter(out)), 'capture')

  def test_merged_warning_says_totals(self):
    out = json.loads(self._tool('get_overview')(str(self.run_dir)))
    self.assertIn('totals across hosts', out['capture']['warnings'][0])

  def test_list_events_warning_says_listed(self):
    out = json.loads(self._tool('list_xplane_events')(str(self.run_dir)))
    self.assertIn('Events listed from 2', out['capture']['warnings'][0])

  @mock.patch.object(sys, 'stderr')
  def test_hlo_text_has_no_combine_warning(self, mock_stderr):
    self._tool('get_hlo_text', result='HloModule m')(str(self.run_dir))
    written = ''.join(call[0][0] for call in mock_stderr.write.call_args_list)
    self.assertIn('"warnings": []', written)


if __name__ == '__main__':
  absltest.main()
