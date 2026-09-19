import inspect
import json
import pathlib
import sys
from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from xprof.cli import xprof_cli


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

    with mock.patch.object(xprof_cli, '_is_oss', return_value=True):
      oss_cli_dict = xprof_cli.cli_main()

    multi_tool_modules = {
        'events_db_tool.py': ('create_events_db', 'query_events_db'),
    }
    for tool_file in tool_files:
      if tool_file.name in multi_tool_modules:
        expected_names = multi_tool_modules[tool_file.name]
      else:
        tool_name = tool_file.stem
        if tool_name.endswith('_tool'):
          tool_name = tool_name[:-5]
        expected_names = (tool_name,)
      for expected_name in expected_names:
        self.assertIn(
            expected_name,
            oss_cli_dict,
            msg=(
                f"Tool '{expected_name}' from '{tool_file.name}' is missing"
                ' registration in cli_main()!'
            ),
        )

    # In 1P mode (_is_oss()=False), OSS-only Parquet tools are not exposed.
    with mock.patch.object(xprof_cli, '_is_oss', return_value=False):
      internal_cli_dict = xprof_cli.cli_main()
    self.assertNotIn('create_events_db', internal_cli_dict)
    self.assertNotIn('query_events_db', internal_cli_dict)

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
  @mock.patch('sys.stdout')
  @mock.patch('sys.stderr')
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
  @mock.patch('sys.stdout')
  @mock.patch('sys.stderr')
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
  @mock.patch('sys.stdout')
  @mock.patch('sys.stderr')
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
  @mock.patch('sys.stdout')
  @mock.patch('sys.stderr')
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
  @mock.patch('sys.stdout')
  @mock.patch('sys.stderr')
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
  @mock.patch('sys.stdout')
  @mock.patch('sys.stderr')
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
  @mock.patch('sys.stdout')
  @mock.patch('sys.stderr')
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
    stderr_output = ''.join(c[0][0] for c in mock_stderr.write.call_args_list)
    self.assertIn('INTERNAL_ERROR', stderr_output)
    self.assertIn('RuntimeError: Unexpected failure', stderr_output)

  @mock.patch.object(
      xprof_cli.XProfCli,
      'upload_trace',
      side_effect=ValueError("Unsupported file format 'trace.txt'"),
  )
  @mock.patch('sys.stdout')
  @mock.patch('sys.stderr')
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
  @mock.patch('sys.stdout')
  @mock.patch('sys.stderr')
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
  @mock.patch('sys.stdout')
  @mock.patch('sys.stderr')
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

  @mock.patch.object(xprof_cli, '_is_oss', return_value=True)
  @mock.patch.object(
      xprof_cli.events_db_tool,
      'create_events_db',
      return_value='{"path": "/tmp/events.parquet", "skipped": false}',
  )
  def test_oss_create_events_db_cli_dispatch(
      self, mock_create_events_db, _
  ):
    trace_file = pathlib.Path(self.create_tempdir().full_path) / 't.xplane.pb'
    trace_file.write_bytes(b'test')
    xprof_cli.main([
        'xprof',
        'create_events_db',
        f'--session_dir={trace_file}',
        '--bypass_cache',
    ])
    mock_create_events_db.assert_called_once_with(
        str(trace_file), bypass_cache=True
    )

  @mock.patch.object(xprof_cli, '_is_oss', return_value=True)
  @mock.patch.object(
      xprof_cli.events_db_tool,
      'query_events_db',
      return_value='{"path": "/tmp/res.json", "skipped": false}',
  )
  def test_oss_query_events_db_single_positional_sql_skips_path_check(
      self, mock_query_events_db, _
  ):
    """Ensures SQL with '.' and '/' is not rejected as a missing trace path."""
    sql = '/* header */ SELECT e.duration_ps / 1000.0 AS us FROM Events AS e'
    xprof_cli.main(['xprof', 'query_events_db', sql])
    mock_query_events_db.assert_called_once_with(sql)


if __name__ == '__main__':
  absltest.main()
