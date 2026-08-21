"""Unit tests for OSS LocalXprofClient logdir configuration and caching."""

import hashlib
import os
import pathlib
import shutil
import sys
import tempfile
import unittest
from unittest import mock

# pylint: disable=g-import-not-at-top
from xprof.cli.internal import decorators
from xprof.cli.internal.oss import xprof_client


class OssXprofClientTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.temp_dir = tempfile.mkdtemp()
    self.addCleanup(lambda: shutil.rmtree(self.temp_dir, ignore_errors=True))
    if hasattr(xprof_client.convert, 'xspace_to_tool_data'):
      self.mock_converter = mock.patch.object(
          xprof_client.convert,
          'xspace_to_tool_data',
          return_value=(b'{}', 'application/json'),
      ).start()
    else:
      self.mock_converter = mock.patch.object(
          xprof_client.convert,
          'raw_to_tool_data',
          return_value=('application/json', b'{}'),
      ).start()
    self.addCleanup(mock.patch.stopall)

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

  def test_fetch_unknown_tool_raises_value_error(self):
    client = xprof_client.LocalXprofClient()
    with self.assertRaisesRegex(ValueError, 'Unknown XProf tool name'):
      client.fetch('non_existent_tool', 'test_session')

  def test_compute_fingerprint_uses_exact_xspace_paths(self):
    """Verifies fingerprint covers input traces only and ignores generated artifacts."""
    run_dir = pathlib.Path(self.temp_dir) / 'run1'
    nested_dir = run_dir / 'plugins' / 'profile' / '2026_08_20'
    nested_dir.mkdir(parents=True, exist_ok=True)

    trace1 = run_dir / 'host.xplane.pb'
    trace1.write_bytes(b'trace_bytes_1')
    trace2 = nested_dir / 'worker.xspace.pb'
    trace2.write_bytes(b'trace_bytes_2')

    client = xprof_client.LocalXprofClient()
    xspace_paths = client.get_xspace_paths(run_dir)
    self.assertEqual(len(xspace_paths), 2)

    fp1 = decorators._compute_path_fingerprint(
        run_dir, xspace_paths=xspace_paths
    )
    self.assertNotEqual(fp1, 'NO_TRACE_INPUTS')

    # Simulate C++ converter generating artifacts in run_dir
    (run_dir / 'ALL_HOSTS.op_stats_v2.pb').write_bytes(b'generated_op_stats')
    (run_dir / 'summary.json').write_text('{"status": "ok"}', encoding='utf-8')
    (run_dir / '.xprof_trace_fingerprint').write_text('stale', encoding='utf-8')

    xspace_paths_after = client.get_xspace_paths(run_dir)
    fp2 = decorators._compute_path_fingerprint(
        run_dir, xspace_paths=xspace_paths_after
    )

    self.assertEqual(
        fp1,
        fp2,
        msg='Fingerprint must remain identical when generated files appear',
    )

  def test_compute_fingerprint_nested_layout_sensitivity(self):
    """Verifies modifying trace files in nested directory hierarchy updates fingerprint."""
    run_dir = pathlib.Path(self.temp_dir) / 'run_nested'
    nested_dir = run_dir / 'plugins' / 'profile' / '2026_08_20'
    nested_dir.mkdir(parents=True, exist_ok=True)

    trace = nested_dir / 'worker.xspace.pb'
    trace.write_bytes(b'original_trace')

    client = xprof_client.LocalXprofClient()
    xspace_paths = client.get_xspace_paths(run_dir)
    fp1 = decorators._compute_path_fingerprint(
        run_dir, xspace_paths=xspace_paths
    )

    # Modify nested trace content and mtime
    trace.write_bytes(b'modified_trace_longer_content')
    stat = trace.stat()
    os.utime(trace, (stat.st_atime + 10, stat.st_mtime + 10))

    fp2 = decorators._compute_path_fingerprint(
        run_dir, xspace_paths=client.get_xspace_paths(run_dir)
    )
    self.assertNotEqual(
        fp1, fp2, msg='Modifying nested trace must alter fingerprint'
    )

  def test_compute_fingerprint_no_trace_inputs_sentinel(self):
    """Verifies empty directories return explicit NO_TRACE_INPUTS sentinel."""
    empty_dir = pathlib.Path(self.temp_dir) / 'empty'
    empty_dir.mkdir()

    fp = decorators._compute_path_fingerprint(empty_dir)
    self.assertEqual(fp, 'NO_TRACE_INPUTS')

    # Directory with non-trace files only
    (empty_dir / 'notes.txt').write_text('not a trace', encoding='utf-8')
    fp_non_trace = decorators._compute_path_fingerprint(empty_dir)
    self.assertEqual(fp_non_trace, 'NO_TRACE_INPUTS')

  def test_external_atomic_fingerprint_storage(self):
    """Verifies fingerprint is stored in /tmp/.../fingerprints/ and not in run_dir."""
    run_dir = pathlib.Path(self.temp_dir) / 'run_external'
    run_dir.mkdir()
    trace = run_dir / 'host.xplane.pb'
    trace.write_bytes(b'trace_data')

    client = xprof_client.LocalXprofClient()
    client.fetch('overview_page.json', str(run_dir))

    # Assert no .xprof_trace_fingerprint in run_dir
    self.assertFalse(
        (run_dir / '.xprof_trace_fingerprint').exists(),
        msg='Trace directory must not contain .xprof_trace_fingerprint',
    )

    # Assert external fingerprint file exists
    run_dir_hash = hashlib.sha256(
        str(run_dir).encode('utf-8')
    ).hexdigest()[:16]
    fp_file = (
        decorators._get_cache_dir() / 'fingerprints' / f'{run_dir_hash}.fp'
    )
    self.assertTrue(
        fp_file.exists(), msg='External fingerprint file must exist'
    )

  def test_fetch_readonly_trace_dir(self):
    """Verifies fetch() succeeds on read-only trace directories."""
    run_dir = pathlib.Path(self.temp_dir) / 'run_readonly'
    run_dir.mkdir()
    trace = run_dir / 'host.xplane.pb'
    trace.write_bytes(b'trace_data')

    # Set directory permissions to read-only (0o555)
    os.chmod(run_dir, 0o555)
    try:
      client = xprof_client.LocalXprofClient()
      content_type, data = client.fetch('overview_page.json', str(run_dir))
      self.assertEqual(content_type, 'application/json')
      self.assertEqual(data, b'{}')
    finally:
      os.chmod(run_dir, 0o755)

  def test_fetch_bypass_cache_flag_forwarded(self):
    """Verifies bypass_cache=True unlinks op_stats and sets use_saved_result='0'."""
    run_dir = pathlib.Path(self.temp_dir) / 'run_bypass'
    run_dir.mkdir()
    trace = run_dir / 'host.xplane.pb'
    trace.write_bytes(b'trace_data')
    op_stats = run_dir / 'ALL_HOSTS.op_stats_v2.pb'
    op_stats.write_bytes(b'existing_op_stats')

    client = xprof_client.LocalXprofClient()
    client.fetch('overview_page.json', str(run_dir), bypass_cache=True)

    # Assert op_stats was unlinked
    self.assertFalse(op_stats.exists(), msg='op_stats should be unlinked')
    # Assert use_saved_result='0' was passed to converter
    call_kwargs = self.mock_converter.call_args[1]
    self.assertEqual(call_kwargs['params']['use_saved_result'], '0')


if __name__ == '__main__':
  unittest.main()

