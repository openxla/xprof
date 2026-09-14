"""Unit tests for OSS upload_trace tool."""

import json
import pathlib
import shutil
import sys
import tempfile
import unittest
from unittest import mock

# pylint: disable=g-import-not-at-top
from xprof.cli.internal.oss import xprof_client
from xprof.cli.tools.oss import upload_trace_tool


class OssUploadTraceToolTest(unittest.TestCase):
  """Tests for upload_trace tool in OSS mode."""

  def setUp(self) -> None:
    super().setUp()
    self.temp_dir = tempfile.mkdtemp()
    self.addCleanup(lambda: shutil.rmtree(self.temp_dir, ignore_errors=True))
    self.client = xprof_client.get_client()
    self.original_logdir = self.client.logdir
    self.addCleanup(
        lambda: self.client.set_logdir(
            str(self.original_logdir)
            if self.original_logdir is not None
            else None
        )
    )

    # Create dummy source trace file
    self.src_dir = pathlib.Path(self.temp_dir) / 'source'
    self.src_dir.mkdir(parents=True, exist_ok=True)
    self.src_file = self.src_dir / 'test_trace.xplane.pb'
    self.trace_content = b'mock_xplane_binary_payload_12345'
    self.src_file.write_bytes(self.trace_content)

  def test_upload_trace_success_with_custom_run_name(self) -> None:
    """Tests trace import with explicit custom run_name."""
    logdir = pathlib.Path(self.temp_dir) / 'logdir'
    self.client.set_logdir(str(logdir))

    res_raw = upload_trace_tool.upload_trace(
        str(self.src_file), run_name='my_custom_run'
    )
    res = json.loads(res_raw)

    self.assertEqual(res.get('status'), 'success')
    self.assertEqual(res.get('run_name'), 'my_custom_run')
    expected_file = (
        logdir
        / 'plugins'
        / 'profile'
        / 'my_custom_run'
        / 'test_trace.xplane.pb'
    )
    self.assertTrue(expected_file.exists())
    self.assertEqual(expected_file.read_bytes(), self.trace_content)

  def test_upload_trace_default_run_name(self) -> None:
    """Tests trace import fallback to 'imported_trace' when run_name is None."""
    logdir = pathlib.Path(self.temp_dir) / 'logdir'
    self.client.set_logdir(str(logdir))

    res_raw = upload_trace_tool.upload_trace(str(self.src_file))
    res = json.loads(res_raw)

    self.assertEqual(res.get('status'), 'success')
    self.assertEqual(res.get('run_name'), 'imported_trace')
    expected_file = (
        logdir
        / 'plugins'
        / 'profile'
        / 'imported_trace'
        / 'test_trace.xplane.pb'
    )
    self.assertTrue(expected_file.exists())
    self.assertEqual(expected_file.read_bytes(), self.trace_content)

  def test_upload_trace_missing_logdir_raises_value_error(self) -> None:
    """Tests that missing client logdir raises ValueError."""
    self.client.set_logdir(None)

    with self.assertRaises(ValueError) as cm:
      upload_trace_tool.upload_trace(str(self.src_file), run_name='test_run')
    self.assertIn('Logdir not set in client', str(cm.exception))

  def test_upload_trace_nonexistent_file_raises_file_not_found(self) -> None:
    """Tests that a non-existent file raises FileNotFoundError."""
    logdir = pathlib.Path(self.temp_dir) / 'logdir'
    self.client.set_logdir(str(logdir))
    nonexistent = str(self.src_dir / 'does_not_exist.xplane.pb')

    with self.assertRaises(FileNotFoundError) as cm:
      upload_trace_tool.upload_trace(nonexistent, run_name='test_run')
    self.assertIn('does not exist', str(cm.exception))

  def test_upload_trace_invalid_extension_rejected(self) -> None:
    """Tests that non-XSpace file extensions raise ValueError."""
    logdir = pathlib.Path(self.temp_dir) / 'logdir'
    self.client.set_logdir(str(logdir))

    for invalid_name in ('trace.trace.json.gz', 'trace.txt', 'trace.json'):
      invalid_file = self.src_dir / invalid_name
      invalid_file.write_bytes(b'dummy_data')
      with self.assertRaises(ValueError) as cm:
        upload_trace_tool.upload_trace(str(invalid_file), run_name='test_run')
      self.assertIn('Unsupported file format', str(cm.exception))

  def test_upload_trace_valid_xspace_pb_accepted(self) -> None:
    """Tests that .xspace.pb files are accepted as valid XSpace trace files."""
    logdir = pathlib.Path(self.temp_dir) / 'logdir'
    self.client.set_logdir(str(logdir))

    xspace_file = self.src_dir / 'trace.xspace.pb'
    xspace_file.write_bytes(b'mock_xspace_binary_data')
    res_raw = upload_trace_tool.upload_trace(
        str(xspace_file), run_name='xspace_run'
    )
    res = json.loads(res_raw)
    self.assertEqual(res.get('status'), 'success')
    self.assertEqual(res.get('run_name'), 'xspace_run')
    expected_file = (
        logdir / 'plugins' / 'profile' / 'xspace_run' / 'trace.xspace.pb'
    )
    self.assertTrue(expected_file.exists())

  def test_upload_trace_path_traversal_rejected(self) -> None:
    """Tests that absolute paths or traversal run_names raise ValueError."""
    logdir = pathlib.Path(self.temp_dir) / 'logdir'
    self.client.set_logdir(str(logdir))

    for malicious_run in (
        '/tmp/ESCAPED',
        '../../ESCAPED',
        'nested/sub_run',
        r'nested\sub_run',
        '..',
    ):
      with self.assertRaises(
          ValueError,
          msg=f"Expected traversal '{malicious_run}' to raise ValueError!",
      ) as cm:
        upload_trace_tool.upload_trace(
            str(self.src_file), run_name=malicious_run
        )
      self.assertIn('Invalid run_name', str(cm.exception))

  def test_upload_trace_copy_failure_raises_os_error(self) -> None:
    """Tests that copy failures raising OSError are re-raised."""
    logdir = pathlib.Path(self.temp_dir) / 'logdir'
    self.client.set_logdir(str(logdir))

    with mock.patch.object(shutil, 'copy', side_effect=OSError('Disk full')):
      with self.assertRaises(OSError) as cm:
        upload_trace_tool.upload_trace(str(self.src_file), run_name='fail_run')
      self.assertIn('Disk full', str(cm.exception))

  def test_upload_trace_unexpected_failure_raises_runtime_error(self) -> None:
    """Tests that unexpected non-OS exceptions raise RuntimeError."""
    logdir = pathlib.Path(self.temp_dir) / 'logdir'
    self.client.set_logdir(str(logdir))

    with mock.patch.object(
        shutil, 'copy', side_effect=Exception('Unexpected internal error')
    ):
      with self.assertRaises(RuntimeError) as cm:
        upload_trace_tool.upload_trace(str(self.src_file), run_name='fail_run')
      self.assertIn('Failed to import trace', str(cm.exception))

  def test_upload_trace_ignores_extra_kwargs(self) -> None:
    """Tests that ttl, tag, and extra kwargs are safely ignored."""
    logdir = pathlib.Path(self.temp_dir) / 'logdir'
    self.client.set_logdir(str(logdir))

    res_raw = upload_trace_tool.upload_trace(
        str(self.src_file),
        ttl=3600,
        tag=['tag1', 'tag2'],
        run_name='tagged_run',
        extra_param='ignored_value',
    )
    res = json.loads(res_raw)

    self.assertEqual(res.get('status'), 'success')
    self.assertEqual(res.get('run_name'), 'tagged_run')


if __name__ == '__main__':
  unittest.main()
