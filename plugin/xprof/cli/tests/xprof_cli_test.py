from typing import Any
import unittest
from unittest import mock

from xprof.cli import xprof_cli


class XProfCliTest(unittest.TestCase):

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

  @mock.patch.object(xprof_cli.XProfCli, 'get_events_db_session_root')
  def test_get_events_db_session_root(self, mock_get_root):
    mock_get_root.return_value = {'status': 'success'}
    result = self.cli.get_events_db_session_root('session_123')
    mock_get_root.assert_called_with('session_123')
    self.assertEqual(result, {'status': 'success'})

  @mock.patch.object(xprof_cli.XProfCli, 'get_profile_summary')
  def test_get_profile_summary(self, mock_get_summary):
    self.cli.get_profile_summary('session_123')
    mock_get_summary.assert_called_with('session_123')

  @mock.patch.object(xprof_cli.XProfCli, 'get_hosts')
  def test_get_hosts(self, mock_get_hosts):
    self.cli.get_hosts('session_123')
    mock_get_hosts.assert_called_with('session_123')

  @mock.patch.object(xprof_cli.XProfCli, 'detect_layout_mismatch_copies')
  def test_detect_layout_mismatch_copies(self, mock_detect):
    self.cli.detect_layout_mismatch_copies('session_123')
    mock_detect.assert_called_with('session_123')

  @mock.patch.object(xprof_cli.fire, 'Fire')
  def test_main(self, mock_fire):
    xprof_cli.main([])
    mock_fire.assert_called_once_with(mock.ANY, command=None, name='xprof')
    self.assertIsInstance(mock_fire.call_args[0][0], xprof_cli.XProfCli)


if __name__ == '__main__':
  unittest.main()
