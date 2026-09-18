"""Unit tests for get_graph_viewer_tool CLI interface in OSS."""

from unittest import mock

from absl.testing import absltest
from xprof.cli.internal.oss import xprof_client
from xprof.cli.tools.oss import get_graph_viewer_tool


class OssGraphViewerToolTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_client = mock.create_autospec(
        xprof_client.CachedXprofClient, instance=True, spec_set=True
    )
    self.enter_context(
        mock.patch.object(
            xprof_client,
            "get_client",
            return_value=self.mock_client,
            autospec=True,
        )
    )

  def test_get_graph_viewer_missing_args(self):
    with self.assertRaises(ValueError) as cm:
      get_graph_viewer_tool.get_graph_viewer()
    self.assertEqual(
        str(cm.exception), "Either session_id or symbol_id must be provided"
    )

  def test_get_graph_viewer_both_args(self):
    with self.assertRaises(ValueError) as cm:
      get_graph_viewer_tool.get_graph_viewer(
          session_id="session_123", symbol_id="symbol_123"
      )
    self.assertEqual(
        str(cm.exception), "Cannot set both session_id and symbol_id"
    )

  def test_get_graph_viewer_with_session_id(self):
    self.mock_client.fetch.return_value = (81, b"hlo content")
    result = get_graph_viewer_tool.get_graph_viewer(session_id="session_123")
    self.assertEqual(result, "hlo content")
    self.mock_client.fetch.assert_called_once_with(
        tool_name="graph_viewer",
        session_id="session_123",
        graph_viewer_options={
            "graph_type": "xla",
            "type": "short_txt",
            "show_metadata": "true",
        },
    )

  def test_get_graph_viewer_with_symbol_id(self):
    self.mock_client.fetch.return_value = (81, b"hlo content")
    result = get_graph_viewer_tool.get_graph_viewer(symbol_id="symbol_123")
    self.assertEqual(result, "hlo content")
    self.mock_client.fetch.assert_called_once_with(
        tool_name="graph_viewer",
        session_id="xsymbol",
        graph_viewer_options={
            "graph_type": "xla",
            "type": "short_txt",
            "show_metadata": "true",
            "symbol_id": "symbol_123",
        },
    )

  def test_get_graph_viewer_with_advanced_params(self):
    self.mock_client.fetch.return_value = (81, b"graph content")
    result = get_graph_viewer_tool.get_graph_viewer(
        session_id="session_123",
        node_name="fusion.112",
        module_name="jit_train_step",
        graph_width=2,
        show_metadata=False,
        merge_fusion=True,
        graph_type="xla",
        tag="graph_viewer",
        tool="hlo_op_profile",
        op_profile_limit=1,
        use_xplane=1,
        output_type="short_txt",
    )
    self.assertEqual(result, "graph content")
    self.mock_client.fetch.assert_called_once_with(
        tool_name="graph_viewer",
        session_id="session_123",
        graph_viewer_options={
            "graph_type": "xla",
            "type": "short_txt",
            "show_metadata": "false",
            "node_name": "fusion.112",
            "module_name": "jit_train_step",
            "graph_width": "2",
            "merge_fusion": "true",
            "tag": "graph_viewer",
            "tool": "hlo_op_profile",
            "op_profile_limit": "1",
            "use_xplane": "1",
        },
    )

  def test_missing_hlo_proto_returns_clean_diagnostic(self):
    """Verifies graph_viewer raises FileNotFoundError when HLO proto is missing."""
    self.mock_client.fetch.side_effect = ValueError(
        "Can not load hlo proto from options."
    )
    with self.assertRaises(FileNotFoundError) as cm:
      get_graph_viewer_tool.get_graph_viewer(session_id="session_without_hlo")
    self.assertIn("xla_flags", str(cm.exception).lower())

  def test_empty_graph_viewer_data_raises_file_not_found(self):
    self.mock_client.fetch.return_value = (81, b"")
    with self.assertRaises(FileNotFoundError) as cm:
      get_graph_viewer_tool.get_graph_viewer(session_id="session_empty")
    self.assertIn("No graph_viewer data found", str(cm.exception))

  @mock.patch.object(get_graph_viewer_tool.hlo_tools, "resolve_module_name")
  @mock.patch.object(get_graph_viewer_tool.hlo_tools, "get_hlo_proto_files")
  def test_short_module_name_resolution(
      self, mock_get_files, mock_resolve_module
  ):
    mock_get_files.return_value = ["jit_train_step(7216021599878099202)"]
    mock_resolve_module.return_value = "jit_train_step(7216021599878099202)"
    self.mock_client.fetch.return_value = (81, b"resolved hlo content")

    result = get_graph_viewer_tool.get_graph_viewer(
        session_id="session_123", module_name="jit_train_step"
    )
    self.assertEqual(result, "resolved hlo content")
    mock_resolve_module.assert_called_once_with("session_123", "jit_train_step")
    self.mock_client.fetch.assert_called_once_with(
        tool_name="graph_viewer",
        session_id="session_123",
        graph_viewer_options={
            "graph_type": "xla",
            "type": "short_txt",
            "show_metadata": "true",
            "module_name": "jit_train_step(7216021599878099202)",
        },
    )

  def test_internal_path_leak_suppression(self):
    self.mock_client.fetch.side_effect = RuntimeError(
        "Failed to open /tmp/xprof_xyz/jit_train_step.hlo_proto.pb: No such"
        " file or directory"
    )
    with self.assertRaises(FileNotFoundError) as cm:
      get_graph_viewer_tool.get_graph_viewer(
          session_id="session_123", module_name="jit_train_step"
      )
    self.assertNotIn("/tmp/", str(cm.exception))
    self.assertIn("jit_train_step", str(cm.exception))


if __name__ == "__main__":
  absltest.main()
