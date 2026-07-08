import json
from unittest import mock

from absl.testing import absltest
from xprof.cli.internal.oss import xprof_client
from xprof.cli.tools import get_graph_viewer_tool


class GetGraphViewerToolTest(absltest.TestCase):

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
    result = get_graph_viewer_tool.get_graph_viewer()
    result_json = json.loads(result)
    self.assertIn("error", result_json)
    self.assertEqual(
        result_json["error"], "Either session_id or symbol_id must be provided"
    )

  def test_get_graph_viewer_both_args(self):
    result = get_graph_viewer_tool.get_graph_viewer(
        session_id="session_123", symbol_id="symbol_123"
    )
    result_json = json.loads(result)
    self.assertIn("error", result_json)
    self.assertEqual(
        result_json["error"], "Cannot set both session_id and symbol_id"
    )

  def test_get_graph_viewer_with_session_id(self):
    self.mock_client.fetch.return_value = (81, b"hlo content")
    result = get_graph_viewer_tool.get_graph_viewer(session_id="session_123")
    self.assertEqual(result, "hlo content")
      self.mock_client.fetch.assert_called_once_with(
          tool_name="graph_viewer.json",
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
          tool_name="graph_viewer.json",
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
        tool_name="graph_viewer.json",
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


if __name__ == "__main__":
  absltest.main()
