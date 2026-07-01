# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Tests for the raw_to_tool_data module."""

from unittest import mock

from absl.testing import absltest

from xprof.convert import (
    raw_to_tool_data,
)
from xprof.protobuf import (
    trace_events_old_pb2,
)
from xprof.convert import _pywrap_profiler_plugin


class RawToToolDataTest(absltest.TestCase):
  """Tests for the raw_to_tool_data module."""

  def test_using_old_tool_format_maps_to_new_format(self):
    wrapper_func = mock.MagicMock(return_value=(b"trace_viewer@", True))
    data, content_type = raw_to_tool_data.xspace_to_tool_data(
        xspace_paths=["/path/to/xspace"],
        tool="trace_viewer@^",
        params={},
        xspace_wrapper_func=wrapper_func,
    )

    self.assertEqual(data, b"trace_viewer@")
    self.assertEqual(content_type, "application/json")
    wrapper_func.assert_called_once_with(
        ["/path/to/xspace"],
        "trace_viewer@",
        {"use_saved_result": True, "hosts": []},
    )

  def test_using_new_tool_format_does_not_map_to_old_format(self):
    wrapper_func = mock.MagicMock(return_value=(b"trace_viewer@", True))
    data, content_type = raw_to_tool_data.xspace_to_tool_data(
        xspace_paths=["/path/to/xspace"],
        tool="trace_viewer@",
        params={},
        xspace_wrapper_func=wrapper_func,
    )

    self.assertEqual(data, b"trace_viewer@")
    self.assertEqual(content_type, "application/json")
    wrapper_func.assert_called_once_with(
        ["/path/to/xspace"],
        "trace_viewer@",
        {"use_saved_result": True, "hosts": []},
    )

  def test_xspace_to_tool_data_trace_viewer_format_pb_returns_pb_data(self):
    params = {"trace_viewer_options": {"format": "pb"}}
    wrapper_func = mock.MagicMock(return_value=(b"compressed_pb", True))

    data, content_type = raw_to_tool_data.xspace_to_tool_data(
        xspace_paths=["/path/to/xspace"],
        tool="trace_viewer",
        params=params,
        xspace_wrapper_func=wrapper_func,
    )

    self.assertEqual(data, b"compressed_pb")
    self.assertEqual(content_type, "application/octet-stream")
    wrapper_func.assert_called_once_with(
        ["/path/to/xspace"],
        "trace_viewer",
        {"format": "pb", "use_saved_result": True},
    )

  def test_xspace_to_tool_data_trace_viewer_streaming_pb_returns_pb_data(self):
    params = {"trace_viewer_options": {"format": "pb"}}
    wrapper_func = mock.MagicMock(return_value=(b"compressed_pb", True))

    data, content_type = raw_to_tool_data.xspace_to_tool_data(
        xspace_paths=["/path/to/xspace"],
        tool="trace_viewer@",
        params=params,
        xspace_wrapper_func=wrapper_func,
    )

    self.assertEqual(data, b"compressed_pb")
    self.assertEqual(content_type, "application/octet-stream")
    wrapper_func.assert_called_once_with(
        ["/path/to/xspace"],
        "trace_viewer@",
        {"format": "pb", "use_saved_result": True, "hosts": []},
    )

  def test_xspace_to_tool_data_trace_viewer_streaming_json_returns_raw_data(
      self,
  ):
    params = {"trace_viewer_options": {"format": "json"}}
    wrapper_func = mock.MagicMock(return_value=(b"raw_json_data", True))

    data, content_type = raw_to_tool_data.xspace_to_tool_data(
        xspace_paths=["/path/to/xspace"],
        tool="trace_viewer@",
        params=params,
        xspace_wrapper_func=wrapper_func,
    )

    self.assertEqual(data, b"raw_json_data")
    self.assertEqual(content_type, "application/json")
    wrapper_func.assert_called_once_with(
        ["/path/to/xspace"],
        "trace_viewer@",
        {"format": "json", "use_saved_result": True, "hosts": []},
    )

  def test_xspace_to_tool_data_trace_viewer_json_returns_json_string(self):
    trace = trace_events_old_pb2.Trace()
    raw_trace = trace.SerializeToString()
    params = {"trace_viewer_options": {"format": "json"}}
    wrapper_func = mock.MagicMock(return_value=(raw_trace, True))

    data, content_type = raw_to_tool_data.xspace_to_tool_data(
        xspace_paths=["/path/to/xspace"],
        tool="trace_viewer",
        params=params,
        xspace_wrapper_func=wrapper_func,
    )

    self.assertIsInstance(data, str)
    self.assertEqual(content_type, "application/json")
    wrapper_func.assert_called_once_with(
        ["/path/to/xspace"],
        "trace_viewer",
        {"format": "json", "use_saved_result": True},
    )

  def test_xspace_to_tool_data_trace_viewer_failure_returns_none(self):
    params = {"trace_viewer_options": {"format": "pb"}}
    wrapper_func = mock.MagicMock(return_value=(b"error", False))

    data, content_type = raw_to_tool_data.xspace_to_tool_data(
        xspace_paths=["/path/to/xspace"],
        tool="trace_viewer",
        params=params,
        xspace_wrapper_func=wrapper_func,
    )

    self.assertIsNone(data)
    self.assertEqual(content_type, "application/json")
    wrapper_func.assert_called_once_with(
        ["/path/to/xspace"],
        "trace_viewer",
        {"format": "pb", "use_saved_result": True},
    )

  def test_xspace_to_tool_data_trace_viewer_streaming_fail_returns_none(self):
    params = {"trace_viewer_options": {"format": "pb"}}
    wrapper_func = mock.MagicMock(return_value=(b"error", False))

    data, content_type = raw_to_tool_data.xspace_to_tool_data(
        xspace_paths=["/path/to/xspace"],
        tool="trace_viewer@",
        params=params,
        xspace_wrapper_func=wrapper_func,
    )

    self.assertIsNone(data)
    self.assertEqual(content_type, "application/json")
    wrapper_func.assert_called_once_with(
        ["/path/to/xspace"],
        "trace_viewer@",
        {"format": "pb", "use_saved_result": True, "hosts": []},
    )

  def test_xspace_to_tools_data_from_bytes_valid_input_returns_tool_data(self):
    mock_func = self.enter_context(
        mock.patch.object(
            _pywrap_profiler_plugin,
            "xspace_to_tools_data_from_byte_string",
            return_value=(b"result", True),
            autospec=True,
        )
    )
    params = {"trace_viewer_options": {"format": "pb"}}

    data, content_type = raw_to_tool_data.xspace_to_tools_data_from_byte_string(
        xspace_byte_list=[b"xspace"],
        filenames=["file.pb"],
        tool="trace_viewer",
        params=params,
    )

    self.assertEqual(data, b"result")
    self.assertEqual(content_type, "application/octet-stream")
    mock_func.assert_called_once_with(
        [b"xspace"],
        ["file.pb"],
        "trace_viewer",
        {"format": "pb", "use_saved_result": True},
    )

  def test_json_to_csv_string_valid_json_returns_csv_string(self):
    json_data = {
        "cols": [{"label": "a"}, {"label": "b"}],
        "rows": [{"c": [{"v": 1}, {"v": 2}]}],
    }

    csv_str = raw_to_tool_data.json_to_csv_string(json_data)

    self.assertIsInstance(csv_str, str)

  def test_xspace_to_tool_data_default_wrapper_valid_returns_tool_data(self):
    mock_func = self.enter_context(
        mock.patch.object(
            _pywrap_profiler_plugin,
            "xspace_to_tools_data",
            return_value=(b"result", True),
            autospec=True,
        )
    )
    params = {"trace_viewer_options": {"format": "pb"}}

    data, content_type = raw_to_tool_data.xspace_to_tool_data(
        xspace_paths=["/path/to/xspace"],
        tool="trace_viewer",
        params=params,
    )

    self.assertEqual(data, b"result")
    self.assertEqual(content_type, "application/octet-stream")
    mock_func.assert_called_once_with(
        ["/path/to/xspace"],
        "trace_viewer",
        {"format": "pb", "use_saved_result": True},
    )

  def test_xspace_to_tool_names_valid_xspace_returns_tool_names(self):
    mock_func = self.enter_context(
        mock.patch.object(
            _pywrap_profiler_plugin,
            "xspace_to_tools_data",
            return_value=(b"trace_viewer,op_profile", True),
            autospec=True,
        )
    )

    names = raw_to_tool_data.xspace_to_tool_names(["/path/to/xspace"])

    self.assertEqual(names, ["trace_viewer", "op_profile"])
    mock_func.assert_called_once_with(["/path/to/xspace"], "tool_names")

  def test_xspace_to_tool_names_invalid_xspace_returns_empty_list(self):
    mock_func = self.enter_context(
        mock.patch.object(
            _pywrap_profiler_plugin,
            "xspace_to_tools_data",
            return_value=(b"error", False),
            autospec=True,
        )
    )

    names = raw_to_tool_data.xspace_to_tool_names(["/path/to/xspace"])

    self.assertEmpty(names)
    mock_func.assert_called_once_with(["/path/to/xspace"], "tool_names")

  def test_xspace_to_tool_data_graph_viewer_pb_returns_octet_stream(self):
    params = {"graph_viewer_options": {"type": "pb"}}
    wrapper_func = mock.MagicMock(return_value=(b"pb_data", True))

    data, content_type = raw_to_tool_data.xspace_to_tool_data(
        xspace_paths=["/path/to/xspace"],
        tool="graph_viewer",
        params=params,
        xspace_wrapper_func=wrapper_func,
    )

    self.assertEqual(data, b"pb_data")
    self.assertEqual(content_type, "application/octet-stream")
    wrapper_func.assert_called_once_with(
        ["/path/to/xspace"],
        "graph_viewer",
        {"type": "pb", "use_saved_result": True},
    )

  def test_xspace_to_tool_data_graph_viewer_graph_returns_html(self):
    params = {"graph_viewer_options": {"type": "graph"}}
    wrapper_func = mock.MagicMock(return_value=(b"<html>", True))

    data, content_type = raw_to_tool_data.xspace_to_tool_data(
        xspace_paths=["/path/to/xspace"],
        tool="graph_viewer",
        params=params,
        xspace_wrapper_func=wrapper_func,
    )

    self.assertEqual(data, b"<html>")
    self.assertEqual(content_type, "text/html")
    wrapper_func.assert_called_once_with(
        ["/path/to/xspace"],
        "graph_viewer",
        {"type": "graph", "use_saved_result": True},
    )

  def test_xspace_to_tool_data_graph_viewer_unknown_returns_plain_text(self):
    params = {"graph_viewer_options": {"type": "unknown"}}
    wrapper_func = mock.MagicMock(return_value=(b"text_data", True))

    data, content_type = raw_to_tool_data.xspace_to_tool_data(
        xspace_paths=["/path/to/xspace"],
        tool="graph_viewer",
        params=params,
        xspace_wrapper_func=wrapper_func,
    )

    self.assertEqual(data, b"text_data")
    self.assertEqual(content_type, "text/plain")
    wrapper_func.assert_called_once_with(
        ["/path/to/xspace"],
        "graph_viewer",
        {"type": "unknown", "use_saved_result": True},
    )

  def test_xspace_to_tool_data_graph_viewer_failure_raises_value_error(self):
    params = {"graph_viewer_options": {"type": "pb"}}
    wrapper_func = mock.MagicMock(return_value=(b"mock error bytes", False))

    with self.assertRaisesRegex(ValueError, "mock error bytes"):
      raw_to_tool_data.xspace_to_tool_data(
          xspace_paths=["/path/to/xspace"],
          tool="graph_viewer",
          params=params,
          xspace_wrapper_func=wrapper_func,
      )
    wrapper_func.assert_called_once_with(
        ["/path/to/xspace"],
        "graph_viewer",
        {"type": "pb", "use_saved_result": True},
    )


if __name__ == "__main__":
  absltest.main()
