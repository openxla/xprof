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

import gzip
import struct
from unittest import mock

from absl.testing import absltest

from xprof.convert import (
    raw_to_tool_data,
)
from xprof.protobuf import (
    trace_events_old_pb2,
)
from xprof.convert import _pywrap_profiler_plugin


def _zstd_frame_with_content_size(content_size: int) -> bytes:
  """Build a minimal zstd frame header declaring ``content_size``.

  Uses Single_Segment + 8-byte Frame_Content_Size so any size fits. The
  frame body is intentionally incomplete; bounds checks only need the header.
  """
  magic = b'\x28\xb5\x2f\xfd'
  # FCS_flag=3 (8 bytes), Single_Segment=1, no dict id, no checksum.
  descriptor = bytes([0xE0])
  fcs = struct.pack('<Q', content_size)
  # Pad a few bytes so the payload is non-empty after the header.
  return magic + descriptor + fcs + b'\x00\x01\x02\x03'


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


class CompressedTraceBoundsTest(absltest.TestCase):
  """Tests for compressed protobuf/gzip size limits and corrupt handling."""

  def test_max_decompressed_trace_bytes_is_positive(self):
    self.assertGreater(raw_to_tool_data.MAX_DECOMPRESSED_TRACE_BYTES, 0)

  def test_uncompressed_payload_passes_bounds_check(self):
    # Happy path for mocks / raw protobufs: no gzip/zstd magic → no-op.
    raw_to_tool_data.ensure_compressed_trace_within_bounds(b"compressed_pb")
    self.assertEqual(
        raw_to_tool_data.decompress_trace_data(b"not-compressed"),
        b"not-compressed",
    )

  def test_gzip_happy_path_decompresses_under_limit(self):
    payload = b"trace-event-payload" * 10
    compressed = gzip.compress(payload)
    raw_to_tool_data.ensure_compressed_trace_within_bounds(
        compressed, max_decompressed_bytes=len(payload) + 1
    )
    self.assertEqual(
        raw_to_tool_data.decompress_trace_data(
            compressed, max_decompressed_bytes=len(payload) + 1
        ),
        payload,
    )

  def test_gzip_corrupt_header_raises(self):
    corrupt = b"\x1f\x8b" + b"\x00\x00\x00\x00\xff\xff"
    with self.assertRaisesRegex(
        raw_to_tool_data.CompressedTraceError, "corrupt gzip"
    ):
      raw_to_tool_data.ensure_compressed_trace_within_bounds(corrupt)

  def test_gzip_oversize_raises(self):
    payload = b"A" * 10_000
    compressed = gzip.compress(payload)
    with self.assertRaisesRegex(
        raw_to_tool_data.CompressedTraceError, "exceeds limit"
    ):
      raw_to_tool_data.ensure_compressed_trace_within_bounds(
          compressed, max_decompressed_bytes=100
      )

  def test_zstd_happy_path_declared_size_under_limit(self):
    frame = _zstd_frame_with_content_size(1024)
    raw_to_tool_data.ensure_compressed_trace_within_bounds(
        frame, max_decompressed_bytes=1024
    )
    # decompress_trace_data size-checks zstd but returns compressed bytes.
    self.assertEqual(
        raw_to_tool_data.decompress_trace_data(
            frame, max_decompressed_bytes=1024
        ),
        frame,
    )

  def test_zstd_corrupt_header_raises(self):
    # Valid magic, truncated / invalid descriptor payload.
    corrupt = b"\x28\xb5\x2f\xfd\xff"
    with self.assertRaisesRegex(
        raw_to_tool_data.CompressedTraceError, "corrupt zstd"
    ):
      raw_to_tool_data.ensure_compressed_trace_within_bounds(corrupt)

  def test_zstd_reserved_bits_raise(self):
    # Descriptor with reserved bit 3 set.
    corrupt = b"\x28\xb5\x2f\xfd" + bytes([0x08]) + b"\x00"
    with self.assertRaisesRegex(
        raw_to_tool_data.CompressedTraceError, "reserved"
    ):
      raw_to_tool_data.ensure_compressed_trace_within_bounds(corrupt)

  def test_zstd_oversize_declared_content_raises(self):
    oversize = raw_to_tool_data.MAX_DECOMPRESSED_TRACE_BYTES + 1
    frame = _zstd_frame_with_content_size(oversize)
    with self.assertRaisesRegex(
        raw_to_tool_data.CompressedTraceError,
        r"decompressed zstd trace exceeds limit",
    ):
      raw_to_tool_data.ensure_compressed_trace_within_bounds(frame)

  def test_trace_viewer_pb_oversize_zstd_raises_value_error(self):
    frame = _zstd_frame_with_content_size(
        raw_to_tool_data.MAX_DECOMPRESSED_TRACE_BYTES + 1
    )
    params = {"trace_viewer_options": {"format": "pb"}}
    wrapper_func = mock.MagicMock(return_value=(frame, True))

    with self.assertRaisesRegex(ValueError, "exceeds limit"):
      raw_to_tool_data.xspace_to_tool_data(
          xspace_paths=["/path/to/xspace"],
          tool="trace_viewer",
          params=params,
          xspace_wrapper_func=wrapper_func,
      )

  def test_trace_viewer_streaming_pb_corrupt_gzip_raises_value_error(self):
    corrupt = b"\x1f\x8b\x00\x00\x00\x00\xff\xff"
    params = {"trace_viewer_options": {"format": "pb"}}
    wrapper_func = mock.MagicMock(return_value=(corrupt, True))

    with self.assertRaisesRegex(ValueError, "corrupt gzip"):
      raw_to_tool_data.xspace_to_tool_data(
          xspace_paths=["/path/to/xspace"],
          tool="trace_viewer@",
          params=params,
          xspace_wrapper_func=wrapper_func,
      )

  def test_trace_viewer_pb_valid_gzip_returns_octet_stream(self):
    payload = b"ok-trace"
    compressed = gzip.compress(payload)
    params = {"trace_viewer_options": {"format": "pb"}}
    wrapper_func = mock.MagicMock(return_value=(compressed, True))

    data, content_type = raw_to_tool_data.xspace_to_tool_data(
        xspace_paths=["/path/to/xspace"],
        tool="trace_viewer",
        params=params,
        xspace_wrapper_func=wrapper_func,
    )

    self.assertEqual(data, compressed)
    self.assertEqual(content_type, "application/octet-stream")


if __name__ == "__main__":
  absltest.main()
