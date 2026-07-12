# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""For conversion of raw files to tool data.

Usage:
    data = xspace_to_tool_data(xplane, tool, params)
    data = tool_proto_to_tool_data(tool_proto, tool, params)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections.abc import Callable, Mapping, Sequence
import gzip
import io
import logging
import struct
from typing import Any

from xprof.convert import csv_writer
from xprof.convert import trace_events_json
from xprof.protobuf import (
    trace_events_old_pb2,
)
from xprof.convert import _pywrap_profiler_plugin


logger = logging.getLogger('tensorboard')

# Hard cap on decompressed protobuf/gzip trace payloads. Guards against
# decompression bombs when serving format=pb compressed traces. Override in
# tests via the max_decompressed_bytes parameter on the helpers below.
MAX_DECOMPRESSED_TRACE_BYTES = 512 * 1024 * 1024  # 512 MiB

_GZIP_MAGIC = b'\x1f\x8b'
_ZSTD_MAGIC = b'\x28\xb5\x2f\xfd'
_GZIP_READ_CHUNK = 1024 * 1024  # 1 MiB


class CompressedTraceError(ValueError):
  """Raised when a compressed trace is corrupt or exceeds size limits."""


def _gzip_decompress_bounded(
    data: bytes, max_decompressed_bytes: int
) -> bytes:
  """Decompress gzip data, rejecting corrupt streams and oversize output."""
  try:
    decoder = gzip.GzipFile(fileobj=io.BytesIO(data), mode='rb')
  except Exception as exc:  # pylint: disable=broad-except
    raise CompressedTraceError(
        f'corrupt gzip compressed trace: {exc}'
    ) from exc

  out = io.BytesIO()
  total = 0
  try:
    while True:
      chunk = decoder.read(_GZIP_READ_CHUNK)
      if not chunk:
        break
      total += len(chunk)
      if total > max_decompressed_bytes:
        raise CompressedTraceError(
            'decompressed gzip trace exceeds limit of'
            f' {max_decompressed_bytes} bytes'
        )
      out.write(chunk)
  except CompressedTraceError:
    raise
  except EOFError as exc:
    raise CompressedTraceError(
        f'corrupt gzip compressed trace: truncated stream: {exc}'
    ) from exc
  except OSError as exc:
    raise CompressedTraceError(
        f'corrupt gzip compressed trace: {exc}'
    ) from exc
  return out.getvalue()


def _zstd_declared_content_size(data: bytes) -> int:
  """Return Frame_Content_Size from a zstd frame header.

  Raises:
    CompressedTraceError: If the frame is truncated, reserved bits are set,
      or content size is not declared in the header.
  """
  if len(data) < 5:
    raise CompressedTraceError('corrupt zstd compressed trace: truncated header')
  if not data.startswith(_ZSTD_MAGIC):
    raise CompressedTraceError('corrupt zstd compressed trace: bad magic')

  descriptor = data[4]
  # Bit 4 is unused (must be 0); bit 3 is reserved (must be 0).
  if descriptor & 0x18:
    raise CompressedTraceError(
        'corrupt zstd compressed trace: reserved/unused header bits set'
    )

  fcs_flag = (descriptor >> 6) & 0x3
  single_segment = bool(descriptor & 0x20)
  dict_id_flag = descriptor & 0x3

  offset = 5  # past magic + descriptor
  if not single_segment:
    # Window_Descriptor is present when Single_Segment is clear.
    if len(data) < offset + 1:
      raise CompressedTraceError(
          'corrupt zstd compressed trace: truncated window descriptor'
      )
    offset += 1

  dict_id_sizes = (0, 1, 2, 4)
  dict_id_size = dict_id_sizes[dict_id_flag]
  if len(data) < offset + dict_id_size:
    raise CompressedTraceError(
        'corrupt zstd compressed trace: truncated dictionary id'
    )
  offset += dict_id_size

  if fcs_flag == 0:
    fcs_size = 1 if single_segment else 0
  elif fcs_flag == 1:
    fcs_size = 2
  elif fcs_flag == 2:
    fcs_size = 4
  else:
    fcs_size = 8

  if fcs_size == 0:
    raise CompressedTraceError(
        'corrupt zstd compressed trace: missing frame content size'
        ' (decompressed size unknown; refusing to process)'
    )
  if len(data) < offset + fcs_size:
    raise CompressedTraceError(
        'corrupt zstd compressed trace: truncated frame content size'
    )

  fcs_bytes = data[offset : offset + fcs_size]
  if fcs_size == 1:
    content_size = fcs_bytes[0]
  elif fcs_size == 2:
    # 2-byte FCS stores (size - 256).
    content_size = struct.unpack('<H', fcs_bytes)[0] + 256
  elif fcs_size == 4:
    content_size = struct.unpack('<I', fcs_bytes)[0]
  else:
    content_size = struct.unpack('<Q', fcs_bytes)[0]
  return content_size


def ensure_compressed_trace_within_bounds(
    data: bytes,
    max_decompressed_bytes: int = MAX_DECOMPRESSED_TRACE_BYTES,
) -> None:
  """Validate compressed protobuf/gzip trace payloads against size limits.

  Uncompressed (non-gzip, non-zstd) bytes are left unchecked so legacy and
  mock payloads keep working. Gzip streams are fully decompressed under a
  hard size cap. Zstd frames are checked via the declared Frame_Content_Size
  in the frame header (what the C++ producer writes) without requiring a
  zstd Python dependency.

  Args:
    data: Raw tool payload bytes (possibly gzip or zstd compressed).
    max_decompressed_bytes: Maximum allowed decompressed size.

  Raises:
    CompressedTraceError: If the payload is a corrupt compressed stream or
      its decompressed size would exceed ``max_decompressed_bytes``.
  """
  if not data:
    return
  if data.startswith(_GZIP_MAGIC):
    _gzip_decompress_bounded(data, max_decompressed_bytes)
    return
  if data.startswith(_ZSTD_MAGIC):
    content_size = _zstd_declared_content_size(data)
    if content_size > max_decompressed_bytes:
      raise CompressedTraceError(
          'decompressed zstd trace exceeds limit of'
          f' {max_decompressed_bytes} bytes (declared {content_size} bytes)'
      )
    return


def decompress_trace_data(
    data: bytes,
    max_decompressed_bytes: int = MAX_DECOMPRESSED_TRACE_BYTES,
) -> bytes:
  """Decompress gzip-compressed trace data with a hard size bound.

  Zstd payloads are size-checked (via frame header) but returned compressed,
  since the plugin path serves zstd protobufs as ``application/octet-stream``
  for frontend/WASM decompression. Uncompressed data is returned as-is.

  Raises:
    CompressedTraceError: On corrupt or oversize compressed input.
  """
  if not data:
    return data
  if data.startswith(_GZIP_MAGIC):
    return _gzip_decompress_bounded(data, max_decompressed_bytes)
  if data.startswith(_ZSTD_MAGIC):
    ensure_compressed_trace_within_bounds(data, max_decompressed_bytes)
    return data
  return data


def _validated_pb_trace_data(raw_data: bytes) -> bytes:
  """Bounds-check compressed pb trace bytes; raise ValueError on failure."""
  try:
    ensure_compressed_trace_within_bounds(raw_data)
  except CompressedTraceError as exc:
    raise ValueError(str(exc)) from exc
  return raw_data


def process_raw_trace(raw_trace: bytes) -> str:
  """Processes raw trace data and returns the UI data."""
  trace = trace_events_old_pb2.Trace()
  trace.ParseFromString(raw_trace)
  return ''.join(trace_events_json.TraceEventsJsonStream(trace))


def xspace_to_tools_data_from_byte_string(
    xspace_byte_list: Sequence[bytes],
    filenames: Sequence[str],
    tool: str,
    params: Mapping[str, Any],
) -> tuple[Any, str]:
  """Helper function for getting an XSpace tool from a bytes string.

  Args:
    xspace_byte_list: A list of byte strings read from a XSpace proto file.
    filenames: Names of the read files.
    tool: A string of tool name.
    params: user input parameters.

  Returns:
    Returns a string of tool data.
  """

  def xspace_wrapper_func(
      xspace_arg: Sequence[bytes],
      tool_arg: str,
      options: Mapping[str, Any],
  ) -> tuple[Any, bool]:
    return _pywrap_profiler_plugin.xspace_to_tools_data_from_byte_string(
        xspace_arg, filenames, tool_arg, options
    )

  return xspace_to_tool_data(
      xspace_byte_list, tool, params, xspace_wrapper_func
  )


def xspace_to_tool_names(xspace_paths: Sequence[Any]) -> list[str]:
  """Converts XSpace to all the available tool names.

  Args:
    xspace_paths: A list of XSpace paths.

  Returns:
    Returns a list of tool names.
  """
  raw_data, success = _pywrap_profiler_plugin.xspace_to_tools_data(
      xspace_paths, 'tool_names'
  )
  if success:
    return raw_data.decode().split(',')
  return []


def xspace_to_tool_data(
    xspace_paths: Sequence[Any],
    tool: str,
    params: Mapping[str, Any],
    xspace_wrapper_func: Callable[..., tuple[Any, bool]] | None = None,
) -> tuple[Any, str]:
  """Converts XSpace to tool data string.

  Args:
    xspace_paths: A list of XSpace paths.
    tool: A string of tool name.
    params: user input parameters.
    xspace_wrapper_func: A callable that takes a list of strings and a tool and
      returns the raw data. If failed, raw data contains the error message.

  Returns:
    Returns a string of tool data and the content type for the response.
  """
  if xspace_wrapper_func is None:
    xspace_wrapper_func = _pywrap_profiler_plugin.xspace_to_tools_data
  if tool.endswith('^'):
    old_tool = tool
    tool = tool[:-1]  # Remove the trailing '^'
    logger.warning(
        'Received old tool format: %s; mapped to new format: %s', old_tool, tool
    )
  data = None
  content_type = 'application/json'
  options = {}
  options['use_saved_result'] = params.get('use_saved_result', True)
  if tool == 'trace_viewer':
    options = dict(params.get('trace_viewer_options', {}))
    options['use_saved_result'] = params.get('use_saved_result', True)
    raw_data, success = xspace_wrapper_func(xspace_paths, tool, options)
    if success:
      if options.get('format') == 'pb':
        data = _validated_pb_trace_data(raw_data)
        content_type = 'application/octet-stream'
      else:
        data = process_raw_trace(raw_data)
  elif tool == 'trace_viewer@':
    options = dict(params.get('trace_viewer_options', {}))
    options['use_saved_result'] = params.get('use_saved_result', True)
    options['hosts'] = params.get('hosts', [])
    raw_data, success = xspace_wrapper_func(xspace_paths, tool, options)
    if success:
      if options.get('format') == 'pb':
        data = _validated_pb_trace_data(raw_data)
        content_type = 'application/octet-stream'
      else:
        data = raw_data
  elif tool == 'overview_page':
    json_data, success = xspace_wrapper_func(xspace_paths, tool, options)
    if success:
      data = json_data
  elif tool == 'input_pipeline_analyzer':
    json_data, success = xspace_wrapper_func(xspace_paths, tool, options)
    if success:
      data = json_data
  elif tool == 'framework_op_stats':
    json_data, success = xspace_wrapper_func(xspace_paths, tool, options)
    if success:
      data = json_data
    # Try legacy tool name: Handle backward compatibility with lower TF version
  elif tool == 'kernel_stats':
    json_data, success = xspace_wrapper_func(xspace_paths, tool, options)
    if success:
      data = json_data
  elif tool == 'memory_profile':
    # Memory profile handles one host at a time.
    assert len(xspace_paths) == 1
    raw_data, success = xspace_wrapper_func(xspace_paths, tool, options)
    if success:
      data = raw_data
  elif tool == 'pod_viewer':
    raw_data, success = xspace_wrapper_func(xspace_paths, tool, options)
    if success:
      data = raw_data
  elif tool == 'op_profile':
    options['group_by'] = params.get('group_by', 'program')
    raw_data, success = xspace_wrapper_func(xspace_paths, tool, options)
    if success:
      data = raw_data
  elif tool == 'hlo_op_profile':
    options['group_by'] = params.get('group_by', 'program')
    raw_data, success = xspace_wrapper_func(xspace_paths, tool, options)
    if success:
      data = raw_data
  elif tool == 'hlo_stats':
    json_data, success = xspace_wrapper_func(xspace_paths, tool, options)
    if success:
      data = json_data
  elif tool == 'roofline_model':
    json_data, success = xspace_wrapper_func(xspace_paths, tool, options)
    if success:
      data = json_data
  elif tool == 'graph_viewer':
    download_hlo_types = ['pb', 'pbtxt', 'json', 'short_txt', 'long_txt']
    graph_html_type = 'graph'
    options = dict(params.get('graph_viewer_options', {}))
    options['use_saved_result'] = params.get('use_saved_result', True)
    raw_data, success = xspace_wrapper_func(xspace_paths, tool, options)
    if success:
      data = raw_data
      content_type = 'text/plain'
      data_type = options.get('type', '')
      if data_type in download_hlo_types:
        content_type = 'application/octet-stream'
      if data_type == graph_html_type:
        content_type = 'text/html'
    else:
      # TODO(tf-profiler) Handle errors for other tools as well,
      # to pass along the error message to client
      if isinstance(raw_data, bytes):
        raw_data = raw_data.decode('utf-8')
      raise ValueError(raw_data)
  elif tool == 'memory_viewer':
    view_memory_allocation_timeline = params.get(
        'view_memory_allocation_timeline', False
    )
    options = {
        'module_name': params.get('module_name'),
        'program_id': params.get('program_id'),
        'view_memory_allocation_timeline': view_memory_allocation_timeline,
        'memory_space': params.get('memory_space', ''),
    }
    raw_data, success = xspace_wrapper_func(xspace_paths, tool, options)
    if success:
      data = raw_data
      if view_memory_allocation_timeline:
        content_type = 'text/html'
  elif tool == 'megascale_stats':
    options = {
        'host_name': params.get('host'),
        'perfetto': params.get('perfetto', False),
    }
    json_data, success = xspace_wrapper_func(xspace_paths, tool, options)
    if success:
      data = json_data
      if options['perfetto']:
        content_type = 'application/octet-stream'
  elif tool == 'inference_profile':
    json_data, success = xspace_wrapper_func(xspace_paths, tool, options)
    if success:
      data = json_data
  elif tool == 'perf_counters':
    json_data, success = xspace_wrapper_func(xspace_paths, tool, options)
    if success:
      data = json_data
  elif tool == 'utilization_viewer':
    json_data, success = xspace_wrapper_func(xspace_paths, tool, options)
    if success:
      data = json_data
  elif tool == 'smart_suggestion':
    options['refresh_suggestion'] = params.get('refresh_suggestion', False)
    json_data, success = xspace_wrapper_func(xspace_paths, tool, options)
    if success:
      data = json_data
  else:
    logger.warning('%s is not a known xplane tool', tool)
  return data, content_type


def json_to_csv_string(json_data: Any) -> str:
  """Converts internal profile JSON format to a CSV string."""
  return csv_writer.json_to_csv_string(json_data)
