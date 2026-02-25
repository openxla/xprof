# Copyright 2026 The XProf Authors. All Rights Reserved.
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

"""Python wrapper for the profiler plugin C API."""

import ctypes
import glob
import os
from typing import Any, Dict, List, Tuple

# Look for the fully-linked shared library and skip any .abi3.so stubs.
_lib_paths = (
    glob.glob(
        os.path.join(os.path.dirname(__file__), "profiler_plugin_c_api*.so")
    )
    + glob.glob(
        os.path.join(os.path.dirname(__file__), "profiler_plugin_c_api*.pyd")
    )
    + glob.glob(
        os.path.join(os.path.dirname(__file__), "profiler_plugin_c_api*.dylib")
    )
    + glob.glob(
        os.path.join(os.path.dirname(__file__), "profiler_plugin_c_api*.dll")
    )
)

if not _lib_paths:
  raise ImportError(
      f"Could not find profiler_plugin_c_api.* at {os.path.dirname(__file__)}"
  )

_lib = ctypes.CDLL(
    sorted(_lib_paths)[0],
    mode=ctypes.RTLD_LOCAL | getattr(os, "RTLD_DEEPBIND", 0),
)

_lib.InitializeProfiler()
_lib.FreeString.argtypes = [ctypes.c_void_p]
_lib.FreeString.restype = None


def _check_error(err_ptr):
  if err_ptr:
    err_msg = ctypes.string_at(err_ptr).decode("utf-8")
    _lib.FreeString(err_ptr)
    raise RuntimeError(err_msg)


def _pack_options(options):
  """Packs Python options dict into ctypes arrays for the C API."""
  if not options:
    return (None, None, None, None, None, 0)

  num_options = len(options)
  keys = (ctypes.c_char_p * num_options)()
  string_vals = (ctypes.c_char_p * num_options)()
  int_vals = (ctypes.c_int * num_options)()
  bool_vals = (ctypes.c_bool * num_options)()
  types = (ctypes.c_int * num_options)()

  for i, (k, v) in enumerate(options.items()):
    keys[i] = str(k).encode("utf-8")
    if isinstance(v, bool):
      types[i] = 0
      bool_vals[i] = v
    elif isinstance(v, int):
      types[i] = 1
      int_vals[i] = v
    elif isinstance(v, str):
      types[i] = 2
      string_vals[i] = v.encode("utf-8")
    else:
      types[i] = -1

  return keys, string_vals, int_vals, bool_vals, types, num_options


_lib.Trace.argtypes = [
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.c_bool,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_char_p),
    ctypes.POINTER(ctypes.c_char_p),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_bool),
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
]
_lib.Trace.restype = ctypes.c_void_p


def trace(
    service_addr: str,
    logdir: str,
    worker_list: str,
    include_dataset_ops: bool,
    duration_ms: int,
    num_tracing_attempts: int,
    options: Dict[str, Any],
) -> None:
  """Traces the profiler."""
  keys, string_vals, int_vals, bool_vals, types, num_options = _pack_options(
      options
  )
  err = _lib.Trace(
      service_addr.encode() if service_addr else None,
      str(logdir).encode() if logdir else None,
      worker_list.encode() if worker_list else None,
      include_dataset_ops,
      duration_ms,
      num_tracing_attempts,
      keys,
      string_vals,
      int_vals,
      bool_vals,
      types,
      num_options,
  )
  _check_error(err)


_lib.Monitor.argtypes = [
    ctypes.c_char_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_bool,
    ctypes.POINTER(ctypes.c_void_p),
]
_lib.Monitor.restype = ctypes.c_void_p


def monitor(
    service_addr: str,
    duration_ms: int,
    monitoring_level: int,
    display_timestamp: bool,
) -> str:
  """Monitors the profiler."""
  result_ptr = ctypes.c_void_p()
  err = _lib.Monitor(
      service_addr.encode() if service_addr else None,
      duration_ms,
      monitoring_level,
      display_timestamp,
      ctypes.byref(result_ptr),
  )
  _check_error(err)
  if result_ptr:
    res = ctypes.string_at(result_ptr).decode("utf-8")
    _lib.FreeString(result_ptr)
    return res
  return ""


_lib.StartContinuousProfiling.argtypes = [
    ctypes.c_char_p,
    ctypes.POINTER(ctypes.c_char_p),
    ctypes.POINTER(ctypes.c_char_p),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_bool),
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
]
_lib.StartContinuousProfiling.restype = ctypes.c_void_p


def start_continuous_profiling(
    service_addr: str, options: Dict[str, Any]
) -> None:
  """Starts continuous profiling."""
  keys, string_vals, int_vals, bool_vals, types, num_options = _pack_options(
      options
  )
  err = _lib.StartContinuousProfiling(
      service_addr.encode() if service_addr else None,
      keys,
      string_vals,
      int_vals,
      bool_vals,
      types,
      num_options,
  )
  _check_error(err)


_lib.StopContinuousProfiling.argtypes = [ctypes.c_char_p]
_lib.StopContinuousProfiling.restype = ctypes.c_void_p


def stop_continuous_profiling(service_addr: str) -> None:
  err = _lib.StopContinuousProfiling(
      service_addr.encode() if service_addr else None
  )
  _check_error(err)


_lib.GetSnapshot.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
_lib.GetSnapshot.restype = ctypes.c_void_p


def get_snapshot(service_addr: str, logdir: str) -> None:
  err = _lib.GetSnapshot(
      service_addr.encode() if service_addr else None,
      str(logdir).encode() if logdir else None,
  )
  _check_error(err)


_lib.XSpaceToToolsData.argtypes = [
    ctypes.POINTER(ctypes.c_char_p),
    ctypes.c_size_t,
    ctypes.c_char_p,
    ctypes.POINTER(ctypes.c_char_p),
    ctypes.POINTER(ctypes.c_char_p),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_bool),
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.POINTER(ctypes.c_bool),
]
_lib.XSpaceToToolsData.restype = ctypes.c_void_p


def xspace_to_tools_data(
    xspace_paths: List[str], tool_name: str, options: Dict[str, Any] = None
) -> Tuple[bytes, bool]:
  """Converts XSpaces to tools data."""
  keys, string_vals, int_vals, bool_vals, types, num_options = _pack_options(
      options or {}
  )

  c_paths = (ctypes.c_char_p * len(xspace_paths))()
  for i, p in enumerate(xspace_paths):
    c_paths[i] = str(p).encode("utf-8")

  result_ptr = ctypes.c_void_p()
  result_size = ctypes.c_size_t(0)
  success = ctypes.c_bool(False)

  err = _lib.XSpaceToToolsData(
      c_paths,
      len(xspace_paths),
      tool_name.encode() if tool_name else None,
      keys,
      string_vals,
      int_vals,
      bool_vals,
      types,
      num_options,
      ctypes.byref(result_ptr),
      ctypes.byref(result_size),
      ctypes.byref(success),
  )
  _check_error(err)

  res_bytes = b""
  if result_ptr:
    res_bytes = ctypes.string_at(result_ptr, result_size.value)
    _lib.FreeString(result_ptr)

  return (res_bytes, success.value)


_lib.XSpaceToToolsDataFromByteString.argtypes = [
    ctypes.POINTER(ctypes.c_char_p),
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.POINTER(ctypes.c_char_p),
    ctypes.c_size_t,
    ctypes.c_char_p,
    ctypes.POINTER(ctypes.c_char_p),
    ctypes.POINTER(ctypes.c_char_p),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_bool),
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.POINTER(ctypes.c_bool),
]
_lib.XSpaceToToolsDataFromByteString.restype = ctypes.c_void_p


def xspace_to_tools_data_from_byte_string(
    xspace_strings: List[bytes],
    filenames_list: List[str],
    tool_name: str,
    options: Dict[str, Any] = None,
) -> Tuple[bytes, bool]:
  """Converts XSpace byte strings to tools data."""
  keys, string_vals, int_vals, bool_vals, types, num_options = _pack_options(
      options or {}
  )

  num_xspaces = len(xspace_strings)
  if len(filenames_list) != num_xspaces:
    raise ValueError("Lengths of xspace_strings and filenames_list must match")

  c_strings = (ctypes.c_char_p * num_xspaces)()
  c_sizes = (ctypes.c_size_t * num_xspaces)()
  c_paths = (ctypes.c_char_p * num_xspaces)()

  for i in range(num_xspaces):
    c_strings[i] = xspace_strings[i]
    c_sizes[i] = len(xspace_strings[i])
    c_paths[i] = str(filenames_list[i]).encode("utf-8")

  result_ptr = ctypes.c_void_p()
  result_size = ctypes.c_size_t(0)
  success = ctypes.c_bool(False)

  err = _lib.XSpaceToToolsDataFromByteString(
      c_strings,
      c_sizes,
      c_paths,
      num_xspaces,
      tool_name.encode() if tool_name else None,
      keys,
      string_vals,
      int_vals,
      bool_vals,
      types,
      num_options,
      ctypes.byref(result_ptr),
      ctypes.byref(result_size),
      ctypes.byref(success),
  )
  _check_error(err)

  res_bytes = b""
  if result_ptr:
    res_bytes = ctypes.string_at(result_ptr, result_size.value)
    _lib.FreeString(result_ptr)

  return (res_bytes, success.value)


_lib.StartGrpcServer.argtypes = [ctypes.c_int, ctypes.c_int]
_lib.StartGrpcServer.restype = None


def start_grpc_server(port: int, max_concurrent_requests: int) -> None:
  _lib.StartGrpcServer(port, max_concurrent_requests)


_lib.InitializeStubs.argtypes = [ctypes.c_char_p]
_lib.InitializeStubs.restype = None


def initialize_stubs(worker_service_addresses: str) -> None:
  _lib.InitializeStubs(
      worker_service_addresses.encode() if worker_service_addresses else None
  )
