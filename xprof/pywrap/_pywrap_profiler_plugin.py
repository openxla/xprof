"""Python wrapper for the profiler plugin C API."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import ctypes
import dataclasses
import enum
import glob
import itertools
import os
import threading
from typing import Any


class OptionType(enum.IntEnum):
  BOOLEAN = 0
  INTEGER = 1
  STRING = 2


@dataclasses.dataclass
class PackedOptions:
  """Packed options for the C API.

  Attributes:
    keys: Ctypes array of option keys.
    string_vals: Ctypes array of string values.
    int_vals: Ctypes array of int values.
    bool_vals: Ctypes array of bool values.
    types: Ctypes array of option types.
    num_options: Number of options.
    keep_alive: List of bytes objects to keep alive for the C API.
  """

  keys: ctypes.Array[ctypes.c_char_p] | None
  string_vals: ctypes.Array[ctypes.c_char_p] | None
  int_vals: ctypes.Array[ctypes.c_int] | None
  bool_vals: ctypes.Array[ctypes.c_bool] | None
  types: ctypes.Array[ctypes.c_int] | None
  num_options: int
  keep_alive: list[bytes]


# Look for the fully-linked shared library and skip any .abi3.so stubs.
_lib_paths = tuple(
    p
    for p in itertools.chain.from_iterable(
        glob.glob(os.path.join(os.path.dirname(__file__), pattern))
        for pattern in (
            "profiler_plugin_c_api*.so",
            "profiler_plugin_c_api*.pyd",
            "profiler_plugin_c_api*.dylib",
            "profiler_plugin_c_api*.dll",
        )
    )
    if ".abi3." not in p
)

if not _lib_paths:
  raise ImportError(
      f"Could not find profiler_plugin_c_api.* at {os.path.dirname(__file__)}"
  )

_lib = ctypes.CDLL(
    sorted(_lib_paths)[0],
    mode=ctypes.RTLD_LOCAL | getattr(os, "RTLD_DEEPBIND", 0),
)

_initialized_lock = threading.Lock()
_initialized = False


def _ensure_initialized() -> None:
  global _initialized
  if not _initialized:
    with _initialized_lock:
      if not _initialized:
        _lib.InitializeProfiler()
        _initialized = True


_lib.FreeString.argtypes = [ctypes.c_void_p]
_lib.FreeString.restype = None


def _check_error(err_ptr: int | None) -> None:
  if err_ptr:
    err_msg = ctypes.string_at(err_ptr).decode("utf-8")
    _lib.FreeString(err_ptr)
    raise RuntimeError(err_msg)


def _pack_options(options: Mapping[str, Any] | None) -> PackedOptions:
  """Packs Python options dict into ctypes arrays for the C API.

  Args:
    options: Dictionary of options to pass to the C API.

  Returns:
    A PackedOptions dataclass containing ctypes arrays and keep-alive list.
  """
  if not options:
    return PackedOptions(
        keys=None,
        string_vals=None,
        int_vals=None,
        bool_vals=None,
        types=None,
        num_options=0,
        keep_alive=[],
    )

  # Filter out None values and complex types (lists, tuples, dicts), matching
  # the old pybind11 behavior of ignoring them.
  options = {
      k: v
      for k, v in options.items()
      if v is not None and not isinstance(v, (list, tuple, dict))
  }
  if not options:
    return PackedOptions(
        keys=None,
        string_vals=None,
        int_vals=None,
        bool_vals=None,
        types=None,
        num_options=0,
        keep_alive=[],
    )

  num_options = len(options)
  keys = (ctypes.c_char_p * num_options)()
  string_vals = (ctypes.c_char_p * num_options)()
  int_vals = (ctypes.c_int * num_options)()
  bool_vals = (ctypes.c_bool * num_options)()
  types = (ctypes.c_int * num_options)()
  keep_alive = []

  for i, (k, v) in enumerate(options.items()):
    key_bytes = k.encode("utf-8")
    keep_alive.append(key_bytes)
    keys[i] = key_bytes
    if isinstance(v, bool):
      types[i] = OptionType.BOOLEAN
      bool_vals[i] = v
    elif isinstance(v, int):
      types[i] = OptionType.INTEGER
      int_vals[i] = v
    elif isinstance(v, str):
      types[i] = OptionType.STRING
      val_bytes = v.encode("utf-8")
      keep_alive.append(val_bytes)
      string_vals[i] = val_bytes
    else:
      raise TypeError(f"Unsupported option type for key {k!r}: {type(v)!r}")

  return PackedOptions(
      keys, string_vals, int_vals, bool_vals, types, num_options, keep_alive
  )


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
    options: Mapping[str, Any] | None = None,
) -> None:
  """Traces the profiler.

  Args:
    service_addr: Address of the profiler service.
    logdir: Directory to save the profile.
    worker_list: List of workers to profile.
    include_dataset_ops: Whether to include dataset ops.
    duration_ms: Duration of the trace in milliseconds.
    num_tracing_attempts: Number of tracing attempts.
    options: Dictionary of options to pass to the C API.
  """
  _ensure_initialized()
  packed = _pack_options(options)
  err = _lib.Trace(
      service_addr.encode() if service_addr else None,
      logdir.encode() if logdir else None,
      worker_list.encode() if worker_list else None,
      include_dataset_ops,
      duration_ms,
      num_tracing_attempts,
      packed.keys,
      packed.string_vals,
      packed.int_vals,
      packed.bool_vals,
      packed.types,
      packed.num_options,
  )
  _check_error(err)
  # Keep Python objects alive until the C API call completes.
  _ = packed.keep_alive


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
  """Monitors the profiler.

  Args:
    service_addr: Address of the profiler service.
    duration_ms: Duration of the monitoring in milliseconds.
    monitoring_level: Monitoring level.
    display_timestamp: Whether to display timestamps.

  Returns:
    The monitoring results as a string.
  """
  _ensure_initialized()
  content_ptr = ctypes.c_void_p()
  err = _lib.Monitor(
      service_addr.encode() if service_addr else None,
      duration_ms,
      monitoring_level,
      display_timestamp,
      ctypes.byref(content_ptr),
  )
  _check_error(err)
  if not content_ptr:
    return ""

  res = ctypes.string_at(content_ptr).decode("utf-8")
  _lib.FreeString(content_ptr)
  return res


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
    service_addr: str,
    options: Mapping[str, Any] | None = None,
) -> None:
  """Starts continuous profiling.

  Args:
    service_addr: Address of the profiler service.
    options: Dictionary of options.
  """
  _ensure_initialized()
  packed = _pack_options(options)
  err = _lib.StartContinuousProfiling(
      service_addr.encode() if service_addr else None,
      packed.keys,
      packed.string_vals,
      packed.int_vals,
      packed.bool_vals,
      packed.types,
      packed.num_options,
  )
  _check_error(err)
  # Keep Python objects alive until the C API call completes.
  _ = packed.keep_alive


_lib.StopContinuousProfiling.argtypes = [ctypes.c_char_p]
_lib.StopContinuousProfiling.restype = ctypes.c_void_p


def stop_continuous_profiling(service_addr: str) -> None:
  _ensure_initialized()
  err = _lib.StopContinuousProfiling(
      service_addr.encode() if service_addr else None
  )
  _check_error(err)


_lib.GetSnapshot.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
_lib.GetSnapshot.restype = ctypes.c_void_p


def get_snapshot(service_addr: str, logdir: str) -> None:
  _ensure_initialized()
  err = _lib.GetSnapshot(
      service_addr.encode() if service_addr else None,
      logdir.encode() if logdir else None,
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
    xspace_paths: Sequence[os.PathLike[str]],
    tool_name: str,
    options: Mapping[str, Any] | None = None,
) -> tuple[bytes, bool]:
  """Converts XSpaces to tools data.

  Args:
    xspace_paths: List of XSpace paths.
    tool_name: Name of the tool.
    options: Dictionary of options.

  Returns:
    A tuple of (result_bytes, success_flag), where result_bytes is the
    tools data as bytes, and success_flag is True if the conversion was
    successful.
  """
  _ensure_initialized()
  packed = _pack_options(options or {})

  c_paths = (ctypes.c_char_p * len(xspace_paths))()
  paths_keep_alive = [str(p).encode("utf-8") for p in xspace_paths]
  for i, path_bytes in enumerate(paths_keep_alive):
    c_paths[i] = path_bytes

  result_ptr = ctypes.c_void_p()
  result_size = ctypes.c_size_t(0)
  success = ctypes.c_bool(False)

  err = _lib.XSpaceToToolsData(
      c_paths,
      len(xspace_paths),
      tool_name.encode() if tool_name else None,
      packed.keys,
      packed.string_vals,
      packed.int_vals,
      packed.bool_vals,
      packed.types,
      packed.num_options,
      ctypes.byref(result_ptr),
      ctypes.byref(result_size),
      ctypes.byref(success),
  )
  _check_error(err)
  # Keep Python objects alive until the C API call completes.
  _ = packed.keep_alive
  _ = paths_keep_alive

  if not result_ptr:
    return b"", success.value

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
    xspace_strings: Sequence[bytes],
    filenames_list: Sequence[str],
    tool_name: str,
    options: Mapping[str, Any] | None = None,
) -> tuple[bytes, bool]:
  """Converts XSpace byte strings to tools data.

  Args:
    xspace_strings: List of XSpace byte strings.
    filenames_list: List of corresponding filenames.
    tool_name: Name of the tool.
    options: Dictionary of options.

  Returns:
    A tuple of (result_bytes, success_flag).
  """
  _ensure_initialized()
  packed = _pack_options(options or {})

  num_xspaces = len(xspace_strings)
  if len(filenames_list) != num_xspaces:
    raise ValueError("Lengths of xspace_strings and filenames_list must match")

  c_strings = (ctypes.c_char_p * num_xspaces)()
  c_sizes = (ctypes.c_size_t * num_xspaces)()
  c_paths = (ctypes.c_char_p * num_xspaces)()
  paths_keep_alive = []

  for i, (string, path) in enumerate(zip(xspace_strings, filenames_list)):
    c_strings[i] = string
    c_sizes[i] = len(string)
    path_bytes = path.encode("utf-8")
    paths_keep_alive.append(path_bytes)
    c_paths[i] = path_bytes

  result_ptr = ctypes.c_void_p()
  result_size = ctypes.c_size_t(0)
  success = ctypes.c_bool(False)

  err = _lib.XSpaceToToolsDataFromByteString(
      c_strings,
      c_sizes,
      c_paths,
      num_xspaces,
      tool_name.encode() if tool_name else None,
      packed.keys,
      packed.string_vals,
      packed.int_vals,
      packed.bool_vals,
      packed.types,
      packed.num_options,
      ctypes.byref(result_ptr),
      ctypes.byref(result_size),
      ctypes.byref(success),
  )
  _check_error(err)
  # Keep Python objects alive until the C API call completes.
  _ = packed.keep_alive
  _ = paths_keep_alive

  if not result_ptr:
    return b"", success.value

  res_bytes = ctypes.string_at(result_ptr, result_size.value)
  _lib.FreeString(result_ptr)
  return (res_bytes, success.value)


_lib.StartGrpcServer.argtypes = [ctypes.c_int, ctypes.c_int]
_lib.StartGrpcServer.restype = None


def start_grpc_server(port: int, max_concurrent_requests: int) -> None:
  _ensure_initialized()
  _lib.StartGrpcServer(port, max_concurrent_requests)


_lib.InitializeStubs.argtypes = [ctypes.c_char_p]
_lib.InitializeStubs.restype = None


def initialize_stubs(worker_service_addresses: str) -> None:
  _ensure_initialized()
  _lib.InitializeStubs(
      worker_service_addresses.encode() if worker_service_addresses else None
  )
