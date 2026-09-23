"""Python wrapper for the profiler plugin C API."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import ctypes
import dataclasses
import enum
import glob
import itertools
import json
import os
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
)

if not _lib_paths:
  raise ImportError(
      f"Could not find profiler_plugin_c_api.* at {os.path.dirname(__file__)}"
  )


def _get_dlopen_mode() -> int:
  """Returns dlopen flags, avoiding RTLD_DEEPBIND under sanitizers."""
  mode = getattr(os, "RTLD_LAZY", 1) | ctypes.RTLD_LOCAL
  # RTLD_DEEPBIND is incompatible with AddressSanitizer/Sanitizers runtime
  # (see https://github.com/google/sanitizers/issues/611).
  is_sanitizer = any(
      var in os.environ
      for var in (
          "ASAN_OPTIONS",
          "MSAN_OPTIONS",
          "TSAN_OPTIONS",
          "HWASAN_OPTIONS",
          "UBSAN_OPTIONS",
      )
  )
  if not is_sanitizer:
    try:
      main_lib = ctypes.CDLL(None)
      is_sanitizer = any(
          hasattr(main_lib, hook)
          for hook in (
              "__asan_init",
              "__msan_init",
              "__tsan_init",
              "__hwasan_init",
              "__ubsan_handle_type_mismatch_v1",
          )
      )
    except (AttributeError, OSError, TypeError):
      pass

  if not is_sanitizer and hasattr(os, "RTLD_DEEPBIND"):
    mode |= os.RTLD_DEEPBIND
  return mode


_lib = ctypes.CDLL(
    sorted(_lib_paths)[0],
    mode=_get_dlopen_mode(),
)
LIB_PATH = sorted(_lib_paths)[0]


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

  # Filter out None values and complex types (lists, tuples, dicts).
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
  packed = _pack_options(options)
  err = _lib.Trace(
      # Marshal absent values to an empty C string, not NULL: the C side
      # strlen()s these, so a None (-> NULL char*) segfaults. worker_list is
      # empty by default for a single-host capture, which is how the "Capture
      # Profile" UI reaches this call.
      (service_addr or "").encode(),
      (logdir or "").encode(),
      (worker_list or "").encode(),
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
  content_ptr = ctypes.c_void_p()
  err = _lib.Monitor(
      (service_addr or "").encode(),
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
  packed = _pack_options(options)
  err = _lib.StartContinuousProfiling(
      (service_addr or "").encode(),
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
  err = _lib.StopContinuousProfiling(
      (service_addr or "").encode()
  )
  _check_error(err)


_lib.GetSnapshot.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
_lib.GetSnapshot.restype = ctypes.c_void_p


def get_snapshot(service_addr: str, logdir: str) -> None:
  err = _lib.GetSnapshot(
      (service_addr or "").encode(),
      (logdir or "").encode(),
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
      (tool_name or "").encode(),
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
      (tool_name or "").encode(),
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
  _lib.StartGrpcServer(port, max_concurrent_requests)


_lib.InitializeStubs.argtypes = [ctypes.c_char_p]
_lib.InitializeStubs.restype = None


def initialize_stubs(worker_service_addresses: str) -> None:
  _lib.InitializeStubs(
      (worker_service_addresses or "").encode()
  )

_lib.BuiltWithEmbedded.argtypes = []
_lib.BuiltWithEmbedded.restype = ctypes.c_bool


def built_with_embedded() -> bool:
  return _lib.BuiltWithEmbedded()


if built_with_embedded():
  _lib.CreateLloAnalysis.argtypes = [ctypes.c_char_p]
  _lib.CreateLloAnalysis.restype = ctypes.c_void_p

  _lib.GetTotalInstructions.argtypes = [ctypes.c_void_p]
  _lib.GetTotalInstructions.restype = ctypes.c_int

  _lib.GetUniqueRegisters.argtypes = [ctypes.c_void_p]
  _lib.GetUniqueRegisters.restype = ctypes.c_int

  _lib.GetNumUniqueOpcodes.argtypes = [ctypes.c_void_p]
  _lib.GetNumUniqueOpcodes.restype = ctypes.c_int

  _lib.GetOpcodeAtIndex.argtypes = [ctypes.c_void_p, ctypes.c_int]
  _lib.GetOpcodeAtIndex.restype = ctypes.c_int

  _lib.GetOpcodeCountAtIndex.argtypes = [ctypes.c_void_p, ctypes.c_int]
  _lib.GetOpcodeCountAtIndex.restype = ctypes.c_int

  _lib.GetLloAnalysisJson.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
  _lib.GetLloAnalysisJson.restype = ctypes.c_char_p

  if hasattr(_lib, "GetLloStaticAnalysisJson"):
    _lib.GetLloStaticAnalysisJson.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_int,
    ]
    _lib.GetLloStaticAnalysisJson.restype = ctypes.c_char_p

  _lib.GetLloDebugString.argtypes = [ctypes.c_void_p]
  _lib.GetLloDebugString.restype = ctypes.c_char_p

  _lib.FreeLloAnalysis.argtypes = [ctypes.c_void_p]
  _lib.FreeLloAnalysis.restype = None


def _run_llo_static_analysis_py(
    xspace_filename: str,
    mode: str = "region_tree",
    hlo_op: str = "",
    bundle: int = 0,
) -> str:
  """Runs LLO static analysis using the pure-Python llo_static_analysis library."""
  try:
    from xprof.cli.internal.llo_static_analysis import bdi_stalls  # pylint: disable=g-import-not-at-top
    from xprof.cli.internal.llo_static_analysis import llo_allocations  # pylint: disable=g-import-not-at-top
    from xprof.cli.internal.llo_static_analysis import llo_bundle_utilization  # pylint: disable=g-import-not-at-top
    from xprof.cli.internal.llo_static_analysis import llo_opcode_stats  # pylint: disable=g-import-not-at-top
    from xprof.cli.internal.llo_static_analysis import llo_region_tree  # pylint: disable=g-import-not-at-top
    from xprof.cli.internal.llo_static_analysis import llo_source_map  # pylint: disable=g-import-not-at-top
    from xprof.cli.internal.llo_static_analysis import register_pressure  # pylint: disable=g-import-not-at-top
    from xprof.cli.internal.llo_static_analysis import target_info  # pylint: disable=g-import-not-at-top
    from xprof.cli.internal.llo_static_analysis import xspace_llo_reader  # pylint: disable=g-import-not-at-top
  except ImportError as exc:
    return json.dumps({
        "status": "ERROR",
        "semantics": "static_modelled_schedule",
        "mode": mode,
        "error": f"llo_static_analysis library unavailable: {exc}",
    })

  valid_modes = (
      "list_ops",
      "region_tree",
      "bundle_util",
      "opcodes",
      "register_pressure",
      "spills",
      "bdi_stalls",
      "target_info",
      "source_for_bundle",
  )
  if mode not in valid_modes:
    return json.dumps({
        "status": "ERROR",
        "semantics": "static_modelled_schedule",
        "mode": mode,
        "error": (
            f"Unknown mode '{mode}'. Valid modes: {', '.join(valid_modes)}."
        ),
    })

  try:
    xspace = xspace_llo_reader.load_xspace(xspace_filename)
  except Exception as exc:  # pylint: disable=broad-exception-caught
    return json.dumps({
        "status": "ERROR",
        "semantics": "static_modelled_schedule",
        "mode": mode,
        "error": str(exc),
    })

  if mode == "list_ops":
    names = xspace_llo_reader.list_hlo_op_names(xspace)
    return json.dumps({
        "status": "OK",
        "semantics": "static_modelled_schedule",
        "semantics_note": (
            "Static / modelled compiler schedule estimate (NOT measured"
            " hardware counters)"
        ),
        "mode": mode,
        "data": {"ops": names},
    })

  matched = (
      xspace_llo_reader.find_llo_modules_for_op(xspace, hlo_op)
      if hlo_op
      else list(xspace_llo_reader.iter_llo_modules(xspace))
  )
  if not matched:
    return json.dumps({
        "status": "ERROR",
        "semantics": "static_modelled_schedule",
        "mode": mode,
        "error": "No LLO module found matching filter",
    })
  module = matched[0]

  if mode == "region_tree":
    raw = llo_region_tree.render_tree_json(module)
  elif mode == "bundle_util":
    paths = llo_region_tree.build_bundle_to_region_path(module)
    raw = llo_bundle_utilization.render_bundle_util_json(
        module, region_path_map=paths
    )
  elif mode == "opcodes":
    raw = llo_opcode_stats.render_opcode_stats_json(module)
  elif mode == "register_pressure":
    raw = register_pressure.render_register_pressure_json(module)
  elif mode == "spills":
    raw = llo_allocations.render_allocations_json(module)
  elif mode == "bdi_stalls":
    raw = bdi_stalls.render_bdi_stalls_json(module)
  elif mode == "target_info":
    raw = target_info.render_target_info_json(module)
  else:
    raw = llo_source_map.render_source_for_bundle(module, int(bundle))

  return json.dumps({
      "status": "OK",
      "semantics": "static_modelled_schedule",
      "semantics_note": (
          "Static / modelled compiler schedule estimate (NOT measured hardware"
          " counters)"
      ),
      "mode": mode,
      "data": json.loads(raw),
  })


if built_with_embedded():

  def analyze_llo(xspace_filename: str, kernel: str = "") -> dict[str, Any]:
    """Analyzes an LLO file."""
    handle = _lib.CreateLloAnalysis(xspace_filename.encode("utf-8"))
    if not handle:
      return {"success": False}
    try:
      kernel_bytes = kernel.encode("utf-8") if kernel else None
      json_str = _lib.GetLloAnalysisJson(handle, kernel_bytes)
      if not json_str:
        return {"success": False}
      result = json.loads(json_str.decode("utf-8"))
      modules = result.get("modules")
      if not isinstance(modules, list):
        return result
      result["source_map_by_kernel"] = {
          (mod.get("kernel_name") or mod.get("hlo_instruction_name", "")): dict(
              mod.get("source_map", {})
          )
          for mod in modules
          if isinstance(mod, dict) and isinstance(mod.get("source_map"), dict)
      }
      if len(modules) == 1 and isinstance(modules[0], dict):
        result["source_map"] = dict(modules[0].get("source_map", {}))
      return result
    finally:
      _lib.FreeLloAnalysis(handle)

  def get_llo_static_analysis_json(
      xspace_filename: str,
      mode: str = "region_tree",
      hlo_op: str = "",
      bundle: int = 0,
  ) -> str:
    """Gets static analysis JSON for an LLO file."""
    if not hasattr(_lib, "GetLloStaticAnalysisJson"):
      return _run_llo_static_analysis_py(
          xspace_filename, mode=mode, hlo_op=hlo_op, bundle=bundle
      )
    handle = _lib.CreateLloAnalysis(xspace_filename.encode("utf-8"))
    if not handle:
      return json.dumps({
          "status": "ERROR",
          "semantics": "static_modelled_schedule",
          "mode": mode,
          "error": "Could not load XSpace or no LLO modules found.",
      })
    try:
      mode_bytes = mode.encode("utf-8") if mode else b"region_tree"
      hlo_op_bytes = hlo_op.encode("utf-8") if hlo_op else b""
      json_str = _lib.GetLloStaticAnalysisJson(
          handle, mode_bytes, hlo_op_bytes, int(bundle)
      )
      if not json_str:
        return json.dumps({
            "status": "ERROR",
            "semantics": "static_modelled_schedule",
            "mode": mode,
            "error": "Failed to produce static analysis JSON.",
        })
      return json_str.decode("utf-8")
    finally:
      _lib.FreeLloAnalysis(handle)

  def get_llo_debug_string(xspace_filename: str) -> str:
    """Gets the debug string of an LLO file."""
    handle = _lib.CreateLloAnalysis(xspace_filename.encode("utf-8"))
    if not handle:
      return ""

    try:
      debug_str = _lib.GetLloDebugString(handle)
      if debug_str:
        return debug_str.decode("utf-8")
      return ""
    finally:
      _lib.FreeLloAnalysis(handle)


else:

  def analyze_llo(xspace_filename: str, kernel: str = "") -> dict[str, Any]:
    del xspace_filename, kernel
    raise NotImplementedError("analyze_llo is not supported in this build")

  def get_llo_static_analysis_json(
      xspace_filename: str,
      mode: str = "region_tree",
      hlo_op: str = "",
      bundle: int = 0,
  ) -> str:
    return _run_llo_static_analysis_py(
        xspace_filename, mode=mode, hlo_op=hlo_op, bundle=bundle
    )

  def get_llo_debug_string(xspace_filename: str) -> str:
    del xspace_filename
    raise NotImplementedError(
        "get_llo_debug_string is not supported in this build"
    )
