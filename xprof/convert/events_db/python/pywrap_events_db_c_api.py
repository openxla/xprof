# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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
"""Python ctypes wrapper for the Events DB C API."""

from __future__ import annotations

import ctypes
import ctypes.util
import dataclasses
import enum
import functools
import glob
import itertools
import os

from etils import epath


class ArrowCompressionType(enum.Enum):
  """Compression codec applied to Parquet data pages."""

  # go/keep-sorted start
  SNAPPY = "SNAPPY"
  UNCOMPRESSED = "UNCOMPRESSED"
  ZSTD = "ZSTD"
  # go/keep-sorted end


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class ParquetExportOptions:
  """Configuration options for exporting events DB records to Apache Parquet.

  Attributes:
    max_record_count: If set, at most this many records will be written before
      early stopping.
    batch_size: Number of records buffered before flushing a batch to disk.
    compression_type: Compression codec applied to Parquet data pages.
    compression_level: Compressor-specific compression level.
  """

  max_record_count: int | None = None
  batch_size: int | None = None
  compression_type: ArrowCompressionType | None = None
  compression_level: int | None = None

  def __post_init__(self) -> None:
    """Validate the options."""
    if self.max_record_count is not None and self.max_record_count <= 0:
      raise ValueError(
          f"max_record_count must be positive, got {self.max_record_count}"
      )
    if self.batch_size is not None and self.batch_size <= 0:
      raise ValueError(f"batch_size must be positive, got {self.batch_size}")
    if self.compression_level is not None and self.compression_type is None:
      raise ValueError("compression_level requires compression_type to be set.")


def _find_lib_paths() -> tuple[str, ...]:
  """Discovers available paths for the Events DB C API shared library.

  Searches for the compiled shared library (`pywrap_events_db_c_api` /
  `events_db_c_api` `.so`, `.pyd`, `.dylib`, or `.dll`).

  Returns:
    A tuple of existing filesystem paths where the library was found.
  """
  paths = tuple(
      p
      for p in itertools.chain.from_iterable(
          glob.glob(os.path.join(os.path.dirname(__file__), pattern))
          for pattern in (
              "*pywrap_events_db_c_api*.so",
              "*pywrap_events_db_c_api*.pyd",
              "*pywrap_events_db_c_api*.dylib",
              "*pywrap_events_db_c_api*.dll",
              "*events_db_c_api*.so",
              "*events_db_c_api*.pyd",
              "*events_db_c_api*.dylib",
              "*events_db_c_api*.dll",
          )
      )
  )
  if paths:
    return paths

  try:
    from python.runfiles import runfiles  # pylint: disable=g-import-not-at-top

    runfiles_inst = runfiles.Create()
  except (ImportError, ModuleNotFoundError, IOError, AttributeError):
    runfiles_inst = None

  if runfiles_inst:
    for runfiles_path in (
        "org_xprof/xprof/convert/events_db/python/libpywrap_events_db_c_api.so",
        "org_xprof/xprof/convert/events_db/python/pywrap_events_db_c_api.so",
    ):
      resolved = runfiles_inst.Rlocation(runfiles_path)
      if resolved and os.path.exists(resolved):
        return (resolved,)

  found = ctypes.util.find_library("pywrap_events_db_c_api")
  if found and os.path.exists(found):
    return (found,)

  return ()


def _get_dlopen_mode() -> int:
  """Returns dlopen flags for loading the C API library."""
  return getattr(os, "RTLD_LAZY", 1) | getattr(ctypes, "RTLD_GLOBAL", 0)


@functools.cache
def _get_lib() -> ctypes.CDLL:
  """Loads and initializes the Events DB C API shared library."""
  lib_paths: tuple[str, ...] = _find_lib_paths()

  if not lib_paths:
    raise ImportError(
        "Could not find pywrap_events_db_c_api.* at "
        f"{os.path.dirname(__file__)!r}"
    )

  lib = ctypes.CDLL(min(lib_paths), mode=_get_dlopen_mode())

  lib.XProfEventsDbFreeString.argtypes = [ctypes.c_void_p]
  lib.XProfEventsDbFreeString.restype = None

  lib.XProfEventsDbXSpaceToParquet.argtypes = [
      ctypes.c_char_p,
      ctypes.c_char_p,
      ctypes.c_uint32,
      ctypes.c_char_p,
      ctypes.c_int,
      ctypes.c_uint64,
      ctypes.POINTER(ctypes.c_void_p),
  ]
  lib.XProfEventsDbXSpaceToParquet.restype = ctypes.c_bool
  return lib


def _get_and_free_error(
    lib: ctypes.CDLL, error_ptr: ctypes.c_void_p
) -> str | None:
  """Extracts error message from pointer and frees C-allocated memory."""
  if not error_ptr.value:
    return None
  msg = ctypes.string_at(error_ptr.value).decode("utf-8")
  lib.XProfEventsDbFreeString(error_ptr)
  return msg


def xspace_to_parquet(
    *,
    input_path: epath.PathLike,
    output_path: epath.PathLike,
    options: ParquetExportOptions | None = None,
) -> None:
  """Converts an XSpace trace file to an Events DB Parquet file.

  Args:
    input_path: Path to the input XSpace/trace file (e.g. `.xplane.pb`).
    output_path: Destination path for the generated `.parquet` file.
    options: Optional `ParquetExportOptions` configuring batch size, compression
      codec, compression level, and maximum record count.

  Raises:
    RuntimeError: If the C API conversion fails.
    ValueError: If argument values are invalid.
  """
  if not input_path:
    raise ValueError("input_path must not be null or empty.")
  if not output_path:
    raise ValueError("output_path must not be null or empty.")

  opts = options or ParquetExportOptions()
  c_input_path = os.fspath(input_path).encode("utf-8")
  c_output_path = os.fspath(output_path).encode("utf-8")
  c_batch_size = opts.batch_size if opts.batch_size is not None else 0
  c_compression_level = (
      opts.compression_level if opts.compression_level is not None else -1
  )
  c_max_record_count = (
      opts.max_record_count if opts.max_record_count is not None else 0
  )

  c_compression_type: bytes | None = None
  if opts.compression_type is not None:
    c_compression_type = str(opts.compression_type.value).encode("utf-8")

  lib = _get_lib()
  error_ptr = ctypes.c_void_p()
  success = lib.XProfEventsDbXSpaceToParquet(
      c_input_path,
      c_output_path,
      c_batch_size,
      c_compression_type,
      c_compression_level,
      c_max_record_count,
      ctypes.byref(error_ptr),
  )

  err_msg = _get_and_free_error(lib, error_ptr)
  if not success:
    if err_msg:
      raise RuntimeError(err_msg)
    raise RuntimeError("XProfEventsDbXSpaceToParquet failed.")


__all__ = [
    # go/keep-sorted start
    "ArrowCompressionType",
    "ParquetExportOptions",
    "xspace_to_parquet",
    # go/keep-sorted end
]
