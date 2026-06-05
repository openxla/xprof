"""Native profiler extension wrapper using ctypes.

Reuses the profiler_plugin_c_api shared library (already loaded by
_pywrap_profiler_plugin) and provides ProfileData / ProfilePlane /
ProfileLine / ProfileEvent classes for XSpace parsing.

TODO: b/519652306 - Move to using the XLA ProfileData implementation via
nanobind with unified Python ABI once we have a minimum Python version of 3.12.
"""

from __future__ import annotations

from collections.abc import Callable
import ctypes
import os
import types
from typing import NewType
import weakref

from etils import epath

from xprof.convert import _pywrap_profiler_plugin

# --- Private NewTypes for C handles ---
_ProfileDataHandle = NewType("_ProfileDataHandle", int)
_ProfilePlaneHandle = NewType("_ProfilePlaneHandle", int)
_ProfileLineHandle = NewType("_ProfileLineHandle", int)
_ProfileEventHandle = NewType("_ProfileEventHandle", int)


_lib = _pywrap_profiler_plugin._lib  # pylint: disable=protected-access

# --- Register argtypes/restypes for profile_data C API ---

_lib.profile_data_from_bytes.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
_lib.profile_data_from_bytes.restype = ctypes.c_void_p

_lib.profile_data_free.argtypes = [ctypes.c_void_p]
_lib.profile_data_free.restype = None

_lib.profile_data_num_planes.argtypes = [ctypes.c_void_p]
_lib.profile_data_num_planes.restype = ctypes.c_int

_lib.profile_data_get_plane.argtypes = [ctypes.c_void_p, ctypes.c_int]
_lib.profile_data_get_plane.restype = ctypes.c_void_p

_lib.profile_plane_name.argtypes = [ctypes.c_void_p]
_lib.profile_plane_name.restype = ctypes.c_char_p

_lib.profile_plane_num_lines.argtypes = [ctypes.c_void_p]
_lib.profile_plane_num_lines.restype = ctypes.c_int

_lib.profile_plane_get_line.argtypes = [ctypes.c_void_p, ctypes.c_int]
_lib.profile_plane_get_line.restype = ctypes.c_void_p

_lib.profile_plane_num_stats.argtypes = [ctypes.c_void_p]
_lib.profile_plane_num_stats.restype = ctypes.c_int

_lib.profile_plane_get_stat.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_char_p),
    ctypes.POINTER(ctypes.c_char_p),
]
_lib.profile_plane_get_stat.restype = None

_lib.profile_plane_free.argtypes = [ctypes.c_void_p]
_lib.profile_plane_free.restype = None

_lib.profile_line_name.argtypes = [ctypes.c_void_p]
_lib.profile_line_name.restype = ctypes.c_char_p

_lib.profile_line_num_events.argtypes = [ctypes.c_void_p]
_lib.profile_line_num_events.restype = ctypes.c_int

_lib.profile_line_get_event.argtypes = [ctypes.c_void_p, ctypes.c_int]
_lib.profile_line_get_event.restype = ctypes.c_void_p

_lib.profile_line_free.argtypes = [ctypes.c_void_p]
_lib.profile_line_free.restype = None

_lib.profile_event_name.argtypes = [ctypes.c_void_p]
_lib.profile_event_name.restype = ctypes.c_char_p

_lib.profile_event_start_ns.argtypes = [ctypes.c_void_p]
_lib.profile_event_start_ns.restype = ctypes.c_double

_lib.profile_event_duration_ns.argtypes = [ctypes.c_void_p]
_lib.profile_event_duration_ns.restype = ctypes.c_double

_lib.profile_event_num_stats.argtypes = [ctypes.c_void_p]
_lib.profile_event_num_stats.restype = ctypes.c_int

_lib.profile_event_get_stat.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_char_p),
    ctypes.POINTER(ctypes.c_char_p),
]
_lib.profile_event_get_stat.restype = None

_lib.profile_event_free.argtypes = [ctypes.c_void_p]
_lib.profile_event_free.restype = None

# Bind direct references for finalizer closures to prevent None conversions.
# This prevents issues during shutdown.
_free_data = _lib.profile_data_free
_free_plane = _lib.profile_plane_free
_free_line = _lib.profile_line_free
_free_event = _lib.profile_event_free

# --- Helper ---


def _unpack_stat(
    handle: int, *, index: int, get_stat_fn: Callable[..., None]
) -> tuple[str, str]:
  """Unpacks a (name, value) stat from C API and returns a tuple."""
  name_p = ctypes.c_char_p()
  val_p = ctypes.c_char_p()
  get_stat_fn(handle, index, ctypes.byref(name_p), ctypes.byref(val_p))
  # pylint: disable=using-constant-test  # False positives due to ctypes byref modification
  name = name_p.value.decode("utf-8", errors="replace") if name_p.value else ""
  val = val_p.value.decode("utf-8", errors="replace") if val_p.value else ""
  # pylint: enable=using-constant-test
  return name, val


# --- Python wrapper classes ---


class ProfileEvent:
  """A single XEvent from the trace."""

  __slots__ = (
      "_handle",
      "_finalizer",
      "_name",
      "_start_ns",
      "_duration_ns",
      "_stats",
      "__weakref__",
  )

  def __init__(self, handle: _ProfileEventHandle):
    """Initializes the instance."""
    self._handle = handle
    self._finalizer = weakref.finalize(self, _free_event, handle)
    self._name = None
    self._start_ns = None
    self._duration_ns = None
    self._stats = None

  def __repr__(self) -> str:
    """Returns an informative string representation."""
    if self.closed:
      return "ProfileEvent(closed=True)"
    return (
        f"ProfileEvent(name={self.name!r}, start_ns={self.start_ns},"
        f" duration_ns={self.duration_ns})"
    )

  def close(self) -> None:
    """Closes the event and frees internal resources."""
    self._finalizer()
    self._handle = None

  @property
  def closed(self) -> bool:
    """The closed status of this event."""
    return not self._finalizer.alive

  def __enter__(self) -> ProfileEvent:
    """Enters the runtime context for resource management."""
    return self

  def __exit__(
      self,
      exc_type: type[BaseException] | None,
      exc_val: BaseException | None,
      exc_tb: types.TracebackType | None,
  ) -> None:
    """Exits the runtime context and releases C resources."""
    self.close()

  @property
  def name(self) -> str:
    """The name of this event."""
    if self.closed:
      raise ValueError("I/O operation on closed ProfileEvent")
    if self._name is None:
      raw_name = _lib.profile_event_name(self._handle)
      self._name = (
          raw_name.decode("utf-8", errors="replace") if raw_name else ""
      )
    return self._name

  @property
  def start_ns(self) -> float:
    """The start time of this event in nanoseconds."""
    if self.closed:
      raise ValueError("I/O operation on closed ProfileEvent")
    if self._start_ns is None:
      self._start_ns = _lib.profile_event_start_ns(self._handle)
    return self._start_ns

  @property
  def duration_ns(self) -> float:
    """The duration of this event in nanoseconds."""
    if self.closed:
      raise ValueError("I/O operation on closed ProfileEvent")
    if self._duration_ns is None:
      self._duration_ns = _lib.profile_event_duration_ns(self._handle)
    return self._duration_ns

  @property
  def stats(self) -> tuple[tuple[str, str], ...]:
    """The list of (name, value) tuples for each stat on this event."""
    if self.closed:
      raise ValueError("I/O operation on closed ProfileEvent")
    if self._stats is None:
      n = _lib.profile_event_num_stats(self._handle)
      self._stats = tuple(
          _unpack_stat(
              self._handle, index=i, get_stat_fn=_lib.profile_event_get_stat
          )
          for i in range(n)
      )
    return self._stats


class ProfileLine:
  """A single XLine from the trace."""

  __slots__ = ("_handle", "_finalizer", "_name", "_events", "__weakref__")

  def __init__(self, handle: _ProfileLineHandle):
    """Initializes the instance."""
    self._handle = handle
    self._finalizer = weakref.finalize(self, _free_line, handle)
    self._name = None
    self._events = None

  def __repr__(self) -> str:
    """Returns an informative string representation."""
    if self.closed:
      return "ProfileLine(closed=True)"
    return f"ProfileLine(name={self.name!r}, num_events={len(self.events)})"

  def close(self) -> None:
    """Closes the line and frees internal resources."""
    if self.closed:
      return
    if self._events is not None:
      for event in self._events:
        event.close()
    self._finalizer()
    self._handle = None

  @property
  def closed(self) -> bool:
    """The closed status of this line."""
    return not self._finalizer.alive

  def __enter__(self) -> ProfileLine:
    """Enters the runtime context for resource management."""
    return self

  def __exit__(
      self,
      exc_type: type[BaseException] | None,
      exc_val: BaseException | None,
      exc_tb: types.TracebackType | None,
  ) -> None:
    """Exits the runtime context and releases C resources."""
    self.close()

  @property
  def name(self) -> str:
    """The name of this line."""
    if self.closed:
      raise ValueError("I/O operation on closed ProfileLine")
    if self._name is None:
      raw_name = _lib.profile_line_name(self._handle)
      self._name = (
          raw_name.decode("utf-8", errors="replace") if raw_name else ""
      )
    return self._name

  @property
  def events(self) -> tuple[ProfileEvent, ...]:
    """The list of events on this line."""
    if self.closed:
      raise ValueError("I/O operation on closed ProfileLine")
    if self._events is None:
      n = _lib.profile_line_num_events(self._handle)
      self._events = tuple(
          ProfileEvent(
              _ProfileEventHandle(_lib.profile_line_get_event(self._handle, i))
          )
          for i in range(n)
      )
    return self._events


class ProfilePlane:
  """A single XPlane from the trace."""

  __slots__ = (
      "_handle",
      "_finalizer",
      "_name",
      "_lines",
      "_stats",
      "__weakref__",
  )

  def __init__(self, handle: _ProfilePlaneHandle):
    """Initializes the instance."""
    self._handle = handle
    self._finalizer = weakref.finalize(self, _free_plane, handle)
    self._name = None
    self._lines = None
    self._stats = None

  def __repr__(self) -> str:
    """Returns an informative string representation."""
    if self.closed:
      return "ProfilePlane(closed=True)"
    return f"ProfilePlane(name={self.name!r}, num_lines={len(self.lines)})"

  def close(self) -> None:
    """Closes the plane and frees internal resources."""
    if self.closed:
      return
    if self._lines is not None:
      for line in self._lines:
        line.close()
    self._finalizer()
    self._handle = None

  @property
  def closed(self) -> bool:
    """The closed status of this plane."""
    return not self._finalizer.alive

  def __enter__(self) -> ProfilePlane:
    """Enters the runtime context for resource management."""
    return self

  def __exit__(
      self,
      exc_type: type[BaseException] | None,
      exc_val: BaseException | None,
      exc_tb: types.TracebackType | None,
  ) -> None:
    """Exits the runtime context and releases C resources."""
    self.close()

  @property
  def name(self) -> str:
    """The name of this plane."""
    if self.closed:
      raise ValueError("I/O operation on closed ProfilePlane")
    if self._name is None:
      raw_name = _lib.profile_plane_name(self._handle)
      self._name = (
          raw_name.decode("utf-8", errors="replace") if raw_name else ""
      )
    return self._name

  @property
  def lines(self) -> tuple[ProfileLine, ...]:
    """The list of lines on this plane."""
    if self.closed:
      raise ValueError("I/O operation on closed ProfilePlane")
    if self._lines is None:
      n = _lib.profile_plane_num_lines(self._handle)
      self._lines = tuple(
          ProfileLine(
              _ProfileLineHandle(_lib.profile_plane_get_line(self._handle, i))
          )
          for i in range(n)
      )
    return self._lines

  @property
  def stats(self) -> tuple[tuple[str, str], ...]:
    """The list of (name, value) tuples for each stat on this plane."""
    if self.closed:
      raise ValueError("I/O operation on closed ProfilePlane")
    if self._stats is None:
      n = _lib.profile_plane_num_stats(self._handle)
      self._stats = tuple(
          _unpack_stat(
              self._handle, index=i, get_stat_fn=_lib.profile_plane_get_stat
          )
          for i in range(n)
      )
    return self._stats


class ProfileData:
  """Profiling data loaded from an XSpace protobuf.

  Usage:
    with ProfileData.from_file(path) as pd:
      for plane in pd.planes:
        for line in plane.lines:
          for event in line.events:
            print(event.name, event.duration_ns)
  """

  __slots__ = ("_handle", "_finalizer", "_planes", "__weakref__")

  def __init__(self, handle: _ProfileDataHandle):
    """Initializes the instance."""
    self._handle = handle
    self._finalizer = weakref.finalize(self, _free_data, handle)
    self._planes = None

  def __repr__(self) -> str:
    """Returns an informative string representation."""
    if self.closed:
      return "ProfileData(closed=True)"
    return f"ProfileData(num_planes={len(self.planes)}, closed=False)"

  @classmethod
  def from_serialized_xspace(cls, data: bytes) -> ProfileData:
    """Parses profiling data from a serialized XSpace protobuf.

    Args:
      data: The serialized XSpace protobuf bytes.

    Returns:
      A ProfileData instance.

    Raises:
      ValueError: If parsing the XSpace protobuf data fails.
    """
    handle = _lib.profile_data_from_bytes(data, len(data))
    if not handle:
      raise ValueError("Failed to parse XSpace protobuf data")
    return cls(_ProfileDataHandle(handle))

  @classmethod
  def from_file(cls, path: str | os.PathLike[str]) -> ProfileData:
    """Parses profiling data from an XSpace file.

    Args:
      path: The path to the XSpace file.

    Returns:
      A ProfileData instance.

    Raises:
      ValueError: If reading the file fails or if parsing the XSpace protobuf
        data fails.
    """
    file_path = epath.Path(path)
    try:
      data = file_path.read_bytes()
    except Exception as e:
      raise ValueError(f"Failed to read XSpace file at {path!r}") from e

    return cls.from_serialized_xspace(data)

  @property
  def planes(self) -> tuple[ProfilePlane, ...]:
    """The list of planes in this profiling dataset.

    Raises:
      ValueError: If the ProfileData instance has been closed.
    """
    if self.closed:
      raise ValueError("I/O operation on closed ProfileData")
    if self._planes is None:
      n = _lib.profile_data_num_planes(self._handle)
      self._planes = tuple(
          ProfilePlane(
              _ProfilePlaneHandle(_lib.profile_data_get_plane(self._handle, i))
          )
          for i in range(n)
      )
    return self._planes

  def close(self) -> None:
    """Closes the profile data instance and frees internal resources."""
    if self.closed:
      return
    if self._planes is not None:
      for plane in self._planes:
        plane.close()
    self._finalizer()
    self._handle = None

  @property
  def closed(self) -> bool:
    """The closed status of this dataset."""
    return not self._finalizer.alive

  def __enter__(self) -> ProfileData:
    """Enters the runtime context for resource management."""
    return self

  def __exit__(
      self,
      exc_type: type[BaseException] | None,
      exc_val: BaseException | None,
      exc_tb: types.TracebackType | None,
  ) -> None:
    """Exits the runtime context and releases C resources."""
    self.close()
