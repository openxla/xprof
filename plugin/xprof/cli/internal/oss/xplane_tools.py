"""XPlane-related tools for XProf MCP."""

import collections
import json
import logging
import os
import pathlib
import re
import statistics
from typing import Any, Iterator

# pylint: disable=g-import-not-at-top
try:
  from xprof import profile_data as profiler  # pytype: disable=import-error  # pyrefly: ignore[missing-import]
except ImportError:
  try:
    from xprof import profile_data as profiler  # pytype: disable=import-error  # pyrefly: ignore[missing-import]
  except ImportError:
    from google3.perftools.accelerators.xprof.api.python.profile_data import profile_data as profiler  # pytype: disable=import-error  # pyrefly: ignore[missing-import]

from xprof.cli.internal import decorators

from . import xprof_client


def iter_planes(source: Any) -> Iterator[Any]:
  """Yields all XPlanes from a session ID, file path, or in-memory object.

  Supports polymorphic inputs to eliminate repeated filesystem I/O and JSON
  conversions across client loops:

  - str session_id (e.g., "xid/..."): Fetches from XProf server.
  - str path or os.PathLike: Reads .xplane.pb files from a directory or file.
  - bytes: Deserializes as a serialized XSpace proto.
  - ProfileData / XSpace / any object with .planes: Yields planes directly.

  Args:
    source: The XProf session ID, local file/directory path, serialized XSpace
      bytes, or an in-memory ProfileData/XSpace object.

  Yields:
    Each XPlane proto found in the session's XSpace.

  Raises:
    FileNotFoundError: If source is an absolute path (starts with "/") that
      does not exist on the local filesystem, or if the run directory or xspace
      files cannot be found.
    TypeError: If source is not a supported type.
    ValueError: If a session ID is provided but logdir has not been set.
  """
  # Handle in-memory objects (ProfileData, XSpace, etc.) with .planes attribute.
  if hasattr(source, "planes"):
    yield from source.planes
    return

  # Handle raw bytes (serialized XSpace proto).
  if isinstance(source, bytes):
    pd = profiler.ProfileData.from_serialized_xspace(source)
    yield from pd.planes
    return

  # Handle os.PathLike by converting to string.
  if isinstance(source, os.PathLike):
    source = str(source)

  # Handle string inputs.
  if isinstance(source, str):
    # If it's a local path (file or directory).
    if (
        os.path.exists(source)
        or source.startswith("/")
        or source.endswith((".xplane.pb", ".xspace.pb"))
    ):
      if not os.path.exists(source):
        raise FileNotFoundError(f"Path does not exist: {source!r}")
      if os.path.isdir(source):
        xplane_paths = []
        for pattern in ("**/*.xplane.pb", "**/*.xspace.pb"):
          xplane_paths.extend(pathlib.Path(source).glob(pattern))
        xplane_paths = sorted(set(xplane_paths))
        if not xplane_paths:
          raise FileNotFoundError(
              "No .xplane.pb or .xspace.pb files found in directory:"
              f" {source!r}"
          )
        for path in xplane_paths:
          with open(path, "rb") as f:
            pd = profiler.ProfileData.from_serialized_xspace(f.read())
          yield from pd.planes
      elif os.path.isfile(source):
        with open(source, "rb") as f:
          pd = profiler.ProfileData.from_serialized_xspace(f.read())
        yield from pd.planes
      return

    # Otherwise, treat as session_id and fetch from XProf server.
    client = xprof_client.get_client()
    run_dir = client.get_run_dir(source)
    xspace_paths = client.get_xspace_paths(run_dir)

    for path in xspace_paths:
      with open(path, "rb") as f:
        pd = profiler.ProfileData.from_serialized_xspace(f.read())
      yield from pd.planes
    return

  raise TypeError(f"Unsupported source type: {type(source)}")


@decorators.cached(expire=86_400)
def list_xplane_events(
    session_id: str,
    *,
    plane_regex: str = ".*",
    event_regex: str = ".*",
    start_time_ps: int | None = None,
    end_time_ps: int | None = None,
    max_events: int = 100,
    offset: int = 0,
) -> str:
  """Searches and filters timeline events across XPlanes.

  **Use this** to find specific instances of slow kernels or gaps in
  computation. Supports regex filtering on both the plane (host/device)
  and the event name.

  **Examples:**
  - `plane_regex='Device.*'`, `event_regex='Fusion.*'`: Find all device-side
  fusion kernels.
  - `plane_regex='host.*'`, `event_regex='.*Wait.*'`: Find host-side
  synchronization waits.
  - `plane_regex='PCIe'`, `event_regex='Copy.*'`: Find HBM/PCIe data transfers.

  Args:
    session_id: The unique XProf session ID.
    plane_regex: Regex to filter XPlanes (e.g., 'Device.*').
    event_regex: Regex to filter event names (e.g., 'Fusion.*').
    start_time_ps: Filter by starting time in picoseconds.
    end_time_ps: Filter by ending time in picoseconds.
    max_events: Limit results to this count (default 100).
    offset: Skip this many events before returning (useful for pagination).

  Returns:
    A JSON-formatted list of matching timeline events.
  """
  try:
    events = []
    skipped_count = 0

    p_re = re.compile(plane_regex)
    e_re = re.compile(event_regex)

    for plane in iter_planes(session_id):
      if not p_re.search(plane.name):
        continue

      for line in plane.lines:
        for event in line.events:
          # Check start/end time
          offset_ps = int(event.start_ns * 1000)
          duration_ps = int(event.duration_ns * 1000)

          if start_time_ps is not None and offset_ps < start_time_ps:
            continue
          if end_time_ps is not None:
            event_end_ps = offset_ps + duration_ps
            if event_end_ps > end_time_ps:
              continue

          event_name = event.name

          # If name is a simple number, try to find a better name in stats
          if event_name.isdigit():
            for stat in event.stats:
              stat_name, stat_val = stat
              if stat_name in ("msg", "message", "annotation", "label"):
                if stat_val:
                  event_name = str(stat_val)
                  break

          if not e_re.search(event_name):
            continue

          if skipped_count < offset:
            skipped_count += 1
            continue

          events.append({
              "plane": plane.name,
              "line_id": line.name,
              "event": event_name,
              "offset_ps": offset_ps,
              "duration_ps": duration_ps,
          })

          if len(events) >= max_events:
            break
        if len(events) >= max_events:
          break
      if len(events) >= max_events:
        break

    return json.dumps(events, indent=2)

  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.exception(
        "Error listing XPlane events for session_id=%r, plane_regex=%r,"
        " event_regex=%r",
        session_id,
        plane_regex,
        event_regex,
    )
    return f"Error listing XPlane events: {e!r}"


@decorators.cached(expire=86_400)
def aggregate_xplane_events(
    session_id: str,
    plane_regex: str = ".*",
    event_regex: str = ".*",
) -> str:
  """Calculates statistical aggregates for matching timeline events.

  **Systemic Analysis.** Use this to determine if a specific kernel type is
  consistently slow or has high variance (noise). Returns count, average,
  min, max, and standard deviation of durations.

  **Examples:**
  - `plane_regex='Device'`, `event_regex='Fusion'`: Aggregate performance of all
  device fusions.
  - `plane_regex='.*'`, `event_regex='collective'`: Check stability of
  cross-replica communication.

  Args:
    session_id: The unique XProf session ID.
    plane_regex: Regex to filter XPlanes.
    event_regex: Regex to filter event names.

  Returns:
    A JSON string containing statistical aggregates grouped by event name.
  """
  try:
    p_re = re.compile(plane_regex)
    e_re = re.compile(event_regex)

    total_events_scanned = 0
    max_events_to_scan = 500_000

    stats_data = collections.defaultdict(list)

    for plane in iter_planes(session_id):
      if not p_re.search(plane.name):
        continue

      for line in plane.lines:
        for event in line.events:
          name_info = event.name

          # If name is a simple number, try to find a better name in stats
          if name_info.isdigit():
            for stat in event.stats:
              stat_name, stat_val = stat
              # Common stat names that might contain the real event name
              if stat_name in ("msg", "message", "annotation", "label"):
                if stat_val:
                  name_info = str(stat_val)
                  break

          if e_re.search(name_info):
            stats_data[name_info].append(int(event.duration_ns * 1000))

          # Safety check
          total_events_scanned += 1
          if total_events_scanned > max_events_to_scan:
            break
        if total_events_scanned > max_events_to_scan:
          break
      if total_events_scanned > max_events_to_scan:
        break

    # Calculate stats
    results = []
    for name, durations in stats_data.items():
      count = len(durations)
      total_duration = sum(durations)
      avg_duration = total_duration / count
      results.append({
          "event": name,
          "count": count,
          "total_duration_ps": total_duration,
          "avg_duration_ps": avg_duration,
          "min_duration_ps": min(durations),
          "max_duration_ps": max(durations),
          "std_dev_ps": statistics.stdev(durations) if count > 1 else 0.0,
      })

    results.sort(key=lambda x: x["total_duration_ps"], reverse=True)
    return json.dumps(results, indent=2)

  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.exception(
        "Error aggregating XPlane events for session_id=%r, plane_regex=%r,"
        " event_regex=%r",
        session_id,
        plane_regex,
        event_regex,
    )
    return f"Error aggregating XPlane events: {e!r}"


def get_xspace_proto(
    session_id: str,
    as_text: bool = False,
    output_path: str | None = None,
    **kwargs,
) -> str | bytes:
  """Returns or saves the serialized XSpace proto for a session."""
  del kwargs
  if as_text:
    raise NotImplementedError(
        "as_text=True is not supported in OSS because xplane_pb2 is not"
        " exposed."
    )

  client = xprof_client.get_client()
  data = client.get_serialized_xspace(session_id)

  if output_path:
    p = pathlib.Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
      f.write(data)
    return str(p)

  return data
