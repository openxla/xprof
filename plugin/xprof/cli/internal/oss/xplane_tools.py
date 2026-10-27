"""XPlane-related tools for XProf MCP."""

import collections
from collections.abc import Sequence
import json
import logging
import os
import pathlib
import re
from typing import Any, Iterator

# pylint: disable=g-import-not-at-top
try:
  from xprof import profile_data as profiler  # pyrefly: ignore[missing-import,missing-module-attribute]
except ImportError:
  try:
    from xprof import profile_data as profiler  # pyrefly: ignore[missing-import,missing-module-attribute]
  except ImportError:
    from google3.perftools.accelerators.xprof.api.python.profile_data import profile_data as profiler  # pyrefly: ignore[missing-import,missing-module-attribute]

from xprof.cli.internal import async_overlap
from xprof.cli.internal import decorators
from xprof.cli.internal import xplane_event_query

from . import xprof_client

# XLine names that indicate a capture carries LLO-level detail. Keep in sync
# with cli/internal/google/xplane_tools.py.
_LLO_LINE_NAMES = frozenset({
    "Pallas Primitives",
    "Source Lines",
    "Source code",
    "LLO Instructions",
    "SALU Instructions",
    "VALU Instructions",
    "EUP Instructions",
    "XLU Instructions",
    "VLD Instructions",
    "VST Instructions",
})


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
        xplane_paths = xprof_client.get_client().get_xspace_paths(source)
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
    xspace_paths = client.get_xspace_paths(source)

    for path in xspace_paths:
      with open(path, "rb") as f:
        pd = profiler.ProfileData.from_serialized_xspace(f.read())
      yield from pd.planes
    return

  raise TypeError(f"Unsupported source type: {type(source)}")


@decorators.cached(expire=86_400)
def list_xplane_events(
    session_id: str,
    plane_regex: str = ".*",
    event_regex: str = ".*",
    start_time_ps: int | None = None,
    end_time_ps: int | None = None,
    max_events: int = 100,
    offset: int = 0,
    bypass_cache: bool = False,
    window_semantics: str = "containment",
    min_duration_ps: int | None = None,
    max_duration_ps: int | None = None,
    filter: str = "",  # pylint: disable=redefined-builtin
) -> str:
  """Searches and filters timeline events across XPlanes.

  **Use this** to find specific instances of slow kernels or gaps in
  computation. Supports regex filtering on both the plane (host/device)
  and the event name, containment or overlap time-window semantics,
  duration bounds, and boolean filter expressions.

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
    bypass_cache: Whether to bypass cache and recompute metrics.
    window_semantics: Time-window matching mode: 'containment' (default, events
      fully inside [start_time_ps, end_time_ps]) or 'overlap' (any event
      intersecting the window).
    min_duration_ps: Optional minimum event duration in picoseconds.
    max_duration_ps: Optional maximum event duration in picoseconds.
    filter: Optional boolean filter expression over plane, line, event,
      duration_ps/ns/us/ms, offset_ps/start_ps, and end_ps.

  Returns:
    A JSON-formatted object (not a bare list) with the shape
    ``{"events": [...], "returned": int, "total_matched": int,
    "truncated": bool}``. Each event is
    ``{"plane", "line_id", "event", "offset_ps", "duration_ps"}``.

  Raises:
    FileNotFoundError: If path or session run is not found.
    ValueError: If parameters are invalid.
    RuntimeError: If listing events fails.
  """
  del bypass_cache
  try:
    window_mode = xplane_event_query.validate_window_semantics(window_semantics)
    xplane_event_query.matches_window(
        0, 0, start_time_ps, end_time_ps, window_mode
    )
    xplane_event_query.matches_duration(0, min_duration_ps, max_duration_ps)
    filter_pred = xplane_event_query.compile_filter(filter)
    p_re = re.compile(plane_regex)
    e_re = re.compile(event_regex)

    events = []
    total_matched = 0
    skipped_count = 0

    for plane in iter_planes(session_id):
      if not p_re.search(plane.name):
        continue

      for line in plane.lines:
        line_label = (
            getattr(line, "display_name", "") or getattr(line, "name", "") or ""
        )
        for event in line.events:
          offset_ps = int(event.start_ns * 1000)
          duration_ps = int(event.duration_ns * 1000)

          if not xplane_event_query.matches_window(
              offset_ps,
              duration_ps,
              start_time_ps,
              end_time_ps,
              window_mode,
          ):
            continue
          if not xplane_event_query.matches_duration(
              duration_ps, min_duration_ps, max_duration_ps
          ):
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

          record = xplane_event_query.EventRecord(
              plane=plane.name,
              line=line_label,
              event=event_name,
              offset_ps=offset_ps,
              duration_ps=duration_ps,
              line_id=str(line.name),
          )
          if not filter_pred(record):
            continue

          total_matched += 1

          if skipped_count < offset:
            skipped_count += 1
            continue

          if max_events <= 0 or len(events) < max_events:
            events.append({
                "plane": plane.name,
                "line_id": line.name,
                "event": event_name,
                "offset_ps": offset_ps,
                "duration_ps": duration_ps,
            })

    result_payload = {
        "events": events,
        "returned": len(events),
        "total_matched": total_matched,
        "truncated": total_matched > (len(events) + offset),
    }
    return json.dumps(result_payload, indent=2)

  except (FileNotFoundError, ValueError):
    raise
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.exception(
        "Error listing XPlane events for session_id=%r, plane_regex=%r,"
        " event_regex=%r",
        session_id,
        plane_regex,
        event_regex,
    )
    raise RuntimeError(
        f"Error listing XPlane events for session {session_id}: {e}"
    ) from e


@decorators.cached(expire=86_400)
def aggregate_xplane_events(
    session_id: str,
    plane_regex: str = ".*",
    event_regex: str = ".*",
    bypass_cache: bool = False,
    start_time_ps: int | None = None,
    end_time_ps: int | None = None,
    window_semantics: str = "containment",
    min_duration_ps: int | None = None,
    max_duration_ps: int | None = None,
    filter: str = "",  # pylint: disable=redefined-builtin
    group_by: str | Sequence[str] = "event",
) -> str:
  """Calculates statistical aggregates for matching timeline events.

  **Systemic Analysis.** Use this to determine if a specific kernel type is
  consistently slow or has high variance (noise). Returns count, average,
  min, p50, p90, p99, max, and standard deviation of durations.

  **Examples:**
  - `plane_regex='Device'`, `event_regex='Fusion'`: Aggregate performance of all
  device fusions.
  - `plane_regex='.*'`, `event_regex='collective'`: Check stability of
  cross-replica communication.

  Args:
    session_id: The unique XProf session ID.
    plane_regex: Regex to filter XPlanes.
    event_regex: Regex to filter event names.
    bypass_cache: Whether to bypass cache and recompute metrics.
    start_time_ps: Optional start of time window in picoseconds.
    end_time_ps: Optional end of time window in picoseconds.
    window_semantics: Time-window matching mode ('containment' or 'overlap').
    min_duration_ps: Optional minimum event duration in picoseconds.
    max_duration_ps: Optional maximum event duration in picoseconds.
    filter: Optional boolean filter expression.
    group_by: Grouping dimensions from {'event', 'plane', 'line'} (default
      'event'), as a comma-separated string or a sequence of field names.

  Returns:
    A JSON string containing statistical aggregates grouped by `group_by`.

  Raises:
    FileNotFoundError: If path or session run is not found.
    ValueError: If parameters are invalid.
    RuntimeError: If aggregating events fails.
  """
  del bypass_cache
  try:
    window_mode = xplane_event_query.validate_window_semantics(window_semantics)
    group_fields = xplane_event_query.parse_group_by(group_by)
    xplane_event_query.matches_window(
        0, 0, start_time_ps, end_time_ps, window_mode
    )
    xplane_event_query.matches_duration(0, min_duration_ps, max_duration_ps)
    filter_pred = xplane_event_query.compile_filter(filter)
    p_re = re.compile(plane_regex)
    e_re = re.compile(event_regex)

    total_events_scanned = 0
    max_events_to_scan = 5_000_000

    stats_data: dict[tuple[str, ...], list[int]] = collections.defaultdict(list)

    for plane in iter_planes(session_id):
      if not p_re.search(plane.name):
        continue

      for line in plane.lines:
        line_label = (
            getattr(line, "display_name", "") or getattr(line, "name", "") or ""
        )
        for event in line.events:
          offset_ps = int(event.start_ns * 1000)
          duration_ps = int(event.duration_ns * 1000)

          if not xplane_event_query.matches_window(
              offset_ps,
              duration_ps,
              start_time_ps,
              end_time_ps,
              window_mode,
          ):
            total_events_scanned += 1
            if total_events_scanned > max_events_to_scan:
              break
            continue
          if not xplane_event_query.matches_duration(
              duration_ps, min_duration_ps, max_duration_ps
          ):
            total_events_scanned += 1
            if total_events_scanned > max_events_to_scan:
              break
            continue

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
            record = xplane_event_query.EventRecord(
                plane=plane.name,
                line=line_label,
                event=name_info,
                offset_ps=offset_ps,
                duration_ps=duration_ps,
                line_id=str(line.name),
            )
            if filter_pred(record):
              group_key = xplane_event_query.record_group_key(
                  record, group_fields
              )
              stats_data[group_key].append(duration_ps)

          # Safety check
          total_events_scanned += 1
          if total_events_scanned > max_events_to_scan:
            break
        if total_events_scanned > max_events_to_scan:
          break
      if total_events_scanned > max_events_to_scan:
        break

    results = xplane_event_query.summarize_groups(stats_data, group_fields)
    result_payload = {
        "aggregates": results,
        "returned": len(results),
        "total_matched": len(results),
        "events_scanned": total_events_scanned,
        "truncated": total_events_scanned > max_events_to_scan,
    }
    return json.dumps(result_payload, indent=2)

  except (FileNotFoundError, ValueError):
    raise
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.exception(
        "Error aggregating XPlane events for session_id=%r, plane_regex=%r,"
        " event_regex=%r",
        session_id,
        plane_regex,
        event_regex,
    )
    raise RuntimeError(
        f"Error aggregating XPlane events for session {session_id}: {e}"
    ) from e


def get_xspace_proto(
    session_id: str,
    as_text: bool = False,
    output_path: str | None = None,
    bypass_cache: bool = False,
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
  data = client.get_serialized_xspace(session_id, bypass_cache=bypass_cache)

  if output_path:
    p = pathlib.Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
      f.write(data)
    return str(p)

  # Guard against massive binary dumps (e.g. >10MB) flooding stdout or
  # overflowing AI agent context windows.
  if isinstance(data, bytes) and len(data) > 10 * 1024 * 1024:
    safe_session = re.sub(r"[^a-zA-Z0-9_-]", "_", str(session_id))
    tmp_path = pathlib.Path(f"/tmp/xspace_{safe_session}.pb")
    with open(tmp_path, "wb") as f:
      f.write(data)
    return json.dumps(
        {
            "status": "SAVED_TO_FILE",
            "size_bytes": len(data),
            "size_mib": round(len(data) / (1024 * 1024), 2),
            "file_path": str(tmp_path),
            "message": (
                "XSpace payload exceeded 10 MB. Saved to file to prevent"
                " terminal buffer overflow. Specify --output_path to customize."
            ),
        },
        indent=2,
    )

  return data


def inspect_capture(session_id: str) -> str:
  """Returns plane/line inventory and capture-verification gate flags as JSON."""
  planes_inventory = []
  valid_device = False
  valid_llo = False
  valid_counters = False

  for plane in iter_planes(session_id):
    plane_name = getattr(plane, "name", "") or ""
    lines_list = []
    plane_events = 0

    if "/host:metadata" in plane_name:
      for stat_meta in getattr(plane, "stat_metadata", {}).values():
        if getattr(stat_meta, "name", "") == "llo_proto":
          valid_llo = True
          break
      # A `ProfilePlane` exposes stats as (name, value) tuples, so unlike the
      # 1P `XPlane` proto these carry their own names and must be scanned
      # separately. Keep in sync with cli/internal/google/xplane_tools.py.
      for stat in getattr(plane, "stats", []) or []:
        stat_name = (
            stat[0] if isinstance(stat, tuple) else getattr(stat, "name", "")
        )
        if stat_name == "llo_proto":
          valid_llo = True
          break

    for line in getattr(plane, "lines", []) or []:
      line_name = (
          getattr(line, "display_name", "") or getattr(line, "name", "") or ""
      )
      events = getattr(line, "events", []) or []
      ev_count = len(events)
      plane_events += ev_count
      lines_list.append({"name": line_name, "event_count": ev_count})

      if ev_count > 0:
        if (
            "/device:" in plane_name
            or "TPU:" in plane_name
            or "GPU:" in plane_name
        ) and line_name in ("XLA Modules", "XLA Ops", "XLA TraceMe"):
          valid_device = True
        if (
            line_name in _LLO_LINE_NAMES
            or "MXU" in line_name
            or ("/host:metadata" in plane_name and "llo" in line_name.lower())
        ):
          valid_llo = True
        if (
            "_counters_" in line_name
            or "TC Stats" in line_name
            or "SC Stats" in line_name
            or "CMN Stats" in line_name
        ):
          valid_counters = True

    planes_inventory.append({
        "name": plane_name,
        "lines": lines_list,
        "total_events": plane_events,
    })

  return json.dumps(
      {
          "session_id": str(session_id),
          "planes": planes_inventory,
          "capture_valid_for_device_analysis": valid_device,
          "capture_valid_for_llo_analysis": valid_llo,
          "capture_valid_for_counter_sampling": valid_counters,
      },
      indent=2,
  )


def iter_timeline_events(
    session_id: Any,
    host: str = "",
    bypass_cache: bool = False,
) -> list[async_overlap.TimelineEvent]:
  """Extracts device timeline events from an XSpace for overlap analysis.

  Args:
    session_id: The session ID, local trace file or directory, serialized XSpace
      bytes, or an in-memory ProfileData/XSpace object.
    host: Optional host. Selects one trace file of the session, matched by file
      stem or `get_hosts` name. Only supported for session IDs and paths.
    bypass_cache: Unused; kept for parity with the 1P implementation.

  Returns:
    The events of the `async_overlap.LINE_NAMES` lines. When the session holds
    several trace files and no `host` is given, each event's `host` is its
    trace file stem (or, if two files share a stem, its path relative to the
    files' common directory), so events of different hosts are never merged.

  Raises:
    ValueError: If `host` does not match exactly one trace file, or is given
      for an in-memory source.
  """
  del bypass_cache
  sources: list[tuple[str, Any]] = []
  if isinstance(session_id, (str, os.PathLike)):
    trace = pathlib.Path(session_id)
    if trace.is_file() and (
        not host
        or host
        in (xprof_client.trace_stem(trace), xprof_client.trace_host_name(trace))
    ):
      # The CLI may already have resolved `--host` to this file and passed its
      # `get_hosts` name, which sibling traces can share.
      paths = [str(trace)]
    else:
      paths = xprof_client.get_client().select_paths(
          str(session_id), host=host or None
      )
    sources.extend(zip(xprof_client.trace_labels(paths), paths))
  elif host:
    raise ValueError(
        "host is only supported for a session ID or trace path, not for an"
        f" in-memory {type(session_id).__name__}."
    )
  else:
    sources.append(("", session_id))

  events: list[async_overlap.TimelineEvent] = []
  for label, source in sources:
    for plane in iter_planes(source):
      plane_name = getattr(plane, "name", "") or ""
      for line in getattr(plane, "lines", []) or []:
        line_name = (
            getattr(line, "display_name", "") or getattr(line, "name", "") or ""
        )
        if line_name not in async_overlap.LINE_NAMES:
          continue
        for event in getattr(line, "events", []) or []:
          events.append(
              async_overlap.TimelineEvent(
                  plane=plane_name,
                  line=line_name,
                  name=getattr(event, "name", "") or "",
                  start_ps=int(event.start_ns * 1000),
                  duration_ps=int(event.duration_ns * 1000),
                  host=label,
              )
          )
  return events
