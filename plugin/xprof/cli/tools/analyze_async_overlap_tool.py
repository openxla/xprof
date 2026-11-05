"""Tool to analyze async XLA op overlap with compute and cross-engine concurrency."""

import json
from typing import Any, Literal

from xprof.cli.internal import async_overlap
from xprof.cli.internal import async_overlap_sql
from xprof.cli.internal import decorators

from xprof.cli.internal.oss import events_db_sql
from xprof.cli.internal.oss import xplane_tools

_EVENTS_DB_NOTE = (
    "Computed in SQL over the Events DB. Windows always come from"
    " <op>-start / <op>-done pairs, times have nanosecond resolution, and"
    " cross-engine concurrency is not computed (use backend='xplane')."
)


def _analyze_events_db(
    session_id: str,
    host: str,
    hidden_threshold: float,
    exposed_threshold: float,
    top_k: int,
    bypass_cache: bool,
) -> dict[str, Any]:
  """Runs the async overlap query on the session's Events DB."""
  host_column = events_db_sql.HOST_COLUMN
  host_param = "host" if host and host_column else None
  sql = async_overlap_sql.build_query(
      events_db_sql.DIALECT,
      hidden_threshold=hidden_threshold,
      exposed_threshold=exposed_threshold,
      top_k=top_k,
      host_column=host_column,
      host_param=host_param,
  )
  rows = events_db_sql.run_query(
      session_id,
      sql,
      parameters={host_param: host} if host_param else None,
      bypass_cache=bypass_cache,
  )
  result = async_overlap_sql.assemble(
      rows,
      hidden_threshold=hidden_threshold,
      exposed_threshold=exposed_threshold,
  )
  result["note"] = _EVENTS_DB_NOTE
  if not rows:
    result["status"] = "NO_DEVICE_TIMELINE_EVENTS"
    result["reason"] = (
        "No async op start or done events were found in the XLA Ops or"
        " Sparse Core Ops categories of the Events DB."
    )
  else:
    result["status"] = "OK"
  return result


def _analyze_xplane(
    session_id: str,
    host: str,
    hidden_threshold: float,
    exposed_threshold: float,
    top_k: int,
    bypass_cache: bool,
) -> dict[str, Any]:
  """Analyzes the timeline events of the session's XSpace."""
  events = xplane_tools.iter_timeline_events(
      session_id, host=host, bypass_cache=bypass_cache
  )
  result = async_overlap.analyze(
      events,
      hidden_threshold=hidden_threshold,
      exposed_threshold=exposed_threshold,
      top_k=top_k,
  )
  if not events:
    result["status"] = "NO_DEVICE_TIMELINE_EVENTS"
    result["reason"] = (
        "No XLA Ops, Async XLA Ops, or Sparse Core Ops timeline events were"
        " found in the capture."
    )
  else:
    result["status"] = "OK"
  return result


@decorators.cached(expire=86400)
def analyze_async_overlap(
    session_id: str,
    host: str = "",
    hidden_threshold: float = async_overlap.DEFAULT_HIDDEN_THRESHOLD,
    exposed_threshold: float = async_overlap.DEFAULT_EXPOSED_THRESHOLD,
    top_k: int = 20,
    backend: Literal["xplane", "events_db"] = "xplane",
    bypass_cache: bool = False,
) -> str:
  """Measures async XLA op overlap with compute and cross-engine concurrency.

  Classifies each async op window (from the `Async XLA Ops` line when present,
  or from paired `<op>-start` / `<op>-done` events on `XLA Ops` /
  `Sparse Core Ops`) as `hidden` (overlap >= `hidden_threshold`), `exposed`
  (overlap <= `exposed_threshold`), or `partial`, and computes pairwise
  concurrent busy time across engines on the same TPU chip.

  Args:
    session_id: The XProf session ID or path to a local XPlane file/directory.
    host: Optional host filter. With `backend="xplane"` it selects the host
      fetched from an XProf server, or one trace file (by file stem or
      `get_hosts` name) of a local trace directory; without it, the trace files
      of a local directory are analyzed per host and never merged. With
      `backend="events_db"` it selects one host's Events DB file of a 1P
      session.
    hidden_threshold: Covered fraction at or above which an async op is
      classified as `hidden` (default 0.9).
    exposed_threshold: Covered fraction at or below which an async op is
      classified as `exposed` (default 0.1).
    top_k: Maximum number of individual async op windows to include in
      `top_exposed` sorted by exposed duration descending (default 20).
    backend: `"xplane"` (default) reads the XSpace and analyzes it in Python.
      `"events_db"` runs the analysis as SQL over the session's Events DB (the
      1P Events DB service, DuckDB over a parquet file for local traces),
      which avoids loading every trace event; it pairs start and done ops only
      and does not compute `concurrency`.
    bypass_cache: Whether to bypass cache and recompute metrics.

  Returns:
    A JSON string containing `status`, `session_id`, `backend`, `summary`,
    `by_op`, `by_plane`, `top_exposed`, `concurrency`, and `thresholds`.

  Raises:
    ValueError: If thresholds do not satisfy `0 <= exposed < hidden <= 1`, or
      `backend` is unknown.
  """
  if not 0.0 <= exposed_threshold < hidden_threshold <= 1.0:
    raise ValueError(
        "thresholds must satisfy 0 <= exposed_threshold < hidden_threshold"
        f" <= 1, got exposed={exposed_threshold}, hidden={hidden_threshold}"
    )
  if backend == "events_db":
    analyze = _analyze_events_db
  elif backend == "xplane":
    analyze = _analyze_xplane
  else:
    raise ValueError(
        f"backend must be 'xplane' or 'events_db', got {backend!r}"
    )

  result = analyze(
      session_id,
      host,
      hidden_threshold,
      exposed_threshold,
      top_k,
      bypass_cache,
  )
  result["session_id"] = str(session_id)
  result["backend"] = backend
  if host:
    result["host"] = host
  return json.dumps(result, indent=2)
