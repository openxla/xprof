"""Tool to analyze async XLA op overlap with compute and cross-engine concurrency."""

import json

from xprof.cli.internal import async_overlap
from xprof.cli.internal import decorators

from xprof.cli.internal.oss import xplane_tools


@decorators.cached(expire=86400)
def analyze_async_overlap(
    session_id: str,
    host: str = "",
    hidden_threshold: float = async_overlap.DEFAULT_HIDDEN_THRESHOLD,
    exposed_threshold: float = async_overlap.DEFAULT_EXPOSED_THRESHOLD,
    top_k: int = 20,
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
    host: Optional host. Selects the host fetched from an XProf server, or one
      trace file (by file stem or `get_hosts` name) of a local trace directory.
      Without it, the trace files of a local directory are analyzed per host and
      never merged.
    hidden_threshold: Covered fraction at or above which an async op is
      classified as `hidden` (default 0.9).
    exposed_threshold: Covered fraction at or below which an async op is
      classified as `exposed` (default 0.1).
    top_k: Maximum number of individual async op windows to include in
      `top_exposed` sorted by exposed duration descending (default 20).
    bypass_cache: Whether to bypass cache and recompute metrics.

  Returns:
    A JSON string containing `status`, `session_id`, `summary`, `by_op`,
    `by_plane`, `top_exposed`, `concurrency`, and `thresholds`.

  Raises:
    ValueError: If thresholds do not satisfy `0 <= exposed < hidden <= 1`.
  """
  if not 0.0 <= exposed_threshold < hidden_threshold <= 1.0:
    raise ValueError(
        "thresholds must satisfy 0 <= exposed_threshold < hidden_threshold"
        f" <= 1, got exposed={exposed_threshold}, hidden={hidden_threshold}"
    )

  events = xplane_tools.iter_timeline_events(
      session_id, host=host, bypass_cache=bypass_cache
  )
  result = async_overlap.analyze(
      events,
      hidden_threshold=hidden_threshold,
      exposed_threshold=exposed_threshold,
      top_k=top_k,
  )
  result["session_id"] = str(session_id)
  if host:
    result["host"] = host
  if not events:
    result["status"] = "NO_DEVICE_TIMELINE_EVENTS"
    result["reason"] = (
        "No XLA Ops, Async XLA Ops, or Sparse Core Ops timeline events were"
        " found in the capture."
    )
  else:
    result["status"] = "OK"
  return json.dumps(result, indent=2)
