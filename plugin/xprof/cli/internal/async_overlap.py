"""Overlap of asynchronous XLA ops with compute, and cross-engine concurrency.

An asynchronous op (a collective, an async copy, a send/recv, ...) runs from
its start op to its done op. Device time inside that window is either covered
by other ops on the same device, which hides the transfer, or not covered,
which is time the transfer may be costing the step. This module measures that
coverage for each async op and classifies it as hidden, partial or exposed.

It also measures how long the engines of one TPU chip (the TensorCore and its
SparseCores) are busy at the same time.

The module works on plain `TimelineEvent`s so the same code serves every
trace backend; the backends only convert their planes into events.

Async op windows come from the device's "Async XLA Ops" line, where each event
already spans one async op. A plane without that line falls back to pairing
`<op>-start` / `<op>-done` events on its op line.

Coverage is measured against the same plane's op line only, and excludes the
start and done ops themselves. It does not account for work on other engines.

Events from different hosts (`TimelineEvent.host`) are never paired, merged or
compared, even when their planes have the same name.
"""

import bisect
import collections
from collections.abc import Iterable, Sequence
import dataclasses
import re
from typing import Any

ASYNC_OPS_LINE = "Async XLA Ops"
OPS_LINES = frozenset({"XLA Ops", "Sparse Core Ops"})
LINE_NAMES = OPS_LINES | {ASYNC_OPS_LINE}

HIDDEN = "hidden"
PARTIAL = "partial"
EXPOSED = "exposed"

DEFAULT_HIDDEN_THRESHOLD = 0.9
DEFAULT_EXPOSED_THRESHOLD = 0.1

# `<base>-start[.N]` / `<base>-done[.N]`, e.g. `all-gather-start.3`,
# `copy-done`, `send-done.1`, `fusion.7.call-start`.
_START_DONE_RE = re.compile(r"^(?P<base>.+)-(?P<role>start|done)(?P<n>\.\d+)?$")
# `send` and `recv` start an async op without a `-start` suffix.
_SEND_RECV_RE = re.compile(r"^(?P<base>send|recv)(?P<n>\.\d+)?$")
_TPU_PLANE_RE = re.compile(
    r"^/device:TPU:(?P<chip>\d+)(?: SparseCore(?: [A-Za-z]+)? (?P<sc>\d+))?$"
)


@dataclasses.dataclass(frozen=True)
class TimelineEvent:
  """One trace event, with an absolute start time.

  Attributes:
    plane: Name of the plane the event is on, e.g. `/device:TPU:0`.
    line: Name of the line the event is on, e.g. `XLA Ops`.
    name: Event name. A full HLO text such as `%copy-start.1 = ...` is accepted;
      see `instruction_name`.
    start_ps: Absolute start time in picoseconds.
    duration_ps: Duration in picoseconds.
    host: The host or trace file the event came from, e.g. the trace file stem
      of one host in a multi-host capture. Empty when all events come from one
      host.
  """

  plane: str
  line: str
  name: str
  start_ps: int
  duration_ps: int
  host: str = ""

  @property
  def end_ps(self) -> int:
    """Absolute end time of the event in picoseconds."""
    return self.start_ps + self.duration_ps


def instruction_name(name: str) -> str:
  """Returns the HLO instruction name in an event name.

  Trace event names are either a bare instruction name (`copy-start.1`) or
  the full HLO text (`%copy-start.1 = (f32[8]) copy-start(...)`).

  Args:
    name: The event name.

  Returns:
    The instruction name without the leading `%`.
  """
  name = name.strip()
  if name.startswith("%"):
    name = name[1:].split(" = ", 1)[0].strip()
  return name


def async_role(name: str) -> tuple[str, str] | None:
  """Classifies an instruction name as the start or done op of an async op.

  Args:
    name: An instruction name, see `instruction_name`.

  Returns:
    `(key, role)`, where `role` is `"start"` or `"done"` and `key` is the same
    for the start and done op of one async op; or None for any other op.
  """
  match = _START_DONE_RE.match(name)
  if match:
    return match["base"] + (match["n"] or ""), match["role"]
  match = _SEND_RECV_RE.match(name)
  if match:
    return match["base"] + (match["n"] or ""), "start"
  return None


def op_family(name: str) -> str:
  """Returns the op kind of an async op, e.g. `all-gather` or `copy`.

  The kind is the last dot-separated token of the op's base name that is not a
  number: `all-gather-start.3` and `all-gather.3-start` are `all-gather`, and
  `fusion.7.call-start` is `call`.

  Args:
    name: An instruction name, see `instruction_name`.

  Returns:
    The op kind.
  """
  match = _START_DONE_RE.match(name) or _SEND_RECV_RE.match(name)
  base = match["base"] if match else name
  tokens = [token for token in base.split(".") if token and not token.isdigit()]
  return tokens[-1] if tokens else base


def _merge(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
  """Merges intervals into a sorted list of disjoint, non-empty intervals."""
  merged: list[tuple[int, int]] = []
  for lo, hi in sorted(intervals):
    if hi <= lo:
      continue
    if merged and lo <= merged[-1][1]:
      if hi > merged[-1][1]:
        merged[-1] = (merged[-1][0], hi)
    else:
      merged.append((lo, hi))
  return merged


class _Coverage:
  """Answers "how much of [lo, hi) is covered" over fixed disjoint intervals."""

  def __init__(self, merged: Sequence[tuple[int, int]]):
    self._starts = [lo for lo, _ in merged]
    self._merged = merged
    self._prefix = [0]
    for lo, hi in merged:
      self._prefix.append(self._prefix[-1] + hi - lo)

  @property
  def total(self) -> int:
    """Total covered duration across all intervals."""
    return self._prefix[-1]

  def covered(self, lo: int, hi: int) -> int:
    """Returns the duration inside `[lo, hi)` covered by the intervals."""
    if hi <= lo or not self._merged:
      return 0
    # First interval that could end after `lo`, last that starts before `hi`.
    first = max(bisect.bisect_right(self._starts, lo) - 1, 0)
    last = bisect.bisect_left(self._starts, hi) - 1
    if last < first:
      return 0
    total = self._prefix[last + 1] - self._prefix[first]
    first_lo, first_hi = self._merged[first]
    total -= max(0, min(lo, first_hi) - first_lo)
    last_lo, last_hi = self._merged[last]
    total -= max(0, last_hi - max(hi, last_lo))
    return max(total, 0)


def _intersection(
    a: Sequence[tuple[int, int]], b: Sequence[tuple[int, int]]
) -> int:
  """Returns the total length of the intersection of two merged lists."""
  i = j = 0
  total = 0
  while i < len(a) and j < len(b):
    lo = max(a[i][0], b[j][0])
    hi = min(a[i][1], b[j][1])
    if hi > lo:
      total += hi - lo
    if a[i][1] < b[j][1]:
      i += 1
    else:
      j += 1
  return total


@dataclasses.dataclass(frozen=True)
class _AsyncWindow:
  plane: str
  name: str
  start_ps: int
  end_ps: int


def _windows_from_async_line(
    plane: str, events: Iterable[TimelineEvent]
) -> list[_AsyncWindow]:
  return [
      _AsyncWindow(plane, instruction_name(e.name), e.start_ps, e.end_ps)
      for e in events
  ]


def _windows_from_start_done(
    plane: str, events: Iterable[TimelineEvent]
) -> tuple[list[_AsyncWindow], int]:
  """Pairs start and done ops in time order; returns windows and unpaired."""
  open_starts: dict[str, collections.deque[TimelineEvent]] = (
      collections.defaultdict(collections.deque)
  )
  windows = []
  unpaired = 0
  for event in sorted(events, key=lambda e: e.start_ps):
    role = async_role(instruction_name(event.name))
    if role is None:
      continue
    key, kind = role
    if kind == "start":
      open_starts[key].append(event)
    elif open_starts[key]:
      start = open_starts[key].popleft()
      windows.append(
          _AsyncWindow(
              plane, instruction_name(start.name), start.start_ps, event.end_ps
          )
      )
    else:
      unpaired += 1
  unpaired += sum(len(q) for q in open_starts.values())
  return windows, unpaired


def classify(
    overlap_ratio: float,
    hidden_threshold: float = DEFAULT_HIDDEN_THRESHOLD,
    exposed_threshold: float = DEFAULT_EXPOSED_THRESHOLD,
) -> str:
  """Returns `HIDDEN`, `PARTIAL` or `EXPOSED` for a covered fraction."""
  if overlap_ratio >= hidden_threshold:
    return HIDDEN
  if overlap_ratio <= exposed_threshold:
    return EXPOSED
  return PARTIAL


def _new_bucket() -> dict[str, Any]:
  return {
      "count": 0,
      "span_ps": 0,
      "exposed_ps": 0,
      HIDDEN: 0,
      PARTIAL: 0,
      EXPOSED: 0,
  }


def _add(bucket: dict[str, Any], span: int, exposed: int, label: str) -> None:
  bucket["count"] += 1
  bucket["span_ps"] += span
  bucket["exposed_ps"] += exposed
  bucket[label] += 1


def _engine_concurrency(
    busy: dict[tuple[str, str, str], list[tuple[int, int]]],
) -> list[dict[str, Any]]:
  """Groups op lines by host and TPU chip and measures pairwise concurrency.

  Args:
    busy: Merged busy intervals keyed by `(host, plane, line)`.

  Returns:
    One entry per host and TPU chip with at least two busy engines. Entries
    carry `host` when it is not empty.
  """
  chips: dict[tuple[str, int], list[tuple[str, str]]] = collections.defaultdict(
      list
  )
  for host, plane, line in busy:
    match = _TPU_PLANE_RE.match(plane)
    if match:
      chips[(host, int(match["chip"]))].append((plane, line))

  result = []
  for host, chip in sorted(chips):
    engines = sorted(chips[(host, chip)])
    if len(engines) < 2:
      continue
    intervals = {e: busy[(host, *e)] for e in engines}
    totals = {e: sum(hi - lo for lo, hi in intervals[e]) for e in engines}
    pairs = []
    for i, a in enumerate(engines):
      for b in engines[i + 1 :]:
        concurrent = _intersection(intervals[a], intervals[b])
        shorter = min(totals[a], totals[b])
        pairs.append({
            "a": {"plane": a[0], "line": a[1]},
            "b": {"plane": b[0], "line": b[1]},
            "concurrent_ps": concurrent,
            # Fraction of the less busy engine's time spent concurrently.
            "concurrent_ratio": concurrent / shorter if shorter else 0.0,
        })
    pairs.sort(key=lambda p: p["concurrent_ps"], reverse=True)
    entry: dict[str, Any] = {
        "chip": chip,
        "engines": [
            {"plane": p, "line": l, "busy_ps": totals[(p, l)]}
            for p, l in engines
        ],
        "pairs": pairs,
    }
    if host:
      entry["host"] = host
    result.append(entry)
  return result


def analyze(
    events: Iterable[TimelineEvent],
    hidden_threshold: float = DEFAULT_HIDDEN_THRESHOLD,
    exposed_threshold: float = DEFAULT_EXPOSED_THRESHOLD,
    top_k: int = 20,
) -> dict[str, Any]:
  """Measures async-op overlap and cross-engine concurrency.

  Args:
    events: Events from the planes to analyze. Only `LINE_NAMES` lines are used.
    hidden_threshold: Covered fraction at or above which an async op is
      `hidden`.
    exposed_threshold: Covered fraction at or below which an async op is
      `exposed`. Anything in between is `partial`.
    top_k: Number of async ops to list in `top_exposed`. Values <= 0 list none.

  Returns:
    A JSON-serializable dict with `summary`, `by_op` (per op kind, most exposed
    time first), `by_plane`, `top_exposed`, `concurrency` (per host and TPU
    chip with at least two busy engines) and `thresholds`. When events come
    from several hosts, `by_plane` keys are `<host>:<plane>` and its entries
    carry `host`. `top_exposed` and `concurrency` entries carry `host` when it
    is not empty.

  Raises:
    ValueError: If the thresholds are not `0 <= exposed < hidden <= 1`.
  """
  if not 0.0 <= exposed_threshold < hidden_threshold <= 1.0:
    raise ValueError(
        "thresholds must satisfy 0 <= exposed_threshold < hidden_threshold"
        f" <= 1, got exposed={exposed_threshold}, hidden={hidden_threshold}"
    )

  by_plane_line: dict[tuple[str, str, str], list[TimelineEvent]] = (
      collections.defaultdict(list)
  )
  for event in events:
    if event.line in LINE_NAMES:
      by_plane_line[(event.host, event.plane, event.line)].append(event)

  planes = sorted({(host, plane) for host, plane, _ in by_plane_line})
  multi_host = len({host for host, _ in planes}) > 1
  busy: dict[tuple[str, str, str], list[tuple[int, int]]] = {}
  summary = _new_bucket()
  summary["unpaired_ops"] = 0
  by_op: dict[str, dict[str, Any]] = collections.defaultdict(_new_bucket)
  by_plane: dict[str, dict[str, Any]] = {}
  rows = []

  for host, plane in planes:
    op_events: list[TimelineEvent] = []
    for line in sorted(OPS_LINES):
      op_events.extend(by_plane_line.get((host, plane, line), []))
    for line in OPS_LINES:
      if (host, plane, line) in by_plane_line:
        busy[(host, plane, line)] = _merge(
            (e.start_ps, e.end_ps) for e in by_plane_line[(host, plane, line)]
        )

    compute = _Coverage(
        _merge(
            (e.start_ps, e.end_ps)
            for e in op_events
            if async_role(instruction_name(e.name)) is None
        )
    )

    async_events = by_plane_line.get((host, plane, ASYNC_OPS_LINE))
    if async_events:
      windows = _windows_from_async_line(plane, async_events)
      source = ASYNC_OPS_LINE
    else:
      windows, unpaired = _windows_from_start_done(plane, op_events)
      summary["unpaired_ops"] += unpaired
      source = "start_done_pairs"
    if not windows:
      continue

    plane_bucket = _new_bucket()
    plane_bucket["window_source"] = source
    if multi_host:
      plane_bucket["host"] = host
    for window in windows:
      span = window.end_ps - window.start_ps
      if span <= 0:
        continue
      covered = compute.covered(window.start_ps, window.end_ps)
      ratio = covered / span
      exposed = span - covered
      label = classify(ratio, hidden_threshold, exposed_threshold)
      family = op_family(window.name)
      _add(summary, span, exposed, label)
      _add(by_op[family], span, exposed, label)
      _add(plane_bucket, span, exposed, label)
      row: dict[str, Any] = {
          "plane": plane,
          "name": window.name,
          "op": family,
          "start_ps": window.start_ps,
          "span_ps": span,
          "exposed_ps": exposed,
          "overlap_ratio": ratio,
          "class": label,
      }
      if host:
        row["host"] = host
      rows.append(row)
    by_plane[f"{host}:{plane}" if multi_host else plane] = plane_bucket

  rows.sort(
      key=lambda r: (
          -r["exposed_ps"],
          r.get("host", ""),
          r["plane"],
          r["start_ps"],
      )
  )
  summary["overlap_ratio"] = (
      1.0 - summary["exposed_ps"] / summary["span_ps"]
      if summary["span_ps"]
      else None
  )
  return {
      "summary": summary,
      "by_op": dict(
          sorted(by_op.items(), key=lambda kv: (-kv[1]["exposed_ps"], kv[0]))
      ),
      "by_plane": by_plane,
      "top_exposed": rows[: max(top_k, 0)],
      "concurrency": _engine_concurrency(busy),
      "thresholds": {
          "hidden": hidden_threshold,
          "exposed": exposed_threshold,
      },
  }
