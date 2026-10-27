"""Async XLA op overlap computed in SQL over an Events DB `Events` table.

This is the Events DB counterpart of `async_overlap.analyze`. The pairing,
merging and coverage run inside the query engine (the 1P Events DB service,
which runs GoogleSQL, or DuckDB over a parquet file in 3P), so only one
aggregated row per async op key comes back instead of every trace event.

Rules match `async_overlap`:

* Async ops are the `<op>-start[.N]` / `<op>-done[.N]` (and `send` / `recv`)
  events on the `XLA Ops` and `Sparse Core Ops` categories of one device.
* Start and done ops with the same key are paired first-in, first-out in time
  order. A done op without an open start, and a start op that is never done,
  count as unpaired.
* A window runs from the start op's start to the done op's end. Its coverage is
  the time inside it covered by the merged non-async ops of the same device.

The Events DB has no `Async XLA Ops` category, so windows always come from
start / done pairs, and it stores nanoseconds, so times are scaled to
picoseconds in the result. Cross-engine concurrency is not computed.
"""

import collections
from collections.abc import Iterable, Mapping
from typing import Any, Literal

from xprof.cli.internal import async_overlap

Dialect = Literal["googlesql", "duckdb"]

WINDOW_SOURCE = "start_done_pairs"
OPS_CATEGORIES = ("XLA Ops", "Sparse Core Ops")

# Row types returned by the query.
_WINDOWS = "w"
_UNPAIRED = "u"

_NS_TO_PS = 1000

# Patterns use RE2 syntax, which both GoogleSQL and DuckDB implement.
# `%name = (shape) op(...)` -> `name`; see `async_overlap.instruction_name`.
_HLO_TEXT_RE = r"^%(.*?)(?: = .*)?$"
_START_RE = r"^.+-start([.][0-9]+)?$"
_SEND_RECV_RE = r"^(send|recv)([.][0-9]+)?$"
_DONE_RE = r"^.+-done([.][0-9]+)?$"
# `<base>-start[.N]` / `<base>-done[.N]` -> `<base>[.N]`.
_KEY_RE = r"^(.+)-(?:start|done)((?:[.][0-9]+)?)$"


def _str(dialect: Dialect, value: str) -> str:
  """Returns a SQL string literal for a constant pattern or name."""
  if "'" in value:
    raise ValueError(f"unexpected quote in SQL constant: {value!r}")
  # GoogleSQL needs raw strings to keep backslashes; DuckDB strings keep them as
  # is.
  return f"r'{value}'" if dialect == "googlesql" else f"'{value}'"


def _matches(dialect: Dialect, column: str, pattern: str) -> str:
  func = "REGEXP_CONTAINS" if dialect == "googlesql" else "regexp_matches"
  return f"{func}({column}, {_str(dialect, pattern)})"


def build_query(
    dialect: Dialect,
    *,
    hidden_threshold: float = async_overlap.DEFAULT_HIDDEN_THRESHOLD,
    exposed_threshold: float = async_overlap.DEFAULT_EXPOSED_THRESHOLD,
    top_k: int = 20,
    host_column: str | None = None,
    host_param: str | None = None,
) -> str:
  """Returns the async overlap query over a table named `Events`.

  Args:
    dialect: `"googlesql"` for GoogleSQL, `"duckdb"` for DuckDB.
    hidden_threshold: Covered fraction at or above which a window is hidden.
    exposed_threshold: Covered fraction at or below which a window is exposed.
    top_k: Number of most exposed windows returned individually.
    host_column: SQL expression naming the host of a row, e.g. one derived from
      the source file. Rows of different hosts are never paired or merged
      together. Defaults to a single unnamed host.
    host_param: Name of a query parameter (GoogleSQL `@name`) to filter the
      host on. Requires `host_column`.

  Returns:
    A query returning rows with columns `row_type`, `host`, `device`,
    `pair_key`, `rk`, `name`, `n`, `span_ns`, `covered_ns`, `hidden_n`,
    `exposed_n` and `start_ns`. See `assemble`.

  Raises:
    ValueError: On an unknown dialect or invalid thresholds.
  """
  if dialect not in ("googlesql", "duckdb"):
    raise ValueError(f"unknown SQL dialect: {dialect!r}")
  hidden = float(hidden_threshold)
  exposed = float(exposed_threshold)
  if not 0.0 <= exposed < hidden <= 1.0:
    raise ValueError(
        "thresholds must satisfy 0 <= exposed_threshold < hidden_threshold"
        f" <= 1, got exposed={exposed}, hidden={hidden}"
    )
  if host_param and not host_column:
    raise ValueError("host_param requires host_column")
  top = max(int(top_k), 0)
  int_type = "INT64" if dialect == "googlesql" else "BIGINT"
  float_type = "FLOAT64" if dialect == "googlesql" else "DOUBLE"
  host_expr = host_column or "''"
  categories = ", ".join(_str(dialect, c) for c in OPS_CATEGORIES)
  host_filter = f"AND {host_expr} = @{host_param}" if host_param else ""
  raw_name = "TRIM(COALESCE(NULLIF(hlo_op, ''), kernel_name, ''))"
  name_expr = (
      f"TRIM(REGEXP_REPLACE({raw_name}, {_str(dialect, _HLO_TEXT_RE)},"
      f" {_str(dialect, chr(92) + '1')}))"
  )
  role_expr = (
      f"CASE WHEN {_matches(dialect, 'name', _START_RE)}"
      f" OR {_matches(dialect, 'name', _SEND_RECV_RE)} THEN 0"
      f" WHEN {_matches(dialect, 'name', _DONE_RE)} THEN 1 ELSE 2 END"
  )
  key_expr = (
      f"REGEXP_REPLACE(name, {_str(dialect, _KEY_RE)},"
      f" {_str(dialect, chr(92) + '1' + chr(92) + '2')})"
  )
  # Order of events of one async op key: by time, start before done on ties.
  key_order = "PARTITION BY host, device, pair_key ORDER BY t0, role, t1"
  running = "ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW"
  before = "ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING"
  # Lowest open-start count seen before a row, never above zero. A done op
  # that takes the count below it has no open start.
  floor_prev = "LEAST(0, COALESCE(min_prev, 0))"
  ratio = f"CAST(covered AS {float_type}) / CAST(span AS {float_type})"
  return f"""
WITH
ops AS (
  SELECT
    {host_expr} AS host,
    device,
    {name_expr} AS name,
    CAST(start_ns AS {int_type}) AS t0,
    CAST(end_ns AS {int_type}) AS t1
  FROM Events
  WHERE category IN ({categories}) {host_filter}
),
classified AS (
  SELECT host, device, name, t0, t1, {role_expr} AS role FROM ops
),
async_events AS (
  SELECT host, device, name, t0, t1, role, {key_expr} AS pair_key
  FROM classified
  WHERE role < 2
),
counted AS (
  SELECT
    host, device, name, t0, t1, role, pair_key,
    SUM(1 - 2 * role) OVER ({key_order} {running}) AS x,
    SUM(1 - role) OVER ({key_order} {running}) AS s_idx,
    SUM(role) OVER ({key_order} {running}) AS d_cum
  FROM async_events
),
sequenced AS (
  SELECT
    host, device, name, t0, t1, role, pair_key, x, s_idx, d_cum,
    MIN(x) OVER ({key_order} {before}) AS min_prev
  FROM counted
),
starts AS (
  SELECT host, device, pair_key, s_idx, name, t0 AS w_start
  FROM sequenced
  WHERE role = 0
),
dones AS (
  SELECT
    host, device, pair_key,
    d_cum + LEAST(0, COALESCE(min_prev, 0), x) AS m_idx,
    t1 AS w_end
  FROM sequenced
  WHERE role = 1 AND x >= {floor_prev}
),
pairs AS (
  SELECT s.host, s.device, s.pair_key, s.s_idx, s.name, s.w_start, d.w_end
  FROM starts s
  LEFT JOIN dones d
    ON d.host = s.host AND d.device = s.device
    AND d.pair_key = s.pair_key AND d.m_idx = s.s_idx
),
windows AS (
  SELECT host, device, pair_key, s_idx, name, w_start, w_end
  FROM pairs
  WHERE w_end > w_start
),
unpaired AS (
  SELECT host, device, COUNT(*) AS n
  FROM (
    SELECT host, device FROM pairs WHERE w_end IS NULL
    UNION ALL
    SELECT host, device FROM sequenced WHERE role = 1 AND x < {floor_prev}
  ) u
  GROUP BY host, device
),
compute AS (
  SELECT
    host, device, t0, t1,
    MAX(t1) OVER (PARTITION BY host, device ORDER BY t0, t1 {before})
      AS prev_end
  FROM classified
  WHERE role = 2 AND t1 > t0
),
compute_tagged AS (
  SELECT
    host, device, t0, t1,
    SUM(CASE WHEN prev_end IS NULL OR t0 > prev_end THEN 1 ELSE 0 END)
      OVER (PARTITION BY host, device ORDER BY t0, t1 {running}) AS island
  FROM compute
),
islands AS (
  SELECT host, device, MIN(t0) AS m0, MAX(t1) AS m1
  FROM compute_tagged
  GROUP BY host, device, island
),
timeline AS (
  SELECT
    host, device, m0 AS ts, 0 AS kind, m1 - m0 AS len, m1 AS m_end,
    '' AS pair_key, 0 AS s_idx, 0 AS is_end
  FROM islands
  UNION ALL
  SELECT host, device, w_start, 1, 0, 0, pair_key, s_idx, 0 FROM windows
  UNION ALL
  SELECT host, device, w_end, 1, 0, 0, pair_key, s_idx, 1 FROM windows
),
busy_before AS (
  SELECT
    host, device, kind, pair_key, s_idx, is_end, ts,
    SUM(len) OVER (PARTITION BY host, device ORDER BY ts, kind {running})
      AS cum_len,
    MAX(m_end) OVER (PARTITION BY host, device ORDER BY ts, kind {running})
      AS last_end
  FROM timeline
),
coverage AS (
  SELECT
    host, device, pair_key, s_idx,
    SUM(
      (CASE WHEN is_end = 1 THEN 1 ELSE -1 END)
      * (cum_len - GREATEST(0, last_end - ts))
    ) AS covered
  FROM busy_before
  WHERE kind = 1
  GROUP BY host, device, pair_key, s_idx
),
scored AS (
  SELECT
    w.host, w.device, w.pair_key, w.name, w.w_start,
    w.w_end - w.w_start AS span,
    c.covered
  FROM windows w
  JOIN coverage c
    ON c.host = w.host AND c.device = w.device
    AND c.pair_key = w.pair_key AND c.s_idx = w.s_idx
),
ranked AS (
  SELECT
    host, device, pair_key, name, w_start, span, covered,
    ROW_NUMBER() OVER (
      ORDER BY span - covered DESC, host, device, w_start, pair_key
    ) AS rk
  FROM scored
)
SELECT
  '{_WINDOWS}' AS row_type,
  host,
  device,
  pair_key,
  CASE WHEN rk <= {top} THEN rk ELSE 0 END AS rk,
  MIN(name) AS name,
  COUNT(*) AS n,
  SUM(span) AS span_ns,
  SUM(covered) AS covered_ns,
  SUM(CASE WHEN {ratio} >= CAST({hidden!r} AS {float_type}) THEN 1 ELSE 0 END) AS hidden_n,
  SUM(CASE WHEN {ratio} <= CAST({exposed!r} AS {float_type}) THEN 1 ELSE 0 END) AS exposed_n,
  MIN(w_start) AS start_ns
FROM ranked
GROUP BY 1, 2, 3, 4, 5
UNION ALL
SELECT
  '{_UNPAIRED}', host, device, '', 0, '', n, 0, 0, 0, 0, 0
FROM unpaired
"""


def _plane(device: str) -> str:
  return f"/device:{device}"


def assemble(
    rows: Iterable[Mapping[str, Any]],
    *,
    hidden_threshold: float = async_overlap.DEFAULT_HIDDEN_THRESHOLD,
    exposed_threshold: float = async_overlap.DEFAULT_EXPOSED_THRESHOLD,
) -> dict[str, Any]:
  """Builds the `async_overlap.analyze` result shape from query rows.

  Args:
    rows: Rows returned by the `build_query` query.
    hidden_threshold: The threshold the query was built with.
    exposed_threshold: The threshold the query was built with.

  Returns:
    A JSON-serializable dict with the same `summary`, `by_op`, `by_plane`,
    `top_exposed` and `thresholds` keys as `async_overlap.analyze`, times in
    picoseconds. `concurrency` is empty. Planes are named
    `/device:<device>`, prefixed with `<host>:` when rows span several hosts.
  """
  rows = list(rows)
  hosts = {str(r["host"] or "") for r in rows}
  multi_host = len(hosts) > 1

  def plane_key(row: Mapping[str, Any]) -> str:
    plane = _plane(str(row["device"]))
    return f"{row['host']}:{plane}" if multi_host else plane

  summary = async_overlap._new_bucket()  # pylint: disable=protected-access
  summary["unpaired_ops"] = 0
  by_op: dict[str, dict[str, Any]] = collections.defaultdict(
      async_overlap._new_bucket  # pylint: disable=protected-access
  )
  by_plane: dict[str, dict[str, Any]] = {}
  top: list[tuple[int, dict[str, Any]]] = []

  for row in rows:
    if row["row_type"] == _UNPAIRED:
      summary["unpaired_ops"] += int(row["n"])
      continue
    n = int(row["n"])
    span = int(row["span_ns"]) * _NS_TO_PS
    exposed = span - int(row["covered_ns"]) * _NS_TO_PS
    hidden_n = int(row["hidden_n"])
    exposed_n = int(row["exposed_n"])
    counts = {
        async_overlap.HIDDEN: hidden_n,
        async_overlap.PARTIAL: n - hidden_n - exposed_n,
        async_overlap.EXPOSED: exposed_n,
    }
    key = plane_key(row)
    if key not in by_plane:
      by_plane[key] = async_overlap._new_bucket()  # pylint: disable=protected-access
      by_plane[key]["window_source"] = WINDOW_SOURCE
      if multi_host:
        by_plane[key]["host"] = row["host"]
    family = async_overlap.op_family(str(row["pair_key"]))
    for bucket in (summary, by_op[family], by_plane[key]):
      bucket["count"] += n
      bucket["span_ps"] += span
      bucket["exposed_ps"] += exposed
      for label, count in counts.items():
        bucket[label] += count

    rank = int(row["rk"])
    if rank > 0:
      ratio = (span - exposed) / span
      entry = {
          "plane": _plane(str(row["device"])),
          "name": row["name"],
          "op": family,
          "start_ps": int(row["start_ns"]) * _NS_TO_PS,
          "span_ps": span,
          "exposed_ps": exposed,
          "overlap_ratio": ratio,
          "class": async_overlap.classify(
              ratio, hidden_threshold, exposed_threshold
          ),
      }
      if row["host"]:
        entry["host"] = row["host"]
      top.append((rank, entry))

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
      "by_plane": dict(sorted(by_plane.items())),
      "top_exposed": [entry for _, entry in sorted(top, key=lambda t: t[0])],
      "concurrency": [],
      "thresholds": {
          "hidden": hidden_threshold,
          "exposed": exposed_threshold,
      },
  }
