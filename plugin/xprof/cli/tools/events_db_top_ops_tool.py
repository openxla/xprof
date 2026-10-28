"""Tool to list the ops with the most time in a session's Events DB."""

import json
from typing import Any, Literal

from xprof.cli.internal import decorators
from xprof.cli.internal import events_db_queries

from xprof.cli.internal.oss import events_db_sql


def _op_row(row: dict[str, Any], with_hlo_meta: bool) -> dict[str, Any]:
  """Returns one result op from a query row."""
  self_time_ns = int(row["self_time_ns"])
  total = int(row["total_self_time_ns"])
  op = {
      "host": str(row["host"]),
      "hlo_module": str(row["hlo_module"]),
      "op": str(row["op"]),
      "occurrences": int(row["occurrences"]),
      "self_time_ns": self_time_ns,
      "self_time_fraction": self_time_ns / total if total else 0.0,
      "inclusive_ns": int(row["inclusive_ns"]),
      "flops": int(row["flops"]),
      "bytes_accessed": int(row["bytes_accessed"]),
  }
  if with_hlo_meta:
    op["op_category"] = str(row["op_category"] or "")
  return op


@decorators.cached(expire=86400)
def events_db_top_ops(
    session_id: str,
    device: str = events_db_queries.DEFAULT_DEVICE,
    category: str = events_db_queries.DEFAULT_CATEGORY,
    order_by: Literal["self", "inclusive"] = "self",
    limit: int = 20,
    host: str = "",
    with_hlo_meta: bool = False,
    bypass_cache: bool = False,
) -> str:
  """Lists the ops with the most time on one device in a session's Events DB.

  Sums `self_time_ns` (time not spent in nested ops) per op on one device and
  one category, so a parent op such as a `while` loop does not count the time
  of the ops inside it. Run `events_db_info` to get the `device` and
  `category` values of a session.

  Args:
    session_id: The XProf session ID (1P) or a local trace path (3P).
    device: The `device` value, for example `TPU:0` or `TPU:0 SparseCore 0`.
    category: The `category` value, for example `XLA Ops` or `Sparse Core Ops`.
    order_by: `"self"` (default) orders by self time, `"inclusive"` by inclusive
      time (`end_ns - start_ns`, which includes nested ops).
    limit: Number of ops returned, from 1 to 1000.
    host: Optional host of a 1P session, one `<host>.events_db` file. Without
      it, hosts are listed separately. Local traces hold one host.
    with_hlo_meta: Whether to add each op's `op_category` from `HloOpMeta`,
      joined on module and op name. 1P only.
    bypass_cache: Whether to bypass cache and run the query again.

  Returns:
    A JSON string with `session_id`, `device`, `category`, `order_by`,
    `total_self_time_ns` (all ops of the device and category) and `ops`. Each
    op has `host`, `hlo_module`, `op`, `occurrences`, `self_time_ns`,
    `self_time_fraction`, `inclusive_ns`, `flops` and `bytes_accessed`, plus
    `op_category` with `with_hlo_meta`.

  Raises:
    ValueError: If `order_by` or `limit` is out of range, or `with_hlo_meta` is
      set and the Events DB has no `HloOpMeta` table.
  """
  meta_table = events_db_sql.HLO_META_TABLE
  if with_hlo_meta and not meta_table:
    raise ValueError(
        "with_hlo_meta needs an HloOpMeta table, which local Events DB files"
        " do not have."
    )
  host_column = events_db_sql.HOST_COLUMN
  filter_host = bool(host and host_column)
  sql = events_db_queries.build_top_ops_query(
      events_db_sql.DIALECT,
      order_by=order_by,
      limit=limit,
      host_column=host_column,
      filter_host=filter_host,
      hlo_meta_table=meta_table if with_hlo_meta else None,
  )
  parameters = {
      events_db_queries.DEVICE_PARAM: device,
      events_db_queries.CATEGORY_PARAM: category,
  }
  if filter_host:
    parameters[events_db_queries.HOST_PARAM] = host
  rows = events_db_sql.run_query(
      session_id, sql, parameters=parameters, bypass_cache=bypass_cache
  )
  result: dict[str, Any] = {
      "session_id": str(session_id),
      "device": device,
      "category": category,
      "order_by": order_by,
      "total_self_time_ns": int(rows[0]["total_self_time_ns"]) if rows else 0,
      "ops": [_op_row(row, with_hlo_meta) for row in rows],
  }
  if host:
    result["host"] = host
  if not rows:
    result["status"] = "NO_EVENTS"
    result["reason"] = (
        f"No events with device {device!r} and category {category!r}. Run"
        " events_db_info to list the devices and categories."
    )
  else:
    result["status"] = "OK"
  return json.dumps(result, indent=2)
