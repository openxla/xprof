"""Tool to list the hosts, devices and categories of a session's Events DB."""

import json
from typing import Any

from xprof.cli.internal import decorators
from xprof.cli.internal import events_db_queries

from xprof.cli.internal.oss import events_db_sql

# Rules for summing Events DB time, returned with every result.
_RULES = (
    (
        "Sum self_time_ns, not end_ns - start_ns (inclusive time counts nested"
        " ops again)."
    ),
    (
        "Sum one device and one category at a time: other categories (for"
        " example 'XLA Modules' or 'TC Overlay') and the host rows cover the"
        " same wall time again."
    ),
    (
        "Join HloOpMeta on hlo_module = module_name AND hlo_op = op_name;"
        " op_fingerprint is not unique per op."
    ),
)


@decorators.cached(expire=86400)
def events_db_info(
    session_id: str,
    bypass_cache: bool = False,
) -> str:
  """Lists the hosts, devices and categories of a session's Events DB.

  Use it before writing Events DB SQL, to get the exact `device` and
  `category` values (for example `TPU:0` and `XLA Ops`) and where the Events DB
  is.

  Args:
    session_id: The XProf session ID (1P) or a local trace path (3P).
    bypass_cache: Whether to bypass cache and run the query again.

  Returns:
    A JSON string with `session_id`, the Events DB location (`session_root`
    and `total_hosts` in 1P, `events_db_path` for local traces), `dialect`,
    `tables`, `rules`, and `categories`: one row per `host`, `device` and
    `category` with `events`, `self_time_ns`, `first_start_ns` and
    `last_end_ns`.
  """
  source = events_db_sql.describe_source(session_id, bypass_cache=bypass_cache)
  sql = events_db_queries.build_info_query(
      events_db_sql.DIALECT, host_column=events_db_sql.HOST_COLUMN
  )
  rows = events_db_sql.run_query(session_id, sql, bypass_cache=bypass_cache)
  tables = ["Events"]
  if events_db_sql.HLO_META_TABLE:
    tables.append(events_db_sql.HLO_META_TABLE)
  result: dict[str, Any] = {"session_id": str(session_id)}
  result.update(source)
  result.update(
      dialect=events_db_sql.DIALECT,
      tables=tables,
      rules=list(_RULES),
      categories=[
          {
              "host": str(row["host"]),
              "device": str(row["device"]),
              "category": str(row["category"]),
              "events": int(row["events"]),
              "self_time_ns": int(row["self_time_ns"]),
              "first_start_ns": int(row["first_start_ns"]),
              "last_end_ns": int(row["last_end_ns"]),
          }
          for row in rows
      ],
  )
  return json.dumps(result, indent=2)
