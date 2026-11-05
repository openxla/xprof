"""Runs Events DB SQL for a local trace with DuckDB over a parquet file."""

from collections.abc import Mapping
import json
from typing import Any

import duckdb

from xprof.cli.tools.oss import events_db_tool

# SQL dialect of `run_query`.
DIALECT = "duckdb"
# The parquet file holds one trace, so rows carry no host.
HOST_COLUMN = None
# The parquet file has no `HloOpMeta` table.
HLO_META_TABLE = None


def _create(session_id: str, bypass_cache: bool) -> str:
  """Creates the Events DB parquet file if needed and returns its path."""
  created = json.loads(
      events_db_tool.create_events_db(session_id, bypass_cache=bypass_cache)
  )
  return str(created["path"])


def describe_source(
    session_id: str, *, bypass_cache: bool = False
) -> dict[str, Any]:
  """Returns where the trace's Events DB is.

  Args:
    session_id: Session ID, logdir path, run directory, or trace file.
    bypass_cache: Whether to rebuild the parquet file.

  Returns:
    `events_db_path`, the path of the Events DB parquet file.
  """
  return {"events_db_path": _create(session_id, bypass_cache)}


def run_query(
    session_id: str,
    sql: str,
    *,
    parameters: Mapping[str, str] | None = None,
    bypass_cache: bool = False,
) -> list[dict[str, Any]]:
  """Runs `sql` against the trace's `Events` table.

  The Events DB parquet file is created on first use, see
  `events_db_tool.create_events_db`.

  Args:
    session_id: Session ID, logdir path, run directory, or trace file.
    sql: A query over a table named `Events`.
    parameters: Optional query parameters, referenced as `$name`.
    bypass_cache: Whether to rebuild the parquet file.

  Returns:
    The result rows.
  """
  db_path = _create(session_id, bypass_cache).replace("'", "''")
  with duckdb.connect(":memory:") as con:
    con.execute(
        f"CREATE VIEW Events AS SELECT * FROM read_parquet('{db_path}')"
    )
    con.execute(f"SET allowed_paths = ['{db_path}']")
    con.execute("SET enable_external_access = false")
    con.execute("SET lock_configuration = true")
    cursor = con.execute(sql, dict(parameters) if parameters else None)
    columns = [d[0] for d in cursor.description or ()]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]
