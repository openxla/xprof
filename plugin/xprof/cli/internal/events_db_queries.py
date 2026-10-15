"""Ready-made queries over an Events DB `Events` table.

The queries run unchanged on the 1P Events DB service (GoogleSQL) and on DuckDB
over a parquet file (3P). They encode the rules that hand-written Events DB SQL
often gets wrong:

* Op time is `self_time_ns`. `end_ns - start_ns` is inclusive time, so a parent
  op (for example a `while` loop) also counts the time of every op inside it.
* Time is summed for one `device` and one `category` at a time. Rows of other
  devices, of the host, and of other timeline lines (`XLA Modules`,
  `TC Overlay`, ...) cover the same wall time again.
* `HloOpMeta` rows are joined on `hlo_module = module_name AND hlo_op =
  op_name`. `op_fingerprint` is not unique per op, and op names repeat across
  modules.
"""

from typing import Literal

Dialect = Literal["googlesql", "duckdb"]
OrderBy = Literal["self", "inclusive"]

DEFAULT_DEVICE = "TPU:0"
DEFAULT_CATEGORY = "XLA Ops"
MAX_LIMIT = 1000

# Query parameter names; values are bound by the caller.
DEVICE_PARAM = "device"
CATEGORY_PARAM = "category"
HOST_PARAM = "host"

_ORDER_COLUMNS = {"self": "self_time_ns", "inclusive": "inclusive_ns"}


def _check_dialect(dialect: Dialect) -> None:
  if dialect not in ("googlesql", "duckdb"):
    raise ValueError(f"unknown SQL dialect: {dialect!r}")


def _int_type(dialect: Dialect) -> str:
  return "INT64" if dialect == "googlesql" else "BIGINT"


def _param(dialect: Dialect, name: str) -> str:
  """Returns a named query parameter reference."""
  return f"@{name}" if dialect == "googlesql" else f"${name}"


def _sum(dialect: Dialect, expr: str) -> str:
  """Returns an integer `SUM` of `expr` that is 0 for no rows."""
  return f"CAST(COALESCE(SUM({expr}), 0) AS {_int_type(dialect)})"


def build_info_query(
    dialect: Dialect, *, host_column: str | None = None
) -> str:
  """Returns a query that lists the devices and categories of `Events`.

  Args:
    dialect: `"googlesql"` for GoogleSQL, `"duckdb"` for DuckDB.
    host_column: SQL expression naming the host of a row. Defaults to a single
      unnamed host.

  Returns:
    A query returning one row per `host`, `device` and `category`, with columns
    `host`, `device`, `category`, `events`, `self_time_ns`, `first_start_ns`
    and `last_end_ns`.

  Raises:
    ValueError: On an unknown dialect.
  """
  _check_dialect(dialect)
  int_type = _int_type(dialect)
  host_expr = host_column or "''"
  return f"""\
WITH scoped AS (
  SELECT
    {host_expr} AS host,
    COALESCE(device, '') AS device,
    COALESCE(category, '') AS category,
    self_time_ns,
    start_ns,
    end_ns
  FROM Events
)
SELECT
  host,
  device,
  category,
  COUNT(*) AS events,
  {_sum(dialect, "self_time_ns")} AS self_time_ns,
  CAST(MIN(start_ns) AS {int_type}) AS first_start_ns,
  CAST(MAX(end_ns) AS {int_type}) AS last_end_ns
FROM scoped
GROUP BY host, device, category
ORDER BY host, device, category
"""


def build_top_ops_query(
    dialect: Dialect,
    *,
    order_by: OrderBy = "self",
    limit: int = 20,
    host_column: str | None = None,
    filter_host: bool = False,
    hlo_meta_table: str | None = None,
) -> str:
  """Returns a query for the ops with the most time on one device and category.

  The query reads the `device` and `category` parameters, and the `host`
  parameter when `filter_host` is set.

  Args:
    dialect: `"googlesql"` for GoogleSQL, `"duckdb"` for DuckDB.
    order_by: `"self"` orders by self time, `"inclusive"` by inclusive time.
    limit: Number of ops returned, from 1 to `MAX_LIMIT`.
    host_column: SQL expression naming the host of a row. Defaults to a single
      unnamed host.
    filter_host: Whether to keep only rows of the `host` parameter. Requires
      `host_column`.
    hlo_meta_table: Name of an `HloOpMeta` table. When set, each op gets the
      `op_category` of its `HloOpMeta` row, or `""` without one.

  Returns:
    A query returning columns `host`, `hlo_module`, `op`, `occurrences`,
    `self_time_ns`, `inclusive_ns`, `flops`, `bytes_accessed` and
    `total_self_time_ns` (the self time of all ops of the device and category),
    plus `op_category` with `hlo_meta_table`.

  Raises:
    ValueError: On an unknown dialect or `order_by`, a `limit` out of range, or
      `filter_host` without `host_column`.
  """
  _check_dialect(dialect)
  if order_by not in _ORDER_COLUMNS:
    raise ValueError(
        f"order_by must be 'self' or 'inclusive', got {order_by!r}"
    )
  if not 1 <= limit <= MAX_LIMIT:
    raise ValueError(f"limit must be in [1, {MAX_LIMIT}], got {limit}")
  if filter_host and not host_column:
    raise ValueError("filter_host requires host_column")
  host_expr = host_column or "''"
  host_filter = (
      f"\n    AND {host_expr} = {_param(dialect, HOST_PARAM)}"
      if filter_host
      else ""
  )
  meta_select = ""
  meta_join = ""
  if hlo_meta_table:
    meta_select = ",\n  COALESCE(meta.op_category, '') AS op_category"
    meta_join = f"""
LEFT JOIN (
  SELECT module_name, op_name, ANY_VALUE(op_category) AS op_category
  FROM {hlo_meta_table}
  GROUP BY module_name, op_name
) AS meta
  ON meta.module_name = agg.hlo_module AND meta.op_name = agg.op"""
  order_column = _ORDER_COLUMNS[order_by]
  return f"""\
WITH ops AS (
  SELECT
    {host_expr} AS host,
    COALESCE(hlo_module, '') AS hlo_module,
    COALESCE(NULLIF(hlo_op, ''), kernel_name, '') AS op,
    self_time_ns,
    end_ns - start_ns AS inclusive_ns,
    flops,
    memory_accessed
  FROM Events
  WHERE device = {_param(dialect, DEVICE_PARAM)}
    AND category = {_param(dialect, CATEGORY_PARAM)}{host_filter}
),
agg AS (
  SELECT
    host,
    hlo_module,
    op,
    COUNT(*) AS occurrences,
    {_sum(dialect, "self_time_ns")} AS self_time_ns,
    {_sum(dialect, "inclusive_ns")} AS inclusive_ns,
    {_sum(dialect, "flops")} AS flops,
    {_sum(dialect, "memory_accessed")} AS bytes_accessed
  FROM ops
  GROUP BY host, hlo_module, op
)
SELECT
  agg.host AS host,
  agg.hlo_module AS hlo_module,
  agg.op AS op,
  agg.occurrences AS occurrences,
  agg.self_time_ns AS self_time_ns,
  agg.inclusive_ns AS inclusive_ns,
  agg.flops AS flops,
  agg.bytes_accessed AS bytes_accessed,
  CAST(SUM(agg.self_time_ns) OVER () AS {_int_type(dialect)})
    AS total_self_time_ns{meta_select}
FROM agg{meta_join}
ORDER BY {order_column} DESC, hlo_module, op
LIMIT {int(limit)}
"""
