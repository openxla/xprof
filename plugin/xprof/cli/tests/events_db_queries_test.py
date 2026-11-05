"""Tests for the ready-made Events DB queries, run on DuckDB."""

from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import duckdb

from xprof.cli.internal import events_db_queries

# (host, device, category, hlo_module, hlo_op, kernel_name, start_ns, end_ns,
#  self_time_ns, flops, memory_accessed)
_EVENTS = [
    # A `while` op that contains two ops: big inclusive time, small self time.
    ("h0", "TPU:0", "XLA Ops", "m", "while.1", "", 0, 100, 10, 0, 0),
    ("h0", "TPU:0", "XLA Ops", "m", "fusion.1", "", 10, 50, 40, 1000, 64),
    ("h0", "TPU:0", "XLA Ops", "m", "fusion.1", "", 50, 100, 50, 1000, 64),
    # The same op name in another module is a different op.
    ("h0", "TPU:0", "XLA Ops", "m2", "fusion.1", "", 200, 230, 30, 10, 8),
    # SparseCore ops carry the name in `kernel_name` only.
    (
        "h0",
        "TPU:0 SparseCore 0",
        "Sparse Core Ops",
        "",
        None,
        "sc.1",
        0,
        70,
        70,
        None,
        None,
    ),
    # Rows of other lines, devices and hosts cover the same time again.
    ("h0", "TPU:0", "XLA Modules", "m", "", "m", 0, 100, 100, 0, 0),
    ("h0", "TPU:1", "XLA Ops", "m", "fusion.1", "", 0, 999, 999, 0, 0),
    ("h1", "TPU:0", "XLA Ops", "m", "fusion.1", "", 0, 500, 500, 0, 0),
]

# (module_name, op_name, op_category). `fusion.1` of `m` has two rows that
# must not double its time; `while.1` has none.
_HLO_META = [
    ("m", "fusion.1", "convolution"),
    ("m", "fusion.1", "convolution"),
    ("m2", "fusion.1", "loop fusion"),
]


def _run(sql: str, parameters: dict[str, str]) -> list[dict[str, Any]]:
  with duckdb.connect(":memory:") as con:
    con.execute(
        "CREATE TABLE Events (host VARCHAR, device VARCHAR, category VARCHAR,"
        " hlo_module VARCHAR, hlo_op VARCHAR, kernel_name VARCHAR, start_ns"
        " UBIGINT, end_ns UBIGINT, self_time_ns UBIGINT, flops UBIGINT,"
        " memory_accessed UBIGINT)"
    )
    con.executemany(
        "INSERT INTO Events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", _EVENTS
    )
    con.execute(
        "CREATE TABLE HloOpMeta (module_name VARCHAR, op_name VARCHAR,"
        " op_category VARCHAR)"
    )
    con.executemany("INSERT INTO HloOpMeta VALUES (?, ?, ?)", _HLO_META)
    cursor = con.execute(sql, parameters)
    columns = [d[0] for d in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def _top_ops(
    device: str = "TPU:0",
    category: str = "XLA Ops",
    host: str | None = "h0",
    **kwargs: Any,
) -> list[dict[str, Any]]:
  parameters = {"device": device, "category": category}
  if host is not None:
    parameters["host"] = host
    kwargs.setdefault("host_column", "host")
    kwargs.setdefault("filter_host", True)
  return _run(
      events_db_queries.build_top_ops_query("duckdb", **kwargs), parameters
  )


def _names(rows: list[dict[str, Any]]) -> list[tuple[str, str]]:
  return [(r["hlo_module"], r["op"]) for r in rows]


class TopOpsQueryTest(parameterized.TestCase):

  def test_orders_by_self_time(self):
    rows = _top_ops()
    self.assertEqual(
        _names(rows), [("m", "fusion.1"), ("m2", "fusion.1"), ("m", "while.1")]
    )
    self.assertEqual(rows[0]["occurrences"], 2)
    self.assertEqual(rows[0]["self_time_ns"], 90)
    self.assertEqual(rows[0]["inclusive_ns"], 90)
    self.assertEqual(rows[0]["flops"], 2000)
    self.assertEqual(rows[0]["bytes_accessed"], 128)
    self.assertEqual(rows[2]["self_time_ns"], 10)
    self.assertEqual(rows[2]["inclusive_ns"], 100)
    # Device and category scoping: no `XLA Modules`, TPU:1 or h1 time.
    self.assertEqual({r["total_self_time_ns"] for r in rows}, {130})
    self.assertEqual({r["host"] for r in rows}, {"h0"})

  def test_orders_by_inclusive_time(self):
    rows = _top_ops(order_by="inclusive")
    self.assertEqual(_names(rows)[0], ("m", "while.1"))
    # The total stays a self time total.
    self.assertEqual(rows[0]["total_self_time_ns"], 130)

  def test_limit(self):
    rows = _top_ops(limit=1)
    self.assertEqual(_names(rows), [("m", "fusion.1")])
    self.assertEqual(rows[0]["total_self_time_ns"], 130)

  def test_sparse_core_name_from_kernel_name(self):
    rows = _top_ops(device="TPU:0 SparseCore 0", category="Sparse Core Ops")
    self.assertEqual(_names(rows), [("", "sc.1")])
    self.assertEqual(rows[0]["self_time_ns"], 70)
    self.assertEqual(rows[0]["flops"], 0)

  def test_without_host_column_merges_hosts(self):
    rows = _top_ops(host=None)
    self.assertEqual(_names(rows)[0], ("m", "fusion.1"))
    self.assertEqual(rows[0]["self_time_ns"], 590)
    self.assertEqual({r["host"] for r in rows}, {""})

  def test_no_rows(self):
    self.assertEqual(_top_ops(device="TPU:7"), [])

  def test_hlo_meta_join_on_module_and_op(self):
    rows = _top_ops(hlo_meta_table="HloOpMeta")
    by_name = {(r["hlo_module"], r["op"]): r for r in rows}
    self.assertLen(rows, 3)
    self.assertEqual(by_name[("m", "fusion.1")]["op_category"], "convolution")
    self.assertEqual(by_name[("m", "fusion.1")]["self_time_ns"], 90)
    self.assertEqual(by_name[("m", "fusion.1")]["occurrences"], 2)
    self.assertEqual(by_name[("m2", "fusion.1")]["op_category"], "loop fusion")
    self.assertEqual(by_name[("m", "while.1")]["op_category"], "")

  def test_googlesql_dialect_uses_at_parameters(self):
    sql = events_db_queries.build_top_ops_query(
        "googlesql", host_column="h", filter_host=True
    )
    self.assertIn("device = @device", sql)
    self.assertIn("category = @category", sql)
    self.assertIn("h = @host", sql)
    self.assertIn("AS INT64", sql)
    self.assertNotIn("$", sql)

  @parameterized.named_parameters(
      ("bad_dialect", {"dialect": "sqlite"}),
      ("bad_order_by", {"order_by": "total"}),
      ("zero_limit", {"limit": 0}),
      ("big_limit", {"limit": events_db_queries.MAX_LIMIT + 1}),
      ("filter_host_without_column", {"filter_host": True}),
  )
  def test_invalid_arguments(self, kwargs):
    dialect = kwargs.pop("dialect", "duckdb")
    with self.assertRaises(ValueError):
      events_db_queries.build_top_ops_query(dialect, **kwargs)


class InfoQueryTest(absltest.TestCase):

  def test_lists_hosts_devices_and_categories(self):
    rows = _run(
        events_db_queries.build_info_query("duckdb", host_column="host"), {}
    )
    keys = [(r["host"], r["device"], r["category"]) for r in rows]
    self.assertEqual(
        keys,
        [
            ("h0", "TPU:0", "XLA Modules"),
            ("h0", "TPU:0", "XLA Ops"),
            ("h0", "TPU:0 SparseCore 0", "Sparse Core Ops"),
            ("h0", "TPU:1", "XLA Ops"),
            ("h1", "TPU:0", "XLA Ops"),
        ],
    )
    ops = rows[1]
    self.assertEqual(ops["events"], 4)
    self.assertEqual(ops["self_time_ns"], 130)
    self.assertEqual(ops["first_start_ns"], 0)
    self.assertEqual(ops["last_end_ns"], 230)

  def test_without_host_column(self):
    rows = _run(events_db_queries.build_info_query("duckdb"), {})
    self.assertEqual({r["host"] for r in rows}, {""})
    self.assertLen(rows, 4)


if __name__ == "__main__":
  absltest.main()
