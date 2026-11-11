from collections.abc import Mapping, Sequence
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import duckdb

from xprof.cli.internal import events_db_queries

_TPU = "TPU:0"
_SPARSE_CORE = "TPU:0 SparseCore 0"
_XLA_OPS = "XLA Ops"
_SPARSE_CORE_OPS = "Sparse Core Ops"
_HOST = "h0"
_FUSION = ("m", "fusion.1")
_FUSION_M2 = ("m2", "fusion.1")
_WHILE = ("m", "while.1")
# Self time of all `XLA Ops` rows of `_TPU` on `_HOST`.
_TOTAL_SELF_TIME_NS = 130

# (host, device, category, hlo_module, hlo_op, kernel_name, start_ns, end_ns,
#  self_time_ns, flops, memory_accessed)
_EVENTS = (
    # A `while` op that contains two ops: big inclusive time, small self time.
    (_HOST, _TPU, _XLA_OPS, "m", "while.1", "", 0, 100, 10, 0, 0),
    (_HOST, _TPU, _XLA_OPS, "m", "fusion.1", "", 10, 50, 40, 1000, 64),
    (_HOST, _TPU, _XLA_OPS, "m", "fusion.1", "", 50, 100, 50, 1000, 64),
    # The same op name in another module is a different op.
    (_HOST, _TPU, _XLA_OPS, "m2", "fusion.1", "", 200, 230, 30, 10, 8),
    # SparseCore ops carry the name in `kernel_name` only.
    (
        _HOST,
        _SPARSE_CORE,
        _SPARSE_CORE_OPS,
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
    (_HOST, _TPU, "XLA Modules", "m", "", "m", 0, 100, 100, 0, 0),
    (_HOST, "TPU:1", _XLA_OPS, "m", "fusion.1", "", 0, 999, 999, 0, 0),
    ("h1", _TPU, _XLA_OPS, "m", "fusion.1", "", 0, 500, 500, 0, 0),
)

# (module_name, op_name, op_category). `fusion.1` of `m` has two rows that
# must not double its time; `while.1` has none.
_HLO_META = (
    ("m", "fusion.1", "convolution"),
    ("m", "fusion.1", "convolution"),
    ("m2", "fusion.1", "loop fusion"),
)


def _run(sql: str, parameters: Mapping[str, str]) -> list[dict[str, Any]]:
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
    cursor = con.execute(sql, dict(parameters))
    columns = [d[0] for d in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def _top_ops(
    device: str = _TPU,
    category: str = _XLA_OPS,
    host: str | None = _HOST,
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


def _names(rows: Sequence[Mapping[str, Any]]) -> list[tuple[str, str]]:
  return [(r["hlo_module"], r["op"]) for r in rows]


def _by_name(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], Mapping[str, Any]]:
  return {(r["hlo_module"], r["op"]): r for r in rows}


def _fields(row: Mapping[str, Any], *fields: str) -> dict[str, Any]:
  return {field: row[field] for field in fields}


class TopOpsQueryTest(parameterized.TestCase):

  def test_orders_by_self_time(self):
    self.assertEqual(_names(_top_ops()), [_FUSION, _FUSION_M2, _WHILE])

  def test_sums_repeated_op(self):
    fusion = _by_name(_top_ops())[_FUSION]

    self.assertEqual(
        _fields(
            fusion,
            "occurrences",
            "self_time_ns",
            "inclusive_ns",
            "flops",
            "bytes_accessed",
        ),
        dict(
            occurrences=2,
            self_time_ns=90,
            inclusive_ns=90,
            flops=2000,
            bytes_accessed=128,
        ),
    )

  def test_parent_op_has_small_self_and_big_inclusive_time(self):
    while_op = _by_name(_top_ops())[_WHILE]

    self.assertEqual(
        _fields(while_op, "self_time_ns", "inclusive_ns"),
        dict(self_time_ns=10, inclusive_ns=100),
    )

  def test_total_counts_only_one_device_category_and_host(self):
    # No `XLA Modules`, TPU:1 or h1 time.
    totals = {r["total_self_time_ns"] for r in _top_ops()}

    self.assertEqual(totals, {_TOTAL_SELF_TIME_NS})

  def test_rows_carry_filtered_host(self):
    self.assertEqual({r["host"] for r in _top_ops()}, {_HOST})

  def test_orders_by_inclusive_time(self):
    rows = _top_ops(order_by="inclusive")

    self.assertEqual(_names(rows)[0], _WHILE)

  def test_inclusive_order_keeps_self_time_total(self):
    rows = _top_ops(order_by="inclusive")

    self.assertEqual(rows[0]["total_self_time_ns"], _TOTAL_SELF_TIME_NS)

  def test_limit(self):
    rows = _top_ops(limit=1)

    self.assertEqual(_names(rows), [_FUSION])

  def test_limit_keeps_total_of_all_ops(self):
    rows = _top_ops(limit=1)

    self.assertEqual(rows[0]["total_self_time_ns"], _TOTAL_SELF_TIME_NS)

  def test_sparse_core_name_from_kernel_name(self):
    rows = _top_ops(device=_SPARSE_CORE, category=_SPARSE_CORE_OPS)

    self.assertEqual(
        [_fields(r, "hlo_module", "op", "self_time_ns", "flops") for r in rows],
        [dict(hlo_module="", op="sc.1", self_time_ns=70, flops=0)],
    )

  def test_without_host_column_merges_hosts(self):
    rows = _top_ops(host=None)

    self.assertEqual(
        _fields(rows[0], "host", "hlo_module", "op", "self_time_ns"),
        dict(host="", hlo_module="m", op="fusion.1", self_time_ns=590),
    )

  def test_no_rows(self):
    self.assertEmpty(_top_ops(device="TPU:7"))

  def test_hlo_meta_join_does_not_duplicate_ops(self):
    rows = _top_ops(hlo_meta_table="HloOpMeta")

    self.assertEqual(
        {
            name: _fields(row, "occurrences", "self_time_ns")
            for name, row in _by_name(rows).items()
        },
        {
            _FUSION: dict(occurrences=2, self_time_ns=90),
            _FUSION_M2: dict(occurrences=1, self_time_ns=30),
            _WHILE: dict(occurrences=1, self_time_ns=10),
        },
    )

  def test_hlo_meta_join_on_module_and_op(self):
    rows = _top_ops(hlo_meta_table="HloOpMeta")

    self.assertEqual(
        {name: row["op_category"] for name, row in _by_name(rows).items()},
        {_FUSION: "convolution", _FUSION_M2: "loop fusion", _WHILE: ""},
    )

  @parameterized.parameters(
      "device = @device", "category = @category", "h = @host", "AS INT64"
  )
  def test_googlesql_dialect(self, expected_sql):
    sql = events_db_queries.build_top_ops_query(
        "googlesql", host_column="h", filter_host=True
    )

    self.assertIn(expected_sql, sql)

  def test_googlesql_dialect_has_no_duckdb_parameters(self):
    sql = events_db_queries.build_top_ops_query(
        "googlesql", host_column="h", filter_host=True
    )

    self.assertNotIn("$", sql)

  @parameterized.named_parameters(
      dict(
          testcase_name="bad_dialect",
          dialect="sqlite",
          kwargs=dict(),
          message="unknown SQL dialect",
      ),
      dict(
          testcase_name="bad_order_by",
          dialect="duckdb",
          kwargs=dict(order_by="total"),
          message="order_by must be",
      ),
      dict(
          testcase_name="zero_limit",
          dialect="duckdb",
          kwargs=dict(limit=0),
          message="limit must be in",
      ),
      dict(
          testcase_name="big_limit",
          dialect="duckdb",
          kwargs=dict(limit=events_db_queries.MAX_LIMIT + 1),
          message="limit must be in",
      ),
      dict(
          testcase_name="filter_host_without_column",
          dialect="duckdb",
          kwargs=dict(filter_host=True),
          message="filter_host requires host_column",
      ),
  )
  def test_invalid_arguments(self, dialect, kwargs, message):
    with self.assertRaisesRegex(ValueError, message):
      events_db_queries.build_top_ops_query(dialect, **kwargs)


class InfoQueryTest(absltest.TestCase):

  def test_lists_hosts_devices_and_categories(self):
    rows = _run(
        events_db_queries.build_info_query("duckdb", host_column="host"), {}
    )

    self.assertEqual(
        [(r["host"], r["device"], r["category"]) for r in rows],
        [
            (_HOST, _TPU, "XLA Modules"),
            (_HOST, _TPU, _XLA_OPS),
            (_HOST, _SPARSE_CORE, _SPARSE_CORE_OPS),
            (_HOST, "TPU:1", _XLA_OPS),
            ("h1", _TPU, _XLA_OPS),
        ],
    )

  def test_summarizes_one_device_and_category(self):
    rows = _run(
        events_db_queries.build_info_query("duckdb", host_column="host"), {}
    )
    xla_ops = next(
        r
        for r in rows
        if (r["host"], r["device"], r["category"]) == (_HOST, _TPU, _XLA_OPS)
    )

    self.assertEqual(
        _fields(
            xla_ops, "events", "self_time_ns", "first_start_ns", "last_end_ns"
        ),
        dict(
            events=4,
            self_time_ns=_TOTAL_SELF_TIME_NS,
            first_start_ns=0,
            last_end_ns=230,
        ),
    )

  def test_without_host_column_merges_hosts(self):
    rows = _run(events_db_queries.build_info_query("duckdb"), {})

    self.assertEqual(
        {(r["host"], r["device"], r["category"]) for r in rows},
        {
            ("", _TPU, "XLA Modules"),
            ("", _TPU, _XLA_OPS),
            ("", _SPARSE_CORE, _SPARSE_CORE_OPS),
            ("", "TPU:1", _XLA_OPS),
        },
    )


if __name__ == "__main__":
  absltest.main()
