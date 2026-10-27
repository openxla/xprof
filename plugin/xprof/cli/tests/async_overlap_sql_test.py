"""Tests that the Events DB async overlap SQL matches `async_overlap.analyze`."""

import random
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import duckdb

from xprof.cli.internal import async_overlap
from xprof.cli.internal import async_overlap_sql

_TC = "TPU:0"
_SC = "TPU:0 SparseCore 0"
_CATEGORY = {_TC: "XLA Ops", _SC: "Sparse Core Ops"}

# (device, hlo_op, kernel_name, start_ns, end_ns, category)
_Row = tuple[str, str | None, str, int, int, str]


def _row(
    device: str,
    name: str,
    start_ns: int,
    end_ns: int,
    *,
    category: str | None = None,
    in_kernel_name: bool = False,
) -> _Row:
  """An Events row; SparseCore rows carry the name in `kernel_name` only."""
  in_kernel_name = in_kernel_name or device == _SC
  return (
      device,
      None if in_kernel_name else name,
      name if in_kernel_name else f"raw.{name}",
      start_ns,
      end_ns,
      category or _CATEGORY[device],
  )


def _run_sql(rows: list[_Row], **kwargs: Any) -> list[dict[str, Any]]:
  with duckdb.connect(":memory:") as con:
    con.execute(
        "CREATE TABLE Events (device VARCHAR, hlo_op VARCHAR, kernel_name"
        " VARCHAR, start_ns UBIGINT, end_ns UBIGINT, category VARCHAR)"
    )
    if rows:
      con.executemany("INSERT INTO Events VALUES (?, ?, ?, ?, ?, ?)", rows)
    cursor = con.execute(async_overlap_sql.build_query("duckdb", **kwargs))
    columns = [d[0] for d in cursor.description]
    return [dict(zip(columns, r)) for r in cursor.fetchall()]


def _via_sql(rows: list[_Row], **kwargs: Any) -> dict[str, Any]:
  thresholds = {
      k: v
      for k, v in kwargs.items()
      if k in ("hidden_threshold", "exposed_threshold")
  }
  return async_overlap_sql.assemble(_run_sql(rows, **kwargs), **thresholds)


def _via_python(rows: list[_Row], **kwargs: Any) -> dict[str, Any]:
  line_of = {"XLA Ops": "XLA Ops", "Sparse Core Ops": "Sparse Core Ops"}
  events = [
      async_overlap.TimelineEvent(
          plane=f"/device:{device}",
          line=line_of.get(category, category),
          name=hlo_op or kernel_name,
          start_ps=start * 1000,
          duration_ps=(end - start) * 1000,
      )
      for device, hlo_op, kernel_name, start, end, category in rows
  ]
  result = async_overlap.analyze(events, **kwargs)
  result["concurrency"] = []
  return result


class AsyncOverlapSqlTest(parameterized.TestCase):

  def assert_same_as_python(self, rows: list[_Row], **kwargs: Any) -> None:
    self.assertEqual(_via_sql(rows, **kwargs), _via_python(rows, **kwargs))

  def test_pairs_fifo_and_counts_unpaired(self):
    rows = [
        # Done op before any start: unpaired.
        _row(_TC, "all-gather-done.1", 0, 5),
        # Two overlapping all-gather.1 ops pair first-in, first-out.
        _row(_TC, "all-gather-start.1", 10, 11),
        _row(_TC, "all-gather-start.1", 20, 21),
        _row(_TC, "all-gather-done.1", 40, 41),
        _row(_TC, "all-gather-done.1", 60, 61),
        # Start op that never finishes: unpaired.
        _row(_TC, "all-gather-start.1", 90, 91),
        # send / recv start without a suffix; send-done finishes send.2.
        _row(_TC, "send.2", 12, 13),
        _row(_TC, "send-done.2", 70, 71),
        # Compute covering parts of the windows, nested and touching.
        _row(_TC, "fusion.1", 15, 35),
        _row(_TC, "fusion.2", 18, 22),
        _row(_TC, "fusion.3", 35, 38),
        _row(_TC, "fusion.4", 55, 59),
        # Not an op category: ignored.
        _row(_TC, "fusion.5", 40, 90, category="XLA Modules"),
    ]
    result = _via_sql(rows)
    self.assertEqual(result["summary"]["count"], 3)
    self.assertEqual(result["summary"]["unpaired_ops"], 2)
    self.assertEqual(
        result["by_plane"]["/device:TPU:0"]["window_source"],
        async_overlap_sql.WINDOW_SOURCE,
    )
    self.assert_same_as_python(rows)

  def test_hlo_text_names_and_kernel_name_fallback(self):
    rows = [
        _row(_TC, "%copy-start.3 = (f32[8]) copy-start(f32[8] %p)", 0, 1),
        _row(_TC, "copy-done.3", 99, 100, in_kernel_name=True),
        _row(_TC, "%fusion.9 = f32[8] fusion()", 10, 60),
    ]
    result = _via_sql(rows)
    self.assertEqual(result["top_exposed"][0]["name"], "copy-start.3")
    self.assertEqual(list(result["by_op"]), ["copy"])
    self.assert_same_as_python(rows)

  def test_devices_are_independent(self):
    rows = [
        _row(_TC, "all-reduce-start", 0, 1),
        _row(_SC, "fusion.1", 0, 100),
        _row(_TC, "all-reduce-done", 99, 100),
        _row(_SC, "all-reduce-start", 10, 11),
        _row(_SC, "all-reduce-done", 50, 51),
    ]
    result = _via_sql(rows)
    self.assertEqual(
        result["by_plane"]["/device:TPU:0"][async_overlap.EXPOSED], 1
    )
    self.assertEqual(
        result["by_plane"]["/device:TPU:0 SparseCore 0"][async_overlap.HIDDEN],
        1,
    )
    self.assert_same_as_python(rows)

  def test_no_async_ops_returns_no_rows(self):
    rows = [_row(_TC, "fusion.1", 0, 100)]
    self.assertEqual(_run_sql(rows), [])
    self.assert_same_as_python(rows)

  @parameterized.parameters(
      dict(top_k=0, hidden_threshold=0.9, exposed_threshold=0.1),
      dict(top_k=3, hidden_threshold=0.5, exposed_threshold=0.5 - 1e-9),
      dict(top_k=1000, hidden_threshold=1.0, exposed_threshold=0.0),
  )
  def test_random_traces_match_python(self, **kwargs):
    rng = random.Random(1234 + kwargs["top_k"])
    families = ["all-gather", "copy", "collective-permute", "fusion.7.call"]
    for _ in range(20):
      rows = []
      # Async events get distinct times so the time order is unambiguous.
      times = iter(rng.sample(range(0, 100_000, 3), 200))
      for device in (_TC, _SC):
        for _ in range(rng.randint(0, 60)):
          start = rng.randrange(0, 100_000)
          rows.append(
              _row(
                  device,
                  f"fusion.{rng.randrange(9)}",
                  start,
                  start + rng.randint(0, 3_000),
              )
          )
        for _ in range(rng.randint(0, 25)):
          family = rng.choice(families)
          suffix = rng.choice(["", ".1", ".2"])
          role = rng.choice(["start", "start", "done"])
          if rng.random() < 0.1:
            name = rng.choice(["send", "recv"]) + suffix
          else:
            name = f"{family}-{role}{suffix}"
          start = next(times)
          rows.append(_row(device, name, start, start + rng.randint(1, 2)))
      self.assert_same_as_python(rows, **kwargs)

  def test_invalid_arguments_raise(self):
    unsupported_dialect: Any = "sqlite"
    with self.assertRaisesRegex(ValueError, "dialect"):
      async_overlap_sql.build_query(unsupported_dialect)
    with self.assertRaisesRegex(ValueError, "thresholds"):
      async_overlap_sql.build_query(
          "googlesql", hidden_threshold=0.1, exposed_threshold=0.2
      )
    with self.assertRaisesRegex(ValueError, "host_column"):
      async_overlap_sql.build_query("googlesql", host_param="host")

  def test_googlesql_query_uses_raw_strings_and_host_parameter(self):
    sql = async_overlap_sql.build_query(
        "googlesql", host_column="h(x)", host_param="host"
    )
    self.assertIn("REGEXP_CONTAINS(name, r'^.+-done([.][0-9]+)?$')", sql)
    self.assertIn(r"r'\1\2'", sql)
    self.assertIn("AND h(x) = @host", sql)
    self.assertNotIn("regexp_matches", sql)

  def test_assemble_prefixes_planes_with_host_when_several(self):
    base = dict(
        row_type="w",
        device="TPU:0",
        pair_key="copy.1",
        rk=0,
        name="",
        n=1,
        span_ns=10,
        covered_ns=10,
        hidden_n=1,
        exposed_n=0,
        start_ns=0,
    )
    result = async_overlap_sql.assemble(
        [dict(base, host="a"), dict(base, host="b", rk=1, name="copy-start.1")]
    )
    self.assertEqual(
        list(result["by_plane"]), ["a:/device:TPU:0", "b:/device:TPU:0"]
    )
    self.assertEqual(result["top_exposed"][0]["host"], "b")
    self.assertEqual(result["summary"]["count"], 2)
    self.assertEqual(result["summary"]["span_ps"], 20_000)


if __name__ == "__main__":
  absltest.main()
