"""Tests async overlap over an OSS Events DB parquet file built from a trace."""

import os
import pathlib
import shutil
from unittest import mock

from absl.testing import absltest

from python.runfiles import runfiles

from xprof.cli.internal import async_overlap
from xprof.cli.internal import async_overlap_sql
from xprof.cli.internal import decorators
from xprof.cli.internal.oss import events_db_sql
from xprof.cli.internal.oss import xplane_tools

_DEMO_TRACE_REL_PATH = (
    "org_xprof/xprof/demo/plugins/profile/v6e-4-training/"
    "t1v-n-9bfa07b4-w-0.xplane.pb"
)


class OssAsyncOverlapEventsDbTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    root = pathlib.Path(self.create_tempdir().full_path)
    cache_dir = root / "cache"
    cache_dir.mkdir()
    self.enter_context(
        mock.patch.object(decorators, "get_cache_dir", return_value=cache_dir)
    )
    trace = runfiles.Create().Rlocation(_DEMO_TRACE_REL_PATH)
    if not trace or not os.path.exists(trace):
      raise FileNotFoundError(f"Missing test trace: {_DEMO_TRACE_REL_PATH!r}")
    self._trace = root / "trace.xplane.pb"
    shutil.copyfile(trace, self._trace)

  def test_sql_matches_python_on_start_done_pairs(self):
    sql = async_overlap_sql.build_query(events_db_sql.DIALECT, top_k=10)
    rows = events_db_sql.run_query(str(self._trace), sql)
    via_sql = async_overlap_sql.assemble(rows)

    # The Events DB has no `Async XLA Ops` category, so compare against the
    # Python analysis of the same trace without that line.
    events = [
        e
        for e in xplane_tools.iter_timeline_events(str(self._trace))
        if e.line != async_overlap.ASYNC_OPS_LINE
    ]
    via_python = async_overlap.analyze(events, top_k=10)

    self.assertGreater(via_sql["summary"]["count"], 0)
    self.assert_buckets_match(
        {"summary": via_sql["summary"]}, {"summary": via_python["summary"]}
    )
    self.assert_buckets_match(via_sql["by_op"], via_python["by_op"])
    self.assert_buckets_match(via_sql["by_plane"], via_python["by_plane"])
    self.assertEqual(
        [_identity(r) for r in via_sql["top_exposed"]],
        [_identity(r) for r in via_python["top_exposed"]],
    )
    for got, want in zip(via_sql["top_exposed"], via_python["top_exposed"]):
      # The Events DB truncates start times to nanoseconds.
      self.assertAlmostEqual(got["start_ps"], want["start_ps"], delta=1000)
      self.assertAlmostEqual(got["span_ps"], want["span_ps"], delta=2000)

  def assert_buckets_match(self, got, want):
    """Counts match exactly; durations up to nanosecond rounding."""
    self.assertEqual(got.keys(), want.keys())
    for key in want:
      exact = {k: v for k, v in want[key].items() if not k.endswith("_ps")}
      exact.pop("overlap_ratio", None)
      self.assertEqual(
          {k: got[key][k] for k in exact}, exact, msg=f"bucket {key}"
      )
      # Every window endpoint and every overlapping compute boundary is
      # truncated to nanoseconds, so the error grows with the window count.
      # Bound it by 1e-3 of the total span as well.
      span = want[key]["span_ps"]
      delta = max(2000 * want[key]["count"], 1e-3 * span)
      for field in ("span_ps", "exposed_ps"):
        self.assertAlmostEqual(
            got[key][field],
            want[key][field],
            delta=delta,
            msg=f"{key}.{field}",
        )


def _identity(row):
  return (row["plane"], row["name"], row["op"], row["class"])


if __name__ == "__main__":
  absltest.main()
