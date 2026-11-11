"""Tests async overlap over an OSS Events DB parquet file built from a trace."""

from collections.abc import Mapping
import os
import pathlib
import shutil
import tempfile
from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized

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

_TOP_K = 10
_DURATION_FIELDS = ("span_ps", "exposed_ps")
# The Events DB truncates times to nanoseconds, so a window's start time may
# differ by up to 1 ns and its span by up to 2 ns.
_MAX_START_ERROR_PS = 1000
_MAX_SPAN_ERROR_PS = 2000


def _copy_demo_trace(root: pathlib.Path) -> pathlib.Path:
  """Copies the demo trace into `root` and returns the copy's path."""
  trace = runfiles.Create().Rlocation(_DEMO_TRACE_REL_PATH)
  if not trace or not os.path.exists(trace):
    raise FileNotFoundError(f"Missing test trace: {_DEMO_TRACE_REL_PATH!r}")
  copy = root / "trace.xplane.pb"
  shutil.copyfile(trace, copy)
  return copy


def _make_cache_dir(root: pathlib.Path) -> pathlib.Path:
  cache_dir = root / "cache"
  cache_dir.mkdir()
  return cache_dir


def _buckets(
    result: Mapping[str, Any], group: str
) -> Mapping[str, Mapping[str, Any]]:
  """Returns the buckets of `group`; `summary` is one bucket of that name."""
  if group == "summary":
    return {group: result[group]}
  return result[group]


def _exact_fields(
    buckets: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
  """Returns the fields of each bucket that both backends compute exactly."""
  return {
      key: {
          field: value
          for field, value in bucket.items()
          if not field.endswith("_ps") and field != "overlap_ratio"
      }
      for key, bucket in buckets.items()
  }


def _duration_errors(
    got: Mapping[str, Mapping[str, Any]],
    want: Mapping[str, Mapping[str, Any]],
) -> dict[str, tuple[int, int]]:
  """Returns the bucket durations that differ by more than ns rounding.

  Every window endpoint and every overlapping compute boundary is truncated to
  nanoseconds, so the error grows with the window count. It is also bounded by
  1e-3 of the bucket span.

  Args:
    got: Buckets computed by the Events DB backend.
    want: The same buckets computed from the XSpace.

  Returns:
    `(got, want)` durations keyed by `<bucket>.<field>`, for every duration
    outside the allowed error.
  """
  errors = {}
  for key, bucket in want.items():
    max_error = max(2000 * bucket["count"], 1e-3 * bucket["span_ps"])
    for field in _DURATION_FIELDS:
      if abs(got[key][field] - bucket[field]) > max_error:
        errors[f"{key}.{field}"] = (got[key][field], bucket[field])
  return errors


def _window_identity(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
  return (row["plane"], row["name"], row["op"], row["class"])


def _window_time_errors(
    got: list[Mapping[str, Any]], want: list[Mapping[str, Any]]
) -> list[tuple[str, int, int]]:
  """Returns the window times that differ by more than ns rounding."""
  errors = []
  for got_row, want_row in zip(got, want):
    for field, max_error in (
        ("start_ps", _MAX_START_ERROR_PS),
        ("span_ps", _MAX_SPAN_ERROR_PS),
    ):
      if abs(got_row[field] - want_row[field]) > max_error:
        errors.append((field, got_row[field], want_row[field]))
  return errors


class OssAsyncOverlapEventsDbTest(parameterized.TestCase):
  """Compares the Events DB backend with the XSpace analysis of one trace."""

  _via_sql: dict[str, Any]
  _via_python: dict[str, Any]

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    root = pathlib.Path(tempfile.mkdtemp(dir=absltest.TEST_TMPDIR.value))
    cache_dir = _make_cache_dir(root)
    trace = str(_copy_demo_trace(root))
    with mock.patch.object(
        decorators, "get_cache_dir", autospec=True, return_value=cache_dir
    ):
      sql = async_overlap_sql.build_query(events_db_sql.DIALECT, top_k=_TOP_K)
      cls._via_sql = async_overlap_sql.assemble(
          events_db_sql.run_query(trace, sql)
      )
      # The Events DB has no `Async XLA Ops` category, so compare against the
      # Python analysis of the same trace without that line.
      events = [
          e
          for e in xplane_tools.iter_timeline_events(trace)
          if e.line != async_overlap.ASYNC_OPS_LINE
      ]
      cls._via_python = async_overlap.analyze(events, top_k=_TOP_K)

  def test_finds_windows(self):
    self.assertGreater(self._via_sql["summary"]["count"], 0)

  @parameterized.parameters("summary", "by_op", "by_plane")
  def test_counts_match(self, group):
    self.assertEqual(
        _exact_fields(_buckets(self._via_sql, group)),
        _exact_fields(_buckets(self._via_python, group)),
    )

  @parameterized.parameters("summary", "by_op", "by_plane")
  def test_durations_match_up_to_rounding(self, group):
    self.assertEmpty(
        _duration_errors(
            _buckets(self._via_sql, group), _buckets(self._via_python, group)
        )
    )

  def test_top_exposed_windows_match(self):
    self.assertEqual(
        [_window_identity(r) for r in self._via_sql["top_exposed"]],
        [_window_identity(r) for r in self._via_python["top_exposed"]],
    )

  def test_top_exposed_times_match_up_to_rounding(self):
    self.assertEmpty(
        _window_time_errors(
            self._via_sql["top_exposed"], self._via_python["top_exposed"]
        )
    )


class OssEventsDbBypassCacheTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    root = pathlib.Path(self.create_tempdir().full_path)
    self.enter_context(
        mock.patch.object(
            decorators,
            "get_cache_dir",
            autospec=True,
            return_value=_make_cache_dir(root),
        )
    )
    self._trace = str(_copy_demo_trace(root))
    self._create = self.enter_context(
        mock.patch.object(
            events_db_sql.events_db_tool,
            "create_events_db",
            autospec=True,
            side_effect=events_db_sql.events_db_tool.create_events_db,
        )
    )

  def test_reuses_parquet_file_by_default(self):
    events_db_sql.run_query(self._trace, "SELECT 1")

    self._create.assert_called_once_with(self._trace, bypass_cache=False)

  def test_bypass_cache_rebuilds_parquet_file(self):
    events_db_sql.run_query(self._trace, "SELECT 1", bypass_cache=True)

    self._create.assert_called_once_with(self._trace, bypass_cache=True)


if __name__ == "__main__":
  absltest.main()
