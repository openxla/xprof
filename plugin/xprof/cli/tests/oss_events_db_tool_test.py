# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from collections.abc import Mapping, Sequence
import functools
import json
import os
import pathlib
import tempfile
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized

from python.runfiles import runfiles

from xprof.convert.events_db.python import pywrap_events_db_c_api as events_db
from xprof.cli.internal import decorators
from xprof.cli.internal.oss import xprof_client
from xprof.cli.tools.oss import events_db_tool

_DEMO_TRACE_REL_PATH = (
    "org_xprof/xprof/demo/plugins/profile/v6e-4-training/"
    "t1v-n-9bfa07b4-w-0.xplane.pb"
)


@functools.cache
def _trace_bytes() -> bytes:
  r = runfiles.Create()
  trace_file = r.Rlocation(_DEMO_TRACE_REL_PATH)
  if not trace_file or not os.path.exists(trace_file):
    raise FileNotFoundError(
        f"Test fixture not found via runfiles: {_DEMO_TRACE_REL_PATH!r}"
    )
  return pathlib.Path(trace_file).read_bytes()


@functools.cache
def _parquet_bytes() -> bytes:
  with tempfile.TemporaryDirectory() as tmp_dir:
    tmp_path = pathlib.Path(tmp_dir)
    trace_path = tmp_path / "trace.xplane.pb"
    parquet_path = tmp_path / "events.parquet"
    trace_path.write_bytes(_trace_bytes())
    events_db.xspace_to_parquet(
        input_path=trace_path,
        output_path=parquet_path,
        options=events_db.ParquetExportOptions(
            compression_type=events_db.ArrowCompressionType.ZSTD,
            compression_level=3,
        ),
    )
    return parquet_path.read_bytes()


class OssCreateEventsDbToolTest(parameterized.TestCase):
  """Tests for create_events_db tool in OSS mode."""

  def setUp(self) -> None:
    super().setUp()
    self._root = pathlib.Path(self.create_tempdir().full_path)
    self._cache_dir = self._root / "cache"
    self._cache_dir.mkdir()
    self.enter_context(
        mock.patch.object(
            decorators, "get_cache_dir", return_value=self._cache_dir
        )
    )

    # 1. Directory with a single trace file
    self._single_dir = self._root / "single_run"
    self._single_dir.mkdir()
    self._single_trace = self._single_dir / "trace.xplane.pb"
    self._single_trace.write_bytes(_trace_bytes())

    # 2. Directory with multiple trace files (worker0 invalid, worker1 valid)
    self._multi_dir = self._root / "multi_run"
    self._multi_dir.mkdir()
    (self._multi_dir / "worker0.xplane.pb").write_bytes(b"invalid_proto_bytes")
    self._multi_worker1 = self._multi_dir / "worker1.xplane.pb"
    self._multi_worker1.write_bytes(_trace_bytes())

    # 3. Empty directory
    self._empty_dir = self._root / "empty_run"
    self._empty_dir.mkdir()

    # 4. TensorBoard logdir structure with session "session_1"
    self._logdir = self._root / "tb_logdir"
    self._session_dir = self._logdir / "plugins" / "profile" / "session_1"
    self._session_dir.mkdir(parents=True)
    (self._session_dir / "trace.xplane.pb").write_bytes(_trace_bytes())
    xprof_client.get_client().set_logdir(str(self._logdir))

    self._logdirs: Mapping[str, str | None] = {
        "configured": str(self._logdir),
        "none": None,
    }
    self._sources: Mapping[str, str | None] = {
        "file": str(self._single_trace),
        "single_dir": str(self._single_dir),
        "specific_file_in_multi_dir": str(self._multi_worker1),
        "multi_dir": str(self._multi_dir),
        "session_id": "session_1",
        "empty_dir": str(self._empty_dir),
        "nonexistent": str(self._root / "does_not_exist"),
        "none": None,
    }
    self._expected_sources: Mapping[str, pathlib.Path] = {
        "file": self._single_trace,
        "single_dir": self._single_trace,
        "specific_file_in_multi_dir": self._multi_worker1,
        "session_id": self._session_dir / "trace.xplane.pb",
        "none": self._session_dir / "trace.xplane.pb",
    }

  def tearDown(self) -> None:
    xprof_client.get_client().set_logdir(None)
    super().tearDown()

  @parameterized.named_parameters(
      dict(
          testcase_name="below_display_limit",
          num_files=3,
          expected_substrings=(
              "Multiple (3) trace files found in /test/search/dir:",
              "file_000.xplane.pb",
              "file_002.xplane.pb",
              "Multi-file Events DB generation is not supported yet.",
          ),
          excluded_substrings=("... and",),
      ),
      dict(
          testcase_name="exact_display_limit",
          num_files=10,
          expected_substrings=(
              "Multiple (10) trace files found in /test/search/dir:",
              "file_000.xplane.pb",
              "file_009.xplane.pb",
              "Multi-file Events DB generation is not supported yet.",
          ),
          excluded_substrings=("... and",),
      ),
      dict(
          testcase_name="exceeds_display_limit_truncates",
          num_files=12,
          expected_substrings=(
              "Multiple (12) trace files found in /test/search/dir:",
              "file_009.xplane.pb",
              "... and 2 more.",
          ),
          excluded_substrings=("file_010.xplane.pb", "file_011.xplane.pb"),
      ),
  )
  def test_create_multi_file_error_formatting(
      self,
      num_files: int,
      expected_substrings: Sequence[str],
      excluded_substrings: Sequence[str],
  ) -> None:
    files = tuple(
        f"/test/search/dir/file_{i:03d}.xplane.pb" for i in range(num_files)
    )
    err = events_db_tool._create_multi_file_error(
        pathlib.Path("/test/search/dir"), files
    )
    msg = str(err)

    self.assertIsInstance(err, NotImplementedError)
    self.assertContainsInOrder(expected_substrings, msg)
    self.assertEmpty([s for s in excluded_substrings if s in msg])

  @parameterized.named_parameters(
      dict(
          testcase_name="from_file",
          source_key="file",
          prepopulate=False,
          bypass_cache=False,
      ),
      dict(
          testcase_name="from_single_dir",
          source_key="single_dir",
          prepopulate=False,
          bypass_cache=False,
      ),
      dict(
          testcase_name="from_session_id_via_logdir",
          source_key="session_id",
          prepopulate=False,
          bypass_cache=False,
      ),
      dict(
          testcase_name="from_specific_file_in_multi_trace_dir",
          source_key="specific_file_in_multi_dir",
          prepopulate=False,
          bypass_cache=False,
      ),
      dict(
          testcase_name="from_none_source_via_logdir",
          source_key="none",
          prepopulate=False,
          bypass_cache=False,
      ),
      dict(
          testcase_name="regenerates_existing_file_when_bypass_cache_true",
          source_key="file",
          prepopulate=True,
          bypass_cache=True,
      ),
  )
  def test_create_events_db_generates_parquet(
      self,
      source_key: str,
      prepopulate: bool,
      bypass_cache: bool,
  ) -> None:
    session_id = self._sources[source_key]
    expected_source = self._expected_sources[source_key]
    expected_path = events_db_tool._cache_path_for_create(expected_source)
    if prepopulate:
      expected_path.parent.mkdir(parents=True, exist_ok=True)
      expected_path.write_bytes(b"PAR1_STALE_CACHED_DATA_PAR1")

    raw_res = events_db_tool.create_events_db(
        session_id=session_id,
        bypass_cache=bypass_cache,
    )
    res = json.loads(raw_res)

    self.assertEqual(res[events_db_tool._KEY_PATH], str(expected_path))
    self.assertFalse(res[events_db_tool._KEY_SKIPPED])
    content = expected_path.read_bytes()
    self.assertNotEqual(content, b"PAR1_STALE_CACHED_DATA_PAR1")
    self.assertTrue(content.startswith(b"PAR1"))
    self.assertTrue(content.endswith(b"PAR1"))

  @parameterized.named_parameters(
      dict(
          testcase_name="from_file",
          source_key="file",
      ),
      dict(
          testcase_name="from_single_dir",
          source_key="single_dir",
      ),
      dict(
          testcase_name="from_session_id_via_logdir",
          source_key="session_id",
      ),
      dict(
          testcase_name="from_specific_file_in_multi_dir",
          source_key="specific_file_in_multi_dir",
      ),
      dict(
          testcase_name="from_none_source_via_logdir",
          source_key="none",
      ),
  )
  def test_create_events_db_skips_existing_parquet(
      self, source_key: str
  ) -> None:
    session_id = self._sources[source_key]
    expected_source = self._expected_sources[source_key]
    expected_path = events_db_tool._cache_path_for_create(expected_source)
    sentinel = b"PAR1_PREEXISTING_SENTINEL_PAR1"
    expected_path.parent.mkdir(parents=True, exist_ok=True)
    expected_path.write_bytes(sentinel)

    raw_res = events_db_tool.create_events_db(
        session_id=session_id,
        bypass_cache=False,
    )
    res = json.loads(raw_res)

    self.assertEqual(res[events_db_tool._KEY_PATH], str(expected_path))
    self.assertTrue(res[events_db_tool._KEY_SKIPPED])
    self.assertEqual(expected_path.read_bytes(), sentinel)

  @parameterized.named_parameters(
      dict(
          testcase_name="empty_dir_raises_file_not_found",
          source_key="empty_dir",
          logdir_key="configured",
          expected_exception=FileNotFoundError,
          expected_regex=r"No \.xplane\.pb or \.xspace\.pb files found",
      ),
      dict(
          testcase_name="nonexistent_path_raises_file_not_found",
          source_key="nonexistent",
          logdir_key="configured",
          expected_exception=FileNotFoundError,
          expected_regex=r"not found",
      ),
      dict(
          testcase_name="no_source_and_no_logdir_raises_value_error",
          source_key="none",
          logdir_key="none",
          expected_exception=ValueError,
          expected_regex=r"Logdir not set",
      ),
      dict(
          testcase_name="multi_trace_dir_raises_not_implemented_error",
          source_key="multi_dir",
          logdir_key="configured",
          expected_exception=NotImplementedError,
          expected_regex=r"Multiple \(2\) trace files found",
      ),
  )
  def test_create_events_db_raises_error(
      self,
      source_key: str,
      logdir_key: str,
      expected_exception: type[Exception],
      expected_regex: str,
  ) -> None:
    xprof_client.get_client().set_logdir(self._logdirs[logdir_key])
    session_id = self._sources[source_key]

    with self.assertRaisesRegex(expected_exception, expected_regex):
      events_db_tool.create_events_db(
          session_id=session_id,
          bypass_cache=False,
      )


class OssQueryEventsDbToolTest(parameterized.TestCase):
  """Tests for query_events_db tool in OSS mode."""

  def setUp(self) -> None:
    super().setUp()
    self._root = pathlib.Path(self.create_tempdir().full_path)
    self._cache_dir = self._root / "cache"
    self._cache_dir.mkdir()
    self.enter_context(
        mock.patch.object(
            decorators, "get_cache_dir", return_value=self._cache_dir
        )
    )

    # 1. Directory with a single trace file
    self._single_dir = self._root / "single_run"
    self._single_dir.mkdir()
    self._single_trace = self._single_dir / "trace.xplane.pb"
    self._single_trace.write_bytes(_trace_bytes())

    # 2. TensorBoard logdir structure with session "session_1"
    self._logdir = self._root / "tb_logdir"
    self._session_dir = self._logdir / "plugins" / "profile" / "session_1"
    self._session_dir.mkdir(parents=True)
    self._session_trace = self._session_dir / "trace.xplane.pb"
    self._session_trace.write_bytes(_trace_bytes())
    xprof_client.get_client().set_logdir(str(self._logdir))

    # Pre-populate cached Parquet files to avoid redundant C++ XSpace-to-Parquet
    # conversions across query test cases.
    for trace_path in (self._single_trace, self._session_trace):
      cache_path = events_db_tool._cache_path_for_create(trace_path)
      cache_path.parent.mkdir(parents=True, exist_ok=True)
      cache_path.write_bytes(_parquet_bytes())
    self._single_db_path = events_db_tool._cache_path_for_create(
        self._single_trace
    )

    # 3. Unauthorized files for security sandbox tests
    self._secret_file = self._root / "secret.txt"
    self._secret_file.write_text("top_secret_data", encoding="utf-8")
    self._exfil_file = self._root / "exfil.json"

  def tearDown(self) -> None:
    xprof_client.get_client().set_logdir(None)
    super().tearDown()

  @parameterized.named_parameters(
      dict(testcase_name="empty", query=""),
      dict(testcase_name="whitespace", query="   \n\t  "),
      dict(testcase_name="only_line_comment", query="-- just a comment"),
      dict(testcase_name="only_block_comment", query="/* block comment */"),
      dict(testcase_name="only_semicolons", query=";;;"),
  )
  def test_query_events_db_raises_value_error_for_empty_query(
      self, query: str
  ) -> None:
    with self.assertRaisesRegex(ValueError, "SQL query must not be empty"):
      events_db_tool.query_events_db(
          session_id=str(self._single_trace),
          query=query,
      )

  def test_query_events_db_with_real_duckdb_counts_events(self) -> None:
    raw_res = events_db_tool.query_events_db(
        session_id=str(self._single_trace),
        query="SELECT count(*) AS total_count FROM Events",
    )
    res = json.loads(raw_res)
    self.assertIn(events_db_tool._KEY_PATH, res)
    self.assertFalse(res[events_db_tool._KEY_SKIPPED])
    res_file = pathlib.Path(res[events_db_tool._KEY_PATH])
    self.assertTrue(res_file.is_file())

    data = json.loads(res_file.read_text(encoding="utf-8"))
    self.assertLen(data, 1)
    self.assertIn("total_count", data[0])
    self.assertGreater(data[0]["total_count"], 0)

  def test_query_events_db_caches_and_skips_duplicate_query(self) -> None:
    raw_res1 = events_db_tool.query_events_db(
        session_id=str(self._single_trace),
        query="SELECT count(*) AS total_count FROM Events",
    )
    res1 = json.loads(raw_res1)
    self.assertFalse(res1[events_db_tool._KEY_SKIPPED])

    raw_res2 = events_db_tool.query_events_db(
        session_id=str(self._single_trace),
        query="SELECT count(*) AS total_count FROM Events",
        bypass_cache=False,
    )
    res2 = json.loads(raw_res2)
    self.assertEqual(
        res1[events_db_tool._KEY_PATH], res2[events_db_tool._KEY_PATH]
    )
    self.assertTrue(res2[events_db_tool._KEY_SKIPPED])

    raw_res3 = events_db_tool.query_events_db(
        session_id=str(self._single_trace),
        query="SELECT count(*) AS total_count FROM Events",
        bypass_cache=True,
    )
    res3 = json.loads(raw_res3)
    self.assertEqual(
        res1[events_db_tool._KEY_PATH], res3[events_db_tool._KEY_PATH]
    )
    self.assertFalse(res3[events_db_tool._KEY_SKIPPED])

  def test_query_events_db_generates_missing_parquet_automatically(
      self,
  ) -> None:
    self._single_db_path.unlink()
    self.assertFalse(self._single_db_path.exists())

    raw_res = events_db_tool.query_events_db(
        session_id=str(self._single_trace),
        query="SELECT category FROM Events LIMIT 3",
    )
    res = json.loads(raw_res)
    self.assertTrue(self._single_db_path.is_file())
    res_file = pathlib.Path(res[events_db_tool._KEY_PATH])
    self.assertTrue(res_file.is_file())

    data = json.loads(res_file.read_text(encoding="utf-8"))
    self.assertNotEmpty(data)
    self.assertIn("category", data[0])

  @parameterized.named_parameters(
      dict(
          testcase_name="single_positional_sql",
          args=("SELECT count(*) AS c FROM Events",),
      ),
      dict(
          testcase_name="single_positional_sql_with_leading_comment",
          args=("/* header */ SELECT count(*) AS c FROM Events",),
      ),
      dict(
          testcase_name="single_positional_sql_with_trailing_comment",
          args=("SELECT count(*) AS c FROM Events -- trailing comment",),
      ),
      dict(
          testcase_name="single_positional_sql_with_semicolon_and_comment",
          args=("SELECT count(*) AS c FROM Events; -- trailing comment",),
      ),
      dict(
          testcase_name="single_positional_from_first_sql",
          args=("FROM Events SELECT count(*) AS c",),
      ),
      dict(
          testcase_name="single_positional_sql_with_empty_second_arg",
          args=("SELECT count(*) AS c FROM Events", ""),
      ),
      dict(
          testcase_name="query_as_second_positional",
          args=("session_1", "SELECT count(*) AS c FROM Events"),
      ),
  )
  def test_query_events_db_positional_arguments_resolution(
      self, args: Sequence[str | None]
  ) -> None:
    raw_res = events_db_tool.query_events_db(*args)
    res = json.loads(raw_res)
    data = json.loads(
        pathlib.Path(res[events_db_tool._KEY_PATH]).read_text(encoding="utf-8")
    )
    self.assertLen(data, 1)
    self.assertGreater(data[0]["c"], 0)

  def test_query_events_db_raises_error_on_invalid_sql(self) -> None:
    with self.assertRaises(Exception):
      events_db_tool.query_events_db(
          session_id=str(self._single_trace),
          query="SELECT * FROM TableThatDoesNotExist",
      )

  def test_query_events_db_preserves_order_by(self) -> None:
    raw_res = events_db_tool.query_events_db(
        session_id=str(self._single_trace),
        query="SELECT start_ns FROM Events ORDER BY start_ns DESC",
    )
    res = json.loads(raw_res)
    data = json.loads(
        pathlib.Path(res[events_db_tool._KEY_PATH]).read_text(encoding="utf-8")
    )

    self.assertNotEmpty(data)
    keys = [row["start_ns"] for row in data]
    self.assertEqual(keys, sorted(keys, reverse=True))

  def test_query_events_db_serializes_sql_data_types(self) -> None:
    raw_res = events_db_tool.query_events_db(
        session_id=str(self._single_trace),
        query=(
            "SELECT 123.45::DECIMAL(10,2) AS dec_val, "
            "DATE '2026-09-16' AS date_val, "
            "INTERVAL 10 SECOND AS duration_val, "
            "'sample_bytes'::BLOB AS blob_val"
        ),
    )
    res = json.loads(raw_res)
    data = json.loads(
        pathlib.Path(res[events_db_tool._KEY_PATH]).read_text(encoding="utf-8")
    )
    self.assertLen(data, 1)
    row = data[0]
    self.assertAlmostEqual(row["dec_val"], 123.45)
    self.assertEqual(row["date_val"], "2026-09-16")
    self.assertEqual(row["duration_val"], "00:00:10")
    self.assertEqual(row["blob_val"], "sample_bytes")

  @parameterized.named_parameters(
      dict(
          testcase_name="read_unauthorized_file_via_read_text",
          query_template="SELECT * FROM read_text('{secret}')",
          error_regex="Permission Error",
      ),
      dict(
          testcase_name="read_unauthorized_file_via_read_csv",
          query_template="SELECT * FROM read_csv('{secret}')",
          error_regex="Permission Error",
      ),
      dict(
          testcase_name="copy_breakout_to_unauthorized_file",
          query_template="SELECT 1) TO '{exfil}' (FORMAT JSON, ARRAY true); --",
          error_regex="Invalid SQL query",
      ),
      dict(
          testcase_name="copy_breakout_to_cached_parquet_db",
          query_template=(
              "SELECT 1) TO '{db_path}' (FORMAT JSON, ARRAY true); --"
          ),
          error_regex="Invalid SQL query",
      ),
      dict(
          testcase_name="multi_statement_override_allowed_paths",
          query_template="SELECT 1; SET allowed_paths = ['{secret}']; --",
          error_regex="Expected a single SQL statement",
      ),
      dict(
          testcase_name="multi_statement_reset_allowed_paths",
          query_template="SELECT 1; RESET allowed_paths; --",
          error_regex="Expected a single SQL statement",
      ),
      dict(
          testcase_name="multi_statement_disable_lock_configuration",
          query_template="SELECT 1; SET lock_configuration = false; --",
          error_regex="Expected a single SQL statement",
      ),
      dict(
          testcase_name="standalone_set_statement",
          query_template="SET lock_configuration = false",
          error_regex="Only SELECT queries are allowed",
      ),
      dict(
          testcase_name="standalone_copy_statement",
          query_template="COPY Events TO '{exfil}'",
          error_regex="Only SELECT queries are allowed",
      ),
  )
  def test_query_events_db_blocks_security_violations(
      self, query_template: str, error_regex: str
  ) -> None:
    query = query_template.format(
        secret=self._secret_file,
        exfil=self._exfil_file,
        db_path=self._single_db_path,
    )
    with self.assertRaisesRegex(Exception, error_regex):
      events_db_tool.query_events_db(
          session_id=str(self._single_trace),
          query=query,
      )
    self.assertFalse(self._exfil_file.exists())
    self.assertEqual(self._single_db_path.read_bytes(), _parquet_bytes())


if __name__ == "__main__":
  absltest.main()
