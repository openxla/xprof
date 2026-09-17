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
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized

from python.runfiles import runfiles

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


if __name__ == "__main__":
  absltest.main()
