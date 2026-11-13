# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for profiler_wrapper.cc pybind methods."""

from concurrent import futures
import json
import os
import subprocess
import sys
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
# pylint: disable=g-direct-tensorflow-import
from tensorflow.tsl.profiler.protobuf import xplane_pb2

# pylint: enable=g-direct-tensorflow-import
from google3.third_party.xprof.embedded.llo_analysis import llo_lite_pb2
from xprof.convert import _pywrap_profiler_plugin as profiler_wrapper_plugin

_INIT_STUBS_SUBPROCESS_ENV = "XPROF_TEST_INIT_STUBS_SUBPROCESS"


def _create_multi_module_llo_xspace_bytes() -> bytes:
  space = xplane_pb2.XSpace()
  plane = space.planes.add(name="/host:metadata")
  plane.stat_metadata[1].name = "llo_proto"

  for ev_id, kernel_name, hlo_name, opcode, src_path, line in (
      (
          10,
          "jit_kernel_a",
          "custom-call.a",
          llo_lite_pb2.OPCODE_VECTOR_MATMUL,
          "kernel_a.py",
          10,
      ),
      (
          20,
          "jit_kernel_b",
          "custom-call.b",
          llo_lite_pb2.OPCODE_VECTOR_ADD_F32,
          "kernel_b.py",
          20,
      ),
  ):
    ev_meta = plane.event_metadata[ev_id]
    ev_meta.name = kernel_name
    module = llo_lite_pb2.LloModuleProto(hlo_instruction_name=hlo_name)
    inst = module.top_region.members.add().instruction
    inst.ordinal = 0
    inst.opcode = opcode
    module.source_map.strings.append(src_path)
    loc = module.source_map.locations.add()
    frame = loc.frames.add()
    frame.path = 0
    frame.line_start = line
    loc.ordinals.append(0)
    stat = ev_meta.stats.add()
    stat.metadata_id = 1
    stat.bytes_value = module.SerializeToString()

  return space.SerializeToString()


def _call_xspace_to_tools_data_invalid():
  return profiler_wrapper_plugin._lib.XSpaceToToolsData(
      None,
      0,
      b"trace_viewer",
      None,
      None,
      None,
      None,
      None,
      1,
      None,
      None,
      None,
  )


class ProfilerSessionTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("default_options", "trace_viewer", None),
      ("with_list_options", "trace_viewer@", {"hosts": ["host1", "host2"]}),
      ("utilization_viewer", "utilization_viewer", None),
      ("empty_tool_name", "", None),
  )
  def test_xspace_to_tools_data(self, tool_name, options):
    res, success = profiler_wrapper_plugin.xspace_to_tools_data(
        xspace_paths=[], tool_name=tool_name, options=options
    )
    self.assertEmpty(res)
    self.assertFalse(success)

  def test_xspace_to_tools_data_from_byte_string_empty_tool_name(self):
    res, success = (
        profiler_wrapper_plugin.xspace_to_tools_data_from_byte_string(
            xspace_strings=[], filenames_list=[], tool_name=""
        )
    )
    self.assertEmpty(res)
    self.assertFalse(success)

  def test_empty_and_none_string_marshaling_passes_empty_bytes(self):
    with mock.patch.object(
        profiler_wrapper_plugin, "_lib", autospec=True
    ) as mock_lib:
      mock_lib.Trace.return_value = None
      mock_lib.Monitor.return_value = None
      mock_lib.StartContinuousProfiling.return_value = None
      mock_lib.StopContinuousProfiling.return_value = None
      mock_lib.GetSnapshot.return_value = None
      mock_lib.XSpaceToToolsData.return_value = None
      mock_lib.XSpaceToToolsDataFromByteString.return_value = None

      profiler_wrapper_plugin.trace(
          "localhost:8466", "/tmp/logdir", "", False, 100, 1
      )
      self.assertEqual(
          mock_lib.Trace.call_args.args[:3],
          (b"localhost:8466", b"/tmp/logdir", b""),
      )

      profiler_wrapper_plugin.trace(
          "localhost:8466", "/tmp/logdir", None, False, 100, 1
      )
      self.assertEqual(
          mock_lib.Trace.call_args.args[:3],
          (b"localhost:8466", b"/tmp/logdir", b""),
      )

      profiler_wrapper_plugin.trace("", "", "", False, 100, 1)
      self.assertEqual(mock_lib.Trace.call_args.args[:3], (b"", b"", b""))

      self.assertEqual(profiler_wrapper_plugin.monitor("", 100, 1, False), "")
      self.assertEqual(mock_lib.Monitor.call_args.args[0], b"")

      profiler_wrapper_plugin.start_continuous_profiling("")
      self.assertEqual(mock_lib.StartContinuousProfiling.call_args.args[0], b"")

      profiler_wrapper_plugin.stop_continuous_profiling("")
      self.assertEqual(mock_lib.StopContinuousProfiling.call_args.args[0], b"")

      profiler_wrapper_plugin.get_snapshot("", "")
      self.assertEqual(mock_lib.GetSnapshot.call_args.args, (b"", b""))

      profiler_wrapper_plugin.xspace_to_tools_data([], "")
      self.assertEqual(mock_lib.XSpaceToToolsData.call_args.args[2], b"")

      profiler_wrapper_plugin.xspace_to_tools_data_from_byte_string([], [], "")
      self.assertEqual(
          mock_lib.XSpaceToToolsDataFromByteString.call_args.args[4], b""
      )

      profiler_wrapper_plugin.initialize_stubs("")
      mock_lib.InitializeStubs.assert_called_once_with(b"")

  def test_c_api_null_pointer_guards_return_invalid_argument(self):
    lib = profiler_wrapper_plugin._lib
    error_ptrs = [
        lib.Trace(
            None, b"/tmp", b"", False, 1, 1, None, None, None, None, None, 0
        ),
        lib.Trace(
            b"localhost:8466",
            None,
            b"",
            False,
            1,
            1,
            None,
            None,
            None,
            None,
            None,
            0,
        ),
        lib.Trace(
            b"", b"/tmp", None, False, 1, 1, None, None, None, None, None, 0
        ),
        lib.Monitor(None, 1, 1, False, None),
        lib.StartContinuousProfiling(None, None, None, None, None, None, 0),
        lib.StopContinuousProfiling(None),
        lib.GetSnapshot(None, b"/tmp"),
        lib.GetSnapshot(b"localhost:8466", None),
        lib.XSpaceToToolsData(
            None, 0, None, None, None, None, None, None, 0, None, None, None
        ),
        lib.XSpaceToToolsDataFromByteString(
            None,
            None,
            None,
            0,
            None,
            None,
            None,
            None,
            None,
            None,
            0,
            None,
            None,
            None,
        ),
    ]
    for err_ptr in error_ptrs:
      self.assertIsNotNone(err_ptr)
      with self.assertRaisesRegex(RuntimeError, r"^INVALID_ARGUMENT"):
        profiler_wrapper_plugin._check_error(err_ptr)

    if profiler_wrapper_plugin.built_with_embedded():
      self.assertIsNone(lib.CreateLloAnalysis(None))

  def test_initialize_stubs_empty_and_null_in_subprocess(self):
    env = dict(os.environ)
    env[_INIT_STUBS_SUBPROCESS_ENV] = "1"
    proc = subprocess.run(
        [sys.argv[0]],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    self.assertEqual(proc.returncode, 0, msg=proc.stderr)

  def test_xspace_to_tools_data_invalid_options_c_api(self):
    err = _call_xspace_to_tools_data_invalid()
    self.assertIsNotNone(err)
    self.addCleanup(profiler_wrapper_plugin._lib.FreeString, err)

  def test_check_error_raises_runtime_error(self):
    err = _call_xspace_to_tools_data_invalid()
    with self.assertRaisesRegex(RuntimeError, r"^INVALID_ARGUMENT"):
      profiler_wrapper_plugin._check_error(err)

  def test_analyze_llo(self):
    if not profiler_wrapper_plugin.built_with_embedded():
      self.skipTest("analyze_llo is not supported in this build")

    test_file = self.create_tempfile().full_path
    result = profiler_wrapper_plugin.analyze_llo(test_file)
    self.assertFalse(result["success"])

  def test_analyze_llo_source_map_by_kernel_and_filter_isolation(self):
    if not profiler_wrapper_plugin.built_with_embedded():
      self.skipTest("analyze_llo is not supported in this build")

    test_file = self.create_tempfile(
        content=_create_multi_module_llo_xspace_bytes()
    ).full_path

    # Unfiltered multi-module profile: source_map_by_kernel preserves all
    # colliding ordinals without loss; ambiguous flat source_map is omitted.
    all_res = profiler_wrapper_plugin.analyze_llo(test_file)
    self.assertTrue(all_res["success"])
    self.assertEqual(
        all_res["source_map_by_kernel"],
        {
            "jit_kernel_a": {"0": "kernel_a.py:10"},
            "jit_kernel_b": {"0": "kernel_b.py:20"},
        },
    )
    self.assertNotIn("source_map", all_res)

    # Filtered to a single kernel: both source_map_by_kernel and unambiguous
    # flat source_map are populated for that kernel only.
    filtered_a = profiler_wrapper_plugin.analyze_llo(
        test_file, kernel="kernel_a"
    )
    self.assertTrue(filtered_a["success"])
    self.assertEqual(filtered_a["source_map"], {"0": "kernel_a.py:10"})
    self.assertEqual(
        filtered_a["source_map_by_kernel"],
        {"jit_kernel_a": {"0": "kernel_a.py:10"}},
    )

    # Non-matching kernel filter: returns {"success": False} with no leaked
    # source data from other modules.
    no_match = profiler_wrapper_plugin.analyze_llo(
        test_file, kernel="no_such_kernel"
    )
    self.assertEqual(no_match, {"success": False})
    self.assertNotIn("source_map", no_match)
    self.assertNotIn("source_map_by_kernel", no_match)

  def test_llo_static_analysis_unavailable_envelope(self):
    res = json.loads(
        profiler_wrapper_plugin._llo_static_analysis_unavailable("region_tree")
    )
    self.assertEqual(res["status"], "UNAVAILABLE")
    self.assertEqual(res["mode"], "region_tree")
    self.assertEqual(res["semantics"], "static_modelled_schedule")
    self.assertEqual(res["reason"], "LLO_STATIC_ANALYSIS_UNSUPPORTED_IN_BUILD")
    self.assertNotIn("data", res)

  def test_get_llo_static_analysis_json_unavailable_without_embedded(self):
    if profiler_wrapper_plugin.built_with_embedded():
      self.skipTest("LLO static analysis is supported in this build")

    res = json.loads(
        profiler_wrapper_plugin.get_llo_static_analysis_json(
            self.create_tempfile().full_path, mode="bdi_stalls"
        )
    )
    self.assertEqual(res["status"], "UNAVAILABLE")
    self.assertEqual(res["mode"], "bdi_stalls")

  def test_utilization_viewer_conversion(self):
    """Tests that utilization_viewer is supported via fallback."""
    xspace = xplane_pb2.XSpace()
    res, success = (
        profiler_wrapper_plugin.xspace_to_tools_data_from_byte_string(
            xspace_strings=[xspace.SerializeToString()],
            filenames_list=["test_host.xplane.pb"],
            tool_name="utilization_viewer",
        )
    )
    self.assertTrue(success)
    self.assertIn(b"rows", res)

  def test_concurrent_uncached_xspace_to_tools_data_does_not_race(self):
    """Verifies concurrent use_saved_result=False calls serialize cache resets."""
    session_dir = self.create_tempdir().full_path
    xspace = xplane_pb2.XSpace()
    xspace.planes.add(name="/host:CPU")
    payload = xspace.SerializeToString()
    paths = []
    for host in ("host0", "host1"):
      path = os.path.join(session_dir, f"{host}.xplane.pb")
      with open(path, "wb") as f:
        f.write(payload)
      paths.append(path)

    def _convert(idx: int) -> tuple[bytes, bool]:
      tool = "overview_page" if idx % 2 == 0 else "trace_viewer@"
      return profiler_wrapper_plugin.xspace_to_tools_data(
          xspace_paths=paths,
          tool_name=tool,
          options={"use_saved_result": False},
      )

    with futures.ThreadPoolExecutor(max_workers=6) as pool:
      results = list(pool.map(_convert, range(12)))

    for data, success in results:
      self.assertTrue(success)
      self.assertNotEmpty(data)

    entries = os.listdir(session_dir)
    self.assertEqual([f for f in entries if ".tmp." in f], [])
    self.assertNotIn(".cache_cleared", entries)
    self.assertTrue(any(f.endswith(".op_stats_v2.pb") for f in entries))
    self.assertTrue(any(f.endswith(".SSTABLE") for f in entries))

    # Once the concurrent wave completes, a subsequent use_saved_result=False
    # request must still clear stale cache files instead of being suppressed.
    stale_cache = os.path.join(session_dir, "stale.smart_suggestion.pb")
    with open(stale_cache, "wb") as f:
      f.write(b"stale")
    refreshed, refreshed_ok = profiler_wrapper_plugin.xspace_to_tools_data(
        xspace_paths=paths,
        tool_name="overview_page",
        options={"use_saved_result": False},
    )
    self.assertTrue(refreshed_ok)
    self.assertNotEmpty(refreshed)
    self.assertFalse(os.path.exists(stale_cache))

  def test_uncached_xspace_to_tools_data_on_readonly_dir(self):
    """Verifies use_saved_result=False succeeds on read-only directories."""
    session_dir = self.create_tempdir().full_path
    xspace = xplane_pb2.XSpace()
    xspace.planes.add(name="/host:CPU")
    path = os.path.join(session_dir, "host0.xplane.pb")
    with open(path, "wb") as f:
      f.write(xspace.SerializeToString())

    os.chmod(session_dir, 0o555)
    self.addCleanup(os.chmod, session_dir, 0o755)

    data, success = profiler_wrapper_plugin.xspace_to_tools_data(
        xspace_paths=[path],
        tool_name="overview_page",
        options={"use_saved_result": False},
    )
    self.assertTrue(success)
    self.assertNotEmpty(data)


if __name__ == "__main__":
  if os.environ.get(_INIT_STUBS_SUBPROCESS_ENV) == "1":
    profiler_wrapper_plugin.initialize_stubs("")
    profiler_wrapper_plugin._lib.InitializeStubs(None)
    sys.exit(0)
  absltest.main()
