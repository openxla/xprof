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

from absl.testing import absltest
from absl.testing import parameterized
from xprof.convert import _pywrap_profiler_plugin as profiler_wrapper_plugin


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
  )
  def test_xspace_to_tools_data(self, tool_name, options):
    res, success = profiler_wrapper_plugin.xspace_to_tools_data(
        xspace_paths=[], tool_name=tool_name, options=options
    )
    self.assertEmpty(res)
    self.assertFalse(success)

  def test_xspace_to_tools_data_invalid_options_c_api(self):
    err = _call_xspace_to_tools_data_invalid()
    self.assertIsNotNone(err)
    self.addCleanup(profiler_wrapper_plugin._lib.FreeString, err)

  def test_check_error_raises_runtime_error(self):
    err = _call_xspace_to_tools_data_invalid()
    with self.assertRaisesRegex(RuntimeError, r"^INVALID_ARGUMENT"):
      profiler_wrapper_plugin._check_error(err)

  # A None/empty tool_name reaches the C API as a NULL c_char_p; the guard now
  # rejects it with a clean RuntimeError instead of dereferencing it downstream.
  @parameterized.named_parameters(
      ("none_tool", None),
      ("empty_tool", ""),
  )
  def test_xspace_to_tools_data_null_tool_name_raises(self, tool_name):
    with self.assertRaisesRegex(RuntimeError, "tool_name must be a non-empty"):
      profiler_wrapper_plugin.xspace_to_tools_data(
          xspace_paths=[], tool_name=tool_name, options=None
      )

  @parameterized.named_parameters(
      ("none_tool", None),
      ("empty_tool", ""),
  )
  def test_xspace_to_tools_data_from_byte_string_null_tool_name_raises(
      self, tool_name
  ):
    with self.assertRaisesRegex(RuntimeError, "tool_name must be a non-empty"):
      profiler_wrapper_plugin.xspace_to_tools_data_from_byte_string(
          xspace_strings=[],
          filenames_list=[],
          tool_name=tool_name,
          options=None,
      )

  # The profiling-control wrappers marshal None/'' to a NULL c_char_p. The C
  # boundary guards reject those with a clean RuntimeError; since the wrapper
  # already collapsed None/'' to NULL, rejecting NULL breaks no working caller.
  @parameterized.named_parameters(("none_addr", None), ("empty_addr", ""))
  def test_trace_null_service_addr_raises(self, service_addr):
    with self.assertRaisesRegex(
        RuntimeError, "service_addr must be a non-empty"
    ):
      profiler_wrapper_plugin.trace(
          service_addr=service_addr,
          logdir="/tmp",
          worker_list="",
          include_dataset_ops=False,
          duration_ms=1,
          num_tracing_attempts=1,
      )

  def test_trace_null_logdir_raises(self):
    with self.assertRaisesRegex(RuntimeError, "logdir must be non-NULL"):
      profiler_wrapper_plugin.trace(
          service_addr="localhost:1234",
          logdir=None,
          worker_list="",
          include_dataset_ops=False,
          duration_ms=1,
          num_tracing_attempts=1,
      )

  @parameterized.named_parameters(("none_addr", None), ("empty_addr", ""))
  def test_monitor_null_service_addr_raises(self, service_addr):
    with self.assertRaisesRegex(
        RuntimeError, "service_addr must be a non-empty"
    ):
      profiler_wrapper_plugin.monitor(
          service_addr=service_addr,
          duration_ms=1,
          monitoring_level=1,
          display_timestamp=False,
      )

  @parameterized.named_parameters(("none_addr", None), ("empty_addr", ""))
  def test_start_continuous_profiling_null_service_addr_raises(
      self, service_addr
  ):
    with self.assertRaisesRegex(
        RuntimeError, "service_addr must be a non-empty"
    ):
      profiler_wrapper_plugin.start_continuous_profiling(
          service_addr=service_addr
      )

  @parameterized.named_parameters(("none_addr", None), ("empty_addr", ""))
  def test_stop_continuous_profiling_null_service_addr_raises(
      self, service_addr
  ):
    with self.assertRaisesRegex(
        RuntimeError, "service_addr must be a non-empty"
    ):
      profiler_wrapper_plugin.stop_continuous_profiling(
          service_addr=service_addr
      )

  @parameterized.named_parameters(("none_addr", None), ("empty_addr", ""))
  def test_get_snapshot_null_service_addr_raises(self, service_addr):
    with self.assertRaisesRegex(
        RuntimeError, "service_addr must be a non-empty"
    ):
      profiler_wrapper_plugin.get_snapshot(
          service_addr=service_addr, logdir="/tmp"
      )

  def test_get_snapshot_null_logdir_raises(self):
    with self.assertRaisesRegex(RuntimeError, "logdir must be non-NULL"):
      profiler_wrapper_plugin.get_snapshot(
          service_addr="localhost:1234", logdir=None
      )

  def test_initialize_stubs_none_is_noop(self):
    # Void-returning guard: None must not raise or crash (fail-closed no-op).
    self.assertIsNone(
        profiler_wrapper_plugin.initialize_stubs(worker_service_addresses=None)
    )

  def test_analyze_llo(self):
    if not profiler_wrapper_plugin.built_with_embedded():
      self.skipTest("analyze_llo is not supported in this build")

    test_file = self.create_tempfile().full_path
    result = profiler_wrapper_plugin.analyze_llo(test_file)
    self.assertFalse(result["success"])


if __name__ == "__main__":
  absltest.main()
