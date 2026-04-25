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

  def test_get_module_digest(self):
    if hasattr(profiler_wrapper_plugin, "get_module_digest"):
      test_file = self.create_tempfile().full_path
      result = profiler_wrapper_plugin.get_module_digest(test_file)
      self.assertFalse(result["success"])


if __name__ == "__main__":
  absltest.main()
