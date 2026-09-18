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
# pylint: disable=g-direct-tensorflow-import
from tensorflow.tsl.profiler.protobuf import xplane_pb2

# pylint: enable=g-direct-tensorflow-import
from google3.third_party.xprof.embedded.llo_analysis import llo_lite_pb2
from xprof.convert import _pywrap_profiler_plugin as profiler_wrapper_plugin


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


if __name__ == "__main__":
  absltest.main()
