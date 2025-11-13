# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for the raw_to_tool_data module."""

import tensorflow as tf

from xprof.convert import raw_to_tool_data


class RawToToolDataTest(tf.test.TestCase):

  def test_using_old_tool_format_maps_to_new_format(self):
    data, content_type = raw_to_tool_data.xspace_to_tool_data(
        xspace_paths=["/path/to/xspace"],
        tool="trace_viewer@^",
        params={},
        xspace_wrapper_func=lambda paths, tool, options: (tool.encode(), True),
    )

    self.assertEqual(data, b"trace_viewer@")
    self.assertEqual(content_type, "application/json")

  def test_using_new_tool_format_does_not_map_to_old_format(self):
    data, content_type = raw_to_tool_data.xspace_to_tool_data(
        xspace_paths=["/path/to/xspace"],
        tool="trace_viewer@",
        params={},
        xspace_wrapper_func=lambda paths, tool, options: (tool.encode(), True),
    )

    self.assertEqual(data, b"trace_viewer@")
    self.assertEqual(content_type, "application/json")


if __name__ == "__main__":
  tf.test.main()
