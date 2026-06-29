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

"""Tests for the counter_extractor module."""

from unittest import mock

from absl.testing import absltest

from xprof.convert import counter_extractor


class CounterExtractorTest(absltest.TestCase):

  def test_get_all_counters_success(self):
    mock_content = """
    #ifndef THIRD_PARTY_XPROF_UTILS_TPU_COUNTER_IDS_H_
    #define THIRD_PARTY_XPROF_UTILS_TPU_COUNTER_IDS_H_
    namespace xprof {
    enum TpuCounterIdsTpu7x : uint64_t {
      TPU_COUNTER_ID_FOO = 12345,
      // Some comment
      TPU_COUNTER_ID_BAR = 67890,
    };
    }
    #endif
    """
    with mock.patch("os.path.exists", return_value=True), mock.patch(
        "builtins.open", mock.mock_open(read_data=mock_content)
    ):
      counters = counter_extractor.get_all_counters("v7x")
      self.assertEqual(
          counters,
          [
              {"name": "tpu_counter_id_foo", "val": 12345},
              {"name": "tpu_counter_id_bar", "val": 67890},
          ],
      )

  def test_get_all_counters_missing_file(self):
    with mock.patch("os.path.exists", return_value=False):
      with self.assertRaises(FileNotFoundError):
        counter_extractor.get_all_counters("invalid_device")

  def test_get_all_counters_no_match(self):
    mock_content = "some file without enum"
    with mock.patch("os.path.exists", return_value=True), mock.patch(
        "builtins.open", mock.mock_open(read_data=mock_content)
    ):
      counters = counter_extractor.get_all_counters("v7x")
      self.assertEqual(counters, [])

  def test_get_all_counters_real_file(self):
    # This checks that we can run it on the actual v6e or v7x header files.
    # The header files are mapped in data dependency of the library.
    counters = counter_extractor.get_all_counters("v6e")
    self.assertNotEmpty(counters)
    self.assertEqual(
        counters[0]["name"],
        "vf_chip_tc_tcs_tc_misc_tcs_stats_tcs_stats_counters_unprivileged_count_cycles",
    )
    self.assertEqual(counters[0]["val"], 3257466888)


if __name__ == "__main__":
  absltest.main()
