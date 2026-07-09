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

  def setUp(self):
    super().setUp()
    counter_extractor.get_all_counters.cache_clear()

  @mock.patch("importlib.resources.files")
  def test_get_all_counters_success(self, mock_files):
    mock_content = """
    #ifndef THIRD_PARTY_XPROF_UTILS_TPU_COUNTER_IDS_H_
    #define THIRD_PARTY_XPROF_UTILS_TPU_COUNTER_IDS_H_
    namespace xprof {
    enum TpuCounterIdsTpu7x : uint64_t {
      // NOLINTBEGIN
      TPU_COUNTER_ID_FOO = 12345,
      // Some comment
      TPU_COUNTER_ID_BAR =
          67890,
      TPU_COUNTER_ID_HEX = 0x1234,
      TPU_COUNTER_ID_MULTI_EQ = 1 = 2,
      // NOLINTEND
    };
    }
    #endif
    """

    mock_file = mock.MagicMock()
    mock_file.read_text.return_value = mock_content
    # Chain: files() -> / 'utils' -> / filename
    mock_pkg = mock.MagicMock()
    mock_utils = mock.MagicMock()
    mock_pkg.__truediv__.return_value = mock_utils
    mock_utils.__truediv__.return_value = mock_file
    mock_files.return_value = mock_pkg

    counters = counter_extractor.get_all_counters("v7x")
    self.assertEqual(
        counters,
        [
            {"name": "tpu_counter_id_foo", "val": 12345},
            {"name": "tpu_counter_id_bar", "val": 67890},
            {"name": "tpu_counter_id_hex", "val": 4660},
        ],
    )

  @mock.patch("importlib.resources.files")
  def test_get_all_counters_fallback_success(self, mock_files):
    mock_content = """
    #ifndef THIRD_PARTY_XPROF_UTILS_TPU_COUNTER_IDS_H_
    #define THIRD_PARTY_XPROF_UTILS_TPU_COUNTER_IDS_H_
    namespace xprof {
    enum TpuCounterIdsTpu7x : uint64_t {
      TPU_COUNTER_ID_FOO = 12345,
    };
    }
    #endif
    """

    mock_file = mock.MagicMock()
    mock_file.read_text.return_value = mock_content
    mock_pkg = mock.MagicMock()
    mock_utils = mock.MagicMock()
    mock_pkg.__truediv__.return_value = mock_utils
    mock_utils.__truediv__.return_value = mock_file

    def side_effect(package_name):
      if package_name == "google3.third_party.xprof":
        raise ImportError("No module named google3")
      elif package_name == "xprof":
        return mock_pkg
      raise ValueError(f"Unexpected package: {package_name}")

    mock_files.side_effect = side_effect

    counters = counter_extractor.get_all_counters("v7x")
    self.assertEqual(
        counters,
        [
            {"name": "tpu_counter_id_foo", "val": 12345},
        ],
    )

  @mock.patch("importlib.resources.files")
  def test_get_all_counters_missing_file(self, mock_files):
    mock_utils = mock.MagicMock()
    # Simulate FileNotFound when reading text
    mock_file = mock.MagicMock()
    mock_file.read_text.side_effect = FileNotFoundError("File not found")
    mock_pkg = mock.MagicMock()
    mock_pkg.__truediv__.return_value = mock_utils
    mock_utils.__truediv__.return_value = mock_file
    mock_files.return_value = mock_pkg

    with self.assertRaises(FileNotFoundError):
      counter_extractor.get_all_counters("v7x")

  @mock.patch("importlib.resources.files")
  def test_get_all_counters_no_match(self, mock_files):
    mock_content = "some file without enum"
    mock_file = mock.MagicMock()
    mock_file.read_text.return_value = mock_content
    mock_pkg = mock.MagicMock()
    mock_utils = mock.MagicMock()
    mock_pkg.__truediv__.return_value = mock_utils
    mock_utils.__truediv__.return_value = mock_file
    mock_files.return_value = mock_pkg

    counters = counter_extractor.get_all_counters("v7x")
    self.assertEqual(counters, [])

  @mock.patch("importlib.resources.files")
  def test_get_all_counters_empty_enum(self, mock_files):
    mock_content_empty = "enum TpuCounterIdsTpu7x : uint64_t {};"
    mock_content_whitespace = "enum TpuCounterIdsTpu7x : uint64_t { \n };"

    mock_file = mock.MagicMock()
    mock_pkg = mock.MagicMock()
    mock_utils = mock.MagicMock()
    mock_pkg.__truediv__.return_value = mock_utils
    mock_utils.__truediv__.return_value = mock_file
    mock_files.return_value = mock_pkg

    # Test completely empty
    mock_file.read_text.return_value = mock_content_empty
    counters = counter_extractor.get_all_counters("v7x")
    self.assertEqual(counters, [])

    # Test whitespace only
    mock_file.read_text.return_value = mock_content_whitespace
    counters = counter_extractor.get_all_counters("v7x")
    self.assertEqual(counters, [])

  def test_get_all_counters_invalid_device_type(self):
    with self.assertRaises(ValueError):
      counter_extractor.get_all_counters("../invalid")

  def test_get_all_counters_security_path_separators(self):
    with self.assertRaises(ValueError):
      counter_extractor.get_all_counters("v7x/../etc/passwd")


if __name__ == "__main__":
  absltest.main()
