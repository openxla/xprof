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
"""Tests for Events DB C API Python bindings."""

import os

from absl.testing import absltest
from absl.testing import parameterized

from python.runfiles import runfiles

from xprof.convert.events_db.python import pywrap_events_db_c_api as events_db

_TEST_DATA_PATH = (
    "org_xprof/xprof/convert/events_db/python/test_data/test.xplane.pb"
)


class ParquetExportOptionsTest(parameterized.TestCase):

  def test_default_options(self):
    opts = events_db.ParquetExportOptions()
    self.assertIsNone(opts.batch_size)
    self.assertIsNone(opts.max_record_count)
    self.assertIsNone(opts.compression_type)
    self.assertIsNone(opts.compression_level)

  def test_explicit_none_batch_size(self):
    opts = events_db.ParquetExportOptions(batch_size=None)
    self.assertIsNone(opts.batch_size)

  def test_custom_options(self):
    opts = events_db.ParquetExportOptions(
        max_record_count=1000,
        batch_size=1024,
        compression_type=events_db.ArrowCompressionType.SNAPPY,
        compression_level=3,
    )
    self.assertEqual(opts.max_record_count, 1000)
    self.assertEqual(opts.batch_size, 1024)
    self.assertEqual(
        opts.compression_type,
        events_db.ArrowCompressionType.SNAPPY,
    )
    self.assertEqual(opts.compression_level, 3)

  @parameterized.parameters(0, -1)
  def test_rejects_non_positive_batch_size(self, batch_size: int):
    with self.assertRaisesRegex(
        ValueError, f"batch_size must be positive, got {batch_size}"
    ):
      events_db.ParquetExportOptions(batch_size=batch_size)

  @parameterized.parameters(0, -1)
  def test_rejects_non_positive_max_record_count(self, max_record_count: int):
    with self.assertRaisesRegex(
        ValueError, f"max_record_count must be positive, got {max_record_count}"
    ):
      events_db.ParquetExportOptions(max_record_count=max_record_count)

  def test_rejects_compression_level_without_type(self):
    with self.assertRaisesRegex(
        ValueError, "compression_level requires compression_type to be set."
    ):
      events_db.ParquetExportOptions(compression_level=3)


class ArrowCompressionTypeTest(parameterized.TestCase):

  @parameterized.parameters(
      ("SNAPPY", "SNAPPY"),
      ("UNCOMPRESSED", "UNCOMPRESSED"),
      ("ZSTD", "ZSTD"),
  )
  def test_enum_values(self, member_name: str, expected_val: str):
    self.assertEqual(
        events_db.ArrowCompressionType[member_name].value,
        expected_val,
    )


class XSpaceToParquetTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls) -> None:
    super().setUpClass()
    file_path = runfiles.Create().Rlocation(_TEST_DATA_PATH)
    cls._xspace_path = file_path

  def test_xspace_to_parquet_default_options(self):
    output_path = self.create_tempfile("test_default.parquet").full_path
    events_db.xspace_to_parquet(
        input_path=self._xspace_path,
        output_path=output_path,
    )
    self.assertTrue(os.path.exists(output_path))
    self.assertGreater(os.path.getsize(output_path), 0)
    with open(output_path, "rb") as f:
      content = f.read()
      self.assertTrue(content.startswith(b"PAR1"))
      self.assertTrue(content.endswith(b"PAR1"))

  def test_xspace_to_parquet_explicit_none_options(self):
    output_path = self.create_tempfile("test_none_options.parquet").full_path
    events_db.xspace_to_parquet(
        input_path=self._xspace_path,
        output_path=output_path,
        options=None,
    )
    self.assertTrue(os.path.exists(output_path))
    self.assertGreater(os.path.getsize(output_path), 0)
    with open(output_path, "rb") as f:
      content = f.read()
      self.assertTrue(content.startswith(b"PAR1"))
      self.assertTrue(content.endswith(b"PAR1"))

  def test_xspace_to_parquet_explicit_none_batch_size(self):
    output_path = self.create_tempfile("test_none_batch.parquet").full_path
    events_db.xspace_to_parquet(
        input_path=self._xspace_path,
        output_path=output_path,
        options=events_db.ParquetExportOptions(batch_size=None),
    )
    self.assertTrue(os.path.exists(output_path))
    self.assertGreater(os.path.getsize(output_path), 0)
    with open(output_path, "rb") as f:
      content = f.read()
      self.assertTrue(content.startswith(b"PAR1"))
      self.assertTrue(content.endswith(b"PAR1"))

  @parameterized.named_parameters(
      (
          "uncompressed",
          events_db.ArrowCompressionType.UNCOMPRESSED,
          False,
      ),
      (
          "snappy",
          events_db.ArrowCompressionType.SNAPPY,
          True,
      ),
      (
          "zstd",
          events_db.ArrowCompressionType.ZSTD,
          True,
      ),
  )
  def test_xspace_to_parquet_compression(
      self,
      compression_type: events_db.ArrowCompressionType,
      expect_diff_from_uncompressed: bool,
  ):
    default_output_path = self.create_tempfile("test_default.parquet").full_path
    events_db.xspace_to_parquet(
        input_path=self._xspace_path,
        output_path=default_output_path,
    )
    with open(default_output_path, "rb") as f:
      default_content = f.read()

    output_path = self.create_tempfile(
        f"test_{compression_type.value.lower()}.parquet"
    ).full_path
    options = events_db.ParquetExportOptions(
        compression_type=compression_type,
    )
    events_db.xspace_to_parquet(
        input_path=self._xspace_path,
        output_path=output_path,
        options=options,
    )
    self.assertTrue(os.path.exists(output_path))
    self.assertGreater(os.path.getsize(output_path), 0)
    with open(output_path, "rb") as f:
      content = f.read()
      self.assertTrue(content.startswith(b"PAR1"))
      self.assertTrue(content.endswith(b"PAR1"))

    if expect_diff_from_uncompressed:
      self.assertNotEqual(content, default_content)

  def test_xspace_to_parquet_custom_options(self):
    output_path = self.create_tempfile("test_custom.parquet").full_path
    options = events_db.ParquetExportOptions(
        max_record_count=1,
        batch_size=1024,
        compression_type=events_db.ArrowCompressionType.ZSTD,
        compression_level=3,
    )
    events_db.xspace_to_parquet(
        input_path=self._xspace_path,
        output_path=output_path,
        options=options,
    )
    self.assertTrue(os.path.exists(output_path))
    self.assertGreater(os.path.getsize(output_path), 0)
    with open(output_path, "rb") as f:
      content = f.read()
      self.assertTrue(content.startswith(b"PAR1"))
      self.assertTrue(content.endswith(b"PAR1"))

  def test_xspace_to_parquet_rejects_empty_input_path(self):
    with self.assertRaisesRegex(
        ValueError, "input_path must not be null or empty."
    ):
      events_db.xspace_to_parquet(
          input_path="",
          output_path="/tmp/output.parquet",
      )

  def test_xspace_to_parquet_rejects_empty_output_path(self):
    with self.assertRaisesRegex(
        ValueError, "output_path must not be null or empty."
    ):
      events_db.xspace_to_parquet(
          input_path=self._xspace_path,
          output_path="",
      )

  def test_xspace_to_parquet_c_api_failure_raises_runtime_error(self):
    output_path = self.create_tempfile("test_error.parquet").full_path
    with self.assertRaisesRegex(RuntimeError, "file.xplane.pb"):
      events_db.xspace_to_parquet(
          input_path="/nonexistent/invalid/file.xplane.pb",
          output_path=output_path,
      )


if __name__ == "__main__":
  absltest.main()
