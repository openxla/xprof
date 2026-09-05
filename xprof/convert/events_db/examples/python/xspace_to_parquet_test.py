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
"""Tests for xspace_to_parquet example binary."""

import contextlib
import io
import os

from absl import app
from absl import flags
from absl.testing import absltest
from absl.testing import flagsaver
from etils import epath

from xprof.convert.events_db.examples.python import xspace_to_parquet
from xprof.convert.events_db.python import events_db


class XspaceToParquetTest(absltest.TestCase):

  def setUp(self) -> None:
    super().setUp()
    self._input_path = self.create_tempfile("empty.xplane.pb").full_path
    self._output_path = self.create_tempfile("out.parquet").full_path

  def test_main_runs_with_flags(self) -> None:
    f = io.StringIO()
    with (
        flagsaver.flagsaver(
            input_path=self._input_path,
            output_path=self._output_path,
            batch_size=512,
            compression_type=events_db.ArrowCompressionType.SNAPPY,
            max_record_count=100,
        ),
        contextlib.redirect_stdout(f),
    ):
      xspace_to_parquet.main(["xspace_to_parquet"])

    with self.subTest("output_created"):
      self.assertTrue(os.path.exists(self._output_path))
      self.assertGreater(os.path.getsize(self._output_path), 0)
    with self.subTest("output_is_parquet"), open(
        self._output_path, "rb"
    ) as parquet_file:
      content = parquet_file.read()
      # Binary content. Do not `assertStartsWith` or `assertEndsWith`.
      self.assertTrue(content.startswith(b"PAR1"))
      self.assertTrue(content.endswith(b"PAR1"))
    with self.subTest("stdout_contains_expected_output"):
      output = f.getvalue()
      self.assertIn("Converting XSpace trace", output)
      self.assertIn("Successfully finished parsing in", output)
      self.assertIn("with status: ParseStatus.COMPLETE", output)

  def test_invalid_batch_size_raises(self) -> None:
    with self.assertRaises(flags.IllegalFlagValueError):
      with flagsaver.flagsaver(batch_size=0):
        xspace_to_parquet.main(["xspace_to_parquet"])

  def test_invalid_max_record_count_raises(self) -> None:
    with self.assertRaises(flags.IllegalFlagValueError):
      with flagsaver.flagsaver(max_record_count=0):
        xspace_to_parquet.main(["xspace_to_parquet"])

  def test_compression_level_without_type_raises(self) -> None:
    with (
        flagsaver.flagsaver(
            compression_level=3,
            compression_type=None,
        ),
        self.assertRaisesRegex(
            SystemExit, "compression_level requires compression_type to be set"
        ),
    ):
      xspace_to_parquet.main(["xspace_to_parquet"])

  def test_too_many_args_raises(self) -> None:
    with flagsaver.flagsaver(
        input_path=self._input_path,
        output_path=self._output_path,
    ):
      with self.assertRaisesRegex(
          app.UsageError, "Too many command-line arguments"
      ):
        xspace_to_parquet.main(["xspace_to_parquet", "unexpected_arg"])


if __name__ == "__main__":
  flags.FLAGS.set_default("input_path", epath.Path("dummy"))
  flags.FLAGS.set_default("output_path", epath.Path("dummy"))
  absltest.main()
