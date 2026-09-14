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
"""Tests for count_zero_self_time_events example binary."""

import contextlib
import io

from absl import app
from absl import flags
from absl.testing import absltest
from absl.testing import flagsaver
from etils import epath

from python.runfiles import runfiles

from xprof.convert.events_db.examples.python import count_zero_self_time_events

_TEST_DATA_PATH = (
    "org_xprof/xprof/convert/events_db/examples/test_data/test.xplane.pb"
)


class CountZeroSelfTimeEventsTest(absltest.TestCase):

  def setUp(self) -> None:
    super().setUp()
    self._input_path = runfiles.Create().Rlocation(_TEST_DATA_PATH)

  def test_main_runs_with_flags(self) -> None:
    f = io.StringIO()
    with (
        flagsaver.flagsaver(input_path=self._input_path),
        contextlib.redirect_stdout(f),
    ):
      count_zero_self_time_events.main(["count_zero_self_time_events"])

    with self.subTest("stdout_contains_expected_output"):
      output = f.getvalue()
      self.assertIn("Counting events with zero self_time_ns", output)
      self.assertIn("Successfully finished parsing in", output)
      self.assertIn("with status: ParseStatus.COMPLETE", output)
      self.assertIn("Total records processed: 2", output)
      self.assertIn("Zero self_time_ns events: 1", output)

  def test_too_many_args_raises(self) -> None:
    with flagsaver.flagsaver(
        input_path=self._input_path,
    ), self.assertRaisesRegex(
        app.UsageError, "Too many command-line arguments"
    ):
      count_zero_self_time_events.main(
          ["count_zero_self_time_events", "unexpected_arg"]
      )


if __name__ == "__main__":
  flags.FLAGS.set_default("input_path", epath.Path("dummy"))
  absltest.main()
