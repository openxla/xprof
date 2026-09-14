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
r"""Example CLI to count events with zero self_time_ns in an XSpace trace.

Usage:
  ```shell
    bazel run \
        //xprof/convert/events_db/examples/python:count_zero_self_time_events \
      -- \
      --input_path=/path/to/trace.xplane.pb
  ```
"""

from collections.abc import Sequence
import os
import threading
import time

from absl import app
from etils import epath

from xprof.convert.events_db.python import events_db

_INPUT_PATH = epath.DEFINE_path(
    "input_path",
    None,
    "Path to input XSpace/trace file (e.g., .xplane.pb).",
    required=True,
)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError(f"Too many command-line arguments: {argv[1:]}")

  print(
      "Counting events with zero self_time_ns in XSpace trace...\n"
      f"  Input XSpace: {os.fspath(_INPUT_PATH.value)!r}"
  )

  lock = threading.Lock()
  schema = events_db.Schema()
  field = schema.register_field_name("self_time_ns")
  total_records = 0
  zero_self_time_count = 0

  def consume(record: events_db.Record) -> events_db.StepControl:
    nonlocal total_records, zero_self_time_count
    is_zero = record.get(field) == 0
    with lock:
      total_records += 1
      if is_zero:
        zero_self_time_count += 1
    return events_db.StepControl.CONTINUE

  start_time = time.perf_counter()
  status = events_db.parse_xspace_file(
      file_path=os.fspath(_INPUT_PATH.value),
      schema=schema,
      consumer=consume,
  )
  elapsed_seconds = time.perf_counter() - start_time

  print(
      f"Successfully finished parsing in {elapsed_seconds:.2f}s with status: "
      f"{status!r}\n"
      f"  Total records processed: {total_records}\n"
      f"  Zero self_time_ns events: {zero_self_time_count}"
  )


if __name__ == "__main__":
  app.run(main)
