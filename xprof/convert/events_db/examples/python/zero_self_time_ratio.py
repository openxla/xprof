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
r"""Example CLI to find the ratio of events with zero `self_time_ns`.

This script uses sqlite3 to compute the zero self-time ratio.

Usage:
  ```shell
    bazel run \
        //xprof/convert/events_db/examples/python:zero_self_time_ratio \
      -- \
      --input_path=/path/to/trace.xplane.pb
  ```
"""

from collections.abc import Sequence
import os
import sqlite3
import textwrap
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

  input_path = os.fspath(_INPUT_PATH.value)
  print(
      "Calculating ratio of events with zero self_time_ns using sqlite3...\n"
      f"  Input XSpace: {input_path!r}"
  )

  start_time = time.perf_counter()
  lock = threading.Lock()
  schema = events_db.Schema()
  self_time_field = schema.register_field_name("self_time_ns")
  records: list[tuple[float | None]] = []

  def consume(record: events_db.Record) -> events_db.StepControl:
    self_time = record.get(self_time_field)
    with lock:
      records.append((self_time,))
    return events_db.StepControl.CONTINUE

  status = events_db.parse_xspace_file(
      file_path=input_path,
      schema=schema,
      consumer=consume,
  )
  if status != events_db.ParseStatus.COMPLETE:
    print(f"Parsing failed with status: {status!r}.")
    return

  with sqlite3.connect(":memory:") as conn:
    conn.execute("CREATE TABLE events (self_time_ns REAL)")
    conn.executemany("INSERT INTO events VALUES (?)", records)
    query = textwrap.dedent("""\
        SELECT
          COUNT(*) AS total_records,
          IFNULL(SUM(self_time_ns = 0), 0) AS zero_self_time_count,
          IFNULL(AVG(self_time_ns = 0), 0) AS zero_self_time_ratio
        FROM events""")
    total_records, zero_self_time_count, zero_self_time_ratio = conn.execute(
        query
    ).fetchone()
  elapsed_seconds = time.perf_counter() - start_time

  print(
      f"Successfully finished parsing in {elapsed_seconds:.2f}s with status: "
      f"{status!r}\n"
      f"  Total records processed: {total_records}\n"
      f"  Zero self_time_ns events: {zero_self_time_count}\n"
      f"  Zero self_time_ns ratio: {zero_self_time_ratio:.4f} "
      f"({zero_self_time_ratio * 100:.2f}%)"
  )


if __name__ == "__main__":
  app.run(main)
