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
r"""Example CLI to convert an XSpace trace into an Events DB Parquet file.

Usage:
  Only the first two flags are required.
  ```shell
    bazel run \
        //xprof/convert/events_db/examples/python:xspace_to_parquet \
      -- \
      --input_path=/path/to/trace.xplane.pb \
      --output_path=/path/to/events.parquet \
      --batch_size=65536 \
      --compression_type=SNAPPY \
      --max_record_count=10
  ```
"""

from collections.abc import Sequence
import os
import sys
import time
from absl import app
from absl import flags
from etils import epath
from xprof.convert.events_db.python import events_db

_INPUT_PATH = epath.DEFINE_path(
    "input_path",
    None,
    "Path to input XSpace/trace file (e.g., .xplane.pb).",
    required=True,
)
_OUTPUT_PATH = epath.DEFINE_path(
    "output_path",
    None,
    "Path to output Events DB Parquet file.",
    required=True,
)

# Arrow / Parquet export options flags.
_BATCH_SIZE = flags.DEFINE_integer(
    "batch_size",
    65536,
    "Number of records buffered before flushing a batch to disk.",
    lower_bound=1,
)
_COMPRESSION_TYPE = flags.DEFINE_enum_class(
    "compression_type",
    None,
    events_db.ArrowCompressionType,
    "Compression codec applied to Parquet data pages (e.g., SNAPPY, ZSTD).",
)
_COMPRESSION_LEVEL = flags.DEFINE_integer(
    "compression_level",
    None,
    "Compressor-specific compression level (e.g. 1-22 for ZSTD, 1-9 for GZIP).",
)
_MAX_RECORD_COUNT = flags.DEFINE_integer(
    "max_record_count",
    None,
    "If set, at most this many records will be written before stopping.",
    lower_bound=1,
)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError(f"Too many command-line arguments: {argv[1:]}")

  try:
    options = events_db.ParquetExportOptions(
        batch_size=_BATCH_SIZE.value,
        compression_type=_COMPRESSION_TYPE.value,
        compression_level=_COMPRESSION_LEVEL.value,
        max_record_count=_MAX_RECORD_COUNT.value,
    )
  except ValueError as err:
    sys.exit(f"Error: {err}")  # Use `str` not `repr`

  compression_str = (
      options.compression_type.value if options.compression_type else "None"
  )
  print(
      "Converting XSpace trace to Events DB...\n"
      f"  Input XSpace: {os.fspath(_INPUT_PATH.value)!r}\n"
      f"  Output Parquet: {os.fspath(_OUTPUT_PATH.value)!r}\n"
      f"  Batch Size: {options.batch_size!r}\n"
      f"  Compression Type: {compression_str!r}\n"
      f"  Compression Level: {options.compression_level!r}\n"
      f"  Max Record Count: {options.max_record_count!r}"
  )

  start_time = time.perf_counter()
  schema = events_db.Schema()
  consumer = events_db.ParquetRecordConsumer(
      schema=schema,
      file_path=os.fspath(_OUTPUT_PATH.value),
      options=options,
  )
  status = events_db.parse_xspace_file(
      file_path=os.fspath(_INPUT_PATH.value),
      schema=schema,
      consumer=consumer,
  )
  elapsed_seconds = time.perf_counter() - start_time

  print(
      f"Successfully finished parsing in {elapsed_seconds:.2f}s with status: "
      f"{status!r}"
  )


if __name__ == "__main__":
  app.run(main)
