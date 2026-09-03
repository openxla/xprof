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
"""Events DB Python API for high-performance accelerator trace analysis.

Record Consumer Semantics & Lifecycle (Transient Record Views):
  When consuming streamed events (via `RecordConsumerRef` or callbacks passed
  to trace parsers), the `Record` instance passed to `consume(record)` is a
  transient, non-owning view directly referencing memory managed by the C++
  parser.

  - Parser Ownership & Memory Reuse:
    To maximize parsing throughput and avoid allocation churn across millions
    of events, the C++ parser retains ownership of the underlying `Record`
    buffer and reuses it across iterations.

  - Lifetime & Cross-Thread Invalidation Warning:
    Consumers MUST NOT retain references or pointers to the `Record` after the
    callback returns, nor pass the `Record` instance to other threads (e.g.
    `threading.Thread`, `queue.Queue`, or thread pools). Doing so causes data
    races, memory corruption, or segmentation faults when the parser clears,
    reuses, or destructs the underlying C++ record:

      ```py
      # DANGEROUS: Storing raw `Record` instances retains transient views!
      def bad_consumer(record: events_db.Record):
        records_list.append(record)  # BUG: Cleared/mutated on next iteration!
        background_queue.put(record)  # BUG: Not thread-safe! Data race!
      ```

  - Safe Usage:
    Extract required field values or copy needed data into independent Python
    structures (such as `dict`, `tuple`, or dataclasses) during the callback:

      ```py
      # SAFE: Extract field values into independent Python objects
      def safe_consumer(record: events_db.Record):
        records_list.append({
            name_field: record[name_field],
            dur_field: record[dur_field],
            tensors_field: tuple(record[tensors_field]),
        })
      ```
"""

from __future__ import annotations

from collections.abc import Sequence
import enum
from typing import TypeAlias

from xprof.convert.events_db.python import pywrap_events_db

# Re-export core C++ classes, enums, and parser functions
# go/keep-sorted start
FieldIndex = pywrap_events_db.FieldIndex
ParseStatus = pywrap_events_db.ParseStatus
Record = pywrap_events_db.Record
RecordConsumerRef = pywrap_events_db.RecordConsumerRef
Schema = pywrap_events_db.Schema
StepControl = pywrap_events_db.StepControl
parse_xspace_bytes = pywrap_events_db.parse_xspace_bytes
parse_xspace_file = pywrap_events_db.parse_xspace_file
# go/keep-sorted end

FieldValue: TypeAlias = (
    None
    | bool
    | int
    | float
    | str
    | Sequence[int]
    | Sequence[float]
    | Sequence[str]
)
"""Type alias for any supported field value in an Events DB `Record`.

Type Conversion (Untyped Python to Strongly-Typed C++):
  Python is dynamically typed and represents all integers as arbitrary-precision
  `int` objects and all sequences as heterogeneous containers (e.g., `list` or
  `tuple`). In contrast, the underlying C++ `Record` stores values in a strictly
  typed object supporting specific scalar widths (`bool`, `int32_t`, `uint32_t`,
  `int64_t`, `uint64_t`, `double`, `std::string`) and homogeneous sequences of
  these types.

  When using generic assignment (`record[field] = val` or
  `record.set(field, val)`), the type is inferred using default mappings:
  - Scalar integers map to `int64_t` (or `uint64_t` if >= 2**63).
  - Floats map to `double`.
  - Homogeneous integer sequences map to sequences of `int64_t`.
  - Homogeneous float sequences map to sequences of `double`.
  - Homogeneous string sequences map to sequences of `std::string`.
  - Empty sequences default to sequences of `int64_t`.

  Explicit Typed Setters:
  Because generic assignment cannot infer narrower integer types (such as
  `int32_t` or `uint32_t`), and incurs dynamic element-inspection overhead for
  sequences, `Record` provides explicit typed setter methods. These allow
  direct specification of the underlying C++ storage type to optimize memory
  and performance:
  - Scalars: `set_bool`, `set_int32`, `set_uint32`, `set_int64`, `set_uint64`,
    `set_double`, `set_string`
  - Sequences: `set_int32_sequence`, `set_uint32_sequence`,
    `set_int64_sequence`, `set_uint64_sequence`, `set_double_sequence`,
    `set_string_sequence`

Repeated Field Semantics (Zero-Copy Sequence Views):
  When reading repeated fields (i.e., `Sequence[int]`, `Sequence[float]`, or
  `Sequence[str]`), the returned object is a read-only, zero-copy view directly
  into the underlying C++ memory.

  - Complexity: Supports true O(1) constant-time random access (`__getitem__`)
    and O(1) length calculation (`__len__`).

  - Lifetime & Invalidation Warning:
    The view's underlying memory is tied to the parent `Record`. While Python's
    garbage collector ensures the parent `Record` remains alive as long as the
    view exists, mutative operations on the `Record` (such as overwriting the
    field or calling `record.clear()`) destroy or reallocate the C++ underlying
    memory. Dereferencing a view after modifying that field causes undefined
    behavior or segmentation faults:

      ```py
      tensors = record[input_tensors]  # Zero-copy view into C++ container

      # DANGEROUS: Overwriting the field destroys the C++ container!
      record[input_tensors] = ["new_tensor"]  # or: record.clear()

      # CRASH / UNDEFINED BEHAVIOR: Dereferencing dangling C++ memory
      print(tensors[0])
      ```

  - Safe Usage:
    Treat returned sequence views as transient. If values must be retained
    while modifying the parent `Record`, explicitly copy elements into an
    independent Python `tuple` or `list`:

      ```py
      # SAFE: Create an independent Python copy before modifying record
      safe_tensors = tuple(record[input_tensors])

      record[input_tensors] = ["new_tensor"]
      print(safe_tensors[0])  # Safe: safe_tensors is an independent copy
      ```
"""


class ArrowCompressionType(enum.Enum):
  """Compression codec applied to Parquet data pages."""

  # go/keep-sorted start
  BROTLI = "BROTLI"
  BZ2 = "BZ2"
  GZIP = "GZIP"
  LZ4 = "LZ4"
  LZ4_FRAME = "LZ4_FRAME"
  LZ4_HADOOP = "LZ4_HADOOP"
  LZO = "LZO"
  SNAPPY = "SNAPPY"
  UNCOMPRESSED = "UNCOMPRESSED"
  ZSTD = "ZSTD"
  # go/keep-sorted end


__all__ = [
    # go/keep-sorted start
    "ArrowCompressionType",
    "FieldIndex",
    "FieldValue",
    "ParseStatus",
    "Record",
    "RecordConsumerRef",
    "Schema",
    "StepControl",
    "parse_xspace_bytes",
    "parse_xspace_file",
    # go/keep-sorted end
]
