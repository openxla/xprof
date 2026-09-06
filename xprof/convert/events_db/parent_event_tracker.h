/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef THIRD_PARTY_XPROF_CONVERT_EVENTS_DB_PARENT_EVENT_TRACKER_H_
#define THIRD_PARTY_XPROF_CONVERT_EVENTS_DB_PARENT_EVENT_TRACKER_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "xprof/convert/events_db/event_utils.h"
#include "xprof/convert/events_db/record_consumer.h"
#include "xprof/convert/events_db/schema.h"

namespace xprof::events_db::internal {

// Stateful interval stack that tracks the active parent hierarchy on a
// timeline, deducts child durations from enclosing parent self-times, and
// streams finalized `Record`s to a `RecordConsumerRef`.
//
// As events on a line are processed chronologically, this class maintains an
// active parent stack to identify enclosing parent intervals and finalize
// records as soon as their intervals expire.
//
// To minimize allocation overhead, this class maintains an internal buffer of
// reusable `Record`s with a configurable capacity `buffer_capacity`. When
// records are finished (e.g., popped from the active parent stack), they are
// retained in the buffer up to `buffer_capacity`. Callers can call
// `GetOrCreateRecord()` to obtain a recycled record (or construct a new one if
// the buffer is empty).
class ParentEventTracker {
 public:
  // Constructs a new instance with the specified maximum record buffer
  // capacity. When `buffer_capacity > 0`, up to `buffer_capacity` recycled
  // `Record` instances are cached in a buffer when their lifecycles finish,
  // allowing them to be reused via `GetOrCreateRecord()`.
  explicit ParentEventTracker(uint32_t buffer_capacity = 0);

  // Obtains a `Record` instance.
  //
  // If the internal buffer contains cached records, pops and returns one
  // (cleared of previous fields). Otherwise, constructs and returns a new
  // `Record`.
  Record GetOrCreateRecord();

  // Ingests a new `Record` along the timeline:
  // 1. Pops all stack entries that do not enclose
  //    `[start_ns, start_ns + duration_ns)` and emits their finalized records
  //    to `consumer`. If the buffer has capacity, recycled records are moved
  //    to the buffer.
  // 2. If `duration_ns == 0` (instant event), emits `record` directly to
  //    `consumer` without pushing it to the stack, and retains it in the buffer
  //    if capacity allows.
  // 3. If `duration_ns > 0`:
  //    - If an active parent encloses this event, deducts `duration_ns` from
  //      the parent's `self_time_ns` field.
  //    - Pushes `{std::move(record), start_ns, start_ns + duration_ns}` onto
  //      the stack.
  //
  // Callers must pass events in chronological order by `start_ns` time. If
  // multiple events share the exact same `start_ns` time, they must be ordered
  // by `duration_ns` in descending order so that larger enclosing parent events
  // are processed before their nested children.
  //
  // Returns:
  // - `StepControl::kContinue`: Ingestion should proceed normally.
  // - `StepControl::kStop`: The `consumer` requested a clean early stop (e.g.,
  //   a record limit was reached or a target event was matched). Processing
  //   halts immediately, no further records are popped or ingested, and the
  //   caller should terminate parsing cleanly.
  // - An error `absl::Status` if `consumer` returned an error, aborting
  //   ingestion immediately.
  absl::StatusOr<StepControl> AddRecord(Record record, uint64_t start_ns,
                                        uint64_t duration_ns,
                                        const FieldIndices& indices,
                                        RecordConsumerRef consumer);

  // Flushes and emits all remaining active records on the stack to `consumer`
  // at the end of the timeline. Emitted records are moved to the buffer if
  // capacity allows. Returns `StepControl::kStop` if `consumer` requested an
  // early stop during flushing.
  absl::StatusOr<StepControl> Flush(RecordConsumerRef consumer);

  // Resets the active stack and timeline tracking state for processing a new
  // timeline track, while preserving buffered records for reuse.
  void Reset();

  // Returns the maximum number of records that can be buffered for reuse.
  uint32_t buffer_capacity() const { return buffer_capacity_; }

  // Returns the number of cached records currently available in the buffer.
  size_t buffer_size() const { return record_buffer_.size(); }

 private:
  struct StackEntry {
    StackEntry(Record record, uint64_t start_ns, uint64_t end_ns)
        : record(std::move(record)), start_ns(start_ns), end_ns(end_ns) {}

    Record record;
    uint64_t start_ns;
    uint64_t end_ns;
  };
  uint32_t buffer_capacity_ = 0;
  std::vector<Record> record_buffer_;
  std::vector<StackEntry> stack_;

  // Only used in debug mode for assertions. They are not enclosed in `NDEBUG`
  // to avoid violating the C++ One Definition Rule.
  std::optional<uint64_t> last_start_ns_;
  uint64_t last_duration_ns_ = 0;
};

}  // namespace xprof::events_db::internal

#endif  // THIRD_PARTY_XPROF_CONVERT_EVENTS_DB_PARENT_EVENT_TRACKER_H_
