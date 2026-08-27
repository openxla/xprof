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

#include "xprof/convert/events_db/parent_event_tracker.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "xprof/convert/events_db/event_utils.h"
#include "xprof/convert/events_db/record_consumer.h"
#include "xprof/convert/events_db/schema.h"

namespace xprof::events_db::internal {

namespace {

// Deducts child duration from parent self-time. Replaces underflow with 0.
inline uint64_t DeductDuration(uint64_t parent_self_time,
                               uint64_t child_duration) {
  return parent_self_time - std::min(parent_self_time, child_duration);
}

// Returns true if [parent_start_ns, parent_end_ns) encloses
// [child_start_ns, child_end_ns). `child_start_ns < parent_end_ns` enforces
// strict half-open interval semantics for zero-duration (instant) events at the
// exclusive end boundary.
inline bool Encloses(uint64_t parent_start_ns, uint64_t parent_end_ns,
                     uint64_t child_start_ns, uint64_t child_end_ns) {
  return parent_start_ns <= child_start_ns && child_end_ns <= parent_end_ns &&
         child_start_ns < parent_end_ns;
}

void AddIfHasCapacity(Record&& record, std::vector<Record>& buffer,
                      uint32_t buffer_capacity) {
  if (buffer.size() < buffer_capacity) {
    buffer.push_back(std::move(record));
  }
}

}  // namespace

ParentEventTracker::ParentEventTracker(uint32_t buffer_capacity)
    : buffer_capacity_(buffer_capacity) {
  record_buffer_.reserve(buffer_capacity_);
}

Record ParentEventTracker::GetOrCreateRecord() {
  if (!record_buffer_.empty()) {
    Record record = std::move(record_buffer_.back());
    record_buffer_.pop_back();
    record.clear();
    return record;
  }
  return Record();
}

absl::StatusOr<StepControl> ParentEventTracker::AddRecord(
    Record record, uint64_t start_ns, uint64_t duration_ns,
    const FieldIndices& indices, RecordConsumerRef consumer) {
#ifndef NDEBUG
  if (last_start_ns_.has_value()) {
    DCHECK_LE(*last_start_ns_, start_ns);
    if (*last_start_ns_ == start_ns) {
      DCHECK_LE(duration_ns, last_duration_ns_);
    }
  }
  last_start_ns_ = start_ns;
  last_duration_ns_ = duration_ns;
#endif

  const uint64_t end_ns = start_ns + duration_ns;
  DCHECK_GE(end_ns, start_ns);

  while (!stack_.empty() && !Encloses(stack_.back().start_ns,
                                      stack_.back().end_ns, start_ns, end_ns)) {
    ASSIGN_OR_RETURN(const StepControl status, consumer(stack_.back().record));
    AddIfHasCapacity(std::move(stack_.back().record), record_buffer_,
                     buffer_capacity_);
    stack_.pop_back();
    if (status == StepControl::kStop) return status;
  }

  if (duration_ns == 0) {
    ASSIGN_OR_RETURN(const StepControl status, consumer(record));
    AddIfHasCapacity(std::move(record), record_buffer_, buffer_capacity_);
    return status;
  }

  if (!stack_.empty()) {
    Record& parent = stack_.back().record;
    parent[indices.self_time_ns] =
        DeductDuration(parent[indices.self_time_ns], duration_ns);
  }

  stack_.emplace_back(std::move(record), start_ns, end_ns);
  return StepControl::kContinue;
}

absl::StatusOr<StepControl> ParentEventTracker::Flush(
    RecordConsumerRef consumer) {
  while (!stack_.empty()) {
    ASSIGN_OR_RETURN(const StepControl status, consumer(stack_.back().record));
    AddIfHasCapacity(std::move(stack_.back().record), record_buffer_,
                     buffer_capacity_);
    stack_.pop_back();
    if (status == StepControl::kStop) return status;
  }
  return StepControl::kContinue;
}

void ParentEventTracker::Reset() {
  stack_.clear();
#ifndef NDEBUG
  last_start_ns_ = std::nullopt;
  last_duration_ns_ = 0;
#endif
}

}  // namespace xprof::events_db::internal
