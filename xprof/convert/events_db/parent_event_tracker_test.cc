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

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "testing/base/public/gmock.h"
#include "<gtest/gtest.h>"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xprof/convert/events_db/event_utils.h"
#include "xprof/convert/events_db/record_consumer.h"
#include "xprof/convert/events_db/schema.h"

namespace xprof::events_db::internal {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;

Record CreateTestRecord(const FieldIndices& indices, absl::string_view name,
                        uint64_t start_ns, uint64_t duration_ns) {
  Record record;
  record[indices.kernel_name] = std::string(name);
  record[indices.start_ns] = start_ns;
  record[indices.end_ns] = start_ns + duration_ns;
  record[indices.self_time_ns] = duration_ns;
  return record;
}

TEST(ParentEventTrackerTest, SingleEventFlushedAtEnd) {
  Schema schema;
  FieldIndices indices(schema);
  ParentEventTracker tracker;
  std::vector<Record> emitted;

  Record record = CreateTestRecord(indices, "event_0", 100, 500);
  EXPECT_THAT(tracker.AddRecord(std::move(record), 100, 500, indices,
                                [&](Record& r) -> absl::StatusOr<StepControl> {
                                  emitted.push_back(r);
                                  return StepControl::kContinue;
                                }),
              IsOkAndHolds(StepControl::kContinue));
  EXPECT_TRUE(emitted.empty());

  EXPECT_THAT(tracker.Flush([&](Record& r) -> absl::StatusOr<StepControl> {
    emitted.push_back(r);
    return StepControl::kContinue;
  }),
              IsOkAndHolds(StepControl::kContinue));

  ASSERT_EQ(emitted.size(), 1);
  EXPECT_EQ(emitted[0][indices.kernel_name], "event_0");
  EXPECT_EQ(emitted[0][indices.self_time_ns], 500);
}

TEST(ParentEventTrackerTest, InstantEventsAreEmittedImmediately) {
  Schema schema;
  FieldIndices indices(schema);
  ParentEventTracker tracker;
  std::vector<Record> emitted;

  auto consumer = [&](Record& r) -> absl::StatusOr<StepControl> {
    emitted.push_back(r);
    return StepControl::kContinue;
  };

  Record instant_record = CreateTestRecord(indices, "instant", 100, 0);
  EXPECT_THAT(
      tracker.AddRecord(std::move(instant_record), 100, 0, indices, consumer),
      IsOkAndHolds(StepControl::kContinue));
  ASSERT_EQ(emitted.size(), 1);
  EXPECT_EQ(emitted[0][indices.kernel_name], "instant");
  EXPECT_EQ(emitted[0][indices.self_time_ns], 0);

  // A subsequent event is not enclosed by the instant event.
  Record next_record = CreateTestRecord(indices, "next", 150, 200);
  EXPECT_THAT(
      tracker.AddRecord(std::move(next_record), 150, 200, indices, consumer),
      IsOkAndHolds(StepControl::kContinue));
  EXPECT_EQ(emitted.size(), 1);

  EXPECT_THAT(tracker.Flush(consumer), IsOkAndHolds(StepControl::kContinue));
  ASSERT_EQ(emitted.size(), 2);
  EXPECT_EQ(emitted[1][indices.kernel_name], "next");
  EXPECT_EQ(emitted[1][indices.self_time_ns], 200);
}

TEST(ParentEventTrackerTest, NonOverlappingSequentialEvents) {
  Schema schema;
  FieldIndices indices(schema);
  ParentEventTracker tracker;
  std::vector<Record> emitted;

  auto consumer = [&](Record& r) -> absl::StatusOr<StepControl> {
    emitted.push_back(r);
    return StepControl::kContinue;
  };

  // Event 0: [0, 100)
  EXPECT_THAT(tracker.AddRecord(CreateTestRecord(indices, "e0", 0, 100), 0, 100,
                                indices, consumer),
              IsOkAndHolds(StepControl::kContinue));
  EXPECT_TRUE(emitted.empty());

  // Event 1: [100, 250) (starts when e0 ends -> pops e0)
  EXPECT_THAT(tracker.AddRecord(CreateTestRecord(indices, "e1", 100, 150), 100,
                                150, indices, consumer),
              IsOkAndHolds(StepControl::kContinue));
  ASSERT_EQ(emitted.size(), 1);
  EXPECT_EQ(emitted[0][indices.kernel_name], "e0");
  EXPECT_EQ(emitted[0][indices.self_time_ns], 100);

  // Event 2: [300, 500) (starts after e1 ends -> pops e1)
  EXPECT_THAT(tracker.AddRecord(CreateTestRecord(indices, "e2", 300, 200), 300,
                                200, indices, consumer),
              IsOkAndHolds(StepControl::kContinue));
  ASSERT_EQ(emitted.size(), 2);
  EXPECT_EQ(emitted[1][indices.kernel_name], "e1");
  EXPECT_EQ(emitted[1][indices.self_time_ns], 150);

  EXPECT_THAT(tracker.Flush(consumer), IsOkAndHolds(StepControl::kContinue));
  ASSERT_EQ(emitted.size(), 3);
  EXPECT_EQ(emitted[2][indices.kernel_name], "e2");
  EXPECT_EQ(emitted[2][indices.self_time_ns], 200);
}

TEST(ParentEventTrackerTest, SingleLevelNestingDeductsSelfTime) {
  // Parent: [0, 1000) (dur = 1000)
  //   Child 1: [100, 400) (dur = 300)
  //   Child 2: [500, 700) (dur = 200)
  Schema schema;
  FieldIndices indices(schema);
  ParentEventTracker tracker;
  std::vector<Record> emitted;

  auto consumer = [&](Record& r) -> absl::StatusOr<StepControl> {
    emitted.push_back(r);
    return StepControl::kContinue;
  };

  EXPECT_THAT(tracker.AddRecord(CreateTestRecord(indices, "parent", 0, 1000), 0,
                                1000, indices, consumer),
              IsOkAndHolds(StepControl::kContinue));
  EXPECT_TRUE(emitted.empty());

  EXPECT_THAT(tracker.AddRecord(CreateTestRecord(indices, "c1", 100, 300), 100,
                                300, indices, consumer),
              IsOkAndHolds(StepControl::kContinue));
  EXPECT_TRUE(emitted.empty());

  // When c2 arrives at 500, c1 [100, 400) has ended -> c1 is popped and
  // emitted.
  EXPECT_THAT(tracker.AddRecord(CreateTestRecord(indices, "c2", 500, 200), 500,
                                200, indices, consumer),
              IsOkAndHolds(StepControl::kContinue));
  ASSERT_EQ(emitted.size(), 1);
  EXPECT_EQ(emitted[0][indices.kernel_name], "c1");
  EXPECT_EQ(emitted[0][indices.self_time_ns], 300);

  EXPECT_THAT(tracker.Flush(consumer), IsOkAndHolds(StepControl::kContinue));
  ASSERT_EQ(emitted.size(), 3);
  EXPECT_EQ(emitted[1][indices.kernel_name], "c2");
  EXPECT_EQ(emitted[1][indices.self_time_ns], 200);

  // Parent self-time = 1000 - 300 (c1) - 200 (c2) = 500.
  EXPECT_EQ(emitted[2][indices.kernel_name], "parent");
  EXPECT_EQ(emitted[2][indices.self_time_ns], 500);
}

TEST(ParentEventTrackerTest, MultiLevelNestingDeductsSelfTime) {
  // Level 0: [0, 2000) (dur = 2000)
  //   Level 1: [100, 1100) (dur = 1000)
  //     Level 2: [200, 600) (dur = 400)
  //       Level 3: [300, 400) (dur = 100)
  Schema schema;
  FieldIndices indices(schema);
  ParentEventTracker tracker;
  std::vector<Record> emitted;

  auto consumer = [&](Record& r) -> absl::StatusOr<StepControl> {
    emitted.push_back(r);
    return StepControl::kContinue;
  };

  EXPECT_THAT(tracker.AddRecord(CreateTestRecord(indices, "L0", 0, 2000), 0,
                                2000, indices, consumer),
              IsOkAndHolds(StepControl::kContinue));
  EXPECT_THAT(tracker.AddRecord(CreateTestRecord(indices, "L1", 100, 1000), 100,
                                1000, indices, consumer),
              IsOkAndHolds(StepControl::kContinue));
  EXPECT_THAT(tracker.AddRecord(CreateTestRecord(indices, "L2", 200, 400), 200,
                                400, indices, consumer),
              IsOkAndHolds(StepControl::kContinue));
  EXPECT_THAT(tracker.AddRecord(CreateTestRecord(indices, "L3", 300, 100), 300,
                                100, indices, consumer),
              IsOkAndHolds(StepControl::kContinue));

  EXPECT_THAT(tracker.Flush(consumer), IsOkAndHolds(StepControl::kContinue));
  ASSERT_EQ(emitted.size(), 4);

  // Popped from top to bottom on flush:
  EXPECT_EQ(emitted[0][indices.kernel_name], "L3");
  EXPECT_EQ(emitted[0][indices.self_time_ns], 100);

  EXPECT_EQ(emitted[1][indices.kernel_name], "L2");
  EXPECT_EQ(emitted[1][indices.self_time_ns], 400 - 100);  // 300

  EXPECT_EQ(emitted[2][indices.kernel_name], "L1");
  EXPECT_EQ(emitted[2][indices.self_time_ns], 1000 - 400);  // 600

  EXPECT_EQ(emitted[3][indices.kernel_name], "L0");
  EXPECT_EQ(emitted[3][indices.self_time_ns], 2000 - 1000);  // 1000
}

TEST(ParentEventTrackerTest, FullDurationCoverageDeduction) {
  // Parent: [100, 500) (dur = 400)
  //   Child: [100, 500) (dur = 400)
  Schema schema;
  FieldIndices indices(schema);
  ParentEventTracker tracker;
  std::vector<Record> emitted;

  auto consumer = [&](Record& r) -> absl::StatusOr<StepControl> {
    emitted.push_back(r);
    return StepControl::kContinue;
  };

  EXPECT_THAT(tracker.AddRecord(CreateTestRecord(indices, "parent", 100, 400),
                                100, 400, indices, consumer),
              IsOkAndHolds(StepControl::kContinue));
  EXPECT_THAT(tracker.AddRecord(CreateTestRecord(indices, "child", 100, 400),
                                100, 400, indices, consumer),
              IsOkAndHolds(StepControl::kContinue));

  EXPECT_THAT(tracker.Flush(consumer), IsOkAndHolds(StepControl::kContinue));
  ASSERT_EQ(emitted.size(), 2);
  EXPECT_EQ(emitted[0][indices.kernel_name], "child");
  EXPECT_EQ(emitted[0][indices.self_time_ns], 400);

  EXPECT_EQ(emitted[1][indices.kernel_name], "parent");
  EXPECT_EQ(emitted[1][indices.self_time_ns], 0);
}

TEST(ParentEventTrackerTest, PartiallyOverlappingIntervals) {
  // First event: [100, 500) (dur = 400)
  // Second event: [300, 600) (dur = 300) - partially overlaps, so first event
  // is popped and emitted without deduction from second event.
  Schema schema;
  FieldIndices indices(schema);
  ParentEventTracker tracker;
  std::vector<Record> emitted;

  auto consumer = [&](Record& r) -> absl::StatusOr<StepControl> {
    emitted.push_back(r);
    return StepControl::kContinue;
  };

  EXPECT_THAT(tracker.AddRecord(CreateTestRecord(indices, "e0", 100, 400), 100,
                                400, indices, consumer),
              IsOkAndHolds(StepControl::kContinue));
  EXPECT_THAT(tracker.AddRecord(CreateTestRecord(indices, "e1", 300, 300), 300,
                                300, indices, consumer),
              IsOkAndHolds(StepControl::kContinue));
  ASSERT_EQ(emitted.size(), 1);
  EXPECT_EQ(emitted[0][indices.kernel_name], "e0");
  EXPECT_EQ(emitted[0][indices.self_time_ns], 400);

  // Child: [350, 450) (dur = 100) is enclosed in e1.
  EXPECT_THAT(tracker.AddRecord(CreateTestRecord(indices, "child", 350, 100),
                                350, 100, indices, consumer),
              IsOkAndHolds(StepControl::kContinue));

  EXPECT_THAT(tracker.Flush(consumer), IsOkAndHolds(StepControl::kContinue));
  ASSERT_EQ(emitted.size(), 3);
  EXPECT_EQ(emitted[1][indices.kernel_name], "child");
  EXPECT_EQ(emitted[1][indices.self_time_ns], 100);

  EXPECT_EQ(emitted[2][indices.kernel_name], "e1");
  EXPECT_EQ(emitted[2][indices.self_time_ns], 300 - 100);  // 200
}

TEST(ParentEventTrackerTest, StopsEarlyWhenConsumerRequestsStop) {
  Schema schema;
  FieldIndices indices(schema);
  ParentEventTracker tracker;

  EXPECT_THAT(tracker.AddRecord(CreateTestRecord(indices, "e0", 0, 100), 0, 100,
                                indices,
                                [](Record&) -> absl::StatusOr<StepControl> {
                                  return StepControl::kContinue;
                                }),
              IsOkAndHolds(StepControl::kContinue));

  // When e1 arrives at 100, e0 pops. Consumer requests stop.
  EXPECT_THAT(tracker.AddRecord(CreateTestRecord(indices, "e1", 100, 100), 100,
                                100, indices,
                                [](Record&) -> absl::StatusOr<StepControl> {
                                  return StepControl::kStop;
                                }),
              IsOkAndHolds(StepControl::kStop));
}

TEST(ParentEventTrackerTest, PropagatesConsumerError) {
  Schema schema;
  FieldIndices indices(schema);
  ParentEventTracker tracker;

  EXPECT_THAT(tracker.AddRecord(CreateTestRecord(indices, "e0", 0, 100), 0, 100,
                                indices,
                                [](Record&) -> absl::StatusOr<StepControl> {
                                  return StepControl::kContinue;
                                }),
              IsOkAndHolds(StepControl::kContinue));

  EXPECT_THAT(tracker.AddRecord(CreateTestRecord(indices, "e1", 100, 100), 100,
                                100, indices,
                                [](Record&) -> absl::StatusOr<StepControl> {
                                  return absl::InternalError("disk failure");
                                }),
              StatusIs(absl::StatusCode::kInternal, "disk failure"));
}

TEST(ParentEventTrackerTest, DeductsWithUnderflowProtection) {
  Schema schema;
  FieldIndices indices(schema);
  ParentEventTracker tracker;
  std::vector<Record> emitted;

  auto consumer = [&](Record& r) -> absl::StatusOr<StepControl> {
    emitted.push_back(r);
    return StepControl::kContinue;
  };

  // Parent duration = 100, self_time initialized to 50
  Record parent = CreateTestRecord(indices, "parent", 0, 100);
  parent[indices.self_time_ns] = 50;
  EXPECT_THAT(tracker.AddRecord(std::move(parent), 0, 100, indices, consumer),
              IsOkAndHolds(StepControl::kContinue));

  // Child duration = 80 > 50 (underflow condition)
  EXPECT_THAT(tracker.AddRecord(CreateTestRecord(indices, "child", 10, 80), 10,
                                80, indices, consumer),
              IsOkAndHolds(StepControl::kContinue));

  EXPECT_THAT(tracker.Flush(consumer), IsOkAndHolds(StepControl::kContinue));
  ASSERT_EQ(emitted.size(), 2);
  EXPECT_EQ(emitted[1][indices.kernel_name], "parent");
  EXPECT_EQ(emitted[1][indices.self_time_ns], 0);
}

TEST(ParentEventTrackerTest, GetOrCreateRecordReturnsNewRecordWhenBufferEmpty) {
  ParentEventTracker tracker(4);
  EXPECT_EQ(tracker.buffer_capacity(), 4);
  EXPECT_EQ(tracker.buffer_size(), 0);

  Record record = tracker.GetOrCreateRecord();
  EXPECT_EQ(record.size(), 0);
  EXPECT_EQ(tracker.buffer_size(), 0);
}

TEST(ParentEventTrackerTest, BufferRecyclesPoppedRecords) {
  Schema schema;
  FieldIndices indices(schema);
  ParentEventTracker tracker(2);
  std::vector<Record> emitted;

  auto consumer = [&](Record& r) -> absl::StatusOr<StepControl> {
    emitted.push_back(r);
    return StepControl::kContinue;
  };

  // Parent [0, 1000) and Child [100, 400)
  EXPECT_THAT(tracker.AddRecord(CreateTestRecord(indices, "parent", 0, 1000), 0,
                                1000, indices, consumer),
              IsOkAndHolds(StepControl::kContinue));
  EXPECT_THAT(tracker.AddRecord(CreateTestRecord(indices, "child", 100, 300),
                                100, 300, indices, consumer),
              IsOkAndHolds(StepControl::kContinue));
  EXPECT_EQ(tracker.buffer_size(), 0);

  // When next event [500, 700) arrives, child [100, 400) is popped, emitted,
  // and recycled to the buffer.
  EXPECT_THAT(tracker.AddRecord(CreateTestRecord(indices, "next", 500, 200),
                                500, 200, indices, consumer),
              IsOkAndHolds(StepControl::kContinue));
  EXPECT_EQ(emitted.size(), 1);
  EXPECT_EQ(tracker.buffer_size(), 1);

  // Reuse recycled record from buffer.
  Record recycled = tracker.GetOrCreateRecord();
  EXPECT_EQ(recycled.size(), 0);
  EXPECT_EQ(tracker.buffer_size(), 0);

  // Flush remaining records (next and parent). Both are recycled up to capacity
  // 2.
  EXPECT_THAT(tracker.Flush(consumer), IsOkAndHolds(StepControl::kContinue));
  EXPECT_EQ(emitted.size(), 3);
  EXPECT_EQ(tracker.buffer_size(), 2);
}

TEST(ParentEventTrackerTest, BufferRecyclesInstantEvents) {
  Schema schema;
  FieldIndices indices(schema);
  ParentEventTracker tracker(2);
  std::vector<Record> emitted;

  auto consumer = [&](Record& r) -> absl::StatusOr<StepControl> {
    emitted.push_back(r);
    return StepControl::kContinue;
  };

  // Instant event is emitted immediately and recycled to the buffer.
  Record instant_record = CreateTestRecord(indices, "instant", 100, 0);
  EXPECT_THAT(
      tracker.AddRecord(std::move(instant_record), 100, 0, indices, consumer),
      IsOkAndHolds(StepControl::kContinue));
  EXPECT_EQ(emitted.size(), 1);
  EXPECT_EQ(tracker.buffer_size(), 1);

  Record recycled = tracker.GetOrCreateRecord();
  EXPECT_EQ(recycled.size(), 0);
  EXPECT_EQ(tracker.buffer_size(), 0);
}

TEST(ParentEventTrackerTest, BufferRespectsCapacityLimit) {
  Schema schema;
  FieldIndices indices(schema);
  ParentEventTracker tracker(2);
  std::vector<Record> emitted;

  auto consumer = [&](Record& r) -> absl::StatusOr<StepControl> {
    emitted.push_back(r);
    return StepControl::kContinue;
  };

  // Add 5 sequential non-overlapping events.
  for (int i = 0; i < 5; ++i) {
    EXPECT_THAT(tracker.AddRecord(CreateTestRecord(indices, "e", i * 100, 50),
                                  i * 100, 50, indices, consumer),
                IsOkAndHolds(StepControl::kContinue));
  }
  EXPECT_THAT(tracker.Flush(consumer), IsOkAndHolds(StepControl::kContinue));
  EXPECT_EQ(emitted.size(), 5);

  // Only 2 records should be buffered since capacity is 2.
  EXPECT_EQ(tracker.buffer_size(), 2);

  // Pop both buffered records.
  EXPECT_EQ(tracker.GetOrCreateRecord().size(), 0);
  EXPECT_EQ(tracker.buffer_size(), 1);
  EXPECT_EQ(tracker.GetOrCreateRecord().size(), 0);
  EXPECT_EQ(tracker.buffer_size(), 0);

  // Next call constructs a fresh record.
  EXPECT_EQ(tracker.GetOrCreateRecord().size(), 0);
  EXPECT_EQ(tracker.buffer_size(), 0);
}

TEST(ParentEventTrackerTest, ZeroCapacityBufferDoesNotCache) {
  Schema schema;
  FieldIndices indices(schema);
  ParentEventTracker tracker(0);
  EXPECT_EQ(tracker.buffer_capacity(), 0);

  std::vector<Record> emitted;
  auto consumer = [&](Record& r) -> absl::StatusOr<StepControl> {
    emitted.push_back(r);
    return StepControl::kContinue;
  };

  EXPECT_THAT(tracker.AddRecord(CreateTestRecord(indices, "e0", 0, 100), 0, 100,
                                indices, consumer),
              IsOkAndHolds(StepControl::kContinue));
  EXPECT_THAT(tracker.Flush(consumer), IsOkAndHolds(StepControl::kContinue));
  EXPECT_EQ(emitted.size(), 1);
  EXPECT_EQ(tracker.buffer_size(), 0);

  Record record = tracker.GetOrCreateRecord();
  EXPECT_EQ(record.size(), 0);
  EXPECT_EQ(tracker.buffer_size(), 0);
}

TEST(ParentEventTrackerTest,
     ResetAllowsStartingNewTimelineWithEarlierTimestamp) {
  Schema schema;
  FieldIndices indices(schema);
  ParentEventTracker tracker(2);

  std::vector<Record> emitted;
  auto consumer = [&](Record& r) -> absl::StatusOr<StepControl> {
    emitted.push_back(r);
    return StepControl::kContinue;
  };

  // Timeline 1: [1000, 2000)
  EXPECT_THAT(
      tracker.AddRecord(CreateTestRecord(indices, "line1_event", 1000, 1000),
                        1000, 1000, indices, consumer),
      IsOkAndHolds(StepControl::kContinue));
  EXPECT_THAT(tracker.Flush(consumer), IsOkAndHolds(StepControl::kContinue));
  ASSERT_EQ(emitted.size(), 1);
  EXPECT_EQ(emitted[0][indices.kernel_name], "line1_event");

  // Reset tracker for Timeline 2.
  tracker.Reset();

  // Timeline 2: [100, 500) (earlier timestamp than Timeline 1's end time)
  EXPECT_THAT(
      tracker.AddRecord(CreateTestRecord(indices, "line2_event", 100, 400), 100,
                        400, indices, consumer),
      IsOkAndHolds(StepControl::kContinue));
  EXPECT_THAT(tracker.Flush(consumer), IsOkAndHolds(StepControl::kContinue));
  ASSERT_EQ(emitted.size(), 2);
  EXPECT_EQ(emitted[1][indices.kernel_name], "line2_event");
}

TEST(ParentEventTrackerTest, ResetPreservesBufferedRecords) {
  Schema schema;
  FieldIndices indices(schema);
  ParentEventTracker tracker(2);

  std::vector<Record> emitted;
  auto consumer = [&](Record& r) -> absl::StatusOr<StepControl> {
    emitted.push_back(r);
    return StepControl::kContinue;
  };

  EXPECT_THAT(tracker.AddRecord(CreateTestRecord(indices, "e0", 0, 100), 0, 100,
                                indices, consumer),
              IsOkAndHolds(StepControl::kContinue));
  EXPECT_THAT(tracker.Flush(consumer), IsOkAndHolds(StepControl::kContinue));
  EXPECT_EQ(tracker.buffer_size(), 1);

  // Reset preserves the recycled record in the buffer.
  tracker.Reset();
  EXPECT_EQ(tracker.buffer_size(), 1);

  Record recycled = tracker.GetOrCreateRecord();
  EXPECT_EQ(recycled.size(), 0);
  EXPECT_EQ(tracker.buffer_size(), 0);
}

TEST(ParentEventTrackerTest, ResetClearsActiveStack) {
  Schema schema;
  FieldIndices indices(schema);
  ParentEventTracker tracker;

  std::vector<Record> emitted;
  auto consumer = [&](Record& r) -> absl::StatusOr<StepControl> {
    emitted.push_back(r);
    return StepControl::kContinue;
  };

  EXPECT_THAT(tracker.AddRecord(CreateTestRecord(indices, "unflushed", 0, 1000),
                                0, 1000, indices, consumer),
              IsOkAndHolds(StepControl::kContinue));
  EXPECT_TRUE(emitted.empty());

  tracker.Reset();

  EXPECT_THAT(tracker.Flush(consumer), IsOkAndHolds(StepControl::kContinue));
  EXPECT_TRUE(emitted.empty());
}

}  // namespace
}  // namespace xprof::events_db::internal
