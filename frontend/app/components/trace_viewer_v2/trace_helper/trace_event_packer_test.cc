#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_event_packer.h"

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include "absl/types/span.h"
#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_event.h"

namespace traceviewer {
namespace {

std::vector<const TraceEvent*> GetEventPointers(
    absl::Span<const TraceEvent> events) {
  std::vector<const TraceEvent*> events_ptr;
  events_ptr.reserve(events.size());
  for (const TraceEvent& event : events) {
    events_ptr.push_back(&event);
  }
  return events_ptr;
}

TEST(TraceEventPackerTest, EmptyEvents) {
  const std::vector<TraceEvent> events;
  auto events_ptr = GetEventPointers(events);
  const auto packed = PackTraceEvents(events_ptr);
  EXPECT_TRUE(packed.empty());
}

TEST(TraceEventPackerTest, SingleEvent) {
  const std::vector<TraceEvent> events = {{.ts = 10.0, .dur = 20.0}};
  auto events_ptr = GetEventPointers(events);
  const auto packed = PackTraceEvents(events_ptr);

  ASSERT_EQ(packed.size(), 1);
  EXPECT_EQ(packed[0].event->ts, 10.0);
  EXPECT_EQ(packed[0].level, 0);
}

TEST(TraceEventPackerTest, SequentialEventsOnSameLevel) {
  const std::vector<TraceEvent> events = {{.ts = 10.0, .dur = 20.0},
                                          {.ts = 30.0, .dur = 10.0}};
  auto events_ptr = GetEventPointers(events);
  const auto packed = PackTraceEvents(events_ptr);

  ASSERT_EQ(packed.size(), 2);
  EXPECT_EQ(packed[0].event->ts, 10.0);
  EXPECT_EQ(packed[0].level, 0);
  EXPECT_EQ(packed[1].event->ts, 30.0);
  EXPECT_EQ(packed[1].level, 0);
}

TEST(TraceEventPackerTest, NestedEventsOnDifferentLevels) {
  const std::vector<TraceEvent> events = {
      {.ts = 10.0, .dur = 30.0},  // Parent
      {.ts = 15.0, .dur = 10.0}   // Child
  };
  auto events_ptr = GetEventPointers(events);
  const auto packed = PackTraceEvents(events_ptr);

  ASSERT_EQ(packed.size(), 2);
  // Sort order check
  EXPECT_EQ(packed[0].event->ts, 10.0);
  EXPECT_EQ(packed[0].level, 0);
  EXPECT_EQ(packed[1].event->ts, 15.0);
  EXPECT_EQ(packed[1].level, 1);
}

TEST(TraceEventPackerTest, NestedSameStartTieBreaker) {
  const std::vector<TraceEvent> events = {
      {.name = "Child", .ts = 10.0, .dur = 10.0},
      {.name = "Parent", .ts = 10.0, .dur = 30.0}};
  auto events_ptr = GetEventPointers(events);
  const auto packed = PackTraceEvents(events_ptr);

  ASSERT_EQ(packed.size(), 2);
  // Parent should be first in sorted output and have level 0.
  EXPECT_EQ(packed[0].event->name, "Parent");
  EXPECT_EQ(packed[0].level, 0);
  EXPECT_EQ(packed[1].event->name, "Child");
  EXPECT_EQ(packed[1].level, 1);
}

TEST(TraceEventPackerTest, OverlappingEventsOnDifferentLevels) {
  // Overlapping but not nested
  // Event A: [10, 30]
  // Event B: [20, 40]
  const std::vector<TraceEvent> events = {
      {.name = "A", .ts = 10.0, .dur = 20.0},
      {.name = "B", .ts = 20.0, .dur = 20.0}};
  auto events_ptr = GetEventPointers(events);
  const auto packed = PackTraceEvents(events_ptr);

  ASSERT_EQ(packed.size(), 2);
  EXPECT_EQ(packed[0].event->name, "A");
  EXPECT_EQ(packed[0].level, 0);
  EXPECT_EQ(packed[1].event->name, "B");
  EXPECT_EQ(packed[1].level, 1);
}

TEST(TraceEventPackerTest, OverlapReusesRow) {
  // Event A: [10, 30]
  // Event B: [20, 40] (overlaps A, goes to level 1)
  // Event C: [35, 50] (starts after A ends, overlaps B. Reuses level 0)
  const std::vector<TraceEvent> events = {
      {.name = "A", .ts = 10.0, .dur = 20.0},
      {.name = "B", .ts = 20.0, .dur = 20.0},
      {.name = "C", .ts = 35.0, .dur = 15.0}};
  auto events_ptr = GetEventPointers(events);
  const auto packed = PackTraceEvents(events_ptr);

  ASSERT_EQ(packed.size(), 3);
  EXPECT_EQ(packed[0].event->name, "A");
  EXPECT_EQ(packed[0].level, 0);
  EXPECT_EQ(packed[1].event->name, "B");
  EXPECT_EQ(packed[1].level, 1);
  EXPECT_EQ(packed[2].event->name, "C");
  EXPECT_EQ(packed[2].level, 0);
}

TEST(TraceEventPackerTest, IgnoresNullPointers) {
  const TraceEvent event_a = {.name = "A", .ts = 10.0, .dur = 20.0};
  const TraceEvent event_b = {.name = "B", .ts = 20.0, .dur = 20.0};
  const std::vector<const TraceEvent*> events_ptr = {&event_a, nullptr,
                                                     &event_b, nullptr};
  const auto packed = PackTraceEvents(events_ptr);

  ASSERT_EQ(packed.size(), 2);
  EXPECT_EQ(packed[0].event->name, "A");
  EXPECT_EQ(packed[0].level, 0);
  EXPECT_EQ(packed[1].event->name, "B");
  EXPECT_EQ(packed[1].level, 1);
}

TEST(TraceEventPackerTest,
     PicosecondAdjacentEventsWithDoubleRoundingStayOnSameLevel) {
  constexpr uint64_t kStart0Ps = 14043458000ULL;
  constexpr uint64_t kDur0Ps = 22113441836ULL;
  constexpr uint64_t kStart1Ps = kStart0Ps + kDur0Ps;
  constexpr uint64_t kDur1Ps = 721597984164ULL;

  const std::vector<TraceEvent> events = {
      {.name = "A", .ts = kStart0Ps / 1e6, .dur = kDur0Ps / 1e6},
      {.name = "B", .ts = kStart1Ps / 1e6, .dur = kDur1Ps / 1e6},
  };
  // Verify that independent `ps / 1e6` conversions round `ts + dur` 1 ULP past
  // the next event's `ts`.
  ASSERT_GT(events[0].ts + events[0].dur, events[1].ts);

  auto events_ptr = GetEventPointers(events);
  const auto packed = PackTraceEvents(events_ptr);

  ASSERT_EQ(packed.size(), 2);
  EXPECT_EQ(packed[0].event->name, "A");
  EXPECT_EQ(packed[0].level, 0);
  EXPECT_EQ(packed[1].event->name, "B");
  EXPECT_EQ(packed[1].level, 0);
}

TEST(TraceEventPackerTest, OnePicosecondOverlapMovesToNextLevel) {
  constexpr uint64_t kStart0Ps = 14043458000ULL;
  constexpr uint64_t kDur0Ps = 22113441837ULL;
  constexpr uint64_t kStart1Ps = 36156899836ULL;
  constexpr uint64_t kDur1Ps = 1000000ULL;
  static_assert(kStart0Ps + kDur0Ps == kStart1Ps + 1);

  const std::vector<TraceEvent> events = {
      {.name = "A", .ts = kStart0Ps / 1e6, .dur = kDur0Ps / 1e6},
      {.name = "B", .ts = kStart1Ps / 1e6, .dur = kDur1Ps / 1e6},
  };
  auto events_ptr = GetEventPointers(events);
  const auto packed = PackTraceEvents(events_ptr);

  ASSERT_EQ(packed.size(), 2);
  EXPECT_EQ(packed[0].event->name, "A");
  EXPECT_EQ(packed[0].level, 0);
  EXPECT_EQ(packed[1].event->name, "B");
  EXPECT_EQ(packed[1].level, 1);
}

}  // namespace
}  // namespace traceviewer
