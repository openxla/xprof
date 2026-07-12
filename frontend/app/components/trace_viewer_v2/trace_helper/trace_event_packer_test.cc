#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_event_packer.h"

#include <string>
#include <vector>

#include "<gtest/gtest.h>"
#include "absl/types/span.h"
#include "frontend/app/components/trace_viewer_v2/trace_helper/overlap_interval_test_fixtures.h"
#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_event.h"

namespace traceviewer {
namespace {

using test_fixtures::ExclusiveEndpointScenario;
using test_fixtures::ExpectedLevels;
using test_fixtures::ExpectedNames;
using test_fixtures::MakePackerEvents;
using test_fixtures::NestedOverlapScenario;
using test_fixtures::OverlapRowReuseScenario;
using test_fixtures::OverlapScenarioEvent;
using test_fixtures::PartialOverlapScenario;

std::vector<const TraceEvent*> GetEventPointers(
    absl::Span<const TraceEvent> events) {
  std::vector<const TraceEvent*> events_ptr;
  events_ptr.reserve(events.size());
  for (const TraceEvent& event : events) {
    events_ptr.push_back(&event);
  }
  return events_ptr;
}

// Asserts packer preserves every input event and assigns the fixture's
// expected visual levels. Shared with data_provider self-time tests via
// overlap_interval_test_fixtures.h.
void ExpectPackedMatchesScenario(
    const std::vector<OverlapScenarioEvent>& scenario) {
  const std::vector<TraceEvent> events = MakePackerEvents(scenario);
  auto events_ptr = GetEventPointers(events);
  const auto packed = PackTraceEvents(events_ptr);

  // No events may be dropped or merged by packing.
  ASSERT_EQ(packed.size(), scenario.size());

  const std::vector<std::string> expected_names = ExpectedNames(scenario);
  const std::vector<int> expected_levels = ExpectedLevels(scenario);
  for (size_t i = 0; i < packed.size(); ++i) {
    EXPECT_EQ(packed[i].event->name, expected_names[i]) << "index " << i;
    EXPECT_EQ(packed[i].level, expected_levels[i]) << "index " << i;
  }
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

// --- Shared overlap fixtures (joint with data_provider self-time tests) ---

TEST(TraceEventPackerTest, SharedFixtureNestedOverlap) {
  ExpectPackedMatchesScenario(NestedOverlapScenario());
}

TEST(TraceEventPackerTest, SharedFixturePartialOverlap) {
  ExpectPackedMatchesScenario(PartialOverlapScenario());
}

TEST(TraceEventPackerTest, SharedFixtureExclusiveEndpoint) {
  // Exclusive end: B starts at A's end and must reuse level 0.
  ExpectPackedMatchesScenario(ExclusiveEndpointScenario());
}

TEST(TraceEventPackerTest, SharedFixtureOverlapRowReuse) {
  ExpectPackedMatchesScenario(OverlapRowReuseScenario());
}

}  // namespace
}  // namespace traceviewer
