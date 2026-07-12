#ifndef THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TRACE_HELPER_OVERLAP_INTERVAL_TEST_FIXTURES_H_
#define THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TRACE_HELPER_OVERLAP_INTERVAL_TEST_FIXTURES_H_

#include <string>
#include <vector>

#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_event.h"

namespace traceviewer {
namespace test_fixtures {

// Shared overlapping-interval scenarios used by both the packer
// (trace_event_packer_test) and self-time / data-provider paths
// (data_provider_test). Keeping one fixture source prevents packing row
// assignment and hierarchy-aware self-time from silently diverging
// (see openxla/xprof#2939, #2919).
//
// Interval model matches PackTraceEvents:
//   Events occupy half-open ranges [ts, ts+dur). An event that starts at
//   exactly another event's end time does not overlap it (exclusive end).

// One event in a shared scenario plus the expected packer level and
// hierarchy-aware self-time after packing + tree construction.
struct OverlapScenarioEvent {
  const char* name;
  Microseconds ts;
  Microseconds dur;
  // 0-based visual level assigned by PackTraceEvents within the track.
  int expected_level;
  // Self-time after BuildTree subtraction of fully-contained children.
  Microseconds expected_self_time;
};

// Nested parent/child chain:
//
//   |---------- Parent [100,200) ----------|
//      |---- Child [110,160) ----|
//         | Grand [120,140) |
//
// Pack levels: Parent=0, Child=1, Grand=2
// Self times:  Parent=50 (100-50), Child=30 (50-20), Grand=20
// Invariant: sum(self_times) == root wall time for pure nesting.
inline std::vector<OverlapScenarioEvent> NestedOverlapScenario() {
  return {
      {"Parent", 100.0, 100.0, /*expected_level=*/0, /*expected_self_time=*/50.0},
      {"Child", 110.0, 50.0, /*expected_level=*/1, /*expected_self_time=*/30.0},
      {"Grand", 120.0, 20.0, /*expected_level=*/2, /*expected_self_time=*/20.0},
  };
}

// Non-nested partial overlap: B starts inside A but ends after A.
// The tree does not treat B as a child of A; both keep full duration as
// self-time. The packer still stacks them on different rows.
//
//   |----- A [100,150) -----|
//          |----- B [120,170) -----|
//
// Pack levels: A=0, B=1
// Self times:  A=50, B=50
inline std::vector<OverlapScenarioEvent> PartialOverlapScenario() {
  return {
      {"A", 100.0, 50.0, /*expected_level=*/0, /*expected_self_time=*/50.0},
      {"B", 120.0, 50.0, /*expected_level=*/1, /*expected_self_time=*/50.0},
  };
}

// Endpoint adjacency under the exclusive-end policy: B starts exactly when
// A ends, so they must share a packer row and keep independent self-times.
//
//   |-- A [10,30) --||-- B [30,50) --|
//
// Pack levels: A=0, B=0
// Self times:  A=20, B=20
inline std::vector<OverlapScenarioEvent> ExclusiveEndpointScenario() {
  return {
      {"A", 10.0, 20.0, /*expected_level=*/0, /*expected_self_time=*/20.0},
      {"B", 30.0, 20.0, /*expected_level=*/0, /*expected_self_time=*/20.0},
  };
}

// Overlap with later row reuse (classic packer case):
//
//   |-- A [10,30) --|
//          |-- B [20,40) --|
//                     |-- C [35,50) --|  reuses level 0 after A ends
//
// Pack levels: A=0, B=1, C=0
// Self times:  A=20, B=20, C=15 (none nested)
inline std::vector<OverlapScenarioEvent> OverlapRowReuseScenario() {
  return {
      {"A", 10.0, 20.0, /*expected_level=*/0, /*expected_self_time=*/20.0},
      {"B", 20.0, 20.0, /*expected_level=*/1, /*expected_self_time=*/20.0},
      {"C", 35.0, 15.0, /*expected_level=*/0, /*expected_self_time=*/15.0},
  };
}

// Minimal TraceEvent list for packer unit tests (name/ts/dur only).
inline std::vector<TraceEvent> MakePackerEvents(
    const std::vector<OverlapScenarioEvent>& scenario) {
  std::vector<TraceEvent> events;
  events.reserve(scenario.size());
  for (const auto& e : scenario) {
    events.push_back(TraceEvent{.name = e.name, .ts = e.ts, .dur = e.dur});
  }
  return events;
}

// Complete-phase TraceEvent list for DataProvider / self-time integration.
inline std::vector<TraceEvent> MakeDataProviderEvents(
    const std::vector<OverlapScenarioEvent>& scenario, ProcessId pid = 1,
    ThreadId tid = 101) {
  std::vector<TraceEvent> events;
  events.reserve(scenario.size());
  for (const auto& e : scenario) {
    events.push_back(TraceEvent{.ph = Phase::kComplete,
                                .pid = pid,
                                .tid = tid,
                                .name = e.name,
                                .ts = e.ts,
                                .dur = e.dur});
  }
  return events;
}

// Expected packer levels in chronological (packer output) order.
inline std::vector<int> ExpectedLevels(
    const std::vector<OverlapScenarioEvent>& scenario) {
  std::vector<int> levels;
  levels.reserve(scenario.size());
  for (const auto& e : scenario) {
    levels.push_back(e.expected_level);
  }
  return levels;
}

// Expected self-times in chronological (packer output) order.
inline std::vector<Microseconds> ExpectedSelfTimes(
    const std::vector<OverlapScenarioEvent>& scenario) {
  std::vector<Microseconds> self_times;
  self_times.reserve(scenario.size());
  for (const auto& e : scenario) {
    self_times.push_back(e.expected_self_time);
  }
  return self_times;
}

// Expected names in chronological (packer output) order.
inline std::vector<std::string> ExpectedNames(
    const std::vector<OverlapScenarioEvent>& scenario) {
  std::vector<std::string> names;
  names.reserve(scenario.size());
  for (const auto& e : scenario) {
    names.push_back(e.name);
  }
  return names;
}

}  // namespace test_fixtures
}  // namespace traceviewer

#endif  // THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TRACE_HELPER_OVERLAP_INTERVAL_TEST_FIXTURES_H_
