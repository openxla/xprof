#include "frontend/app/components/trace_viewer_v2/timeline/data_provider.h"

#include <algorithm>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "testing/base/public/gmock.h"
#include "<gtest/gtest.h>"
#include "absl/algorithm/container.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "imgui.h"
#include "tsl/profiler/lib/context_types.h"
#include "frontend/app/components/trace_viewer_v2/color/colors.h"
#include "frontend/app/components/trace_viewer_v2/timeline/time_range.h"
#include "frontend/app/components/trace_viewer_v2/timeline/timeline.h"
#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_event.h"

namespace traceviewer {

namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::SizeIs;
using ::testing::UnorderedElementsAre;

class DataProviderTest : public ::testing::Test {
 public:
  DataProviderTest() = default;

 protected:
  TraceEvent CreateMetadataEvent(std::string event_name, ProcessId pid,
                                 ThreadId tid,
                                 std::string thread_or_process_name) {
    return {.ph = Phase::kMetadata,
            .pid = pid,
            .tid = tid,
            .name = std::move(event_name),
            .ts = 0.0,
            .dur = 0.0,
            .id = "",
            .args = {{std::string(kName), std::move(thread_or_process_name)}}};
  }

  CounterEvent CreateCounterEvent(ProcessId pid, std::string name,
                                  std::vector<double> timestamps,
                                  std::vector<double> values) {
    CounterEvent event;
    event.pid = pid;
    event.name = std::move(name);
    event.timestamps = timestamps;
    event.values = values;
    for (double val : values) {
      event.min_value = std::min(event.min_value, val);
      event.max_value = std::max(event.max_value, val);
    }
    return event;
  }

  Timeline timeline_;
  DataProvider data_provider_;
};

TEST_F(DataProviderTest, ProcessEmptyTraceData) {
  const std::vector<TraceEvent> events;

  data_provider_.ProcessTraceEvents({events, {}}, timeline_);

  EXPECT_THAT(timeline_.timeline_data().groups, IsEmpty());
  EXPECT_THAT(timeline_.timeline_data().entry_start_times, IsEmpty());
}

TEST_F(DataProviderTest, ProcessMetadataEvents) {
  const std::vector<TraceEvent> events = {
      CreateMetadataEvent(std::string(kThreadName), 1, 101, "Thread_A"),
      CreateMetadataEvent(std::string(kProcessName), 1, 0, "Process_1")};

  data_provider_.ProcessTraceEvents({events, {}}, timeline_);

  // Metadata alone doesn't create entries in timeline_data
  EXPECT_THAT(timeline_.timeline_data().groups, IsEmpty());
}

TEST_F(DataProviderTest, ProcessMetadataEventsWithEmptyName) {
  const std::vector<TraceEvent> events = {
      CreateMetadataEvent(std::string(kProcessName), 1, 0, ""),
      CreateMetadataEvent(std::string(kThreadName), 1, 101, ""),
      {.ph = Phase::kComplete,
       .pid = 1,
       .tid = 101,
       .name = "Task A",
       .ts = 5000.0,
       .dur = 1000.0,
       .id = "",
       .args = {}},
  };

  data_provider_.ProcessTraceEvents({events, {}}, timeline_);

  const FlameChartTimelineData& data = timeline_.timeline_data();

  ASSERT_THAT(data.groups, SizeIs(2));

  EXPECT_EQ(data.groups[0].name, "Process_1");
  EXPECT_EQ(data.groups[0].nesting_level, 0);
  EXPECT_EQ(data.groups[1].name, "Thread_101");
  EXPECT_EQ(data.groups[1].nesting_level, 1);
}

TEST_F(DataProviderTest, ProcessMetadataEventsWithNoNameArg) {
  const std::vector<TraceEvent> events = {
      {.ph = Phase::kMetadata,
       .pid = 1,
       .tid = 0,
       .name = std::string(kProcessName),
       .ts = 0.0,
       .dur = 0.0,
       .id = "",
       .args = {}},
      {.ph = Phase::kMetadata,
       .pid = 1,
       .tid = 101,
       .name = std::string(kThreadName),
       .ts = 0.0,
       .dur = 0.0,
       .id = "",
       .args = {}},
      {.ph = Phase::kComplete,
       .pid = 1,
       .tid = 101,
       .name = "Task A",
       .ts = 5000.0,
       .dur = 1000.0,
       .id = "",
       .args = {}},
  };

  data_provider_.ProcessTraceEvents({events, {}}, timeline_);

  const FlameChartTimelineData& data = timeline_.timeline_data();

  ASSERT_THAT(data.groups, SizeIs(2));

  EXPECT_EQ(data.groups[0].name, "Process_1");
  EXPECT_EQ(data.groups[0].nesting_level, 0);
  EXPECT_EQ(data.groups[1].name, "Thread_101");
  EXPECT_EQ(data.groups[1].nesting_level, 1);
}

TEST_F(DataProviderTest, ProcessCompleteEvents) {
  const std::vector<TraceEvent> events = {{.ph = Phase::kComplete,
                                           .pid = 1,
                                           .tid = 101,
                                           .name = "Event 1",
                                           .ts = 1000.0,
                                           .dur = 200.0,
                                           .id = "",
                                           .args = {}},
                                          {.ph = Phase::kComplete,
                                           .pid = 1,
                                           .tid = 102,
                                           .name = "Event 2",
                                           .ts = 1100.0,
                                           .dur = 300.0,
                                           .id = "",
                                           .args = {}}};

  data_provider_.ProcessTraceEvents({events, {}}, timeline_);

  const FlameChartTimelineData& data = timeline_.timeline_data();

  ASSERT_THAT(data.groups, SizeIs(3));

  EXPECT_EQ(data.groups[0].name, "Process_1");
  EXPECT_EQ(data.groups[0].start_level, 0);
  EXPECT_EQ(data.groups[0].nesting_level, 0);
  EXPECT_EQ(data.groups[1].name, "Thread_101");
  EXPECT_EQ(data.groups[1].start_level, 0);
  EXPECT_EQ(data.groups[1].nesting_level, 1);
  EXPECT_EQ(data.groups[2].name, "Thread_102");
  EXPECT_EQ(data.groups[2].start_level, 1);
  EXPECT_EQ(data.groups[2].nesting_level, 1);

  EXPECT_THAT(data.entry_start_times, ElementsAre(1000.0, 1100.0));
  EXPECT_THAT(data.entry_total_times, ElementsAre(200.0, 300.0));
  EXPECT_THAT(data.entry_levels, ElementsAre(0, 1));
  EXPECT_THAT(data.entry_names, ElementsAre("Event 1", "Event 2"));

  ASSERT_THAT(data.events_by_level, SizeIs(2));

  EXPECT_THAT(data.events_by_level[0], ElementsAre(0));
  EXPECT_THAT(data.events_by_level[1], ElementsAre(1));

  EXPECT_DOUBLE_EQ(timeline_.visible_range().start(), 1000.0);
  EXPECT_DOUBLE_EQ(timeline_.visible_range().end(), 1400.0);
}

TEST_F(DataProviderTest, ProcessNestedCompleteEvents) {
  const std::vector<TraceEvent> events = {{.ph = Phase::kComplete,
                                           .pid = 1,
                                           .tid = 101,
                                           .name = "Event A",
                                           .ts = 100.0,
                                           .dur = 100.0},
                                          {.ph = Phase::kComplete,
                                           .pid = 1,
                                           .tid = 101,
                                           .name = "Event B",
                                           .ts = 110.0,
                                           .dur = 50.0},
                                          {.ph = Phase::kComplete,
                                           .pid = 1,
                                           .tid = 101,
                                           .name = "Event C",
                                           .ts = 120.0,
                                           .dur = 20.0}};

  data_provider_.ProcessTraceEvents({events, {}}, timeline_);

  const FlameChartTimelineData& data = timeline_.timeline_data();

  ASSERT_THAT(data.groups, SizeIs(2));

  EXPECT_EQ(data.groups[0].name, "Process_1");
  EXPECT_EQ(data.groups[0].start_level, 0);
  EXPECT_EQ(data.groups[0].nesting_level, 0);
  EXPECT_EQ(data.groups[1].name, "Thread_101");
  EXPECT_EQ(data.groups[1].start_level, 0);
  EXPECT_EQ(data.groups[1].nesting_level, 1);

  EXPECT_THAT(data.entry_start_times, ElementsAre(100.0, 110.0, 120.0));
  EXPECT_THAT(data.entry_total_times, ElementsAre(100.0, 50.0, 20.0));
  EXPECT_THAT(data.entry_levels, ElementsAre(0, 1, 2));
  EXPECT_THAT(data.entry_names, ElementsAre("Event A", "Event B", "Event C"));

  ASSERT_THAT(data.events_by_level, SizeIs(3));

  EXPECT_THAT(data.events_by_level[0], ElementsAre(0));
  EXPECT_THAT(data.events_by_level[1], ElementsAre(1));
  EXPECT_THAT(data.events_by_level[2], ElementsAre(2));

  EXPECT_DOUBLE_EQ(timeline_.visible_range().start(), 100.0);
  EXPECT_DOUBLE_EQ(timeline_.visible_range().end(), 200.0);
}

TEST_F(DataProviderTest, ProcessNonOverlappingCompleteEventsSameThread) {
  const std::vector<TraceEvent> events = {{.ph = Phase::kComplete,
                                           .pid = 1,
                                           .tid = 101,
                                           .name = "Event A",
                                           .ts = 100.0,
                                           .dur = 50.0},
                                          {.ph = Phase::kComplete,
                                           .pid = 1,
                                           .tid = 101,
                                           .name = "Event B",
                                           .ts = 160.0,
                                           .dur = 50.0}};

  data_provider_.ProcessTraceEvents({events, {}}, timeline_);

  const FlameChartTimelineData& data = timeline_.timeline_data();

  ASSERT_THAT(data.groups, SizeIs(2));  // Process and Thread

  EXPECT_EQ(data.groups[1].name, "Thread_101");
  EXPECT_EQ(data.groups[1].start_level, 0);

  EXPECT_THAT(data.entry_start_times, ElementsAre(100.0, 160.0));
  EXPECT_THAT(data.entry_total_times, ElementsAre(50.0, 50.0));
  // Both should be on level 0 (row reuse)
  EXPECT_THAT(data.entry_levels, ElementsAre(0, 0));
  EXPECT_THAT(data.entry_names, ElementsAre("Event A", "Event B"));
}

TEST_F(DataProviderTest, PopulateThreadTrackWithPackedLayoutSorting) {
  const std::vector<TraceEvent> events = {{.ph = Phase::kComplete,
                                           .pid = 1,
                                           .tid = 101,
                                           .name = "AsyncEvent",
                                           .ts = 150.0,
                                           .dur = 50.0,
                                           .is_async = true},
                                          {.ph = Phase::kComplete,
                                           .pid = 1,
                                           .tid = 101,
                                           .name = "AsyncEvent",
                                           .ts = 100.0,
                                           .dur = 50.0,
                                           .is_async = true}};

  data_provider_.ProcessTraceEvents({events, {}}, timeline_);

  const FlameChartTimelineData& data = timeline_.timeline_data();

  // Verify they are sorted by timestamp and packed onto the same row.
  EXPECT_THAT(data.entry_start_times, ElementsAre(100.0, 150.0));
  EXPECT_THAT(data.entry_levels, ElementsAre(0, 0));
}

TEST_F(DataProviderTest, TimeRangeCoversDuration) {
  const std::vector<TraceEvent> events = {{.ph = Phase::kComplete,
                                           .pid = 1,
                                           .tid = 101,
                                           .name = "Event 1",
                                           .ts = 100.0,
                                           .dur = 100.0},
                                          {.ph = Phase::kComplete,
                                           .pid = 1,
                                           .tid = 101,
                                           .name = "Event 2",
                                           .ts = 150.0,
                                           .dur = 10.0}};
  data_provider_.ProcessTraceEvents({events, {}}, timeline_);
  EXPECT_DOUBLE_EQ(timeline_.visible_range().start(), 100.0);
  EXPECT_DOUBLE_EQ(timeline_.visible_range().end(), 200.0);
}

TEST_F(DataProviderTest, ProcessMixedEvents) {
  const std::vector<TraceEvent> events = {
      CreateMetadataEvent(std::string(kProcessName), 1, 0, "Main_Process"),
      CreateMetadataEvent(std::string(kThreadName), 1, 101, "Worker Thread"),
      {.ph = Phase::kComplete,
       .pid = 1,
       .tid = 101,
       .name = "Task A",
       .ts = 5000.0,
       .dur = 1000.0,
       .id = "",
       .args = {}},
      // No metadata for tid 102, uses default "Thread_102".
      {.ph = Phase::kComplete,
       .pid = 1,
       .tid = 102,
       .name = "Task B",
       .ts = 5500.0,
       .dur = 1500.0,
       .id = "",
       .args = {}},
      // No metadata for pid 2, uses default "Process_2".
      {.ph = Phase::kComplete,
       .pid = 2,
       .tid = 201,
       .name = "Task C",
       .ts = 6000.0,
       .dur = 500.0,
       .id = "",
       .args = {}}};

  data_provider_.ProcessTraceEvents({events, {}}, timeline_);

  const FlameChartTimelineData& data = timeline_.timeline_data();

  ASSERT_THAT(data.groups, SizeIs(5));

  EXPECT_EQ(data.groups[0].name, "Main_Process");
  EXPECT_EQ(data.groups[0].nesting_level, 0);
  EXPECT_EQ(data.groups[1].name, "Worker Thread");
  EXPECT_EQ(data.groups[1].nesting_level, 1);
  EXPECT_EQ(data.groups[2].name, "Thread_102");
  EXPECT_EQ(data.groups[2].nesting_level, 1);
  EXPECT_EQ(data.groups[3].name, "Process_2");
  EXPECT_EQ(data.groups[3].nesting_level, 0);
  EXPECT_EQ(data.groups[4].name, "Thread_201");
  EXPECT_EQ(data.groups[4].nesting_level, 1);

  EXPECT_THAT(data.entry_start_times, ElementsAre(5000.0, 5500.0, 6000.0));
  EXPECT_THAT(data.entry_total_times, ElementsAre(1000.0, 1500.0, 500.0));
  EXPECT_THAT(data.entry_levels, ElementsAre(0, 1, 2));
  EXPECT_THAT(data.entry_names, ElementsAre("Task A", "Task B", "Task C"));

  EXPECT_DOUBLE_EQ(timeline_.visible_range().start(), 5000.0);
  EXPECT_DOUBLE_EQ(timeline_.visible_range().end(), 7000.0);
}

TEST_F(DataProviderTest, ProcessMultipleProcesses) {
  const std::vector<TraceEvent> events = {
      CreateMetadataEvent(std::string(kProcessName), 1, 0, "Process_A"),
      CreateMetadataEvent(std::string(kThreadName), 1, 101, "Thread_A1"),
      {.ph = Phase::kComplete,
       .pid = 1,
       .tid = 101,
       .name = "Event A1",
       .ts = 1000.0,
       .dur = 100.0,
       .id = "",
       .args = {}},
      CreateMetadataEvent(std::string(kProcessName), 2, 0, "Process_B"),
      CreateMetadataEvent(std::string(kThreadName), 2, 201, "Thread_B1"),
      {.ph = Phase::kComplete,
       .pid = 2,
       .tid = 201,
       .name = "Event B1",
       .ts = 1200.0,
       .dur = 100.0,
       .id = "",
       .args = {}},
      CreateMetadataEvent(std::string(kThreadName), 1, 102, "Thread_A2"),
      {.ph = Phase::kComplete,
       .pid = 1,
       .tid = 102,
       .name = "Event A2",
       .ts = 1100.0,
       .dur = 100.0,
       .id = "",
       .args = {}},
  };

  data_provider_.ProcessTraceEvents({events, {}}, timeline_);

  const FlameChartTimelineData& data = timeline_.timeline_data();
  ASSERT_THAT(data.groups, SizeIs(5));

  // Process A
  EXPECT_EQ(data.groups[0].name, "Process_A");
  EXPECT_EQ(data.groups[0].nesting_level, 0);
  EXPECT_EQ(data.groups[1].name, "Thread_A1");
  EXPECT_EQ(data.groups[1].nesting_level, 1);
  EXPECT_EQ(data.groups[1].start_level, 0);
  EXPECT_EQ(data.groups[2].name, "Thread_A2");
  EXPECT_EQ(data.groups[2].nesting_level, 1);
  EXPECT_EQ(data.groups[2].start_level, 1);

  // Process B
  EXPECT_EQ(data.groups[3].name, "Process_B");
  EXPECT_EQ(data.groups[3].nesting_level, 0);
  EXPECT_EQ(data.groups[4].name, "Thread_B1");
  EXPECT_EQ(data.groups[4].nesting_level, 1);
  EXPECT_EQ(data.groups[4].start_level, 2);

  EXPECT_THAT(data.entry_levels, ElementsAre(0, 1, 2));
  EXPECT_THAT(data.entry_start_times, ElementsAre(1000.0, 1100.0, 1200.0));
}

TEST_F(DataProviderTest, ProcessSingleCounterEvent) {
  const std::vector<CounterEvent> events = {
      CreateCounterEvent(1, "Counter A", {10.0, 20.0, 30.0}, {1.0, 5.0, 2.0})};

  data_provider_.ProcessTraceEvents({{}, events}, timeline_);

  const FlameChartTimelineData& data = timeline_.timeline_data();

  ASSERT_THAT(data.groups, SizeIs(2));  // Process group + Counter group

  EXPECT_EQ(data.groups[0].name, "Process_1");
  EXPECT_EQ(data.groups[0].nesting_level, 0);

  EXPECT_EQ(data.groups[1].name, "Counter A");
  EXPECT_EQ(data.groups[1].type, Group::Type::kCounter);
  EXPECT_EQ(data.groups[1].nesting_level, 1);
  EXPECT_TRUE(data.groups[1].expanded);

  ASSERT_TRUE(data.counter_data_by_group_index.count(1));

  const CounterData& counter_data = data.counter_data_by_group_index.at(1);

  EXPECT_THAT(counter_data.timestamps, ElementsAre(10.0, 20.0, 30.0));
  EXPECT_THAT(counter_data.values, ElementsAre(1.0, 5.0, 2.0));
  EXPECT_DOUBLE_EQ(counter_data.min_value, 1.0);
  EXPECT_DOUBLE_EQ(counter_data.max_value, 5.0);

  EXPECT_DOUBLE_EQ(timeline_.visible_range().start(), 10.0);
  EXPECT_DOUBLE_EQ(timeline_.visible_range().end(), 30.0);
}

TEST_F(DataProviderTest, ProcessCounterEventWithNegativeValues) {
  const std::vector<CounterEvent> events = {CreateCounterEvent(
      1, "Counter A", {10.0, 20.0, 30.0}, {-1.0, -5.0, -2.0})};

  data_provider_.ProcessTraceEvents({{}, events}, timeline_);

  const FlameChartTimelineData& data = timeline_.timeline_data();
  ASSERT_TRUE(data.counter_data_by_group_index.count(1));
  const CounterData& counter_data = data.counter_data_by_group_index.at(1);

  EXPECT_DOUBLE_EQ(counter_data.min_value, -5.0);
  EXPECT_DOUBLE_EQ(counter_data.max_value, -1.0);
}

TEST_F(DataProviderTest, ProcessCounterEventWithSingleValue) {
  const std::vector<CounterEvent> events = {
      CreateCounterEvent(1, "Counter A", {10.0}, {42.0})};

  data_provider_.ProcessTraceEvents({{}, events}, timeline_);

  const FlameChartTimelineData& data = timeline_.timeline_data();
  ASSERT_TRUE(data.counter_data_by_group_index.count(1));
  const CounterData& counter_data = data.counter_data_by_group_index.at(1);

  EXPECT_DOUBLE_EQ(counter_data.min_value, 42.0);
  EXPECT_DOUBLE_EQ(counter_data.max_value, 42.0);
}

TEST_F(DataProviderTest, ProcessCounterEventWithEmptyValues) {
  const std::vector<CounterEvent> events = {
      CreateCounterEvent(1, "Counter A", {}, {})};

  data_provider_.ProcessTraceEvents({{}, events}, timeline_);

  const FlameChartTimelineData& data = timeline_.timeline_data();
  ASSERT_EQ(data.counter_data_by_group_index.size(), 1);
  ASSERT_TRUE(data.counter_data_by_group_index.count(1));
  const CounterData& counter_data = data.counter_data_by_group_index.at(1);
  EXPECT_TRUE(counter_data.timestamps.empty());
  EXPECT_TRUE(counter_data.values.empty());
}

TEST_F(DataProviderTest, ProcessMultipleCounterEventsSorted) {
  CounterEvent event1 =
      CreateCounterEvent(1, "Counter A", {100.0, 110.0}, {10.0, 11.0});

  CounterEvent event2 =
      CreateCounterEvent(1, "Counter A", {50.0, 60.0}, {5.0, 6.0});

  data_provider_.ProcessTraceEvents({{{.ph = Phase::kComplete,
                                       .pid = 1,
                                       .tid = 1,
                                       .name = "Complete Event",
                                       .ts = 0.0,
                                       .dur = 10.0}},
                                     {event1, event2}},
                                    timeline_);

  const FlameChartTimelineData& data = timeline_.timeline_data();

  ASSERT_TRUE(data.counter_data_by_group_index.count(2));

  const CounterData& counter_data = data.counter_data_by_group_index.at(2);

  EXPECT_THAT(counter_data.timestamps, ElementsAre(50.0, 60.0, 100.0, 110.0));
  EXPECT_THAT(counter_data.values, ElementsAre(5.0, 6.0, 10.0, 11.0));
}

TEST_F(DataProviderTest, ProcessCounterEventAndCompleteEvent) {
  CounterEvent counter_event =
      CreateCounterEvent(1, "Counter A", {10.0, 20.0, 30.0}, {1.0, 5.0, 2.0});

  data_provider_.ProcessTraceEvents({{{.ph = Phase::kComplete,
                                       .pid = 1,
                                       .tid = 1,
                                       .name = "Complete Event",
                                       .ts = 0.0,
                                       .dur = 10.0}},
                                     {counter_event}},
                                    timeline_);

  const FlameChartTimelineData& data = timeline_.timeline_data();

  ASSERT_THAT(data.groups,
              SizeIs(3));  // Process group + Thread group + Counter group

  EXPECT_EQ(data.groups[0].name, "Process_1");
  EXPECT_EQ(data.groups[0].nesting_level, 0);

  EXPECT_EQ(data.groups[1].name, "Thread_1");
  EXPECT_EQ(data.groups[1].nesting_level, 1);

  EXPECT_EQ(data.groups[2].name, "Counter A");
  EXPECT_EQ(data.groups[2].type, Group::Type::kCounter);
  EXPECT_EQ(data.groups[2].nesting_level, 1);

  ASSERT_TRUE(data.counter_data_by_group_index.count(2));

  const CounterData& counter_data = data.counter_data_by_group_index.at(2);

  EXPECT_THAT(counter_data.timestamps, ElementsAre(10.0, 20.0, 30.0));
  EXPECT_THAT(counter_data.values, ElementsAre(1.0, 5.0, 2.0));
  EXPECT_DOUBLE_EQ(counter_data.min_value, 1.0);
  EXPECT_DOUBLE_EQ(counter_data.max_value, 5.0);

  EXPECT_DOUBLE_EQ(timeline_.visible_range().start(), 0.0);
  EXPECT_DOUBLE_EQ(timeline_.visible_range().end(), 30.0);
}

TEST_F(DataProviderTest, ProcessCounterEventAndCompleteEventInDifferentPid) {
  CounterEvent counter_event =
      CreateCounterEvent(1, "Counter A", {10.0, 20.0, 30.0}, {1.0, 5.0, 2.0});

  data_provider_.ProcessTraceEvents({{{.ph = Phase::kComplete,
                                       .pid = 2,
                                       .tid = 1,
                                       .name = "Complete Event",
                                       .ts = 0.0,
                                       .dur = 10.0}},
                                     {counter_event}},
                                    timeline_);

  const FlameChartTimelineData& data = timeline_.timeline_data();

  ASSERT_THAT(data.groups,
              SizeIs(4));  // Process 1, Counter A, Process 2, Thread 1

  EXPECT_EQ(data.groups[0].name, "Process_1");
  EXPECT_EQ(data.groups[0].nesting_level, 0);

  EXPECT_EQ(data.groups[1].name, "Counter A");
  EXPECT_EQ(data.groups[1].type, Group::Type::kCounter);
  EXPECT_EQ(data.groups[1].nesting_level, 1);

  EXPECT_EQ(data.groups[2].name, "Process_2");
  EXPECT_EQ(data.groups[2].nesting_level, 0);

  EXPECT_EQ(data.groups[3].name, "Thread_1");
  EXPECT_EQ(data.groups[3].nesting_level, 1);

  ASSERT_TRUE(data.counter_data_by_group_index.count(1));
}

TEST_F(DataProviderTest, CounterTrackIncrementsLevel) {
  // Process 1: Thread 1 (1 level), Counter A
  // Process 2: Thread 2
  TraceEvent t1_event{.ph = Phase::kComplete,
                      .pid = 1,
                      .tid = 1,
                      .name = "Thread1Event",
                      .ts = 0.0,
                      .dur = 10.0};

  CounterEvent counter_event = CreateCounterEvent(1, "CounterA", {0.0}, {0.0});

  TraceEvent t2_event{.ph = Phase::kComplete,
                      .pid = 2,
                      .tid = 2,
                      .name = "Thread2Event",
                      .ts = 0.0,
                      .dur = 10.0};

  data_provider_.ProcessTraceEvents({{t1_event, t2_event}, {counter_event}},
                                    timeline_);

  const FlameChartTimelineData& data = timeline_.timeline_data();

  // Expected Groups:
  // 0: Process 1
  // 1: Thread 1 (pid 1, tid 1). start_level = 0.
  // 2: CounterA (pid 1). start_level = 1.
  // 3: Process 2
  // 4: Thread 2 (pid 2, tid 2). start_level = 2 (IF incremented) OR 1 (IF
  // NOT).

  ASSERT_THAT(data.groups, SizeIs(5));

  EXPECT_EQ(data.groups[1].name, "Thread_1");
  EXPECT_EQ(data.groups[1].start_level, 0);

  EXPECT_EQ(data.groups[2].name, "CounterA");
  EXPECT_EQ(data.groups[2].start_level, 1);

  EXPECT_EQ(data.groups[4].name, "Thread_2");
  EXPECT_EQ(data.groups[4].start_level, 2);
}

TEST_F(DataProviderTest, ProcessCounterEventReservesCapacityCorrectly) {
  // Use a number that is likely to cause capacity mismatch if not reserved.
  // 100 elements.
  // Without reserve: 1 -> 2 -> 4 -> 8 -> 16 -> 32 -> 64 -> 128. Capacity = 128.
  // With reserve(100): Capacity = 100 (typically).
  const int kNumEntries = 100;
  std::vector<double> timestamps;
  std::vector<double> values;
  timestamps.reserve(kNumEntries);
  values.reserve(kNumEntries);
  for (int i = 0; i < kNumEntries; ++i) {
    timestamps.push_back(static_cast<double>(i));
    values.push_back(static_cast<double>(i));
  }
  CounterEvent counter_event = CreateCounterEvent(
      1, "Counter A", std::move(timestamps), std::move(values));

  const std::vector<CounterEvent> events = {counter_event};

  data_provider_.ProcessTraceEvents({{}, events}, timeline_);

  const FlameChartTimelineData& data = timeline_.timeline_data();

  // Group 0 is process, Group 1 is counter.
  ASSERT_TRUE(data.counter_data_by_group_index.count(1));

  const CounterData& counter_data = data.counter_data_by_group_index.at(1);

  EXPECT_THAT(counter_data.timestamps, SizeIs(kNumEntries));

  // Verify that capacity matches size, implying reserve was called with correct
  // size. Without reserve, capacity would likely be the next power of 2 (e.g.,
  // 128 for 100 elements).
  EXPECT_EQ(counter_data.timestamps.capacity(), kNumEntries);
  EXPECT_EQ(counter_data.values.capacity(), kNumEntries);
}

TEST_F(DataProviderTest, ProcessesSortedBySortIndex) {
  const std::vector<TraceEvent> events = {
      CreateMetadataEvent(std::string(kProcessName), 1, 0, "Process_1"),
      {.ph = Phase::kMetadata,
       .pid = 1,
       .tid = 0,
       .name = "process_sort_index",
       .ts = 0.0,
       .dur = 0.0,
       .args = {{"sort_index", "2"}}},
      // Add a complete event for Process 1
      {.ph = Phase::kComplete,
       .pid = 1,
       .tid = 101,
       .name = "Event 1",
       .ts = 0.0,
       .dur = 10.0},
      CreateMetadataEvent(std::string(kProcessName), 2, 0, "Process_2"),
      {.ph = Phase::kMetadata,
       .pid = 2,
       .tid = 0,
       .name = "process_sort_index",
       .ts = 0.0,
       .dur = 0.0,
       .args = {{"sort_index", "1"}}},
      // Add a complete event for Process 2
      {.ph = Phase::kComplete,
       .pid = 2,
       .tid = 201,
       .name = "Event 2",
       .ts = 0.0,
       .dur = 10.0},
      CreateMetadataEvent(std::string(kProcessName), 3, 0, "Process_3"),
      // Process 3 has no sort index, defaults to pid (3)
      // Add a complete event for Process 3
      {.ph = Phase::kComplete,
       .pid = 3,
       .tid = 301,
       .name = "Event 3",
       .ts = 0.0,
       .dur = 10.0},
  };

  data_provider_.ProcessTraceEvents({events, {}}, timeline_);

  const FlameChartTimelineData& data = timeline_.timeline_data();
  // 3 processes, each having 1 thread track -> 6 groups total.
  ASSERT_THAT(data.groups, SizeIs(6));

  // Expected order: Process 2 (index 1), Process 1 (index 2), Process 3 (index
  // 3 / default)
  // Groups for Process 2 are at indices 0 (process) and 1 (thread)
  // Groups for Process 1 are at indices 2 (process) and 3 (thread)
  // Groups for Process 3 are at indices 4 (process) and 5 (thread)
  EXPECT_EQ(data.groups[0].name, "Process_2");
  EXPECT_EQ(data.groups[2].name, "Process_1");
  EXPECT_EQ(data.groups[4].name, "Process_3");
}

TEST_F(DataProviderTest, ProcessesSortedBySortIndexStable) {
  const std::vector<TraceEvent> events = {
      CreateMetadataEvent(std::string(kProcessName), 1, 0, "Process_1"),
      {.ph = Phase::kMetadata,
       .pid = 1,
       .tid = 0,
       .name = "process_sort_index",
       .ts = 0.0,
       .dur = 0.0,
       .args = {{"sort_index", "1"}}},
      // Add a complete event for Process 1
      {.ph = Phase::kComplete,
       .pid = 1,
       .tid = 101,
       .name = "Event 1",
       .ts = 0.0,
       .dur = 10.0},
      CreateMetadataEvent(std::string(kProcessName), 2, 0, "Process_2"),
      {.ph = Phase::kMetadata,
       .pid = 2,
       .tid = 0,
       .name = "process_sort_index",
       .ts = 0.0,
       .dur = 0.0,
       .args = {{"sort_index", "1"}}},
      // Add a complete event for Process 2
      {.ph = Phase::kComplete,
       .pid = 2,
       .tid = 201,
       .name = "Event 2",
       .ts = 0.0,
       .dur = 10.0},
  };

  data_provider_.ProcessTraceEvents({events, {}}, timeline_);

  const FlameChartTimelineData& data = timeline_.timeline_data();
  // 2 processes, each having 1 thread track -> 4 groups total.
  ASSERT_THAT(data.groups, SizeIs(4));

  // Stable sort: Process 1 (pid 1) comes before Process 2 (pid 2) as they have
  // same sort index.
  // Groups for Process 1 are at indices 0 (process) and 1 (thread)
  // Groups for Process 2 are at indices 2 (process) and 3 (thread)
  EXPECT_EQ(data.groups[0].name, "Process_1");
  EXPECT_EQ(data.groups[2].name, "Process_2");
}

TEST_F(DataProviderTest, ProcessesSortedByAsyncPriority) {
  const std::vector<TraceEvent> events = {
      CreateMetadataEvent(std::string(kProcessName), 1, 0, "Host Process"),
      {.ph = Phase::kComplete,
       .pid = 1,
       .tid = 101,
       .name = "Task 1",
       .ts = 0.0,
       .dur = 10.0},
      CreateMetadataEvent(std::string(kProcessName), 2, 0,
                          std::string(kAsyncXlaOps)),
      {.ph = Phase::kComplete,
       .pid = 2,
       .tid = 201,
       .name = "Async 1",
       .ts = 0.0,
       .dur = 10.0},
      CreateMetadataEvent(std::string(kProcessName), 3, 0, "Device DMA"),
      {.ph = Phase::kComplete,
       .pid = 3,
       .tid = 301,
       .name = "DMA 1",
       .ts = 0.0,
       .dur = 10.0},
      CreateMetadataEvent(std::string(kProcessName), 4, 0, "Device Process"),
      {.ph = Phase::kMetadata,
       .pid = 4,
       .tid = 0,
       .name = std::string(kProcessSortIndex),
       .ts = 0.0,
       .dur = 0.0,
       .args = {{std::string(kSortIndex), "0"}}},
      {.ph = Phase::kComplete,
       .pid = 4,
       .tid = 401,
       .name = "Task 2",
       .ts = 0.0,
       .dur = 10.0},
  };

  data_provider_.ProcessTraceEvents({events, {}}, timeline_);

  const FlameChartTimelineData& data = timeline_.timeline_data();
  // 4 processes -> 8 groups (process + thread each)
  ASSERT_THAT(data.groups, SizeIs(8));

  // Expected order:
  // 1. Async XLA Ops (priority 2)
  // 2. Device DMA (priority 1)
  // 3. Device Process (priority 0, but sort_index 0)
  // 4. Host Process (priority 0, fallback pid 1)

  EXPECT_EQ(data.groups[0].name, kAsyncXlaOps);
  EXPECT_EQ(data.groups[2].name, "Device DMA");
  EXPECT_EQ(data.groups[4].name, "Device Process");
  EXPECT_EQ(data.groups[6].name, "Host Process");
}

TEST_F(DataProviderTest, ProcessesSortedByAsyncThreadPriority) {
  const std::vector<TraceEvent> events = {
      CreateMetadataEvent(std::string(kProcessName), 1, 0, "Host Process"),
      {.ph = Phase::kComplete,
       .pid = 1,
       .tid = 101,
       .name = "Task 1",
       .ts = 0.0,
       .dur = 10.0},
      CreateMetadataEvent(std::string(kProcessName), 2, 0, "Async Host"),
      CreateMetadataEvent(std::string(kThreadName), 2, 201,
                          std::string(kAsyncXlaOps)),
      {.ph = Phase::kComplete,
       .pid = 2,
       .tid = 201,
       .name = "Async 1",
       .ts = 0.0,
       .dur = 10.0},
  };

  data_provider_.ProcessTraceEvents({events, {}}, timeline_);

  const FlameChartTimelineData& data = timeline_.timeline_data();
  // 2 processes -> 4 groups
  ASSERT_THAT(data.groups, SizeIs(4));

  // Expected order:
  // 1. Async Host (priority 2 because of thread name)
  // 2. Host Process (priority 0)
  EXPECT_EQ(data.groups[0].name, "Async Host");
  EXPECT_EQ(data.groups[2].name, "Host Process");
}

TEST_F(DataProviderTest, AsyncProcessesGroupedByName) {
  const std::vector<TraceEvent> events = {
      CreateMetadataEvent(std::string(kProcessName), 1, 0, "Generic Process"),
      {.ph = Phase::kComplete,
       .pid = 1,
       .tid = 0,
       .name = "async-copy-1",
       .ts = 10.0,
       .dur = 10.0,
       .is_async = true},
      {.ph = Phase::kComplete,
       .pid = 1,
       .tid = 0,
       .name = "async-copy-2",
       .ts = 20.0,
       .dur = 10.0,
       .is_async = true},
  };

  data_provider_.ProcessTraceEvents({events, {}}, timeline_);

  const FlameChartTimelineData& data = timeline_.timeline_data();
  // 1 process + 2 named tracks = 3 groups
  ASSERT_THAT(data.groups, SizeIs(3));

  EXPECT_EQ(data.groups[0].name, "Generic Process");
  EXPECT_EQ(data.groups[1].name, "async-copy-1");
  EXPECT_EQ(data.groups[2].name, "async-copy-2");
}

TEST_F(DataProviderTest, AsyncProcessesMixedGrouping) {
  const std::vector<TraceEvent> events = {
      CreateMetadataEvent(std::string(kProcessName), 1, 0, "Mixed Process"),
      {.ph = Phase::kComplete,
       .pid = 1,
       .tid = 55,
       .name = "async-op",
       .ts = 10.0,
       .dur = 10.0,
       .is_async = true},
      {.ph = Phase::kComplete,
       .pid = 1,
       .tid = 55,
       .name = "sync-op",
       .ts = 30.0,
       .dur = 10.0,
       .is_async = false},
  };

  data_provider_.ProcessTraceEvents({events, {}}, timeline_);

  const FlameChartTimelineData& data = timeline_.timeline_data();
  // 1 process header + 1 thread track + 1 async track = 3 groups
  ASSERT_THAT(data.groups, SizeIs(3));

  EXPECT_EQ(data.groups[0].name, "Mixed Process");
  EXPECT_EQ(data.groups[1].name, "async-op");
  EXPECT_EQ(data.groups[2].name, "Thread_55");
}

TEST_F(DataProviderTest, AsyncProcessesConcurrentPacking) {
  const std::vector<TraceEvent> events = {
      CreateMetadataEvent(std::string(kProcessName), 1, 0, "Packed Process"),
      {.ph = Phase::kComplete,
       .pid = 1,
       .tid = 55,
       .name = "async-op",
       .ts = 10.0,
       .dur = 20.0,
       .is_async = true},
      {.ph = Phase::kComplete,
       .pid = 1,
       .tid = 55,
       .name = "async-op",
       .ts = 15.0,
       .dur = 10.0,
       .is_async = true},
  };

  data_provider_.ProcessTraceEvents({events, {}}, timeline_);

  const FlameChartTimelineData& data = timeline_.timeline_data();
  ASSERT_THAT(data.groups, SizeIs(2));

  EXPECT_EQ(data.groups[0].name, "Packed Process");
  EXPECT_EQ(data.groups[1].name, "async-op");

  ASSERT_THAT(data.entry_start_times, SizeIs(2));

  int start_level = data.groups[1].start_level;
  EXPECT_EQ(data.entry_levels[0], start_level);
  EXPECT_EQ(data.entry_levels[1], start_level + 1);
}

TEST_F(DataProviderTest, ProcessesSortedWithMalformedAndMissingSortIndex) {
  const std::vector<TraceEvent> events = {
      CreateMetadataEvent(std::string(kProcessName), 1, 0, "Process_1"),
      // Process 1 has malformed sort index, fallback to pid 1.
      {.ph = Phase::kMetadata,
       .pid = 1,
       .tid = 0,
       .name = "process_sort_index",
       .ts = 0.0,
       .dur = 0.0,
       .args = {{"sort_index", "invalid"}}},
      {.ph = Phase::kComplete,
       .pid = 1,
       .tid = 101,
       .name = "Event 1",
       .ts = 0.0,
       .dur = 10.0},
      CreateMetadataEvent(std::string(kProcessName), 2, 0, "Process_2"),
      // Process 2 has sort index 0.
      {.ph = Phase::kMetadata,
       .pid = 2,
       .tid = 0,
       .name = "process_sort_index",
       .ts = 0.0,
       .dur = 0.0,
       .args = {{"sort_index", "0"}}},
      {.ph = Phase::kComplete,
       .pid = 2,
       .tid = 201,
       .name = "Event 2",
       .ts = 0.0,
       .dur = 10.0},
      CreateMetadataEvent(std::string(kProcessName), 3, 0, "Process_3"),
      // Process 3 has process_sort_index event but missing sort_index arg,
      // fallback to pid 3.
      {.ph = Phase::kMetadata,
       .pid = 3,
       .tid = 0,
       .name = "process_sort_index",
       .ts = 0.0,
       .dur = 0.0,
       .args = {}},
      {.ph = Phase::kComplete,
       .pid = 3,
       .tid = 301,
       .name = "Event 3",
       .ts = 0.0,
       .dur = 10.0},
  };

  data_provider_.ProcessTraceEvents({events, {}}, timeline_);

  const FlameChartTimelineData& data = timeline_.timeline_data();
  ASSERT_THAT(data.groups, SizeIs(6));

  // pid 1 -> sort key 1 (fallback)
  // pid 2 -> sort key 0
  // pid 3 -> sort key 3 (fallback)
  // Expected order: 2, 1, 3
  EXPECT_EQ(data.groups[0].name, "Process_2");
  EXPECT_EQ(data.groups[2].name, "Process_1");
  EXPECT_EQ(data.groups[4].name, "Process_3");
}

TEST_F(DataProviderTest, MpmdPipelineViewEnabledPropagated) {
  ParsedTraceEvents events;
  events.mpmd_pipeline_view = true;
  // Add a dummy event to prevent early return
  events.flame_events.push_back({.ph = Phase::kComplete,
                                 .pid = 1,
                                 .tid = 1,
                                 .name = "Event",
                                 .ts = 0.0,
                                 .dur = 10.0});

  data_provider_.ProcessTraceEvents(events, timeline_);

  EXPECT_TRUE(timeline_.mpmd_pipeline_view_enabled());

  events.mpmd_pipeline_view = false;
  // Clear timeline data to process again (or just process again as it
  // overwrites)
  data_provider_.ProcessTraceEvents(events, timeline_);
  EXPECT_FALSE(timeline_.mpmd_pipeline_view_enabled());
}

TEST_F(DataProviderTest, ProcessTraceEventsWithFullTimespan) {
  const std::vector<TraceEvent> events = {{.ph = Phase::kComplete,
                                           .pid = 1,
                                           .tid = 1,
                                           .name = "Event 1",
                                           .ts = 10.0,
                                           .dur = 10.0}};
  ParsedTraceEvents parsed_events;
  parsed_events.flame_events = events;
  // full_timespan is in milliseconds. 0.1ms = 100us.
  parsed_events.full_timespan = std::make_pair(0.0, 0.1);

  data_provider_.ProcessTraceEvents(parsed_events, timeline_);

  // visible_range and fetched_data_time_range should be set to the event's
  // timespan (10.0 to 20.0).
  // Add this sanity check to make sure nothing is broken.
  EXPECT_DOUBLE_EQ(timeline_.visible_range().start(), 10.0);
  EXPECT_DOUBLE_EQ(timeline_.visible_range().end(), 20.0);
  EXPECT_DOUBLE_EQ(timeline_.fetched_data_time_range().start(), 10.0);
  EXPECT_DOUBLE_EQ(timeline_.fetched_data_time_range().end(), 20.0);

  EXPECT_DOUBLE_EQ(timeline_.data_time_range().start(), 0.0);
  EXPECT_DOUBLE_EQ(timeline_.data_time_range().end(), 100.0);
}

TEST_F(DataProviderTest, ProcessTraceEventsWithoutFullTimespan) {
  const std::vector<TraceEvent> events = {{.ph = Phase::kComplete,
                                           .pid = 1,
                                           .tid = 1,
                                           .name = "Event 1",
                                           .ts = 10.0,
                                           .dur = 10.0}};
  ParsedTraceEvents parsed_events;
  parsed_events.flame_events = events;
  // full_timespan is not set

  data_provider_.ProcessTraceEvents(parsed_events, timeline_);

  // visible_range and fetched_data_time_range should be set to the event's
  // timespan (10.0 to 20.0).
  // Add this sanity check to make sure the code before data_time_range
  // calculation is correct.
  EXPECT_DOUBLE_EQ(timeline_.visible_range().start(), 10.0);
  EXPECT_DOUBLE_EQ(timeline_.visible_range().end(), 20.0);
  EXPECT_DOUBLE_EQ(timeline_.fetched_data_time_range().start(), 10.0);
  EXPECT_DOUBLE_EQ(timeline_.fetched_data_time_range().end(), 20.0);

  // data_time_range should fallback to fetched_data_time_range (10.0 to 20.0)
  EXPECT_DOUBLE_EQ(timeline_.data_time_range().start(), 10.0);
  EXPECT_DOUBLE_EQ(timeline_.data_time_range().end(), 20.0);
}

TEST_F(DataProviderTest, ProcessTraceEventsWithVisibleRangeFromUrl) {
  const std::vector<TraceEvent> events = {{.ph = Phase::kComplete,
                                           .pid = 1,
                                           .tid = 1,
                                           .name = "Event 1",
                                           .ts = 10.0,
                                           .dur = 10.0}};
  ParsedTraceEvents parsed_events;
  parsed_events.flame_events = events;
  // Initial visible range in milliseconds. 0.015ms = 15us. 0.018ms = 18us.
  parsed_events.visible_range_from_url = std::make_pair(0.015, 0.018);

  data_provider_.ProcessTraceEvents(parsed_events, timeline_);

  EXPECT_DOUBLE_EQ(timeline_.visible_range().start(), 15.0);
  EXPECT_DOUBLE_EQ(timeline_.visible_range().end(), 18.0);

  // fetched_data_time_range should still be the event's timespan (10.0 to 20.0)
  EXPECT_DOUBLE_EQ(timeline_.fetched_data_time_range().start(), 10.0);
  EXPECT_DOUBLE_EQ(timeline_.fetched_data_time_range().end(), 20.0);
}

TEST_F(DataProviderTest,
       ProcessMultipleCounterEventsReservesCapacityCorrectly) {
  // Use sizes that trigger reallocation if not reserved upfront.
  // 64 is a common power of 2. Adding 1 more should trigger growth if capacity
  // is exactly 64.
  const int kNumEntries1 = 64;
  const int kNumEntries2 = 1;
  const int kTotalEntries = kNumEntries1 + kNumEntries2;

  std::vector<double> timestamps1(kNumEntries1, 0.0);
  std::vector<double> values1(kNumEntries1, 0.0);
  CounterEvent event1 = CreateCounterEvent(
      1, "Counter A", std::move(timestamps1), std::move(values1));

  std::vector<double> timestamps2(kNumEntries2, 0.0);
  std::vector<double> values2(kNumEntries2, 0.0);
  CounterEvent event2 = CreateCounterEvent(
      1, "Counter A", std::move(timestamps2), std::move(values2));

  // The events will be sorted by first timestamp. Since all are 0.0,
  // relative order is preserved or arbitrary. Both have same name/pid so
  // they end up in same track.

  data_provider_.ProcessTraceEvents({{}, {event1, event2}}, timeline_);

  const FlameChartTimelineData& data = timeline_.timeline_data();
  ASSERT_TRUE(data.counter_data_by_group_index.count(1));
  const CounterData& counter_data = data.counter_data_by_group_index.at(1);

  EXPECT_THAT(counter_data.timestamps, SizeIs(kTotalEntries));

  // If reserve(65) is called, capacity should be 65 (or slightly more if
  // implementation rounds up, but typically exact for reserve on empty).
  // If reserve(0) is called:
  // Insert 64 -> Cap 64.
  // Insert 1 -> Realloc -> Cap 128 (usually).
  // So we expect Cap == 65.
  // Note: This test assumes std::vector doubles capacity.
  // To be safe, we can check that capacity is NOT >= 128 if we expect strict
  // reservation. Or better, just check it equals TotalEntries.
  // However, std::vector::reserve(n) might reserve more.
  // But usually it reserves exactly n if vector is empty.
  EXPECT_EQ(counter_data.timestamps.capacity(), kTotalEntries);
  EXPECT_EQ(counter_data.values.capacity(), kTotalEntries);
}

TEST_F(DataProviderTest, ProcessFlowEvents) {
  const std::vector<TraceEvent> all_events = {
      // Process 1, Thread 101
      {.ph = Phase::kComplete,
       .event_id = 10,
       .pid = 1,
       .tid = 101,
       .name = "Task A",
       .ts = 100.0,
       .dur = 50.0},
      {.ph = Phase::kFlowStart,
       .event_id = 1,
       .pid = 1,
       .tid = 101,
       .name = "flow1",
       .ts = 120.0,
       .id = "1",
       .category = tsl::profiler::ContextType::kGpuLaunch},
      // Process 1, Thread 102
      {.ph = Phase::kComplete,
       .event_id = 11,
       .pid = 1,
       .tid = 102,
       .name = "Task B",
       .ts = 200.0,
       .dur = 50.0},
      {.ph = Phase::kFlowStart,
       .event_id = 2,
       .pid = 1,
       .tid = 102,
       .name = "flow1",
       .ts = 210.0,
       .id = "1",
       .category = tsl::profiler::ContextType::kGpuLaunch},
      {.ph = Phase::kFlowEnd,
       .event_id = 3,
       .pid = 1,
       .tid = 102,
       .name = "flow1",
       .ts = 230.0,
       .id = "1",
       .category = tsl::profiler::ContextType::kGpuLaunch},
      // Process 2, Thread 201
      {.ph = Phase::kComplete,
       .event_id = 12,
       .pid = 2,
       .tid = 201,
       .name = "Task C",
       .ts = 300.0,
       .dur = 100.0},
      {.ph = Phase::kFlowStart,
       .event_id = 4,
       .pid = 2,
       .tid = 201,
       .name = "flow2",
       .ts = 310.0,
       .id = "2",
       .category = tsl::profiler::ContextType::kGeneric},
      {.ph = Phase::kFlowEnd,
       .event_id = 5,
       .pid = 2,
       .tid = 201,
       .name = "flow2",
       .ts = 380.0,
       .id = "2",
       .category = tsl::profiler::ContextType::kGeneric},
  };
  std::vector<TraceEvent> flame_events;
  std::vector<TraceEvent> flow_events;
  for (const auto& event : all_events) {
    if (event.ph == Phase::kComplete) {
      flame_events.push_back(event);
    }
    if (!event.id.empty()) {
      flow_events.push_back(event);
    }
  }

  data_provider_.ProcessTraceEvents({flame_events, {}, flow_events}, timeline_);

  FlameChartTimelineData* mutable_data =
      const_cast<FlameChartTimelineData*>(&timeline_.timeline_data());
  absl::c_sort(mutable_data->flow_lines,
               [](const FlowLine& a, const FlowLine& b) {
                 if (a.source_ts != b.source_ts) {
                   return a.source_ts < b.source_ts;
                 }
                 return a.target_ts < b.target_ts;
               });

  const FlameChartTimelineData& data = timeline_.timeline_data();

  // Check categories - list should be sorted.
  EXPECT_THAT(data_provider_.GetFlowCategories(),
              UnorderedElementsAre(
                  static_cast<int>(tsl::profiler::ContextType::kGeneric),
                  static_cast<int>(tsl::profiler::ContextType::kGpuLaunch)));

  // Check flow lines
  // Flow 1: 101 -> 102 (120->210), 102 -> 102 (210->230)
  // Flow 2: 201 -> 201 (310->380)
  ASSERT_THAT(data.flow_lines, SizeIs(3));

  // pid 1 tid 101 is level 0
  // pid 1 tid 102 is level 1
  // pid 2 tid 201 is level 2
  ImU32 flow1_color = kOrange80;
  ImU32 flow2_color = kRed80;
  // Flow 1, part 1
  EXPECT_EQ(data.flow_lines[0].source_ts, 120.0);
  EXPECT_EQ(data.flow_lines[0].target_ts, 210.0);
  EXPECT_EQ(data.flow_lines[0].source_level, 0);
  EXPECT_EQ(data.flow_lines[0].target_level, 1);
  EXPECT_EQ(data.flow_lines[0].category,
            tsl::profiler::ContextType::kGpuLaunch);
  EXPECT_EQ(data.flow_lines[0].color, flow1_color);

  // Flow 1, part 2
  EXPECT_EQ(data.flow_lines[1].source_ts, 210.0);
  EXPECT_EQ(data.flow_lines[1].target_ts, 230.0);
  EXPECT_EQ(data.flow_lines[1].source_level, 1);
  EXPECT_EQ(data.flow_lines[1].target_level, 1);
  EXPECT_EQ(data.flow_lines[1].category,
            tsl::profiler::ContextType::kGpuLaunch);
  EXPECT_EQ(data.flow_lines[1].color, flow1_color);

  // Flow 2
  EXPECT_EQ(data.flow_lines[2].source_ts, 310.0);
  EXPECT_EQ(data.flow_lines[2].target_ts, 380.0);
  EXPECT_EQ(data.flow_lines[2].source_level, 2);
  EXPECT_EQ(data.flow_lines[2].target_level, 2);
  EXPECT_EQ(data.flow_lines[2].category, tsl::profiler::ContextType::kGeneric);
  EXPECT_EQ(data.flow_lines[2].color, flow2_color);

  // Check flow_lines_by_flow_id
  ASSERT_TRUE(data.flow_lines_by_flow_id.contains("1"));
  EXPECT_THAT(data.flow_lines_by_flow_id.at("1"), SizeIs(2));
  ASSERT_TRUE(data.flow_lines_by_flow_id.contains("2"));
  EXPECT_THAT(data.flow_lines_by_flow_id.at("2"), SizeIs(1));

  // Check flow_ids_by_event_id for flow events
  EXPECT_THAT(data.flow_ids_by_event_id.at(1), ElementsAre("1"));
  EXPECT_THAT(data.flow_ids_by_event_id.at(2), ElementsAre("1"));
  EXPECT_THAT(data.flow_ids_by_event_id.at(3), ElementsAre("1"));
  EXPECT_THAT(data.flow_ids_by_event_id.at(4), ElementsAre("2"));
  EXPECT_THAT(data.flow_ids_by_event_id.at(5), ElementsAre("2"));
}

TEST_F(DataProviderTest, FlowEventsAffectTimeRange) {
  const std::vector<TraceEvent> all_events = {
      {.ph = Phase::kComplete,
       .pid = 1,
       .tid = 101,
       .name = "Task A",
       .ts = 100.0,
       .dur = 50.0},
      {.ph = Phase::kFlowStart,
       .pid = 1,
       .tid = 101,
       .name = "flow1",
       .ts = 50.0,
       .id = "1"},
      {.ph = Phase::kFlowEnd,
       .pid = 1,
       .tid = 101,
       .name = "flow1",
       .ts = 200.0,
       .id = "1"},
  };
  std::vector<TraceEvent> flame_events;
  std::vector<TraceEvent> flow_events;
  for (const auto& event : all_events) {
    if (event.ph == Phase::kComplete) {
      flame_events.push_back(event);
    }
    if (!event.id.empty()) {
      flow_events.push_back(event);
    }
  }

  data_provider_.ProcessTraceEvents({flame_events, {}, flow_events}, timeline_);
  EXPECT_DOUBLE_EQ(timeline_.visible_range().start(), 50.0);
  EXPECT_DOUBLE_EQ(timeline_.visible_range().end(), 200.0);
}

TEST_F(DataProviderTest, FlowEventFindLevelDefaultsToThreadStartLevel) {
  const std::vector<TraceEvent> all_events = {
      // Process 1, Thread 101
      {.ph = Phase::kComplete,
       .event_id = 10,
       .pid = 1,
       .tid = 101,
       .name = "Task A",
       .ts = 100.0,
       .dur = 50.0},
      {.ph = Phase::kFlowStart,
       .event_id = 1,
       .pid = 1,
       .tid = 101,
       .name = "flow1",
       .ts = 120.0,
       .id = "1",
       .category = tsl::profiler::ContextType::kGpuLaunch},
      // Process 1, Thread 102
      {.ph = Phase::kComplete,
       .event_id = 11,
       .pid = 1,
       .tid = 102,
       .name = "Task B",
       .ts = 200.0,
       .dur = 50.0},
      // Flow end ts=180 doesn't fall in Task B
      {.ph = Phase::kFlowEnd,
       .event_id = 2,
       .pid = 1,
       .tid = 102,
       .name = "flow1",
       .ts = 180.0,
       .id = "1",
       .category = tsl::profiler::ContextType::kGpuLaunch},
  };
  std::vector<TraceEvent> flame_events;
  std::vector<TraceEvent> flow_events;
  for (const auto& event : all_events) {
    if (event.ph == Phase::kComplete) {
      flame_events.push_back(event);
    }
    if (!event.id.empty()) {
      flow_events.push_back(event);
    }
  }

  data_provider_.ProcessTraceEvents({flame_events, {}, flow_events}, timeline_);
  const FlameChartTimelineData& data = timeline_.timeline_data();
  ASSERT_THAT(data.flow_lines, SizeIs(1));
  // p1/t101 is level 0, p1/t102 is level 1.
  // Flow start is at 120, falls in Task A, so source_level should be 0.
  // Flow end is at 180, doesn't fall in Task B (200-250), so target_level
  // should be thread 102's start_level, which is 1.
  EXPECT_EQ(data.flow_lines[0].source_ts, 120.0);
  EXPECT_EQ(data.flow_lines[0].target_ts, 180.0);
  EXPECT_EQ(data.flow_lines[0].source_level, 0);
  EXPECT_EQ(data.flow_lines[0].target_level, 1);
}

TEST_F(DataProviderTest, FlowEventFindLevelDeepestLevel) {
  const std::vector<TraceEvent> all_events = {
      // Process 1, Thread 101
      // Task A contains Task B
      {.ph = Phase::kComplete,
       .event_id = 100,
       .pid = 1,
       .tid = 101,
       .name = "Task A",
       .ts = 100.0,
       .dur = 100.0},
      {.ph = Phase::kComplete,
       .event_id = 101,
       .pid = 1,
       .tid = 101,
       .name = "Task B",
       .ts = 120.0,
       .dur = 50.0},
      // Flow starts in Task B
      {.ph = Phase::kFlowStart,
       .event_id = 200,
       .pid = 1,
       .tid = 101,
       .name = "flow1",
       .ts = 130.0,
       .id = "1",
       .category = tsl::profiler::ContextType::kGpuLaunch},
      // Flow ends in Task B
      {.ph = Phase::kFlowEnd,
       .event_id = 201,
       .pid = 1,
       .tid = 101,
       .name = "flow1",
       .ts = 140.0,
       .id = "1",
       .category = tsl::profiler::ContextType::kGpuLaunch},
  };
  std::vector<TraceEvent> flame_events;
  std::vector<TraceEvent> flow_events;
  for (const auto& event : all_events) {
    if (event.ph == Phase::kComplete) {
      flame_events.push_back(event);
    }
    if (!event.id.empty()) {
      flow_events.push_back(event);
    }
  }

  data_provider_.ProcessTraceEvents({flame_events, {}, flow_events}, timeline_);
  const FlameChartTimelineData& data = timeline_.timeline_data();
  ASSERT_THAT(data.flow_lines, SizeIs(1));
  // p1/t101 has 2 levels. Task A is level 0, Task B is level 1.
  // Flow starts at 130, falls in Task B, deepest level is 1.
  // Flow ends at 140, falls in Task B, deepest level is 1.
  EXPECT_EQ(data.flow_lines[0].source_ts, 130.0);
  EXPECT_EQ(data.flow_lines[0].target_ts, 140.0);
  EXPECT_EQ(data.flow_lines[0].source_level, 1);
  EXPECT_EQ(data.flow_lines[0].target_level, 1);
}

TEST_F(DataProviderTest, FlowEventFindLevelMultipleEventsOnLevel) {
  const std::vector<TraceEvent> all_events = {
      // Process 1, Thread 101
      {.ph = Phase::kComplete,
       .event_id = 100,
       .pid = 1,
       .tid = 101,
       .name = "Task A",
       .ts = 100.0,
       .dur = 100.0},
      {.ph = Phase::kComplete,
       .event_id = 101,
       .pid = 1,
       .tid = 101,
       .name = "Task B",
       .ts = 120.0,
       .dur = 10.0},
      {.ph = Phase::kComplete,
       .event_id = 102,
       .pid = 1,
       .tid = 101,
       .name = "Task C",
       .ts = 140.0,
       .dur = 10.0},
      // Flow starts in Task C
      {.ph = Phase::kFlowStart,
       .event_id = 200,
       .pid = 1,
       .tid = 101,
       .name = "flow1",
       .ts = 145.0,
       .id = "1",
       .category = tsl::profiler::ContextType::kGpuLaunch},
      // Flow ends in Task C
      {.ph = Phase::kFlowEnd,
       .event_id = 201,
       .pid = 1,
       .tid = 101,
       .name = "flow1",
       .ts = 146.0,
       .id = "1",
       .category = tsl::profiler::ContextType::kGpuLaunch},
  };
  std::vector<TraceEvent> flame_events;
  std::vector<TraceEvent> flow_events;
  for (const auto& event : all_events) {
    if (event.ph == Phase::kComplete) {
      flame_events.push_back(event);
    }
    if (!event.id.empty()) {
      flow_events.push_back(event);
    }
  }

  data_provider_.ProcessTraceEvents({flame_events, {}, flow_events}, timeline_);
  const FlameChartTimelineData& data = timeline_.timeline_data();
  ASSERT_THAT(data.flow_lines, SizeIs(1));
  // p1/t101: Task A lvl 0, Task B lvl 1, Task C lvl 1.
  // Flow starts at 145, falls in Task C, deepest level is 1.
  EXPECT_EQ(data.flow_lines[0].source_ts, 145.0);
  EXPECT_EQ(data.flow_lines[0].target_ts, 146.0);
  EXPECT_EQ(data.flow_lines[0].source_level, 1);
  EXPECT_EQ(data.flow_lines[0].target_level, 1);
}

TEST_F(DataProviderTest, CompleteEventWithIdIsHandledAsFlowEvent) {
  const std::vector<TraceEvent> all_events = {
      // Process 1, Thread 101
      {.ph = Phase::kComplete,
       .event_id = 10,
       .pid = 1,
       .tid = 101,
       .name = "Task A",
       .ts = 100.0,
       .dur = 50.0,
       .id = "1",
       .category = tsl::profiler::ContextType::kGpuLaunch},
      // Process 1, Thread 102
      {.ph = Phase::kComplete,
       .event_id = 11,
       .pid = 1,
       .tid = 102,
       .name = "Task B",
       .ts = 200.0,
       .dur = 50.0,
       .id = "1",
       .category = tsl::profiler::ContextType::kGpuLaunch},
  };
  std::vector<TraceEvent> flame_events;
  std::vector<TraceEvent> flow_events;
  for (const auto& event : all_events) {
    if (event.ph == Phase::kComplete) {
      flame_events.push_back(event);
    }
    if (!event.id.empty()) {
      flow_events.push_back(event);
    }
  }

  data_provider_.ProcessTraceEvents({flame_events, {}, flow_events}, timeline_);
  const FlameChartTimelineData& data = timeline_.timeline_data();
  ASSERT_THAT(data.flow_lines, SizeIs(1));
  EXPECT_EQ(data.flow_lines[0].source_ts, 100.0);
  EXPECT_EQ(data.flow_lines[0].target_ts, 200.0);
  EXPECT_EQ(data.flow_lines[0].source_level, 0);
  EXPECT_EQ(data.flow_lines[0].target_level, 1);
  EXPECT_EQ(data.flow_lines[0].category,
            tsl::profiler::ContextType::kGpuLaunch);
  EXPECT_EQ(data.flow_lines[0].color, kOrange80);
  EXPECT_THAT(data_provider_.GetFlowCategories(),
              UnorderedElementsAre(
                  static_cast<int>(tsl::profiler::ContextType::kGpuLaunch)));
}

TEST_F(DataProviderTest, FlowEventWithSameIdAndEventId) {
  const std::vector<TraceEvent> all_events = {
      {.ph = Phase::kFlowStart,
       .event_id = 10,
       .pid = 1,
       .tid = 101,
       .name = "flow1",
       .ts = 100.0,
       .id = "1"},
      {.ph = Phase::kFlowEnd,
       .event_id = 10,
       .pid = 1,
       .tid = 101,
       .name = "flow1",
       .ts = 200.0,
       .id = "1"},
  };
  std::vector<TraceEvent> flow_events;
  for (const auto& event : all_events) {
    if (!event.id.empty()) {
      flow_events.push_back(event);
    }
  }

  data_provider_.ProcessTraceEvents({{}, {}, flow_events}, timeline_);
  const FlameChartTimelineData& data = timeline_.timeline_data();
  EXPECT_THAT(data.flow_ids_by_event_id.at(10), ElementsAre("1"));
}

TEST_F(DataProviderTest, ProcessTraceEventsPreservesVisibleRange) {
  // Initial load
  const std::vector<TraceEvent> events1 = {{.ph = Phase::kComplete,
                                            .pid = 1,
                                            .tid = 1,
                                            .name = "Event 1",
                                            .ts = 1000.0,
                                            .dur = 100.0}};
  data_provider_.ProcessTraceEvents({events1, {}}, timeline_);

  // Set visible range to something specific (simulating zoom).
  timeline_.SetVisibleRange({1020.0, 1050.0});
  TimeRange visible_before = timeline_.visible_range();

  // Incremental load (new events, but within or related to current view)
  const std::vector<TraceEvent> events2 = {{.ph = Phase::kComplete,
                                            .pid = 1,
                                            .tid = 1,
                                            .name = "Event 1",
                                            .ts = 1000.0,
                                            .dur = 100.0},
                                           {.ph = Phase::kComplete,
                                            .pid = 1,
                                            .tid = 1,
                                            .name = "Event 2",
                                            .ts = 1200.0,
                                            .dur = 100.0}};

  data_provider_.ProcessTraceEvents({events2, {}}, timeline_);

  // Verify visible range is preserved.
  EXPECT_DOUBLE_EQ(timeline_.visible_range().start(), visible_before.start());
  EXPECT_DOUBLE_EQ(timeline_.visible_range().end(), visible_before.end());

  // Verify fetched data range is updated.
  EXPECT_DOUBLE_EQ(timeline_.fetched_data_time_range().start(), 1000.0);
  EXPECT_DOUBLE_EQ(timeline_.fetched_data_time_range().end(), 1300.0);
}

TEST_F(DataProviderTest, FlowLineColoringWithTop5Categories) {
  std::vector<TraceEvent> flow_events;
  int flow_id_counter = 0;
  auto add_cat_flows = [&](tsl::profiler::ContextType cat, int count) {
    for (int i = 0; i < count; ++i) {
      std::string flow_id = absl::StrCat("f", flow_id_counter++);
      flow_events.push_back({.ph = Phase::kFlowStart,
                             .pid = 1,
                             .tid = 1,
                             .ts = 1.0,
                             .id = flow_id,
                             .category = cat});
      flow_events.push_back({.ph = Phase::kFlowEnd,
                             .pid = 1,
                             .tid = 1,
                             .ts = 2.0,
                             .id = flow_id,
                             .category = cat});
    }
  };

  add_cat_flows(tsl::profiler::ContextType::kGeneric, 110);
  add_cat_flows(tsl::profiler::ContextType::kGpuLaunch, 100);
  add_cat_flows(tsl::profiler::ContextType::kTfExecutor, 90);
  add_cat_flows(tsl::profiler::ContextType::kPjRt, 80);
  add_cat_flows(tsl::profiler::ContextType::kTfrtTpuRuntime, 70);
  add_cat_flows(tsl::profiler::ContextType::kTpuEmbeddingEngine, 60);
  add_cat_flows(tsl::profiler::ContextType::kSharedBatchScheduler, 50);
  add_cat_flows(tsl::profiler::ContextType::kLegacy, 40);

  data_provider_.ProcessTraceEvents({{}, {}, flow_events}, timeline_);
  const FlameChartTimelineData& data = timeline_.timeline_data();

  auto get_color = [&](tsl::profiler::ContextType cat) {
    for (const auto& line : data.flow_lines) {
      if (line.category == cat) return line.color;
    }
    return static_cast<ImU32>(0);
  };

  EXPECT_EQ(get_color(tsl::profiler::ContextType::kGeneric), kRed80);
  EXPECT_EQ(get_color(tsl::profiler::ContextType::kGpuLaunch), kOrange80);
  EXPECT_EQ(get_color(tsl::profiler::ContextType::kTfExecutor), kYellow80);
  EXPECT_EQ(get_color(tsl::profiler::ContextType::kPjRt), kGreen80);
  EXPECT_EQ(get_color(tsl::profiler::ContextType::kTfrtTpuRuntime), kBlue80);
  EXPECT_EQ(get_color(tsl::profiler::ContextType::kTpuEmbeddingEngine),
            kCyan80);
  EXPECT_EQ(get_color(tsl::profiler::ContextType::kSharedBatchScheduler),
            kPurple80);
  EXPECT_EQ(get_color(tsl::profiler::ContextType::kLegacy), kPurple80);
}

TEST_F(DataProviderTest, FlowLineColoringWithTieBreaker) {
  ParsedTraceEvents parsed_events;
  parsed_events.flame_events = {
      CreateMetadataEvent(std::string(kProcessName), 1, 0, "Process A"),
      CreateMetadataEvent(std::string(kThreadName), 1, 101, "Thread A"),
  };

  // Add 50 flow events for kTfExecutor and 50 for kGpuLaunch
  for (int i = 0; i < 50; ++i) {
    std::string id_executor = absl::StrCat("executor_", i);
    std::string id_gpu = absl::StrCat("gpu_", i);

    parsed_events.flow_events.push_back(
        {.ph = Phase::kFlowStart,
         .pid = 1,
         .tid = 101,
         .id = id_executor,
         .category = tsl::profiler::ContextType::kTfExecutor});
    parsed_events.flow_events.push_back(
        {.ph = Phase::kFlowEnd,
         .pid = 1,
         .tid = 101,
         .id = id_executor,
         .category = tsl::profiler::ContextType::kTfExecutor});

    parsed_events.flow_events.push_back(
        {.ph = Phase::kFlowStart,
         .pid = 1,
         .tid = 101,
         .id = id_gpu,
         .category = tsl::profiler::ContextType::kGpuLaunch});
    parsed_events.flow_events.push_back(
        {.ph = Phase::kFlowEnd,
         .pid = 1,
         .tid = 101,
         .id = id_gpu,
         .category = tsl::profiler::ContextType::kGpuLaunch});
  }

  data_provider_.ProcessTraceEvents(parsed_events, timeline_);

  const FlameChartTimelineData& data = timeline_.timeline_data();

  auto get_color = [&](tsl::profiler::ContextType cat) -> ImU32 {
    for (const auto& line : data.flow_lines) {
      if (line.category == cat) return line.color;
    }
    return 0;
  };

  // Since kTfExecutor (2) < kGpuLaunch (9), kTfExecutor should come first and
  // get kOrange80!
  EXPECT_EQ(get_color(tsl::profiler::ContextType::kTfExecutor), kOrange80);
  EXPECT_EQ(get_color(tsl::profiler::ContextType::kGpuLaunch), kYellow80);
}

TEST_F(DataProviderTest, MultipleProcessTraceEventsClearsTop5FlowCategories) {
  auto create_flow_events =
      [](tsl::profiler::ContextType cat) -> std::vector<TraceEvent> {
    return {{.ph = Phase::kFlowStart,
             .pid = 1,
             .tid = 1,
             .ts = 1.0,
             .id = "1",
             .category = cat},
            {.ph = Phase::kFlowEnd,
             .pid = 1,
             .tid = 1,
             .ts = 2.0,
             .id = "1",
             .category = cat}};
  };

  data_provider_.ProcessTraceEvents(
      {{}, {}, create_flow_events(tsl::profiler::ContextType::kGpuLaunch)},
      timeline_);
  {
    const FlameChartTimelineData& data = timeline_.timeline_data();
    ASSERT_THAT(data.flow_lines, SizeIs(1));
    EXPECT_EQ(data.flow_lines[0].category,
              tsl::profiler::ContextType::kGpuLaunch);
    EXPECT_EQ(data.flow_lines[0].color, kOrange80);
  }

  data_provider_.ProcessTraceEvents(
      {{}, {}, create_flow_events(tsl::profiler::ContextType::kTfExecutor)},
      timeline_);
  {
    const FlameChartTimelineData& data = timeline_.timeline_data();
    ASSERT_THAT(data.flow_lines, SizeIs(1));
    EXPECT_EQ(data.flow_lines[0].category,
              tsl::profiler::ContextType::kTfExecutor);
    EXPECT_EQ(data.flow_lines[0].color, kOrange80);
  }
}

TEST_F(DataProviderTest, MultipleProcessTraceEventsClearsFlowCategories) {
  const std::vector<TraceEvent> flow_events1 = {
      {.ph = Phase::kFlowStart,
       .pid = 1,
       .tid = 101,
       .name = "flow1",
       .ts = 120.0,
       .id = "1",
       .category = tsl::profiler::ContextType::kGpuLaunch},
      {.ph = Phase::kFlowEnd,
       .pid = 1,
       .tid = 101,
       .name = "flow1",
       .ts = 130.0,
       .id = "1",
       .category = tsl::profiler::ContextType::kGpuLaunch}};
  data_provider_.ProcessTraceEvents({{}, {}, flow_events1}, timeline_);
  EXPECT_THAT(data_provider_.GetFlowCategories(),
              UnorderedElementsAre(
                  static_cast<int>(tsl::profiler::ContextType::kGpuLaunch)));

  const std::vector<TraceEvent> flow_events2 = {
      {.ph = Phase::kFlowStart,
       .pid = 1,
       .tid = 101,
       .name = "flow2",
       .ts = 140.0,
       .id = "2",
       .category = tsl::profiler::ContextType::kGeneric},
      {.ph = Phase::kFlowEnd,
       .pid = 1,
       .tid = 101,
       .name = "flow2",
       .ts = 150.0,
       .id = "2",
       .category = tsl::profiler::ContextType::kGeneric}};
  data_provider_.ProcessTraceEvents({{}, {}, flow_events2}, timeline_);
  EXPECT_THAT(data_provider_.GetFlowCategories(),
              UnorderedElementsAre(
                  static_cast<int>(tsl::profiler::ContextType::kGeneric)));
}

TEST_F(DataProviderTest, FlowLinesNestedEventLevelTest) {
  const std::vector<TraceEvent> events = {
      CreateMetadataEvent(std::string(kProcessName), 1, 0, "Process_1"),
      CreateMetadataEvent(std::string(kThreadName), 1, 101, "Thread_101"),
      // Level 0 event
      {.ph = Phase::kComplete,
       .pid = 1,
       .tid = 101,
       .name = "Task A",
       .ts = 10.0,
       .dur = 100.0},
      // Level 1 nested event starting exactly at 20.0 (tests binary search
      // boundary)
      {.ph = Phase::kComplete,
       .pid = 1,
       .tid = 101,
       .name = "Task A1",
       .ts = 20.0,
       .dur = 30.0}};

  const std::vector<TraceEvent> flow_events = {
      // Flow start at exactly 20.0 (inside A1)
      {.ph = Phase::kFlowStart,
       .pid = 1,
       .tid = 101,
       .name = "flow1",
       .ts = 20.0,
       .id = "1",
       .category = tsl::profiler::ContextType::kGeneric},
      // Flow end at 50.0 (inside A)
      {.ph = Phase::kFlowEnd,
       .pid = 1,
       .tid = 101,
       .name = "flow1",
       .ts = 50.0,
       .id = "1",
       .category = tsl::profiler::ContextType::kGeneric}};

  data_provider_.ProcessTraceEvents({events, {}, flow_events}, timeline_);

  const FlameChartTimelineData& data = timeline_.timeline_data();
  ASSERT_THAT(data.flow_lines, SizeIs(1));

  // The base thread level is 1 (group 0 is Process, group 1 is Thread)
  // Level 0 events have source_level = 0, Level 1 events have source_level
  // = 1. Flow start at 20.0 inside A1 -> 1 Flow end at 50.0 also inside A1
  // (boundary) -> 1
  EXPECT_EQ(data.flow_lines[0].source_level, 1);
  EXPECT_EQ(data.flow_lines[0].target_level, 1);
}

TEST_F(DataProviderTest, ProcessTraceEventsPreservesExpandedState) {
  const std::vector<TraceEvent> events = {
      CreateMetadataEvent(std::string(kProcessName), 1, 0, "Process_1"),
      // Thread 101 has multiple levels (Task A1 is nested inside Task A).
      // This makes Thread 101 collapsible.
      {.ph = Phase::kComplete,
       .pid = 1,
       .tid = 101,
       .name = "Task A",
       .ts = 100.0,
       .dur = 100.0},
      {.ph = Phase::kComplete,
       .pid = 1,
       .tid = 101,
       .name = "Task A1",
       .ts = 110.0,
       .dur = 50.0},
      CreateMetadataEvent(std::string(kProcessName), 2, 0, "Process_2"),
      // Thread 201 has only a single level, making it non-collapsible.
      {.ph = Phase::kComplete,
       .pid = 2,
       .tid = 201,
       .name = "Task B",
       .ts = 200.0,
       .dur = 50.0}};

  // Initial load
  data_provider_.ProcessTraceEvents({events, {}}, timeline_);

  // By default, first process is expanded, others collapsed.
  // Group 0: Process 1, Group 1: Thread 101, Group 2: Process 2, Group 3:
  // Thread 201
  ASSERT_THAT(timeline_.timeline_data().groups, SizeIs(4));
  EXPECT_TRUE(timeline_.timeline_data().groups[0].expanded);   // Process 1
  EXPECT_TRUE(timeline_.timeline_data().groups[1].expanded);   // Thread 101
  EXPECT_FALSE(timeline_.timeline_data().groups[2].expanded);  // Process 2
  EXPECT_TRUE(timeline_.timeline_data().groups[3].expanded);   // Thread 201

  // Manually collapse Process 1 and Thread 101.
  // Manually expand Process 2 and collapse Thread 201 (which shouldn't stick).
  {
    FlameChartTimelineData data = timeline_.timeline_data();
    data.groups[0].expanded = false;  // Process 1 -> false
    data.groups[1].expanded = false;  // Thread 101 (collapsible) -> false
    data.groups[2].expanded = true;   // Process 2 -> true
    data.groups[3].expanded = false;  // Thread 201 (non-collapsible) -> false
    timeline_.SetTimelineData(std::move(data));
  }

  // Simulate incremental loading (new events)
  const std::vector<TraceEvent> new_events = {{.ph = Phase::kComplete,
                                               .pid = 1,
                                               .tid = 101,
                                               .name = "Task C",
                                               .ts = 300.0,
                                               .dur = 50.0},
                                              {.ph = Phase::kComplete,
                                               .pid = 2,
                                               .tid = 201,
                                               .name = "Task D",
                                               .ts = 400.0,
                                               .dur = 50.0}};

  std::vector<TraceEvent> all_events = events;
  all_events.insert(all_events.end(), new_events.begin(), new_events.end());

  data_provider_.ProcessTraceEvents({all_events, {}}, timeline_);

  // Verify that expansion states are preserved for collapsible tracks,
  // but unconditionally forced to true for non-collapsible tracks.
  ASSERT_THAT(timeline_.timeline_data().groups, SizeIs(4));
  EXPECT_FALSE(timeline_.timeline_data()
                   .groups[0]
                   .expanded);  // Process 1 (PRESERVED false)
  EXPECT_FALSE(timeline_.timeline_data()
                   .groups[1]
                   .expanded);  // Thread 101 (PRESERVED false, multi-level)
  EXPECT_TRUE(timeline_.timeline_data()
                  .groups[2]
                  .expanded);  // Process 2 (PRESERVED true)
  EXPECT_TRUE(timeline_.timeline_data()
                  .groups[3]
                  .expanded);  // Thread 201 (FORCED TRUE, single-level)
}

TEST_F(DataProviderTest, ProcessTraceEventsForcesExpansionForOneLineThreads) {
  // Initial load
  timeline_.SetTimelineData({});
  const std::vector<TraceEvent> events = {
      // Process 1: Thread 101 (one line)
      {.ph = Phase::kComplete,
       .pid = 1,
       .tid = 101,
       .name = "Event 1",
       .ts = Microseconds(0),
       .dur = Microseconds(10)},
  };
  ParsedTraceEvents parsed_events;
  parsed_events.flame_events = events;
  DataProvider provider;
  provider.ProcessTraceEvents(parsed_events, timeline_);

  // Thread 101 should be expanded by default (one line)
  ASSERT_THAT(timeline_.timeline_data().groups, SizeIs(2));
  EXPECT_TRUE(timeline_.timeline_data().groups[1].expanded);

  // Manually collapse Thread 101
  {
    FlameChartTimelineData data = timeline_.timeline_data();
    data.groups[1].expanded = false;
    timeline_.SetTimelineData(std::move(data));
  }

  // Reload same data (simulate update)
  provider.ProcessTraceEvents(parsed_events, timeline_);

  // One-line threads should be expanded by default on initial load, but
  // even if collapsed by user, the current implementation overrides it to true
  // upon reload.
  EXPECT_TRUE(timeline_.timeline_data().groups[1].expanded);
}

TEST_F(DataProviderTest,
       ProcessTraceEventsPreservesExpandedStateForCounterTrack) {
  const std::vector<TraceEvent> events = {
      CreateMetadataEvent(std::string(kProcessName), 1, 0, "Process_1"),
      {.ph = Phase::kComplete,
       .pid = 1,
       .tid = 101,
       .name = "Task A",
       .ts = 100.0,
       .dur = 50.0}};

  const std::vector<CounterEvent> counter_events = {
      {.pid = 1,
       .name = "Test Counter",
       .timestamps = {100.0, 150.0},
       .values = {10.0, 20.0}}};

  // Initial load
  data_provider_.ProcessTraceEvents({events, counter_events}, timeline_);

  // Timeline groups will have:
  // 0: Process 1
  // 1: Thread 101
  // 2: Test Counter
  ASSERT_THAT(timeline_.timeline_data().groups, SizeIs(3));
  EXPECT_TRUE(timeline_.timeline_data().groups[0].expanded);  // Process 1
  EXPECT_TRUE(timeline_.timeline_data().groups[1].expanded);  // Thread 101
  // Counter groups are typically expanded by default depending on the
  // name/process.
  EXPECT_TRUE(timeline_.timeline_data().groups[2].expanded);  // Counter

  {
    FlameChartTimelineData data = timeline_.timeline_data();
    data.groups[0].expanded = false;  // Manually collapse Process 1
    data.groups[1].expanded = false;  // Manually collapse single-line thread
    data.groups[2].expanded = false;  // Manually collapse counter track
    timeline_.SetTimelineData(std::move(data));
  }

  // Simulate incremental loading (new events)
  const std::vector<TraceEvent> new_events = {{.ph = Phase::kComplete,
                                               .pid = 1,
                                               .tid = 101,
                                               .name = "Task C",
                                               .ts = 300.0,
                                               .dur = 50.0}};

  std::vector<TraceEvent> all_events = events;
  all_events.insert(all_events.end(), new_events.begin(), new_events.end());

  data_provider_.ProcessTraceEvents({all_events, counter_events}, timeline_);

  // Verify that expansion states are preserved for processes, but
  // single-line threads and counters are unconditionally forced to true,
  // making them effectively uncollapsible.
  ASSERT_THAT(timeline_.timeline_data().groups, SizeIs(3));
  EXPECT_FALSE(timeline_.timeline_data()
                   .groups[0]
                   .expanded);  // Process 1 (PRESERVED false)
  EXPECT_TRUE(timeline_.timeline_data()
                  .groups[1]
                  .expanded);  // Thread 101 (FORCED TRUE)
  EXPECT_TRUE(timeline_.timeline_data()
                  .groups[2]
                  .expanded);  // Test Counter (FORCED TRUE)
}

TEST_F(DataProviderTest, ProcessesSortedByThreadDmaPriority) {
  const std::vector<TraceEvent> events = {
      CreateMetadataEvent(std::string(kProcessName), 1, 0, "Normal Process"),
      CreateMetadataEvent(std::string(kThreadName), 1, 101, "Normal Thread"),
      {.ph = Phase::kComplete,
       .pid = 1,
       .tid = 101,
       .name = "Task 1",
       .ts = 100.0,
       .dur = 10.0},
      CreateMetadataEvent(std::string(kProcessName), 2, 0, "Another Process"),
      CreateMetadataEvent(std::string(kThreadName), 2, 201,
                          "Thread containing DMA"),
      {.ph = Phase::kComplete,
       .pid = 2,
       .tid = 201,
       .name = "Task 2",
       .ts = 110.0,
       .dur = 10.0},
  };

  data_provider_.ProcessTraceEvents({events, {}}, timeline_);

  const FlameChartTimelineData& data = timeline_.timeline_data();
  ASSERT_THAT(data.groups, SizeIs(4));

  EXPECT_EQ(data.groups[0].name, "Another Process");
  EXPECT_EQ(data.groups[2].name, "Normal Process");
}

TEST_F(DataProviderTest, ProcessesSortedByAsyncEventPriority) {
  const std::vector<TraceEvent> events = {
      CreateMetadataEvent(std::string(kProcessName), 1, 0, "Normal Process"),
      {.ph = Phase::kComplete,
       .pid = 1,
       .tid = 101,
       .name = "Task 1",
       .ts = 100.0,
       .dur = 10.0},
      CreateMetadataEvent(std::string(kProcessName), 2, 0,
                          "Async Process Event"),
      {.ph = Phase::kComplete,
       .pid = 2,
       .tid = 201,
       .name = "Task 2",
       .ts = 110.0,
       .dur = 10.0,
       .id = "",
       .args = {},
       .is_async = true},
      {.ph = Phase::kComplete,
       .pid = 2,
       .tid = 202,  // Different tid, but same name and process
       .name = "Task 2",
       .ts = 130.0,
       .dur = 10.0,
       .id = "",
       .args = {},
       .is_async = true},
  };

  data_provider_.ProcessTraceEvents({events, {}}, timeline_);

  const FlameChartTimelineData& data = timeline_.timeline_data();
  ASSERT_THAT(data.groups, SizeIs(4));

  EXPECT_EQ(data.groups[0].name, "Async Process Event");
  EXPECT_EQ(data.groups[2].name, "Normal Process");
}

TEST_F(DataProviderTest,
       AppendEventToTimelineDataWithDataMotionLayersUtilizationKillsMutant) {
  const std::vector<TraceEvent> events = {
      CreateMetadataEvent(std::string(kThreadName), 1, 101,
                          std::string(kDataMotionLayersUtilization)),
      {.ph = Phase::kComplete,
       .pid = 1,
       .tid = 101,
       .name = "Task A",
       .ts = 100.0,
       .dur = 50.0,
       .args = {{"Name", "HloOpValue"}}}};

  data_provider_.ProcessTraceEvents({events, {}}, timeline_);

  const FlameChartTimelineData& data = timeline_.timeline_data();
  ASSERT_THAT(data.entry_args, SizeIs(1));
  EXPECT_EQ(data.entry_args[0].at(std::string(kHloOp)), "HloOpValue");
}

TEST_F(DataProviderTest,
       AppendEventToTimelineDataNormalThreadWithOnlyHloModuleDoesNotAddHloOp) {
  const std::vector<TraceEvent> events = {
      CreateMetadataEvent(std::string(kThreadName), 1, 101, "Normal Thread"),
      {.ph = Phase::kComplete,
       .pid = 1,
       .tid = 101,
       .name = "Normal Task",
       .ts = 100.0,
       .dur = 50.0,
       .args = {{std::string(kHloModule), "ModuleValue"}}}};

  data_provider_.ProcessTraceEvents({events, {}}, timeline_);

  const FlameChartTimelineData& data = timeline_.timeline_data();
  ASSERT_THAT(data.entry_args, SizeIs(1));

  // Normal thread should NOT enter HLO processing block and should NOT create
  // kHloOp if it wasn't there.
  EXPECT_EQ(data.entry_args[0].count(std::string(kHloOp)), 0);
}

TEST_F(DataProviderTest, AppendEventToTimelineDataDecoratesHloModule) {
  const std::vector<TraceEvent> events = {
      {.ph = Phase::kComplete,
       .pid = 1,
       .tid = 101,
       .name = "Task A",
       .ts = 100.0,
       .dur = 50.0,
       .args = {{std::string(kHloOp), "OpValue"},
                {std::string(kHloModule), "ModuleValue"},
                {std::string(kHloModuleId), "123"}}}};

  data_provider_.ProcessTraceEvents({events, {}}, timeline_);

  const FlameChartTimelineData& data = timeline_.timeline_data();
  ASSERT_THAT(data.entry_args, SizeIs(1));

  // Should be decorated as "ModuleValue(123)"
  EXPECT_EQ(data.entry_args[0].at(std::string(kHloModule)), "ModuleValue(123)");
}

TEST_F(DataProviderTest, PopulateSyncProcessTrackDoesNotUseAsyncLayout) {
  const std::vector<TraceEvent> events = {
      {.ph = Phase::kComplete,
       .pid = 1,
       .tid = 101,
       .name = "Task A",
       .ts = 100.0,
       .dur = 50.0},
      {.ph = Phase::kComplete,
       .pid = 1,
       .tid = 101,
       .name = "Task B",
       .ts = 120.0,
       .dur = 10.0},
  };

  data_provider_.ProcessTraceEvents({events, {}}, timeline_);

  const FlameChartTimelineData& data = timeline_.timeline_data();

  // In Sync mode, it should group by TID. We expect:
  // - 1 Process Track
  // - 1 Thread Track
  // Total 2 groups.
  // If the mutant makes it Async, it will split by name into "Task A" and "Task
  // B" tracks (Total 3 groups).
  ASSERT_THAT(data.groups, SizeIs(2));
}

TEST_F(DataProviderTest,
       PopulateProcessTrackWithDataMotionLayersUtilizationNameIsAsync) {
  const std::vector<TraceEvent> events = {
      CreateMetadataEvent(std::string(kProcessName), 2, 0,
                          std::string(kDataMotionLayersUtilization)),
      {.ph = Phase::kComplete,
       .pid = 2,
       .tid = 101,
       .name = "Task A",
       .ts = 100.0,
       .dur = 50.0,
       .is_async = false},
      CreateMetadataEvent(std::string(kProcessName), 1, 0, "Normal Process"),
      {.ph = Phase::kComplete,
       .pid = 1,
       .tid = 102,
       .name = "Task B",
       .ts = 100.0,
       .dur = 50.0,
       .is_async = false},
  };

  ParsedTraceEvents parsed_events;
  parsed_events.flame_events = events;

  data_provider_.ProcessTraceEvents(parsed_events, timeline_);

  const FlameChartTimelineData& data = timeline_.timeline_data();

  // Process 2 (Data Motion Layers Utilization) has priority 1.
  // Process 1 (Normal Process) has priority 0.
  // So Process 2 should be first!
  ASSERT_THAT(data.groups, Not(IsEmpty()));
  EXPECT_EQ(data.groups[0].name, std::string(kDataMotionLayersUtilization));
}

TEST_F(DataProviderTest, PopulateProcessTrackWithDmaThreadNameIsAsync) {
  const std::vector<TraceEvent> events = {
      CreateMetadataEvent(std::string(kThreadName), 2, 101, "Device DMA"),
      {.ph = Phase::kComplete,
       .pid = 2,
       .tid = 101,
       .name = "Task A",
       .ts = 100.0,
       .dur = 50.0,
       .is_async = false},
      CreateMetadataEvent(std::string(kProcessName), 1, 0, "Normal Process"),
      {.ph = Phase::kComplete,
       .pid = 1,
       .tid = 102,
       .name = "Task B",
       .ts = 100.0,
       .dur = 50.0,
       .is_async = false},
  };

  ParsedTraceEvents parsed_events;
  parsed_events.flame_events = events;

  data_provider_.ProcessTraceEvents(parsed_events, timeline_);

  const FlameChartTimelineData& data = timeline_.timeline_data();

  // Process 2 (has DMA thread) has priority 1.
  // Process 1 (Normal Process) has priority 0.
  // So Process 2 should be first!
  ASSERT_THAT(data.groups, Not(IsEmpty()));
  EXPECT_EQ(data.groups[0].name, "Process_2");
}

TEST_F(DataProviderTest,
       PopulateProcessTrackWithAsyncProcessNameButSyncEventsIsTreatedAsAsync) {
  const std::vector<TraceEvent> events = {
      CreateMetadataEvent(std::string(kProcessName), 1, 0, "Normal Process"),
      {.ph = Phase::kComplete,
       .pid = 1,
       .tid = 101,
       .name = "Task B",
       .ts = 100.0,
       .dur = 50.0},
      CreateMetadataEvent(std::string(kProcessName), 2, 0,
                          std::string(kDataMotionLayersUtilization)),
      {.ph = Phase::kComplete,
       .pid = 2,
       .tid = 102,
       .name = "Task A",
       .ts = 100.0,
       .dur = 50.0},
  };

  ParsedTraceEvents parsed_events;
  parsed_events.flame_events = events;

  data_provider_.ProcessTraceEvents(parsed_events, timeline_);

  const FlameChartTimelineData& data = timeline_.timeline_data();

  // Process 2 (DataMotionLayersUtilization) has higher priority because of its
  // name. Process 1 (Normal Process) has priority 0. So Process 2 should be
  // first!
  ASSERT_THAT(data.groups, Not(IsEmpty()));
  EXPECT_EQ(data.groups[0].name, std::string(kDataMotionLayersUtilization));
}

TEST_F(DataProviderTest,
       PopulateProcessTrackWithDataMotionLayersUtilizationPackedEfficiently) {
  const std::vector<TraceEvent> events = {
      CreateMetadataEvent(std::string(kProcessName), 1, 0,
                          std::string(kDataMotionLayersUtilization)),
      {.ph = Phase::kComplete,
       .pid = 1,
       .tid = 101,
       .name = "Task A",
       .ts = 100.0,
       .dur = 10.0,  // Non-overlapping!
       .id = "",
       .args = {},
       .is_async = true},
      {.ph = Phase::kComplete,
       .pid = 1,
       .tid = 101,
       .name = "Task A",
       .ts = 120.0,
       .dur = 10.0,
       .id = "",
       .args = {},
       .is_async = true},
  };

  data_provider_.ProcessTraceEvents({events, {}}, timeline_);

  const FlameChartTimelineData& data = timeline_.timeline_data();

  // In Async mode with non-overlapping events, it should PACK them into ONE
  // track. We expect:
  // - 1 Process Track
  // - 1 packed track for both "Task A" and "Task B" (since they don't overlap).
  // Total 2 groups.
  // If the mutant makes it create a new row unconditionally, it will split into
  // "Task A" and "Task B" tracks (Total 3 groups).
  ASSERT_THAT(data.groups, SizeIs(2));
}

TEST_F(DataProviderTest, PopulateProcessTrackWithAsyncThreadName) {
  const std::vector<TraceEvent> events = {
      CreateMetadataEvent(std::string(kThreadName), 1, 101, std::string(kDma)),
      {.ph = Phase::kComplete,
       .pid = 1,
       .tid = 101,
       .name = "SyncTask",
       .ts = 100.0,
       .dur = 50.0}};

  ParsedTraceEvents parsed_events;
  parsed_events.flame_events = events;

  data_provider_.ProcessTraceEvents(parsed_events, timeline_);

  const FlameChartTimelineData& data = timeline_.timeline_data();

  // Verify that the thread is present and named correctly.
  ASSERT_THAT(data.groups, SizeIs(2));
  EXPECT_EQ(data.groups[1].name, std::string(kDma));
}

TEST_F(DataProviderTest, PopulateThreadTrackWithPackedLayoutNonNestedOverlap) {
  const std::vector<TraceEvent> events = {{.ph = Phase::kComplete,
                                           .pid = 1,
                                           .tid = 101,
                                           .name = "AsyncOp",
                                           .ts = 50.0,
                                           .dur = 100.0,
                                           .is_async = true},
                                          {.ph = Phase::kComplete,
                                           .pid = 1,
                                           .tid = 102,
                                           .name = "AsyncOp",
                                           .ts = 100.0,
                                           .dur = 100.0,
                                           .is_async = true}};

  ParsedTraceEvents parsed_events;
  parsed_events.flame_events = events;

  data_provider_.ProcessTraceEvents(parsed_events, timeline_);

  const FlameChartTimelineData& data = timeline_.timeline_data();

  // We expect Process Track and Synthetic Async Track.
  ASSERT_THAT(data.groups, SizeIs(2));
  // We expect BOTH events to be rendered in Packed Layout.
  EXPECT_THAT(data.entry_names, SizeIs(2));
}

TEST_F(DataProviderTest, PopulateProcessTrackWithAsyncProcessNamePriorityFlip) {
  const std::vector<TraceEvent> events = {
      CreateMetadataEvent(std::string(kProcessName), 1, 0, "NormalProcess"),
      CreateMetadataEvent(std::string(kProcessName), 2, 0,
                          std::string(kDataMotionLayersUtilization)),
      {.ph = Phase::kComplete,
       .pid = 1,
       .tid = 101,
       .name = "TaskA",
       .ts = 100.0,
       .dur = 50.0},
      {.ph = Phase::kComplete,
       .pid = 2,
       .tid = 201,
       .name = "TaskB",
       .ts = 100.0,
       .dur = 50.0}};

  ParsedTraceEvents parsed_events;
  parsed_events.flame_events = events;

  data_provider_.ProcessTraceEvents(parsed_events, timeline_);

  const FlameChartTimelineData& data = timeline_.timeline_data();

  // Async process named kDataMotionLayersUtilization should be prioritized
  // over normal process, even with a higher PID.
  // Track order: Process 2 (Async) -> Thread 201 -> Process 1 (Normal) ->
  // Thread 101.
  ASSERT_THAT(data.groups, SizeIs(4));
  EXPECT_EQ(data.groups[0].name, std::string(kDataMotionLayersUtilization));
  EXPECT_EQ(data.groups[2].name, "NormalProcess");
}

TEST_F(DataProviderTest, PopulateProcessTrackWithAsyncThreadNamePriorityFlip) {
  const std::vector<TraceEvent> events = {
      CreateMetadataEvent(std::string(kProcessName), 1, 0, "NormalProcess"),
      CreateMetadataEvent(std::string(kProcessName), 2, 0,
                          "AsyncThreadProcess"),
      CreateMetadataEvent(std::string(kThreadName), 2, 201, std::string(kDma)),
      {.ph = Phase::kComplete,
       .pid = 1,
       .tid = 101,
       .name = "TaskA",
       .ts = 100.0,
       .dur = 50.0},
      {.ph = Phase::kComplete,
       .pid = 2,
       .tid = 201,
       .name = "TaskB",
       .ts = 100.0,
       .dur = 50.0}};

  ParsedTraceEvents parsed_events;
  parsed_events.flame_events = events;

  data_provider_.ProcessTraceEvents(parsed_events, timeline_);

  const FlameChartTimelineData& data = timeline_.timeline_data();

  // Process 2 (with kDma thread) should be prioritized over normal process 1,
  // even with a higher PID.
  // Track order: Process 2 Group -> Thread 201 Group -> Process 1 Group ->
  // Thread 101 Group.
  ASSERT_THAT(data.groups, SizeIs(4));
  EXPECT_EQ(data.groups[0].name, "AsyncThreadProcess");
  EXPECT_EQ(data.groups[2].name, "NormalProcess");
}

TEST_F(DataProviderTest,
       AppendEventToTimelineDataWithProgramIdDecoratesHloModule) {
  const std::vector<TraceEvent> events = {
      {.ph = Phase::kComplete,
       .pid = 1,
       .tid = 101,
       .name = "Task A",
       .ts = 100.0,
       .dur = 50.0,
       .args = {{std::string(kHloOp), "OpValue"},
                {std::string(kHloModule), "ModuleValue"},
                {std::string(kProgramId), "456"}}}};

  data_provider_.ProcessTraceEvents({events, {}}, timeline_);

  const FlameChartTimelineData& data = timeline_.timeline_data();
  ASSERT_THAT(data.entry_args, SizeIs(1));
  EXPECT_EQ(data.entry_args[0].at(std::string(kHloModule)), "ModuleValue(456)");
}

TEST_F(DataProviderTest,
       AppendEventToTimelineDataWithKernelDetailsDecoratesHloModule) {
  const std::vector<TraceEvent> events = {
      {.ph = Phase::kComplete,
       .pid = 1,
       .tid = 101,
       .name = "Task A",
       .ts = 100.0,
       .dur = 50.0,
       .args = {{std::string(kHloOp), "OpValue"},
                {std::string(kHloModule), "ModuleValue"},
                {std::string(kKernelDetails), "some module:foo_789 inside"}}}};

  data_provider_.ProcessTraceEvents({events, {}}, timeline_);

  const FlameChartTimelineData& data = timeline_.timeline_data();
  ASSERT_THAT(data.entry_args, SizeIs(1));
  EXPECT_EQ(data.entry_args[0].at(std::string(kHloModule)), "ModuleValue(789)");
}

TEST_F(
    DataProviderTest,
    AppendEventToTimelineDataWithKernelDetailsNoMatchDoesNotDecorateHloModule) {
  const std::vector<TraceEvent> events = {
      {.ph = Phase::kComplete,
       .pid = 1,
       .tid = 101,
       .name = "Task A",
       .ts = 100.0,
       .dur = 50.0,
       .args = {{std::string(kHloOp), "OpValue"},
                {std::string(kHloModule), "ModuleValue"},
                {std::string(kKernelDetails), "no match"}}}};

  data_provider_.ProcessTraceEvents({events, {}}, timeline_);

  const FlameChartTimelineData& data = timeline_.timeline_data();
  ASSERT_THAT(data.entry_args, SizeIs(1));
  EXPECT_EQ(data.entry_args[0].at(std::string(kHloModule)), "ModuleValue");
}

TEST_F(DataProviderTest, AppendEventToTimelineDataViaXlaModulesThread) {
  ParsedTraceEvents parsed_events;
  parsed_events.flame_events = {
      CreateMetadataEvent(std::string(kProcessName), 1, 0, "Process A"),
      CreateMetadataEvent(std::string(kThreadName), 1, 101,
                          std::string(kXlaOps)),
      CreateMetadataEvent(std::string(kThreadName), 1, 102,
                          std::string(kXlaModules)),
      {.ph = Phase::kComplete,
       .pid = 1,
       .tid = 102,
       .name = "ModuleValue(789)",
       .ts = 50.0,
       .dur = 200.0},
      {.ph = Phase::kComplete,
       .pid = 1,
       .tid = 101,
       .name = "Task A",
       .ts = 100.0,
       .dur = 50.0,
       .args = {{std::string(kHloOp), "OpValue"}}}  // No kHloModule!
  };

  data_provider_.ProcessTraceEvents(parsed_events, timeline_);

  const FlameChartTimelineData& data = timeline_.timeline_data();
  // We expect 2 entries in entry_args:
  // 1. Task A (which gets decorated from XLA Modules)
  // 2. The module event in XLA Modules thread itself (which does not get
  // decorated, so gets default)
  ASSERT_THAT(data.entry_args, SizeIs(2));

  // Find which one is for Task A (it has hlo_op).
  int task_a_index = -1;
  for (size_t i = 0; i < data.entry_args.size(); ++i) {
    if (data.entry_args[i].count(std::string(kHloOp)) > 0) {
      task_a_index = i;
      break;
    }
  }
  ASSERT_NE(task_a_index, -1);
  EXPECT_EQ(data.entry_args[task_a_index].at(std::string(kHloModule)),
            "ModuleValue(789)");
}

TEST_F(DataProviderTest,
       AppendEventToTimelineDataViaXlaModulesThreadModuleEventEndsBefore) {
  ParsedTraceEvents parsed_events;
  parsed_events.flame_events = {
      CreateMetadataEvent(std::string(kProcessName), 1, 0, "Process A"),
      CreateMetadataEvent(std::string(kThreadName), 1, 101,
                          std::string(kXlaOps)),
      CreateMetadataEvent(std::string(kThreadName), 1, 102,
                          std::string(kXlaModules)),
      {.ph = Phase::kComplete,
       .pid = 1,
       .tid = 102,
       .name = "ModuleValue(789)",
       .ts = 50.0,
       .dur = 30.0},  // Ends at 80.0
      {.ph = Phase::kComplete,
       .pid = 1,
       .tid = 101,
       .name = "Task A",
       .ts = 100.0,
       .dur = 50.0,
       .args = {{std::string(kHloOp), "OpValue"}}}  // No kHloModule!
  };

  data_provider_.ProcessTraceEvents(parsed_events, timeline_);

  const FlameChartTimelineData& data = timeline_.timeline_data();
  // We expect 2 entries in entry_args:
  // 1. Task A (should NOT be decorated)
  // 2. The module event in XLA Modules thread itself
  ASSERT_THAT(data.entry_args, SizeIs(2));

  // Find which one is for Task A (it has hlo_op).
  int task_a_index = -1;
  for (size_t i = 0; i < data.entry_args.size(); ++i) {
    if (data.entry_args[i].count(std::string(kHloOp)) > 0) {
      task_a_index = i;
      break;
    }
  }
  ASSERT_NE(task_a_index, -1);
  // Should be default because it ended before Task A started.
  EXPECT_EQ(data.entry_args[task_a_index].at(std::string(kHloModule)),
            "default");
}

TEST_F(DataProviderTest, UnnamedProcessWithEvents) {
  const std::vector<TraceEvent> events = {{.ph = Phase::kComplete,
                                           .pid = 4,
                                           .tid = 401,
                                           .name = "Task A",
                                           .ts = 100.0,
                                           .dur = 50.0}};

  data_provider_.ProcessTraceEvents({events, {}}, timeline_);

  const FlameChartTimelineData& data = timeline_.timeline_data();
  ASSERT_THAT(data.groups, SizeIs(2));  // Process group and thread group

  EXPECT_EQ(data.groups[0].name, "Process_4");  // Default name!
  EXPECT_EQ(data.groups[0].nesting_level, 0);
  EXPECT_EQ(data.groups[1].name, "Thread_401");  // Default name!
  EXPECT_EQ(data.groups[1].nesting_level, 1);
}

TEST_F(DataProviderTest,
       AppendEventToTimelineDataWithXlaOpsThreadDefaultHloOp) {
  ParsedTraceEvents parsed_events;
  parsed_events.flame_events = {
      CreateMetadataEvent(std::string(kProcessName), 1, 0, "Process A"),
      CreateMetadataEvent(std::string(kThreadName), 1, 101,
                          std::string(kXlaOps)),
      {.ph = Phase::kComplete,
       .pid = 1,
       .tid = 101,
       .name = "OpName",
       .ts = 100.0,
       .dur = 50.0}};

  data_provider_.ProcessTraceEvents(parsed_events, timeline_);

  const FlameChartTimelineData& data = timeline_.timeline_data();
  ASSERT_THAT(data.entry_args, SizeIs(1));
  EXPECT_EQ(data.entry_args[0].at(std::string(kHloOp)), "OpName");
}

TEST_F(DataProviderTest,
       AppendEventToTimelineDataWithHloModuleOnlyNoDecoration) {
  const std::vector<TraceEvent> events = {
      {.ph = Phase::kComplete,
       .pid = 1,
       .tid = 101,
       .name = "Task A",
       .ts = 100.0,
       .dur = 50.0,
       .args = {{std::string(kHloOp), "OpValue"},
                {std::string(kHloModule), "ModuleValue"}}}};

  data_provider_.ProcessTraceEvents({events, {}}, timeline_);

  const FlameChartTimelineData& data = timeline_.timeline_data();
  ASSERT_THAT(data.entry_args, SizeIs(1));
  EXPECT_EQ(data.entry_args[0].at(std::string(kHloModule)), "ModuleValue");
}

TEST_F(DataProviderTest, IgnoredPhaseEvent) {
  const std::vector<TraceEvent> events = {{.ph = Phase::kAsyncBegin,
                                           .pid = 1,
                                           .tid = 101,
                                           .name = "Task A",
                                           .ts = 100.0,
                                           .dur = 50.0}};

  data_provider_.ProcessTraceEvents({events, {}}, timeline_);

  const FlameChartTimelineData& data = timeline_.timeline_data();
  EXPECT_THAT(data.entry_start_times, IsEmpty());
}

TEST_F(DataProviderTest, EventSortingWithTieBreaker) {
  const std::vector<TraceEvent> events = {{.ph = Phase::kComplete,
                                           .pid = 1,
                                           .tid = 101,
                                           .name = "ShortEvent",
                                           .ts = 100.0,
                                           .dur = 10.0},
                                          {.ph = Phase::kComplete,
                                           .pid = 1,
                                           .tid = 101,
                                           .name = "LongEvent",
                                           .ts = 100.0,
                                           .dur = 50.0}};

  data_provider_.ProcessTraceEvents({events, {}}, timeline_);

  const FlameChartTimelineData& data = timeline_.timeline_data();
  ASSERT_THAT(data.entry_names, SizeIs(2));
  EXPECT_EQ(data.entry_names[0], "LongEvent");
  EXPECT_EQ(data.entry_names[1], "ShortEvent");
}

TEST_F(DataProviderTest, CounterSortingWithEmptyTimestamps) {
  CounterEvent empty_event;
  empty_event.name = "CounterA";
  empty_event.pid = 1;

  CounterEvent normal_event;
  normal_event.name = "CounterA";
  normal_event.pid = 1;
  normal_event.timestamps = {100.0};
  normal_event.values = {10.0};

  ParsedTraceEvents parsed_events;
  parsed_events.counter_events = {empty_event, normal_event};

  data_provider_.ProcessTraceEvents(parsed_events, timeline_);

  const FlameChartTimelineData& data = timeline_.timeline_data();
  ASSERT_THAT(data.groups, Not(IsEmpty()));
}

TEST_F(DataProviderTest, SyncProcessWithAsyncEventsRetainsOriginalTids) {
  const std::vector<TraceEvent> events = {{.ph = Phase::kComplete,
                                           .pid = 1,
                                           .tid = 101,
                                           .name = "AsyncOp",
                                           .ts = 100.0,
                                           .dur = 50.0,
                                           .is_async = true}};

  data_provider_.ProcessTraceEvents({events, {}}, timeline_);

  const FlameChartTimelineData& data = timeline_.timeline_data();
  ASSERT_THAT(data.entry_tids, Not(IsEmpty()));
  EXPECT_EQ(data.entry_tids[0], 101);
  EXPECT_LT(data.entry_tids[0], 0x80000000);
}

TEST_F(DataProviderTest, HloModuleDecorationChecksProcessId) {
  TraceEvent t1 = {.ph = Phase::kMetadata,
                   .pid = 1,
                   .tid = 101,
                   .name = std::string(kThreadName),
                   .args = {{"name", "XLA Modules"}}};
  TraceEvent t2 = {.ph = Phase::kMetadata,
                   .pid = 2,
                   .tid = 101,
                   .name = std::string(kThreadName),
                   .args = {{"name", "Ordinary Thread"}}};
  TraceEvent t3 = {.ph = Phase::kMetadata,
                   .pid = 2,
                   .tid = 202,
                   .name = std::string(kThreadName),
                   .args = {{"name", "Ordinary Thread 2"}}};

  TraceEvent p1 = {.ph = Phase::kMetadata,
                   .pid = 1,
                   .name = std::string(kProcessName),
                   .args = {{"name", "Process 1"}}};
  TraceEvent p2 = {.ph = Phase::kMetadata,
                   .pid = 2,
                   .name = std::string(kProcessName),
                   .args = {{"name", "Process 2"}}};

  TraceEvent m1 = {.ph = Phase::kComplete,
                   .pid = 1,
                   .tid = 101,
                   .name = "module: module1",
                   .ts = 100.0,
                   .dur = 50.0};
  TraceEvent e1 = {.ph = Phase::kComplete,
                   .pid = 2,
                   .tid = 101,
                   .name = "OrdinaryEvent",
                   .ts = 100.0,
                   .dur = 50.0};
  TraceEvent e2 = {.ph = Phase::kComplete,
                   .pid = 2,
                   .tid = 202,
                   .name = "EventE",
                   .ts = 120.0,
                   .dur = 10.0};

  ParsedTraceEvents parsed_events;
  parsed_events.flame_events = {t1, t2, t3, p1, p2, m1, e1, e2};

  data_provider_.ProcessTraceEvents(parsed_events, timeline_);

  const FlameChartTimelineData& data = timeline_.timeline_data();

  bool found = false;
  for (size_t i = 0; i < data.entry_names.size(); ++i) {
    if (absl::StrContains(data.entry_names[i], "EventE")) {
      found = true;
      EXPECT_EQ(data.entry_names[i], "EventE");
    }
  }
  EXPECT_TRUE(found);
}

TEST_F(DataProviderTest, HloModuleDecorationChecksDuration) {
  TraceEvent t1 = {.ph = Phase::kMetadata,
                   .pid = 1,
                   .tid = 101,
                   .name = std::string(kThreadName),
                   .args = {{"name", "XLA Modules"}}};
  TraceEvent t2 = {.ph = Phase::kMetadata,
                   .pid = 1,
                   .tid = 102,
                   .name = std::string(kThreadName),
                   .args = {{"name", "Ordinary Thread"}}};

  TraceEvent p1 = {.ph = Phase::kMetadata,
                   .pid = 1,
                   .name = std::string(kProcessName),
                   .args = {{"name", "Process 1"}}};

  TraceEvent m1 = {.ph = Phase::kComplete,
                   .pid = 1,
                   .tid = 101,
                   .name = "module: module1",
                   .ts = 100.0,
                   .dur = 50.0};
  TraceEvent e2 = {.ph = Phase::kComplete,
                   .pid = 1,
                   .tid = 102,
                   .name = "EventE",
                   .ts = 200.0,
                   .dur = 10.0};

  ParsedTraceEvents parsed_events;
  parsed_events.flame_events = {t1, t2, p1, m1, e2};

  data_provider_.ProcessTraceEvents(parsed_events, timeline_);

  const FlameChartTimelineData& data = timeline_.timeline_data();

  bool found = false;
  for (size_t i = 0; i < data.entry_names.size(); ++i) {
    if (absl::StrContains(data.entry_names[i], "EventE")) {
      found = true;
      EXPECT_EQ(data.entry_names[i], "EventE");
    }
  }
  EXPECT_TRUE(found);
}

TEST_F(DataProviderTest, AsyncProcessNamedAsyncXlaOpsIsTreatedAsAsync) {
  TraceEvent p1 =
      CreateMetadataEvent(std::string(kProcessName), 1, 0, "Async XLA Ops");

  TraceEvent e1 = {.ph = Phase::kComplete,
                   .pid = 1,
                   .tid = 101,
                   .name = "AsyncOp",
                   .ts = 100.0,
                   .dur = 50.0,
                   .is_async = true};

  ParsedTraceEvents parsed_events;
  parsed_events.flame_events = {p1, e1};

  data_provider_.ProcessTraceEvents(parsed_events, timeline_);

  const FlameChartTimelineData& data = timeline_.timeline_data();
  ASSERT_THAT(data.groups, SizeIs(testing::Ge(2)));
  EXPECT_EQ(data.groups[1].name, "AsyncOp");
}

TEST_F(DataProviderTest, ProcessLargeIds) {
  // IDs that would collide if truncated to 32-bit.
  const ProcessId pid = 1;
  const ThreadId tid1 = 0x100000001ULL;
  const ThreadId tid2 = 0x200000001ULL;

  const std::vector<TraceEvent> events = {{.ph = Phase::kComplete,
                                           .pid = pid,
                                           .tid = tid1,
                                           .name = "Event 1",
                                           .ts = 1000.0,
                                           .dur = 100.0},
                                          {.ph = Phase::kComplete,
                                           .pid = pid,
                                           .tid = tid2,
                                           .name = "Event 2",
                                           .ts = 1100.0,
                                           .dur = 100.0}};

  data_provider_.ProcessTraceEvents({events, {}}, timeline_);

  const FlameChartTimelineData& data = timeline_.timeline_data();

  // 1 process group + 2 thread groups = 3 groups total.
  // If IDs were truncated, they would be merged into 1 thread group.
  ASSERT_THAT(data.groups, SizeIs(3));

  EXPECT_EQ(data.groups[0].name, "Process_1");
  EXPECT_EQ(data.groups[1].name, "Thread_4294967297");
  EXPECT_EQ(data.groups[2].name, "Thread_8589934593");

  EXPECT_THAT(data.entry_tids, ElementsAre(tid1, tid2));
}

}  // namespace
}  // namespace traceviewer
