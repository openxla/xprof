#include "xprof/frontend/app/components/trace_viewer_v2/timeline/data_provider.h"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "testing/base/public/gmock.h"
#include "<gtest/gtest.h>"
#include "xprof/frontend/app/components/trace_viewer_v2/timeline/time_range.h"
#include "xprof/frontend/app/components/trace_viewer_v2/timeline/timeline.h"
#include "xprof/frontend/app/components/trace_viewer_v2/trace_helper/trace_event.h"

namespace traceviewer {

namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::SizeIs;

class DataProviderTest : public ::testing::Test {
 public:
  DataProviderTest() = default;

 protected:
  TraceEvent CreateMetadataEvent(std::string event_name, ProcessId pid,
                                 ThreadId tid,
                                 std::string thread_or_process_name) {
    return {Phase::kMetadata,
            pid,
            tid,
            std::move(event_name),
            0.0,
            0.0,
            {{std::string(kName), std::move(thread_or_process_name)}}};
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
      CreateMetadataEvent(std::string(kThreadName), 1, 101, "Thread A"),
      CreateMetadataEvent(std::string(kProcessName), 1, 0, "Process 1")};

  data_provider_.ProcessTraceEvents({events, {}}, timeline_);

  // Metadata alone doesn't create entries in timeline_data
  EXPECT_THAT(timeline_.timeline_data().groups, IsEmpty());
}

TEST_F(DataProviderTest, ProcessMetadataEventsWithEmptyName) {
  const std::vector<TraceEvent> events = {
      CreateMetadataEvent(std::string(kProcessName), 1, 0, ""),
      CreateMetadataEvent(std::string(kThreadName), 1, 101, ""),
      TraceEvent{Phase::kComplete, 1, 101, "Task A", 5000.0, 1000.0},
  };

  data_provider_.ProcessTraceEvents({events, {}}, timeline_);

  const FlameChartTimelineData& data = timeline_.timeline_data();

  ASSERT_THAT(data.groups, SizeIs(2));

  EXPECT_EQ(data.groups[0].name, "Process 1");
  EXPECT_EQ(data.groups[0].nesting_level, 0);
  EXPECT_EQ(data.groups[1].name, "Thread 101");
  EXPECT_EQ(data.groups[1].nesting_level, 1);
}

TEST_F(DataProviderTest, ProcessMetadataEventsWithNoNameArg) {
  const std::vector<TraceEvent> events = {
      TraceEvent{
          Phase::kMetadata, 1, 0, std::string(kProcessName), 0.0, 0.0, {}},
      TraceEvent{
          Phase::kMetadata, 1, 101, std::string(kThreadName), 0.0, 0.0, {}},
      TraceEvent{Phase::kComplete, 1, 101, "Task A", 5000.0, 1000.0},
  };

  data_provider_.ProcessTraceEvents({events, {}}, timeline_);

  const FlameChartTimelineData& data = timeline_.timeline_data();

  ASSERT_THAT(data.groups, SizeIs(2));

  EXPECT_EQ(data.groups[0].name, "Process 1");
  EXPECT_EQ(data.groups[0].nesting_level, 0);
  EXPECT_EQ(data.groups[1].name, "Thread 101");
  EXPECT_EQ(data.groups[1].nesting_level, 1);
}

TEST_F(DataProviderTest, ProcessCompleteEvents) {
  const std::vector<TraceEvent> events = {
      TraceEvent{Phase::kComplete, 1, 101, "Event 1", 1000.0, 200.0},
      TraceEvent{Phase::kComplete, 1, 102, "Event 2", 1100.0, 300.0}};

  data_provider_.ProcessTraceEvents({events, {}}, timeline_);

  const FlameChartTimelineData& data = timeline_.timeline_data();

  ASSERT_THAT(data.groups, SizeIs(3));

  EXPECT_EQ(data.groups[0].name, "Process 1");
  EXPECT_EQ(data.groups[0].start_level, 0);
  EXPECT_EQ(data.groups[0].nesting_level, 0);
  EXPECT_EQ(data.groups[1].name, "Thread 101");
  EXPECT_EQ(data.groups[1].start_level, 0);
  EXPECT_EQ(data.groups[1].nesting_level, 1);
  EXPECT_EQ(data.groups[2].name, "Thread 102");
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

TEST_F(DataProviderTest, ProcessMixedEvents) {
  const std::vector<TraceEvent> events = {
      CreateMetadataEvent(std::string(kProcessName), 1, 0, "Main Process"),
      CreateMetadataEvent(std::string(kThreadName), 1, 101, "Worker Thread"),
      TraceEvent{Phase::kComplete, 1, 101, "Task A", 5000.0, 1000.0},
      // No metadata for tid 102, uses default "Thread 102".
      TraceEvent{Phase::kComplete, 1, 102, "Task B", 5500.0, 1500.0},
      // No metadata for pid 2, uses default "Process 2".
      TraceEvent{Phase::kComplete, 2, 201, "Task C", 6000.0, 500.0}};

  data_provider_.ProcessTraceEvents({events, {}}, timeline_);

  const FlameChartTimelineData& data = timeline_.timeline_data();

  ASSERT_THAT(data.groups, SizeIs(5));

  EXPECT_EQ(data.groups[0].name, "Main Process");
  EXPECT_EQ(data.groups[0].nesting_level, 0);
  EXPECT_EQ(data.groups[1].name, "Worker Thread");
  EXPECT_EQ(data.groups[1].nesting_level, 1);
  EXPECT_EQ(data.groups[2].name, "Thread 102");
  EXPECT_EQ(data.groups[2].nesting_level, 1);
  EXPECT_EQ(data.groups[3].name, "Process 2");
  EXPECT_EQ(data.groups[3].nesting_level, 0);
  EXPECT_EQ(data.groups[4].name, "Thread 201");
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
      CreateMetadataEvent(std::string(kProcessName), 1, 0, "Process A"),
      CreateMetadataEvent(std::string(kThreadName), 1, 101, "Thread A1"),
      TraceEvent(Phase::kComplete, 1, 101, "Event A1", 1000.0, 100.0),
      CreateMetadataEvent(std::string(kProcessName), 2, 0, "Process B"),
      CreateMetadataEvent(std::string(kThreadName), 2, 201, "Thread B1"),
      TraceEvent(Phase::kComplete, 2, 201, "Event B1", 1200.0, 100.0),
      CreateMetadataEvent(std::string(kThreadName), 1, 102, "Thread A2"),
      TraceEvent(Phase::kComplete, 1, 102, "Event A2", 1100.0, 100.0),
  };

  data_provider_.ProcessTraceEvents({events, {}}, timeline_);

  const FlameChartTimelineData& data = timeline_.timeline_data();
  ASSERT_THAT(data.groups, SizeIs(5));

  // Process A
  EXPECT_EQ(data.groups[0].name, "Process A");
  EXPECT_EQ(data.groups[0].nesting_level, 0);
  EXPECT_EQ(data.groups[1].name, "Thread A1");
  EXPECT_EQ(data.groups[1].nesting_level, 1);
  EXPECT_EQ(data.groups[1].start_level, 0);
  EXPECT_EQ(data.groups[2].name, "Thread A2");
  EXPECT_EQ(data.groups[2].nesting_level, 1);
  EXPECT_EQ(data.groups[2].start_level, 1);

  // Process B
  EXPECT_EQ(data.groups[3].name, "Process B");
  EXPECT_EQ(data.groups[3].nesting_level, 0);
  EXPECT_EQ(data.groups[4].name, "Thread B1");
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

  EXPECT_EQ(data.groups[0].name, "Process 1");
  EXPECT_EQ(data.groups[0].nesting_level, 0);

  EXPECT_EQ(data.groups[1].name, "Counter A");
  EXPECT_EQ(data.groups[1].type, Group::Type::kCounter);
  EXPECT_EQ(data.groups[1].nesting_level, 1);

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

  data_provider_.ProcessTraceEvents(
      {{TraceEvent{Phase::kComplete, 1, 1, "Complete Event", 0.0, 10.0}},
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

  data_provider_.ProcessTraceEvents(
      {{TraceEvent{Phase::kComplete, 1, 1, "Complete Event", 0.0, 10.0}},
       {counter_event}},
      timeline_);

  const FlameChartTimelineData& data = timeline_.timeline_data();

  ASSERT_THAT(data.groups,
              SizeIs(3));  // Process group + Thread group + Counter group

  EXPECT_EQ(data.groups[0].name, "Process 1");
  EXPECT_EQ(data.groups[0].nesting_level, 0);

  EXPECT_EQ(data.groups[1].name, "Thread 1");
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

  data_provider_.ProcessTraceEvents(
      {{TraceEvent{Phase::kComplete, 2, 1, "Complete Event", 0.0, 10.0}},
       {counter_event}},
      timeline_);

  const FlameChartTimelineData& data = timeline_.timeline_data();

  ASSERT_THAT(data.groups,
              SizeIs(4));  // Process 1, Counter A, Process 2, Thread 1

  EXPECT_EQ(data.groups[0].name, "Process 1");
  EXPECT_EQ(data.groups[0].nesting_level, 0);

  EXPECT_EQ(data.groups[1].name, "Counter A");
  EXPECT_EQ(data.groups[1].type, Group::Type::kCounter);
  EXPECT_EQ(data.groups[1].nesting_level, 1);

  EXPECT_EQ(data.groups[2].name, "Process 2");
  EXPECT_EQ(data.groups[2].nesting_level, 0);

  EXPECT_EQ(data.groups[3].name, "Thread 1");
  EXPECT_EQ(data.groups[3].nesting_level, 1);

  ASSERT_TRUE(data.counter_data_by_group_index.count(1));
}

TEST_F(DataProviderTest, CounterTrackIncrementsLevel) {
  // Process 1: Thread 1 (1 level), Counter A
  // Process 2: Thread 2
  TraceEvent t1_event{Phase::kComplete, 1, 1, "Thread1Event", 0.0, 10.0};

  CounterEvent counter_event = CreateCounterEvent(1, "CounterA", {0.0}, {0.0});

  TraceEvent t2_event{Phase::kComplete, 2, 2, "Thread2Event", 0.0, 10.0};

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

  EXPECT_EQ(data.groups[1].name, "Thread 1");
  EXPECT_EQ(data.groups[1].start_level, 0);

  EXPECT_EQ(data.groups[2].name, "CounterA");
  EXPECT_EQ(data.groups[2].start_level, 1);

  EXPECT_EQ(data.groups[4].name, "Thread 2");
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
      CreateMetadataEvent(std::string(kProcessName), 1, 0, "Process 1"),
      TraceEvent{Phase::kMetadata,
                 1,
                 0,
                 "process_sort_index",
                 0.0,
                 0.0,
                 {{"sort_index", "2"}}},
      // Add a complete event for Process 1
      TraceEvent{Phase::kComplete, 1, 101, "Event 1", 0.0, 10.0},
      CreateMetadataEvent(std::string(kProcessName), 2, 0, "Process 2"),
      TraceEvent{Phase::kMetadata,
                 2,
                 0,
                 "process_sort_index",
                 0.0,
                 0.0,
                 {{"sort_index", "1"}}},
      // Add a complete event for Process 2
      TraceEvent{Phase::kComplete, 2, 201, "Event 2", 0.0, 10.0},
      CreateMetadataEvent(std::string(kProcessName), 3, 0, "Process 3"),
      // Process 3 has no sort index, defaults to pid (3)
      // Add a complete event for Process 3
      TraceEvent{Phase::kComplete, 3, 301, "Event 3", 0.0, 10.0},
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
  EXPECT_EQ(data.groups[0].name, "Process 2");
  EXPECT_EQ(data.groups[2].name, "Process 1");
  EXPECT_EQ(data.groups[4].name, "Process 3");
}

TEST_F(DataProviderTest, ProcessesSortedBySortIndexStable) {
  const std::vector<TraceEvent> events = {
      CreateMetadataEvent(std::string(kProcessName), 1, 0, "Process 1"),
      TraceEvent{Phase::kMetadata,
                 1,
                 0,
                 "process_sort_index",
                 0.0,
                 0.0,
                 {{"sort_index", "1"}}},
      // Add a complete event for Process 1
      TraceEvent{Phase::kComplete, 1, 101, "Event 1", 0.0, 10.0},
      CreateMetadataEvent(std::string(kProcessName), 2, 0, "Process 2"),
      TraceEvent{Phase::kMetadata,
                 2,
                 0,
                 "process_sort_index",
                 0.0,
                 0.0,
                 {{"sort_index", "1"}}},
      // Add a complete event for Process 2
      TraceEvent{Phase::kComplete, 2, 201, "Event 2", 0.0, 10.0},
  };

  data_provider_.ProcessTraceEvents({events, {}}, timeline_);

  const FlameChartTimelineData& data = timeline_.timeline_data();
  // 2 processes, each having 1 thread track -> 4 groups total.
  ASSERT_THAT(data.groups, SizeIs(4));

  // Stable sort: Process 1 (pid 1) comes before Process 2 (pid 2) as they have
  // same sort index.
  // Groups for Process 1 are at indices 0 (process) and 1 (thread)
  // Groups for Process 2 are at indices 2 (process) and 3 (thread)
  EXPECT_EQ(data.groups[0].name, "Process 1");
  EXPECT_EQ(data.groups[2].name, "Process 2");
}

TEST_F(DataProviderTest, MpmdPipelineViewEnabledPropagated) {
  ParsedTraceEvents events;
  events.mpmd_pipeline_view = true;
  // Add a dummy event to prevent early return
  events.flame_events.push_back(
      TraceEvent{Phase::kComplete, 1, 1, "Event", 0.0, 10.0});

  data_provider_.ProcessTraceEvents(events, timeline_);

  EXPECT_TRUE(timeline_.mpmd_pipeline_view_enabled());

  events.mpmd_pipeline_view = false;
  // Clear timeline data to process again (or just process again as it
  // overwrites)
  data_provider_.ProcessTraceEvents(events, timeline_);
  EXPECT_FALSE(timeline_.mpmd_pipeline_view_enabled());
}

TEST_F(DataProviderTest, ProcessTraceEventsWithFullTimespan) {
  const std::vector<TraceEvent> events = {
      TraceEvent{Phase::kComplete, 1, 1, "Event 1", 10.0, 10.0}};
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
  const std::vector<TraceEvent> events = {
      TraceEvent{Phase::kComplete, 1, 1, "Event 1", 10.0, 10.0}};
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
  const std::vector<TraceEvent> events = {
      TraceEvent{Phase::kComplete, 1, 1, "Event 1", 10.0, 10.0}};
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

TEST_F(DataProviderTest, ProcessTraceEventsPreservesVisibleRange) {
  // Initial load
  const std::vector<TraceEvent> events1 = {
      TraceEvent{Phase::kComplete, 1, 1, "Event 1", 1000.0, 100.0}};
  data_provider_.ProcessTraceEvents({events1, {}}, timeline_);

  // Set visible range to something specific (simulating zoom).
  timeline_.SetVisibleRange({1020.0, 1050.0});
  TimeRange visible_before = timeline_.visible_range();

  // Incremental load (new events, but within or related to current view)
  const std::vector<TraceEvent> events2 = {
      TraceEvent{Phase::kComplete, 1, 1, "Event 1", 1000.0, 100.0},
      TraceEvent{Phase::kComplete, 1, 1, "Event 2", 1200.0, 100.0}};

  data_provider_.ProcessTraceEvents({events2, {}}, timeline_);

  // Verify visible range is preserved.
  EXPECT_DOUBLE_EQ(timeline_.visible_range().start(), visible_before.start());
  EXPECT_DOUBLE_EQ(timeline_.visible_range().end(), visible_before.end());

  // Verify fetched data range is updated.
  EXPECT_DOUBLE_EQ(timeline_.fetched_data_time_range().start(), 1000.0);
  EXPECT_DOUBLE_EQ(timeline_.fetched_data_time_range().end(), 1300.0);
}

}  // namespace
}  // namespace traceviewer
