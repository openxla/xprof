#include "xprof/frontend/app/components/trace_viewer_v2/timeline/data_provider.h"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "testing/base/public/gmock.h"
#include "<gtest/gtest.h>"
#include "absl/algorithm/container.h"
#include "tsl/profiler/lib/context_types.h"
#include "xprof/frontend/app/components/trace_viewer_v2/timeline/time_range.h"
#include "xprof/frontend/app/components/trace_viewer_v2/timeline/timeline.h"
#include "xprof/frontend/app/components/trace_viewer_v2/trace_helper/trace_event.h"

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

  EXPECT_EQ(data.groups[0].name, "Process 1");
  EXPECT_EQ(data.groups[0].nesting_level, 0);
  EXPECT_EQ(data.groups[1].name, "Thread 101");
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

  EXPECT_EQ(data.groups[0].name, "Process 1");
  EXPECT_EQ(data.groups[0].nesting_level, 0);
  EXPECT_EQ(data.groups[1].name, "Thread 101");
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

  EXPECT_EQ(data.groups[0].name, "Process 1");
  EXPECT_EQ(data.groups[0].start_level, 0);
  EXPECT_EQ(data.groups[0].nesting_level, 0);
  EXPECT_EQ(data.groups[1].name, "Thread 101");
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
      CreateMetadataEvent(std::string(kProcessName), 1, 0, "Main Process"),
      CreateMetadataEvent(std::string(kThreadName), 1, 101, "Worker Thread"),
      {.ph = Phase::kComplete,
       .pid = 1,
       .tid = 101,
       .name = "Task A",
       .ts = 5000.0,
       .dur = 1000.0,
       .id = "",
       .args = {}},
      // No metadata for tid 102, uses default "Thread 102".
      {.ph = Phase::kComplete,
       .pid = 1,
       .tid = 102,
       .name = "Task B",
       .ts = 5500.0,
       .dur = 1500.0,
       .id = "",
       .args = {}},
      // No metadata for pid 2, uses default "Process 2".
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
      {.ph = Phase::kComplete,
       .pid = 1,
       .tid = 101,
       .name = "Event A1",
       .ts = 1000.0,
       .dur = 100.0,
       .id = "",
       .args = {}},
      CreateMetadataEvent(std::string(kProcessName), 2, 0, "Process B"),
      CreateMetadataEvent(std::string(kThreadName), 2, 201, "Thread B1"),
      {.ph = Phase::kComplete,
       .pid = 2,
       .tid = 201,
       .name = "Event B1",
       .ts = 1200.0,
       .dur = 100.0,
       .id = "",
       .args = {}},
      CreateMetadataEvent(std::string(kThreadName), 1, 102, "Thread A2"),
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
      CreateMetadataEvent(std::string(kProcessName), 2, 0, "Process 2"),
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
      CreateMetadataEvent(std::string(kProcessName), 3, 0, "Process 3"),
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
  EXPECT_EQ(data.groups[0].name, "Process 2");
  EXPECT_EQ(data.groups[2].name, "Process 1");
  EXPECT_EQ(data.groups[4].name, "Process 3");
}

TEST_F(DataProviderTest, ProcessesSortedBySortIndexStable) {
  const std::vector<TraceEvent> events = {
      CreateMetadataEvent(std::string(kProcessName), 1, 0, "Process 1"),
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
      CreateMetadataEvent(std::string(kProcessName), 2, 0, "Process 2"),
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
  EXPECT_EQ(data.groups[0].name, "Process 1");
  EXPECT_EQ(data.groups[2].name, "Process 2");
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
  // Flow 1, part 1
  EXPECT_EQ(data.flow_lines[0].source_ts, 120.0);
  EXPECT_EQ(data.flow_lines[0].target_ts, 210.0);
  EXPECT_EQ(data.flow_lines[0].source_level, 0);
  EXPECT_EQ(data.flow_lines[0].target_level, 1);
  EXPECT_EQ(data.flow_lines[0].category,
            tsl::profiler::ContextType::kGpuLaunch);

  // Flow 1, part 2
  EXPECT_EQ(data.flow_lines[1].source_ts, 210.0);
  EXPECT_EQ(data.flow_lines[1].target_ts, 230.0);
  EXPECT_EQ(data.flow_lines[1].source_level, 1);
  EXPECT_EQ(data.flow_lines[1].target_level, 1);
  EXPECT_EQ(data.flow_lines[1].category,
            tsl::profiler::ContextType::kGpuLaunch);

  // Flow 2
  EXPECT_EQ(data.flow_lines[2].source_ts, 310.0);
  EXPECT_EQ(data.flow_lines[2].target_ts, 380.0);
  EXPECT_EQ(data.flow_lines[2].source_level, 2);
  EXPECT_EQ(data.flow_lines[2].target_level, 2);
  EXPECT_EQ(data.flow_lines[2].category, tsl::profiler::ContextType::kGeneric);

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

}  // namespace
}  // namespace traceviewer
