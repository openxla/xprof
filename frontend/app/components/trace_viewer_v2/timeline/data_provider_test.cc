#include "xprof/frontend/app/components/trace_viewer_v2/timeline/data_provider.h"

#include <string>
#include <utility>
#include <vector>

#include "xprof/frontend/app/components/trace_viewer_v2/timeline/time_range.h"
#include "xprof/frontend/app/components/trace_viewer_v2/timeline/timeline.h"
#include "xprof/frontend/app/components/trace_viewer_v2/trace_helper/trace_event.h"
#include "testing/base/public/gmock.h"
#include "<gtest/gtest.h>"

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

  Timeline timeline_;
  DataProvider data_provider_;
};

TEST_F(DataProviderTest, ProcessEmptyTraceData) {
  const std::vector<TraceEvent> events;

  data_provider_.ProcessTraceEvents(events, timeline_);

  EXPECT_THAT(timeline_.timeline_data().groups, IsEmpty());
  EXPECT_THAT(timeline_.timeline_data().entry_start_times, IsEmpty());
}

TEST_F(DataProviderTest, ProcessMetadataEvents) {
  const std::vector<TraceEvent> events = {
      CreateMetadataEvent(std::string(kThreadName), 1, 101, "Thread A"),
      CreateMetadataEvent(std::string(kProcessName), 1, 0, "Process 1")};

  data_provider_.ProcessTraceEvents(events, timeline_);

  // Metadata alone doesn't create entries in timeline_data
  EXPECT_THAT(timeline_.timeline_data().groups, IsEmpty());
}

TEST_F(DataProviderTest, ProcessMetadataEventsWithEmptyName) {
  const std::vector<TraceEvent> events = {
      CreateMetadataEvent(std::string(kProcessName), 1, 0, ""),
      CreateMetadataEvent(std::string(kThreadName), 1, 101, ""),
      TraceEvent{Phase::kComplete, 1, 101, "Task A", 5000.0, 1000.0},
  };

  data_provider_.ProcessTraceEvents(events, timeline_);

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

  data_provider_.ProcessTraceEvents(events, timeline_);

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

  data_provider_.ProcessTraceEvents(events, timeline_);

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

  data_provider_.ProcessTraceEvents(events, timeline_);

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

  data_provider_.ProcessTraceEvents(events, timeline_);

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

}  // namespace
}  // namespace traceviewer
