#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_event_parser_core.h"

#include <string>
#include <vector>

#include "<gtest/gtest.h>"
#include "tsl/profiler/lib/context_types.h"
#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_event.h"
#include "plugin/xprof/protobuf/trace_data_response.pb.h"

namespace traceviewer {
namespace {

TEST(TraceEventParserCoreTest, ProcessMetadata) {
  xprof::TraceDataResponse response;

  auto* process = response.mutable_metadata()->add_processes();
  process->set_id(1);
  process->set_name("Main Process");
  process->set_sort_index(5);

  auto* thread = process->add_threads();
  thread->set_id(10);
  thread->set_name("Worker Thread");

  ParsedTraceEvents result;
  ProcessMetadataEvents(response, result);

  ASSERT_EQ(result.flame_events.size(), 3);

  EXPECT_EQ(result.flame_events[0].ph, Phase::kMetadata);
  EXPECT_EQ(result.flame_events[0].pid, 1);
  EXPECT_EQ(result.flame_events[0].name, kProcessName);
  EXPECT_EQ(result.flame_events[0].args.at(std::string(kName)), "Main Process");

  EXPECT_EQ(result.flame_events[1].ph, Phase::kMetadata);
  EXPECT_EQ(result.flame_events[1].pid, 1);
  EXPECT_EQ(result.flame_events[1].name, kProcessSortIndex);
  EXPECT_EQ(result.flame_events[1].args.at(std::string(kSortIndex)), "5");

  EXPECT_EQ(result.flame_events[2].ph, Phase::kMetadata);
  EXPECT_EQ(result.flame_events[2].pid, 1);
  EXPECT_EQ(result.flame_events[2].tid, 10);
  EXPECT_EQ(result.flame_events[2].name, kThreadName);
  EXPECT_EQ(result.flame_events[2].args.at(std::string(kName)),
            "Worker Thread");
}

TEST(TraceEventParserCoreTest, ProcessCompleteEvents) {
  xprof::TraceDataResponse response;
  response.add_interned_strings("");            // index 0
  response.add_interned_strings("compute_op");  // index 1
  response.add_interned_strings("Tpu Launch");  // index 2

  auto* series = response.add_complete_events();
  series->mutable_metadata()->set_process_id(1);
  series->mutable_metadata()->set_thread_id(2);

  series->add_deltas(1000000);     // 1 us
  series->add_durations(5000000);  // 5 us
  series->add_name_refs(1);        // "compute_op"
  auto* meta1 = series->add_event_metadata();
  meta1->set_serial(123);

  series->add_deltas(2000000);     // +2 us = 3 us absolute
  series->add_durations(3000000);  // 3 us
  series->add_name_refs(1);
  auto* meta2 = series->add_event_metadata();
  meta2->set_flow_id(999);
  meta2->set_flow_category(2);  // "TpuLaunch"
  meta2->set_serial(124);

  ParsedTraceEvents result;
  ProcessCompleteEvents(response, result);

  ASSERT_EQ(result.flame_events.size(), 2);

  EXPECT_EQ(result.flame_events[0].ph, Phase::kComplete);
  EXPECT_EQ(result.flame_events[0].pid, 1);
  EXPECT_EQ(result.flame_events[0].tid, 2);
  EXPECT_DOUBLE_EQ(result.flame_events[0].ts, 1.0);
  EXPECT_DOUBLE_EQ(result.flame_events[0].dur, 5.0);
  EXPECT_EQ(result.flame_events[0].name, "compute_op");
  EXPECT_EQ(result.flame_events[0].args.at("uid"), "123");

  EXPECT_EQ(result.flame_events[1].ph, Phase::kComplete);
  EXPECT_DOUBLE_EQ(result.flame_events[1].ts, 3.0);
  EXPECT_DOUBLE_EQ(result.flame_events[1].dur, 3.0);
  EXPECT_EQ(result.flame_events[1].args.at("uid"), "124");
  EXPECT_EQ(result.flame_events[1].flow_id, 999);
  EXPECT_EQ(result.flame_events[1].category,
            tsl::profiler::ContextType::kTpuLaunch);
}

TEST(TraceEventParserCoreTest, ProcessCounterEvents) {
  xprof::TraceDataResponse response;
  response.add_interned_strings("MemoryUsage");

  auto* series = response.add_counter_events();
  series->mutable_metadata()->set_process_id(1);
  series->mutable_metadata()->set_name_ref(0);

  series->add_deltas(1000000);  // 1 us
  series->add_event_metadata()->set_counter_value_double(100.5);

  series->add_deltas(2000000);  // +2 us = 3 us
  series->add_event_metadata()->set_counter_value_double(50.2);

  ParsedTraceEvents result;
  ProcessCounterEvents(response, result);

  ASSERT_EQ(result.counter_events.size(), 1);
  const auto& counter = result.counter_events[0];
  EXPECT_EQ(counter.pid, 1);
  EXPECT_EQ(counter.name, "MemoryUsage");

  ASSERT_EQ(counter.timestamps.size(), 2);
  EXPECT_DOUBLE_EQ(counter.timestamps[0], 1.0);
  EXPECT_DOUBLE_EQ(counter.timestamps[1], 3.0);

  ASSERT_EQ(counter.values.size(), 2);
  EXPECT_DOUBLE_EQ(counter.values[0], 100.5);
  EXPECT_DOUBLE_EQ(counter.values[1], 50.2);

  EXPECT_DOUBLE_EQ(counter.min_value, 50.2);
  EXPECT_DOUBLE_EQ(counter.max_value, 100.5);
}

TEST(TraceEventParserCoreTest, ProcessAsyncEventsWithDuration) {
  xprof::TraceDataResponse response;
  response.add_interned_strings("async_op");

  auto* series = response.add_async_events();
  series->mutable_metadata()->set_name_ref(0);
  series->mutable_metadata()->set_process_id(1);

  series->add_deltas(1000000);     // 1 us
  series->add_durations(5000000);  // 5 us

  auto* meta = series->add_event_metadata();
  meta->set_flow_id(500);
  meta->set_serial(42);
  meta->set_group_id(99);

  ParsedTraceEvents result;
  ProcessAsyncEvents(response, result);

  ASSERT_EQ(result.flame_events.size(), 1);
  EXPECT_TRUE(result.flame_events[0].is_async);
  EXPECT_EQ(result.flame_events[0].ph, Phase::kComplete);
  EXPECT_EQ(result.flame_events[0].pid, 1);
  EXPECT_DOUBLE_EQ(result.flame_events[0].ts, 1.0);
  EXPECT_DOUBLE_EQ(result.flame_events[0].dur, 5.0);
  EXPECT_EQ(result.flame_events[0].name, "async_op");
  EXPECT_EQ(result.flame_events[0].args.at("uid"), "42");
  EXPECT_EQ(result.flame_events[0].args.at("group_id"), "99");
}

TEST(TraceEventParserCoreTest, ProcessAsyncEventsBeginEndPair) {
  xprof::TraceDataResponse response;
  response.add_interned_strings("dma_transfer");

  auto* series = response.add_async_events();
  series->mutable_metadata()->set_name_ref(0);
  series->mutable_metadata()->set_process_id(1);

  series->add_deltas(1000000);  // 1 us
  series->add_durations(0);     // 0 duration -> part of Pair
  auto* meta1 = series->add_event_metadata();
  meta1->set_flow_id(777);
  meta1->set_serial(101);

  series->add_deltas(4000000);  // +4 us = 5 us absolute
  series->add_durations(0);
  auto* meta2 = series->add_event_metadata();
  meta2->set_flow_id(777);
  meta2->set_serial(102);

  ParsedTraceEvents result;
  ProcessAsyncEvents(response, result);

  ASSERT_EQ(result.flame_events.size(), 1);
  EXPECT_TRUE(result.flame_events[0].is_async);
  EXPECT_EQ(result.flame_events[0].ph, Phase::kComplete);
  EXPECT_DOUBLE_EQ(result.flame_events[0].ts, 1.0);
  EXPECT_DOUBLE_EQ(result.flame_events[0].dur, 4.0);
  EXPECT_EQ(result.flame_events[0].name, "dma_transfer");
  EXPECT_EQ(result.flame_events[0].args.at("uid"), "101");
}

}  // namespace
}  // namespace traceviewer
