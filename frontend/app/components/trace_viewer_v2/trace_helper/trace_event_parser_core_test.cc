#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_event_parser_core.h"

#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "tsl/profiler/lib/context_types.h"
#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_event.h"
#include "plugin/xprof/protobuf/trace_data_response.pb.h"

namespace traceviewer {
namespace {

TEST(TraceEventParserCoreTest, ProcessMetadata) {
  // Input setup: A process with explicit sort_index=5 and a thread with
  // explicit sort_index=3.
  xprof::TraceDataResponse response;

  auto* process = response.mutable_metadata()->add_processes();
  process->set_id(1);
  process->set_name("Main Process");
  process->set_sort_index(5);

  auto* thread = process->add_threads();
  thread->set_id(10);
  thread->set_name("Worker Thread");
  thread->set_sort_index(3);

  ParsedTraceEvents result;
  ProcessMetadataEvents(response, result);

  // Expectation: Both process and thread emit their respective name and
  // sort_index metadata events.
  ASSERT_EQ(result.flame_events.size(), 4);

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

  EXPECT_EQ(result.flame_events[3].ph, Phase::kMetadata);
  EXPECT_EQ(result.flame_events[3].pid, 1);
  EXPECT_EQ(result.flame_events[3].tid, 10);
  EXPECT_EQ(result.flame_events[3].name, kThreadSortIndex);
  EXPECT_EQ(result.flame_events[3].args.at(std::string(kSortIndex)), "3");
}

TEST(TraceEventParserCoreTest, ProcessMetadataDefaultSortIndexEmitted) {
  // Input setup: Process and thread with sort_index explicitly set to 0.
  xprof::TraceDataResponse response;

  auto* process = response.mutable_metadata()->add_processes();
  process->set_id(2);
  process->set_name("Default Process");
  process->set_sort_index(0);

  auto* thread = process->add_threads();
  thread->set_id(20);
  thread->set_name("Default Thread");
  thread->set_sort_index(0);

  ParsedTraceEvents result;
  ProcessMetadataEvents(response, result);

  // Expectation: A sort_index of 0 is explicitly emitted as "0" rather than
  // omitted, allowing track ordering at index 0.
  ASSERT_EQ(result.flame_events.size(), 4);

  EXPECT_EQ(result.flame_events[0].ph, Phase::kMetadata);
  EXPECT_EQ(result.flame_events[0].pid, 2);
  EXPECT_EQ(result.flame_events[0].name, kProcessName);
  EXPECT_EQ(result.flame_events[0].args.at(std::string(kName)),
            "Default Process");

  EXPECT_EQ(result.flame_events[1].ph, Phase::kMetadata);
  EXPECT_EQ(result.flame_events[1].pid, 2);
  EXPECT_EQ(result.flame_events[1].name, kProcessSortIndex);
  EXPECT_EQ(result.flame_events[1].args.at(std::string(kSortIndex)), "0");

  EXPECT_EQ(result.flame_events[2].ph, Phase::kMetadata);
  EXPECT_EQ(result.flame_events[2].pid, 2);
  EXPECT_EQ(result.flame_events[2].tid, 20);
  EXPECT_EQ(result.flame_events[2].name, kThreadName);
  EXPECT_EQ(result.flame_events[2].args.at(std::string(kName)),
            "Default Thread");

  EXPECT_EQ(result.flame_events[3].ph, Phase::kMetadata);
  EXPECT_EQ(result.flame_events[3].pid, 2);
  EXPECT_EQ(result.flame_events[3].tid, 20);
  EXPECT_EQ(result.flame_events[3].name, kThreadSortIndex);
  EXPECT_EQ(result.flame_events[3].args.at(std::string(kSortIndex)), "0");
}

TEST(TraceEventParserCoreTest, ProcessMetadataUnsetSortIndexNotEmitted) {
  // Input setup: Process and thread with names but no sort_index set on proto.
  xprof::TraceDataResponse response;

  auto* process = response.mutable_metadata()->add_processes();
  process->set_id(2);
  process->set_name("Unindexed Process");

  auto* thread = process->add_threads();
  thread->set_id(20);
  thread->set_name("Unindexed Thread");

  ParsedTraceEvents result;
  ProcessMetadataEvents(response, result);

  // Expectation: Only name metadata events are emitted; no sort_index events.
  ASSERT_EQ(result.flame_events.size(), 2);

  EXPECT_EQ(result.flame_events[0].ph, Phase::kMetadata);
  EXPECT_EQ(result.flame_events[0].pid, 2);
  EXPECT_EQ(result.flame_events[0].name, kProcessName);
  EXPECT_EQ(result.flame_events[0].args.at(std::string(kName)),
            "Unindexed Process");

  EXPECT_EQ(result.flame_events[1].ph, Phase::kMetadata);
  EXPECT_EQ(result.flame_events[1].pid, 2);
  EXPECT_EQ(result.flame_events[1].tid, 20);
  EXPECT_EQ(result.flame_events[1].name, kThreadName);
  EXPECT_EQ(result.flame_events[1].args.at(std::string(kName)),
            "Unindexed Thread");
}

TEST(TraceEventParserCoreTest, ProcessMetadataMixedThreadsSortIndex) {
  // Input setup: Process with sort_index=10 and four threads with varying
  // sort_index configurations:
  // - Thread 100: sort_index=5
  // - Thread 200: sort_index=0
  // - Thread 300: sort_index=20
  // - Thread 400: sort_index unset
  xprof::TraceDataResponse response;

  auto* process = response.mutable_metadata()->add_processes();
  process->set_id(1);
  process->set_name("Process 1");
  process->set_sort_index(10);

  // Thread 1: sort_index > 0
  auto* thread1 = process->add_threads();
  thread1->set_id(100);
  thread1->set_name("Thread 100");
  thread1->set_sort_index(5);

  // Thread 2: sort_index == 0 (explicitly set)
  auto* thread2 = process->add_threads();
  thread2->set_id(200);
  thread2->set_name("Thread 200");
  thread2->set_sort_index(0);

  // Thread 3: sort_index > 0
  auto* thread3 = process->add_threads();
  thread3->set_id(300);
  thread3->set_name("Thread 300");
  thread3->set_sort_index(20);

  // Thread 4: sort_index NOT set
  auto* thread4 = process->add_threads();
  thread4->set_id(400);
  thread4->set_name("Thread 400");

  ParsedTraceEvents result;
  ProcessMetadataEvents(response, result);

  // Expectation: All four threads emit name events, but only threads with
  // explicit sort_index set (threads 100, 200, 300) emit sort_index events.
  ASSERT_EQ(result.flame_events.size(), 9);

  EXPECT_EQ(result.flame_events[0].ph, Phase::kMetadata);
  EXPECT_EQ(result.flame_events[0].pid, 1);
  EXPECT_EQ(result.flame_events[0].name, kProcessName);
  EXPECT_EQ(result.flame_events[0].args.at(std::string(kName)), "Process 1");

  EXPECT_EQ(result.flame_events[1].ph, Phase::kMetadata);
  EXPECT_EQ(result.flame_events[1].pid, 1);
  EXPECT_EQ(result.flame_events[1].name, kProcessSortIndex);
  EXPECT_EQ(result.flame_events[1].args.at(std::string(kSortIndex)), "10");

  EXPECT_EQ(result.flame_events[2].ph, Phase::kMetadata);
  EXPECT_EQ(result.flame_events[2].pid, 1);
  EXPECT_EQ(result.flame_events[2].tid, 100);
  EXPECT_EQ(result.flame_events[2].name, kThreadName);
  EXPECT_EQ(result.flame_events[2].args.at(std::string(kName)), "Thread 100");

  EXPECT_EQ(result.flame_events[3].ph, Phase::kMetadata);
  EXPECT_EQ(result.flame_events[3].pid, 1);
  EXPECT_EQ(result.flame_events[3].tid, 100);
  EXPECT_EQ(result.flame_events[3].name, kThreadSortIndex);
  EXPECT_EQ(result.flame_events[3].args.at(std::string(kSortIndex)), "5");

  EXPECT_EQ(result.flame_events[4].ph, Phase::kMetadata);
  EXPECT_EQ(result.flame_events[4].pid, 1);
  EXPECT_EQ(result.flame_events[4].tid, 200);
  EXPECT_EQ(result.flame_events[4].name, kThreadName);
  EXPECT_EQ(result.flame_events[4].args.at(std::string(kName)), "Thread 200");

  EXPECT_EQ(result.flame_events[5].ph, Phase::kMetadata);
  EXPECT_EQ(result.flame_events[5].pid, 1);
  EXPECT_EQ(result.flame_events[5].tid, 200);
  EXPECT_EQ(result.flame_events[5].name, kThreadSortIndex);
  EXPECT_EQ(result.flame_events[5].args.at(std::string(kSortIndex)), "0");

  EXPECT_EQ(result.flame_events[6].ph, Phase::kMetadata);
  EXPECT_EQ(result.flame_events[6].pid, 1);
  EXPECT_EQ(result.flame_events[6].tid, 300);
  EXPECT_EQ(result.flame_events[6].name, kThreadName);
  EXPECT_EQ(result.flame_events[6].args.at(std::string(kName)), "Thread 300");

  EXPECT_EQ(result.flame_events[7].ph, Phase::kMetadata);
  EXPECT_EQ(result.flame_events[7].pid, 1);
  EXPECT_EQ(result.flame_events[7].tid, 300);
  EXPECT_EQ(result.flame_events[7].name, kThreadSortIndex);
  EXPECT_EQ(result.flame_events[7].args.at(std::string(kSortIndex)), "20");

  EXPECT_EQ(result.flame_events[8].ph, Phase::kMetadata);
  EXPECT_EQ(result.flame_events[8].pid, 1);
  EXPECT_EQ(result.flame_events[8].tid, 400);
  EXPECT_EQ(result.flame_events[8].name, kThreadName);
  EXPECT_EQ(result.flame_events[8].args.at(std::string(kName)), "Thread 400");
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
  EXPECT_NE(result.flame_events[0].event_id, 0);
  EXPECT_EQ(result.flame_events[0].event_id,
            GenerateEventId("compute_op", 1.0, 5.0));

  EXPECT_EQ(result.flame_events[1].ph, Phase::kComplete);
  EXPECT_DOUBLE_EQ(result.flame_events[1].ts, 3.0);
  EXPECT_DOUBLE_EQ(result.flame_events[1].dur, 3.0);
  EXPECT_EQ(result.flame_events[1].args.at("uid"), "124");
  EXPECT_EQ(result.flame_events[1].id, "999");
  EXPECT_EQ(result.flame_events[1].category,
            tsl::profiler::ContextType::kTpuLaunch);
  EXPECT_NE(result.flame_events[1].event_id, 0);
  EXPECT_EQ(result.flame_events[1].event_id,
            GenerateEventId("compute_op", 3.0, 3.0));

  ASSERT_EQ(result.flow_events.size(), 1);
  EXPECT_EQ(result.flow_events[0].id, "999");
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
  EXPECT_NE(result.flame_events[0].event_id, 0);
  EXPECT_EQ(result.flame_events[0].event_id,
            GenerateEventId("async_op", 1.0, 5.0));
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
  EXPECT_NE(result.flame_events[0].event_id, 0);
  EXPECT_EQ(result.flame_events[0].event_id,
            GenerateEventId("dma_transfer", 1.0, 4.0));
}

TEST(TraceEventParserCoreTest, GenerateEventIdTest) {
  // Stable hashing: Identical inputs produce the same ID
  EventId id1 = GenerateEventId("event_a", 10.123456, 5.654321);
  EventId id2 = GenerateEventId("event_a", 10.123456, 5.654321);
  EXPECT_EQ(id1, id2);

  // Floating point precision normalization:
  // e.g. 10.123456 vs 10.123456000000001 (very close floats)
  // They should round to the same picosecond and yield the same ID.
  EventId id_approx =
      GenerateEventId("event_a", 10.123456000000001, 5.6543210000000004);
  EXPECT_EQ(id1, id_approx);

  // Different inputs produce different IDs
  EventId id_diff_name = GenerateEventId("event_b", 10.123456, 5.654321);
  EXPECT_NE(id1, id_diff_name);

  EventId id_diff_ts = GenerateEventId("event_a", 10.123457, 5.654321);
  EXPECT_NE(id1, id_diff_ts);

  EventId id_diff_dur = GenerateEventId("event_a", 10.123456, 5.654322);
  EXPECT_NE(id1, id_diff_dur);
}

TEST(TraceEventParserCoreTest, GenerateEventIdExactValues) {
  // Use dummy value 0ULL to trigger failure and capture exact fingerprint.
  EXPECT_EQ(GenerateEventId("test_event", 1.0, 2.0), 16168300061312540288ULL);
}


}  // namespace
}  // namespace traceviewer
