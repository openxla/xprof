#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_pb_event_parser.h"

#include <emscripten/bind.h>
#include <emscripten/em_asm.h>
#include <emscripten/val.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xprof/convert/trace_viewer/delta_series/zstd_compression.h"
#include "frontend/app/components/trace_viewer_v2/color/colors.h"
#include "frontend/app/components/trace_viewer_v2/timeline/data_provider.h"
#include "frontend/app/components/trace_viewer_v2/timeline/timeline.h"
#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_event_parser_core.h"
#include "plugin/xprof/protobuf/trace_data_response.pb.h"

namespace traceviewer {
class TracePbEventParserTest : public ::testing::Test {
 protected:
  void SetUp() override {
    EM_ASM({
      if (typeof global != 'undefined' &&
          typeof global.CustomEvent == 'undefined') {
        global.CustomEvent = function(type, params) {
          this.type = type;
          this.detail = params ? params.detail : null;
        };
      }
      if (typeof window == 'undefined') {
        global.window = {};
        global.window.testResults = {};
        global.window.listeners = {};
        global.window.addEventListener = function(type, listener) {
          global.window.listeners[type] = listener;
        };
        global.window.dispatchEvent = function(event) {
          if (global.window.listeners[event.type]) {
            global.window.listeners[event.type](event);
          }
        };
      } else {
        window.testResults = {};
        window.listeners = {};
        window.addEventListener = function(type, listener) {
          window.listeners[type] = listener;
        };
        window.dispatchEvent = function(event) {
          if (window.listeners[event.type]) {
            window.listeners[event.type](event);
          }
        };
      }
      window.addEventListener(
          'details_received', function(e) {
            window.testResults['details_received'] = {};
            window.testResults['details_received'].received = true;
            window.testResults['details_received'].details = e.detail.details;
          });
    });
  }

  ColorPalette palette_ = ColorPalette::Default();
  Timeline timeline_{palette_};
  DataProvider data_provider_;
};

TEST_F(TracePbEventParserTest, InvalidZstdBuffer) {
  const std::string invalid_zstd = "invalid zstd buffer content";
  const emscripten::val visible_range = emscripten::val::null();
  ParseAndProcessCompressedTraceEvents(
      reinterpret_cast<uintptr_t>(invalid_zstd.data()), invalid_zstd.size(),
      visible_range, data_provider_, timeline_);
  EXPECT_EQ(timeline_.data_time_range().duration(), 0);
}

TEST_F(TracePbEventParserTest, InvalidProtobufBuffer) {
  const std::string invalid_proto = "invalid protobuf buffer content";
  const std::string compressed_buffer =
      *tensorflow::profiler::ZstdCompression::Compress(invalid_proto);
  const emscripten::val visible_range = emscripten::val::null();
  ParseAndProcessCompressedTraceEvents(
      reinterpret_cast<uintptr_t>(compressed_buffer.data()),
      compressed_buffer.size(), visible_range, data_provider_, timeline_);
  EXPECT_EQ(timeline_.data_time_range().duration(), 0);
}

TEST_F(TracePbEventParserTest, ValidTraceDataWithDetailsAndTimespan) {
  xprof::TraceDataResponse response;
  response.set_full_timespan_start_ps(1000000000);  // 1 ms
  response.set_full_timespan_end_ps(5000000000);    // 5 ms

  auto* detail = response.add_details();
  detail->set_name("full_dma");
  detail->set_value(true);

  std::string serialized_proto;
  ASSERT_TRUE(response.SerializeToString(&serialized_proto));
  const std::string compressed_buffer =
      *tensorflow::profiler::ZstdCompression::Compress(serialized_proto);

  emscripten::val visible_range = emscripten::val::array();
  visible_range.call<void>("push", emscripten::val(2.0));
  visible_range.call<void>("push", emscripten::val(4.0));

  ParseAndProcessCompressedTraceEvents(
      reinterpret_cast<uintptr_t>(compressed_buffer.data()),
      compressed_buffer.size(), visible_range, data_provider_, timeline_);

  const emscripten::val results =
      emscripten::val::global("window")["testResults"]["details_received"];
  ASSERT_TRUE(results["received"].as<bool>());
  const emscripten::val details_map = results["details"];
  EXPECT_TRUE(details_map.call<bool>("get", emscripten::val("full_dma")));
}

TEST_F(TracePbEventParserTest, ValidTraceDataWithInvalidTimespan) {
  xprof::TraceDataResponse response;
  response.set_full_timespan_start_ps(5000000000);  // 5 ms
  response.set_full_timespan_end_ps(1000000000);    // 1 ms (start > end)

  std::string serialized_proto;
  ASSERT_TRUE(response.SerializeToString(&serialized_proto));
  const std::string compressed_buffer =
      *tensorflow::profiler::ZstdCompression::Compress(serialized_proto);

  const emscripten::val visible_range = emscripten::val::undefined();

  ParseAndProcessCompressedTraceEvents(
      reinterpret_cast<uintptr_t>(compressed_buffer.data()),
      compressed_buffer.size(), visible_range, data_provider_, timeline_);
}

TEST_F(TracePbEventParserTest, IncrementalLoadPreservesLastFetchRequestRange) {
  xprof::TraceDataResponse response;
  response.add_interned_strings("");
  response.add_interned_strings("op");
  response.set_full_timespan_start_ps(0);
  response.set_full_timespan_end_ps(10000000000ULL);  // 10 ms
  auto* series = response.add_complete_events();
  series->mutable_metadata()->set_process_id(1);
  series->mutable_metadata()->set_thread_id(1);
  series->add_deltas(1000000000ULL);     // 1 ms
  series->add_durations(1000000000ULL);  // 1 ms
  series->add_name_refs(1);
  series->add_event_metadata();

  std::string serialized_proto;
  ASSERT_TRUE(response.SerializeToString(&serialized_proto));
  const std::string compressed_buffer =
      *tensorflow::profiler::ZstdCompression::Compress(serialized_proto);

  // Initial load initializes last_fetch_request_range_ to full data_time_range_
  ParseAndProcessCompressedTraceEvents(
      reinterpret_cast<uintptr_t>(compressed_buffer.data()),
      compressed_buffer.size(), emscripten::val::undefined(), data_provider_,
      timeline_);
  EXPECT_EQ(timeline_.last_fetch_request_range(), TimeRange(0.0, 10000.0));

  // Simulate MaybeRequestData setting last_fetch_request_range_ to an expanded
  // fetch window [2000 us, 5000 us] (2.0 ms to 5.0 ms) before incremental load.
  timeline_.InitializeLastFetchRequestRange(TimeRange(3000.0, 4000.0));
  const TimeRange expected_fetch_range = timeline_.last_fetch_request_range();
  ASSERT_EQ(expected_fetch_range, TimeRange(2000.0, 5000.0));

  emscripten::val fetch_range_ms = emscripten::val::array();
  fetch_range_ms.call<void>("push", emscripten::val(2.0));
  fetch_range_ms.call<void>("push", emscripten::val(5.0));

  ParseAndProcessCompressedTraceEvents(
      reinterpret_cast<uintptr_t>(compressed_buffer.data()),
      compressed_buffer.size(), fetch_range_ms, data_provider_, timeline_);

  // Incremental load must not re-scale [2.0 ms, 5.0 ms] by kFetchRatio again.
  EXPECT_EQ(timeline_.last_fetch_request_range(), expected_fetch_range);
}

TEST_F(
    TracePbEventParserTest,
    InitialOverviewLoadCompactsLargeTraceAndSubWindowPreservesFullResolution) {
  xprof::TraceDataResponse response;
  response.add_interned_strings("");
  response.add_interned_strings("op");
  response.set_full_timespan_start_ps(0);
  response.set_full_timespan_end_ps(5000000000ULL);  // 5 ms
  auto* process = response.mutable_metadata()->add_processes();
  process->set_id(1);
  process->set_name("Host");
  auto* thread = process->add_threads();
  thread->set_id(1);
  thread->set_name("Thread 1");

  auto* series = response.add_complete_events();
  series->mutable_metadata()->set_process_id(1);
  series->mutable_metadata()->set_thread_id(1);
  constexpr int kNumEvents = 500;
  for (int i = 0; i < kNumEvents; ++i) {
    series->add_deltas(10000000ULL);    // +10 us each
    series->add_durations(5000000ULL);  // 5 us duration
    series->add_name_refs(1);
    series->add_event_metadata();
  }

  std::string serialized_proto;
  ASSERT_TRUE(response.SerializeToString(&serialized_proto));
  const std::string compressed_buffer =
      *tensorflow::profiler::ZstdCompression::Compress(serialized_proto);

  timeline_.set_max_uncompacted_overview_entries_for_test(64);
  timeline_.set_incremental_fetch_enabled(true);

  // 1. Initial overview load (no visible_range_from_url) configures
  // is_compacted(true) and compacts the 500 events down to MIP dominant events.
  ParseAndProcessCompressedTraceEvents(
      reinterpret_cast<uintptr_t>(compressed_buffer.data()),
      compressed_buffer.size(), emscripten::val::undefined(), data_provider_,
      timeline_);
  EXPECT_TRUE(timeline_.is_compacted());
  EXPECT_FALSE(timeline_.window_at_full_resolution());
  EXPECT_FALSE(timeline_.full_trace_loaded());
  EXPECT_LT(timeline_.timeline_data().entry_start_times.size(), 300u);

  // 2. Subsequent sub-window fetch (with visible_range_from_url) configures
  // is_compacted(false) and preserves all 500 events at full resolution.
  emscripten::val fetch_range_ms = emscripten::val::array();
  fetch_range_ms.call<void>("push", emscripten::val(0.0));
  fetch_range_ms.call<void>("push", emscripten::val(5.0));
  ParseAndProcessCompressedTraceEvents(
      reinterpret_cast<uintptr_t>(compressed_buffer.data()),
      compressed_buffer.size(), fetch_range_ms, data_provider_, timeline_);
  EXPECT_FALSE(timeline_.is_compacted());
  EXPECT_TRUE(timeline_.window_at_full_resolution());
  EXPECT_EQ(timeline_.timeline_data().entry_start_times.size(), 500u);
}

TEST_F(TracePbEventParserTest, SortIndicesAndAsyncFlowExclusion) {
  xprof::TraceDataResponse response;
  response.add_interned_strings("");
  response.add_interned_strings("EventA");
  response.add_interned_strings("EventB");
  response.add_interned_strings("AsyncOp");

  // Process 20 has sort_index 0; Process 10 has sort_index 1.
  xprof::Process* p10 = response.mutable_metadata()->add_processes();
  p10->set_id(10);
  p10->set_name("Process 10");
  p10->set_sort_index(1);
  xprof::Thread* t100 = p10->add_threads();
  t100->set_id(100);
  t100->set_name("Thread 100");
  t100->set_sort_index(1);
  xprof::Thread* t200 = p10->add_threads();
  t200->set_id(200);
  t200->set_name("Thread 200");
  t200->set_sort_index(0);

  xprof::Process* p20 = response.mutable_metadata()->add_processes();
  p20->set_id(20);
  p20->set_name("Process 20");
  p20->set_sort_index(0);
  xprof::Thread* t300 = p20->add_threads();
  t300->set_id(300);
  t300->set_name("Thread 300");

  xprof::TraceEventSeries* s1 = response.add_complete_events();
  s1->mutable_metadata()->set_process_id(10);
  s1->mutable_metadata()->set_thread_id(100);
  s1->add_deltas(1000000);
  s1->add_durations(500000);
  s1->add_name_refs(1);
  s1->add_event_metadata();

  xprof::TraceEventSeries* s2 = response.add_complete_events();
  s2->mutable_metadata()->set_process_id(10);
  s2->mutable_metadata()->set_thread_id(200);
  s2->add_deltas(1000000);
  s2->add_durations(500000);
  s2->add_name_refs(2);
  s2->add_event_metadata();

  xprof::TraceEventSeries* s3 = response.add_complete_events();
  s3->mutable_metadata()->set_process_id(20);
  s3->mutable_metadata()->set_thread_id(300);
  s3->add_deltas(1000000);
  s3->add_durations(500000);
  s3->add_name_refs(1);
  s3->add_event_metadata();

  // Add an async event with flow_id set; it should not be treated as a flow
  // line endpoint.
  xprof::TraceEventSeries* async_series = response.add_async_events();
  async_series->mutable_metadata()->set_process_id(20);
  async_series->mutable_metadata()->set_name_ref(3);
  async_series->add_deltas(1000000);
  async_series->add_durations(500000);
  xprof::TraceEventMetadata* async_meta = async_series->add_event_metadata();
  async_meta->set_flow_id(42);

  data_provider_.ProcessTraceEvents(response, timeline_, std::nullopt);

  EXPECT_TRUE(data_provider_.GetFlowCategories().empty());
  const FlameChartTimelineData& data = timeline_.timeline_data();
  // Process 20 (sort_index 0) must precede Process 10 (sort_index 1).
  // Within Process 10, Thread 200 (sort_index 0) must precede Thread 100
  // (sort_index 1).
  ASSERT_GE(data.groups.size(), 5);
  EXPECT_EQ(data.groups[0].name, "Process 20");
  int p10_idx = -1;
  int t200_idx = -1;
  int t100_idx = -1;
  for (size_t i = 0; i < data.groups.size(); ++i) {
    if (data.groups[i].name == "Process 10") p10_idx = static_cast<int>(i);
    if (data.groups[i].name == "Thread 200") t200_idx = static_cast<int>(i);
    if (data.groups[i].name == "Thread 100") t100_idx = static_cast<int>(i);
  }
  EXPECT_GT(p10_idx, 0);
  EXPECT_GT(t200_idx, p10_idx);
  EXPECT_GT(t100_idx, t200_idx);
}

TEST_F(TracePbEventParserTest,
       PrePairedAsyncEventsWithZeroFlowIdOrOmittedMetadataArePreserved) {
  xprof::TraceDataResponse response;
  response.add_interned_strings("");
  response.add_interned_strings("AsyncTrack");
  response.add_interned_strings("CustomAsyncEventName");

  xprof::Process* p1 = response.mutable_metadata()->add_processes();
  p1->set_id(1);
  p1->set_name("Process 1");

  // Series 1: pre-paired async event (dur > 0) with flow_id == 0 in metadata.
  xprof::TraceEventSeries* async_zero_flow = response.add_async_events();
  async_zero_flow->mutable_metadata()->set_process_id(1);
  async_zero_flow->mutable_metadata()->set_name_ref(1);
  async_zero_flow->add_deltas(1000000);
  async_zero_flow->add_durations(500000);
  xprof::TraceEventMetadata* meta = async_zero_flow->add_event_metadata();
  meta->set_flow_id(0);
  meta->set_serial(77);

  // Series 2: pre-paired async event (dur > 0) with omitted event_metadata
  // and per-event name_refs(i).
  xprof::TraceEventSeries* async_no_meta = response.add_async_events();
  async_no_meta->mutable_metadata()->set_process_id(1);
  async_no_meta->mutable_metadata()->set_name_ref(1);
  async_no_meta->add_deltas(2000000);
  async_no_meta->add_durations(400000);
  async_no_meta->add_name_refs(2);

  // Series 3: Begin/End async event pair (dur == 0) with flow_id != 0.
  xprof::TraceEventSeries* async_pair = response.add_async_events();
  async_pair->mutable_metadata()->set_process_id(1);
  async_pair->mutable_metadata()->set_name_ref(1);
  async_pair->add_deltas(3000000);
  async_pair->add_durations(0);
  xprof::TraceEventMetadata* begin_meta = async_pair->add_event_metadata();
  begin_meta->set_flow_id(99);
  async_pair->add_deltas(600000);
  async_pair->add_durations(0);
  xprof::TraceEventMetadata* end_meta = async_pair->add_event_metadata();
  end_meta->set_flow_id(99);
  end_meta->set_serial(88);

  data_provider_.ProcessTraceEvents(response, timeline_, std::nullopt);

  const FlameChartTimelineData& data = timeline_.timeline_data();
  ASSERT_EQ(data.entry_start_times.size(), 3u);
  // "AsyncTrack" events (Series 1 at 1.0 us and Series 3 at 3.0 us) precede
  // "CustomAsyncEventName" (Series 2 at 2.0 us) in async_groups btree_map.
  EXPECT_DOUBLE_EQ(data.entry_start_times[0], 1.0);
  EXPECT_DOUBLE_EQ(data.entry_total_times[0], 0.5);
  EXPECT_EQ(data.non_zero_uids_by_event_index.at(0), 77u);
  EXPECT_DOUBLE_EQ(data.entry_start_times[1], 3.0);
  EXPECT_DOUBLE_EQ(data.entry_total_times[1], 0.6);
  EXPECT_EQ(data.non_zero_uids_by_event_index.at(1), 88u);
  EXPECT_DOUBLE_EQ(data.entry_start_times[2], 2.0);
  EXPECT_DOUBLE_EQ(data.entry_total_times[2], 0.4);
  EXPECT_EQ(data.entry_name_indices[2], 2u);
}

TEST_F(TracePbEventParserTest, EarlyReturnsResetIncrementalLoading) {
  const emscripten::val visible_range = emscripten::val::null();

  timeline_.set_is_incremental_loading(true);
  ParseAndProcessCompressedTraceEvents(0, 0, visible_range, data_provider_,
                                       timeline_);
  EXPECT_FALSE(timeline_.is_incremental_loading());

  timeline_.set_is_incremental_loading(true);
  const std::string invalid_zstd = "not zstd";
  ParseAndProcessCompressedTraceEvents(
      reinterpret_cast<uintptr_t>(invalid_zstd.data()), invalid_zstd.size(),
      visible_range, data_provider_, timeline_);
  EXPECT_FALSE(timeline_.is_incremental_loading());
}

TEST_F(TracePbEventParserTest, ProcessCompleteEventsDecodesRleGroupIds) {
  xprof::TraceDataResponse response;
  response.add_interned_strings("");
  response.add_interned_strings("StepOpA");
  response.add_interned_strings("StepOpB");

  xprof::TraceEventSeries* s = response.add_complete_events();
  s->mutable_metadata()->set_process_id(1);
  s->mutable_metadata()->set_thread_id(10);
  s->add_deltas(1000000);
  s->add_durations(500000);
  s->add_name_refs(1);
  s->add_deltas(1000000);
  s->add_durations(500000);
  s->add_name_refs(2);
  s->add_group_id_run_values(7);
  s->add_group_id_run_lengths(1);
  s->add_group_id_run_values(9);
  s->add_group_id_run_lengths(1);

  ParsedTraceEvents parsed;
  ProcessCompleteEvents(response, parsed);
  ASSERT_EQ(parsed.flame_events.size(), 2u);
  EXPECT_TRUE(parsed.flame_events[0].has_group_id);
  EXPECT_EQ(parsed.flame_events[0].group_id, 7);
  EXPECT_TRUE(parsed.flame_events[1].has_group_id);
  EXPECT_EQ(parsed.flame_events[1].group_id, 9);
}

}  // namespace traceviewer
