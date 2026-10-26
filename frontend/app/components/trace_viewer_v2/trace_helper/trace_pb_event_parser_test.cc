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

TEST_F(TracePbEventParserTest,
       MultiChunkAsyncEventsPreservedAcrossCompressedChunks) {
  // Chunk 1: Async Begin event (duration = 0, flow_id = 42) at 1 ms
  xprof::TraceDataResponse chunk1;
  chunk1.add_interned_strings("AsyncDMA");
  auto* process = chunk1.mutable_metadata()->add_processes();
  process->set_id(1);
  process->set_name("TPU:0");
  auto* series1 = chunk1.add_async_events();
  series1->mutable_metadata()->set_process_id(1);
  series1->mutable_metadata()->set_name_ref(0);
  series1->add_deltas(1000000000ULL);  // 1000 us
  series1->add_durations(0);
  series1->add_event_metadata()->set_flow_id(42);

  std::string serialized1;
  ASSERT_TRUE(chunk1.SerializeToString(&serialized1));
  const std::string compressed1 =
      *tensorflow::profiler::ZstdCompression::Compress(serialized1);

  ParseAndProcessCompressedTraceEvents(
      reinterpret_cast<uintptr_t>(compressed1.data()), compressed1.size(),
      emscripten::val::null(), data_provider_, timeline_);
  EXPECT_EQ(data_provider_.open_async_events().size(), 1u);

  // Chunk 2: Matching Async End event (duration = 0, flow_id = 42) at 3 ms
  xprof::TraceDataResponse chunk2;
  chunk2.add_interned_strings("AsyncDMA");
  auto* series2 = chunk2.add_async_events();
  series2->mutable_metadata()->set_process_id(1);
  series2->mutable_metadata()->set_name_ref(0);
  series2->add_deltas(3000000000ULL);  // 3000 us
  series2->add_durations(0);
  series2->add_event_metadata()->set_flow_id(42);

  std::string serialized2;
  ASSERT_TRUE(chunk2.SerializeToString(&serialized2));
  const std::string compressed2 =
      *tensorflow::profiler::ZstdCompression::Compress(serialized2);

  ParseAndProcessCompressedTraceEvents(
      reinterpret_cast<uintptr_t>(compressed2.data()), compressed2.size(),
      emscripten::val::null(), data_provider_, timeline_);
  EXPECT_TRUE(data_provider_.open_async_events().empty());
  ASSERT_EQ(timeline_.timeline_data().entry_names.size(), 1u);
  EXPECT_EQ(timeline_.timeline_data().entry_names[0], "AsyncDMA");
  EXPECT_DOUBLE_EQ(timeline_.timeline_data().entry_start_times[0], 1000.0);
  EXPECT_DOUBLE_EQ(timeline_.timeline_data().entry_total_times[0], 2000.0);
}

}  // namespace traceviewer
