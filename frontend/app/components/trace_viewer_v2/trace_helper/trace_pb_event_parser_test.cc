#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_pb_event_parser.h"

#include <emscripten/bind.h>
#include <emscripten/emscripten.h>
#include <emscripten/val.h>

#include <cstdint>
#include <string>

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

TEST_F(TracePbEventParserTest, IncrementalTraceEventsXchnMultiFrame) {
  auto append_framed = [](std::string& out,
                          const xprof::TraceDataResponse& frame) {
    std::string serialized;
    ASSERT_TRUE(frame.SerializeToString(&serialized));
    const std::string compressed =
        *tensorflow::profiler::ZstdCompression::Compress(serialized);
    const uint32_t len = static_cast<uint32_t>(compressed.size());
    const char len_bytes[4] = {
        static_cast<char>(len & 0xff),
        static_cast<char>((len >> 8) & 0xff),
        static_cast<char>((len >> 16) & 0xff),
        static_cast<char>((len >> 24) & 0xff),
    };
    out.append(len_bytes, 4);
    out.append(compressed);
  };

  xprof::TraceDataResponse frame1;
  frame1.set_full_timespan_start_ps(1000000000);
  frame1.set_full_timespan_end_ps(5000000000);
  frame1.add_interned_strings("");
  frame1.add_interned_strings("op_a");
  auto* proc1 = frame1.mutable_metadata()->add_processes();
  proc1->set_id(1);
  proc1->set_name("host:0 /device:TPU:0");
  auto* th1 = proc1->add_threads();
  th1->set_id(10);
  th1->set_name("XLA Ops");
  auto* series1 = frame1.add_complete_events();
  series1->mutable_metadata()->set_process_id(1);
  series1->mutable_metadata()->set_thread_id(10);
  series1->add_deltas(1000000000);
  series1->add_durations(500000000);
  series1->add_name_refs(1);
  series1->add_event_metadata();

  xprof::TraceDataResponse frame2;
  frame2.add_interned_strings("op_b");
  auto* proc2 = frame2.mutable_metadata()->add_processes();
  proc2->set_id(2);
  proc2->set_name("host:0 /device:TPU:1");
  auto* th2 = proc2->add_threads();
  th2->set_id(20);
  th2->set_name("XLA Ops");
  auto* series2 = frame2.add_complete_events();
  series2->mutable_metadata()->set_process_id(2);
  series2->mutable_metadata()->set_thread_id(20);
  series2->add_deltas(2000000000);
  series2->add_durations(500000000);
  series2->add_name_refs(2);
  series2->add_event_metadata();

  std::string xchn_payload(tensorflow::profiler::kXChunkMagic);
  append_framed(xchn_payload, frame1);
  append_framed(xchn_payload, frame2);
  xchn_payload.append("\0\0\0\0", 4);

  const emscripten::val visible_range = emscripten::val::undefined();
  BeginIncrementalTraceEvents(visible_range, data_provider_, timeline_);
  AppendCompressedTraceEventsIncremental(
      reinterpret_cast<uintptr_t>(xchn_payload.data()), xchn_payload.size());

  EXPECT_TRUE(StepIncrementalTraceEvents(1, data_provider_, timeline_));
  EXPECT_EQ(timeline_.timeline_data().entry_names.size(), 1u);
  EXPECT_EQ(timeline_.timeline_data().entry_names[0], "op_a");

  EXPECT_FALSE(StepIncrementalTraceEvents(1, data_provider_, timeline_));
  EXPECT_EQ(timeline_.timeline_data().entry_names.size(), 2u);
  EXPECT_EQ(timeline_.timeline_data().entry_names[1], "op_b");

  FinishIncrementalTraceEvents(data_provider_, timeline_);
  EXPECT_EQ(timeline_.data_time_range().duration(), 4000.0);
  EXPECT_TRUE(timeline_.full_trace_loaded());
}

TEST_F(TracePbEventParserTest,
       IncrementalTraceEventsXchnOpenSpanExtensionAcrossFrames) {
  constexpr uint64_t kDurationPhaseBeginBit = 1ULL << 61;
  constexpr uint64_t kDurationPhaseEndBit = 1ULL << 62;

  auto append_framed = [](std::string& out,
                          const xprof::TraceDataResponse& frame) {
    std::string serialized;
    ASSERT_TRUE(frame.SerializeToString(&serialized));
    const std::string compressed =
        *tensorflow::profiler::ZstdCompression::Compress(serialized);
    const uint32_t len = static_cast<uint32_t>(compressed.size());
    const char len_bytes[4] = {
        static_cast<char>(len & 0xff),
        static_cast<char>((len >> 8) & 0xff),
        static_cast<char>((len >> 16) & 0xff),
        static_cast<char>((len >> 24) & 0xff),
    };
    out.append(len_bytes, 4);
    out.append(compressed);
  };

  // Frame 1: while_loop starts at 1000 us with partial duration 500 us (tagged
  // with kDurationPhaseBeginBit), and child_1 runs [1100, 1300] us (200 us).
  xprof::TraceDataResponse frame1;
  frame1.add_interned_strings("");
  frame1.add_interned_strings("while_loop");
  frame1.add_interned_strings("child_1");
  auto* proc1 = frame1.mutable_metadata()->add_processes();
  proc1->set_id(1);
  proc1->set_name("host:0 /device:TPU:0");
  auto* th1 = proc1->add_threads();
  th1->set_id(10);
  th1->set_name("XLA Ops");
  auto* series1 = frame1.add_complete_events();
  series1->mutable_metadata()->set_process_id(1);
  series1->mutable_metadata()->set_thread_id(10);
  series1->add_deltas(1000000000ULL);
  series1->add_durations(500000000ULL | kDurationPhaseBeginBit);
  series1->add_name_refs(1);
  series1->add_event_metadata();
  series1->add_deltas(100000000ULL);
  series1->add_durations(200000000ULL);
  series1->add_name_refs(2);
  series1->add_event_metadata();

  // Frame 2: while_loop closes at 2500 us with full duration 1500 us (tagged
  // with kDurationPhaseEndBit), and child_2 runs [1600, 1900] us (300 us).
  xprof::TraceDataResponse frame2;
  frame2.add_interned_strings("child_2");
  auto* series2 = frame2.add_complete_events();
  series2->mutable_metadata()->set_process_id(1);
  series2->mutable_metadata()->set_thread_id(10);
  series2->add_deltas(1000000000ULL);
  series2->add_durations(1500000000ULL | kDurationPhaseEndBit);
  series2->add_name_refs(1);
  series2->add_event_metadata();
  series2->add_deltas(600000000ULL);
  series2->add_durations(300000000ULL);
  series2->add_name_refs(3);
  series2->add_event_metadata();

  std::string xchn_payload(tensorflow::profiler::kXChunkMagic);
  append_framed(xchn_payload, frame1);
  append_framed(xchn_payload, frame2);
  xchn_payload.append("\0\0\0\0", 4);

  const emscripten::val visible_range = emscripten::val::undefined();
  BeginIncrementalTraceEvents(visible_range, data_provider_, timeline_);
  AppendCompressedTraceEventsIncremental(
      reinterpret_cast<uintptr_t>(xchn_payload.data()), xchn_payload.size());

  EXPECT_TRUE(StepIncrementalTraceEvents(1, data_provider_, timeline_));
  ASSERT_EQ(timeline_.timeline_data().entry_names.size(), 2u);
  EXPECT_DOUBLE_EQ(timeline_.timeline_data().entry_total_times[0], 500.0);
  EXPECT_DOUBLE_EQ(timeline_.timeline_data().entry_self_times[0], 300.0);

  EXPECT_FALSE(StepIncrementalTraceEvents(1, data_provider_, timeline_));
  FinishIncrementalTraceEvents(data_provider_, timeline_);

  const auto& data = timeline_.timeline_data();
  ASSERT_EQ(data.entry_names.size(), 3u);
  EXPECT_EQ(data.entry_names[0], "while_loop");
  EXPECT_DOUBLE_EQ(data.entry_start_times[0], 1000.0);
  EXPECT_DOUBLE_EQ(data.entry_total_times[0], 1500.0);
  EXPECT_DOUBLE_EQ(data.entry_self_times[0], 1000.0);
  EXPECT_EQ(data.entry_levels[0], 0);
  EXPECT_EQ(data.entry_levels[1], 1);
  EXPECT_EQ(data.entry_levels[2], 1);
}

TEST_F(TracePbEventParserTest,
       IncrementalTraceEventsZoomRefetchFiltersEventsToWindow) {
  xprof::TraceDataResponse frame;
  frame.add_interned_strings("");
  frame.add_interned_strings("op_early");
  frame.add_interned_strings("op_middle");
  frame.add_interned_strings("op_late");
  auto* proc = frame.mutable_metadata()->add_processes();
  proc->set_id(1);
  proc->set_name("host:0 /device:TPU:0");
  auto* th = proc->add_threads();
  th->set_id(10);
  th->set_name("XLA Ops");
  auto* series = frame.add_complete_events();
  series->mutable_metadata()->set_process_id(1);
  series->mutable_metadata()->set_thread_id(10);
  // op_early: [1000, 1200] us
  series->add_deltas(1000000000ULL);
  series->add_durations(200000000ULL);
  series->add_name_refs(1);
  series->add_event_metadata();
  // op_middle: [2000, 2200] us
  series->add_deltas(1000000000ULL);
  series->add_durations(200000000ULL);
  series->add_name_refs(2);
  series->add_event_metadata();
  // op_late: [3000, 3200] us
  series->add_deltas(1000000000ULL);
  series->add_durations(200000000ULL);
  series->add_name_refs(3);
  series->add_event_metadata();

  std::string serialized;
  ASSERT_TRUE(frame.SerializeToString(&serialized));
  const std::string compressed =
      *tensorflow::profiler::ZstdCompression::Compress(serialized);

  // Initial load: all 3 events loaded, full_trace_loaded == true.
  BeginIncrementalTraceEvents(emscripten::val::undefined(), data_provider_,
                              timeline_);
  AppendCompressedTraceEventsIncremental(
      reinterpret_cast<uintptr_t>(compressed.data()), compressed.size());
  EXPECT_TRUE(FinishIncrementalTraceEvents(data_provider_, timeline_));
  ASSERT_EQ(timeline_.timeline_data().entry_names.size(), 3u);
  EXPECT_DOUBLE_EQ(timeline_.data_time_range().start(), 1000.0);
  EXPECT_DOUBLE_EQ(timeline_.data_time_range().end(), 3200.0);

  // Zoom refetch for [1.8, 2.4] ms ([1800, 2400] us): only op_middle is kept,
  // while data_time_range preserves [1000, 3200] us.
  emscripten::val zoom_range = emscripten::val::array();
  zoom_range.set(0, 1.8);
  zoom_range.set(1, 2.4);
  BeginIncrementalTraceEvents(zoom_range, data_provider_, timeline_);
  AppendCompressedTraceEventsIncremental(
      reinterpret_cast<uintptr_t>(compressed.data()), compressed.size());
  // Intermediate step during refetch must not replace existing timeline_data
  // until FinishIncrementalTraceEvents completes.
  EXPECT_FALSE(StepIncrementalTraceEvents(1, data_provider_, timeline_));
  EXPECT_EQ(timeline_.timeline_data().entry_names.size(), 3u);

  EXPECT_FALSE(FinishIncrementalTraceEvents(data_provider_, timeline_));
  ASSERT_EQ(timeline_.timeline_data().entry_names.size(), 1u);
  EXPECT_EQ(timeline_.timeline_data().entry_names[0], "op_middle");
  EXPECT_DOUBLE_EQ(timeline_.data_time_range().start(), 1000.0);
  EXPECT_DOUBLE_EQ(timeline_.data_time_range().end(), 3200.0);
  EXPECT_TRUE(timeline_.window_at_full_resolution());
  EXPECT_DOUBLE_EQ(timeline_.last_fetch_request_range().start(), 1800.0);
  EXPECT_DOUBLE_EQ(timeline_.last_fetch_request_range().end(), 2400.0);

  // Zooming back out restores the stashed full-trace overview in O(1) without
  // re-decompressing or re-parsing frames.
  EXPECT_TRUE(RestoreFullTraceOverviewIfAvailable(timeline_));
  ASSERT_EQ(timeline_.timeline_data().entry_names.size(), 3u);
  EXPECT_EQ(timeline_.timeline_data().entry_names[0], "op_early");
  EXPECT_EQ(timeline_.timeline_data().entry_names[1], "op_middle");
  EXPECT_EQ(timeline_.timeline_data().entry_names[2], "op_late");
  EXPECT_DOUBLE_EQ(timeline_.last_fetch_request_range().start(), 1000.0);
  EXPECT_DOUBLE_EQ(timeline_.last_fetch_request_range().end(), 3200.0);
}

TEST_F(TracePbEventParserTest,
       IncrementalTraceEventsZoomRefetchSkipsNonOverlappingFrames) {
  auto compress_frame = [](const xprof::TraceDataResponse& frame) {
    std::string serialized;
    EXPECT_TRUE(frame.SerializeToString(&serialized));
    return *tensorflow::profiler::ZstdCompression::Compress(serialized);
  };

  // Frame 0: [100, 200] ms (100,000..200,000 us), interned strings ["", "op0"]
  xprof::TraceDataResponse f0;
  f0.add_interned_strings("");
  f0.add_interned_strings("op0");
  auto* proc = f0.mutable_metadata()->add_processes();
  proc->set_id(1);
  proc->set_name("host:0 /device:TPU:0");
  auto* th = proc->add_threads();
  th->set_id(10);
  th->set_name("XLA Ops");
  auto* s0 = f0.add_complete_events();
  s0->mutable_metadata()->set_process_id(1);
  s0->mutable_metadata()->set_thread_id(10);
  s0->add_deltas(100000000000ULL);
  s0->add_durations(100000000000ULL);
  s0->add_name_refs(1);
  s0->add_event_metadata();

  // Frame 1: [2000, 2100] ms, cumulative interned string ["op1"] (id 2)
  xprof::TraceDataResponse f1;
  f1.add_interned_strings("op1");
  auto* s1 = f1.add_complete_events();
  s1->mutable_metadata()->set_process_id(1);
  s1->mutable_metadata()->set_thread_id(10);
  s1->add_deltas(2000000000000ULL);
  s1->add_durations(100000000000ULL);
  s1->add_name_refs(2);
  s1->add_event_metadata();

  // Frame 2: [5000, 5100] ms, cumulative interned string ["op2"] (id 3)
  xprof::TraceDataResponse f2;
  f2.add_interned_strings("op2");
  auto* s2 = f2.add_complete_events();
  s2->mutable_metadata()->set_process_id(1);
  s2->mutable_metadata()->set_thread_id(10);
  s2->add_deltas(5000000000000ULL);
  s2->add_durations(100000000000ULL);
  s2->add_name_refs(3);
  s2->add_event_metadata();

  const std::string c0 = compress_frame(f0);
  const std::string c1 = compress_frame(f1);
  const std::string c2 = compress_frame(f2);

  BeginIncrementalTraceEvents(emscripten::val::undefined(), data_provider_,
                              timeline_);
  AppendCompressedTraceEventsIncremental(reinterpret_cast<uintptr_t>(c0.data()),
                                         c0.size());
  AppendCompressedTraceEventsIncremental(reinterpret_cast<uintptr_t>(c1.data()),
                                         c1.size());
  AppendCompressedTraceEventsIncremental(reinterpret_cast<uintptr_t>(c2.data()),
                                         c2.size());
  EXPECT_TRUE(FinishIncrementalTraceEvents(data_provider_, timeline_));
  ASSERT_EQ(timeline_.timeline_data().entry_names.size(), 3u);

  // Zoom refetch for [1900, 2200] ms: Frame 0 ([100, 200] ms) and Frame 2
  // ([5000, 5100] ms) are outside the 500 ms margin around [1900, 2200] ms, so
  // ShouldEnqueueCompressedFrameIncremental() returns false for Frames 0 and 2
  // and true for Frame 1.
  emscripten::val zoom_range = emscripten::val::array();
  zoom_range.set(0, 1900.0);
  zoom_range.set(1, 2200.0);
  BeginIncrementalTraceEvents(zoom_range, data_provider_, timeline_);
  EXPECT_FALSE(ShouldEnqueueCompressedFrameIncremental());  // Frame 0 skipped
  EXPECT_TRUE(ShouldEnqueueCompressedFrameIncremental());   // Frame 1 needed
  AppendCompressedTraceEventsIncremental(reinterpret_cast<uintptr_t>(c1.data()),
                                         c1.size());
  EXPECT_FALSE(ShouldEnqueueCompressedFrameIncremental());  // Frame 2 skipped
  EXPECT_FALSE(FinishIncrementalTraceEvents(data_provider_, timeline_));

  ASSERT_EQ(timeline_.timeline_data().entry_names.size(), 1u);
  EXPECT_EQ(timeline_.timeline_data().entry_names[0], "op1");
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

}  // namespace traceviewer
