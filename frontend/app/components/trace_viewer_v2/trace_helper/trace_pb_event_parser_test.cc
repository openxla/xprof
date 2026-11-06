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
  // Trailing frame length 0xFFFFFFF0 exercises the 32-bit size_t overflow guard
  // (frame_len > data_size - offset).
  xchn_payload.append("\xF0\xFF\xFF\xFF", 4);

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
  // Trailing frame length 0xFFFFFFF0 exercises the 32-bit size_t overflow guard
  // (frame_len > data_size - offset).
  xchn_payload.append("\xF0\xFF\xFF\xFF", 4);

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

TEST_F(TracePbEventParserTest,
       CompactsFullOverviewToMipPyramidsAndRestoresOnZoomOut) {
  xprof::TraceDataResponse f0;
  f0.add_interned_strings("");
  f0.add_interned_strings("sc_op");
  auto* proc = f0.mutable_metadata()->add_processes();
  proc->set_id(1);
  proc->set_name("host:0 /device:TPU:0 SparseCore:0");
  auto* th = proc->add_threads();
  th->set_id(10);
  th->set_name("SparseCore Ops");
  auto* s0 = f0.add_complete_events();
  s0->mutable_metadata()->set_process_id(1);
  s0->mutable_metadata()->set_thread_id(10);

  // Add 2000 closely-packed back-to-back events on the same level so that
  // BuildLevelMipPyramids merges multiple events into MIP tiles and
  // CompactTimelineDataToMipOverview compacts raw entries down to dominant
  // tile events.
  constexpr int kNumEvents = 2000;
  for (int i = 0; i < kNumEvents; ++i) {
    s0->add_deltas(i == 0 ? 1000000ULL : 1000000ULL);
    s0->add_durations(900000ULL);
    s0->add_name_refs(1);
    s0->add_event_metadata();
  }

  std::string serialized;
  ASSERT_TRUE(f0.SerializeToString(&serialized));
  const std::string c0 =
      *tensorflow::profiler::ZstdCompression::Compress(serialized);

  timeline_.set_max_uncompacted_overview_entries_for_test(512);
  BeginIncrementalTraceEvents(emscripten::val::undefined(), data_provider_,
                              timeline_);
  AppendCompressedTraceEventsIncremental(reinterpret_cast<uintptr_t>(c0.data()),
                                         c0.size());
  EXPECT_FALSE(FinishIncrementalTraceEvents(data_provider_, timeline_));
  EXPECT_TRUE(timeline_.is_compacted());
  EXPECT_TRUE(HasCachedTraceEventsIncremental());
  EXPECT_TRUE(timeline_.is_showing_full_overview());

  const size_t compacted_overview_entries =
      timeline_.timeline_data().entry_start_times.size();
  EXPECT_GT(compacted_overview_entries, 0u);
  EXPECT_LT(compacted_overview_entries, static_cast<size_t>(kNumEvents));
  ASSERT_FALSE(timeline_.level_mip_pyramids_for_test().empty());
  EXPECT_TRUE(timeline_.level_mip_pyramids_for_test()[0].has_pyramid);
  EXPECT_TRUE(timeline_.level_mip_pyramids_for_test()[0].raw_events_compacted);

  // Zoom in to a narrow window [0.0, 0.5] ms (~500 events, <= 512 so raw
  // uncompacted events are retained in the zoomed-in window). Because
  // HasCachedTraceEventsIncremental() is true, no compressed frame needs to be
  // re-enqueued or re-decompressed.
  emscripten::val zoom_range = emscripten::val::array();
  zoom_range.set(0, 0.0);
  zoom_range.set(1, 0.5);
  BeginIncrementalTraceEvents(zoom_range, data_provider_, timeline_);
  EXPECT_FALSE(FinishIncrementalTraceEvents(data_provider_, timeline_));
  EXPECT_FALSE(timeline_.is_showing_full_overview());
  EXPECT_TRUE(timeline_.HasSavedFullOverview());
  EXPECT_EQ(timeline_.timeline_data().entry_start_times.size(), 500u);

  // Zoom out and restore the mip-compacted full overview from memory.
  EXPECT_TRUE(RestoreFullTraceOverviewIfAvailable(timeline_));
  EXPECT_TRUE(timeline_.is_showing_full_overview());
  EXPECT_TRUE(timeline_.is_compacted());
  EXPECT_EQ(timeline_.timeline_data().entry_start_times.size(),
            compacted_overview_entries);
  ASSERT_FALSE(timeline_.level_mip_pyramids_for_test().empty());
  EXPECT_TRUE(timeline_.level_mip_pyramids_for_test()[0].has_pyramid);
  EXPECT_TRUE(timeline_.level_mip_pyramids_for_test()[0].raw_events_compacted);
}

TEST_F(TracePbEventParserTest,
       FiltersRefetchByExpandedTracksAndBorrowsOverviewPyramids) {
  xprof::TraceDataResponse f0;
  f0.add_interned_strings("");
  f0.add_interned_strings("tc_op");
  f0.add_interned_strings("sc_op");

  // Process 1 (TensorCore, expanded by default) and Process 2 (SparseCore,
  // collapsed by default because its name contains "SparseCore").
  auto* proc1 = f0.mutable_metadata()->add_processes();
  proc1->set_id(1);
  proc1->set_name("host:0 /device:TPU:0");
  auto* th1 = proc1->add_threads();
  th1->set_id(10);
  th1->set_name("XLA Ops");

  auto* proc2 = f0.mutable_metadata()->add_processes();
  proc2->set_id(2);
  proc2->set_name("host:0 /device:TPU:0 SparseCore:0");
  auto* th2 = proc2->add_threads();
  th2->set_id(20);
  th2->set_name("SparseCore Ops");

  auto* s1 = f0.add_complete_events();
  s1->mutable_metadata()->set_process_id(1);
  s1->mutable_metadata()->set_thread_id(10);
  for (int i = 0; i < 100; ++i) {
    s1->add_deltas(10000000ULL);
    s1->add_durations(9000000ULL);
    s1->add_name_refs(1);
    s1->add_event_metadata();
  }

  auto* s2 = f0.add_complete_events();
  s2->mutable_metadata()->set_process_id(2);
  s2->mutable_metadata()->set_thread_id(20);
  for (int i = 0; i < 1000; ++i) {
    s2->add_deltas(1000000ULL);
    s2->add_durations(900000ULL);
    s2->add_name_refs(2);
    s2->add_event_metadata();
  }

  std::string serialized;
  ASSERT_TRUE(f0.SerializeToString(&serialized));
  const std::string c0 =
      *tensorflow::profiler::ZstdCompression::Compress(serialized);

  timeline_.set_max_uncompacted_overview_entries_for_test(512);
  BeginIncrementalTraceEvents(emscripten::val::undefined(), data_provider_,
                              timeline_);
  AppendCompressedTraceEventsIncremental(reinterpret_cast<uintptr_t>(c0.data()),
                                         c0.size());
  EXPECT_FALSE(FinishIncrementalTraceEvents(data_provider_, timeline_));

  // On zoom refetch into [0.0, 0.5] ms, only Process 1 (expanded) is in
  // ComputeActiveTracksForFetch(); Process 2 (SparseCore, collapsed) is
  // skipped, so only Process 1's 50 events are decoded into entry_names, while
  // Process 2's level borrows its overview MIP pyramid from
  // saved_full_overview_.
  emscripten::val zoom_range = emscripten::val::array();
  zoom_range.set(0, 0.0);
  zoom_range.set(1, 0.5);
  BeginIncrementalTraceEvents(zoom_range, data_provider_, timeline_);
  EXPECT_FALSE(FinishIncrementalTraceEvents(data_provider_, timeline_));

  EXPECT_EQ(timeline_.timeline_data().entry_names.size(), 50u);
  for (const std::string& name : timeline_.timeline_data().entry_names) {
    EXPECT_EQ(name, "tc_op");
  }
  ASSERT_GE(timeline_.level_mip_pyramids_for_test().size(), 2u);
  // Process 2's level borrowed its MIP pyramid from saved_full_overview_.
  EXPECT_TRUE(timeline_.level_mip_pyramids_for_test().back().has_pyramid);
  EXPECT_FALSE(
      timeline_.level_mip_pyramids_for_test().back().levels[1].tiles.empty());
  EXPECT_FALSE(timeline_.HasUnloadedVisibleExpandedTracks());
}

TEST_F(TracePbEventParserTest,
       ZoomInSynchronousRefetchThenZoomOutRestoresFullOverview) {
  xprof::TraceDataResponse f0;
  f0.add_interned_strings("");
  f0.add_interned_strings("early_op");
  f0.add_interned_strings("late_op");
  f0.set_full_timespan_start_ps(0ULL);
  f0.set_full_timespan_end_ps(10000000000000ULL);  // 10,000 ms

  auto* proc = f0.mutable_metadata()->add_processes();
  proc->set_id(1);
  proc->set_name("host:0 /device:TPU:0");
  auto* th = proc->add_threads();
  th->set_id(10);
  th->set_name("XLA Ops");

  auto* s1 = f0.add_complete_events();
  s1->mutable_metadata()->set_process_id(1);
  s1->mutable_metadata()->set_thread_id(10);

  // Add 600 events in [0 ms, 1200 ms] ("early_op") and 600 events in
  // [8850 ms, 10000 ms] ("late_op") across a 10,000 ms trace.
  for (int i = 0; i < 600; ++i) {
    s1->add_deltas(i == 0 ? 1000000000ULL : 1000000000ULL);  // +1 ms each
    s1->add_durations(800000000ULL);                         // 0.8 ms
    s1->add_name_refs(1);
    s1->add_event_metadata();
  }
  for (int i = 0; i < 600; ++i) {
    // Jump from 600 ms to 8850 ms on the first late event (+8250 ms).
    s1->add_deltas(i == 0 ? 8250000000000ULL : 1000000000ULL);
    s1->add_durations(800000000ULL);
    s1->add_name_refs(2);
    s1->add_event_metadata();
  }

  std::string serialized;
  ASSERT_TRUE(f0.SerializeToString(&serialized));
  const std::string c0 =
      *tensorflow::profiler::ZstdCompression::Compress(serialized);

  timeline_.set_max_uncompacted_overview_entries_for_test(512);
  BeginIncrementalTraceEvents(emscripten::val::undefined(), data_provider_,
                              timeline_);
  AppendCompressedTraceEventsIncremental(reinterpret_cast<uintptr_t>(c0.data()),
                                         c0.size());
  EXPECT_FALSE(FinishIncrementalTraceEvents(data_provider_, timeline_));
  EXPECT_TRUE(timeline_.is_showing_full_overview());
  EXPECT_TRUE(timeline_.is_compacted());
  EXPECT_FALSE(timeline_.is_incremental_loading());
  const size_t full_overview_count =
      timeline_.timeline_data().entry_names.size();
  EXPECT_GT(full_overview_count, 0u);

  // Wire kFetchData to execute the incremental cached refetch synchronously,
  // reproducing the exact call order when stepIncrementalTraceEvents finishes
  // in a single synchronous step.
  int fetch_count = 0;
  timeline_.set_event_callback(
      [&](absl::string_view type, const EventData& detail) {
        if (type != kFetchData) return;
        ++fetch_count;
        const double start_ms =
            std::any_cast<double>(detail.at(std::string(kFetchDataStart)));
        const double end_ms =
            std::any_cast<double>(detail.at(std::string(kFetchDataEnd)));
        emscripten::val range = emscripten::val::array();
        range.set(0, start_ms);
        range.set(1, end_ms);
        BeginIncrementalTraceEvents(range, data_provider_, timeline_);
        while (StepIncrementalTraceEvents(65536, data_provider_, timeline_)) {
        }
        FinishIncrementalTraceEvents(data_provider_, timeline_);
      });

  // 1. Zoom into [9000 ms, 9500 ms] (500 ms window -> 1500 ms fetch window
  // [8500 ms, 10000 ms]).
  timeline_.SetVisibleRange({9000000.0, 9500000.0});
  timeline_.MaybeRequestData();
  EXPECT_EQ(fetch_count, 1);
  // Regression check: synchronous completion inside kFetchData callback must
  // NOT leave is_incremental_loading() stuck at true.
  EXPECT_FALSE(timeline_.is_incremental_loading());
  EXPECT_FALSE(timeline_.is_showing_full_overview());
  EXPECT_TRUE(timeline_.HasSavedFullOverview());
  EXPECT_EQ(timeline_.timeline_data().entry_names.size(), 600u);
  for (const std::string& name : timeline_.timeline_data().entry_names) {
    EXPECT_EQ(name, "late_op");
  }

  // 2. Zoom out to [550 ms, 9950 ms] (9400 ms visible window, 94% of the
  // 10,000 ms trace) and call MaybeRequestData(). This must immediately restore
  // the full overview including all "early_op" and "late_op" events.
  timeline_.SetVisibleRange({550000.0, 9950000.0});
  timeline_.MaybeRequestData();
  EXPECT_TRUE(timeline_.is_showing_full_overview());
  EXPECT_FALSE(timeline_.is_incremental_loading());
  EXPECT_EQ(timeline_.timeline_data().entry_names.size(), full_overview_count);
  bool has_early_op = false;
  bool has_late_op = false;
  for (const std::string& name : timeline_.timeline_data().entry_names) {
    if (name == "early_op") has_early_op = true;
    if (name == "late_op") has_late_op = true;
  }
  EXPECT_TRUE(has_early_op);
  EXPECT_TRUE(has_late_op);

  // 3. Also verify that if a filtered refetch is in flight when the user zooms
  // out and restores the full overview, finishing the stale filtered refetch
  // does not overwrite the restored full overview.
  emscripten::val zoom_range = emscripten::val::array();
  zoom_range.set(0, 8850.0);
  zoom_range.set(1, 10000.0);
  BeginIncrementalTraceEvents(zoom_range, data_provider_, timeline_);
  StepIncrementalTraceEvents(100, data_provider_, timeline_);
  // User zooms out to [550 ms, 9950 ms] before FinishIncrementalTraceEvents:
  timeline_.SetVisibleRange({550000.0, 9950000.0});
  timeline_.MaybeRequestData();
  EXPECT_TRUE(timeline_.is_showing_full_overview());
  // Stale filtered refetch finishes late:
  FinishIncrementalTraceEvents(data_provider_, timeline_);
  EXPECT_TRUE(timeline_.is_showing_full_overview());
  EXPECT_EQ(timeline_.timeline_data().entry_names.size(), full_overview_count);
}

TEST_F(TracePbEventParserTest,
       ZoomRefetchWhenAllVisibleProcessesCollapsedDoesNotDropAllEvents) {
  xprof::TraceDataResponse f0;
  f0.add_interned_strings("");
  f0.add_interned_strings("sc_op");
  // SparseCore processes default to collapsed, so ComputeActiveTracksForFetch
  // returns an empty set when no other process is expanded.
  auto* proc = f0.mutable_metadata()->add_processes();
  proc->set_id(2);
  proc->set_name("host:0 /device:TPU:0 SparseCore:0");
  auto* th = proc->add_threads();
  th->set_id(20);
  th->set_name("SparseCore Ops");

  auto* s0 = f0.add_complete_events();
  s0->mutable_metadata()->set_process_id(2);
  s0->mutable_metadata()->set_thread_id(20);
  for (int i = 0; i < 100; ++i) {
    s0->add_deltas(10000000ULL);
    s0->add_durations(9000000ULL);
    s0->add_name_refs(1);
    s0->add_event_metadata();
  }

  std::string serialized;
  ASSERT_TRUE(f0.SerializeToString(&serialized));
  const std::string c0 =
      *tensorflow::profiler::ZstdCompression::Compress(serialized);

  timeline_.set_max_uncompacted_overview_entries_for_test(32);
  BeginIncrementalTraceEvents(emscripten::val::undefined(), data_provider_,
                              timeline_);
  AppendCompressedTraceEventsIncremental(reinterpret_cast<uintptr_t>(c0.data()),
                                         c0.size());
  FinishIncrementalTraceEvents(data_provider_, timeline_);
  ASSERT_TRUE(HasCachedTraceEventsIncremental());

  emscripten::val zoom_range = emscripten::val::array();
  zoom_range.set(0, 0.0);
  zoom_range.set(1, 0.5);
  BeginIncrementalTraceEvents(zoom_range, data_provider_, timeline_);
  FinishIncrementalTraceEvents(data_provider_, timeline_);

  EXPECT_EQ(timeline_.timeline_data().entry_names.size(), 50u);
  EXPECT_FALSE(timeline_.HasUnloadedVisibleExpandedTracks());
}

TEST_F(TracePbEventParserTest,
       OverviewPreCompactionPreservedWhenExceedingMaxCachedPackedBytes) {
  struct ScopedMaxPackedBytes {
    explicit ScopedMaxPackedBytes(size_t bytes) {
      SetMaxCachedPackedBytesForTest(bytes);
    }
    ~ScopedMaxPackedBytes() {
      SetMaxCachedPackedBytesForTest(512ULL * 1024 * 1024);
    }
  } scoped_limit(60000);

  xprof::TraceDataResponse f0;
  f0.add_interned_strings("");
  f0.add_interned_strings("dense_op");
  f0.set_full_timespan_start_ps(0ULL);
  f0.set_full_timespan_end_ps(10000000000ULL);

  auto* proc1 = f0.mutable_metadata()->add_processes();
  proc1->set_id(1);
  proc1->set_name("host:0 /device:TPU:0");
  auto* th1 = proc1->add_threads();
  th1->set_id(10);
  th1->set_name("XLA Ops 1");

  auto* proc2 = f0.mutable_metadata()->add_processes();
  proc2->set_id(2);
  proc2->set_name("host:0 /device:TPU:1");
  auto* th2 = proc2->add_threads();
  th2->set_id(20);
  th2->set_name("XLA Ops 2");

  // Process 1 has 9000 sub-quantum events (> 8192, ~54 KB packed) so
  // ProcessOverviewSampledSeriesSlice pre-compacts them and sets
  // overview_pre_compacted = true, while Process 2's packed blocks push
  // building_packed_bytes past the 60 KB test limit and trigger
  // ResetBuildingTraceStore().
  auto* s1 = f0.add_complete_events();
  s1->mutable_metadata()->set_process_id(1);
  s1->mutable_metadata()->set_thread_id(10);
  for (int i = 0; i < 9000; ++i) {
    s1->add_deltas(1000ULL);
    s1->add_durations(500ULL);
    s1->add_name_refs(1);
    s1->add_event_metadata();
  }

  auto* s2 = f0.add_complete_events();
  s2->mutable_metadata()->set_process_id(2);
  s2->mutable_metadata()->set_thread_id(20);
  for (int i = 0; i < 2000; ++i) {
    s2->add_deltas(1000000ULL + static_cast<uint64_t>(i) * 1000ULL);
    s2->add_durations(500000ULL + static_cast<uint64_t>(i) * 100ULL);
    s2->add_name_refs(1);
    auto* meta = s2->add_event_metadata();
    meta->set_serial(static_cast<uint64_t>(i + 1) * 1000ULL);
  }

  std::string serialized;
  ASSERT_TRUE(f0.SerializeToString(&serialized));
  const std::string c0 =
      *tensorflow::profiler::ZstdCompression::Compress(serialized);

  // Set a high uncompacted entry threshold so surviving post-sampled events are
  // well below max_uncompacted_overview_entries().
  timeline_.set_max_uncompacted_overview_entries_for_test(100000);
  BeginIncrementalTraceEvents(emscripten::val::undefined(), data_provider_,
                              timeline_);
  AppendCompressedTraceEventsIncremental(reinterpret_cast<uintptr_t>(c0.data()),
                                         c0.size());
  while (StepIncrementalTraceEvents(5000, data_provider_, timeline_)) {
  }
  EXPECT_FALSE(FinishIncrementalTraceEvents(data_provider_, timeline_));
  EXPECT_FALSE(HasCachedTraceEventsIncremental());
  EXPECT_TRUE(timeline_.is_compacted());
  EXPECT_FALSE(timeline_.full_trace_loaded());
  EXPECT_FALSE(timeline_.window_at_full_resolution());
}

TEST_F(TracePbEventParserTest,
       InitialUrlVisibleRangeFullSessionLoadArmsCachedTraceStore) {
  xprof::TraceDataResponse f0;
  f0.add_interned_strings("");
  f0.add_interned_strings("url_window_op");
  f0.set_full_timespan_start_ps(0ULL);
  f0.set_full_timespan_end_ps(100000000000ULL);

  auto* proc = f0.mutable_metadata()->add_processes();
  proc->set_id(1);
  proc->set_name("host:0 /device:TPU:0");
  auto* th = proc->add_threads();
  th->set_id(10);
  th->set_name("XLA Ops");

  auto* series = f0.add_complete_events();
  series->mutable_metadata()->set_process_id(1);
  series->mutable_metadata()->set_thread_id(10);
  for (int i = 0; i < 64; ++i) {
    series->add_deltas(1000000ULL);
    series->add_durations(500000ULL);
    series->add_name_refs(1);
    series->add_event_metadata();
  }

  std::string serialized;
  ASSERT_TRUE(f0.SerializeToString(&serialized));
  const std::string c0 =
      *tensorflow::profiler::ZstdCompression::Compress(serialized);

  timeline_.set_max_uncompacted_overview_entries_for_test(16);
  emscripten::val url_range = emscripten::val::array();
  url_range.set(0, 10.0);
  url_range.set(1, 20.0);
  BeginIncrementalTraceEvents(url_range, data_provider_, timeline_);
  AppendCompressedTraceEventsIncremental(reinterpret_cast<uintptr_t>(c0.data()),
                                         c0.size());
  while (StepIncrementalTraceEvents(100, data_provider_, timeline_)) {
  }
  EXPECT_FALSE(FinishIncrementalTraceEvents(data_provider_, timeline_));
  EXPECT_TRUE(HasCachedTraceEventsIncremental());
}

TEST_F(TracePbEventParserTest,
       CachedRefetchFiltersAsyncFlowAndCounterEventsToWindow) {
  xprof::TraceDataResponse f0;
  f0.add_interned_strings("");
  f0.add_interned_strings("sync_op");
  f0.add_interned_strings("async_op");
  f0.add_interned_strings("mem_bw");
  f0.set_full_timespan_start_ps(0ULL);
  f0.set_full_timespan_end_ps(100000000000ULL);

  auto* proc = f0.mutable_metadata()->add_processes();
  proc->set_id(1);
  proc->set_name("host:0 /device:TPU:0");
  auto* th = proc->add_threads();
  th->set_id(10);
  th->set_name("XLA Ops");

  // In-window complete events at 1.0 ms and 2.0 ms linked by sync flow_id=777,
  // plus enough events to exceed max_uncompacted_overview_entries (16) so the
  // initial load caches the block store.
  auto* sync_series = f0.add_complete_events();
  sync_series->mutable_metadata()->set_process_id(1);
  sync_series->mutable_metadata()->set_thread_id(10);
  for (int i = 0; i < 40; ++i) {
    sync_series->add_deltas(100000ULL);
    sync_series->add_durations(50000ULL);
    sync_series->add_name_refs(1);
    auto* meta = sync_series->add_event_metadata();
    if (i == 10 || i == 20) {
      meta->set_flow_id(777);
    }
  }

  // >16384 out-of-window async flow events starting at 50 ms (50,000,000,000
  // ps). Without filtering cached_async_flow_events on a zoomed refetch, these
  // saturate kMaxIncrementalFlowEvents (16384) and drop sync flow_id=777.
  auto* async_series = f0.add_async_events();
  async_series->mutable_metadata()->set_process_id(1);
  async_series->mutable_metadata()->set_name_ref(2);
  for (int i = 0; i < 16500; ++i) {
    async_series->add_deltas(i == 0 ? 50000000000ULL : 1000ULL);
    async_series->add_durations(500ULL);
    auto* meta = async_series->add_event_metadata();
    meta->set_flow_id(static_cast<uint64_t>(1000 + i / 2));
  }

  // Counter series with in-window points at 1 ms / 2 ms (values 10.0, 20.0) and
  // an out-of-window spike at 90 ms (value 999.0).
  auto* counter_series = f0.add_counter_events();
  counter_series->mutable_metadata()->set_process_id(1);
  counter_series->mutable_metadata()->set_name_ref(3);
  counter_series->add_deltas(1000000000ULL);
  counter_series->add_event_metadata()->set_counter_value_double(10.0);
  counter_series->add_deltas(1000000000ULL);
  counter_series->add_event_metadata()->set_counter_value_double(20.0);
  counter_series->add_deltas(88000000000ULL);
  counter_series->add_event_metadata()->set_counter_value_double(999.0);

  std::string serialized;
  ASSERT_TRUE(f0.SerializeToString(&serialized));
  const std::string c0 =
      *tensorflow::profiler::ZstdCompression::Compress(serialized);

  timeline_.set_max_uncompacted_overview_entries_for_test(16);
  BeginIncrementalTraceEvents(emscripten::val::undefined(), data_provider_,
                              timeline_);
  AppendCompressedTraceEventsIncremental(reinterpret_cast<uintptr_t>(c0.data()),
                                         c0.size());
  while (StepIncrementalTraceEvents(20000, data_provider_, timeline_)) {
  }
  EXPECT_FALSE(FinishIncrementalTraceEvents(data_provider_, timeline_));
  ASSERT_TRUE(HasCachedTraceEventsIncremental());

  // Zoomed cached refetch over [0.0, 5.0] ms ([0, 5000] us).
  emscripten::val zoom_range = emscripten::val::array();
  zoom_range.set(0, 0.0);
  zoom_range.set(1, 5.0);
  BeginIncrementalTraceEvents(zoom_range, data_provider_, timeline_);
  while (StepIncrementalTraceEvents(5000, data_provider_, timeline_)) {
  }
  FinishIncrementalTraceEvents(data_provider_, timeline_);

  EXPECT_TRUE(
      timeline_.timeline_data().flow_lines_by_flow_id.contains("777"));
  EXPECT_LE(timeline_.fetched_data_time_range().end(), 5000.0);
  bool found_counter = false;
  for (const auto& group : timeline_.timeline_data().groups) {
    if (group.type == Group::Type::kCounter && group.name == "mem_bw") {
      found_counter = true;
      const auto& counters =
          timeline_.timeline_data().counter_data_by_group_index;
      const auto it = counters.find(
          static_cast<int>(&group - timeline_.timeline_data().groups.data()));
      ASSERT_NE(it, counters.end());
      EXPECT_DOUBLE_EQ(it->second.min_value, 10.0);
      EXPECT_DOUBLE_EQ(it->second.max_value, 20.0);
    }
  }
  EXPECT_TRUE(found_counter);
}

TEST_F(TracePbEventParserTest,
       NetworkRefetchesWithWindowSpecificStringTablesPreserveEventNames) {
  auto build_window_response = [](absl::string_view name_1,
                                  absl::string_view name_2, uint64_t ts1_ps,
                                  uint64_t ts2_ps) {
    xprof::TraceDataResponse resp;
    resp.set_full_timespan_start_ps(0ULL);
    resp.set_full_timespan_end_ps(10000000000ULL);  // 10 ms
    resp.add_interned_strings("");
    resp.add_interned_strings(std::string(name_1));
    resp.add_interned_strings(std::string(name_2));
    auto* proc = resp.mutable_metadata()->add_processes();
    proc->set_id(1);
    proc->set_name("host:0 /device:TPU:0");
    auto* th = proc->add_threads();
    th->set_id(10);
    th->set_name("XLA Ops");
    auto* series = resp.add_complete_events();
    series->mutable_metadata()->set_process_id(1);
    series->mutable_metadata()->set_thread_id(10);
    series->add_deltas(ts1_ps);
    series->add_durations(100000000ULL);
    series->add_name_refs(1);
    series->add_event_metadata();
    series->add_deltas(ts2_ps - ts1_ps);
    series->add_durations(100000000ULL);
    series->add_name_refs(2);
    series->add_event_metadata();
    std::string serialized;
    EXPECT_TRUE(resp.SerializeToString(&serialized));
    return *tensorflow::profiler::ZstdCompression::Compress(serialized);
  };

  // 1. Initial load via ParseAndProcessCompressedTraceEvents (legacy overview
  // load when resolution > 0).
  const std::string overview_compressed = build_window_response(
      "overview_op_1", "overview_op_2", 1000000000ULL, 5000000000ULL);
  ParseAndProcessCompressedTraceEvents(
      reinterpret_cast<uintptr_t>(overview_compressed.data()),
      overview_compressed.size(), emscripten::val::undefined(), data_provider_,
      timeline_);
  ASSERT_EQ(timeline_.timeline_data().entry_names.size(), 2u);
  EXPECT_EQ(timeline_.timeline_data().entry_names[0], "overview_op_1");
  EXPECT_EQ(timeline_.timeline_data().entry_names[1], "overview_op_2");

  // 2. First zoom-in network refetch for [1.0, 3.0] ms with its own
  // window-specific interned_strings table.
  const std::string zoom1_compressed = build_window_response(
      "zoom1_op_a", "zoom1_op_b", 1200000000ULL, 2200000000ULL);
  emscripten::val zoom1_range = emscripten::val::array();
  zoom1_range.set(0, 1.0);
  zoom1_range.set(1, 3.0);
  BeginIncrementalTraceEvents(zoom1_range, data_provider_, timeline_);
  AppendCompressedTraceEventsIncremental(
      reinterpret_cast<uintptr_t>(zoom1_compressed.data()),
      zoom1_compressed.size());
  EXPECT_FALSE(FinishIncrementalTraceEvents(data_provider_, timeline_));
  EXPECT_FALSE(HasCachedTraceEventsIncremental());
  ASSERT_EQ(timeline_.timeline_data().entry_names.size(), 2u);
  EXPECT_EQ(timeline_.timeline_data().entry_names[0], "zoom1_op_a");
  EXPECT_EQ(timeline_.timeline_data().entry_names[1], "zoom1_op_b");

  // 3. Second zoom-out / pan network refetch for [0.5, 6.0] ms with a fresh
  // window-specific interned_strings table where indices 1 and 2 map to
  // completely different event names.
  const std::string zoom2_compressed = build_window_response(
      "zoom2_op_x", "zoom2_op_y", 800000000ULL, 4500000000ULL);
  emscripten::val zoom2_range = emscripten::val::array();
  zoom2_range.set(0, 0.5);
  zoom2_range.set(1, 6.0);
  BeginIncrementalTraceEvents(zoom2_range, data_provider_, timeline_);
  AppendCompressedTraceEventsIncremental(
      reinterpret_cast<uintptr_t>(zoom2_compressed.data()),
      zoom2_compressed.size());
  EXPECT_FALSE(FinishIncrementalTraceEvents(data_provider_, timeline_));
  ASSERT_EQ(timeline_.timeline_data().entry_names.size(), 2u);
  EXPECT_EQ(timeline_.timeline_data().entry_names[0], "zoom2_op_x");
  EXPECT_EQ(timeline_.timeline_data().entry_names[1], "zoom2_op_y");
}

TEST_F(TracePbEventParserTest,
       NetworkRefetchAfterUncachedIncrementalLoadUsesResponseStrings) {
  auto compress_frame = [](const xprof::TraceDataResponse& frame) {
    std::string serialized;
    EXPECT_TRUE(frame.SerializeToString(&serialized));
    return *tensorflow::profiler::ZstdCompression::Compress(serialized);
  };

  xprof::TraceDataResponse init_resp;
  init_resp.set_full_timespan_start_ps(0ULL);
  init_resp.set_full_timespan_end_ps(10000000000ULL);
  init_resp.add_interned_strings("");
  init_resp.add_interned_strings("initial_op");
  auto* proc = init_resp.mutable_metadata()->add_processes();
  proc->set_id(1);
  proc->set_name("host:0 /device:TPU:0");
  auto* th = proc->add_threads();
  th->set_id(10);
  th->set_name("XLA Ops");
  auto* s0 = init_resp.add_complete_events();
  s0->mutable_metadata()->set_process_id(1);
  s0->mutable_metadata()->set_thread_id(10);
  s0->add_deltas(1000000000ULL);
  s0->add_durations(200000000ULL);
  s0->add_name_refs(1);
  s0->add_event_metadata();

  const std::string c_init = compress_frame(init_resp);
  BeginIncrementalTraceEvents(emscripten::val::undefined(), data_provider_,
                              timeline_);
  AppendCompressedTraceEventsIncremental(
      reinterpret_cast<uintptr_t>(c_init.data()), c_init.size());
  EXPECT_TRUE(FinishIncrementalTraceEvents(data_provider_, timeline_));
  EXPECT_FALSE(HasCachedTraceEventsIncremental());
  ASSERT_EQ(timeline_.timeline_data().entry_names.size(), 1u);
  EXPECT_EQ(timeline_.timeline_data().entry_names[0], "initial_op");

  // Network refetch response with its own interned_strings table where index 1
  // is "refetch_window_op" (enqueued directly without calling
  // ShouldEnqueueCompressedFrameIncremental()).
  xprof::TraceDataResponse refetch_resp;
  refetch_resp.set_full_timespan_start_ps(0ULL);
  refetch_resp.set_full_timespan_end_ps(10000000000ULL);
  refetch_resp.add_interned_strings("");
  refetch_resp.add_interned_strings("refetch_window_op");
  auto* r_proc = refetch_resp.mutable_metadata()->add_processes();
  r_proc->set_id(1);
  r_proc->set_name("host:0 /device:TPU:0");
  auto* r_th = r_proc->add_threads();
  r_th->set_id(10);
  r_th->set_name("XLA Ops");
  auto* s1 = refetch_resp.add_complete_events();
  s1->mutable_metadata()->set_process_id(1);
  s1->mutable_metadata()->set_thread_id(10);
  s1->add_deltas(2000000000ULL);
  s1->add_durations(200000000ULL);
  s1->add_name_refs(1);
  s1->add_event_metadata();

  const std::string c_refetch = compress_frame(refetch_resp);
  emscripten::val zoom_range = emscripten::val::array();
  zoom_range.set(0, 1.5);
  zoom_range.set(1, 2.5);
  BeginIncrementalTraceEvents(zoom_range, data_provider_, timeline_);
  AppendCompressedTraceEventsIncremental(
      reinterpret_cast<uintptr_t>(c_refetch.data()), c_refetch.size());
  EXPECT_FALSE(FinishIncrementalTraceEvents(data_provider_, timeline_));
  ASSERT_EQ(timeline_.timeline_data().entry_names.size(), 1u);
  EXPECT_EQ(timeline_.timeline_data().entry_names[0], "refetch_window_op");
}
}  // namespace traceviewer
