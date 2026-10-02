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

}  // namespace traceviewer
