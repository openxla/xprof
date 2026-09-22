#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_event_parser.h"

#include <emscripten/val.h>

#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_event.h"

namespace traceviewer {
namespace {

emscripten::val ParseJson(const std::string& json_str) {
  return emscripten::val::global("JSON").call<emscripten::val>(
      "parse", emscripten::val(json_str));
}

TEST(TraceEventParserTest, CompleteEventWithFlowInArray) {
  emscripten::val trace_data = ParseJson(R"({
    "traceEvents": [
      {
        "ph": "X",
        "name": "consumer_op",
        "pid": 1,
        "tid": 2,
        "ts": 100.0,
        "dur": 50.0,
        "flow_in": ["123", "456"]
      }
    ]
  })");

  ParsedTraceEvents result =
      ParseTraceEvents(trace_data, emscripten::val::null());

  ASSERT_EQ(result.flame_events.size(), 1);
  EXPECT_EQ(result.flame_events[0].name, "consumer_op");

  ASSERT_EQ(result.flow_events.size(), 2);
  EXPECT_EQ(result.flow_events[0].ph, Phase::kFlowEnd);
  EXPECT_EQ(result.flow_events[0].id, "123");
  EXPECT_EQ(result.flow_events[0].name, "consumer_op");
  EXPECT_DOUBLE_EQ(result.flow_events[0].ts, 100.0);

  EXPECT_EQ(result.flow_events[1].ph, Phase::kFlowEnd);
  EXPECT_EQ(result.flow_events[1].id, "456");
  EXPECT_EQ(result.flow_events[1].name, "consumer_op");
  EXPECT_DOUBLE_EQ(result.flow_events[1].ts, 100.0);
}

TEST(TraceEventParserTest, CompleteEventWithFlowOutArray) {
  emscripten::val trace_data = ParseJson(R"({
    "traceEvents": [
      {
        "ph": "X",
        "name": "producer_op",
        "pid": 1,
        "tid": 2,
        "ts": 50.0,
        "dur": 30.0,
        "flow_out": ["789"]
      }
    ]
  })");

  ParsedTraceEvents result =
      ParseTraceEvents(trace_data, emscripten::val::null());

  ASSERT_EQ(result.flame_events.size(), 1);
  EXPECT_EQ(result.flame_events[0].name, "producer_op");

  ASSERT_EQ(result.flow_events.size(), 1);
  EXPECT_EQ(result.flow_events[0].ph, Phase::kFlowStart);
  EXPECT_EQ(result.flow_events[0].id, "789");
  EXPECT_EQ(result.flow_events[0].name, "producer_op");
  EXPECT_DOUBLE_EQ(result.flow_events[0].ts, 50.0);
}

TEST(TraceEventParserTest, CompleteEventWithBothFlowInAndFlowOutArrays) {
  emscripten::val trace_data = ParseJson(R"({
    "traceEvents": [
      {
        "ph": "X",
        "name": "step_op",
        "pid": 1,
        "tid": 2,
        "ts": 120.0,
        "dur": 40.0,
        "flow_in": ["123"],
        "flow_out": ["789"]
      }
    ]
  })");

  ParsedTraceEvents result =
      ParseTraceEvents(trace_data, emscripten::val::null());

  ASSERT_EQ(result.flame_events.size(), 1);
  EXPECT_EQ(result.flame_events[0].name, "step_op");

  ASSERT_EQ(result.flow_events.size(), 2);
  EXPECT_EQ(result.flow_events[0].ph, Phase::kFlowEnd);
  EXPECT_EQ(result.flow_events[0].id, "123");
  EXPECT_EQ(result.flow_events[0].name, "step_op");

  EXPECT_EQ(result.flow_events[1].ph, Phase::kFlowStart);
  EXPECT_EQ(result.flow_events[1].id, "789");
  EXPECT_EQ(result.flow_events[1].name, "step_op");
}

TEST(TraceEventParserTest, FlowEventsWithNumericFlowIds) {
  emscripten::val trace_data = ParseJson(R"({
    "traceEvents": [
      {
        "ph": "X",
        "name": "numeric_flow_op",
        "pid": 1,
        "tid": 2,
        "ts": 100.0,
        "dur": 50.0,
        "flow_in": [123, 456],
        "flow_out": [789]
      }
    ]
  })");

  ParsedTraceEvents result =
      ParseTraceEvents(trace_data, emscripten::val::null());

  ASSERT_EQ(result.flame_events.size(), 1);
  ASSERT_EQ(result.flow_events.size(), 3);
  EXPECT_EQ(result.flow_events[0].ph, Phase::kFlowEnd);
  EXPECT_EQ(result.flow_events[0].id, "123");
  EXPECT_EQ(result.flow_events[1].ph, Phase::kFlowEnd);
  EXPECT_EQ(result.flow_events[1].id, "456");
  EXPECT_EQ(result.flow_events[2].ph, Phase::kFlowStart);
  EXPECT_EQ(result.flow_events[2].id, "789");
}

TEST(TraceEventParserTest, CompleteEventWithBooleanFlowFlagsAndId) {
  emscripten::val trace_data = ParseJson(R"({
    "traceEvents": [
      {
        "ph": "X",
        "name": "bool_flow_op",
        "pid": 1,
        "tid": 2,
        "id": "555",
        "ts": 100.0,
        "dur": 50.0,
        "flow_in": true,
        "flow_out": true
      }
    ]
  })");

  ParsedTraceEvents result =
      ParseTraceEvents(trace_data, emscripten::val::null());

  ASSERT_EQ(result.flame_events.size(), 1);
  ASSERT_EQ(result.flow_events.size(), 2);
  EXPECT_EQ(result.flow_events[0].ph, Phase::kFlowEnd);
  EXPECT_EQ(result.flow_events[0].id, "555");
  EXPECT_EQ(result.flow_events[1].ph, Phase::kFlowStart);
  EXPECT_EQ(result.flow_events[1].id, "555");
}

TEST(TraceEventParserTest, EdgeCaseEmptyFlowArrays) {
  emscripten::val trace_data = ParseJson(R"({
    "traceEvents": [
      {
        "ph": "X",
        "name": "empty_flow_op",
        "pid": 1,
        "tid": 2,
        "ts": 100.0,
        "dur": 50.0,
        "flow_in": [],
        "flow_out": []
      }
    ]
  })");

  ParsedTraceEvents result =
      ParseTraceEvents(trace_data, emscripten::val::null());

  ASSERT_EQ(result.flame_events.size(), 1);
  EXPECT_TRUE(result.flow_events.empty());
}

TEST(TraceEventParserTest, EdgeCaseFlowInWithoutMatchingFlowOut) {
  emscripten::val trace_data = ParseJson(R"({
    "traceEvents": [
      {
        "ph": "X",
        "name": "consumer_only_op",
        "pid": 1,
        "tid": 2,
        "ts": 200.0,
        "dur": 50.0,
        "flow_in": ["unmatched_999"]
      }
    ]
  })");

  ParsedTraceEvents result =
      ParseTraceEvents(trace_data, emscripten::val::null());

  ASSERT_EQ(result.flame_events.size(), 1);
  ASSERT_EQ(result.flow_events.size(), 1);
  EXPECT_EQ(result.flow_events[0].ph, Phase::kFlowEnd);
  EXPECT_EQ(result.flow_events[0].id, "unmatched_999");
}

}  // namespace
}  // namespace traceviewer
