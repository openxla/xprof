#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_event_parser.h"

#include <emscripten/val.h>

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tsl/platform/fingerprint.h"
#include "tsl/profiler/lib/context_types.h"
#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_event.h"

namespace traceviewer {
namespace {

using ::testing::ElementsAre;
using ::testing::Field;

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
  EXPECT_EQ(result.flow_events[0].flow_id, 123);
  EXPECT_EQ(result.flow_events[0].name, "consumer_op");
  EXPECT_DOUBLE_EQ(result.flow_events[0].ts, 100.0);

  EXPECT_EQ(result.flow_events[1].ph, Phase::kFlowEnd);
  EXPECT_EQ(result.flow_events[1].id, "456");
  EXPECT_EQ(result.flow_events[1].flow_id, 456);
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
  EXPECT_EQ(result.flow_events[0].flow_id, 789);
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
  EXPECT_EQ(result.flow_events[0].flow_id, 123);
  EXPECT_EQ(result.flow_events[0].name, "step_op");

  EXPECT_EQ(result.flow_events[1].ph, Phase::kFlowStart);
  EXPECT_EQ(result.flow_events[1].id, "789");
  EXPECT_EQ(result.flow_events[1].flow_id, 789);
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
  EXPECT_EQ(result.flow_events[0].flow_id, 123);
  EXPECT_EQ(result.flow_events[1].ph, Phase::kFlowEnd);
  EXPECT_EQ(result.flow_events[1].id, "456");
  EXPECT_EQ(result.flow_events[1].flow_id, 456);
  EXPECT_EQ(result.flow_events[2].ph, Phase::kFlowStart);
  EXPECT_EQ(result.flow_events[2].id, "789");
  EXPECT_EQ(result.flow_events[2].flow_id, 789);
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
  EXPECT_EQ(result.flow_events[0].flow_id, 555);
  EXPECT_EQ(result.flow_events[1].ph, Phase::kFlowStart);
  EXPECT_EQ(result.flow_events[1].id, "555");
  EXPECT_EQ(result.flow_events[1].flow_id, 555);
}

TEST(TraceEventParserTest, CompleteEventWithBooleanFlowFlagsAndNumericBindId) {
  emscripten::val trace_data = ParseJson(R"({
    "traceEvents": [
      {
        "ph": "X",
        "name": "bind_id_flow_op",
        "pid": 1,
        "tid": 2,
        "bind_id": 777,
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
  EXPECT_TRUE(result.flame_events[0].id.empty());
  EXPECT_EQ(result.flame_events[0].flow_id, 0);

  ASSERT_EQ(result.flow_events.size(), 2);
  EXPECT_EQ(result.flow_events[0].ph, Phase::kFlowEnd);
  EXPECT_EQ(result.flow_events[0].id, "777");
  EXPECT_EQ(result.flow_events[0].flow_id, 777);
  EXPECT_EQ(result.flow_events[1].ph, Phase::kFlowStart);
  EXPECT_EQ(result.flow_events[1].id, "777");
  EXPECT_EQ(result.flow_events[1].flow_id, 777);
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
  EXPECT_EQ(result.flow_events[0].flow_id, tsl::Fingerprint64("unmatched_999"));
}

TEST(TraceEventParserTest, AsyncBeginEndSetsIsAsyncAndClampsDuration) {
  emscripten::val trace_data = ParseJson(R"({
    "traceEvents": [
      {
        "ph": "b",
        "name": "async_op",
        "pid": 1,
        "tid": 2,
        "id": 100,
        "ts": 10.0,
        "args": {"begin_k": "begin_v"}
      },
      {
        "ph": "e",
        "name": "async_op",
        "pid": 1,
        "tid": 2,
        "id": 100,
        "ts": 25.0,
        "args": {"end_k": "end_v"}
      },
      {
        "ph": "b",
        "name": "inverted_async_op",
        "pid": 1,
        "tid": 2,
        "id": 200,
        "ts": 50.0
      },
      {
        "ph": "e",
        "name": "inverted_async_op",
        "pid": 1,
        "tid": 2,
        "id": 200,
        "ts": 30.0
      }
    ]
  })");

  ParsedTraceEvents result =
      ParseTraceEvents(trace_data, emscripten::val::null());

  ASSERT_EQ(result.flame_events.size(), 2);
  EXPECT_EQ(result.flame_events[0].ph, Phase::kComplete);
  EXPECT_TRUE(result.flame_events[0].is_async);
  EXPECT_DOUBLE_EQ(result.flame_events[0].ts, 10.0);
  EXPECT_DOUBLE_EQ(result.flame_events[0].dur, 15.0);
  EXPECT_EQ(result.flame_events[0].args.at("begin_k"), "begin_v");
  EXPECT_EQ(result.flame_events[0].args.at("end_k"), "end_v");

  EXPECT_EQ(result.flame_events[1].ph, Phase::kComplete);
  EXPECT_TRUE(result.flame_events[1].is_async);
  EXPECT_DOUBLE_EQ(result.flame_events[1].ts, 50.0);
  EXPECT_DOUBLE_EQ(result.flame_events[1].dur, 0.0);
}

TEST(TraceEventParserTest, AsyncAndFlowEventsWithHexAndSymbolicStringIds) {
  emscripten::val trace_data = ParseJson(R"({
    "traceEvents": [
      {
        "ph": "b",
        "name": "hex_async_op",
        "pid": 1,
        "tid": 2,
        "id": "0x2a",
        "ts": 10.0
      },
      {
        "ph": "e",
        "name": "hex_async_op",
        "pid": 1,
        "tid": 2,
        "id": "0x2a",
        "ts": 20.0
      },
      {
        "ph": "b",
        "name": "sym_async_op",
        "pid": 1,
        "tid": 2,
        "bind_id": "async_req_1",
        "ts": 30.0
      },
      {
        "ph": "e",
        "name": "sym_async_op",
        "pid": 1,
        "tid": 2,
        "bind_id": "async_req_1",
        "ts": 45.0
      },
      {
        "ph": "s",
        "name": "hex_flow_start",
        "pid": 1,
        "tid": 2,
        "id": "0x2a",
        "ts": 10.0
      },
      {
        "ph": "f",
        "name": "sym_flow_end",
        "pid": 1,
        "tid": 2,
        "bind_id": "async_req_1",
        "ts": 45.0
      }
    ]
  })");

  ParsedTraceEvents result =
      ParseTraceEvents(trace_data, emscripten::val::null());

  ASSERT_EQ(result.flame_events.size(), 4);
  EXPECT_EQ(result.flame_events[0].name, "hex_async_op");
  EXPECT_TRUE(result.flame_events[0].is_async);
  EXPECT_EQ(result.flame_events[0].id, "0x2a");
  EXPECT_EQ(result.flame_events[0].flow_id, 42);
  EXPECT_DOUBLE_EQ(result.flame_events[0].dur, 10.0);

  EXPECT_EQ(result.flame_events[1].name, "sym_async_op");
  EXPECT_TRUE(result.flame_events[1].is_async);
  EXPECT_EQ(result.flame_events[1].id, "async_req_1");
  EXPECT_EQ(result.flame_events[1].flow_id, tsl::Fingerprint64("async_req_1"));
  EXPECT_DOUBLE_EQ(result.flame_events[1].dur, 15.0);

  EXPECT_EQ(result.flame_events[2].ph, Phase::kFlowStart);
  EXPECT_EQ(result.flame_events[2].id, "0x2a");
  EXPECT_EQ(result.flame_events[2].flow_id, 42);

  EXPECT_EQ(result.flame_events[3].ph, Phase::kFlowEnd);
  EXPECT_EQ(result.flame_events[3].id, "async_req_1");
  EXPECT_EQ(result.flame_events[3].flow_id, tsl::Fingerprint64("async_req_1"));
}

TEST(TraceEventParserTest, ArgsFormattingAndNumericCategory) {
  emscripten::val trace_data = ParseJson(R"({
    "traceEvents": [
      {
        "ph": "X",
        "name": "kernel_op",
        "pid": 1,
        "tid": 2,
        "ts": 100.0,
        "dur": 20.0,
        "cat": 12,
        "args": {
          "int_val": 42,
          "float_val": 3.5,
          "bool_true": true,
          "bool_false": false,
          "kernel_details": {"occupancy": 1}
        }
      }
    ]
  })");

  ParsedTraceEvents result =
      ParseTraceEvents(trace_data, emscripten::val::null());

  ASSERT_EQ(result.flame_events.size(), 1);
  const TraceEvent& ev = result.flame_events[0];
  EXPECT_EQ(ev.category, tsl::profiler::ContextType::kTpuLaunch);
  EXPECT_EQ(ev.args.at("int_val"), "42");
  EXPECT_EQ(ev.args.at("float_val"), "3.5");
  EXPECT_EQ(ev.args.at("bool_true"), "true");
  EXPECT_EQ(ev.args.at("bool_false"), "false");
  EXPECT_EQ(ev.args.at("kernel_details"), "{\"occupancy\":1}");
}

TEST(TraceEventParserTest, DeprecatedInstantPhaseEmittedToFlameEvents) {
  emscripten::val trace_data = ParseJson(R"({
    "traceEvents": [
      {
        "ph": "I",
        "name": "deprecated_instant_op",
        "pid": 1,
        "tid": 2,
        "ts": 100.0
      }
    ]
  })");

  ParsedTraceEvents result =
      ParseTraceEvents(trace_data, emscripten::val::null());

  ASSERT_EQ(result.flame_events.size(), 1);
  EXPECT_EQ(result.flame_events[0].ph, Phase::kInstantDeprecated);
  EXPECT_EQ(result.flame_events[0].name, "deprecated_instant_op");
  EXPECT_DOUBLE_EQ(result.flame_events[0].ts, 100.0);
}

TEST(TraceEventParserTest, ZeroFlowAndAsyncIdsArePreserved) {
  emscripten::val trace_data = ParseJson(R"({
    "traceEvents": [
      {
        "ph": "b",
        "name": "zero_async_num",
        "pid": 1,
        "tid": 2,
        "id": 0,
        "ts": 10.0
      },
      {
        "ph": "e",
        "name": "zero_async_num",
        "pid": 1,
        "tid": 2,
        "id": 0,
        "ts": 25.0
      },
      {
        "ph": "s",
        "name": "zero_flow_start",
        "pid": 1,
        "tid": 2,
        "id": 0,
        "ts": 10.0
      },
      {
        "ph": "f",
        "name": "zero_flow_end",
        "pid": 1,
        "tid": 2,
        "id": "0",
        "ts": 25.0
      }
    ]
  })");

  ParsedTraceEvents result =
      ParseTraceEvents(trace_data, emscripten::val::null());

  ASSERT_EQ(result.flame_events.size(), 3);
  EXPECT_EQ(result.flame_events[0].name, "zero_async_num");
  EXPECT_TRUE(result.flame_events[0].is_async);
  EXPECT_EQ(result.flame_events[0].flow_id, 0);
  EXPECT_TRUE(result.flame_events[0].has_flow_id);
  EXPECT_DOUBLE_EQ(result.flame_events[0].dur, 15.0);

  EXPECT_EQ(result.flame_events[1].ph, Phase::kFlowStart);
  EXPECT_EQ(result.flame_events[1].flow_id, 0);
  EXPECT_TRUE(result.flame_events[1].has_flow_id);

  EXPECT_EQ(result.flame_events[2].ph, Phase::kFlowEnd);
  EXPECT_EQ(result.flame_events[2].flow_id, 0);
  EXPECT_TRUE(result.flame_events[2].has_flow_id);
}

TEST(TraceEventParserTest,
     ConcurrentAsyncEventsWithSamePidAndIdDifferentNamesDoNotCollide) {
  emscripten::val trace_data = ParseJson(R"({
    "traceEvents": [
      {
        "ph": "b",
        "name": "op_a",
        "pid": 1,
        "tid": 2,
        "id": 42,
        "ts": 10.0
      },
      {
        "ph": "b",
        "name": "op_b",
        "pid": 1,
        "tid": 2,
        "id": 42,
        "ts": 15.0
      },
      {
        "ph": "e",
        "name": "op_a",
        "pid": 1,
        "tid": 2,
        "id": 42,
        "ts": 30.0
      },
      {
        "ph": "e",
        "name": "op_b",
        "pid": 1,
        "tid": 2,
        "id": 42,
        "ts": 50.0
      }
    ]
  })");

  ParsedTraceEvents result =
      ParseTraceEvents(trace_data, emscripten::val::null());

  ASSERT_EQ(result.flame_events.size(), 2);
  EXPECT_EQ(result.flame_events[0].name, "op_a");
  EXPECT_DOUBLE_EQ(result.flame_events[0].ts, 10.0);
  EXPECT_DOUBLE_EQ(result.flame_events[0].dur, 20.0);

  EXPECT_EQ(result.flame_events[1].name, "op_b");
  EXPECT_DOUBLE_EQ(result.flame_events[1].ts, 15.0);
  EXPECT_DOUBLE_EQ(result.flame_events[1].dur, 35.0);
}

TEST(TraceEventParserTest, CounterEventParsesEventStats) {
  emscripten::val trace_data = ParseJson(R"({
    "traceEvents": [
      {
        "ph": "C",
        "name": "COUNT_MXU_BUSY_0",
        "pid": 1,
        "event_stats": "percentage",
        "entries": [[10.0, 25.0], [20.0, 75.0]]
      },
      {
        "ph": "C",
        "name": "Uncategorized Counter",
        "pid": 1,
        "entries": [[10.0, 1.0], [20.0, 2.0]]
      }
    ]
  })");

  ParsedTraceEvents result =
      ParseTraceEvents(trace_data, emscripten::val::null());

  ASSERT_EQ(result.counter_events.size(), 2);
  EXPECT_EQ(result.counter_events[0].name, "COUNT_MXU_BUSY_0");
  EXPECT_EQ(result.counter_events[0].event_stats, "percentage");
  ASSERT_EQ(result.counter_events[0].timestamps.size(), 2);
  EXPECT_DOUBLE_EQ(result.counter_events[0].timestamps[0], 10.0);
  EXPECT_DOUBLE_EQ(result.counter_events[0].timestamps[1], 20.0);
  ASSERT_EQ(result.counter_events[0].values.size(), 2);
  EXPECT_DOUBLE_EQ(result.counter_events[0].values[0], 25.0);
  EXPECT_DOUBLE_EQ(result.counter_events[0].values[1], 75.0);
  EXPECT_EQ(result.counter_events[1].name, "Uncategorized Counter");
  EXPECT_EQ(result.counter_events[1].event_stats, "");
}

TEST(TraceEventParserTest, CounterEventUnit) {
  emscripten::val trace_data = ParseJson(R"({
    "traceEvents": [
      {"ph": "C", "pid": 1, "name": "no unit", "entries": [[0, 1]]},
      {"ph": "C", "pid": 1, "name": "fraction", "unit": "fraction",
       "entries": [[0, 0.5]]},
      {"ph": "C", "pid": 1, "name": "bytes", "unit": "bytes",
       "entries": [[0, 1024]]},
      {"ph": "C", "pid": 1, "name": "unknown", "unit": "furlongs",
       "entries": [[0, 1]]},
      {"ph": "C", "pid": 1, "name": "not a string", "unit": 7,
       "entries": [[0, 1]]}
    ]
  })");

  ParsedTraceEvents result =
      ParseTraceEvents(trace_data, emscripten::val::null());

  EXPECT_THAT(result.counter_events,
              ElementsAre(Field(&CounterEvent::unit, CounterUnit::kNone),
                          Field(&CounterEvent::unit, CounterUnit::kFraction),
                          Field(&CounterEvent::unit, CounterUnit::kBytes),
                          Field(&CounterEvent::unit, CounterUnit::kNone),
                          Field(&CounterEvent::unit, CounterUnit::kNone)));
}

}  // namespace
}  // namespace traceviewer
