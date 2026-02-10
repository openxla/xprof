/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "xprof/convert/trace_viewer/trace_events_to_json.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "testing/base/public/gmock.h"
#include "<gtest/gtest.h>"
#include "absl/container/btree_map.h"
#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "google/protobuf/map.h"
#include "xprof/convert/trace_viewer/trace_viewer_color.h"
#include "plugin/xprof/protobuf/task.pb.h"
#include "plugin/xprof/protobuf/trace_events.pb.h"
#include "plugin/xprof/protobuf/trace_events_raw.pb.h"

namespace tensorflow {
namespace profiler {
namespace {

using ::testing::HasSubstr;
using ::testing::Not;

class TestTraceEventsContainer {
 public:
  void AddEvent(const TraceEvent& event) { events_.push_back(event); }

  void SetTrace(const Trace& trace) { trace_ = trace; }

  const Trace& trace() const { return trace_; }

  size_t NumEvents() const { return events_.size(); }

  bool FilterByVisibility() const { return false; }

  template <typename Callback>
  void ForAllDeviceFirstEvents(Callback callback) const {
    absl::flat_hash_set<uint32_t> visited_devices;
    for (const TraceEvent& event : events_) {
      if (visited_devices.insert(event.device_id()).second) {
        callback(event);
      }
    }
  }

  template <typename Callback>
  void ForAllEvents(Callback callback) const {
    for (const TraceEvent& event : events_) {
      callback(event);
    }
  }

 private:
  std::vector<TraceEvent> events_;
  Trace trace_;
};

using TraceEventsContainer = TestTraceEventsContainer;

class MockTraceEventsColorer : public TraceEventsColorerInterface {
 public:
  void SetUp(const Trace& trace) override {}
  std::optional<uint32_t> GetColor(const TraceEvent& event) const override {
    return 1;
  }
};

TEST(TraceEventsToJsonTest, PicosToMicrosTest) {
  EXPECT_DOUBLE_EQ(PicosToMicros(1000000), 1.0);
  EXPECT_DOUBLE_EQ(PicosToMicros(1), 1E-6);
  EXPECT_DOUBLE_EQ(PicosToMicros(1234567), 1.234567);
}

TEST(TraceEventsToJsonTest, JsonEscapeTest) {
  EXPECT_EQ(JsonEscape(""), R"("")");
  EXPECT_EQ(JsonEscape("abc"), R"("abc")");
  EXPECT_EQ(JsonEscape("a\"b\\c"), R"("a\"b\\c")");
  EXPECT_EQ(JsonEscape("a\nb\rc\td\be\ff"), R"("a\nb\rc\td\be\ff")");
  EXPECT_EQ(JsonEscape("a<b"), R"("a\u003cb")");
  EXPECT_EQ(JsonEscape("b>c"), R"("b\u003ec")");
  EXPECT_EQ(JsonEscape("c&d"), R"("c\u0026d")");
  EXPECT_EQ(JsonEscape("\xe2\x80\xa8"), R"("\u2028")");
  EXPECT_EQ(JsonEscape("\xe2\x80\xa9"), R"("\u2029")");
}

TEST(TraceEventsToJsonTest, BuildStackFrameReferencesTest) {
  Trace trace;
  google::protobuf::Map<uint64_t, std::string>& name_table = *trace.mutable_name_table();
  name_table[1] = "abc";
  name_table[2] = "@@stack1";
  name_table[3] = "def";
  name_table[4] = "@@stack2";
  absl::btree_map<uint64_t, uint64_t> references =
      BuildStackFrameReferences(trace);
  ASSERT_EQ(references.size(), 2);
  EXPECT_EQ(references[2], 1);
  EXPECT_EQ(references[4], 2);
}

TEST(TraceEventsToJsonTest, JsonEventCounterTest) {
  JsonEventCounter counter;
  EXPECT_EQ(counter.GetCounterEventCount(), 0);
  counter.Inc(JsonEventCounter::kCompleteEvent);
  counter.Inc(JsonEventCounter::kCompleteEventWithFlow);
  counter.Inc(JsonEventCounter::kCounterEvent);
  counter.Inc(JsonEventCounter::kAsyncEvent);
  counter.Inc(JsonEventCounter::kCounterEvent);
  EXPECT_EQ(counter.GetCounterEventCount(), 2);
  EXPECT_EQ(counter.ToString(),
            "Generated JSON events: complete: 1 "
            "complete+flow: 1 counter: 2 async: 1");
}

template <typename T>
class JsonSeparatorTypedTest : public ::testing::Test {};
using IOBufferTypes = ::testing::Types<IOBufferAdapter>;
TYPED_TEST_SUITE(JsonSeparatorTypedTest, IOBufferTypes);

TYPED_TEST(JsonSeparatorTypedTest, SeparatorTest) {
  std::string output_str;
  TypeParam output(&output_str);
  JsonSeparator<TypeParam> separator(&output);
  EXPECT_EQ(output_str, "");
  separator.Add();
  EXPECT_EQ(output_str, "");
  separator.Add();
  EXPECT_EQ(output_str, ",");
  separator.Add();
  EXPECT_EQ(output_str, ",,");
}

TEST(TraceEventsToJsonTest, IOBufferAdapterTest) {
  std::string output_str;
  IOBufferAdapter output(&output_str);
  output.Append("hello");
  EXPECT_EQ(output_str, "hello");
  output.Append(" world", "!");
  EXPECT_EQ(output_str, "hello world!");
}

TEST(TraceEventsToJsonTest, ProtoStringTest) {
  TraceEvent event;
  event.set_device_id(123);
  event.set_name("test_event");
  EXPECT_EQ(ProtoString(event), JsonEscape(event.DebugString()));
}

template <typename T>
class WriteDetailsTypedTest : public ::testing::Test {};
TYPED_TEST_SUITE(WriteDetailsTypedTest, IOBufferTypes);

TYPED_TEST(WriteDetailsTypedTest, WriteDetailsTest) {
  std::string output_str;
  TypeParam output(&output_str);
  JsonTraceOptions::Details details = {{"detail1", true}, {"detail2", false}};
  WriteDetails(details, &output);
  EXPECT_EQ(
      output_str,
      R"("details":[{"name":"detail1","value":true},{"name":"detail2","value":false}],)");
}

template <typename T>
class WriteReturnedEventsSizeTypedTest : public ::testing::Test {};
TYPED_TEST_SUITE(WriteReturnedEventsSizeTypedTest, IOBufferTypes);

TYPED_TEST(WriteReturnedEventsSizeTypedTest, WriteReturnedEventsSizeTest) {
  std::string output_str;
  TypeParam output(&output_str);
  WriteReturnedEventsSize(123, &output);
  EXPECT_EQ(output_str, R"("returnedEventsSize":123,)");
}

template <typename T>
class WriteFilteredByVisibilityTypedTest : public ::testing::Test {};
TYPED_TEST_SUITE(WriteFilteredByVisibilityTypedTest, IOBufferTypes);

TYPED_TEST(WriteFilteredByVisibilityTypedTest, WriteFilteredByVisibilityTest) {
  std::string output_str;
  TypeParam output(&output_str);
  WriteFilteredByVisibility(true, &output);
  EXPECT_EQ(output_str, R"("filteredByVisibility":true,)");
}

template <typename T>
class WriteTraceFullTimespanTypedTest : public ::testing::Test {};
TYPED_TEST_SUITE(WriteTraceFullTimespanTypedTest, IOBufferTypes);

TYPED_TEST(WriteTraceFullTimespanTypedTest, WriteTraceFullTimespanTest) {
  Trace trace;
  trace.set_min_timestamp_ps(1000000000);
  trace.set_max_timestamp_ps(2000000000);
  std::string output_str;
  TypeParam output(&output_str);
  WriteTraceFullTimespan(&trace, &output);
  EXPECT_EQ(output_str, R"("fullTimespan":[1,2],)");
}

template <typename T>
class WriteStackFramesTypedTest : public ::testing::Test {};
TYPED_TEST_SUITE(WriteStackFramesTypedTest, IOBufferTypes);

TYPED_TEST(WriteStackFramesTypedTest, WriteStackFramesTest) {
  Trace trace;
  google::protobuf::Map<uint64_t, std::string>& name_table = *trace.mutable_name_table();
  name_table[1] = "abc";
  name_table[2] = "@@stack1";
  name_table[3] = "@@stack2";
  absl::btree_map<uint64_t, uint64_t> references =
      BuildStackFrameReferences(trace);
  std::string output_str;
  TypeParam output(&output_str);
  WriteStackFrames(trace, references, &output);
  EXPECT_THAT(output_str, HasSubstr(R"("1":{"name":"stack1"})"));
  EXPECT_THAT(output_str, HasSubstr(R"("2":{"name":"stack2"})"));
}

template <typename T>
class WriteTasksTypedTest : public ::testing::Test {};
TYPED_TEST_SUITE(WriteTasksTypedTest, IOBufferTypes);

TYPED_TEST(WriteTasksTypedTest, WriteTasksTest) {
  Trace trace;
  Task& task = (*trace.mutable_tasks())[123];
  task.set_changelist(12345);
  std::string output_str;
  TypeParam output(&output_str);
  WriteTasks(trace, &output);
  EXPECT_EQ(output_str, R"("tasks":[{"host_id":123,"changelist":12345}],)");
}

template <typename T>
class JsonEventWriterTypedTest : public ::testing::Test {};
TYPED_TEST_SUITE(JsonEventWriterTypedTest, IOBufferTypes);

TYPED_TEST(JsonEventWriterTypedTest, CompleteEventTest) {
  Trace trace;
  DefaultTraceEventsColorer colorer;
  absl::btree_map<uint64_t, uint64_t> references;
  std::string output_str;
  TypeParam output(&output_str);
  JsonEventWriter<TypeParam, RawData> writer(&colorer, trace, references,
                                             &output);

  TraceEvent event;
  event.set_device_id(1);
  event.set_resource_id(2);
  event.set_name("complete_event");
  event.set_timestamp_ps(1000000);
  event.set_duration_ps(500000);
  writer.WriteEvent(event);

  EXPECT_EQ(
      output_str,
      R"({"pid":1,"tid":2,"name":"complete_event","ts":1,"dur":0.5,"ph":"X"})");
}

TYPED_TEST(JsonEventWriterTypedTest, CompleteEventWithFlowTest) {
  Trace trace;
  DefaultTraceEventsColorer colorer;
  absl::btree_map<uint64_t, uint64_t> references;
  std::string output_str;
  TypeParam output(&output_str);
  JsonEventWriter<TypeParam, RawData> writer(&colorer, trace, references,
                                             &output);

  TraceEvent event;
  event.set_device_id(1);
  event.set_resource_id(2);
  event.set_name("complete_event_with_flow");
  event.set_timestamp_ps(1000000);
  event.set_duration_ps(500000);
  event.set_flow_id(123);
  event.set_flow_entry_type(TraceEvent::FLOW_START);
  writer.WriteEvent(event);

  EXPECT_EQ(
      output_str,
      R"({"pid":1,"tid":2,"name":"complete_event_with_flow","ts":1,"dur":0.5,"bind_id":123,"flow_out":true,"ph":"X"})");
}

TYPED_TEST(JsonEventWriterTypedTest, CounterEventTest) {
  Trace trace;
  DefaultTraceEventsColorer colorer;
  absl::btree_map<uint64_t, uint64_t> references;
  std::string output_str;
  TypeParam output(&output_str);
  JsonEventWriter<TypeParam, RawData> writer(&colorer, trace, references,
                                             &output);

  TraceEvent event;
  event.set_device_id(1);
  event.set_name("counter_event");
  event.set_timestamp_ps(1000000);
  RawData raw_data;
  TraceEventArguments::Argument& arg = *raw_data.mutable_args()->add_arg();
  arg.set_name("arg1");
  arg.set_int_value(100);
  event.set_raw_data(raw_data.SerializeAsString());
  writer.AddCounterEvent(event);

  output_str += "]}";

  EXPECT_EQ(
      output_str,
      R"({"pid":1,"name":"counter_event","ph":"C","event_stats":"arg1","entries":[[1,100]]})");
}

TYPED_TEST(JsonEventWriterTypedTest, AsyncEventTest) {
  Trace trace;
  DefaultTraceEventsColorer colorer;
  absl::btree_map<uint64_t, uint64_t> references;
  std::string output_str;
  TypeParam output(&output_str);
  JsonEventWriter<TypeParam, RawData> writer(&colorer, trace, references,
                                             &output);

  TraceEvent event;
  event.set_device_id(1);
  event.set_name("async_event");
  event.set_timestamp_ps(1000000);
  event.set_flow_id(456);
  event.set_flow_entry_type(TraceEvent::FLOW_START);
  writer.WriteEvent(event);

  EXPECT_EQ(output_str,
            R"({"pid":1,"name":"async_event","ts":1,"id":456,"ph":"b"})");
}

TYPED_TEST(JsonEventWriterTypedTest, IsMatchingLastCounterEventTest) {
  Trace trace;
  DefaultTraceEventsColorer colorer;
  absl::btree_map<uint64_t, uint64_t> references;
  std::string output_str;
  TypeParam output(&output_str);
  JsonEventWriter<TypeParam, RawData> writer(&colorer, trace, references,
                                             &output);

  TraceEvent event1;
  event1.set_device_id(1);
  event1.set_name("counter_event");
  event1.set_timestamp_ps(1000000);
  RawData raw_data1;
  TraceEventArguments::Argument& arg1 = *raw_data1.mutable_args()->add_arg();
  arg1.set_name("arg1");
  arg1.set_int_value(100);
  event1.set_raw_data(raw_data1.SerializeAsString());
  writer.AddCounterEvent(event1);

  TraceEvent event2;
  event2.set_device_id(1);
  event2.set_name("counter_event");
  EXPECT_TRUE(writer.isMatchingLastCounterEvent(event2));

  TraceEvent event3;
  event3.set_device_id(2);
  event3.set_name("counter_event");
  EXPECT_FALSE(writer.isMatchingLastCounterEvent(event3));
}

TYPED_TEST(JsonEventWriterTypedTest, WriteEvent_Color) {
  Trace trace;
  MockTraceEventsColorer colorer;
  absl::btree_map<uint64_t, uint64_t> references;
  std::string output_str;
  TypeParam output(&output_str);
  JsonEventWriter<TypeParam, RawData> writer(&colorer, trace, references,
                                             &output);

  TraceEvent event;
  event.set_device_id(1);
  event.set_resource_id(2);
  event.set_name("color_event");
  event.set_timestamp_ps(1000000);
  event.set_duration_ps(500000);
  writer.WriteEvent(event);

  EXPECT_THAT(output_str, HasSubstr(R"("cname":)"));
}

TYPED_TEST(JsonEventWriterTypedTest, WriteEvent_Flows) {
  Trace trace;
  DefaultTraceEventsColorer colorer;
  absl::btree_map<uint64_t, uint64_t> references;
  std::string output_str;
  TypeParam output(&output_str);
  JsonEventWriter<TypeParam, RawData> writer(&colorer, trace, references,
                                             &output);

  // Flow Start
  {
    output_str.clear();
    TraceEvent event;
    event.set_device_id(1);
    event.set_resource_id(2);
    event.set_name("flow_event");
    event.set_timestamp_ps(1000000);
    event.set_duration_ps(1000);
    event.set_flow_id(100);
    event.set_flow_entry_type(TraceEvent::FLOW_START);
    event.set_flow_category(1);  // Assuming 1 maps to something or generic
    writer.WriteEvent(event);
    EXPECT_THAT(output_str, HasSubstr(R"("flow_out":true)"));
    EXPECT_THAT(output_str, HasSubstr(R"("bind_id":100)"));
  }

  // Flow Mid
  {
    output_str.clear();
    TraceEvent event;
    event.set_device_id(1);
    event.set_resource_id(2);
    event.set_name("flow_event");
    event.set_timestamp_ps(2000000);
    event.set_duration_ps(1000);
    event.set_flow_id(100);
    event.set_flow_entry_type(TraceEvent::FLOW_MID);
    writer.WriteEvent(event);
    EXPECT_THAT(output_str, HasSubstr(R"("flow_in":true)"));
    EXPECT_THAT(output_str, HasSubstr(R"("flow_out":true)"));
  }

  // Flow End
  {
    output_str.clear();
    TraceEvent event;
    event.set_device_id(1);
    event.set_resource_id(2);
    event.set_name("flow_event");
    event.set_timestamp_ps(3000000);
    event.set_duration_ps(1000);
    event.set_flow_id(100);
    event.set_flow_entry_type(TraceEvent::FLOW_END);
    writer.WriteEvent(event);
    EXPECT_THAT(output_str, HasSubstr(R"("flow_in":true)"));
    EXPECT_THAT(output_str, Not(HasSubstr(R"("flow_out":true)")));
  }
}

TYPED_TEST(JsonEventWriterTypedTest, WriteEvent_Async_FlowMid) {
  Trace trace;
  DefaultTraceEventsColorer colorer;
  absl::btree_map<uint64_t, uint64_t> references;
  std::string output_str;
  TypeParam output(&output_str);
  JsonEventWriter<TypeParam, RawData> writer(&colorer, trace, references,
                                             &output);

  TraceEvent event;
  event.set_device_id(1);
  event.set_name("async_event");
  event.set_timestamp_ps(1000000);
  event.set_duration_ps(500000);
  event.set_flow_id(456);
  event.set_flow_entry_type(TraceEvent::FLOW_MID);

  writer.WriteEvent(event);

  // Expect two events: one for begin (ph:b) and one for end (ph:e)
  // The first one is the original event (modified to be 'b' implicitly by the
  // switch logic which appends "ph":"b") The second one is the emplaced
  // async_event which is set to 'e' Actually, the code appends "ph":"b" for
  // FLOW_MID, then emplaces a new event for FLOW_END.

  EXPECT_THAT(output_str, HasSubstr(R"("ph":"b")"));
  EXPECT_THAT(output_str, HasSubstr(R"("ph":"e")"));
  EXPECT_THAT(output_str, HasSubstr(R"("ts":1)"));    // Start
  EXPECT_THAT(output_str, HasSubstr(R"("ts":1.5)"));  // End (1 + 0.5)
}

TYPED_TEST(JsonEventWriterTypedTest, WriteEvent_Args) {
  Trace trace;
  google::protobuf::Map<uint64_t, std::string>& name_table = *trace.mutable_name_table();
  name_table[1] = "ref_value";
  name_table[2] = "@@stack_frame";
  DefaultTraceEventsColorer colorer;
  absl::btree_map<uint64_t, uint64_t> references;
  references[2] = 99;  // stack frame ref
  std::string output_str;
  TypeParam output(&output_str);
  JsonEventWriter<TypeParam, RawData> writer(&colorer, trace, references,
                                             &output);

  TraceEvent event;
  event.set_device_id(1);
  event.set_resource_id(2);
  event.set_name("args_event");
  event.set_timestamp_ps(1000);
  event.set_duration_ps(1000);
  event.set_group_id(10);
  event.set_serial(123456);

  RawData raw_data;
  {
    TraceEventArguments::Argument& arg = *raw_data.mutable_args()->add_arg();
    arg.set_name("str_arg");
    arg.set_str_value("value");
  }
  {
    TraceEventArguments::Argument& arg = *raw_data.mutable_args()->add_arg();
    arg.set_name("int_arg");
    arg.set_int_value(42);
  }
  {
    TraceEventArguments::Argument& arg = *raw_data.mutable_args()->add_arg();
    arg.set_name("uint_arg");
    arg.set_uint_value(100);
  }
  {
    TraceEventArguments::Argument& arg = *raw_data.mutable_args()->add_arg();
    arg.set_name("double_arg");
    arg.set_double_value(3.14);
  }
  {
    TraceEventArguments::Argument& arg = *raw_data.mutable_args()->add_arg();
    arg.set_name("ref_arg");
    arg.set_ref_value(1);
  }
  {
    TraceEventArguments::Argument& arg = *raw_data.mutable_args()->add_arg();
    arg.set_name("stack_arg");
    arg.set_ref_value(2);  // Should be treated as stack frame
  }

  event.set_raw_data(raw_data.SerializeAsString());
  writer.WriteEvent(event);

  EXPECT_THAT(output_str, HasSubstr(R"("args":{)"));
  EXPECT_THAT(output_str, HasSubstr(R"("group_id":10)"));
  EXPECT_THAT(output_str, HasSubstr(R"("str_arg":"value")"));
  EXPECT_THAT(output_str, HasSubstr(R"("int_arg":42)"));
  EXPECT_THAT(output_str, HasSubstr(R"("uint_arg":100)"));
  EXPECT_THAT(output_str, HasSubstr(R"("double_arg":3.14)"));
  EXPECT_THAT(output_str, HasSubstr(R"("ref_arg":"ref_value")"));
  EXPECT_THAT(output_str, HasSubstr(R"("sf":99)"));
  EXPECT_THAT(output_str, HasSubstr(R"("z":123456)"));
}

TYPED_TEST(JsonEventWriterTypedTest, CounterEventsGroupingTest) {
  Trace trace;
  DefaultTraceEventsColorer colorer;
  absl::btree_map<uint64_t, uint64_t> references;
  std::string output_str;
  TypeParam output(&output_str);
  JsonEventWriter<TypeParam, RawData> writer(&colorer, trace, references,
                                             &output);

  TraceEvent event1;
  event1.set_device_id(1);
  event1.set_name("counter");
  event1.set_timestamp_ps(1000000);  // 1.0 us
  {
    RawData raw_data;
    TraceEventArguments::Argument& arg = *raw_data.mutable_args()->add_arg();
    arg.set_name("val");
    arg.set_int_value(10);
    event1.set_raw_data(raw_data.SerializeAsString());
  }
  writer.AddCounterEvent(event1);

  output.Append(",");

  TraceEvent event2;
  event2.set_device_id(1);
  event2.set_name("counter");
  event2.set_timestamp_ps(2000000);  // 2.0 us
  {
    RawData raw_data;
    TraceEventArguments::Argument& arg = *raw_data.mutable_args()->add_arg();
    arg.set_name("val");
    arg.set_int_value(20);
    event2.set_raw_data(raw_data.SerializeAsString());
  }
  writer.AddCounterEvent(event2);

  output_str += "]}";

  EXPECT_THAT(output_str, HasSubstr(R"("ph":"C")"));
  EXPECT_THAT(output_str, HasSubstr(R"("entries":[[1,10],[2,20]]})"));
}

template <typename T>
class TraceEventsToJsonTypedTest : public ::testing::Test {};
TYPED_TEST_SUITE(TraceEventsToJsonTypedTest, IOBufferTypes);

TYPED_TEST(TraceEventsToJsonTypedTest, IntegrationTest) {
  Trace trace;
  google::protobuf::Map<uint64_t, Resource>& device =
      *(*trace.mutable_devices())[1].mutable_resources();
  device[1].set_name("thread1");
  (*trace.mutable_devices())[1].set_name("device1");

  TraceEvent event;
  event.set_device_id(1);
  event.set_resource_id(1);
  event.set_name("event1");
  event.set_timestamp_ps(1000);
  event.set_duration_ps(1000);

  TraceEventsContainer events;
  events.AddEvent(event);
  events.SetTrace(trace);

  JsonTraceOptions options;
  options.mpmd_pipeline_view = true;
  options.generate_stack_frames = true;

  std::string output_str;
  TypeParam output(&output_str);

  TraceEventsToJson<TypeParam, TraceEventsContainer, RawData>(options, events,
                                                              &output);

  EXPECT_THAT(output_str, HasSubstr(R"("traceEvents":[)"));
  EXPECT_THAT(output_str, HasSubstr(R"("process_name")"));
  EXPECT_THAT(output_str, HasSubstr(R"("thread_name")"));
  EXPECT_THAT(output_str, HasSubstr(R"("mpmdPipelineView": true)"));
}

template <typename T>
class JsonEventCounterDeathTest : public ::testing::Test {};
TYPED_TEST_SUITE(JsonEventCounterDeathTest, IOBufferTypes);

// Only checking that it runs without crashing, verifying log output is harder
// in unit tests without capturing stderr.

TYPED_TEST(JsonEventCounterDeathTest, DestructorLogs) {
  {
    JsonEventCounter counter;
    counter.Inc(JsonEventCounter::kCompleteEvent);
  }
}

TEST(BuildMpmdDependencyGraphTest, NoDependencies) {
  absl::flat_hash_map<uint32_t, absl::flat_hash_map<std::string, int>>
      device_program_min_layer;
  device_program_min_layer[1] = {{"program1", 0}};
  device_program_min_layer[2] = {{"program2", 0}};

  const internal::MpmdDependencyGraph graph =
      internal::BuildMpmdDependencyGraph(device_program_min_layer);

  EXPECT_TRUE(graph.adj.empty());
  EXPECT_EQ(graph.in_degree.size(), 2);
  EXPECT_EQ(graph.in_degree.at(1), 0);
  EXPECT_EQ(graph.in_degree.at(2), 0);
}

TEST(BuildMpmdDependencyGraphTest, SimpleLinearDependencies) {
  absl::flat_hash_map<uint32_t, absl::flat_hash_map<std::string, int>>
      device_program_min_layer;
  device_program_min_layer[1] = {{"program1", 0}};
  device_program_min_layer[2] = {{"program1", 1}};
  device_program_min_layer[3] = {{"program1", 2}};

  const internal::MpmdDependencyGraph graph =
      internal::BuildMpmdDependencyGraph(device_program_min_layer);

  EXPECT_EQ(graph.adj.size(), 2);
  EXPECT_THAT(graph.adj.at(1), testing::ElementsAre(2));
  EXPECT_THAT(graph.adj.at(2), testing::ElementsAre(3));
  EXPECT_EQ(graph.in_degree.size(), 3);
  EXPECT_EQ(graph.in_degree.at(1), 0);
  EXPECT_EQ(graph.in_degree.at(2), 1);
  EXPECT_EQ(graph.in_degree.at(3), 1);
}

TEST(BuildMpmdDependencyGraphTest, MultipleDependencies) {
  absl::flat_hash_map<uint32_t, absl::flat_hash_map<std::string, int>>
      device_program_min_layer;
  device_program_min_layer[1] = {{"program1", 0}};
  device_program_min_layer[2] = {{"program1", 1}};
  device_program_min_layer[3] = {{"program1", 1}};
  device_program_min_layer[4] = {{"program1", 2}};

  const internal::MpmdDependencyGraph graph =
      internal::BuildMpmdDependencyGraph(device_program_min_layer);

  EXPECT_EQ(graph.adj.size(), 3);
  EXPECT_THAT(graph.adj.at(1), testing::ElementsAre(2, 3));
  EXPECT_THAT(graph.adj.at(2), testing::ElementsAre(4));
  EXPECT_THAT(graph.adj.at(3), testing::ElementsAre(4));
  EXPECT_EQ(graph.in_degree.size(), 4);
  EXPECT_EQ(graph.in_degree.at(1), 0);
  EXPECT_EQ(graph.in_degree.at(2), 1);
  EXPECT_EQ(graph.in_degree.at(3), 1);
  EXPECT_EQ(graph.in_degree.at(4), 2);
}

TEST(PerformMpmdTopologicalSortTest, NoDependencies) {
  absl::flat_hash_map<uint32_t, absl::btree_set<uint32_t>> adj;
  absl::btree_map<uint32_t, int> in_degree;
  in_degree[1] = 0;
  in_degree[2] = 0;
  absl::flat_hash_map<uint32_t, absl::flat_hash_map<std::string, int>>
      device_program_min_layer;
  device_program_min_layer[1] = {};
  device_program_min_layer[2] = {};
  absl::flat_hash_map<uint32_t, uint32_t> device_to_sort_index;

  internal::PerformMpmdTopologicalSort(adj, std::move(in_degree),
                                       device_program_min_layer,
                                       device_to_sort_index);

  EXPECT_EQ(device_to_sort_index.size(), 2);
  EXPECT_EQ(device_to_sort_index[1], 0);
  EXPECT_EQ(device_to_sort_index[2], 1);
}

TEST(PerformMpmdTopologicalSortTest, SimpleLinearDependencies) {
  absl::flat_hash_map<uint32_t, absl::btree_set<uint32_t>> adj;
  adj[1] = {2};
  adj[2] = {3};
  absl::btree_map<uint32_t, int> in_degree;
  in_degree[1] = 0;
  in_degree[2] = 1;
  in_degree[3] = 1;
  absl::flat_hash_map<uint32_t, absl::flat_hash_map<std::string, int>>
      device_program_min_layer;
  device_program_min_layer[1] = {};
  device_program_min_layer[2] = {};
  device_program_min_layer[3] = {};
  absl::flat_hash_map<uint32_t, uint32_t> device_to_sort_index;

  internal::PerformMpmdTopologicalSort(adj, std::move(in_degree),
                                       device_program_min_layer,
                                       device_to_sort_index);

  EXPECT_EQ(device_to_sort_index.size(), 3);
  EXPECT_EQ(device_to_sort_index[1], 0);
  EXPECT_EQ(device_to_sort_index[2], 1);
  EXPECT_EQ(device_to_sort_index[3], 2);
}

TEST(PerformMpmdTopologicalSortTest, MultipleDependencies) {
  absl::flat_hash_map<uint32_t, absl::btree_set<uint32_t>> adj;
  adj[1] = {2, 3};
  adj[2] = {4};
  adj[3] = {4};
  absl::btree_map<uint32_t, int> in_degree;
  in_degree[1] = 0;
  in_degree[2] = 1;
  in_degree[3] = 1;
  in_degree[4] = 2;
  absl::flat_hash_map<uint32_t, absl::flat_hash_map<std::string, int>>
      device_program_min_layer;
  device_program_min_layer[1] = {};
  device_program_min_layer[2] = {};
  device_program_min_layer[3] = {};
  device_program_min_layer[4] = {};
  absl::flat_hash_map<uint32_t, uint32_t> device_to_sort_index;

  internal::PerformMpmdTopologicalSort(adj, std::move(in_degree),
                                       device_program_min_layer,
                                       device_to_sort_index);

  EXPECT_EQ(device_to_sort_index.size(), 4);
  EXPECT_EQ(device_to_sort_index[1], 0);
  EXPECT_EQ(device_to_sort_index[2], 1);
  EXPECT_EQ(device_to_sort_index[3], 2);
  EXPECT_EQ(device_to_sort_index[4], 3);
}

TEST(PerformMpmdTopologicalSortTest, CyclicDependencies) {
  absl::flat_hash_map<uint32_t, absl::btree_set<uint32_t>> adj;
  adj[1] = {2};
  adj[2] = {3};
  adj[3] = {1};
  absl::btree_map<uint32_t, int> in_degree;
  in_degree[1] = 1;
  in_degree[2] = 1;
  in_degree[3] = 1;
  absl::flat_hash_map<uint32_t, absl::flat_hash_map<std::string, int>>
      device_program_min_layer;
  device_program_min_layer[1] = {};
  device_program_min_layer[2] = {};
  device_program_min_layer[3] = {};
  absl::flat_hash_map<uint32_t, uint32_t> device_to_sort_index;

  internal::PerformMpmdTopologicalSort(adj, std::move(in_degree),
                                       device_program_min_layer,
                                       device_to_sort_index);

  EXPECT_EQ(device_to_sort_index.size(), 3);
  EXPECT_EQ(device_to_sort_index[1], 0);
  EXPECT_EQ(device_to_sort_index[2], 1);
  EXPECT_EQ(device_to_sort_index[3], 2);
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
