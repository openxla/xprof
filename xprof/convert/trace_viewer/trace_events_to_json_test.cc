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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
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
using ::testing::UnorderedElementsAre;

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

TEST(NormalizeMpmdProgramKeyTest, StripsBucketSuffixes) {
  EXPECT_EQ(
      internal::NormalizeMpmdProgramKey("inc_prefill_step_4k_bucket_128k"),
      "inc_prefill_step_4k");
  EXPECT_EQ(
      internal::NormalizeMpmdProgramKey("inc_prefill_step_4k_bucket_512k"),
      "inc_prefill_step_4k");
  EXPECT_EQ(
      internal::NormalizeMpmdProgramKey("inc_prefill_step_32k_chunk_4096"),
      "inc_prefill_step_32k_chunk_4096");
}

TEST(ExtractMpmdModuleInfoTest, ParsesShardyModuleWithLoopAndLayer) {
  const std::optional<internal::MpmdModuleInfo> info =
      internal::ExtractMpmdModuleInfo(
          "p0_loop_0_layer_0_0.inc_prefill_step_32k_chunk_4096(12345)");
  ASSERT_TRUE(info.has_value());
  EXPECT_EQ(info->group_id, 0);
  EXPECT_EQ(info->loop_id, 0);
  EXPECT_EQ(info->min_layer, 0);
  EXPECT_TRUE(info->has_explicit_layer);
  EXPECT_EQ(info->program_key, "inc_prefill_step_32k_chunk_4096");

  const std::optional<internal::MpmdModuleInfo> info2 =
      internal::ExtractMpmdModuleInfo(
          "p24_loop_1_layer_24_24.inc_prefill_step_32k_chunk_4096(12345)");
  ASSERT_TRUE(info2.has_value());
  EXPECT_EQ(info2->group_id, 24);
  EXPECT_EQ(info2->loop_id, 1);
  EXPECT_EQ(info2->min_layer, 24);
  EXPECT_TRUE(info2->has_explicit_layer);
  EXPECT_EQ(info2->program_key, "inc_prefill_step_32k_chunk_4096");
}

TEST(ExtractMpmdModuleInfoTest, ParsesShardyModuleWithEllipsisInDescriptor) {
  const std::optional<internal::MpmdModuleInfo> info =
      internal::ExtractMpmdModuleInfo(
          "p0_loop_1_layer_0_3..._fwd.my_prog(123)");
  ASSERT_TRUE(info.has_value());
  EXPECT_EQ(info->group_id, 0);
  EXPECT_EQ(info->loop_id, 1);
  EXPECT_EQ(info->min_layer, 0);
  EXPECT_TRUE(info->has_explicit_layer);
  EXPECT_EQ(info->program_key, "my_prog");

  const std::optional<internal::MpmdModuleInfo> info2 =
      internal::ExtractMpmdModuleInfo("p0_foo..._fwd.my_prog(123)");
  ASSERT_TRUE(info2.has_value());
  EXPECT_EQ(info2->group_id, 0);
  EXPECT_EQ(info2->loop_id, 0);
  EXPECT_EQ(info2->min_layer, 0);
  EXPECT_FALSE(info2->has_explicit_layer);
  EXPECT_EQ(info2->program_key, "my_prog");
}

TEST(ExtractMpmdModuleInfoTest, ParsesShardyNonLayerProgram) {
  const std::optional<internal::MpmdModuleInfo> session_info =
      internal::ExtractMpmdModuleInfo(
          "p1_inferred.inc_prefill_session_32k(12345)");
  ASSERT_TRUE(session_info.has_value());
  EXPECT_EQ(session_info->group_id, 1);
  EXPECT_EQ(session_info->loop_id, 0);
  EXPECT_EQ(session_info->min_layer, 1);
  EXPECT_FALSE(session_info->has_explicit_layer);
  EXPECT_EQ(session_info->program_key, "inc_prefill_session_32k");

  const std::optional<internal::MpmdModuleInfo> final_info =
      internal::ExtractMpmdModuleInfo(
          "p0_inferred.inc_prefill_final_32k(12345)");
  ASSERT_TRUE(final_info.has_value());
  EXPECT_EQ(final_info->group_id, 0);
  EXPECT_EQ(final_info->loop_id, 0);
  EXPECT_EQ(final_info->min_layer, 0);
  EXPECT_FALSE(final_info->has_explicit_layer);
  EXPECT_EQ(final_info->program_key, "inc_prefill_final_32k");
}

TEST(ExtractMpmdModuleInfoTest, ParsesShardyStageWithBucket) {
  const std::optional<internal::MpmdModuleInfo> info =
      internal::ExtractMpmdModuleInfo(
          "p2_stage1.inc_prefill_step_4k_bucket_128k(12345)");
  ASSERT_TRUE(info.has_value());
  EXPECT_EQ(info->group_id, 2);
  EXPECT_EQ(info->loop_id, 0);
  EXPECT_EQ(info->min_layer, 1);
  EXPECT_TRUE(info->has_explicit_layer);
  EXPECT_EQ(info->program_key, "inc_prefill_step_4k");
}

TEST(ExtractMpmdModuleInfoTest, ParsesLegacyPatterns) {
  const std::optional<internal::MpmdModuleInfo> info =
      internal::ExtractMpmdModuleInfo("mesh_stage0");
  ASSERT_TRUE(info.has_value());
  EXPECT_EQ(info->min_layer, 0);
  EXPECT_TRUE(info->has_explicit_layer);

  const std::optional<internal::MpmdModuleInfo> info2 =
      internal::ExtractMpmdModuleInfo("module_layer_2_3.prog");
  ASSERT_TRUE(info2.has_value());
  EXPECT_EQ(info2->min_layer, 2);
  EXPECT_TRUE(info2->has_explicit_layer);
  EXPECT_EQ(info2->program_key, "prog");

  const std::optional<internal::MpmdModuleInfo> info3 =
      internal::ExtractMpmdModuleInfo("module_layer_5.my_prog(1)");
  ASSERT_TRUE(info3.has_value());
  EXPECT_EQ(info3->min_layer, 5);
  EXPECT_TRUE(info3->has_explicit_layer);
  EXPECT_EQ(info3->program_key, "my_prog");

  const std::optional<internal::MpmdModuleInfo> info_overflow =
      internal::ExtractMpmdModuleInfo("p99999999999999999999_stage0.prog(1)");
  EXPECT_FALSE(info_overflow.has_value());
}

TEST(SortMpmdDevicesTest, SortMpmdDevicesByLoopAndLayerWithinProgram) {
  Trace trace;
  TraceEventsContainer events;

  auto add_event = [&](uint32_t device_id, const std::string& name,
                       uint64_t ts) {
    (*trace.mutable_devices())[device_id]
        .mutable_resources()
        ->operator[](1)
        .set_name("XLA Modules");
    TraceEvent event;
    event.set_device_id(device_id);
    event.set_resource_id(1);
    event.set_name(name);
    event.set_timestamp_ps(ts);
    event.set_duration_ps(1000);
    events.AddEvent(event);
  };

  add_event(3, "p2_loop_1_layer_0_0.program(100)", 1000);
  add_event(2, "p1_loop_0_layer_1_1.program(100)", 1000);
  add_event(1, "p0_loop_0_layer_0_0.program(100)", 1000);
  events.SetTrace(trace);

  absl::flat_hash_map<uint32_t, uint32_t> device_to_sort_index;
  SortMpmdDevices(events, device_to_sort_index);

  EXPECT_EQ(device_to_sort_index.size(), 3);
  EXPECT_EQ(device_to_sort_index.at(1), 0);
  EXPECT_EQ(device_to_sort_index.at(2), 1);
  EXPECT_EQ(device_to_sort_index.at(3), 2);
}

TEST(SortMpmdDevicesTest, SortMpmdDevicesMultiProgramByEarliestTimestamp) {
  Trace trace;
  TraceEventsContainer events;

  auto add_event = [&](uint32_t device_id, const std::string& name,
                       uint64_t ts) {
    (*trace.mutable_devices())[device_id]
        .mutable_resources()
        ->operator[](1)
        .set_name("XLA Modules");
    TraceEvent event;
    event.set_device_id(device_id);
    event.set_resource_id(1);
    event.set_name(name);
    event.set_timestamp_ps(ts);
    event.set_duration_ps(1000);
    events.AddEvent(event);
  };

  // Program A (inc_prefill) runs at t = 79 ms on device 10 and 11.
  add_event(10, "p0_loop_0_layer_0_0.inc_prefill(1)", 79000000000ULL);
  add_event(11, "p1_loop_0_layer_1_1.inc_prefill(1)", 80000000000ULL);

  // Program B (local_recovery_prefill) runs at t = 5583 ms on device 20 and 21.
  add_event(20, "p0_loop_0_layer_0_0.local_recovery_prefill(1)",
            5583000000000ULL);
  add_event(21, "p1_loop_0_layer_1_1.local_recovery_prefill(1)",
            5584000000000ULL);

  events.SetTrace(trace);

  absl::flat_hash_map<uint32_t, uint32_t> device_to_sort_index;
  SortMpmdDevices(events, device_to_sort_index);

  EXPECT_EQ(device_to_sort_index.size(), 4);
  EXPECT_EQ(device_to_sort_index.at(10), 0);
  EXPECT_EQ(device_to_sort_index.at(11), 1);
  EXPECT_EQ(device_to_sort_index.at(20), 2);
  EXPECT_EQ(device_to_sort_index.at(21), 3);
}

TEST(SortMpmdDevicesTest,
     SortMpmdDevicesMakeSessionAndFinalSharedAndDedicatedDevices) {
  Trace trace;
  TraceEventsContainer events;

  auto add_event = [&](uint32_t device_id, const std::string& name,
                       uint64_t ts) {
    (*trace.mutable_devices())[device_id]
        .mutable_resources()
        ->operator[](1)
        .set_name("XLA Modules");
    TraceEvent event;
    event.set_device_id(device_id);
    event.set_resource_id(1);
    event.set_name(name);
    event.set_timestamp_ps(ts);
    event.set_duration_ps(1000);
    events.AddEvent(event);
  };

  // Dedicated make_session device 5 (Tier 2, starts at t = 10 ms).
  add_event(5, "p0_inferred.inc_prefill_session_32k(1)", 10000000000ULL);

  // Shared devices 0 and 1 execute make_session (Tier 2 at 62 ms),
  // then prefill (Tier 1 at 79 ms and 85 ms), then final (Tier 2 at 5000 ms).
  add_event(0, "p1_inferred.inc_prefill_session_32k(1)", 62000000000ULL);
  add_event(1, "p1_inferred.inc_prefill_session_32k(1)", 62000000000ULL);
  add_event(0, "p0_loop_0_layer_0_0.prefill(1)", 79000000000ULL);
  add_event(1, "p1_loop_0_layer_1_1.prefill(1)", 85000000000ULL);
  add_event(0, "p0_inferred.inc_prefill_final_32k(1)", 5000000000000ULL);
  add_event(1, "p0_inferred.inc_prefill_final_32k(1)", 5000000000000ULL);

  // Dedicated final device 6 (Tier 2, starts at t = 9000 ms).
  add_event(6, "p0_inferred.inc_prefill_final_32k(1)", 9000000000000ULL);

  events.SetTrace(trace);

  absl::flat_hash_map<uint32_t, uint32_t> device_to_sort_index;
  SortMpmdDevices(events, device_to_sort_index);

  EXPECT_EQ(device_to_sort_index.size(), 4);
  EXPECT_EQ(device_to_sort_index.at(5), 0);  // Dedicated session.
  EXPECT_EQ(device_to_sort_index.at(0), 1);  // Shared prefill layer 0.
  EXPECT_EQ(device_to_sort_index.at(1), 2);  // Shared prefill layer 1.
  EXPECT_EQ(device_to_sort_index.at(6), 3);  // Dedicated final.
}

TEST(SortMpmdDevicesTest,
     SortMpmdDevicesBucketNormalizationAndAdjacentSameStage) {
  Trace trace;
  TraceEventsContainer events;

  auto add_event = [&](uint32_t device_id, const std::string& name,
                       uint64_t ts) {
    (*trace.mutable_devices())[device_id]
        .mutable_resources()
        ->operator[](1)
        .set_name("XLA Modules");
    TraceEvent event;
    event.set_device_id(device_id);
    event.set_resource_id(1);
    event.set_name(name);
    event.set_timestamp_ps(ts);
    event.set_duration_ps(1000);
    events.AddEvent(event);
  };

  // Device 0 and Device 1 run stage 0 (bucket 128k).
  add_event(0, "p0_stage0.inc_prefill_step_4k_bucket_128k(1)", 1000);
  add_event(1, "p0_stage0.inc_prefill_step_4k_bucket_128k(1)", 1000);

  // Device 2 and Device 3 run stage 1 (bucket 512k).
  add_event(2, "p1_stage1.inc_prefill_step_4k_bucket_512k(1)", 1000);
  add_event(3, "p1_stage1.inc_prefill_step_4k_bucket_512k(1)", 1000);

  events.SetTrace(trace);

  absl::flat_hash_map<uint32_t, uint32_t> device_to_sort_index;
  SortMpmdDevices(events, device_to_sort_index);

  EXPECT_EQ(device_to_sort_index.size(), 4);
  EXPECT_EQ(device_to_sort_index.at(0), 0);
  EXPECT_EQ(device_to_sort_index.at(1), 1);
  EXPECT_EQ(device_to_sort_index.at(2), 2);
  EXPECT_EQ(device_to_sort_index.at(3), 3);
}

TEST(TraceEventsToJsonTest, UnmatchedDevicesOffsetBehindMpmdStages) {
  Trace trace;
  // Device 1: TPU with MPMD stage.
  (*trace.mutable_devices())[1].mutable_resources()->operator[](1).set_name(
      "XLA Modules");
  (*trace.mutable_devices())[1].set_name("/device:TPU:0");

  // Device 10: Host CPU without MPMD events.
  (*trace.mutable_devices())[10].mutable_resources()->operator[](1).set_name(
      "Host Thread");
  (*trace.mutable_devices())[10].set_name("/host:CPU:0");

  TraceEventsContainer events;
  TraceEvent event;
  event.set_device_id(1);
  event.set_resource_id(1);
  event.set_name("p0_stage0.program(1)");
  event.set_timestamp_ps(1000);
  event.set_duration_ps(1000);
  events.AddEvent(event);
  events.SetTrace(trace);

  JsonTraceOptions options;
  options.mpmd_pipeline_view = true;

  std::string output_str;
  IOBufferAdapter output(&output_str);
  TraceEventsToJson<IOBufferAdapter, TraceEventsContainer, RawData>(
      options, events, &output);

  // Device 1 gets MPMD sort_index: 0.
  // Device 10 gets fallback sort_index: kMpmdUnrankedSortIndexBase + 10.
  EXPECT_THAT(
      output_str,
      HasSubstr(R"({"args":{"sort_index":0},"name":"process_sort_index",)"
                R"("ph":"M","pid":1})"));
  EXPECT_THAT(
      output_str,
      HasSubstr(
          R"({"args":{"sort_index":1073741834},"name":"process_sort_index",)"
          R"("ph":"M","pid":10})"));
}

TEST(TraceEventsToJsonTest, IdleTpuCoresOrderedAfterMpmdStages) {
  Trace trace;
  // Device 0: Active TPU core 0 with hostname prefix.
  (*trace.mutable_devices())[0].mutable_resources()->operator[](1).set_name(
      "XLA Modules");
  (*trace.mutable_devices())[0].set_name("host0 /device:TPU:0");

  // Device 1: TPU core 1 with non-MPMD event and hostname prefix.
  (*trace.mutable_devices())[1].mutable_resources()->operator[](1).set_name(
      "XLA Modules");
  (*trace.mutable_devices())[1].set_name("host0 /device:TPU:1");

  // Device 10: Host CPU without MPMD events.
  (*trace.mutable_devices())[10].mutable_resources()->operator[](1).set_name(
      "Host Thread");
  (*trace.mutable_devices())[10].set_name("/host:CPU:0");

  TraceEventsContainer events;
  TraceEvent event0;
  event0.set_device_id(0);
  event0.set_resource_id(1);
  event0.set_name("p0_stage0.program(1)");
  event0.set_timestamp_ps(1000);
  event0.set_duration_ps(1000);
  events.AddEvent(event0);

  TraceEvent event1;
  event1.set_device_id(1);
  event1.set_resource_id(1);
  event1.set_name("non_mpmd_compute");
  event1.set_timestamp_ps(1000);
  event1.set_duration_ps(1000);
  events.AddEvent(event1);

  events.SetTrace(trace);

  JsonTraceOptions options;
  options.mpmd_pipeline_view = true;

  std::string output_str;
  IOBufferAdapter output(&output_str);
  TraceEventsToJson<IOBufferAdapter, TraceEventsContainer, RawData>(
      options, events, &output);

  // Device 0 (active TPU core 0) is ranked at sort_index 0.
  EXPECT_THAT(output_str, HasSubstr(R"("pid":0)"));
  EXPECT_THAT(
      output_str,
      HasSubstr(R"({"args":{"sort_index":0},"name":"process_sort_index",)"
                R"("ph":"M","pid":0})"));
  // Device 1 (TPU core 1 without MPMD module events) is preserved and gets
  // kMpmdUnrankedSortIndexBase + 1 (1073741825).
  EXPECT_THAT(output_str, HasSubstr(R"("pid":1)"));
  EXPECT_THAT(
      output_str,
      HasSubstr(
          R"({"args":{"sort_index":1073741825},"name":"process_sort_index",)"
          R"("ph":"M","pid":1})"));
  EXPECT_THAT(output_str, HasSubstr("non_mpmd_compute"));
  // Device 10 (/host:CPU:0) is preserved and gets
  // kMpmdUnrankedSortIndexBase + 10 (1073741834).
  EXPECT_THAT(output_str, HasSubstr(R"("pid":10)"));
  EXPECT_THAT(
      output_str,
      HasSubstr(
          R"({"args":{"sort_index":1073741834},"name":"process_sort_index",)"
          R"("ph":"M","pid":10})"));
}

TEST(TraceEventsToJsonTest, MpmdProcessMetadataIndependentOfLoadedWindow) {
  Trace trace;
  // Device 0: TPU Core 0 (Stage 0).
  (*trace.mutable_devices())[0].mutable_resources()->operator[](1).set_name(
      "XLA Modules");
  (*trace.mutable_devices())[0].set_name("host0 /device:TPU:0");

  // Device 1: TPU Core 1 (Stage 1).
  (*trace.mutable_devices())[1].mutable_resources()->operator[](1).set_name(
      "XLA Modules");
  (*trace.mutable_devices())[1].set_name("host0 /device:TPU:1");

  // Device 3: Idle TPU Core 3 without module events.
  (*trace.mutable_devices())[3].mutable_resources()->operator[](1).set_name(
      "XLA Modules");
  (*trace.mutable_devices())[3].set_name("host0 /device:TPU:3");

  TraceEvent event0;
  event0.set_device_id(0);
  event0.set_resource_id(1);
  event0.set_name("p0_stage0.program(1)");
  event0.set_timestamp_ps(100);
  event0.set_duration_ps(100);

  TraceEvent event1;
  event1.set_device_id(1);
  event1.set_resource_id(1);
  event1.set_name("p0_stage1.program(1)");
  event1.set_timestamp_ps(200);
  event1.set_duration_ps(100);

  // Full trace (K = 2): both Stage 0 and Stage 1 active.
  TraceEventsContainer full_events;
  full_events.AddEvent(event0);
  full_events.AddEvent(event1);
  full_events.SetTrace(trace);

  // Windowed trace (K = 1): only Stage 0 active.
  TraceEventsContainer windowed_events;
  windowed_events.AddEvent(event0);
  windowed_events.SetTrace(trace);

  JsonTraceOptions options;
  options.mpmd_pipeline_view = true;

  std::string full_output;
  IOBufferAdapter full_buffer(&full_output);
  TraceEventsToJson<IOBufferAdapter, TraceEventsContainer, RawData>(
      options, full_events, &full_buffer);

  std::string windowed_output;
  IOBufferAdapter windowed_buffer(&windowed_output);
  TraceEventsToJson<IOBufferAdapter, TraceEventsContainer, RawData>(
      options, windowed_events, &windowed_buffer);

  // In full load (K = 2), device 0 gets 0, device 1 gets 1.
  EXPECT_THAT(
      full_output,
      HasSubstr(R"({"args":{"sort_index":0},"name":"process_sort_index",)"
                R"("ph":"M","pid":0})"));
  EXPECT_THAT(
      full_output,
      HasSubstr(R"({"args":{"sort_index":1},"name":"process_sort_index",)"
                R"("ph":"M","pid":1})"));

  // In windowed load (K = 1), device 0 gets 0, device 1 gets Base + 1.
  EXPECT_THAT(
      windowed_output,
      HasSubstr(R"({"args":{"sort_index":0},"name":"process_sort_index",)"
                R"("ph":"M","pid":0})"));
  EXPECT_THAT(
      windowed_output,
      HasSubstr(
          R"({"args":{"sort_index":1073741825},"name":"process_sort_index",)"
          R"("ph":"M","pid":1})"));

  // For idle device 3, both full and windowed loads emit the identical
  // kMpmdUnrankedSortIndexBase + 3 (1073741827) sort index.
  EXPECT_THAT(
      full_output,
      HasSubstr(
          R"({"args":{"sort_index":1073741827},"name":"process_sort_index",)"
          R"("ph":"M","pid":3})"));
  EXPECT_THAT(
      windowed_output,
      HasSubstr(
          R"({"args":{"sort_index":1073741827},"name":"process_sort_index",)"
          R"("ph":"M","pid":3})"));
}

TEST(TraceEventsToJsonTest, MpmdPipelineViewDeviceOrderingBranches) {
  Trace trace;
  (*trace.mutable_name_table())[1] = "p0_stage0.prog_a(1)";

  // Device 0: TPU 0 with XLA Modules and XLA Ops resources.
  Device device0;
  device0.set_name("host0 /device:TPU:0");
  Resource resource_mod;
  resource_mod.set_name("XLA Modules");
  (*device0.mutable_resources())[1] = resource_mod;
  Resource resource_ops;
  resource_ops.set_name("XLA Ops");
  (*device0.mutable_resources())[2] = resource_ops;
  (*trace.mutable_devices())[0] = device0;

  // Device 1: TPU 1 with XLA Modules resource.
  Device device1;
  device1.set_name("host0 /device:TPU:1");
  (*device1.mutable_resources())[1] = resource_mod;
  (*trace.mutable_devices())[1] = device1;

  // Device 9: TPU 9 with no MPMD events, but with a non-module event on XLA
  // Ops.
  Device device9;
  device9.set_name("host0 /device:TPU:9");
  (*device9.mutable_resources())[2] = resource_ops;
  (*trace.mutable_devices())[9] = device9;

  TraceEventsContainer events;

  // 1. Non-module event on resource 2 (XLA Ops) on device 0 (early return).
  TraceEvent event_non_mod;
  event_non_mod.set_device_id(0);
  event_non_mod.set_resource_id(2);
  event_non_mod.set_name("op_kernel");
  event_non_mod.set_timestamp_ps(50);
  event_non_mod.set_duration_ps(10);
  events.AddEvent(event_non_mod);

  // 2. Event on device 0 using name_ref (has_name_ref() == true).
  TraceEvent event_ref;
  event_ref.set_device_id(0);
  event_ref.set_resource_id(1);
  event_ref.set_name_ref(1);
  event_ref.set_timestamp_ps(1000);
  event_ref.set_duration_ps(100);
  events.AddEvent(event_ref);

  // 3. Multiple Tier-1 programs on device 0 to exercise std::tie ordering.
  TraceEvent event_prog_y;
  event_prog_y.set_device_id(0);
  event_prog_y.set_resource_id(1);
  event_prog_y.set_name("p0_stage0.prog_y(1)");
  event_prog_y.set_timestamp_ps(1200);
  event_prog_y.set_duration_ps(50);
  events.AddEvent(event_prog_y);

  TraceEvent event_prog_z;
  event_prog_z.set_device_id(0);
  event_prog_z.set_resource_id(1);
  event_prog_z.set_name("p0_stage0.prog_z(1)");
  event_prog_z.set_timestamp_ps(1300);
  event_prog_z.set_duration_ps(50);
  events.AddEvent(event_prog_z);

  // 4. Device 1 running prog_b at the same timestamp (ts = 1000).
  TraceEvent event_prog_b;
  event_prog_b.set_device_id(1);
  event_prog_b.set_resource_id(1);
  event_prog_b.set_name("p0_stage0.prog_b(1)");
  event_prog_b.set_timestamp_ps(1000);
  event_prog_b.set_duration_ps(100);
  events.AddEvent(event_prog_b);

  // 5. Device 9 non-module event on XLA Ops.
  TraceEvent event_dev9;
  event_dev9.set_device_id(9);
  event_dev9.set_resource_id(2);
  event_dev9.set_name("hbm_transfer");
  event_dev9.set_timestamp_ps(1100);
  event_dev9.set_duration_ps(100);
  events.AddEvent(event_dev9);

  events.SetTrace(trace);

  JsonTraceOptions options;
  options.mpmd_pipeline_view = true;

  std::string output_str;
  IOBufferAdapter output(&output_str);
  TraceEventsToJson<IOBufferAdapter, TraceEventsContainer, RawData>(
      options, events, &output);

  EXPECT_THAT(output_str, HasSubstr(R"("args":{"name":"host0 /device:TPU:0")"));
  EXPECT_THAT(output_str, HasSubstr(R"("args":{"name":"host0 /device:TPU:1")"));
  EXPECT_THAT(output_str, HasSubstr(R"("args":{"name":"host0 /device:TPU:9")"));
  EXPECT_THAT(
      output_str,
      HasSubstr(R"({"args":{"sort_index":0},"name":"process_sort_index",)"
                R"("ph":"M","pid":0})"));
  EXPECT_THAT(
      output_str,
      HasSubstr(R"({"args":{"sort_index":1},"name":"process_sort_index",)"
                R"("ph":"M","pid":1})"));
  EXPECT_THAT(
      output_str,
      HasSubstr(
          R"({"args":{"sort_index":1073741833},"name":"process_sort_index",)"
          R"("ph":"M","pid":9})"));
}

TEST(TraceEventsToJsonTest, MpmdZeroRankedDevicesOmitsProcessSortIndex) {
  Trace trace;
  (*trace.mutable_devices())[0].mutable_resources()->operator[](1).set_name(
      "XLA Modules");
  (*trace.mutable_devices())[0].set_name("host0 /device:TPU:0");
  (*trace.mutable_devices())[1].mutable_resources()->operator[](1).set_name(
      "XLA Modules");
  (*trace.mutable_devices())[1].set_name("host0 /device:TPU:1");

  TraceEventsContainer events;
  // Non-MPMD event.
  TraceEvent event0;
  event0.set_device_id(0);
  event0.set_resource_id(1);
  event0.set_name("regular_kernel");
  event0.set_timestamp_ps(100);
  event0.set_duration_ps(50);
  events.AddEvent(event0);

  events.SetTrace(trace);

  JsonTraceOptions options;
  options.mpmd_pipeline_view = true;

  std::string output_str;
  IOBufferAdapter output(&output_str);
  TraceEventsToJson<IOBufferAdapter, TraceEventsContainer, RawData>(
      options, events, &output);

  // When K = 0 (no ranked MPMD devices), process_sort_index metadata is
  // omitted.
  EXPECT_THAT(output_str, Not(HasSubstr(R"("name":"process_sort_index")")));
}

TEST(SortMpmdDevicesTest, SortMpmdDevicesSingleDevicePerStage) {
  Trace trace;
  TraceEventsContainer events;

  auto add_event = [&](uint32_t device_id, const std::string& name,
                       uint64_t ts) {
    (*trace.mutable_devices())[device_id]
        .mutable_resources()
        ->operator[](1)
        .set_name("XLA Modules");
    TraceEvent event;
    event.set_device_id(device_id);
    event.set_resource_id(1);
    event.set_name(name);
    event.set_timestamp_ps(ts);
    event.set_duration_ps(1000);
    events.AddEvent(event);
  };

  // Device 0 and Device 1 both execute Stage 0 of program.
  add_event(0, "p0_stage0.program(1)", 1000);
  add_event(1, "p0_stage0.program(1)", 1000);

  // Device 2 executes Stage 1 of program.
  add_event(2, "p1_stage1.program(1)", 2000);

  events.SetTrace(trace);

  // With single_device_per_stage = false, all 3 devices get sort indices.
  absl::flat_hash_map<uint32_t, uint32_t> all_indices;
  SortMpmdDevices(events, all_indices, /*single_device_per_stage=*/false);
  EXPECT_EQ(all_indices.size(), 3);
  EXPECT_EQ(all_indices.at(0), 0);
  EXPECT_EQ(all_indices.at(1), 1);
  EXPECT_EQ(all_indices.at(2), 2);

  // With single_device_per_stage = true and deduplicated_device_ids = nullptr,
  // duplicate stage device 1 is omitted without recording deduplicated IDs.
  absl::flat_hash_map<uint32_t, uint32_t> dedup_indices_no_set;
  SortMpmdDevices(events, dedup_indices_no_set,
                  /*single_device_per_stage=*/true,
                  /*deduplicated_device_ids=*/nullptr);
  EXPECT_EQ(dedup_indices_no_set.size(), 2);
  EXPECT_TRUE(dedup_indices_no_set.contains(0));
  EXPECT_FALSE(dedup_indices_no_set.contains(1));
  EXPECT_TRUE(dedup_indices_no_set.contains(2));
  EXPECT_EQ(dedup_indices_no_set.at(0), 0);
  EXPECT_EQ(dedup_indices_no_set.at(2), 1);

  // With single_device_per_stage = true and non-null deduplicated_device_ids,
  // deduplicated device 1 is captured in the set.
  absl::flat_hash_map<uint32_t, uint32_t> dedup_indices;
  absl::flat_hash_set<uint32_t> deduplicated_device_ids;
  SortMpmdDevices(events, dedup_indices, /*single_device_per_stage=*/true,
                  &deduplicated_device_ids);
  EXPECT_EQ(dedup_indices.size(), 2);
  EXPECT_TRUE(dedup_indices.contains(0));
  EXPECT_FALSE(dedup_indices.contains(1));
  EXPECT_TRUE(dedup_indices.contains(2));
  EXPECT_EQ(dedup_indices.at(0), 0);
  EXPECT_EQ(dedup_indices.at(2), 1);
  EXPECT_THAT(deduplicated_device_ids, UnorderedElementsAre(1));
}

TEST(TraceEventsToJsonTest,
     TraceEventsToJsonSingleDevicePerStagePrunesDuplicateStageTpus) {
  Trace trace;
  // Device 0: TPU core 0 running stage 0 (representative).
  (*trace.mutable_devices())[0].mutable_resources()->operator[](1).set_name(
      "XLA Modules");
  (*trace.mutable_devices())[0].set_name("host0 /device:TPU:0");

  // Device 1: TPU core 1 running duplicate stage 0 (should be pruned).
  (*trace.mutable_devices())[1].mutable_resources()->operator[](1).set_name(
      "XLA Modules");
  (*trace.mutable_devices())[1].set_name("host0 /device:TPU:1");

  // Device 2: TPU core 2 running stage 1 (representative).
  (*trace.mutable_devices())[2].mutable_resources()->operator[](1).set_name(
      "XLA Modules");
  (*trace.mutable_devices())[2].set_name("host0 /device:TPU:2");

  // Device 10: Host CPU without MPMD events.
  (*trace.mutable_devices())[10].mutable_resources()->operator[](1).set_name(
      "Host Thread");
  (*trace.mutable_devices())[10].set_name("/host:CPU:0");

  TraceEventsContainer events;
  TraceEvent event0;
  event0.set_device_id(0);
  event0.set_resource_id(1);
  event0.set_name("p0_stage0.program(1)");
  event0.set_timestamp_ps(1000);
  event0.set_duration_ps(1000);
  events.AddEvent(event0);

  TraceEvent event1;
  event1.set_device_id(1);
  event1.set_resource_id(1);
  event1.set_name("p0_stage0.program(1)");
  event1.set_timestamp_ps(1000);
  event1.set_duration_ps(1000);
  events.AddEvent(event1);

  TraceEvent event2;
  event2.set_device_id(2);
  event2.set_resource_id(1);
  event2.set_name("p1_stage1.program(1)");
  event2.set_timestamp_ps(2000);
  event2.set_duration_ps(1000);
  events.AddEvent(event2);

  events.SetTrace(trace);

  JsonTraceOptions options;
  options.mpmd_pipeline_view = true;
  options.mpmd_single_device_per_stage = true;

  std::string output_str;
  IOBufferAdapter output(&output_str);
  TraceEventsToJson<IOBufferAdapter, TraceEventsContainer, RawData>(
      options, events, &output);

  // Device 0 and Device 2 are present with sort_index 0 and 1.
  EXPECT_THAT(
      output_str,
      HasSubstr(R"({"args":{"sort_index":0},"name":"process_sort_index",)"
                R"("ph":"M","pid":0})"));
  EXPECT_THAT(
      output_str,
      HasSubstr(R"({"args":{"sort_index":1},"name":"process_sort_index",)"
                R"("ph":"M","pid":2})"));

  // Duplicate Device 1 is completely pruned (no metadata and no events).
  EXPECT_THAT(output_str, Not(HasSubstr(R"("pid":1,)")));
  EXPECT_THAT(output_str, Not(HasSubstr(R"("pid":1})")));

  // Device 10 (Host CPU) is preserved behind MPMD stages:
  // kMpmdUnrankedSortIndexBase + 10 = 1073741834.
  EXPECT_THAT(
      output_str,
      HasSubstr(
          R"({"args":{"sort_index":1073741834},"name":"process_sort_index",)"
          R"("ph":"M","pid":10})"));
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
