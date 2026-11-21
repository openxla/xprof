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
#include "xprof/convert/trace_viewer/trace_events.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "net/proto2/contrib/parse_proto/parse_text_proto.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "google/protobuf/arena.h"
#include "xla/tsl/profiler/utils/timespan.h"
#include "xprof/convert/trace_viewer/trace_events_filter_interface.h"
#include "xprof/convert/trace_viewer/trace_events_test_utils.h"
#include "xprof/convert/trace_viewer/trace_viewer_visibility.h"
#include "plugin/xprof/protobuf/trace_events.pb.h"
#include "plugin/xprof/protobuf/trace_events_raw.pb.h"

namespace tensorflow::profiler {
namespace {

using ::google::protobuf::contrib::parse_proto::ParseTextProtoOrDie;
using ::testing::Eq;
using ::testing::EqualsProto;
using ::testing::proto::WhenDeserializedAs;
using ::tsl::profiler::Timespan;

TEST(SerializationTest, SerializeFullEventSkipsTimestampAndKeepsRawData) {
  TraceEvent event = ParseTextProtoOrDie(R"pb(
    device_id: 1
    resource_id: 2
    name: "test_event"
    timestamp_ps: 12345
    duration_ps: 100
    raw_data: "some_raw_data"
  )pb");

  std::string output;
  google::protobuf::Arena arena;
  TraceEvent* reusable_event_copy = google::protobuf::Arena::Create<TraceEvent>(&arena);
  EXPECT_OK(SerializeTraceEventForPersistingFullEvent(
      event, *reusable_event_copy, output));

  EXPECT_THAT(output, WhenDeserializedAs<TraceEvent>(EqualsProto(R"pb(
                device_id: 1
                resource_id: 2
                name: "test_event"
                duration_ps: 100
                raw_data: "some_raw_data"
              )pb")));
}

TEST(SerializationTest, SerializeEventWithoutMetadataSkipsTimestampAndRawData) {
  TraceEvent event = ParseTextProtoOrDie(R"pb(
    device_id: 1
    resource_id: 2
    name: "test_event"
    timestamp_ps: 12345
    raw_data: "some_raw_data"
  )pb");

  std::string output;
  google::protobuf::Arena arena;
  TraceEvent* reusable_event_copy = google::protobuf::Arena::Create<TraceEvent>(&arena);
  EXPECT_OK(SerializeTraceEventForPersistingEventWithoutMetadata(
      event, *reusable_event_copy, output));

  EXPECT_THAT(output, WhenDeserializedAs<TraceEvent>(EqualsProto(R"pb(
                device_id: 1
                resource_id: 2
                name: "test_event"
              )pb")));
}

TEST(SerializationTest, SerializeEventWithoutMetadataKeepsRawDataForCounter) {
  TraceEvent event = ParseTextProtoOrDie(R"pb(
    device_id: 1
    name: "counter_event"
    timestamp_ps: 12345
    raw_data: "some_raw_data"
  )pb");

  std::string output;
  google::protobuf::Arena arena;
  TraceEvent* reusable_event_copy = google::protobuf::Arena::Create<TraceEvent>(&arena);
  EXPECT_OK(SerializeTraceEventForPersistingEventWithoutMetadata(
      event, *reusable_event_copy, output));

  EXPECT_THAT(output, WhenDeserializedAs<TraceEvent>(EqualsProto(R"pb(
                device_id: 1
                name: "counter_event"
                raw_data: "some_raw_data"
              )pb")));
}

TEST(SerializationTest, SerializeOnlyMetadataReturnsMetadata) {
  TraceEvent event = ParseTextProtoOrDie(R"pb(
    resource_id: 1
    raw_data: "metadata_payload"
  )pb");

  std::string output;
  google::protobuf::Arena arena;
  TraceEvent* reusable_event = google::protobuf::Arena::Create<TraceEvent>(&arena);
  EXPECT_OK(SerializeTraceEventForPersistingOnlyMetadata(event, *reusable_event,
                                                         output));

  EXPECT_THAT(output, WhenDeserializedAs<TraceEvent>(EqualsProto(R"pb(
                raw_data: "metadata_payload"
              )pb")));
}

TEST(SerializationTest, SerializeOnlyMetadataReturnsNotFoundForCounter) {
  TraceEvent event = ParseTextProtoOrDie(R"pb(
    raw_data: "metadata_payload"
  )pb");

  std::string output;
  google::protobuf::Arena arena;
  TraceEvent* reusable_event = google::protobuf::Arena::Create<TraceEvent>(&arena);
  absl::Status status = SerializeTraceEventForPersistingOnlyMetadata(
      event, *reusable_event, output);
  EXPECT_TRUE(absl::IsNotFound(status));
}

TEST(SerializationTest, SerializeOnlyMetadataReturnsEmptyRawDataIfMissing) {
  TraceEvent event = ParseTextProtoOrDie(R"pb(
    resource_id: 1
  )pb");

  std::string output;
  google::protobuf::Arena arena;
  TraceEvent* reusable_event = google::protobuf::Arena::Create<TraceEvent>(&arena);
  EXPECT_OK(SerializeTraceEventForPersistingOnlyMetadata(event, *reusable_event,
                                                         output));

  EXPECT_THAT(output, WhenDeserializedAs<TraceEvent>(EqualsProto(R"pb(
                raw_data: ""
              )pb")));
}

TEST(SerializationTest, SerializeEventWithoutMetadataSkipsRawDataForAsync) {
  TraceEvent event = ParseTextProtoOrDie(R"pb(
    device_id: 1
    flow_id: 2
    name: "async_event"
    timestamp_ps: 12345
    raw_data: "some_raw_data"
  )pb");

  std::string output;
  google::protobuf::Arena arena;
  TraceEvent* reusable_event_copy = google::protobuf::Arena::Create<TraceEvent>(&arena);
  EXPECT_OK(SerializeTraceEventForPersistingEventWithoutMetadata(
      event, *reusable_event_copy, output));

  EXPECT_THAT(output, WhenDeserializedAs<TraceEvent>(EqualsProto(R"pb(
                device_id: 1
                flow_id: 2
                name: "async_event"
              )pb")));
}

TEST(SerializationTest, SerializeFullEventHandlesSupportedTypes) {
  TraceEvent event = ParseTextProtoOrDie(R"pb(
    device_id: 42
    resource_id: 1234567890123
    timestamp_ps: 9876543210987
    name: "multi_type_event"
    group_id: -100
    flow_entry_type: FLOW_START
  )pb");

  std::string output;
  google::protobuf::Arena arena;
  TraceEvent* reusable_event_copy = google::protobuf::Arena::Create<TraceEvent>(&arena);
  EXPECT_OK(SerializeTraceEventForPersistingFullEvent(
      event, *reusable_event_copy, output));

  EXPECT_THAT(output, WhenDeserializedAs<TraceEvent>(EqualsProto(R"pb(
                device_id: 42
                resource_id: 1234567890123
                name: "multi_type_event"
                group_id: -100
                flow_entry_type: FLOW_START
              )pb")));
}

TEST(TraceEventsSearchTest, SearchMetadataKeysInTrie) {
  absl::SetFlag(&FLAGS_xprof_enable_metadata_indexing, true);
  // 1. Create and populate a TraceEventsContainerBase
  using TestContainer = TraceEventsContainerBase<EventFactory, RawData>;
  TestContainer input_container;

  // Set up metadata (devices and resources) to make it valid
  Device* device = input_container.MutableDevice(1);
  device->set_device_id(1);
  device->set_name("TestDevice");
  Resource* resource = &(*device->mutable_resources())[2];
  resource->set_resource_id(2);
  resource->set_name("TestResource");

  // Add Event 1: Name "ModelForward" with searchable metadata request_id1:
  // "target_req_123"
  RawData raw_data1;
  TraceEventArguments::Argument* arg1 = raw_data1.mutable_args()->add_arg();
  arg1->set_name("request_id1");
  arg1->set_str_value("target_req_123");

  // And some non-searchable metadata in the same event
  TraceEventArguments::Argument* arg2 = raw_data1.mutable_args()->add_arg();
  arg2->set_name("bytes");
  arg2->set_uint_value(1024);

  input_container.AddCompleteEvent(
      "ModelForward", /*resource_id=*/2, /*device_id=*/1,
      Timespan::FromEndPoints(100, 200), &raw_data1);

  // Add Event 2: Name "OtherEvent" with non-searchable metadata some_other_key:
  // "req_ignored"
  RawData raw_data2;
  TraceEventArguments::Argument* arg3 = raw_data2.mutable_args()->add_arg();
  arg3->set_name("some_other_key");
  arg3->set_str_value("req_ignored");

  input_container.AddCompleteEvent(
      "OtherEvent", /*resource_id=*/2, /*device_id=*/1,
      Timespan::FromEndPoints(300, 400), &raw_data2);

  // 2. Store the container as LevelDB tables
  ASSERT_OK_AND_ASSIGN(
      TraceEventsLevelDbFilePaths file_paths,
      StoreTraceContainerToLevelDbTables(
          ::testing::TempDir(), "metadata_search_keys", input_container));

  // 3. Search using the trie and verify results

  // Scenario A: Search for "target_req" with search_metadata = false (should
  // match event but NOT load metadata arguments)
  {
    TestContainer search_container;
    ASSERT_OK(search_container.SearchInLevelDbTable(file_paths, "target_req"));
    EXPECT_THAT(search_container.NumEvents(), Eq(1));

    search_container.ForAllEvents([](const TraceEvent& event) {
      EXPECT_THAT(event.name(), Eq("ModelForward"));
      RawData raw_data;
      ASSERT_TRUE(raw_data.ParseFromString(event.raw_data()));
      // Slices should only have the dummy 'uid' argument, not the original
      // 'request_id1'
      for (const TraceEventArguments::Argument& arg : raw_data.args().arg()) {
        EXPECT_NE(arg.name(), "request_id1");
      }
    });
  }

  // Scenario A2: Search for "target_req" with search_metadata = true (should
  // match event AND fully load original arguments)
  {
    TestContainer search_container;
    ASSERT_OK(search_container.SearchInLevelDbTable(
        file_paths, "target_req", nullptr, {.search_metadata = true}));
    EXPECT_THAT(search_container.NumEvents(), Eq(1));

    search_container.ForAllEvents([](const TraceEvent& event) {
      EXPECT_THAT(event.name(), Eq("ModelForward"));
      RawData raw_data;
      ASSERT_TRUE(raw_data.ParseFromString(event.raw_data()));
      ASSERT_TRUE(raw_data.has_args());
      bool found_req_id = false;
      for (const TraceEventArguments::Argument& arg : raw_data.args().arg()) {
        if (arg.name() == "request_id1") {
          EXPECT_THAT(arg.str_value(), Eq("target_req_123"));
          found_req_id = true;
        }
      }
      EXPECT_TRUE(found_req_id);
    });
  }

  // Scenario B: Search for "req_ignored" (should NOT match Event 2 because key
  // name doesn't match regex)
  {
    TestContainer search_container;
    ASSERT_OK(search_container.SearchInLevelDbTable(file_paths, "req_ignored"));
    EXPECT_THAT(search_container.NumEvents(), Eq(0));
  }

  // Scenario C: Search for "Model" (should match Event 1 via event name -
  // existing behavior)
  {
    TestContainer search_container;
    ASSERT_OK(search_container.SearchInLevelDbTable(file_paths, "Model"));
    EXPECT_THAT(search_container.NumEvents(), Eq(1));
    search_container.ForAllEvents([](const TraceEvent& event) {
      EXPECT_THAT(event.name(), Eq("ModelForward"));
    });
  }

  // Scenario D: Deduplication verification. Search for "req" which matches both
  // "req_event" name and metadata.
  {
    TestContainer deduplication_container;
    Device* dev = deduplication_container.MutableDevice(1);
    dev->set_device_id(1);
    Resource* res = &(*dev->mutable_resources())[2];
    res->set_resource_id(2);

    RawData raw_data3;
    TraceEventArguments::Argument* arg = raw_data3.mutable_args()->add_arg();
    arg->set_name("request_id1");
    arg->set_str_value("req_val");

    deduplication_container.AddCompleteEvent(
        "req_event", /*resource_id=*/2, /*device_id=*/1,
        Timespan::FromEndPoints(500, 600), &raw_data3);

    ASSERT_OK_AND_ASSIGN(TraceEventsLevelDbFilePaths paths,
                         StoreTraceContainerToLevelDbTables(
                             ::testing::TempDir(), "metadata_search_dedup",
                             deduplication_container));

    TestContainer search_container;
    ASSERT_OK(search_container.SearchInLevelDbTable(paths, "req"));
    // Deduplication ensures we only get 1 unique event.
    EXPECT_THAT(search_container.NumEvents(), Eq(1));
    search_container.ForAllEvents([](const TraceEvent& event) {
      EXPECT_THAT(event.name(), Eq("req_event"));
    });
  }
}

TEST(TraceEventsSearchTest, SearchMetadataInternedAndEdgeCases) {
  absl::SetFlag(&FLAGS_xprof_enable_metadata_indexing, true);
  using TestContainer = TraceEventsContainerBase<EventFactory, RawData>;
  TestContainer input_container;

  Device* device = input_container.MutableDevice(1);
  device->set_device_id(1);
  device->set_name("TestDevice");
  Resource* resource = &(*device->mutable_resources())[2];
  resource->set_resource_id(2);
  resource->set_name("TestResource");

  // 1. Short event name (not interned, so it works without event name
  // resolution)
  std::string event_name = "ShortEventName";

  // 2. Interned metadata value (Long string > 16 chars gets interned
  // automatically)
  std::string long_request_id = "SuperLongRequestIdValueThatWillBeInterned";

  RawData raw_data1;
  TraceEventArguments::Argument* arg1 = raw_data1.mutable_args()->add_arg();
  arg1->set_name("request_id1");
  arg1->set_str_value(long_request_id);

  input_container.AddCompleteEvent(
      event_name, /*resource_id=*/2, /*device_id=*/1,
      Timespan::FromEndPoints(100, 200), &raw_data1);

  // 3. Non-string metadata value (should be ignored by search indexer)
  RawData raw_data2;
  TraceEventArguments::Argument* arg_int = raw_data2.mutable_args()->add_arg();
  arg_int->set_name("request_id1");
  arg_int->set_uint_value(99999);  // Integer instead of string

  input_container.AddCompleteEvent(
      "IntReqEvent", /*resource_id=*/2, /*device_id=*/1,
      Timespan::FromEndPoints(300, 400), &raw_data2);

  // 4. Event with no raw_data (should be indexed by name but have no metadata
  // index)
  input_container.AddCompleteEvent("NoRawDataEvent", /*resource_id=*/2,
                                   /*device_id=*/1,
                                   Timespan::FromEndPoints(500, 600), nullptr);

  ASSERT_OK_AND_ASSIGN(
      TraceEventsLevelDbFilePaths file_paths,
      StoreTraceContainerToLevelDbTables(
          ::testing::TempDir(), "metadata_search_edge", input_container));

  // Test Scenario A: Search for the event name (now short/not interned)
  {
    TestContainer search_container;
    ASSERT_OK(search_container.SearchInLevelDbTable(file_paths, event_name));
    EXPECT_THAT(search_container.NumEvents(), Eq(1));
    search_container.ForAllEvents([&](const TraceEvent& event) {
      EXPECT_THAT(event.name(), Eq(event_name));
    });
  }

  // Test Scenario B: Search for the long interned metadata request_id value
  {
    TestContainer search_container;
    ASSERT_OK(search_container.SearchInLevelDbTable(file_paths,
                                                    "SuperLongRequestId"));
    EXPECT_THAT(search_container.NumEvents(), Eq(1));
    search_container.ForAllEvents([&](const TraceEvent& event) {
      EXPECT_THAT(event.name(), Eq(event_name));
    });
  }

  // Test Scenario C: Search for the integer metadata value (should be converted
  // to string and found)
  {
    TestContainer search_container;
    ASSERT_OK(search_container.SearchInLevelDbTable(file_paths, "99999"));
    EXPECT_THAT(search_container.NumEvents(), Eq(1));
    search_container.ForAllEvents([&](const TraceEvent& event) {
      EXPECT_THAT(event.name(), Eq("IntReqEvent"));
    });
  }

  // Test Scenario D: Search for the event with no raw data (should be found by
  // name)
  {
    TestContainer search_container;
    ASSERT_OK(
        search_container.SearchInLevelDbTable(file_paths, "NoRawDataEvent"));
    EXPECT_THAT(search_container.NumEvents(), Eq(1));
  }
}

TEST(TraceEventsZoomLevelTest, FlowEventsZoomLevelAssignmentAndSafetyGuard) {
  // Test basic flow event zoom level assignment.
  using TestContainer = TraceEventsContainerBase<EventFactory, RawData>;
  TestContainer container;

  Device* device = container.MutableDevice(1);
  device->set_device_id(1);
  device->set_name("TestDevice");
  Resource* resource = &(*device->mutable_resources())[2];
  resource->set_resource_id(2);
  resource->set_name("TestResource");

  // Add a dummy flow event to occupy the first flow event slot on the track.
  container.AddFlowEvent("Dummy", /*resource_id=*/2, /*device_id=*/1,
                         Timespan::FromEndPoints(0, 100), /*flow_id=*/99,
                         TraceEvent::FLOW_START);

  // Flow 1 consists of:
  // - Event A: start 1000ps, duration 100ps (tiny), flow 1
  // - Event B: start 2000ps, duration 2,000,000,000,000ps (2s, very long), flow
  // 1
  container.AddFlowEvent("EventA", /*resource_id=*/2, /*device_id=*/1,
                         Timespan::FromEndPoints(1000, 1100), /*flow_id=*/1,
                         TraceEvent::FLOW_START);
  container.AddFlowEvent("EventB", /*resource_id=*/2, /*device_id=*/1,
                         Timespan::FromEndPoints(2000, 2000 + 2000000000000ull),
                         /*flow_id=*/1, TraceEvent::FLOW_END);

  std::vector<std::vector<const TraceEvent*>> events_by_level =
      container.GetTraceEventsByLevel();

  // Verify Event B is at level 0.
  bool found_b_at_level_0 = false;
  for (const TraceEvent* event : events_by_level[0]) {
    if (event->name() == "EventB") {
      found_b_at_level_0 = true;
    }
  }
  EXPECT_TRUE(found_b_at_level_0);

  // Verify Event A is at level 9 (resolution 1ns).
  bool found_a_at_level_9 = false;
  for (const TraceEvent* event : events_by_level[9]) {
    if (event->name() == "EventA") {
      found_a_at_level_9 = true;
    }
  }
  EXPECT_TRUE(found_a_at_level_9);
}

TEST(TraceEventsZoomLevelTest, SafetyGuardForFallbackFlowEvents) {
  // Verify that the track duration resolution check serves as a safety guard
  // for flow events where flow-level visibility is incorrectly masked by an
  // earlier short event.
  using TestContainer = TraceEventsContainerBase<EventFactory, RawData>;
  TestContainer container;

  Device* device = container.MutableDevice(1);
  device->set_device_id(1);
  device->set_name("TestDevice");

  Resource* resource1 = &(*device->mutable_resources())[2];
  resource1->set_resource_id(2);
  resource1->set_name("TestResource1");

  Resource* resource2 = &(*device->mutable_resources())[3];
  resource2->set_resource_id(3);
  resource2->set_name("TestResource2");

  // Track 1 (Device 1, Resource 2):
  // - Dummy 1: start 0, duration 100ps.
  // - Event A: start 1000ps, duration 100ps, flow 1 (FLOW_START)
  // - Event B: start 2000ps, duration 2s (2e12ps), flow 1 (FLOW_MID)
  container.AddFlowEvent("Dummy1", /*resource_id=*/2, /*device_id=*/1,
                         Timespan::FromEndPoints(0, 100), /*flow_id=*/99,
                         TraceEvent::FLOW_START);
  container.AddFlowEvent("EventA", /*resource_id=*/2, /*device_id=*/1,
                         Timespan::FromEndPoints(1000, 1100), /*flow_id=*/1,
                         TraceEvent::FLOW_START);
  container.AddFlowEvent("EventB", /*resource_id=*/2, /*device_id=*/1,
                         Timespan::FromEndPoints(2000, 2000 + 2000000000000ull),
                         /*flow_id=*/1, TraceEvent::FLOW_MID);

  // Track 2 (Device 1, Resource 3):
  // - Dummy 2: start 3.5s (3.5e12ps), duration 100ps.
  // - Event C: start 4s (4e12ps), duration 100ps, flow 1 (FLOW_END)
  container.AddFlowEvent(
      "Dummy2", /*resource_id=*/3, /*device_id=*/1,
      Timespan::FromEndPoints(3500000000000ull, 3500000000000ull + 100),
      /*flow_id=*/98, TraceEvent::FLOW_START);
  container.AddFlowEvent(
      "EventC", /*resource_id=*/3, /*device_id=*/1,
      Timespan::FromEndPoints(4000000000000ull, 4000000000000ull + 100),
      /*flow_id=*/1, TraceEvent::FLOW_END);

  std::vector<std::vector<const TraceEvent*>> events_by_level =
      container.GetTraceEventsByLevel();

  // Verify Event B is at level 0 (due to fallback duration promotion).
  bool found_b_at_level_0 = false;
  for (const TraceEvent* event : events_by_level[0]) {
    if (event->name() == "EventB") {
      found_b_at_level_0 = true;
    }
  }
  EXPECT_TRUE(found_b_at_level_0);

  // Verify Event C (Track 2) is NOT promoted to level 0 (or level 1, etc.) by
  // Event B's fallback promotion on Track 1. It should remain at level 9.
  for (int level = 0; level < 9; ++level) {
    for (const TraceEvent* event : events_by_level[level]) {
      EXPECT_NE(event->name(), "EventC");
    }
  }

  bool found_c_at_level_9 = false;
  for (const TraceEvent* event : events_by_level[9]) {
    if (event->name() == "EventC") {
      found_c_at_level_9 = true;
    }
  }
  EXPECT_TRUE(found_c_at_level_9);
}

class CountingTraceEventsFilter : public TraceEventsFilterInterface {
 public:
  explicit CountingTraceEventsFilter(int* filter_calls)
      : filter_calls_(filter_calls) {}

  void SetUp(const Trace& trace) override {}

  bool Filter(const TraceEvent& event) override {
    ++(*filter_calls_);
    return false;
  }

 private:
  int* filter_calls_;
};

using TestTraceEventsContainer =
    TraceEventsContainerBase<EventFactory, RawData>;

TestTraceEventsContainer MakeTestTraceContainerWithDeviceAndResource() {
  TestTraceEventsContainer container;
  Device* device = container.MutableDevice(1);
  device->set_device_id(1);
  device->set_name("TestDevice");
  Resource* resource = &(*device->mutable_resources())[2];
  resource->set_resource_id(2);
  resource->set_name("TestResource");
  return container;
}

// Builds a trace with an event at timestamp 0 so scanning LevelDB with a
// zero-duration span would invoke CountingTraceEventsFilter without an
// empty-window short-circuit.
TestTraceEventsContainer MakeZeroOriginTraceContainer() {
  TestTraceEventsContainer container =
      MakeTestTraceContainerWithDeviceAndResource();
  container.AddCompleteEvent("Event0", /*resource_id=*/2, /*device_id=*/1,
                             Timespan::FromEndPoints(0, 100));
  container.AddCompleteEvent("Event1", /*resource_id=*/2, /*device_id=*/1,
                             Timespan::FromEndPoints(1000, 1100));
  container.AddCompleteEvent("Event2", /*resource_id=*/2, /*device_id=*/1,
                             Timespan::FromEndPoints(1500, 2000));
  return container;
}

TEST(TraceEventsLoadTest, OverlappingWindowScansLevelDbAndLoadsEvents) {
  absl::StatusOr<TraceEventsLevelDbFilePaths> file_paths =
      StoreTraceContainerToLevelDbTables(::testing::TempDir(),
                                         "overlapping_window_zero_origin",
                                         MakeZeroOriginTraceContainer());
  ASSERT_OK(file_paths);

  TestTraceEventsContainer loaded;
  int filter_calls = 0;
  auto visibility = std::make_unique<TraceVisibilityFilter>(
      TraceViewerVisibility::Window::FromEndPoints(0, 2100),
      /*resolution=*/0.0);
  EXPECT_OK(loaded.LoadFromLevelDbTable(
      *file_paths, std::make_unique<CountingTraceEventsFilter>(&filter_calls),
      std::move(visibility)));
  EXPECT_GT(filter_calls, 0);
  EXPECT_EQ(loaded.NumEvents(), 3);
}

struct EmptyWindowLoadCase {
  std::string test_name;
  TraceViewerVisibility::Window window;
};

class TraceEventsEmptyWindowLoadTest
    : public ::testing::TestWithParam<EmptyWindowLoadCase> {};

TEST_P(TraceEventsEmptyWindowLoadTest, ShortCircuitsScanAndPreservesMetadata) {
  const EmptyWindowLoadCase& test_case = GetParam();
  absl::StatusOr<TraceEventsLevelDbFilePaths> file_paths =
      StoreTraceContainerToLevelDbTables(
          ::testing::TempDir(),
          absl::StrCat("empty_window_", test_case.test_name),
          MakeZeroOriginTraceContainer());
  ASSERT_OK(file_paths);

  TestTraceEventsContainer loaded;
  int filter_calls = 0;
  auto visibility = std::make_unique<TraceVisibilityFilter>(test_case.window,
                                                            /*resolution=*/0.0);
  EXPECT_OK(loaded.LoadFromLevelDbTable(
      *file_paths, std::make_unique<CountingTraceEventsFilter>(&filter_calls),
      std::move(visibility)));
  EXPECT_EQ(filter_calls, 0);
  EXPECT_EQ(loaded.NumEvents(), 0);
  EXPECT_EQ(loaded.trace().min_timestamp_ps(), 0);
  EXPECT_EQ(loaded.trace().max_timestamp_ps(), 2000);
  ASSERT_TRUE(loaded.trace().devices().contains(1));
  EXPECT_EQ(loaded.trace().devices().at(1).name(), "TestDevice");
  ASSERT_TRUE(loaded.trace().devices().at(1).resources().contains(2));
  EXPECT_EQ(loaded.trace().devices().at(1).resources().at(2).name(),
            "TestResource");
}

INSTANTIATE_TEST_SUITE_P(
    EmptyWindows, TraceEventsEmptyWindowLoadTest,
    ::testing::Values(
        EmptyWindowLoadCase{"ExplicitEmpty",
                            TraceViewerVisibility::Window::Empty()},
        EmptyWindowLoadCase{
            "InvertedBounds",
            TraceViewerVisibility::Window::FromEndPoints(1500, 500)},
        EmptyWindowLoadCase{
            "StartPastTraceMax",
            TraceViewerVisibility::Window::FromEndPoints(3000, std::nullopt)}),
    [](const ::testing::TestParamInfo<EmptyWindowLoadCase>& info) {
      return info.param.test_name;
    });

TEST(TraceEventsLoadTest,
     WindowBeforeTraceMinShortCircuitsScanAndPreservesMetadata) {
  TestTraceEventsContainer nonzero_min_container =
      MakeTestTraceContainerWithDeviceAndResource();
  nonzero_min_container.AddCompleteEvent("Event1", /*resource_id=*/2,
                                         /*device_id=*/1,
                                         Timespan::FromEndPoints(1000, 1100));
  nonzero_min_container.AddCompleteEvent("Event2", /*resource_id=*/2,
                                         /*device_id=*/1,
                                         Timespan::FromEndPoints(1500, 2000));
  absl::StatusOr<TraceEventsLevelDbFilePaths> file_paths =
      StoreTraceContainerToLevelDbTables(::testing::TempDir(),
                                         "empty_window_nonzero_min",
                                         nonzero_min_container);
  ASSERT_OK(file_paths);

  TestTraceEventsContainer loaded;
  int filter_calls = 0;
  auto visibility = std::make_unique<TraceVisibilityFilter>(
      TraceViewerVisibility::Window::FromEndPoints(100, 500),
      /*resolution=*/0.0);
  EXPECT_OK(loaded.LoadFromLevelDbTable(
      *file_paths, std::make_unique<CountingTraceEventsFilter>(&filter_calls),
      std::move(visibility)));
  EXPECT_EQ(filter_calls, 0);
  EXPECT_EQ(loaded.NumEvents(), 0);
  EXPECT_EQ(loaded.trace().min_timestamp_ps(), 1000);
  EXPECT_EQ(loaded.trace().max_timestamp_ps(), 2000);
  ASSERT_TRUE(loaded.trace().devices().contains(1));
  EXPECT_EQ(loaded.trace().devices().at(1).name(), "TestDevice");
  ASSERT_TRUE(loaded.trace().devices().at(1).resources().contains(2));
  EXPECT_EQ(loaded.trace().devices().at(1).resources().at(2).name(),
            "TestResource");
}

TEST(TraceEventsLoadTest,
     UnboundedWindowLoadsHighTimestampEventsWhenTraceOmitsMaxTimestamp) {
  // ExpandTraceSpan ignores timestamps above UINT64_MAX / 2 (~9.22e18 ps), so
  // storing events at 1e19 ps naturally produces a trace omitting
  // max_timestamp_ps. Assumes all events in the trace exceed UINT64_MAX / 2 so
  // `has_max_timestamp_ps()` remains false; mixed normal (`<= UINT64_MAX / 2`)
  // and corrupt (`> UINT64_MAX / 2`) traces are intentionally not required to
  // match between unclamped and clamped filtering paths.
  constexpr uint64_t kHighBasePs = 10000000000000000000ULL;
  TestTraceEventsContainer input_container =
      MakeTestTraceContainerWithDeviceAndResource();
  input_container.AddCompleteEvent("EarlyHighEvent", /*resource_id=*/2,
                                   /*device_id=*/1, Timespan(kHighBasePs, 50));
  input_container.AddCompleteEvent("MidHighEvent", /*resource_id=*/2,
                                   /*device_id=*/1,
                                   Timespan(kHighBasePs + 500, 50));
  input_container.AddCompleteEvent("LateHighEvent", /*resource_id=*/2,
                                   /*device_id=*/1,
                                   Timespan(kHighBasePs + 1500, 50));

  absl::StatusOr<TraceEventsLevelDbFilePaths> file_paths =
      StoreTraceContainerToLevelDbTables(::testing::TempDir(), "high_ts",
                                         input_container);
  ASSERT_OK(file_paths);

  // Unconstrained filter loads all 3 events without breaking at end_ps == 0.
  {
    TestTraceEventsContainer loaded;
    auto visibility = std::make_unique<TraceVisibilityFilter>(
        TraceViewerVisibility::Window::FromEndPoints(0, std::nullopt),
        /*resolution=*/0.0);
    EXPECT_OK(loaded.LoadFromLevelDbTable(*file_paths, /*filter=*/nullptr,
                                          std::move(visibility)));
    EXPECT_FALSE(loaded.trace().has_max_timestamp_ps());
    EXPECT_EQ(loaded.NumEvents(), 3);
  }

  // Open-ended filter with start_ps > 0 drops EarlyHighEvent and loads both
  // later events instead of breaking at event.timestamp_ps() > start_ps.
  {
    TestTraceEventsContainer loaded;
    auto visibility = std::make_unique<TraceVisibilityFilter>(
        TraceViewerVisibility::Window::FromEndPoints(kHighBasePs + 400,
                                                     std::nullopt),
        /*resolution=*/0.0);
    EXPECT_OK(loaded.LoadFromLevelDbTable(*file_paths, /*filter=*/nullptr,
                                          std::move(visibility)));
    EXPECT_FALSE(loaded.trace().has_max_timestamp_ps());
    EXPECT_EQ(loaded.NumEvents(), 2);
  }

  // Null visibility filter also loads all 3 events when max_timestamp_ps is
  // unset.
  {
    TestTraceEventsContainer loaded;
    EXPECT_OK(loaded.LoadFromLevelDbTable(*file_paths, /*filter=*/nullptr,
                                          /*visibility=*/nullptr));
    EXPECT_FALSE(loaded.trace().has_max_timestamp_ps());
    EXPECT_EQ(loaded.NumEvents(), 3);
  }
}

}  // namespace
}  // namespace tensorflow::profiler
