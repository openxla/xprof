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
#include <string>
#include <utility>
#include <vector>

#include "net/proto2/contrib/parse_proto/parse_text_proto.h"
#include "testing/base/public/gmock.h"
#include "<gtest/gtest.h>"
#include "absl/cleanup/cleanup.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "google/protobuf/arena.h"
#include "xla/tsl/lib/io/table_builder.h"
#include "xla/tsl/lib/io/table_options.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/file_system.h"
#include "xla/tsl/profiler/utils/timespan.h"
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

std::string GetTempFilename(absl::string_view suffix) {
  return absl::StrCat(::testing::TempDir(), "/metadata_search_test_", suffix);
}

TEST(TraceEventsSearchTest, SearchMetadataKeysInTrie) {
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
  std::string trace_events_file = GetTempFilename("events.ldb");
  std::string trace_events_metadata_file = GetTempFilename("metadata.ldb");
  std::string trace_events_prefix_trie_file = GetTempFilename("trie.ldb");

  absl::Cleanup cleanup = [&] {
    tsl::Env::Default()->DeleteFile(trace_events_file).IgnoreError();
    tsl::Env::Default()->DeleteFile(trace_events_metadata_file).IgnoreError();
    tsl::Env::Default()
        ->DeleteFile(trace_events_prefix_trie_file)
        .IgnoreError();
  };

  std::unique_ptr<tsl::WritableFile> file1, file2, file3;
  ASSERT_OK(tsl::Env::Default()->NewWritableFile(trace_events_file, &file1));
  ASSERT_OK(
      tsl::Env::Default()->NewWritableFile(trace_events_metadata_file, &file2));
  ASSERT_OK(tsl::Env::Default()->NewWritableFile(trace_events_prefix_trie_file,
                                                 &file3));

  ASSERT_OK(input_container.StoreAsLevelDbTables(
      std::move(file1), std::move(file2), std::move(file3)));

  TraceEventsLevelDbFilePaths file_paths = {
      .trace_events_file_path = trace_events_file,
      .trace_events_metadata_file_path = trace_events_metadata_file,
      .trace_events_prefix_trie_file_path = trace_events_prefix_trie_file,
  };

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

    std::string f1 = GetTempFilename("events_dedup.ldb");
    std::string f2 = GetTempFilename("metadata_dedup.ldb");
    std::string f3 = GetTempFilename("trie_dedup.ldb");

    absl::Cleanup cleanup_dedup = [&] {
      tsl::Env::Default()->DeleteFile(f1).IgnoreError();
      tsl::Env::Default()->DeleteFile(f2).IgnoreError();
      tsl::Env::Default()->DeleteFile(f3).IgnoreError();
    };

    std::unique_ptr<tsl::WritableFile> wf1, wf2, wf3;
    ASSERT_OK(tsl::Env::Default()->NewWritableFile(f1, &wf1));
    ASSERT_OK(tsl::Env::Default()->NewWritableFile(f2, &wf2));
    ASSERT_OK(tsl::Env::Default()->NewWritableFile(f3, &wf3));

    ASSERT_OK(deduplication_container.StoreAsLevelDbTables(
        std::move(wf1), std::move(wf2), std::move(wf3)));

    TraceEventsLevelDbFilePaths paths = {
        .trace_events_file_path = f1,
        .trace_events_metadata_file_path = f2,
        .trace_events_prefix_trie_file_path = f3,
    };

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

  std::string trace_events_file = GetTempFilename("events_edge.ldb");
  std::string trace_events_metadata_file = GetTempFilename("metadata_edge.ldb");
  std::string trace_events_prefix_trie_file = GetTempFilename("trie_edge.ldb");

  absl::Cleanup cleanup = [&] {
    tsl::Env::Default()->DeleteFile(trace_events_file).IgnoreError();
    tsl::Env::Default()->DeleteFile(trace_events_metadata_file).IgnoreError();
    tsl::Env::Default()
        ->DeleteFile(trace_events_prefix_trie_file)
        .IgnoreError();
  };

  std::unique_ptr<tsl::WritableFile> file1, file2, file3;
  ASSERT_OK(tsl::Env::Default()->NewWritableFile(trace_events_file, &file1));
  ASSERT_OK(
      tsl::Env::Default()->NewWritableFile(trace_events_metadata_file, &file2));
  ASSERT_OK(tsl::Env::Default()->NewWritableFile(trace_events_prefix_trie_file,
                                                 &file3));

  ASSERT_OK(input_container.StoreAsLevelDbTables(
      std::move(file1), std::move(file2), std::move(file3)));

  TraceEventsLevelDbFilePaths file_paths = {
      .trace_events_file_path = trace_events_file,
      .trace_events_metadata_file_path = trace_events_metadata_file,
      .trace_events_prefix_trie_file_path = trace_events_prefix_trie_file,
  };

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

TEST(TraceEventsKeyLengthTest, KeyGenerationAndParsing) {
  // Test legacy key behavior
  {
    // Valid repetition
    std::string key10 =
        LevelDbTableKey(/*zoom_level=*/1, /*timestamp=*/123456789ULL,
                        /*repetition=*/255, /*key_length=*/kLegacyKeyLength);
    EXPECT_EQ(key10.size(), kLegacyKeyLength);
    EXPECT_EQ(TimestampFromLevelDbTableKey(key10), 123456789ULL);

    // Out of bounds repetition for legacy key
    std::string key10_oob =
        LevelDbTableKey(/*zoom_level=*/1, /*timestamp=*/123456789ULL,
                        /*repetition=*/256, /*key_length=*/kLegacyKeyLength);
    EXPECT_TRUE(key10_oob.empty());
  }

  // Test extended key behavior
  {
    // Valid repetition under 256
    std::string key11_small =
        LevelDbTableKey(/*zoom_level=*/1, /*timestamp=*/123456789ULL,
                        /*repetition=*/255, /*key_length=*/kExtendedKeyLength);
    EXPECT_EQ(key11_small.size(), kExtendedKeyLength);
    EXPECT_EQ(TimestampFromLevelDbTableKey(key11_small), 123456789ULL);

    // Valid repetition above 256
    std::string key11_large =
        LevelDbTableKey(/*zoom_level=*/1, /*timestamp=*/123456789ULL,
                        /*repetition=*/1000, /*key_length=*/kExtendedKeyLength);
    EXPECT_EQ(key11_large.size(), kExtendedKeyLength);
    EXPECT_EQ(TimestampFromLevelDbTableKey(key11_large), 123456789ULL);

    // Out of bounds repetition for extended key
    std::string key11_oob = LevelDbTableKey(
        /*zoom_level=*/1, /*timestamp=*/123456789ULL,
        /*repetition=*/65536, /*key_length=*/kExtendedKeyLength);
    EXPECT_TRUE(key11_oob.empty());
  }
}

#ifndef NDEBUG
TEST(TraceEventsDeathTest, TimestampFromLevelDbTableKeyUnexpectedKeySize) {
  EXPECT_DEBUG_DEATH(TimestampFromLevelDbTableKey("short"),
                     "Unexpected key size: 5");
}
#endif

TEST(TraceEventsLevelDbTest, LoadFromExtendedLevelDbTable) {
  using TestContainer = TraceEventsContainerBase<EventFactory, RawData>;
  TestContainer input_container;

  // Set up metadata (devices and resources)
  Device* device = input_container.MutableDevice(1);
  device->set_device_id(1);
  device->set_name("TestDevice");
  Resource* resource = &(*device->mutable_resources())[2];
  resource->set_resource_id(2);
  resource->set_name("TestResource");

  // Add event with raw data
  RawData raw_data;
  TraceEventArguments::Argument* arg = raw_data.mutable_args()->add_arg();
  arg->set_name("my_arg");
  arg->set_str_value("my_val");

  input_container.AddCompleteEvent(
      "ExtendedEvent", /*resource_id=*/2, /*device_id=*/1,
      Timespan::FromEndPoints(100, 200), &raw_data);

  // Get temporary paths
  std::string trace_events_file = GetTempFilename("ext_events.ldb");
  std::string trace_events_metadata_file = GetTempFilename("ext_metadata.ldb");
  std::string trace_events_prefix_trie_file = GetTempFilename("ext_trie.ldb");

  absl::Cleanup cleanup = [&] {
    tsl::Env::Default()->DeleteFile(trace_events_file).IgnoreError();
    tsl::Env::Default()->DeleteFile(trace_events_metadata_file).IgnoreError();
    tsl::Env::Default()
        ->DeleteFile(trace_events_prefix_trie_file)
        .IgnoreError();
  };

  std::unique_ptr<tsl::WritableFile> file1_ptr, file2_ptr, file3_ptr;
  ASSERT_OK(
      tsl::Env::Default()->NewWritableFile(trace_events_file, &file1_ptr));
  ASSERT_OK(tsl::Env::Default()->NewWritableFile(trace_events_metadata_file,
                                                 &file2_ptr));
  ASSERT_OK(tsl::Env::Default()->NewWritableFile(trace_events_prefix_trie_file,
                                                 &file3_ptr));

  // Store as LevelDB Tables (uses 11-byte extended keys by default)
  ASSERT_OK(input_container.StoreAsLevelDbTables(
      std::move(file1_ptr), std::move(file2_ptr), std::move(file3_ptr)));

  TraceEventsLevelDbFilePaths file_paths = {
      .trace_events_file_path = trace_events_file,
      .trace_events_metadata_file_path = trace_events_metadata_file,
      .trace_events_prefix_trie_file_path = trace_events_prefix_trie_file,
  };

  // Verify LoadFromLevelDbTable
  TestContainer load_container;
  ASSERT_OK(load_container.LoadFromLevelDbTable(
      file_paths, /*filter=*/nullptr, /*visibility=*/nullptr,
      /*filter_by_visibility_threshold=*/-1LL, /*load_metadata=*/true));

  EXPECT_THAT(load_container.NumEvents(), Eq(1));
  load_container.ForAllEvents([](const TraceEvent& event) {
    EXPECT_THAT(event.name(), Eq("ExtendedEvent"));
    EXPECT_THAT(event.timestamp_ps(), Eq(100));
    EXPECT_THAT(event.duration_ps(), Eq(100));
    RawData loaded_raw;
    ASSERT_TRUE(loaded_raw.ParseFromString(event.raw_data()));
    EXPECT_THAT(loaded_raw, EqualsProto(R"pb(
                  args { arg { name: "my_arg" str_value: "my_val" } }
                )pb"));
  });
}

TEST(TraceEventsLevelDbTest, ReadFullEventFromExtendedLevelDbTable) {
  using TestContainer = TraceEventsContainerBase<EventFactory, RawData>;
  TestContainer input_container;

  Device* device = input_container.MutableDevice(1);
  device->set_device_id(1);
  Resource* resource = &(*device->mutable_resources())[2];
  resource->set_resource_id(2);

  RawData raw_data;
  TraceEventArguments::Argument* arg = raw_data.mutable_args()->add_arg();
  arg->set_name("my_arg");
  arg->set_str_value("my_val");

  input_container.AddCompleteEvent(
      "ExtendedEvent", /*resource_id=*/2, /*device_id=*/1,
      Timespan::FromEndPoints(100, 200), &raw_data);

  std::string trace_events_file = GetTempFilename("ext_events_read.ldb");
  std::string trace_events_metadata_file =
      GetTempFilename("ext_metadata_read.ldb");
  std::string trace_events_prefix_trie_file =
      GetTempFilename("ext_trie_read.ldb");

  absl::Cleanup cleanup = [&] {
    tsl::Env::Default()->DeleteFile(trace_events_file).IgnoreError();
    tsl::Env::Default()->DeleteFile(trace_events_metadata_file).IgnoreError();
    tsl::Env::Default()
        ->DeleteFile(trace_events_prefix_trie_file)
        .IgnoreError();
  };

  std::unique_ptr<tsl::WritableFile> file1, file2, file3;
  ASSERT_OK(tsl::Env::Default()->NewWritableFile(trace_events_file, &file1));
  ASSERT_OK(
      tsl::Env::Default()->NewWritableFile(trace_events_metadata_file, &file2));
  ASSERT_OK(tsl::Env::Default()->NewWritableFile(trace_events_prefix_trie_file,
                                                 &file3));

  ASSERT_OK(input_container.StoreAsLevelDbTables(
      std::move(file1), std::move(file2), std::move(file3)));

  TestContainer read_container;
  ASSERT_OK(read_container.ReadFullEventFromLevelDbTable(
      trace_events_metadata_file, trace_events_file, "ExtendedEvent",
      /*timestamp_ps=*/100, /*duration_ps=*/100, /*unique_id=*/0));

  EXPECT_THAT(read_container.NumEvents(), Eq(1));
  read_container.ForAllEvents([](const TraceEvent& event) {
    EXPECT_THAT(event.name(), Eq("ExtendedEvent"));
    EXPECT_THAT(event.timestamp_ps(), Eq(100));
    EXPECT_THAT(event.duration_ps(), Eq(100));
    RawData loaded_raw;
    ASSERT_TRUE(loaded_raw.ParseFromString(event.raw_data()));
    EXPECT_THAT(loaded_raw, EqualsProto(R"pb(
                  args { arg { name: "my_arg" str_value: "my_val" } }
                )pb"));
  });
}

TEST(TraceEventsLevelDbTest, LoadFromLegacyLevelDbTable) {
  // 1. Prepare trace metadata
  Trace trace;
  trace.set_min_timestamp_ps(0);
  trace.set_max_timestamp_ps(1000);
  trace.set_num_events(1);
  google::protobuf::Map<uint64_t, std::string>* name_table = trace.mutable_name_table();
  (*name_table)[12345] = "LegacyEvent";

  // 2. Set up temporary files
  std::string trace_events_file = GetTempFilename("legacy_events.ldb");
  std::string trace_events_metadata_file =
      GetTempFilename("legacy_metadata.ldb");
  std::string trace_events_prefix_trie_file =
      GetTempFilename("legacy_trie.ldb");

  absl::Cleanup cleanup = [&] {
    tsl::Env::Default()->DeleteFile(trace_events_file).IgnoreError();
    tsl::Env::Default()->DeleteFile(trace_events_metadata_file).IgnoreError();
    tsl::Env::Default()
        ->DeleteFile(trace_events_prefix_trie_file)
        .IgnoreError();
  };

  // 3. Write legacy trace events table (using 10-byte keys manually)
  {
    std::unique_ptr<tsl::WritableFile> wfile;
    ASSERT_OK(tsl::Env::Default()->NewWritableFile(trace_events_file, &wfile));

    tsl::table::Options options;
    tsl::table::TableBuilder builder(options, wfile.get());

    // Add metadata record at standard key
    builder.Add(kTraceMetadataKey, trace.SerializeAsString());

    // Create an event
    TraceEvent event;
    event.set_name_ref(12345);
    event.set_device_id(1);
    event.set_resource_id(2);
    event.set_duration_ps(100);

    // Key length explicitly set to 10
    std::string legacy_key =
        LevelDbTableKey(/*zoom_level=*/0, /*timestamp=*/100,
                        /*repetition=*/0, kLegacyKeyLength);
    std::string serialized_event;
    TraceEvent reusable_event;
    ASSERT_OK(SerializeTraceEventForPersistingFullEvent(event, reusable_event,
                                                        serialized_event));

    builder.Add(legacy_key, serialized_event);
    ASSERT_OK(builder.Finish());
  }

  // 4. Write legacy metadata table (using 10-byte keys manually)
  {
    std::unique_ptr<tsl::WritableFile> wfile;
    ASSERT_OK(tsl::Env::Default()->NewWritableFile(trace_events_metadata_file,
                                                   &wfile));

    tsl::table::Options options;
    tsl::table::TableBuilder builder(options, wfile.get());

    // Serialize a valid RawData protobuf containing metadata arguments
    RawData raw_data;
    TraceEventArguments::Argument* arg = raw_data.mutable_args()->add_arg();
    arg->set_name("my_arg");
    arg->set_str_value("legacy_metadata_payload");

    TraceEvent event_metadata;
    event_metadata.set_device_id(1);
    event_metadata.set_resource_id(2);
    raw_data.SerializeToString(event_metadata.mutable_raw_data());

    std::string legacy_key =
        LevelDbTableKey(/*zoom_level=*/0, /*timestamp=*/100,
                        /*repetition=*/0, kLegacyKeyLength);
    std::string serialized_metadata;
    TraceEvent reusable_metadata;
    ASSERT_OK(SerializeTraceEventForPersistingOnlyMetadata(
        event_metadata, reusable_metadata, serialized_metadata));

    builder.Add(legacy_key, serialized_metadata);
    ASSERT_OK(builder.Finish());
  }

  // 5. Load using LoadFromLevelDbTable (will dynamically detect 10-byte keys)
  using TestContainer = TraceEventsContainerBase<EventFactory, RawData>;
  TestContainer load_container;

  TraceEventsLevelDbFilePaths file_paths = {
      .trace_events_file_path = trace_events_file,
      .trace_events_metadata_file_path = trace_events_metadata_file,
      .trace_events_prefix_trie_file_path = trace_events_prefix_trie_file,
  };

  ASSERT_OK(load_container.LoadFromLevelDbTable(
      file_paths, /*filter=*/nullptr, /*visibility=*/nullptr,
      /*filter_by_visibility_threshold=*/-1LL, /*load_metadata=*/true));

  // 6. Verify loaded event (name_ref is preserved but name remains empty)
  EXPECT_THAT(load_container.NumEvents(), Eq(1));
  load_container.ForAllEvents([](const TraceEvent& event) {
    EXPECT_THAT(event.name_ref(), Eq(12345));
    EXPECT_THAT(event.name(), Eq(""));
    EXPECT_THAT(event.timestamp_ps(), Eq(100));
    EXPECT_THAT(event.duration_ps(), Eq(100));
    RawData loaded_raw;
    ASSERT_TRUE(loaded_raw.ParseFromString(event.raw_data()));
    EXPECT_THAT(
        loaded_raw, EqualsProto(R"pb(
          args { arg { name: "my_arg" str_value: "legacy_metadata_payload" } }
        )pb"));
  });
}

TEST(TraceEventsLevelDbTest, ReadFullEventFromLegacyLevelDbTable) {
  // 1. Prepare trace metadata
  Trace trace;
  trace.set_min_timestamp_ps(0);
  trace.set_max_timestamp_ps(1000);
  trace.set_num_events(1);
  google::protobuf::Map<uint64_t, std::string>* name_table = trace.mutable_name_table();
  (*name_table)[12345] = "LegacyEvent";

  // 2. Set up temporary files
  std::string trace_events_file = GetTempFilename("legacy_events_read.ldb");
  std::string trace_events_metadata_file =
      GetTempFilename("legacy_metadata_read.ldb");
  std::string trace_events_prefix_trie_file =
      GetTempFilename("legacy_trie_read.ldb");

  absl::Cleanup cleanup = [&] {
    tsl::Env::Default()->DeleteFile(trace_events_file).IgnoreError();
    tsl::Env::Default()->DeleteFile(trace_events_metadata_file).IgnoreError();
    tsl::Env::Default()
        ->DeleteFile(trace_events_prefix_trie_file)
        .IgnoreError();
  };

  // 3. Write legacy trace events table (using 10-byte keys manually)
  {
    std::unique_ptr<tsl::WritableFile> wfile;
    ASSERT_OK(tsl::Env::Default()->NewWritableFile(trace_events_file, &wfile));

    tsl::table::Options options;
    tsl::table::TableBuilder builder(options, wfile.get());

    // Add metadata record at standard key
    builder.Add(kTraceMetadataKey, trace.SerializeAsString());

    // Create an event
    TraceEvent event;
    event.set_name_ref(12345);
    event.set_device_id(1);
    event.set_resource_id(2);
    event.set_duration_ps(100);

    // Key length explicitly set to 10
    std::string legacy_key =
        LevelDbTableKey(/*zoom_level=*/0, /*timestamp=*/100,
                        /*repetition=*/0, kLegacyKeyLength);
    std::string serialized_event;
    TraceEvent reusable_event;
    ASSERT_OK(SerializeTraceEventForPersistingFullEvent(event, reusable_event,
                                                        serialized_event));

    builder.Add(legacy_key, serialized_event);
    ASSERT_OK(builder.Finish());
  }

  // 4. Write legacy metadata table (using 10-byte keys manually)
  {
    std::unique_ptr<tsl::WritableFile> wfile;
    ASSERT_OK(tsl::Env::Default()->NewWritableFile(trace_events_metadata_file,
                                                   &wfile));

    tsl::table::Options options;
    tsl::table::TableBuilder builder(options, wfile.get());

    // Serialize a valid RawData protobuf containing metadata arguments
    RawData raw_data;
    TraceEventArguments::Argument* arg = raw_data.mutable_args()->add_arg();
    arg->set_name("my_arg");
    arg->set_str_value("legacy_metadata_payload");

    TraceEvent event_metadata;
    event_metadata.set_device_id(1);
    event_metadata.set_resource_id(2);
    raw_data.SerializeToString(event_metadata.mutable_raw_data());

    std::string legacy_key =
        LevelDbTableKey(/*zoom_level=*/0, /*timestamp=*/100,
                        /*repetition=*/0, kLegacyKeyLength);
    std::string serialized_metadata;
    TraceEvent reusable_metadata;
    ASSERT_OK(SerializeTraceEventForPersistingOnlyMetadata(
        event_metadata, reusable_metadata, serialized_metadata));

    builder.Add(legacy_key, serialized_metadata);
    ASSERT_OK(builder.Finish());
  }

  // 5. Verify ReadFullEventFromLevelDbTable (will dynamically detect 10-byte
  // keys and fully resolve name)
  using TestContainer = TraceEventsContainerBase<EventFactory, RawData>;
  TestContainer read_container;
  ASSERT_OK(read_container.ReadFullEventFromLevelDbTable(
      trace_events_metadata_file, trace_events_file, "LegacyEvent",
      /*timestamp_ps=*/100, /*duration_ps=*/100, /*unique_id=*/0));

  EXPECT_THAT(read_container.NumEvents(), Eq(1));
  read_container.ForAllEvents([](const TraceEvent& event) {
    EXPECT_THAT(event.name(), Eq("LegacyEvent"));
    EXPECT_THAT(event.timestamp_ps(), Eq(100));
    EXPECT_THAT(event.duration_ps(), Eq(100));
    RawData loaded_raw;
    ASSERT_TRUE(loaded_raw.ParseFromString(event.raw_data()));
    EXPECT_THAT(
        loaded_raw, EqualsProto(R"pb(
          args { arg { name: "my_arg" str_value: "legacy_metadata_payload" } }
        )pb"));
  });
}

TEST(TraceEventsLevelDbTest, ReadFullEventFromLevelDbTableWithCounter) {
  using TestContainer = TraceEventsContainerBase<EventFactory, RawData>;
  TestContainer input_container;

  Device* device = input_container.MutableDevice(1);
  device->set_device_id(1);
  Resource* resource = &(*device->mutable_resources())[2];
  resource->set_resource_id(2);

  RawData raw_data;
  TraceEventArguments::Argument* arg = raw_data.mutable_args()->add_arg();
  arg->set_name("counter_val");
  arg->set_int_value(42);

  input_container.AddCompleteEvent(
      "CounterEvent", /*resource_id=*/2, /*device_id=*/1,
      Timespan::FromEndPoints(100, 100), &raw_data);

  std::string trace_events_file = GetTempFilename("counter_events.ldb");
  std::string trace_events_metadata_file =
      GetTempFilename("counter_metadata.ldb");
  std::string trace_events_prefix_trie_file =
      GetTempFilename("counter_trie.ldb");

  absl::Cleanup cleanup = [&] {
    tsl::Env::Default()->DeleteFile(trace_events_file).IgnoreError();
    tsl::Env::Default()->DeleteFile(trace_events_metadata_file).IgnoreError();
    tsl::Env::Default()
        ->DeleteFile(trace_events_prefix_trie_file)
        .IgnoreError();
  };

  std::unique_ptr<tsl::WritableFile> file1, file2, file3;
  ASSERT_OK(tsl::Env::Default()->NewWritableFile(trace_events_file, &file1));
  ASSERT_OK(
      tsl::Env::Default()->NewWritableFile(trace_events_metadata_file, &file2));
  ASSERT_OK(tsl::Env::Default()->NewWritableFile(trace_events_prefix_trie_file,
                                                 &file3));

  ASSERT_OK(input_container.StoreAsLevelDbTables(
      std::move(file1), std::move(file2), std::move(file3)));

  TestContainer read_container;
  ASSERT_OK(read_container.ReadFullEventFromLevelDbTable(
      trace_events_metadata_file, trace_events_file, "CounterEvent",
      /*timestamp_ps=*/100, /*duration_ps=*/0, /*unique_id=*/0));

  EXPECT_THAT(read_container.NumEvents(), Eq(1));
  read_container.ForAllEvents([](const TraceEvent& event) {
    EXPECT_THAT(event.name(), Eq("CounterEvent"));
    EXPECT_THAT(event.timestamp_ps(), Eq(100));
    EXPECT_THAT(event.duration_ps(), Eq(0));
    RawData loaded_raw;
    ASSERT_TRUE(loaded_raw.ParseFromString(event.raw_data()));
    EXPECT_THAT(loaded_raw, EqualsProto(R"pb(
                  args { arg { name: "counter_val" int_value: 42 } }
                )pb"));
  });
}

#ifdef NDEBUG
TEST(TraceEventsKeyLengthTest, TimestampFromLevelDbTableKeyShortKeyNonDebug) {
  EXPECT_EQ(TimestampFromLevelDbTableKey("short"), 0);
}

TEST(TraceEventsKeyLengthTest, LevelDbTableKeyInvalidKeyLengthNonDebug) {
  EXPECT_EQ(LevelDbTableKey(1, 100, 0, 5), "");
}

TEST(TraceEventsKeyLengthTest, LevelDbTableKeyInvalidZoomLevelNonDebug) {
  EXPECT_EQ(LevelDbTableKey(-1, 100, 0, kExtendedKeyLength), "");
  EXPECT_EQ(LevelDbTableKey(NumLevels(), 100, 0, kExtendedKeyLength), "");
}
#else
TEST(TraceEventsDeathTest, LevelDbTableKeyInvalidKeyLength) {
  EXPECT_DEBUG_DEATH(LevelDbTableKey(1, 100, 0, 5),
                     "Invalid key length requested");
}

TEST(TraceEventsDeathTest, LevelDbTableKeyInvalidZoomLevel) {
  EXPECT_DEBUG_DEATH(LevelDbTableKey(-1, 100, 0, kExtendedKeyLength),
                     "Invalid zoom level requested");
}
#endif

TEST(TraceEventsKeyLengthTest, DetectDbKeyLengthInvalidFormat) {
  std::string temp_file = GetTempFilename("corrupted.ldb");
  absl::Cleanup cleanup = [&] {
    tsl::Env::Default()->DeleteFile(temp_file).IgnoreError();
  };

  {
    std::unique_ptr<tsl::WritableFile> wfile;
    ASSERT_OK(tsl::Env::Default()->NewWritableFile(temp_file, &wfile));
    tsl::table::Options options;
    tsl::table::TableBuilder builder(options, wfile.get());
    builder.Add("12345", "value");  // 5 bytes key
    ASSERT_OK(builder.Finish());
  }

  tsl::table::Table* table = nullptr;
  std::unique_ptr<tsl::RandomAccessFile> file;
  ASSERT_OK(OpenLevelDbTable(temp_file, &table, file));
  std::unique_ptr<tsl::table::Table> table_deleter(table);
  std::unique_ptr<tsl::table::Iterator> iterator(table->NewIterator());
  iterator->SeekToFirst();

  auto status = DetectDbKeyLength(iterator.get());
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.status().code(), absl::StatusCode::kInternal);
  EXPECT_THAT(status.status().message(),
              testing::HasSubstr("Invalid or corrupted database key format"));
}

TEST(TraceEventsLevelDbTest,
     ReadFullEventFromLevelDbTableMismatchedOrNotFound) {
  using TestContainer = TraceEventsContainerBase<EventFactory, RawData>;
  TestContainer input_container;

  Device* device = input_container.MutableDevice(1);
  device->set_device_id(1);
  Resource* resource = &(*device->mutable_resources())[2];
  resource->set_resource_id(2);

  input_container.AddCompleteEvent("ExtendedEvent", /*resource_id=*/2,
                                   /*device_id=*/1,
                                   Timespan::FromEndPoints(100, 200), nullptr);

  std::string trace_events_file = GetTempFilename("ext_events_err.ldb");
  std::string trace_events_metadata_file =
      GetTempFilename("ext_metadata_err.ldb");
  std::string trace_events_prefix_trie_file =
      GetTempFilename("ext_trie_err.ldb");

  absl::Cleanup cleanup = [&] {
    tsl::Env::Default()->DeleteFile(trace_events_file).IgnoreError();
    tsl::Env::Default()->DeleteFile(trace_events_metadata_file).IgnoreError();
    tsl::Env::Default()
        ->DeleteFile(trace_events_prefix_trie_file)
        .IgnoreError();
  };

  std::unique_ptr<tsl::WritableFile> file1, file2, file3;
  ASSERT_OK(tsl::Env::Default()->NewWritableFile(trace_events_file, &file1));
  ASSERT_OK(
      tsl::Env::Default()->NewWritableFile(trace_events_metadata_file, &file2));
  ASSERT_OK(tsl::Env::Default()->NewWritableFile(trace_events_prefix_trie_file,
                                                 &file3));

  ASSERT_OK(input_container.StoreAsLevelDbTables(
      std::move(file1), std::move(file2), std::move(file3)));

  TestContainer read_container;
  // Test case 1: Mismatched event name (triggers continue branch)
  {
    auto status = read_container.ReadFullEventFromLevelDbTable(
        trace_events_metadata_file, trace_events_file, "MismatchedEvent",
        /*timestamp_ps=*/100, /*duration_ps=*/100, /*unique_id=*/0);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), absl::StatusCode::kNotFound);
  }

  // Test case 2: Event not found (triggers NotFoundError at the end of loop)
  {
    auto status = read_container.ReadFullEventFromLevelDbTable(
        trace_events_metadata_file, trace_events_file, "ExtendedEvent",
        /*timestamp_ps=*/999, /*duration_ps=*/100, /*unique_id=*/0);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), absl::StatusCode::kNotFound);
  }
}

}  // namespace
}  // namespace tensorflow::profiler
