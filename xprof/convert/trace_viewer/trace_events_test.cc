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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "net/proto2/contrib/parse_proto/parse_text_proto.h"
#include "testing/base/public/gmock.h"
#include "<gtest/gtest.h>"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "google/protobuf/arena.h"
#include "xla/tsl/lib/io/table_builder.h"
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
  auto* arg1 = raw_data1.mutable_args()->add_arg();
  arg1->set_name("request_id1");
  arg1->set_str_value("target_req_123");

  // And some non-searchable metadata in the same event
  auto* arg2 = raw_data1.mutable_args()->add_arg();
  arg2->set_name("bytes");
  arg2->set_uint_value(1024);

  input_container.AddCompleteEvent(
      "ModelForward", /*resource_id=*/2, /*device_id=*/1,
      Timespan::FromEndPoints(100, 200), &raw_data1);

  // Add Event 2: Name "OtherEvent" with non-searchable metadata some_other_key:
  // "req_ignored"
  RawData raw_data2;
  auto* arg3 = raw_data2.mutable_args()->add_arg();
  arg3->set_name("some_other_key");
  arg3->set_str_value("req_ignored");

  input_container.AddCompleteEvent(
      "OtherEvent", /*resource_id=*/2, /*device_id=*/1,
      Timespan::FromEndPoints(300, 400), &raw_data2);

  // 2. Store the container as LevelDB tables
  std::string trace_events_file = GetTempFilename("events.ldb");
  std::string trace_events_metadata_file = GetTempFilename("metadata.ldb");
  std::string trace_events_prefix_trie_file = GetTempFilename("trie.ldb");

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
      for (const auto& arg : raw_data.args().arg()) {
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
      for (const auto& arg : raw_data.args().arg()) {
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
    auto* arg = raw_data3.mutable_args()->add_arg();
    arg->set_name("request_id1");
    arg->set_str_value("req_val");

    deduplication_container.AddCompleteEvent(
        "req_event", /*resource_id=*/2, /*device_id=*/1,
        Timespan::FromEndPoints(500, 600), &raw_data3);

    std::string f1 = GetTempFilename("events_dedup.ldb");
    std::string f2 = GetTempFilename("metadata_dedup.ldb");
    std::string f3 = GetTempFilename("trie_dedup.ldb");

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
  auto* arg1 = raw_data1.mutable_args()->add_arg();
  arg1->set_name("request_id1");
  arg1->set_str_value(long_request_id);

  input_container.AddCompleteEvent(
      event_name, /*resource_id=*/2, /*device_id=*/1,
      Timespan::FromEndPoints(100, 200), &raw_data1);

  // 3. Non-string metadata value (should be ignored by search indexer)
  RawData raw_data2;
  auto* arg_int = raw_data2.mutable_args()->add_arg();
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

TEST(TraceEventsVisibilityTest, FlowEventsZoomLevelAssignmentAndSafetyGuard) {
  Trace trace;
  trace.set_min_timestamp_ps(0);
  trace.set_max_timestamp_ps(10000000000000ULL);  // 10s

  // Prior flow event on the same track to make flow 42 invisible at zoom
  // level 0
  TraceEvent event0;
  event0.set_device_id(1);
  event0.set_resource_id(2);
  event0.set_timestamp_ps(1000);
  event0.set_duration_ps(100);
  event0.set_flow_id(41);
  event0.set_flow_entry_type(TraceEvent::FLOW_START);

  // Event 1: Tiny event in flow 42 close to flow 41 (invisible at level 0)
  TraceEvent event1;
  event1.set_device_id(1);
  event1.set_resource_id(2);
  event1.set_timestamp_ps(5000);
  event1.set_duration_ps(100);
  event1.set_flow_id(42);
  event1.set_flow_entry_type(TraceEvent::FLOW_START);

  // Event 2: Huge event in flow 42 (visible at level 0 due to track duration)
  TraceEvent event2;
  event2.set_device_id(1);
  event2.set_resource_id(2);
  event2.set_name("XlaModuleEvent");
  event2.set_timestamp_ps(100000000);
  event2.set_duration_ps(2000000000000ULL);  // 2s duration (> 1s resolution)
  event2.set_flow_id(42);
  event2.set_flow_entry_type(TraceEvent::FLOW_END);

  TraceEventTrack track = {&event0, &event1, &event2};
  std::vector<const TraceEventTrack*> event_tracks = {&track};

  std::vector<std::vector<const TraceEvent*>> events_by_level =
      GetEventsByLevel(trace, event_tracks);

  // Verification: Zoom Level Assignment for Flow Events
  // Because flow 42 was marked invisible at level 0 (due to event1 being
  // tiny), event2 is only visible at zoom level 0 because of its massive
  // duration (track visibility). This actively verifies the "OR" semantics
  // logic.
  bool found_event2_at_level0 = false;
  for (const auto* event : events_by_level[0]) {
    if (event->name() == "XlaModuleEvent") {
      found_event2_at_level0 = true;
    }
  }
  EXPECT_TRUE(found_event2_at_level0);
}

TEST(TraceEventsVisibilityTest, SafetyGuardForFallbackFlowEvents) {
  Trace trace;
  trace.set_min_timestamp_ps(0);
  trace.set_max_timestamp_ps(1000000000000ULL);  // 1s

  // Prior event to establish depth/resolution baseline
  TraceEvent event0;
  event0.set_device_id(1);
  event0.set_resource_id(2);
  event0.set_timestamp_ps(10000);
  event0.set_duration_ps(100);

  // Tiny flow event placed extremely close (1 ps) to event0
  // distance < kLayerResolutions[kNumLevels-2] (10ps), forces fallback to
  // level 12
  TraceEvent event1;
  event1.set_device_id(1);
  event1.set_resource_id(2);
  event1.set_name("TinyFlowEvent");
  event1.set_timestamp_ps(10101);  // 101ps distance
  event1.set_duration_ps(0);
  event1.set_flow_id(43);
  event1.set_flow_entry_type(TraceEvent::FLOW_START);

  TraceEventTrack track = {&event0, &event1};
  std::vector<const TraceEventTrack*> event_tracks = {&track};

  // This should not crash! Actively exercises the safety guard to prevent
  // out-of-bounds accesses.
  EXPECT_NO_FATAL_FAILURE({
    std::vector<std::vector<const TraceEvent*>> events_by_level =
        GetEventsByLevel(trace, event_tracks);
  });
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
  auto* arg = raw_data.mutable_args()->add_arg();
  arg->set_name("my_arg");
  arg->set_str_value("my_val");

  input_container.AddCompleteEvent(
      "ExtendedEvent", /*resource_id=*/2, /*device_id=*/1,
      Timespan::FromEndPoints(100, 200), &raw_data);

  // Get temporary paths
  std::string trace_events_file = GetTempFilename("ext_events.ldb");
  std::string trace_events_metadata_file = GetTempFilename("ext_metadata.ldb");
  std::string trace_events_prefix_trie_file = GetTempFilename("ext_trie.ldb");

  std::unique_ptr<tsl::WritableFile> file1, file2, file3;
  ASSERT_OK(tsl::Env::Default()->NewWritableFile(trace_events_file, &file1));
  ASSERT_OK(
      tsl::Env::Default()->NewWritableFile(trace_events_metadata_file, &file2));
  ASSERT_OK(tsl::Env::Default()->NewWritableFile(trace_events_prefix_trie_file,
                                                 &file3));

  // Store as LevelDB Tables (uses 11-byte extended keys by default)
  ASSERT_OK(input_container.StoreAsLevelDbTables(
      std::move(file1), std::move(file2), std::move(file3)));

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
    ASSERT_TRUE(loaded_raw.has_args());
    bool found_arg = false;
    for (const auto& arg : loaded_raw.args().arg()) {
      if (arg.name() == "my_arg") {
        EXPECT_THAT(arg.str_value(), Eq("my_val"));
        found_arg = true;
      }
    }
    EXPECT_TRUE(found_arg);
  });

  // Verify ReadFullEventFromLevelDbTable
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
    EXPECT_THAT(loaded_raw.args().arg(0).str_value(), Eq("my_val"));
  });
}

TEST(TraceEventsLevelDbTest, LoadFromLegacyLevelDbTable) {
  // 1. Prepare trace metadata
  Trace trace;
  trace.set_min_timestamp_ps(0);
  trace.set_max_timestamp_ps(1000);
  trace.set_num_events(1);
  auto* name_table = trace.mutable_name_table();
  (*name_table)[12345] = "LegacyEvent";

  // 2. Set up temporary files
  std::string trace_events_file = GetTempFilename("legacy_events.ldb");
  std::string trace_events_metadata_file =
      GetTempFilename("legacy_metadata.ldb");
  std::string trace_events_prefix_trie_file =
      GetTempFilename("legacy_trie.ldb");

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
    auto* arg = raw_data.mutable_args()->add_arg();
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
    ASSERT_TRUE(loaded_raw.has_args());
    EXPECT_THAT(loaded_raw.args().arg(0).str_value(),
                Eq("legacy_metadata_payload"));
  });

  // 7. Verify ReadFullEventFromLevelDbTable (will dynamically detect 10-byte
  // keys and fully resolve name)
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
    ASSERT_TRUE(loaded_raw.has_args());
    EXPECT_THAT(loaded_raw.args().arg(0).str_value(),
                Eq("legacy_metadata_payload"));
  });
}

}  // namespace
}  // namespace tensorflow::profiler
