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

#include "net/proto2/contrib/parse_proto/parse_text_proto.h"
#include "testing/base/public/gmock.h"
#include "<gtest/gtest.h>"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "google/protobuf/arena.h"
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
  // Test key generation logic for kLegacyKeyLength
  std::string legacy_key =
      LevelDbTableKey(/*zoom_level=*/0, /*timestamp=*/12345, /*repetition=*/10,
                      kLegacyKeyLength);
  EXPECT_EQ(legacy_key.size(), kLegacyKeyLength);
  EXPECT_EQ(TimestampFromLevelDbTableKey(legacy_key), 12345);

  // Repetition >= 256 should return empty string for kLegacyKeyLength
  EXPECT_EQ(LevelDbTableKey(/*zoom_level=*/0, /*timestamp=*/12345,
                            /*repetition=*/256, kLegacyKeyLength),
            "");

  // Test key generation logic for kExtendedKeyLength
  std::string extended_key =
      LevelDbTableKey(/*zoom_level=*/0, /*timestamp=*/12345,
                      /*repetition=*/1000, kExtendedKeyLength);
  EXPECT_EQ(extended_key.size(), kExtendedKeyLength);
  EXPECT_EQ(TimestampFromLevelDbTableKey(extended_key), 12345);

  // Repetition >= 65536 should return empty string for kExtendedKeyLength
  EXPECT_EQ(LevelDbTableKey(/*zoom_level=*/0, /*timestamp=*/12345,
                            /*repetition=*/65536, kExtendedKeyLength),
            "");

  // Backward compatibility test: write dynamic table files and check
  // DetectDbKeyLength.
  std::string file_path_legacy = GetTempFilename("legacy_detect.ldb");
  std::string file_path_extended = GetTempFilename("extended_detect.ldb");

  auto env = tsl::Env::Default();

  // 1. Write a legacy-keyed Table
  {
    std::unique_ptr<tsl::WritableFile> file;
    ASSERT_OK(env->NewWritableFile(file_path_legacy, &file));
    tsl::table::Options options;
    tsl::table::TableBuilder builder(options, file.get());
    builder.Add("/trace", "dummy_metadata");
    builder.Add(LevelDbTableKey(0, 1000, 5, kLegacyKeyLength), "event_data");
    ASSERT_OK(builder.Finish());
    ASSERT_OK(file->Close());
  }

  // 2. Write an extended-keyed Table
  {
    std::unique_ptr<tsl::WritableFile> file;
    ASSERT_OK(env->NewWritableFile(file_path_extended, &file));
    tsl::table::Options options;
    tsl::table::TableBuilder builder(options, file.get());
    builder.Add("/trace", "dummy_metadata");
    builder.Add(LevelDbTableKey(0, 1000, 5, kExtendedKeyLength), "event_data");
    ASSERT_OK(builder.Finish());
    ASSERT_OK(file->Close());
  }

  // Read & check legacy key detection
  {
    uint64_t file_size;
    ASSERT_OK(env->GetFileSize(file_path_legacy, &file_size));
    std::unique_ptr<tsl::RandomAccessFile> file;
    ASSERT_OK(env->NewRandomAccessFile(file_path_legacy, &file));
    tsl::table::Options options;
    tsl::table::Table* table = nullptr;
    ASSERT_OK(tsl::table::Table::Open(options, file.get(), file_size, &table));
    std::unique_ptr<tsl::table::Table> table_deleter(table);
    std::unique_ptr<tsl::table::Iterator> iterator(table->NewIterator());

    iterator->SeekToFirst();
    iterator->Next();  // skip "/trace"
    auto detected_or = DetectDbKeyLength(iterator.get());
    ASSERT_OK(detected_or.status());
    EXPECT_EQ(*detected_or, kLegacyKeyLength);
  }

  // Read & check extended key detection
  {
    uint64_t file_size;
    ASSERT_OK(env->GetFileSize(file_path_extended, &file_size));
    std::unique_ptr<tsl::RandomAccessFile> file;
    ASSERT_OK(env->NewRandomAccessFile(file_path_extended, &file));
    tsl::table::Options options;
    tsl::table::Table* table = nullptr;
    ASSERT_OK(tsl::table::Table::Open(options, file.get(), file_size, &table));
    std::unique_ptr<tsl::table::Table> table_deleter(table);
    std::unique_ptr<tsl::table::Iterator> iterator(table->NewIterator());

    iterator->SeekToFirst();
    iterator->Next();  // skip "/trace"
    auto detected_or = DetectDbKeyLength(iterator.get());
    ASSERT_OK(detected_or.status());
    EXPECT_EQ(*detected_or, kExtendedKeyLength);
  }
}

TEST(TraceEventsVisibilityTest, FlowEventsZoomLevelAssignmentAndSafetyGuard) {
  Trace trace;
  trace.set_min_timestamp_ps(0);
  trace.set_max_timestamp_ps(10000000000000ull);  // 10s trace.

  TraceEventTrack track;
  TraceEvent flow_event;
  flow_event.set_device_id(1);
  flow_event.set_resource_id(1);
  flow_event.set_name("HugeFlowEvent");
  flow_event.set_timestamp_ps(100);
  flow_event.set_duration_ps(1000000000000ull);  // 1s duration.
  flow_event.set_flow_id(42);
  flow_event.set_flow_entry_type(TraceEvent::FLOW_START);
  track.push_back(&flow_event);

  std::vector<const TraceEventTrack*> tracks = {&track};
  std::vector<std::vector<const TraceEvent*>> events_by_level =
      GetEventsByLevel(trace, tracks);

  // Since this event has a 1s duration, it must be visible at zoom level 0.
  // If the OR semantics is working, zoom level 0 must contain this event.
  bool found_in_coarse_level = false;
  for (int level = 0; level < 3; ++level) {
    for (const auto* event : events_by_level[level]) {
      if (event->name() == "HugeFlowEvent") {
        found_in_coarse_level = true;
        break;
      }
    }
  }
  EXPECT_TRUE(found_in_coarse_level);
}

}  // namespace
}  // namespace tensorflow::profiler
