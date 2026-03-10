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

#include <string>

#include "testing/base/public/gmock.h"
#include "<gtest/gtest.h>"
#include "absl/status/status.h"
#include "google/protobuf/arena.h"
#include "plugin/xprof/protobuf/trace_events.pb.h"
#include "plugin/xprof/protobuf/trace_events_raw.pb.h"

namespace tensorflow {
namespace profiler {
namespace {

TEST(SerializationTest, SerializeFullEventSkipsTimestampAndKeepsRawData) {
  TraceEvent event;
  event.set_device_id(1);
  event.set_resource_id(2);
  event.set_name("test_event");
  event.set_timestamp_ps(12345);
  event.set_duration_ps(100);
  event.set_raw_data("some_raw_data");

  std::string output;
  google::protobuf::Arena arena;
  TraceEvent* reusable_event_copy = google::protobuf::Arena::Create<TraceEvent>(&arena);
  EXPECT_OK(SerializeTraceEventForPersistingFullEvent(
      &event, reusable_event_copy, output));

  TraceEvent deserialized;
  ASSERT_TRUE(deserialized.ParseFromString(output));

  EXPECT_EQ(deserialized.device_id(), 1);
  EXPECT_EQ(deserialized.resource_id(), 2);
  EXPECT_EQ(deserialized.name(), "test_event");
  EXPECT_EQ(deserialized.duration_ps(), 100);
  EXPECT_FALSE(deserialized.has_timestamp_ps());
  EXPECT_TRUE(deserialized.has_raw_data());
  EXPECT_EQ(deserialized.raw_data(), "some_raw_data");
}

TEST(SerializationTest, SerializeEventWithoutMetadataSkipsTimestampAndRawData) {
  TraceEvent event;
  event.set_device_id(1);
  event.set_resource_id(2);  // Ensure it's not treated as a counter.
  event.set_name("test_event");
  event.set_timestamp_ps(12345);
  event.set_raw_data("some_raw_data");

  std::string output;
  google::protobuf::Arena arena;
  TraceEvent* reusable_event_copy = google::protobuf::Arena::Create<TraceEvent>(&arena);
  EXPECT_OK(SerializeTraceEventForPersistingEventWithoutMetadata(
      &event, reusable_event_copy, output));

  TraceEvent deserialized;
  ASSERT_TRUE(deserialized.ParseFromString(output));

  EXPECT_EQ(deserialized.device_id(), 1);
  EXPECT_EQ(deserialized.resource_id(), 2);
  EXPECT_EQ(deserialized.name(), "test_event");
  EXPECT_FALSE(deserialized.has_timestamp_ps());
  EXPECT_FALSE(deserialized.has_raw_data());
}

TEST(SerializationTest, SerializeEventWithoutMetadataKeepsRawDataForCounter) {
  TraceEvent event;
  event.set_device_id(1);
  // No resource_id -> Counter event.
  event.set_name("counter_event");
  event.set_timestamp_ps(12345);
  event.set_raw_data("some_raw_data");

  std::string output;
  google::protobuf::Arena arena;
  TraceEvent* reusable_event_copy = google::protobuf::Arena::Create<TraceEvent>(&arena);
  EXPECT_OK(SerializeTraceEventForPersistingEventWithoutMetadata(
      &event, reusable_event_copy, output));

  TraceEvent deserialized;
  ASSERT_TRUE(deserialized.ParseFromString(output));

  EXPECT_EQ(deserialized.name(), "counter_event");
  EXPECT_FALSE(deserialized.has_timestamp_ps());
  EXPECT_TRUE(deserialized.has_raw_data());
  EXPECT_EQ(deserialized.raw_data(), "some_raw_data");
}

TEST(SerializationTest, SerializeOnlyMetadataReturnsMetadata) {
  TraceEvent event;
  event.set_raw_data("metadata_payload");
  event.set_resource_id(1);  // Not a counter

  std::string output;
  google::protobuf::Arena arena;
  TraceEvent* reusable_event = google::protobuf::Arena::Create<TraceEvent>(&arena);
  EXPECT_OK(SerializeTraceEventForPersistingOnlyMetadata(&event, reusable_event,
                                                         output));

  TraceEvent deserialized;
  ASSERT_TRUE(deserialized.ParseFromString(output));

  EXPECT_TRUE(deserialized.has_raw_data());
  EXPECT_EQ(deserialized.raw_data(), "metadata_payload");
  EXPECT_FALSE(deserialized.has_resource_id());
}

TEST(SerializationTest, SerializeOnlyMetadataReturnsNotFoundForCounter) {
  TraceEvent event;
  event.set_raw_data("metadata_payload");
  // Counter event (no resource_id, no flow_id)

  std::string output;
  google::protobuf::Arena arena;
  TraceEvent* reusable_event = google::protobuf::Arena::Create<TraceEvent>(&arena);
  absl::Status status = SerializeTraceEventForPersistingOnlyMetadata(
      &event, reusable_event, output);
  EXPECT_TRUE(absl::IsNotFound(status));
}

TEST(SerializationTest, SerializeOnlyMetadataReturnsEmptyRawDataIfMissing) {
  TraceEvent event;
  event.set_resource_id(1);  // Not a counter

  std::string output;
  google::protobuf::Arena arena;
  TraceEvent* reusable_event = google::protobuf::Arena::Create<TraceEvent>(&arena);
  EXPECT_OK(SerializeTraceEventForPersistingOnlyMetadata(&event, reusable_event,
                                                         output));

  TraceEvent deserialized;
  ASSERT_TRUE(deserialized.ParseFromString(output));

  // In proto2, explicitly setting an empty string will make has_raw_data()
  // true.
  EXPECT_TRUE(deserialized.has_raw_data());
  EXPECT_EQ(deserialized.raw_data(), "");
}

TEST(SerializationTest, SerializeEventWithoutMetadataSkipsRawDataForAsync) {
  TraceEvent event;
  event.set_device_id(1);
  event.set_flow_id(2);  // No resource_id, has flow_id -> Async event.
  event.set_name("async_event");
  event.set_timestamp_ps(12345);
  event.set_raw_data("some_raw_data");

  std::string output;
  google::protobuf::Arena arena;
  TraceEvent* reusable_event_copy = google::protobuf::Arena::Create<TraceEvent>(&arena);
  EXPECT_OK(SerializeTraceEventForPersistingEventWithoutMetadata(
      &event, reusable_event_copy, output));

  TraceEvent deserialized;
  ASSERT_TRUE(deserialized.ParseFromString(output));

  EXPECT_EQ(deserialized.device_id(), 1);
  EXPECT_EQ(deserialized.flow_id(), 2);
  EXPECT_EQ(deserialized.name(), "async_event");
  EXPECT_FALSE(deserialized.has_timestamp_ps());
  EXPECT_FALSE(deserialized.has_raw_data());
}

TEST(SerializationTest, SerializeFullEventHandlesSupportedTypes) {
  TraceEvent event;
  event.set_device_id(42);                            // uint32
  event.set_resource_id(1234567890123ull);            // uint64
  event.set_timestamp_ps(9876543210987ull);           // uint64
  event.set_name("multi_type_event");                 // string
  event.set_group_id(-100);                           // int64
  event.set_flow_entry_type(TraceEvent::FLOW_START);  // enum

  std::string output;
  google::protobuf::Arena arena;
  TraceEvent* reusable_event_copy = google::protobuf::Arena::Create<TraceEvent>(&arena);
  EXPECT_OK(SerializeTraceEventForPersistingFullEvent(
      &event, reusable_event_copy, output));

  TraceEvent deserialized;
  ASSERT_TRUE(deserialized.ParseFromString(output));

  EXPECT_EQ(deserialized.device_id(), 42);
  EXPECT_EQ(deserialized.resource_id(), 1234567890123ull);
  EXPECT_EQ(deserialized.name(), "multi_type_event");
  EXPECT_EQ(deserialized.group_id(), -100);
  EXPECT_EQ(deserialized.flow_entry_type(), TraceEvent::FLOW_START);
  EXPECT_FALSE(deserialized.has_timestamp_ps());
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
