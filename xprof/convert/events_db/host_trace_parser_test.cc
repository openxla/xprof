/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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

#include "xprof/convert/events_db/host_trace_parser.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "testing/base/public/gmock.h"
#include "<gtest/gtest.h>"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/tsl/profiler/utils/group_events.h"
#include "xla/tsl/profiler/utils/xplane_builder.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/events_db/event_utils.h"
#include "xprof/convert/events_db/record_consumer.h"
#include "xprof/convert/events_db/schema.h"

namespace xprof::events_db::internal {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;

TEST(HostTraceParserTest, ParsesHostEvents) {
  tensorflow::profiler::XPlane plane;
  plane.set_name("/host:CPU");
  tsl::profiler::XPlaneBuilder plane_builder(&plane);
  tsl::profiler::XLineBuilder line_builder = plane_builder.GetOrCreateLine(10);
  line_builder.SetName("WorkerThread-1");

  tsl::profiler::XEventBuilder event_builder =
      line_builder.AddEvent(*plane_builder.GetOrCreateEventMetadata(
          "training_loop/ForwardPass:MatMul"));
  event_builder.SetTimestampNs(5000);
  event_builder.SetDurationNs(2000);
  event_builder.AddStatValue(
      *plane_builder.GetOrCreateStatMetadata(
          tsl::profiler::GetStatTypeStr(tsl::profiler::StatType::kGroupId)),
      static_cast<int64_t>(1));
  event_builder.AddStatValue(
      *plane_builder.GetOrCreateStatMetadata(tsl::profiler::GetStatTypeStr(
          tsl::profiler::StatType::kCorrelationId)),
      static_cast<uint64_t>(999));
  event_builder.AddStatValue(
      *plane_builder.GetOrCreateStatMetadata(
          tsl::profiler::GetStatTypeStr(tsl::profiler::StatType::kFlow)),
      static_cast<int64_t>(42));
  event_builder.AddStatValue(
      *plane_builder.GetOrCreateStatMetadata(
          tsl::profiler::GetStatTypeStr(tsl::profiler::StatType::kSourceInfo)),
      "train.py:88");
  event_builder.AddStatValue(
      *plane_builder.GetOrCreateStatMetadata("custom_stat"), "val");

  tsl::profiler::GroupMetadataMap group_metadata_map;
  group_metadata_map[1] = {.name = "step:0"};

  Schema schema;
  FieldIndices indices(schema);
  std::vector<Record> parsed_records;

  EXPECT_THAT(
      ParseHostTrace(plane, group_metadata_map, indices,
                     [&](Record& record) -> absl::StatusOr<StepControl> {
                       parsed_records.push_back(std::move(record));
                       return StepControl::kContinue;
                     }),
      IsOkAndHolds(ParseStatus::kComplete));

  ASSERT_EQ(parsed_records.size(), 1);
  const Record& record = parsed_records[0];

  EXPECT_EQ(record[indices.device], "cpu:0");
  EXPECT_EQ(record[indices.thread_id], 10);
  EXPECT_EQ(record[indices.thread_name], "WorkerThread-1");
  EXPECT_EQ(record[indices.kernel_name], "training_loop/ForwardPass:MatMul");
  EXPECT_EQ(record[indices.tf_op_name], "training_loop/ForwardPass");
  EXPECT_EQ(record[indices.tf_op_type], "MatMul");
  EXPECT_EQ(record[indices.start_ns], 5000);
  EXPECT_EQ(record[indices.end_ns], 7000);
  EXPECT_EQ(record[indices.self_time_ns], 2000);
  EXPECT_EQ(record[indices.category], "host");
  EXPECT_EQ(record[indices.step], "step:0");
  EXPECT_EQ(record[indices.correlation_id], 999);
  EXPECT_EQ(record[indices.flow], 42);
  EXPECT_EQ(record[indices.source_line], "train.py:88");
  EXPECT_EQ(record[indices.trace_args], "custom_stat=val");
}

TEST(HostTraceParserTest, StopsEarlyWhenRequested) {
  tensorflow::profiler::XPlane plane;
  tsl::profiler::XPlaneBuilder plane_builder(&plane);
  tsl::profiler::XLineBuilder line_builder = plane_builder.GetOrCreateLine(1);

  for (int i = 0; i < 5; ++i) {
    tsl::profiler::XEventBuilder event_builder = line_builder.AddEvent(
        *plane_builder.GetOrCreateEventMetadata(absl::StrCat("Event_", i)));
    event_builder.SetTimestampNs(100 * (i + 1));
    event_builder.SetDurationNs(10);
  }

  tsl::profiler::GroupMetadataMap group_metadata_map;
  Schema schema;
  FieldIndices indices(schema);
  int records_seen = 0;

  EXPECT_THAT(
      ParseHostTrace(plane, group_metadata_map, indices,
                     [&](Record& record) -> absl::StatusOr<StepControl> {
                       ++records_seen;
                       return records_seen == 2 ? StepControl::kStop
                                                : StepControl::kContinue;
                     }),
      IsOkAndHolds(ParseStatus::kStoppedEarly));

  EXPECT_EQ(records_seen, 2);
}

TEST(HostTraceParserTest, PropagatesConsumerError) {
  tensorflow::profiler::XPlane plane;
  tsl::profiler::XPlaneBuilder plane_builder(&plane);
  tsl::profiler::XLineBuilder line_builder = plane_builder.GetOrCreateLine(1);
  tsl::profiler::XEventBuilder event_builder =
      line_builder.AddEvent(*plane_builder.GetOrCreateEventMetadata("Event_0"));
  event_builder.SetTimestampNs(100);
  event_builder.SetDurationNs(10);

  tsl::profiler::GroupMetadataMap group_metadata_map;
  Schema schema;
  FieldIndices indices(schema);

  EXPECT_THAT(
      ParseHostTrace(plane, group_metadata_map, indices,
                     [&](Record& record) -> absl::StatusOr<StepControl> {
                       return absl::InternalError("consumer error");
                     }),
      StatusIs(absl::StatusCode::kInternal, "consumer error"));
}

TEST(HostTraceParserTest, HandlesEmptyPlane) {
  tensorflow::profiler::XPlane plane;
  tsl::profiler::GroupMetadataMap group_metadata_map;
  Schema schema;
  FieldIndices indices(schema);
  int records_seen = 0;

  EXPECT_THAT(
      ParseHostTrace(plane, group_metadata_map, indices,
                     [&](Record& record) -> absl::StatusOr<StepControl> {
                       ++records_seen;
                       return StepControl::kContinue;
                     }),
      IsOkAndHolds(ParseStatus::kComplete));

  EXPECT_EQ(records_seen, 0);
}

TEST(HostTraceParserTest, HandlesLinesWithoutEvents) {
  tensorflow::profiler::XPlane plane;
  tsl::profiler::XPlaneBuilder plane_builder(&plane);
  plane_builder.GetOrCreateLine(1);

  tsl::profiler::GroupMetadataMap group_metadata_map;
  Schema schema;
  FieldIndices indices(schema);
  int records_seen = 0;

  EXPECT_THAT(
      ParseHostTrace(plane, group_metadata_map, indices,
                     [&](Record& record) -> absl::StatusOr<StepControl> {
                       ++records_seen;
                       return StepControl::kContinue;
                     }),
      IsOkAndHolds(ParseStatus::kComplete));

  EXPECT_EQ(records_seen, 0);
}

}  // namespace
}  // namespace xprof::events_db::internal
