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

#include "xprof/convert/events_db/custom_trace_parser.h"

#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/tsl/profiler/utils/xplane_builder.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/events_db/event_utils.h"
#include "xprof/convert/events_db/record_consumer.h"
#include "xprof/convert/events_db/schema.h"

namespace xprof::events_db::internal {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;

TEST(CustomTraceParserTest, ParsesCustomMegascaleEvents) {
  tensorflow::profiler::XPlane plane;
  plane.set_name("/device:CUSTOM:Megascale Trace");
  tsl::profiler::XPlaneBuilder plane_builder(&plane);
  tsl::profiler::XLineBuilder line_builder = plane_builder.GetOrCreateLine(1);
  line_builder.SetName("Collective Optical Flow");

  tsl::profiler::XEventBuilder event_builder = line_builder.AddEvent(
      *plane_builder.GetOrCreateEventMetadata("AllReduce_Optical_Chunk"));
  event_builder.SetTimestampNs(100);
  event_builder.SetDurationNs(400);

  Schema schema;
  FieldIndices indices(schema);
  std::vector<Record> parsed_records;

  EXPECT_THAT(
      ParseCustomTrace(plane, indices,
                       [&](Record& record) -> absl::StatusOr<StepControl> {
                         parsed_records.push_back(std::move(record));
                         return StepControl::kContinue;
                       }),
      IsOkAndHolds(ParseStatus::kComplete));

  ASSERT_EQ(parsed_records.size(), 1);
  const Record& record = parsed_records[0];

  EXPECT_EQ(record[indices.device], "/device:CUSTOM:Megascale Trace");
  EXPECT_EQ(record[indices.stream_id], 1);
  EXPECT_EQ(record[indices.kernel_name], "AllReduce_Optical_Chunk");
  EXPECT_EQ(record[indices.start_ns], 100);
  EXPECT_EQ(record[indices.end_ns], 500);
  EXPECT_EQ(record[indices.self_time_ns], 400);
  EXPECT_EQ(record[indices.category], "Collective Optical Flow");
}

TEST(CustomTraceParserTest, StopsEarlyWhenRequested) {
  tensorflow::profiler::XPlane plane;
  tsl::profiler::XPlaneBuilder plane_builder(&plane);
  tsl::profiler::XLineBuilder line_builder = plane_builder.GetOrCreateLine(1);

  for (int i = 0; i < 5; ++i) {
    tsl::profiler::XEventBuilder event_builder = line_builder.AddEvent(
        *plane_builder.GetOrCreateEventMetadata(absl::StrCat("Event_", i)));
    event_builder.SetTimestampNs(100 * (i + 1));
    event_builder.SetDurationNs(10);
  }

  Schema schema;
  FieldIndices indices(schema);
  int records_seen = 0;

  EXPECT_THAT(
      ParseCustomTrace(plane, indices,
                       [&](Record&) -> absl::StatusOr<StepControl> {
                         ++records_seen;
                         return records_seen == 2 ? StepControl::kStop
                                                  : StepControl::kContinue;
                       }),
      IsOkAndHolds(ParseStatus::kStoppedEarly));

  EXPECT_EQ(records_seen, 2);
}

TEST(CustomTraceParserTest, PropagatesConsumerError) {
  tensorflow::profiler::XPlane plane;
  tsl::profiler::XPlaneBuilder plane_builder(&plane);
  tsl::profiler::XLineBuilder line_builder = plane_builder.GetOrCreateLine(1);
  tsl::profiler::XEventBuilder event_builder =
      line_builder.AddEvent(*plane_builder.GetOrCreateEventMetadata("Event_0"));
  event_builder.SetTimestampNs(100);
  event_builder.SetDurationNs(10);

  Schema schema;
  FieldIndices indices(schema);

  EXPECT_THAT(
      ParseCustomTrace(plane, indices,
                       [&](Record&) -> absl::StatusOr<StepControl> {
                         return absl::InternalError("consumer error");
                       }),
      StatusIs(absl::StatusCode::kInternal, "consumer error"));
}

}  // namespace
}  // namespace xprof::events_db::internal
