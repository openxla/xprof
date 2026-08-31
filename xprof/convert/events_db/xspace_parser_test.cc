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

#include "xprof/convert/events_db/xspace_parser.h"

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "testing/base/public/gmock.h"
#include "<gtest/gtest.h>"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/service/hlo.pb.h"
#include "xla/tsl/profiler/utils/group_events.h"
#include "xla/tsl/profiler/utils/xplane_builder.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/events_db/event_utils.h"
#include "xprof/convert/events_db/record_consumer.h"
#include "xprof/convert/events_db/schema.h"
#include "xprof/convert/events_db/tpu_component.h"
#include "xprof/convert/executor.h"
#include "xprof/convert/executor_factory.h"
#include "xprof/utils/hlo_module_map.h"

namespace xprof::events_db {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::Eq;
using ::testing::IsEmpty;
using ::testing::Optional;

constexpr absl::string_view kHloText = R"(
  HloModule test_module
  ENTRY main {
    arg0 = f32[2,2] parameter(0)
    ROOT negate = f32[2,2] negate(arg0)
  })";

TEST(XSpaceParserTest, ParsesMultiPlaneXSpaceWithTpuAndHloProtos) {
  tensorflow::profiler::XSpace xspace;

  // 1. Host Plane
  tensorflow::profiler::XPlane* host_plane = xspace.add_planes();
  host_plane->set_name(tsl::profiler::kHostThreadsPlaneName);
  tsl::profiler::XPlaneBuilder host_builder(host_plane);
  tsl::profiler::XLineBuilder host_line = host_builder.GetOrCreateLine(1);
  host_line.SetName("MainThread");
  tsl::profiler::XEventBuilder host_event = host_line.AddEvent(
      *host_builder.GetOrCreateEventMetadata("model/train_step"));
  host_event.SetTimestampNs(100);
  host_event.SetDurationNs(900);

  // 2. TPU Plane
  tensorflow::profiler::XPlane* tpu_plane = xspace.add_planes();
  tpu_plane->set_name("/device:TPU:0");
  tsl::profiler::XPlaneBuilder tpu_builder(tpu_plane);
  tsl::profiler::XLineBuilder tpu_line =
      tpu_builder.GetOrCreateLine(internal::TpuComponent::kTensorCore);
  tpu_line.SetName("Tensor Core");
  tsl::profiler::XEventBuilder tpu_event =
      tpu_line.AddEvent(*tpu_builder.GetOrCreateEventMetadata("fused_mha"));
  tpu_event.SetTimestampNs(200);
  tpu_event.SetDurationNs(500);

  // 3. Custom Plane
  tensorflow::profiler::XPlane* custom_plane = xspace.add_planes();
  custom_plane->set_name("/device:CUSTOM:Megascale Trace");
  tsl::profiler::XPlaneBuilder custom_builder(custom_plane);
  tsl::profiler::XLineBuilder custom_line = custom_builder.GetOrCreateLine(1);
  custom_line.SetName("DCN Ring");
  tsl::profiler::XEventBuilder custom_event = custom_line.AddEvent(
      *custom_builder.GetOrCreateEventMetadata("SendRecv"));
  custom_event.SetTimestampNs(300);
  custom_event.SetDurationNs(200);

  // 4. Metadata Plane containing HLO Proto
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> hlo_module,
                       xla::ParseAndReturnUnverifiedModule(kHloText));
  xla::HloProto hlo_proto;
  *hlo_proto.mutable_hlo_module() = hlo_module->ToProto();

  tensorflow::profiler::XPlane* metadata_plane = xspace.add_planes();
  metadata_plane->set_name(tsl::profiler::kMetadataPlaneName);
  tsl::profiler::XPlaneBuilder metadata_builder(metadata_plane);
  metadata_builder.GetOrCreateStatMetadata(
      tsl::profiler::GetStatTypeStr(tsl::profiler::StatType::kHloProto));
  metadata_builder.GetOrCreateStatMetadata(
      tsl::profiler::GetStatTypeStr(tsl::profiler::StatType::kProgramId));
  tensorflow::profiler::XEventMetadata* hlo_event_meta =
      metadata_builder.GetOrCreateEventMetadata("test_module");
  hlo_event_meta->set_id(12345);
  tensorflow::profiler::XStat* stat = hlo_event_meta->add_stats();
  stat->set_metadata_id(
      metadata_builder
          .GetOrCreateStatMetadata(
              tsl::profiler::GetStatTypeStr(tsl::profiler::StatType::kHloProto))
          ->id());
  stat->set_bytes_value(hlo_proto.SerializeAsString());

  // 5. Unclassified plane to exercise classification fallthrough
  tensorflow::profiler::XPlane* unclassified_plane = xspace.add_planes();
  unclassified_plane->set_name("/other:metadata");

  Schema schema;
  internal::FieldIndices indices(schema);

  std::vector<Record> parsed_records;
  const absl::StatusOr<ParseStatus> status_or = ParseXSpace(
      xspace, {}, schema,
      [&](Record& record) -> absl::StatusOr<StepControl> {
        parsed_records.push_back(record);
        return StepControl::kContinue;
      },
      /*hlo_module_map=*/std::nullopt,
      tensorflow::profiler::InlineExecutorFactory);

  EXPECT_THAT(status_or, IsOkAndHolds(Eq(ParseStatus::kComplete)));
  ASSERT_EQ(parsed_records.size(), 3);

  EXPECT_EQ(parsed_records[0][indices.device], "cpu:0");
  EXPECT_EQ(parsed_records[0][indices.kernel_name], "model/train_step");

  EXPECT_EQ(parsed_records[1][indices.device],
            "/device:CUSTOM:Megascale Trace");
  EXPECT_EQ(parsed_records[1][indices.kernel_name], "SendRecv");

  EXPECT_EQ(parsed_records[2][indices.device], "TPU:0");
  EXPECT_EQ(parsed_records[2][indices.kernel_name], "fused_mha");
}

TEST(XSpaceParserTest, ParsesGpuXSpace) {
  tensorflow::profiler::XSpace xspace;

  // 1. Host Plane
  tensorflow::profiler::XPlane* host_plane = xspace.add_planes();
  host_plane->set_name(tsl::profiler::kHostThreadsPlaneName);
  tsl::profiler::XPlaneBuilder host_builder(host_plane);
  tsl::profiler::XLineBuilder host_line = host_builder.GetOrCreateLine(1);
  host_line.SetName("MainThread");
  tsl::profiler::XEventBuilder host_event =
      host_line.AddEvent(*host_builder.GetOrCreateEventMetadata("host_op"));
  host_event.SetTimestampNs(100);
  host_event.SetDurationNs(900);

  // 2. GPU Plane
  tensorflow::profiler::XPlane* gpu_plane = xspace.add_planes();
  gpu_plane->set_name("/device:GPU:0");
  tsl::profiler::XPlaneBuilder gpu_builder(gpu_plane);
  tsl::profiler::XLineBuilder gpu_line = gpu_builder.GetOrCreateLine(1);
  gpu_line.SetName("Stream: 7");
  tsl::profiler::XEventBuilder gpu_event =
      gpu_line.AddEvent(*gpu_builder.GetOrCreateEventMetadata("matmul_kernel"));
  gpu_event.SetTimestampNs(200);
  gpu_event.SetDurationNs(500);

  Schema schema;
  internal::FieldIndices indices(schema);

  std::vector<Record> parsed_records;
  const absl::StatusOr<ParseStatus> status_or = ParseXSpace(
      xspace, {}, schema,
      [&](Record& record) -> absl::StatusOr<StepControl> {
        parsed_records.push_back(record);
        return StepControl::kContinue;
      },
      /*hlo_module_map=*/std::nullopt,
      tensorflow::profiler::InlineExecutorFactory);

  EXPECT_THAT(status_or, IsOkAndHolds(Eq(ParseStatus::kComplete)));
  ASSERT_EQ(parsed_records.size(), 2);

  EXPECT_EQ(parsed_records[0][indices.device], "cpu:0");
  EXPECT_EQ(parsed_records[0][indices.kernel_name], "host_op");

  EXPECT_EQ(parsed_records[1][indices.device], "gpu:0");
  EXPECT_EQ(parsed_records[1][indices.kernel_name], "matmul_kernel");
}

TEST(XSpaceParserTest, HandlesCpuOnlyXSpace) {
  tensorflow::profiler::XSpace xspace;

  tensorflow::profiler::XPlane* host_plane = xspace.add_planes();
  host_plane->set_name(tsl::profiler::kHostThreadsPlaneName);
  tsl::profiler::XPlaneBuilder host_builder(host_plane);
  tsl::profiler::XLineBuilder host_line = host_builder.GetOrCreateLine(1);
  host_line.SetName("MainThread");
  tsl::profiler::XEventBuilder host_event =
      host_line.AddEvent(*host_builder.GetOrCreateEventMetadata("cpu_event"));
  host_event.SetTimestampNs(100);
  host_event.SetDurationNs(500);

  Schema schema;
  internal::FieldIndices indices(schema);

  std::vector<Record> parsed_records;
  const absl::StatusOr<ParseStatus> status_or = ParseXSpace(
      xspace, {}, schema,
      [&](Record& record) -> absl::StatusOr<StepControl> {
        parsed_records.push_back(record);
        return StepControl::kContinue;
      },
      /*hlo_module_map=*/std::nullopt,
      tensorflow::profiler::InlineExecutorFactory);

  EXPECT_THAT(status_or, IsOkAndHolds(Eq(ParseStatus::kComplete)));
  ASSERT_EQ(parsed_records.size(), 1);
  EXPECT_EQ(parsed_records[0][indices.kernel_name], "cpu_event");
}

TEST(XSpaceParserTest, SupportsEarlyStoppingOnHostPlane) {
  tensorflow::profiler::XSpace xspace;

  tensorflow::profiler::XPlane* host_plane = xspace.add_planes();
  host_plane->set_name(tsl::profiler::kHostThreadsPlaneName);
  tsl::profiler::XPlaneBuilder host_builder(host_plane);
  tsl::profiler::XLineBuilder host_line = host_builder.GetOrCreateLine(1);
  host_line.SetName("MainThread");

  for (int64_t i = 0; i < 10; ++i) {
    tsl::profiler::XEventBuilder ev =
        host_line.AddEvent(*host_builder.GetOrCreateEventMetadata("event"));
    ev.SetTimestampNs(i * 100);
    ev.SetDurationNs(50);
  }

  Schema schema;

  int64_t received_count = 0;
  const absl::StatusOr<ParseStatus> status_or = ParseXSpace(
      xspace, {}, schema,
      [&](Record&) -> absl::StatusOr<StepControl> {
        ++received_count;
        return received_count < 3 ? StepControl::kContinue : StepControl::kStop;
      },
      /*hlo_module_map=*/std::nullopt,
      tensorflow::profiler::InlineExecutorFactory);

  EXPECT_THAT(status_or, IsOkAndHolds(Eq(ParseStatus::kStoppedEarly)));
  EXPECT_EQ(received_count, 3);
}

TEST(XSpaceParserTest, SupportsEarlyStoppingOnCustomPlane) {
  tensorflow::profiler::XSpace xspace;

  tensorflow::profiler::XPlane* custom_plane = xspace.add_planes();
  custom_plane->set_name("/device:CUSTOM:Test");
  tsl::profiler::XPlaneBuilder custom_builder(custom_plane);
  tsl::profiler::XLineBuilder custom_line = custom_builder.GetOrCreateLine(1);
  custom_line.SetName("Stream");

  for (int64_t i = 0; i < 5; ++i) {
    tsl::profiler::XEventBuilder ev =
        custom_line.AddEvent(*custom_builder.GetOrCreateEventMetadata("event"));
    ev.SetTimestampNs(i * 100);
    ev.SetDurationNs(50);
  }

  Schema schema;

  int64_t received_count = 0;
  const absl::StatusOr<ParseStatus> status_or = ParseXSpace(
      xspace, {}, schema,
      [&](Record&) -> absl::StatusOr<StepControl> {
        ++received_count;
        return received_count < 2 ? StepControl::kContinue : StepControl::kStop;
      },
      /*hlo_module_map=*/std::nullopt,
      tensorflow::profiler::InlineExecutorFactory);

  EXPECT_THAT(status_or, IsOkAndHolds(Eq(ParseStatus::kStoppedEarly)));
  EXPECT_EQ(received_count, 2);
}

TEST(XSpaceParserTest, SupportsEarlyStoppingOnDevicePlane) {
  tensorflow::profiler::XSpace xspace;

  tensorflow::profiler::XPlane* tpu_plane = xspace.add_planes();
  tpu_plane->set_name("/device:TPU:0");
  tsl::profiler::XPlaneBuilder tpu_builder(tpu_plane);
  tsl::profiler::XLineBuilder tpu_line = tpu_builder.GetOrCreateLine(1);
  tpu_line.SetName("Tensor Core");

  for (int64_t i = 0; i < 5; ++i) {
    tsl::profiler::XEventBuilder ev =
        tpu_line.AddEvent(*tpu_builder.GetOrCreateEventMetadata("tpu_event"));
    ev.SetTimestampNs(i * 100);
    ev.SetDurationNs(50);
  }

  Schema schema;

  int64_t received_count = 0;
  const absl::StatusOr<ParseStatus> status_or = ParseXSpace(
      xspace, {}, schema,
      [&](Record&) -> absl::StatusOr<StepControl> {
        ++received_count;
        return received_count < 2 ? StepControl::kContinue : StepControl::kStop;
      },
      /*hlo_module_map=*/std::nullopt,
      tensorflow::profiler::InlineExecutorFactory);

  EXPECT_THAT(status_or, IsOkAndHolds(Eq(ParseStatus::kStoppedEarly)));
  EXPECT_EQ(received_count, 2);
}

TEST(XSpaceParserTest, PropagatesHostError) {
  tensorflow::profiler::XSpace xspace;

  tensorflow::profiler::XPlane* host_plane = xspace.add_planes();
  host_plane->set_name(tsl::profiler::kHostThreadsPlaneName);
  tsl::profiler::XPlaneBuilder host_builder(host_plane);
  tsl::profiler::XLineBuilder host_line = host_builder.GetOrCreateLine(1);
  tsl::profiler::XEventBuilder ev =
      host_line.AddEvent(*host_builder.GetOrCreateEventMetadata("event"));
  ev.SetTimestampNs(100);
  ev.SetDurationNs(50);

  Schema schema;

  const absl::StatusOr<ParseStatus> status_or = ParseXSpace(
      xspace, {}, schema,
      [&](Record&) -> absl::StatusOr<StepControl> {
        return absl::InternalError("host disk write failure");
      },
      /*hlo_module_map=*/std::nullopt,
      tensorflow::profiler::InlineExecutorFactory);

  EXPECT_THAT(status_or,
              StatusIs(absl::StatusCode::kInternal, "host disk write failure"));
}

TEST(XSpaceParserTest, PropagatesCustomError) {
  tensorflow::profiler::XSpace xspace;

  tensorflow::profiler::XPlane* custom_plane = xspace.add_planes();
  custom_plane->set_name("/device:CUSTOM:Test");
  tsl::profiler::XPlaneBuilder custom_builder(custom_plane);
  tsl::profiler::XLineBuilder custom_line = custom_builder.GetOrCreateLine(1);
  tsl::profiler::XEventBuilder ev =
      custom_line.AddEvent(*custom_builder.GetOrCreateEventMetadata("event"));
  ev.SetTimestampNs(100);
  ev.SetDurationNs(50);

  Schema schema;

  const absl::StatusOr<ParseStatus> status_or = ParseXSpace(
      xspace, {}, schema,
      [&](Record&) -> absl::StatusOr<StepControl> {
        return absl::InternalError("custom disk write failure");
      },
      /*hlo_module_map=*/std::nullopt,
      tensorflow::profiler::InlineExecutorFactory);

  EXPECT_THAT(status_or, StatusIs(absl::StatusCode::kInternal,
                                  "custom disk write failure"));
}

TEST(XSpaceParserTest, PropagatesDeviceError) {
  tensorflow::profiler::XSpace xspace;

  tensorflow::profiler::XPlane* tpu_plane = xspace.add_planes();
  tpu_plane->set_name("/device:TPU:0");
  tsl::profiler::XPlaneBuilder tpu_builder(tpu_plane);
  tsl::profiler::XLineBuilder tpu_line = tpu_builder.GetOrCreateLine(1);
  tpu_line.SetName("Tensor Core");
  tsl::profiler::XEventBuilder ev =
      tpu_line.AddEvent(*tpu_builder.GetOrCreateEventMetadata("tpu_event"));
  ev.SetTimestampNs(100);
  ev.SetDurationNs(50);

  Schema schema;

  const absl::StatusOr<ParseStatus> status_or = ParseXSpace(
      xspace, {}, schema,
      [&](Record&) -> absl::StatusOr<StepControl> {
        return absl::InternalError("device disk write failure");
      },
      /*hlo_module_map=*/std::nullopt,
      tensorflow::profiler::InlineExecutorFactory);

  EXPECT_THAT(status_or, StatusIs(absl::StatusCode::kInternal,
                                  "device disk write failure"));
}

TEST(XSpaceParserTest, ParsesMultiPlaneXSpaceWithParallelExecutor) {
  tensorflow::profiler::XSpace xspace;

  // 1. Host Plane
  tensorflow::profiler::XPlane* host_plane = xspace.add_planes();
  host_plane->set_name(tsl::profiler::kHostThreadsPlaneName);
  tsl::profiler::XPlaneBuilder host_builder(host_plane);
  tsl::profiler::XLineBuilder host_line = host_builder.GetOrCreateLine(1);
  host_line.SetName("MainThread");
  tsl::profiler::XEventBuilder host_event = host_line.AddEvent(
      *host_builder.GetOrCreateEventMetadata("model/train_step"));
  host_event.SetTimestampNs(100);
  host_event.SetDurationNs(900);

  // 2. Multiple TPU Planes
  for (int i = 0; i < 4; ++i) {
    tensorflow::profiler::XPlane* tpu_plane = xspace.add_planes();
    tpu_plane->set_name(absl::StrCat("/device:TPU:", i));
    tsl::profiler::XPlaneBuilder tpu_builder(tpu_plane);
    tsl::profiler::XLineBuilder tpu_line = tpu_builder.GetOrCreateLine(1);
    tpu_line.SetName("Tensor Core");
    tsl::profiler::XEventBuilder tpu_event = tpu_line.AddEvent(
        *tpu_builder.GetOrCreateEventMetadata(absl::StrCat("op_", i)));
    tpu_event.SetTimestampNs(200 + i * 10);
    tpu_event.SetDurationNs(500);
  }

  // 3. Custom Plane
  tensorflow::profiler::XPlane* custom_plane = xspace.add_planes();
  custom_plane->set_name("/device:CUSTOM:Megascale Trace");
  tsl::profiler::XPlaneBuilder custom_builder(custom_plane);
  tsl::profiler::XLineBuilder custom_line = custom_builder.GetOrCreateLine(1);
  custom_line.SetName("DCN Ring");
  tsl::profiler::XEventBuilder custom_event = custom_line.AddEvent(
      *custom_builder.GetOrCreateEventMetadata("SendRecv"));
  custom_event.SetTimestampNs(300);
  custom_event.SetDurationNs(200);

  Schema schema;

  absl::Mutex mu;
  std::vector<Record> received_records;

  const absl::StatusOr<ParseStatus> status_or = ParseXSpace(
      xspace, {}, schema,
      [&](Record& record) -> absl::StatusOr<StepControl> {
        absl::MutexLock lock(mu);
        received_records.push_back(record);
        return StepControl::kContinue;
      },
      /*hlo_module_map=*/std::nullopt,
      [] {
        return tensorflow::profiler::CreateXprofThreadPoolExecutor(
            "test_executor", 4);
      });

  EXPECT_THAT(status_or, IsOkAndHolds(Eq(ParseStatus::kComplete)));
  // 1 host + 4 TPU + 1 custom = 6 records total
  EXPECT_EQ(received_records.size(), 6);
}

TEST(XSpaceParserTest,
     ParsesMultiPlaneXSpaceWithInlineExecutorFactorySingleThreaded) {
  tensorflow::profiler::XSpace xspace;

  // 1. Host Plane
  tensorflow::profiler::XPlane* host_plane = xspace.add_planes();
  host_plane->set_name(tsl::profiler::kHostThreadsPlaneName);
  tsl::profiler::XPlaneBuilder host_builder(host_plane);
  tsl::profiler::XLineBuilder host_line = host_builder.GetOrCreateLine(1);
  host_line.SetName("MainThread");
  tsl::profiler::XEventBuilder host_event = host_line.AddEvent(
      *host_builder.GetOrCreateEventMetadata("model/train_step"));
  host_event.SetTimestampNs(100);
  host_event.SetDurationNs(900);

  // 2. Multiple TPU Planes
  for (int i = 0; i < 4; ++i) {
    tensorflow::profiler::XPlane* tpu_plane = xspace.add_planes();
    tpu_plane->set_name(absl::StrCat("/device:TPU:", i));
    tsl::profiler::XPlaneBuilder tpu_builder(tpu_plane);
    tsl::profiler::XLineBuilder tpu_line = tpu_builder.GetOrCreateLine(1);
    tpu_line.SetName("Tensor Core");
    tsl::profiler::XEventBuilder tpu_event = tpu_line.AddEvent(
        *tpu_builder.GetOrCreateEventMetadata(absl::StrCat("op_", i)));
    tpu_event.SetTimestampNs(200 + i * 10);
    tpu_event.SetDurationNs(500);
  }

  // 3. Custom Plane
  tensorflow::profiler::XPlane* custom_plane = xspace.add_planes();
  custom_plane->set_name("/device:CUSTOM:Megascale Trace");
  tsl::profiler::XPlaneBuilder custom_builder(custom_plane);
  tsl::profiler::XLineBuilder custom_line = custom_builder.GetOrCreateLine(1);
  custom_line.SetName("DCN Ring");
  tsl::profiler::XEventBuilder custom_event = custom_line.AddEvent(
      *custom_builder.GetOrCreateEventMetadata("SendRecv"));
  custom_event.SetTimestampNs(300);
  custom_event.SetDurationNs(200);

  Schema schema;
  std::vector<Record> received_records;

  const absl::StatusOr<ParseStatus> status_or = ParseXSpace(
      xspace, {}, schema,
      [&](Record& record) -> absl::StatusOr<StepControl> {
        received_records.push_back(record);
        return StepControl::kContinue;
      },
      /*hlo_module_map=*/std::nullopt,
      tensorflow::profiler::InlineExecutorFactory);

  EXPECT_THAT(status_or, IsOkAndHolds(Eq(ParseStatus::kComplete)));
  EXPECT_EQ(received_records.size(), 6);
}

TEST(XSpaceParserTest, ParallelExecutorHandlesEarlyStopping) {
  tensorflow::profiler::XSpace xspace;

  for (int i = 0; i < 8; ++i) {
    tensorflow::profiler::XPlane* tpu_plane = xspace.add_planes();
    tpu_plane->set_name(absl::StrCat("/device:TPU:", i));
    tsl::profiler::XPlaneBuilder tpu_builder(tpu_plane);
    tsl::profiler::XLineBuilder tpu_line = tpu_builder.GetOrCreateLine(1);
    tpu_line.SetName("Tensor Core");
    tsl::profiler::XEventBuilder tpu_event = tpu_line.AddEvent(
        *tpu_builder.GetOrCreateEventMetadata(absl::StrCat("op_", i)));
    tpu_event.SetTimestampNs(200 + i * 10);
    tpu_event.SetDurationNs(500);
  }

  Schema schema;

  std::atomic<int> received_count = 0;
  const absl::StatusOr<ParseStatus> status_or = ParseXSpace(
      xspace, {}, schema,
      [&](Record&) -> absl::StatusOr<StepControl> {
        int prev = received_count.fetch_add(1);
        return prev < 2 ? StepControl::kContinue : StepControl::kStop;
      },
      /*hlo_module_map=*/std::nullopt,
      [] {
        return tensorflow::profiler::CreateXprofThreadPoolExecutor(
            "test_stop_executor", 4);
      });

  EXPECT_THAT(status_or, IsOkAndHolds(Eq(ParseStatus::kStoppedEarly)));
}

TEST(XSpaceParserTest, InlineExecutorFactoryHandlesEarlyStopping) {
  tensorflow::profiler::XSpace xspace;

  for (int i = 0; i < 8; ++i) {
    tensorflow::profiler::XPlane* tpu_plane = xspace.add_planes();
    tpu_plane->set_name(absl::StrCat("/device:TPU:", i));
    tsl::profiler::XPlaneBuilder tpu_builder(tpu_plane);
    tsl::profiler::XLineBuilder tpu_line = tpu_builder.GetOrCreateLine(1);
    tpu_line.SetName("Tensor Core");
    tsl::profiler::XEventBuilder tpu_event = tpu_line.AddEvent(
        *tpu_builder.GetOrCreateEventMetadata(absl::StrCat("op_", i)));
    tpu_event.SetTimestampNs(200 + i * 10);
    tpu_event.SetDurationNs(500);
  }

  Schema schema;

  int received_count = 0;
  const absl::StatusOr<ParseStatus> status_or = ParseXSpace(
      xspace, {}, schema,
      [&](Record&) -> absl::StatusOr<StepControl> {
        return ++received_count < 2 ? StepControl::kContinue
                                    : StepControl::kStop;
      },
      /*hlo_module_map=*/std::nullopt,
      tensorflow::profiler::InlineExecutorFactory);

  EXPECT_THAT(status_or, IsOkAndHolds(Eq(ParseStatus::kStoppedEarly)));
}

TEST(XSpaceParserTest, DefaultExecutorFactoryCreatesValidExecutor) {
  std::unique_ptr<tensorflow::profiler::Executor> executor =
      tensorflow::profiler::DefaultExecutorFactory();
  ASSERT_NE(executor, nullptr);
  bool executed = false;
  executor->Execute([&] { executed = true; });
  executor->JoinAll();
  EXPECT_TRUE(executed);
}

TEST(XSpaceParserTest, ParsesMixedTpuAndGpuPlanesInSameXSpace) {
  tensorflow::profiler::XSpace xspace;

  // 1. TPU Plane
  tensorflow::profiler::XPlane* tpu_plane = xspace.add_planes();
  tpu_plane->set_name("/device:TPU:0");
  tsl::profiler::XPlaneBuilder tpu_builder(tpu_plane);
  tsl::profiler::XLineBuilder tpu_line =
      tpu_builder.GetOrCreateLine(internal::TpuComponent::kTensorCore);
  tpu_line.SetName("Tensor Core");
  tsl::profiler::XEventBuilder tpu_event =
      tpu_line.AddEvent(*tpu_builder.GetOrCreateEventMetadata("tpu_kernel"));
  tpu_event.SetTimestampNs(100);
  tpu_event.SetDurationNs(200);

  // 2. GPU Plane
  tensorflow::profiler::XPlane* gpu_plane = xspace.add_planes();
  gpu_plane->set_name("/device:GPU:0");
  tsl::profiler::XPlaneBuilder gpu_builder(gpu_plane);
  tsl::profiler::XLineBuilder gpu_line = gpu_builder.GetOrCreateLine(1);
  gpu_line.SetName("Stream:7");
  tsl::profiler::XEventBuilder gpu_event =
      gpu_line.AddEvent(*gpu_builder.GetOrCreateEventMetadata("gpu_kernel"));
  gpu_event.SetTimestampNs(300);
  gpu_event.SetDurationNs(400);

  Schema schema;
  const internal::FieldIndices indices(schema);

  std::vector<Record> records;
  const absl::StatusOr<ParseStatus> status_or = ParseXSpace(
      xspace, {}, schema,
      [&](Record& record) -> absl::StatusOr<StepControl> {
        records.push_back(record);
        return StepControl::kContinue;
      },
      /*hlo_module_map=*/std::nullopt,
      tensorflow::profiler::InlineExecutorFactory);

  EXPECT_THAT(status_or, IsOkAndHolds(Eq(ParseStatus::kComplete)));
  ASSERT_EQ(records.size(), 2);
  EXPECT_EQ(records[0][indices.kernel_name], "tpu_kernel");
  EXPECT_EQ(records[1][indices.kernel_name], "gpu_kernel");
}

TEST(XSpaceParserTest, ResolvesStepFromGroupMetadataMap) {
  tensorflow::profiler::XSpace xspace;

  tensorflow::profiler::XPlane* host_plane = xspace.add_planes();
  host_plane->set_name(tsl::profiler::kHostThreadsPlaneName);
  tsl::profiler::XPlaneBuilder host_builder(host_plane);
  tsl::profiler::XLineBuilder host_line = host_builder.GetOrCreateLine(1);
  host_line.SetName("MainThread");
  tsl::profiler::XEventBuilder host_event =
      host_line.AddEvent(*host_builder.GetOrCreateEventMetadata("step_op"));
  host_event.SetTimestampNs(100);
  host_event.SetDurationNs(200);
  host_event.AddStatValue(
      *host_builder.GetOrCreateStatMetadata(
          tsl::profiler::GetStatTypeStr(tsl::profiler::StatType::kGroupId)),
      42);

  tsl::profiler::GroupMetadataMap group_metadata_map{
      {42, tsl::profiler::GroupMetadata{.name = "train_step 42"}}};

  Schema schema;
  const internal::FieldIndices indices(schema);

  std::vector<Record> records;
  const absl::StatusOr<ParseStatus> status_or = ParseXSpace(
      xspace, group_metadata_map, schema,
      [&](Record& record) -> absl::StatusOr<StepControl> {
        records.push_back(record);
        return StepControl::kContinue;
      },
      /*hlo_module_map=*/std::nullopt,
      tensorflow::profiler::InlineExecutorFactory);

  EXPECT_THAT(status_or, IsOkAndHolds(Eq(ParseStatus::kComplete)));
  ASSERT_EQ(records.size(), 1);
  EXPECT_EQ(records[0][indices.step], "train_step 42");
}

TEST(XSpaceParserTest, UsesPrecomputedHloModuleMap) {
  tensorflow::profiler::XSpace xspace;

  tensorflow::profiler::XPlane* gpu_plane = xspace.add_planes();
  gpu_plane->set_name("/device:GPU:0");
  tsl::profiler::XPlaneBuilder gpu_builder(gpu_plane);
  tsl::profiler::XLineBuilder gpu_line = gpu_builder.GetOrCreateLine(1);
  gpu_line.SetName("Stream:7");
  tsl::profiler::XEventBuilder gpu_event =
      gpu_line.AddEvent(*gpu_builder.GetOrCreateEventMetadata("custom_kernel"));
  gpu_event.SetTimestampNs(100);
  gpu_event.SetDurationNs(200);
  gpu_event.AddStatValue(
      *gpu_builder.GetOrCreateStatMetadata(
          tsl::profiler::GetStatTypeStr(tsl::profiler::StatType::kHloOp)),
      "fusion_1");
  gpu_event.AddStatValue(
      *gpu_builder.GetOrCreateStatMetadata(
          tsl::profiler::GetStatTypeStr(tsl::profiler::StatType::kHloModule)),
      "test_module");
  gpu_event.AddStatValue(
      *gpu_builder.GetOrCreateStatMetadata(
          tsl::profiler::GetStatTypeStr(tsl::profiler::StatType::kProgramId)),
      static_cast<uint64_t>(12345));

  absl::string_view kModuleText = R"(
    HloModule test_module

    fused_comp {
      p0 = f32[64,64]{1,0} parameter(0)
      ROOT res = f32[64,128]{1,0} custom-call(p0), custom_call_target="dense"
    }

    ENTRY main {
      param0 = f32[64,64]{1,0} parameter(0)
      ROOT fusion_1 = f32[64,128]{1,0} fusion(param0), kind=kCustom, calls=fused_comp, metadata={op_name="model/layer/Dense" op_type="Dense"}
    }
  )";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> hlo_module,
                       xla::ParseAndReturnUnverifiedModule(kModuleText));
  xla::HloProto hlo_proto;
  *hlo_proto.mutable_hlo_module() = hlo_module->ToProto();

  tensorflow::profiler::HloModuleMap hlo_module_map;
  tensorflow::profiler::AddHloProto(hlo_module_map, 12345, hlo_proto,
                                    /*cost_analysis=*/nullptr);

  Schema schema;
  const internal::FieldIndices indices(schema);

  std::vector<Record> records;
  const absl::StatusOr<ParseStatus> status_or = ParseXSpace(
      xspace, {}, schema,
      [&](Record& record) -> absl::StatusOr<StepControl> {
        records.push_back(record);
        return StepControl::kContinue;
      },
      hlo_module_map, tensorflow::profiler::InlineExecutorFactory);

  EXPECT_THAT(status_or, IsOkAndHolds(Eq(ParseStatus::kComplete)));
  ASSERT_EQ(records.size(), 1);
  EXPECT_EQ(records[0][indices.kernel_name], "custom_kernel");
  EXPECT_EQ(records[0][indices.hlo_module], "test_module");
  EXPECT_EQ(records[0][indices.hlo_op], "fusion_1");
  EXPECT_EQ(records[0][indices.tf_op_name], "model/layer/Dense");
}

TEST(XSpaceParserTest, AutoGroupsEventsWhenGroupMetadataMapOmitted) {
  tensorflow::profiler::XSpace xspace;

  tensorflow::profiler::XPlane* host_plane = xspace.add_planes();
  host_plane->set_name(tsl::profiler::kHostThreadsPlaneName);
  tsl::profiler::XPlaneBuilder host_builder(host_plane);
  tsl::profiler::XLineBuilder host_line = host_builder.GetOrCreateLine(1);
  host_line.SetName("MainThread");
  tsl::profiler::XEventBuilder host_event = host_line.AddEvent(
      *host_builder.GetOrCreateEventMetadata(tsl::profiler::GetHostEventTypeStr(
          tsl::profiler::HostEventType::kTraceContext)));
  host_event.SetTimestampNs(100);
  host_event.SetDurationNs(200);
  host_event.AddStatValue(
      *host_builder.GetOrCreateStatMetadata(
          tsl::profiler::GetStatTypeStr(tsl::profiler::StatType::kGraphType)),
      "train");
  host_event.AddStatValue(
      *host_builder.GetOrCreateStatMetadata(
          tsl::profiler::GetStatTypeStr(tsl::profiler::StatType::kStepNum)),
      123);

  Schema schema;
  const internal::FieldIndices indices(schema);

  std::vector<Record> records;
  const absl::StatusOr<ParseStatus> status_or = ParseXSpace(
      xspace, schema,
      [&](Record& record) -> absl::StatusOr<StepControl> {
        records.push_back(record);
        return StepControl::kContinue;
      },
      /*hlo_module_map=*/std::nullopt,
      /*group_metadata_map=*/std::nullopt,
      tensorflow::profiler::InlineExecutorFactory);

  EXPECT_THAT(status_or, IsOkAndHolds(Eq(ParseStatus::kComplete)));
  ASSERT_EQ(records.size(), 1);
  EXPECT_EQ(records[0][indices.kernel_name],
            tsl::profiler::GetHostEventTypeStr(
                tsl::profiler::HostEventType::kTraceContext));
  EXPECT_EQ(records[0][indices.step], "train 123");
}

TEST(XSpaceParserTest, HandlesEmptyXSpaceGracefully) {
  tensorflow::profiler::XSpace xspace;
  Schema schema;

  std::vector<Record> records;
  const absl::StatusOr<ParseStatus> status_or = ParseXSpace(
      xspace, {}, schema,
      [&](Record& record) -> absl::StatusOr<StepControl> {
        records.push_back(record);
        return StepControl::kContinue;
      },
      /*hlo_module_map=*/std::nullopt,
      tensorflow::profiler::InlineExecutorFactory);

  EXPECT_THAT(status_or, IsOkAndHolds(Eq(ParseStatus::kComplete)));
  EXPECT_THAT(records, IsEmpty());
}

TEST(XSpaceParserTest, ReturnsErrorWhenExecutorFactoryReturnsNull) {
  tensorflow::profiler::XSpace xspace;
  Schema schema;

  const absl::StatusOr<ParseStatus> status_or = ParseXSpace(
      xspace, {}, schema,
      [](Record&) -> absl::StatusOr<StepControl> {
        return StepControl::kContinue;
      },
      /*hlo_module_map=*/std::nullopt,
      []() -> std::unique_ptr<tensorflow::profiler::Executor> {
        return nullptr;
      });

  EXPECT_THAT(status_or.status(), StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(XSpaceParserTest, CallsFinalizeOnStatefulConsumer) {
  tensorflow::profiler::XSpace xspace;
  Schema schema;

  struct StatefulConsumer {
    int records_seen = 0;
    bool finalized = false;
    std::optional<ParseStatus> parse_outcome;

    absl::StatusOr<StepControl> Consume(Record&) {
      records_seen++;
      return StepControl::kContinue;
    }

    absl::Status Finalize(const absl::StatusOr<ParseStatus>& result) {
      finalized = true;
      if (result.ok()) {
        parse_outcome = *result;
      }
      return absl::OkStatus();
    }
  };

  StatefulConsumer consumer;
  const absl::StatusOr<ParseStatus> status_or =
      ParseXSpace(xspace, {}, schema, consumer,
                  /*hlo_module_map=*/std::nullopt,
                  tensorflow::profiler::InlineExecutorFactory);

  EXPECT_THAT(status_or, IsOkAndHolds(Eq(ParseStatus::kComplete)));
  EXPECT_TRUE(consumer.finalized);
  EXPECT_THAT(consumer.parse_outcome, Optional(Eq(ParseStatus::kComplete)));
}

TEST(XSpaceParserTest, FinalizeReceivesStoppedEarlyResult) {
  tensorflow::profiler::XSpace xspace;
  tensorflow::profiler::XPlane* host_plane = xspace.add_planes();
  host_plane->set_name(tsl::profiler::kHostThreadsPlaneName);
  tsl::profiler::XPlaneBuilder host_builder(host_plane);
  tsl::profiler::XLineBuilder host_line = host_builder.GetOrCreateLine(1);
  for (int i = 0; i < 3; ++i) {
    tsl::profiler::XEventBuilder ev =
        host_line.AddEvent(*host_builder.GetOrCreateEventMetadata("event"));
    ev.SetTimestampNs(100 * (i + 1));
    ev.SetDurationNs(50);
  }

  Schema schema;

  struct EarlyStoppingConsumer {
    int records_seen = 0;
    bool finalized = false;
    std::optional<ParseStatus> parse_outcome;

    absl::StatusOr<StepControl> Consume(Record&) {
      records_seen++;
      return StepControl::kStop;
    }

    absl::Status Finalize(const absl::StatusOr<ParseStatus>& result) {
      finalized = true;
      if (result.ok()) {
        parse_outcome = *result;
      }
      return absl::OkStatus();
    }
  };

  EarlyStoppingConsumer consumer;
  const absl::StatusOr<ParseStatus> status_or =
      ParseXSpace(xspace, {}, schema, consumer,
                  /*hlo_module_map=*/std::nullopt,
                  tensorflow::profiler::InlineExecutorFactory);

  EXPECT_THAT(status_or, IsOkAndHolds(Eq(ParseStatus::kStoppedEarly)));
  EXPECT_TRUE(consumer.finalized);
  EXPECT_THAT(consumer.parse_outcome, Optional(Eq(ParseStatus::kStoppedEarly)));
}

TEST(XSpaceParserTest, FinalizeReceivesErrorResultOnFailure) {
  tensorflow::profiler::XSpace xspace;
  tensorflow::profiler::XPlane* host_plane = xspace.add_planes();
  host_plane->set_name(tsl::profiler::kHostThreadsPlaneName);
  tsl::profiler::XPlaneBuilder host_builder(host_plane);
  tsl::profiler::XLineBuilder host_line = host_builder.GetOrCreateLine(1);
  tsl::profiler::XEventBuilder ev =
      host_line.AddEvent(*host_builder.GetOrCreateEventMetadata("event"));
  ev.SetTimestampNs(100);
  ev.SetDurationNs(50);

  Schema schema;

  struct FailingConsumerWithFinalize {
    bool finalized = false;
    absl::Status received_error;

    absl::StatusOr<StepControl> Consume(Record&) {
      return absl::InternalError("worker failed");
    }

    absl::Status Finalize(const absl::StatusOr<ParseStatus>& result) {
      finalized = true;
      if (!result.ok()) {
        received_error = result.status();
      }
      return absl::OkStatus();
    }
  };

  FailingConsumerWithFinalize consumer;
  const absl::StatusOr<ParseStatus> status_or =
      ParseXSpace(xspace, {}, schema, consumer,
                  /*hlo_module_map=*/std::nullopt,
                  tensorflow::profiler::InlineExecutorFactory);

  EXPECT_THAT(status_or,
              StatusIs(absl::StatusCode::kInternal, "worker failed"));
  EXPECT_TRUE(consumer.finalized);
  EXPECT_THAT(consumer.received_error,
              StatusIs(absl::StatusCode::kInternal, "worker failed"));
}

TEST(XSpaceParserTest, PropagatesFinalizeError) {
  tensorflow::profiler::XSpace xspace;
  Schema schema;

  struct FailingFinalizeConsumer {
    absl::StatusOr<StepControl> Consume(Record&) {
      return StepControl::kContinue;
    }

    absl::Status Finalize() {
      return absl::InternalError("flush failed during finalize");
    }
  };

  FailingFinalizeConsumer consumer;
  const absl::StatusOr<ParseStatus> status_or =
      ParseXSpace(xspace, {}, schema, consumer,
                  /*hlo_module_map=*/std::nullopt,
                  tensorflow::profiler::InlineExecutorFactory);

  EXPECT_THAT(status_or.status(), StatusIs(absl::StatusCode::kInternal,
                                           "flush failed during finalize"));
}

TEST(XSpaceParserTest, WorksWithStatefulConsumerWithoutFinalize) {
  tensorflow::profiler::XSpace xspace;
  Schema schema;

  struct ConsumerWithoutFinalize {
    int count = 0;
    absl::StatusOr<StepControl> Consume(Record&) {
      count++;
      return StepControl::kContinue;
    }
  };

  ConsumerWithoutFinalize consumer;
  const absl::StatusOr<ParseStatus> status_or =
      ParseXSpace(xspace, {}, schema, consumer,
                  /*hlo_module_map=*/std::nullopt,
                  tensorflow::profiler::InlineExecutorFactory);

  EXPECT_THAT(status_or, IsOkAndHolds(Eq(ParseStatus::kComplete)));
}

TEST(XSpaceParserTest, WorksWithVoidFinalizeConsumer) {
  tensorflow::profiler::XSpace xspace;
  Schema schema;

  struct VoidFinalizeConsumer {
    int count = 0;
    bool finalized = false;
    absl::StatusOr<StepControl> Consume(Record&) {
      count++;
      return StepControl::kContinue;
    }
    void Finalize() { finalized = true; }
  };

  VoidFinalizeConsumer consumer;
  const absl::StatusOr<ParseStatus> status_or =
      ParseXSpace(xspace, {}, schema, consumer,
                  /*hlo_module_map=*/std::nullopt,
                  tensorflow::profiler::InlineExecutorFactory);

  EXPECT_THAT(status_or, IsOkAndHolds(Eq(ParseStatus::kComplete)));
  EXPECT_TRUE(consumer.finalized);
}

}  // namespace
}  // namespace xprof::events_db
