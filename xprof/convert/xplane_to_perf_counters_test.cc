/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#include "xprof/convert/xplane_to_perf_counters.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/file_system.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/profiler/utils/xplane_builder.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/repository.h"

namespace tensorflow {
namespace profiler {
namespace {

using ::tsl::profiler::StatType;
using ::tsl::profiler::XEventBuilder;
using ::tsl::profiler::XLineBuilder;
using ::tsl::profiler::XPlaneBuilder;

SessionSnapshot CreateSessionSnapshot() {
  std::string test_name =
      ::testing::UnitTest::GetInstance()->current_test_info()->name();
  std::string path = absl::StrCat("ram://", test_name, "/");
  std::unique_ptr<tsl::WritableFile> xplane_file_unused;
  tsl::Env::Default()
      ->NewAppendableFile(absl::StrCat(path, "hostname.xplane.pb"),
                          &xplane_file_unused)
      .IgnoreError();
  std::vector<std::string> paths = {path};
  auto xspace = std::make_unique<XSpace>();
  XPlaneBuilder host_plane_builder(xspace->add_planes());
  host_plane_builder.SetName("host:0");

  XPlaneBuilder device_plane_builder(xspace->add_planes());
  device_plane_builder.SetName(std::string(tsl::profiler::kTpuPlanePrefix) +
                               "0");
  device_plane_builder.AddStatValue(
      *device_plane_builder.GetOrCreateStatMetadata(
          GetStatTypeStr(StatType::kGlobalChipId)),
      0);
  device_plane_builder.AddStatValue(
      *device_plane_builder.GetOrCreateStatMetadata(
          GetStatTypeStr(StatType::kDeviceTypeString)),
      "TPU v7x");

  XLineBuilder line_builder = device_plane_builder.GetOrCreateLine(0);
  line_builder.SetName("Stream 1");

  XEventMetadata* event_metadata =
      device_plane_builder.GetOrCreateEventMetadata("KernelA");
  XStat* id_stat = event_metadata->add_stats();
  id_stat->set_metadata_id(device_plane_builder
                               .GetOrCreateStatMetadata(GetStatTypeStr(
                                   StatType::kPerformanceCounterId))
                               ->id());
  id_stat->set_uint64_value(2701299720ULL);

  XEventBuilder event_builder = line_builder.AddEvent(*event_metadata);
  event_builder.AddStatValue(*device_plane_builder.GetOrCreateStatMetadata(
                                 GetStatTypeStr(StatType::kCounterValue)),
                             123ULL);

  XPlaneBuilder gpu_plane_builder(xspace->add_planes());
  gpu_plane_builder.SetName(std::string(tsl::profiler::kGpuPlanePrefix) + "0");
  gpu_plane_builder.AddStatValue(*gpu_plane_builder.GetOrCreateStatMetadata(
                                     GetStatTypeStr(StatType::kGlobalChipId)),
                                 1);
  XLineBuilder gpu_line_builder = gpu_plane_builder.GetOrCreateLine(0);
  gpu_line_builder.SetName("Stream GPU");
  XEventBuilder gpu_event_builder = gpu_line_builder.AddEvent(
      *gpu_plane_builder.GetOrCreateEventMetadata("GpuKernelB"));
  gpu_event_builder.AddStatValue(*gpu_plane_builder.GetOrCreateStatMetadata(
                                     GetStatTypeStr(StatType::kCounterValue)),
                                 456ULL);
  gpu_event_builder.AddStatValue(
      *gpu_plane_builder.GetOrCreateStatMetadata(
          GetStatTypeStr(StatType::kPerformanceCounterDescription)),
      "Fallback Description B");
  gpu_event_builder.AddStatValue(
      *gpu_plane_builder.GetOrCreateStatMetadata(
          GetStatTypeStr(StatType::kPerformanceCounterSets)),
      "Fallback Set B");

  std::vector<std::unique_ptr<XSpace>> xspaces;
  xspaces.push_back(std::move(xspace));
  absl::StatusOr<SessionSnapshot> session_snapshot =
      SessionSnapshot::Create(paths, std::move(xspaces));
  TF_CHECK_OK(session_snapshot.status());
  return std::move(session_snapshot.value());
}

TEST(XPlaneToPerfCountersTest, ConvertMultiXSpacesToPerfCounters) {
  SessionSnapshot session_snapshot = CreateSessionSnapshot();
  absl::StatusOr<std::string> result =
      ConvertMultiXSpacesToPerfCounters(session_snapshot);
  EXPECT_TRUE(result.ok());
  std::string json = result.value();

  // Basic validation of JSON content.
  // For TPU kernels, we expect the counter description and set to be looked up
  // from the embedded CSV data.
  EXPECT_THAT(json, testing::HasSubstr("kernela"));
  EXPECT_THAT(
      json,
      testing::HasSubstr(
          "Insertion counter: incremented by 1 at each insertion to queue"));
  EXPECT_THAT(json, testing::HasSubstr("insertion queue_stats"));
  // 123.0 -> 0x7B
  EXPECT_THAT(json, testing::HasSubstr("0x7b"));

  // Validation of fallback description/set from XPlane stats.
  EXPECT_THAT(json, testing::HasSubstr("gpukernelb"));
  EXPECT_THAT(json, testing::HasSubstr("Fallback Description B"));
  EXPECT_THAT(json, testing::HasSubstr("Fallback Set B"));
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
