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

#include "testing/base/public/gmock.h"
#include "<gtest/gtest.h>"
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
  device_plane_builder.SetName(std::string(tsl::profiler::kGpuPlanePrefix) +
                               "0");
  device_plane_builder.AddStatValue(
      *device_plane_builder.GetOrCreateStatMetadata(
          GetStatTypeStr(StatType::kGlobalChipId)),
      0);

  XLineBuilder line_builder = device_plane_builder.GetOrCreateLine(0);
  line_builder.SetName("Stream 1");

  XEventBuilder event_builder = line_builder.AddEvent(
      *device_plane_builder.GetOrCreateEventMetadata("KernelA"));
  event_builder.AddStatValue(*device_plane_builder.GetOrCreateStatMetadata(
                                 GetStatTypeStr(StatType::kCounterValue)),
                             123ULL);
  event_builder.AddStatValue(
      *device_plane_builder.GetOrCreateStatMetadata(
          GetStatTypeStr(StatType::kPerformanceCounterDescription)),
      "Description A");
  event_builder.AddStatValue(
      *device_plane_builder.GetOrCreateStatMetadata(
          GetStatTypeStr(StatType::kPerformanceCounterSets)),
      "Set A");

  std::vector<std::unique_ptr<XSpace>> xspaces;
  xspaces.push_back(std::move(xspace));
  absl::StatusOr<SessionSnapshot> session_snapshot =
      SessionSnapshot::Create(paths, std::move(xspaces));
  TF_CHECK_OK(session_snapshot.status());
  return std::move(session_snapshot.value());
}

TEST(XPlaneToPerfCountersTest, ConvertMultiXSpacesToPerfCounters) {
  SessionSnapshot session_snapshot = CreateSessionSnapshot();
  auto result = ConvertMultiXSpacesToPerfCounters(session_snapshot);
  EXPECT_TRUE(result.ok());
  std::string json = result.value();

  // Basic validation of JSON content
  EXPECT_THAT(json, testing::HasSubstr("kernela"));
  EXPECT_THAT(json, testing::HasSubstr("Description A"));
  // 123.0 -> 0x7B
  EXPECT_THAT(json, testing::HasSubstr("0x7b"));
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
