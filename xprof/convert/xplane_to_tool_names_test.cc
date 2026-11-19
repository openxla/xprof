/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "xprof/convert/xplane_to_tool_names.h"

#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "xla/backends/profiler/cpu/metadata_utils.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/file_system.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/profiler/utils/xplane_builder.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_utils.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/repository.h"
#include "xprof/convert/tool_options.h"

namespace tensorflow {
namespace profiler {
namespace {

struct XPlaneToToolsTestCase {
  std::string test_name;
  std::string_view plane_name;
  bool has_hlo_module;
  bool has_dcn_collective_stats;
  bool has_perf_counters;
  std::vector<std::string> expected_tools;
};

SessionSnapshot CreateSessionSnapshot(std::unique_ptr<XSpace> xspace,
                                      bool has_hlo_module,
                                      bool has_dcn_collective_stats,
                                      bool has_perf_counters) {
  std::string test_name =
      ::testing::UnitTest::GetInstance()->current_test_info()->name();
  std::string path = absl::StrCat("ram://", test_name, "/");
  std::unique_ptr<tsl::WritableFile> xplane_file;
  tsl::Env::Default()
      ->NewAppendableFile(absl::StrCat(path, "hostname.xplane.pb"),
                          &xplane_file)
      .IgnoreError();
  std::vector<std::string> paths = {path};

  if (has_hlo_module) {
    XPlane* hlo_module = xspace->add_planes();
    hlo_module->set_name(tsl::profiler::kMetadataPlaneName);
    xla::profiler::MetadataXPlaneBuilder builder(hlo_module);
  }

  if (has_dcn_collective_stats) {
    XPlane* host_thread = xspace->add_planes();
    tsl::profiler::XPlaneBuilder builder(host_thread);

    builder.SetName(tsl::profiler::kHostThreadsPlaneName);
    builder.GetOrCreateEventMetadata("MegaScale:Megascale Event");
  }

  if (has_perf_counters) {
    XPlane* device_plane = xspace->add_planes();
    tsl::profiler::XPlaneBuilder plane_builder(device_plane);
    plane_builder.SetName(tsl::profiler::kTpuPlanePrefix);
    tsl::profiler::XLineBuilder line_builder = plane_builder.GetOrCreateLine(0);
    line_builder
        .AddEvent(*plane_builder.GetOrCreateEventMetadata("dummy_event"))
        .AddStatValue(*plane_builder.GetOrCreateStatMetadata(
                          tsl::profiler::GetStatTypeStr(
                              tsl::profiler::StatType::kCounterValue)),
                      123);
  }

  std::vector<std::unique_ptr<XSpace>> xspaces;
  xspaces.push_back(std::move(xspace));

  absl::StatusOr<SessionSnapshot> session_snapshot =
      SessionSnapshot::Create(paths, std::move(xspaces));
  TF_CHECK_OK(session_snapshot.status());
  return std::move(session_snapshot.value());
}

using XPlaneToToolsTest = ::testing::TestWithParam<XPlaneToToolsTestCase>;

TEST_P(XPlaneToToolsTest, ToolsList) {
  const XPlaneToToolsTestCase& test_case = GetParam();
  auto xspace = std::make_unique<XSpace>();

  tsl::profiler::FindOrAddMutablePlaneWithName(xspace.get(),
                                               test_case.plane_name);

  SessionSnapshot sessionSnapshot = CreateSessionSnapshot(
      std::move(xspace), test_case.has_hlo_module,
      test_case.has_dcn_collective_stats, test_case.has_perf_counters);

  absl::StatusOr<std::string> toolsString =
      GetAvailableToolNames(sessionSnapshot);
  ASSERT_TRUE(toolsString.ok());

  std::vector<std::string> tools = absl::StrSplit(toolsString.value(), ',');

  std::vector<std::string> expected_tools = {
      "trace_viewer",
      "overview_page",
      "input_pipeline_analyzer",
      "framework_op_stats",
      "memory_profile",
      // "pod_viewer",
      "op_profile",
      "hlo_stats",
      "roofline_model",
      // "inference_profile",
  };
  expected_tools.insert(expected_tools.end(), test_case.expected_tools.begin(),
                        test_case.expected_tools.end());

  EXPECT_THAT(tools, ::testing::UnorderedElementsAreArray(expected_tools));
}

INSTANTIATE_TEST_SUITE_P(
    XPlaneToToolsTests, XPlaneToToolsTest,
    ::testing::ValuesIn<XPlaneToToolsTestCase>({
        {"ToolsForTpuWithoutHloModule",
         tsl::profiler::kTpuPlanePrefix,
         false,
         false,
         false,
         {}},
        {"ToolsForTpuWithHloModule",
         tsl::profiler::kTpuPlanePrefix,
         true,
         false,
         false,
         {"graph_viewer", "memory_viewer"}},
        {"ToolsForGpuWithoutHloModule",
         tsl::profiler::kGpuPlanePrefix,
         false,
         false,
         false,
         {"kernel_stats"}},
        {"ToolsForGpuWithHloModule",
         tsl::profiler::kGpuPlanePrefix,
         true,
         false,
         false,
         {"kernel_stats", "graph_viewer", "memory_viewer"}},
        {"ToolsForTpuWithDcnCollectiveStats",
         tsl::profiler::kTpuPlanePrefix,
         false,
         true,
         false,
         {"megascale_stats"}},
        {"ToolsForTpuWithPerfCounters",
         tsl::profiler::kTpuPlanePrefix,
         false,
         false,
         true,
         {"perf_counters", "utilization_viewer", "kernel_utilization"}},
    }),
    [](const ::testing::TestParamInfo<XPlaneToToolsTest::ParamType>& info) {
      return info.param.test_name;
    });

// Returns a one-host session whose host plane has a gRPC flow event when
// `has_grpc_metrics_events` is true.
SessionSnapshot CreateGrpcMetricsSessionSnapshot(bool has_grpc_metrics_events) {
  auto xspace = std::make_unique<XSpace>();
  tsl::profiler::FindOrAddMutablePlaneWithName(xspace.get(),
                                               tsl::profiler::kTpuPlanePrefix);
  tsl::profiler::XPlaneBuilder host_plane(
      tsl::profiler::FindOrAddMutablePlaneWithName(
          xspace.get(), tsl::profiler::kHostThreadsPlaneName));
  host_plane.GetOrCreateEventMetadata(
      has_grpc_metrics_events ? "Megascale_gRPC:SendRpc" : "OtherEvent");

  std::vector<std::unique_ptr<XSpace>> xspaces;
  xspaces.push_back(std::move(xspace));
  absl::StatusOr<SessionSnapshot> session_snapshot = SessionSnapshot::Create(
      {"ram://grpc_metrics_tools/hostname.xplane.pb"}, std::move(xspaces));
  TF_CHECK_OK(session_snapshot.status());
  return std::move(session_snapshot.value());
}

// Why: the DCN Analyzer stays hidden unless it is turned on.
// What: a profile with gRPC flow events lists grpc_metrics only when the
// enable_dcn_analyzer option is true.
TEST(XPlaneToToolsGrpcMetricsTest, ListedOnlyWhenEnabled) {
  // Setup.
  const SessionSnapshot session_snapshot =
      CreateGrpcMetricsSessionSnapshot(/*has_grpc_metrics_events=*/true);
  ToolOptions enabled_options;
  enabled_options[std::string(kEnableDcnAnalyzerOption)] = true;
  ToolOptions disabled_options;
  disabled_options[std::string(kEnableDcnAnalyzerOption)] = false;

  // Execution.
  const absl::StatusOr<std::string> default_tools =
      GetAvailableToolNames(session_snapshot);
  const absl::StatusOr<std::string> disabled_tools =
      GetAvailableToolNames(session_snapshot, disabled_options);
  const absl::StatusOr<std::string> enabled_tools =
      GetAvailableToolNames(session_snapshot, enabled_options);

  // Verification.
  ASSERT_TRUE(default_tools.ok());
  ASSERT_TRUE(disabled_tools.ok());
  ASSERT_TRUE(enabled_tools.ok());
  const std::vector<std::string> default_list =
      absl::StrSplit(*default_tools, ',');
  const std::vector<std::string> disabled_list =
      absl::StrSplit(*disabled_tools, ',');
  const std::vector<std::string> enabled_list =
      absl::StrSplit(*enabled_tools, ',');
  EXPECT_THAT(default_list,
              ::testing::Not(::testing::Contains("grpc_metrics")));
  EXPECT_THAT(disabled_list,
              ::testing::Not(::testing::Contains("grpc_metrics")));
  EXPECT_THAT(enabled_list, ::testing::Contains("grpc_metrics"));
}

// Why: turning the DCN Analyzer on must not list it for profiles without its
// events.
// What: a profile without gRPC flow events does not list grpc_metrics.
TEST(XPlaneToToolsGrpcMetricsTest, NotListedWithoutEvents) {
  // Setup.
  const SessionSnapshot session_snapshot =
      CreateGrpcMetricsSessionSnapshot(/*has_grpc_metrics_events=*/false);
  ToolOptions enabled_options;
  enabled_options[std::string(kEnableDcnAnalyzerOption)] = true;

  // Execution.
  const absl::StatusOr<std::string> enabled_tools =
      GetAvailableToolNames(session_snapshot, enabled_options);

  // Verification.
  ASSERT_TRUE(enabled_tools.ok());
  const std::vector<std::string> enabled_list =
      absl::StrSplit(*enabled_tools, ',');
  EXPECT_THAT(enabled_list,
              ::testing::Not(::testing::Contains("grpc_metrics")));
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
