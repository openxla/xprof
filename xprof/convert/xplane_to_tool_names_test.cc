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
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/cleanup/cleanup.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "xla/backends/profiler/cpu/metadata_utils.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/file_system.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/profiler/utils/xplane_builder.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_utils.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/kernel_viewer_registry.h"
#include "xprof/convert/repository.h"
#include "xprof/convert/tool_options.h"
#include "xprof/convert/unified_session_snapshot.h"

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
  std::string_view device_type = "";
};

SessionSnapshot CreateSessionSnapshot(std::unique_ptr<XSpace> xspace,
                                      bool has_hlo_module,
                                      bool has_dcn_collective_stats,
                                      bool has_perf_counters,
                                      std::string_view device_type = "") {
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
    if (!device_type.empty()) {
      plane_builder.AddStatValue(
          *plane_builder.GetOrCreateStatMetadata(tsl::profiler::GetStatTypeStr(
              tsl::profiler::StatType::kDeviceTypeString)),
          device_type);
    }
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

  SessionSnapshot sessionSnapshot =
      CreateSessionSnapshot(std::move(xspace), test_case.has_hlo_module,
                            test_case.has_dcn_collective_stats,
                            test_case.has_perf_counters, test_case.device_type);

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
         {"perf_counters"}},
        {"ToolsForUnsupportedTpuV5eWithPerfCounters",
         tsl::profiler::kTpuPlanePrefix,
         false,
         false,
         true,
         {"perf_counters"},
         "TPU v5e"},
        {"ToolsForTpuV6eWithPerfCounters",
         tsl::profiler::kTpuPlanePrefix,
         false,
         false,
         true,
         {"perf_counters", "utilization_viewer", "kernel_utilization"},
         "TPU v6 Lite"},
        {"ToolsForTpuV7xWithPerfCounters",
         tsl::profiler::kTpuPlanePrefix,
         false,
         false,
         true,
         {"perf_counters", "utilization_viewer", "kernel_utilization"},
         "TPU v7x"},
    }),
    [](const ::testing::TestParamInfo<XPlaneToToolsTest::ParamType>& info) {
      return info.param.test_name;
    });

// Adds a metadata plane with one LLO module to `xspace`.
void AddLloModule(XSpace* xspace) {
  tsl::profiler::XPlaneBuilder builder(xspace->add_planes());
  builder.SetName(tsl::profiler::kMetadataPlaneName);
  XStat* llo_stat =
      builder.GetOrCreateEventMetadata("module(1)::kernel")->add_stats();
  llo_stat->set_metadata_id(
      builder
          .GetOrCreateStatMetadata(
              tsl::profiler::GetStatTypeStr(tsl::profiler::StatType::kLloProto))
          ->id());
  llo_stat->set_bytes_value("llo-bytes");
}

absl::StatusOr<std::string> FakeKernelViewerHandler(
    const SessionSnapshot& session_snapshot, const ToolOptions& options) {
  return std::string("{}");
}

TEST(XPlaneToToolsKernelViewerTest, AdvertisesKernelViewerOnlyWhenRegistered) {
  absl::Cleanup reset_handler = [] { RegisterKernelViewerHandler(nullptr); };

  auto xspace = std::make_unique<XSpace>();
  tsl::profiler::FindOrAddMutablePlaneWithName(xspace.get(),
                                               tsl::profiler::kTpuPlanePrefix);
  AddLloModule(xspace.get());
  SessionSnapshot session = CreateSessionSnapshot(
      std::move(xspace), /*has_hlo_module=*/true,
      /*has_dcn_collective_stats=*/false, /*has_perf_counters=*/false);

  // Without a registered converter the tool is not advertised even though the
  // profile has LLO modules.
  RegisterKernelViewerHandler(nullptr);
  ASSERT_FALSE(IsKernelViewerRegistered());
  TF_ASSERT_OK_AND_ASSIGN(std::string tools, GetAvailableToolNames(session));
  EXPECT_THAT(absl::StrSplit(tools, ','),
              ::testing::Not(::testing::Contains("kernel_viewer")));

  // With a registered converter, profiles with LLO modules advertise the tool,
  // whereas profiles without LLO modules do not.
  RegisterKernelViewerHandler(&FakeKernelViewerHandler);
  TF_ASSERT_OK_AND_ASSIGN(tools, GetAvailableToolNames(session));
  EXPECT_THAT(absl::StrSplit(tools, ','), ::testing::Contains("kernel_viewer"));

  auto xspace_without_llo = std::make_unique<XSpace>();
  tsl::profiler::FindOrAddMutablePlaneWithName(xspace_without_llo.get(),
                                               tsl::profiler::kTpuPlanePrefix);
  SessionSnapshot session_without_llo = CreateSessionSnapshot(
      std::move(xspace_without_llo), /*has_hlo_module=*/false,
      /*has_dcn_collective_stats=*/false, /*has_perf_counters=*/false);
  TF_ASSERT_OK_AND_ASSIGN(tools, GetAvailableToolNames(session_without_llo));
  EXPECT_THAT(absl::StrSplit(tools, ','),
              ::testing::Not(::testing::Contains("kernel_viewer")));
}

TEST(XPlaneToToolsKernelViewerTest, CachesLloXPlaneForFirstHostOfFileSession) {
  absl::Cleanup reset_handler = [] { RegisterKernelViewerHandler(nullptr); };
  RegisterKernelViewerHandler(&FakeKernelViewerHandler);

  const std::string run_dir =
      absl::StrCat(testing::TempDir(), "/tool_names_llo_cache");
  TF_CHECK_OK(tsl::Env::Default()->RecursivelyCreateDir(run_dir));
  XSpace host_a_xspace;
  tsl::profiler::FindOrAddMutablePlaneWithName(&host_a_xspace,
                                               tsl::profiler::kTpuPlanePrefix);
  AddLloModule(&host_a_xspace);
  const std::string host_a_path = absl::StrCat(run_dir, "/host_a.xplane.pb");
  TF_CHECK_OK(
      tsl::WriteBinaryProto(tsl::Env::Default(), host_a_path, host_a_xspace));

  XSpace host_b_xspace;
  const std::string host_b_path = absl::StrCat(run_dir, "/host_b.xplane.pb");
  TF_CHECK_OK(
      tsl::WriteBinaryProto(tsl::Env::Default(), host_b_path, host_b_xspace));

  TF_ASSERT_OK_AND_ASSIGN(
      SessionSnapshot session,
      SessionSnapshot::Create({host_a_path, host_b_path}, std::nullopt));
  ASSERT_TRUE(session.HasAccessibleRunDir());

  TF_ASSERT_OK_AND_ASSIGN(std::string tools, GetAvailableToolNames(session));
  EXPECT_THAT(absl::StrSplit(tools, ','), ::testing::Contains("kernel_viewer"));

  TF_ASSERT_OK_AND_ASSIGN(
      const bool cache_exists,
      HostDataFileExists(session, StoredDataType::LLO_XPLANE, "host_a"));
  EXPECT_TRUE(cache_exists);
  XPlane cached_plane;
  TF_CHECK_OK(ReadBinaryProto(session, StoredDataType::LLO_XPLANE, "host_a",
                              &cached_plane));
  EXPECT_EQ(cached_plane.event_metadata_size(), 1);
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
