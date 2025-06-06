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

#include "testing/base/public/gmock.h"
#include "<gtest/gtest.h>"
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

namespace tensorflow {
namespace profiler {
namespace {

struct XPlaneToToolsTestCase {
  std::string test_name;
  std::string_view plane_name;
  bool has_hlo_module;
  bool has_dcn_collective_stats;
  std::vector<std::string> expected_tools;
};

SessionSnapshot CreateSessionSnapshot(std::unique_ptr<XSpace> xspace,
                                      bool has_hlo_module,
                                      bool has_dcn_collective_stats) {
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
                            test_case.has_dcn_collective_stats);

  absl::StatusOr<std::string> toolsString =
      GetAvailableToolNames(sessionSnapshot);
  ASSERT_TRUE(toolsString.ok());

  std::vector<std::string> tools = absl::StrSplit(toolsString.value(), ',');

  std::vector<std::string> expected_tools = {
      "trace_viewer",
      "overview_page",
      // TODO(jonahweaver): Re-enable input_pipeline_analyzer when it is ready.
      // b/407096031
      // "input_pipeline_analyzer",
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
         {}},
        {"ToolsForTpuWithHloModule",
         tsl::profiler::kTpuPlanePrefix,
         true,
         false,
         {"graph_viewer", "memory_viewer"}},
        {"ToolsForGpuWithoutHloModule",
         tsl::profiler::kGpuPlanePrefix,
         false,
         false,
         {"kernel_stats"}},
        {"ToolsForGpuWithHloModule",
         tsl::profiler::kGpuPlanePrefix,
         true,
         false,
         {"kernel_stats", "graph_viewer", "memory_viewer"}},
        {"ToolsForTpuWithDcnCollectiveStats",
         tsl::profiler::kTpuPlanePrefix,
         false,
         true,
         {"megascale_stats"}},
    }),
    [](const ::testing::TestParamInfo<XPlaneToToolsTest::ParamType>& info) {
      return info.param.test_name;
    });

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
