/* Copyright 2024 The OpenXLA Authors. All Rights Reserved.

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
#include "xprof/convert/profile_processor.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "file/base/filesystem.h"
#include "file/base/options.h"
#include "file/base/path.h"
#include "testing/base/public/gmock.h"
#include "<gtest/gtest.h>"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/env.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/profile_processor_factory.h"
#include "xprof/convert/repository.h"
#include "xprof/convert/tool_options.h"
#include "xprof/convert/xplane_to_tools_data.h"
#include "plugin/xprof/protobuf/op_stats.pb.h"

namespace xprof {
namespace {

using ::tensorflow::profiler::ConvertMultiXSpacesToToolDataWithProfileProcessor;
using ::tensorflow::profiler::OpStats;
using ::tensorflow::profiler::SessionSnapshot;
using ::tensorflow::profiler::StoredDataType;
using ::tensorflow::profiler::ToolOptions;
using ::tensorflow::profiler::XSpace;
using ::testing::IsEmpty;
using ::testing::Not;

struct ProfileProcessorTestParam {
  std::string test_name;
  std::string tool_name;
};

class ProfileProcessorTest
    : public ::testing::TestWithParam<ProfileProcessorTestParam> {};

TEST_P(ProfileProcessorTest, MapTest) {
  const ProfileProcessorTestParam& test_param = GetParam();
  ToolOptions options;
  auto processor = ProfileProcessorFactory::GetInstance().Create(
      test_param.tool_name, options);
  ASSERT_NE(processor, nullptr);
  XSpace space;
  space.add_planes()->set_name("test_plane");
  std::string output;
  // Create a SessionSnapshot with a minimal XSpace for the test.
  std::string session_dir =
      file::JoinPath(testing::TempDir(), test_param.test_name + "_map_test");
  ASSERT_OK(file::CreateDir(session_dir, file::Defaults()));
  std::string xspace_path = file::JoinPath(session_dir, "test_host.xplane.pb");
  XSpace dummy_space;
  ASSERT_OK(
      tsl::WriteBinaryProto(tsl::Env::Default(), xspace_path, dummy_space));

  auto status_or_session_snapshot =
      SessionSnapshot::Create({xspace_path}, std::nullopt);
  ASSERT_OK(status_or_session_snapshot);
  ASSERT_OK_AND_ASSIGN(
      std::string map_output_path,
      processor->Map(status_or_session_snapshot.value(), "test_host", space));

  // Verify that the output was written to the session snapshot.
  ASSERT_OK(tsl::Env::Default()->FileExists(map_output_path));

  std::string content;
  ASSERT_OK(
      tsl::ReadFileToString(tsl::Env::Default(), map_output_path, &content));
  EXPECT_THAT(content, Not(IsEmpty()));

  OpStats op_stats;
  ASSERT_TRUE(op_stats.ParseFromString(content));

    // Clean up.
  ASSERT_OK(file::RecursivelyDelete(session_dir, file::Defaults()));
}

// Test the Reduce method for different tools.
TEST_P(ProfileProcessorTest, ReduceTest) {
  const ProfileProcessorTestParam& test_param = GetParam();
  ToolOptions options;
  auto processor = ProfileProcessorFactory::GetInstance().Create(
      test_param.tool_name, options);
  ASSERT_NE(processor, nullptr);

  OpStats op_stats1;
  op_stats1.mutable_run_environment()->set_is_training(true);
  std::string output1;
  ASSERT_TRUE(op_stats1.SerializeToString(&output1));

  OpStats op_stats2;
  op_stats2.mutable_run_environment()->set_is_training(true);
  std::string output2;
  ASSERT_TRUE(op_stats2.SerializeToString(&output2));

  // Create temporary files for map outputs.
  std::string session_dir =
      file::JoinPath(testing::TempDir(), test_param.test_name + "_reduce_test");
  ASSERT_OK(file::CreateDir(session_dir, file::Defaults()));

  std::string map_output_path1 = file::JoinPath(session_dir, "map1.pb");
  ASSERT_OK(
      tsl::WriteStringToFile(tsl::Env::Default(), map_output_path1, output1));

  std::string map_output_path2 = file::JoinPath(session_dir, "map2.pb");
  ASSERT_OK(
      tsl::WriteStringToFile(tsl::Env::Default(), map_output_path2, output2));

  std::vector<std::string> map_output_files = {map_output_path1,
                                               map_output_path2};

  // Create a SessionSnapshot with a minimal XSpace for the test.
  std::vector<std::unique_ptr<XSpace>> xspaces;
  xspaces.push_back(std::make_unique<XSpace>());
  auto status_or_session_snapshot =
      SessionSnapshot::Create({map_output_path1}, std::move(xspaces));
  ASSERT_OK(status_or_session_snapshot);
  ASSERT_OK(
      processor->Reduce(status_or_session_snapshot.value(), map_output_files));

  EXPECT_EQ(processor->GetContentType(), "application/json");
  EXPECT_THAT(processor->GetData(), Not(IsEmpty()));
  // TODO(bhupendradubey): Add more specific checks on the JSON output.

  // Clean up.
  ASSERT_OK(file::RecursivelyDelete(session_dir, file::Defaults()));
}

// Test the E2E method for different tools.
TEST_P(ProfileProcessorTest, ProcessorE2ETest) {
  const ProfileProcessorTestParam& test_param = GetParam();
  // Create unique session dir for this test.
  std::string session_dir =
      file::JoinPath(testing::TempDir(), test_param.test_name + "_e2e_test");
  ASSERT_OK(file::CreateDir(session_dir, file::Defaults()));

  std::string xspace_path = file::JoinPath(session_dir, "test.xplane.pb");
  XSpace space;
  space.add_planes()->set_name("test_plane");
  ASSERT_OK(tsl::WriteBinaryProto(tsl::Env::Default(), xspace_path, space));

  ASSERT_OK_AND_ASSIGN(auto session_snapshot,
                       SessionSnapshot::Create({xspace_path}, std::nullopt));

  ToolOptions options;
  // First call - should compute and write to cache.
  ASSERT_OK_AND_ASSIGN(std::string result1,
                       ConvertMultiXSpacesToToolDataWithProfileProcessor(
                           session_snapshot, test_param.tool_name, options));
  EXPECT_THAT(result1, Not(IsEmpty()));

  // Check if cache file exists for the host.
  std::string hostname = session_snapshot.GetHostname(0);
  ASSERT_OK_AND_ASSIGN(
      auto cache_file_path,
      session_snapshot.GetHostDataFilePath(StoredDataType::OP_STATS, hostname));
  EXPECT_TRUE(cache_file_path.has_value());
  ASSERT_OK(tsl::Env::Default()->FileExists(cache_file_path.value()));

  // Second call - should hit the cache.
  ASSERT_OK_AND_ASSIGN(std::string result2,
                       ConvertMultiXSpacesToToolDataWithProfileProcessor(
                           session_snapshot, test_param.tool_name, options));
  EXPECT_EQ(result1, result2);

  // Clean up.
  ASSERT_OK(file::RecursivelyDelete(session_dir, file::Defaults()));
}

INSTANTIATE_TEST_SUITE_P(
    ProfileProcessorTests, ProfileProcessorTest,
    ::testing::ValuesIn<ProfileProcessorTestParam>({
        {"OverviewPage", "overview_page"},
        {"InputPipelineAnalyzer", "input_pipeline_analyzer"},
        {"FrameworkOpStats", "framework_op_stats"},
        {"KernelStats", "kernel_stats"},
    }),
    [](const ::testing::TestParamInfo<ProfileProcessorTest::ParamType>& info) {
      return info.param.test_name;
    });

}  // namespace
}  // namespace xprof
