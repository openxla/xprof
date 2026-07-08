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

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "file/base/filesystem.h"
#include "file/base/options.h"
#include "file/base/path.h"
#include "testing/base/public/benchmark.h"
#include "testing/base/public/gmock.h"
#include "<gtest/gtest.h>"
#include "absl/log/check.h"
#include "absl/cleanup/cleanup.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/env.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/file_utils.h"
#include "xprof/convert/profile_processor_factory.h"
#include "xprof/convert/repository.h"
#include "xprof/convert/tool_options.h"
#include "xprof/convert/unified_session_snapshot.h"
#include "xprof/convert/xplane_to_tools_data_with_profile_processor.h"
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
using ::testing::status::StatusIs;

using GetDummySerializedProtoFn = std::function<std::string()>;
using VerifySerializedProtoFn = std::function<bool(absl::string_view)>;

struct ProfileProcessorTestParam {
  std::string test_name;
  std::string tool_name;
  GetDummySerializedProtoFn get_dummy_serialized_proto;
  VerifySerializedProtoFn verify_serialized_proto;
};

// Returns a dummy serialized OpStats proto as a string instead of a proto
// message object to match the GetDummySerializedProtoFn callback signature.
std::string GetDummyOpStats() {
  OpStats op_stats;
  op_stats.mutable_run_environment()->set_is_training(true);
  std::string output;
  CHECK(op_stats.SerializeToString(&output));
  return output;
}

bool VerifyOpStats(absl::string_view content) {
  return OpStats().ParseFromString(content);
}

// Returns an empty string for processors that do not require a specific proto
// payload.
std::string GetDummyEmptyProto() { return ""; }

bool VerifyEmptyProto(absl::string_view /*content*/) { return true; }

class ProfileProcessorTest
    : public ::testing::TestWithParam<ProfileProcessorTestParam> {};

TEST_P(ProfileProcessorTest, MapTest) {
  const ProfileProcessorTestParam& test_param = GetParam();
  ToolOptions options;
  std::unique_ptr<ProfileProcessor> processor =
      ProfileProcessorFactory::GetInstance().Create(test_param.tool_name,
                                                    options);
  ASSERT_NE(processor, nullptr);
  XSpace space;
  space.add_planes()->set_name("test_plane");
  // Create a SessionSnapshot with a minimal XSpace for the test.
  std::string session_dir =
      file::JoinPath(testing::TempDir(), test_param.test_name + "_map_test");
  ASSERT_OK(file::CreateDir(session_dir, file::Defaults()));
  auto cleanup = absl::MakeCleanup([&session_dir] {
    file::RecursivelyDelete(session_dir, file::Defaults()).IgnoreError();
  });
  std::string xspace_path = file::JoinPath(session_dir, "test_host.xplane.pb");
  XSpace dummy_space;
  ASSERT_OK(xprof::WriteBinaryProto(xspace_path, dummy_space));

  ASSERT_OK_AND_ASSIGN(SessionSnapshot session_snapshot,
                       SessionSnapshot::Create({xspace_path}, std::nullopt));

  std::string dummy_payload = test_param.get_dummy_serialized_proto();
  if (dummy_payload.empty()) {
    EXPECT_THAT(processor->Map(session_snapshot, "test_host", space),
                StatusIs(absl::StatusCode::kUnimplemented));
    return;
  }

  ASSERT_OK_AND_ASSIGN(std::string map_output_path,
                       processor->Map(session_snapshot, "test_host", space));

  // Verify that the output was written to the session snapshot.
  ASSERT_OK(tsl::Env::Default()->FileExists(map_output_path));

  std::string content;
  ASSERT_OK(
      tsl::ReadFileToString(tsl::Env::Default(), map_output_path, &content));
  EXPECT_THAT(content, Not(IsEmpty()));

  EXPECT_TRUE(test_param.verify_serialized_proto(content));
}

// Test the Reduce method for different tools.
TEST_P(ProfileProcessorTest, ReduceTest) {
  const ProfileProcessorTestParam& test_param = GetParam();
  ToolOptions options;
  std::unique_ptr<ProfileProcessor> processor =
      ProfileProcessorFactory::GetInstance().Create(test_param.tool_name,
                                                    options);
  ASSERT_NE(processor, nullptr);

  std::string dummy_payload = test_param.get_dummy_serialized_proto();
  if (dummy_payload.empty()) {
    std::vector<std::string> map_output_files = {"dummy_file.pb"};
    ASSERT_OK_AND_ASSIGN(
        SessionSnapshot session_snapshot,
        SessionSnapshot::Create({"dummy_file.pb"}, std::nullopt));
    EXPECT_THAT(processor->Reduce(session_snapshot, map_output_files),
                StatusIs(absl::StatusCode::kUnimplemented));
    return;
  }

  // Create temporary files for map outputs.
  std::string session_dir =
      file::JoinPath(testing::TempDir(), test_param.test_name + "_reduce_test");
  ASSERT_OK(file::CreateDir(session_dir, file::Defaults()));
  auto cleanup = absl::MakeCleanup([&session_dir] {
    file::RecursivelyDelete(session_dir, file::Defaults()).IgnoreError();
  });

  std::string map_output_path1 = file::JoinPath(session_dir, "map1.pb");
  ASSERT_OK(tsl::WriteStringToFile(tsl::Env::Default(), map_output_path1,
                                   dummy_payload));

  std::string map_output_path2 = file::JoinPath(session_dir, "map2.pb");
  ASSERT_OK(tsl::WriteStringToFile(tsl::Env::Default(), map_output_path2,
                                   dummy_payload));

  std::vector<std::string> map_output_files = {map_output_path1,
                                               map_output_path2};

  // Create a SessionSnapshot with a minimal XSpace for the test.
  std::vector<std::unique_ptr<XSpace>> xspaces;
  xspaces.push_back(std::make_unique<XSpace>());
  ASSERT_OK_AND_ASSIGN(
      SessionSnapshot session_snapshot,
      SessionSnapshot::Create({map_output_path1}, std::move(xspaces)));
  ASSERT_OK(processor->Reduce(session_snapshot, map_output_files));

  EXPECT_EQ(processor->GetContentType(), "application/json");
  EXPECT_THAT(processor->GetData(), Not(IsEmpty()));
}

// Test the E2E method for different tools.
TEST_P(ProfileProcessorTest, ProcessorE2ETest) {
  const ProfileProcessorTestParam& test_param = GetParam();
  // Create unique session dir for this test.
  std::string session_dir =
      file::JoinPath(testing::TempDir(), test_param.test_name + "_e2e_test");
  ASSERT_OK(file::CreateDir(session_dir, file::Defaults()));
  auto cleanup = absl::MakeCleanup([&session_dir] {
    file::RecursivelyDelete(session_dir, file::Defaults()).IgnoreError();
  });

  std::string xspace_path = file::JoinPath(session_dir, "test.xplane.pb");
  XSpace space;
  space.add_planes()->set_name("test_plane");
  ASSERT_OK(xprof::WriteBinaryProto(xspace_path, space));

  ASSERT_OK_AND_ASSIGN(SessionSnapshot session_snapshot,
                       SessionSnapshot::Create({xspace_path}, std::nullopt));

  ToolOptions options;
  std::string dummy_payload = test_param.get_dummy_serialized_proto();

  // First call - should compute and write to cache.
  absl::StatusOr<std::string> result1 =
      ConvertMultiXSpacesToToolDataWithProfileProcessor(
          session_snapshot, test_param.tool_name, options);

  if (dummy_payload.empty()) {
    // For non-OpStats tools, an empty XSpace might cause the processor to
    // return an error (e.g. missing HLO proto) or an empty result.
    if (!result1.ok()) {
      return;
    }
  } else {
    ASSERT_OK(result1);
    EXPECT_THAT(*result1, Not(IsEmpty()));
  }

  if (!dummy_payload.empty()) {
    ASSERT_OK_AND_ASSIGN(std::optional<std::string> cache_file_path,
                         session_snapshot.GetHostDataFilePath(
                             StoredDataType::OP_STATS,
                             tensorflow::profiler::kAllHostsIdentifier));
    EXPECT_TRUE(cache_file_path.has_value());
    ASSERT_OK(tsl::Env::Default()->FileExists(cache_file_path.value()));
  }

  // Second call - should hit the cache (or recompute if no caching).
  ASSERT_OK_AND_ASSIGN(std::string result2,
                       ConvertMultiXSpacesToToolDataWithProfileProcessor(
                           session_snapshot, test_param.tool_name, options));
  EXPECT_EQ(*result1, result2);
}

// Helper to map index to tool name for benchmarks.
std::string GetToolNameForBenchmark(int index) {
  static const std::vector<std::string> tool_names = {
      "overview_page",      "input_pipeline_analyzer",
      "kernel_stats",       "pod_viewer",
      "hlo_stats",          "roofline_model",
      "framework_op_stats", "op_profile"};
  return tool_names[index];
}

// Microbenchmark for the E2E performance of ProfileProcessor.
void BM_ProcessorE2ETest(benchmark::State& state) {
  const std::string tool_name = GetToolNameForBenchmark(state.range(0));
  // Setup: Create session directory and XSpace. This is done once per benchmark
  // run.
  std::string session_dir =
      file::JoinPath(testing::TempDir(), tool_name + "_e2e_benchmark");
  CHECK_OK(file::CreateDir(session_dir, file::Defaults()));
  std::string xspace_path = file::JoinPath(session_dir, "test.xplane.pb");
  XSpace space;
  space.add_planes()->set_name("test_plane");
  CHECK_OK(xprof::WriteBinaryProto(xspace_path, space));
  SessionSnapshot session_snapshot =
      SessionSnapshot::Create({xspace_path}, std::nullopt).value();
  ToolOptions options;

  for (auto s : state) {
    // Clear the cache file before each iteration to measure the full
    // computation.
    // absl::StatusOr<std::optional<std::string>> cache_file_path =
    //     session_snapshot.GetHostDataFilePath(
    //         StoredDataType::OP_STATS,
    //         tensorflow::profiler::kAllHostsIdentifier);
    // if (cache_file_path.has_value()) {
    //   file::Delete(*cache_file_path, file::Defaults()).IgnoreError();
    // }

    // Measure the performance of
    // ConvertMultiXSpacesToToolDataWithProfileProcessor.
    absl::StatusOr<std::string> result =
        ConvertMultiXSpacesToToolDataWithProfileProcessor(session_snapshot,
                                                          tool_name, options);
    benchmark::DoNotOptimize(result);
    CHECK_OK(result.status());
  }

  // Cleanup: Delete the session directory.
  CHECK_OK(file::RecursivelyDelete(session_dir, file::Defaults()));
}

// Register a benchmark for each tool.
BENCHMARK(BM_ProcessorE2ETest)->Arg(0);  // overview_page
                                         // ->Arg(1)  // input_pipeline_analyzer
                                         // ->Arg(2)  // kernel_stats
                                         // ->Arg(3)  // pod_viewer
                                         // ->Arg(4)  // hlo_stats
                                         // ->Arg(5)  // roofline_model
                                         // ->Arg(6)  // framework_op_stats
                                         // ->Arg(7); // op_profile

INSTANTIATE_TEST_SUITE_P(
    ProfileProcessorTests, ProfileProcessorTest,
    ::testing::ValuesIn<ProfileProcessorTestParam>({
        {"OverviewPage", "overview_page", GetDummyOpStats, VerifyOpStats},
        {"InputPipelineAnalyzer", "input_pipeline_analyzer", GetDummyOpStats,
         VerifyOpStats},
        {"KernelStats", "kernel_stats", GetDummyOpStats, VerifyOpStats},
        {"PodViewer", "pod_viewer", GetDummyOpStats, VerifyOpStats},
        {"HloStats", "hlo_stats", GetDummyOpStats, VerifyOpStats},
        {"RooflineModel", "roofline_model", GetDummyOpStats, VerifyOpStats},
        {"FrameworkOpStats", "framework_op_stats", GetDummyOpStats,
         VerifyOpStats},
        {"OpProfile", "op_profile", GetDummyOpStats, VerifyOpStats},
        {"MemoryProfile", "memory_profile", GetDummyEmptyProto,
         VerifyEmptyProto},
        {"MemoryViewer", "memory_viewer", GetDummyEmptyProto, VerifyEmptyProto},
        {"TraceViewer", "trace_viewer", GetDummyEmptyProto, VerifyEmptyProto},
        {"GraphViewer", "graph_viewer", GetDummyEmptyProto, VerifyEmptyProto},
    }),
    [](const ::testing::TestParamInfo<ProfileProcessorTest::ParamType>& info) {
      return info.param.test_name;
    });

}  // namespace
}  // namespace xprof
