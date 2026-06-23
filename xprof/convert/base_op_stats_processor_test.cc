// Copyright 2026 The OpenXLA Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================

#include "xprof/convert/base_op_stats_processor.h"

#include <optional>
#include <string>
#include <vector>

#include "file/base/filesystem.h"
#include "file/base/options.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "xla/tsl/platform/env.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/file_utils.h"
#include "xprof/convert/repository.h"
#include "xprof/convert/tool_options.h"
#include "xprof/convert/unified_session_snapshot.h"
#include "plugin/xprof/protobuf/op_stats.pb.h"

namespace xprof {
namespace {

using ::tensorflow::profiler::OpStats;
using ::tensorflow::profiler::SessionSnapshot;
using ::tensorflow::profiler::ToolOptions;
using ::tensorflow::profiler::XSpace;

class MockOpStatsProcessor : public BaseOpStatsProcessor {
 public:
  explicit MockOpStatsProcessor(const ToolOptions& options)
      : BaseOpStatsProcessor(options), called_(false) {}

  absl::Status ProcessCombinedOpStats(
      const XprofSessionSnapshot& session_snapshot,
      const OpStats& combined_op_stats, const ToolOptions& options) override {
    called_ = true;
    return absl::OkStatus();
  }

  bool called() const { return called_; }

 private:
  bool called_;
};

class BaseOpStatsProcessorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    session_dir_ =
        tsl::io::JoinPath(testing::TempDir(), "base_op_stats_processor_test");
    file::RecursivelyDelete(session_dir_, file::Defaults()).IgnoreError();
    ASSERT_OK(file::CreateDir(session_dir_, file::Defaults()));
  }

  void TearDown() override {
    file::RecursivelyDelete(session_dir_, file::Defaults()).IgnoreError();
  }

  std::string session_dir_;
  ToolOptions options_;
};

TEST_F(BaseOpStatsProcessorTest, MinimalTest) {
  MockOpStatsProcessor processor(options_);

  std::string xspace_path = tsl::io::JoinPath(session_dir_, "test_host.xplane.pb");
  XSpace dummy_space;
  ASSERT_OK(xprof::WriteBinaryProto(xspace_path, dummy_space));

  ASSERT_OK_AND_ASSIGN(auto session_snapshot,
                       SessionSnapshot::Create({xspace_path}, std::nullopt));

  EXPECT_FALSE(processor.ShouldUseWorkerService(session_snapshot, options_));
}

TEST_F(BaseOpStatsProcessorTest, MapTest) {
  MockOpStatsProcessor processor(options_);

  std::string xspace_path = tsl::io::JoinPath(session_dir_, "test_host.xplane.pb");
  XSpace dummy_space;
  dummy_space.add_planes()->set_name("test_plane");
  ASSERT_OK(xprof::WriteBinaryProto(xspace_path, dummy_space));

  ASSERT_OK_AND_ASSIGN(auto session_snapshot,
                       SessionSnapshot::Create({xspace_path}, std::nullopt));

  ASSERT_OK_AND_ASSIGN(
      auto cache_path,
      processor.Map(session_snapshot, "test_host", dummy_space));

  EXPECT_TRUE(tsl::Env::Default()->FileExists(cache_path).ok());
}

TEST_F(BaseOpStatsProcessorTest, ReduceTest) {
  MockOpStatsProcessor processor(options_);

  std::string xspace_path = tsl::io::JoinPath(session_dir_, "test_host.xplane.pb");
  XSpace dummy_space;
  ASSERT_OK(xprof::WriteBinaryProto(xspace_path, dummy_space));
  ASSERT_OK_AND_ASSIGN(auto session_snapshot,
                       SessionSnapshot::Create({xspace_path}, std::nullopt));

  std::string op_stats_path =
      tsl::io::JoinPath(session_dir_, "test_host.op_stats.pb");
  OpStats dummy_op_stats;
  dummy_op_stats.mutable_run_environment()->set_device_type("TPU");
  ASSERT_OK(xprof::WriteBinaryProto(op_stats_path, dummy_op_stats));

  std::vector<std::string> map_output_files = {op_stats_path};

  ASSERT_OK(processor.Reduce(session_snapshot, map_output_files));

  EXPECT_TRUE(processor.called());
}

}  // namespace
}  // namespace xprof
