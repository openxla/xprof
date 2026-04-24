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
#include "file/base/path.h"
#include "testing/base/public/gmock.h"
#include "<gtest/gtest.h>"
#include "absl/status/status.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/file_utils.h"
#include "xprof/convert/repository.h"
#include "xprof/convert/tool_options.h"
#include "plugin/xprof/protobuf/op_stats.pb.h"

namespace xprof {
namespace {

using ::tensorflow::profiler::OpStats;
using ::tensorflow::profiler::SessionSnapshot;
using ::tensorflow::profiler::ToolOptions;
using ::tensorflow::profiler::XSpace;

class DummyOpStatsProcessor : public BaseOpStatsProcessor {
 public:
  explicit DummyOpStatsProcessor(const ToolOptions& options)
      : BaseOpStatsProcessor(options) {}

  absl::Status ProcessCombinedOpStats(
      const SessionSnapshot& session_snapshot,
      const OpStats& combined_op_stats,
      const ToolOptions& options) override {
    return absl::OkStatus();
  }
};

TEST(BaseOpStatsProcessorTest, MinimalTest) {
  ToolOptions options;
  DummyOpStatsProcessor processor(options);

  std::string session_dir =
      file::JoinPath(testing::TempDir(), "base_op_stats_processor_test");
  ASSERT_OK(file::CreateDir(session_dir, file::Defaults()));
  std::string xspace_path = file::JoinPath(session_dir, "test_host.xplane.pb");
  XSpace dummy_space;
  ASSERT_OK(xprof::WriteBinaryProto(xspace_path, dummy_space));

  auto status_or_session_snapshot =
      SessionSnapshot::Create({xspace_path}, std::nullopt);
  ASSERT_OK(status_or_session_snapshot);

  EXPECT_FALSE(processor.ShouldUseWorkerService(
      status_or_session_snapshot.value(), options));

  // Clean up.
  ASSERT_OK(file::RecursivelyDelete(session_dir, file::Defaults()));
}

}  // namespace
}  // namespace xprof
