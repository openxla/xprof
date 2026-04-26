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


#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "file/base/filesystem.h"
#include "file/base/options.h"
#include "file/base/path.h"
#include "<gtest/gtest.h>"
#include "absl/status/status_matchers.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/file_utils.h"
#include "xprof/convert/repository.h"
#include "xprof/convert/tool_options.h"
#include "xprof/convert/unified_profile_processor_factory.h"
#include "plugin/xprof/protobuf/op_stats.pb.h"

namespace xprof {
namespace {

using ::tensorflow::profiler::SessionSnapshot;
using ::tensorflow::profiler::ToolOptions;
using ::tensorflow::profiler::XSpace;

TEST(UnifiedHloStatsProcessorTest, MinimalTest) {
  ToolOptions options;
  auto processor = UnifiedProfileProcessorFactory::GetInstance().Create(
      "hlo_stats", options);
  ASSERT_NE(processor, nullptr);

  std::string session_dir =
      file::JoinPath(testing::TempDir(), "unified_hlo_stats_processor_test");
  ASSERT_OK(file::CreateDir(session_dir, file::Defaults()));
  std::string xspace_path = file::JoinPath(session_dir, "test_host.xplane.pb");
  XSpace dummy_space;
  ASSERT_OK(xprof::WriteBinaryProto(xspace_path, dummy_space));

  auto status_or_session_snapshot =
      SessionSnapshot::Create({xspace_path}, std::nullopt);
  ASSERT_OK(status_or_session_snapshot);

  EXPECT_EQ(processor->GetData(), "");  // Initially empty

  // Clean up.
  ASSERT_OK(file::RecursivelyDelete(session_dir, file::Defaults()));
}

}  // namespace
}  // namespace xprof
