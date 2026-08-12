/* Copyright 2026 The OpenXLA Authors. All Rights Reserved.

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

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "testing/base/public/gmock.h"
#include "<gtest/gtest.h>"
#include "file/base/filesystem.h"
#include "file/base/options.h"
#include "file/base/path.h"
#include "absl/log/check.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/file_utils.h"
#include "xprof/convert/repository.h"
#include "xprof/convert/tool_options.h"
#include "xprof/convert/unified_profile_processor.h"
#include "xprof/convert/unified_profile_processor_factory.h"
#include "xprof/convert/unified_tools_registration.h"

namespace xprof {
namespace {

using ::tensorflow::profiler::SessionSnapshot;
using ::tensorflow::profiler::ToolOptions;
using ::tensorflow::profiler::XSpace;

class UnifiedFrameworkOpStatsProcessorTest : public testing::Test {
 protected:
  void SetUp() override {
    session_dir_ = file::JoinPath(
        testing::TempDir(), "unified_framework_op_stats_processor_test");
    file::RecursivelyDelete(session_dir_, file::Defaults()).IgnoreError();
    CHECK_OK(file::CreateDir(session_dir_, file::Defaults()));
    RegisterUnifiedToolRegistrations();
  }

  void TearDown() override {
    file::RecursivelyDelete(session_dir_, file::Defaults()).IgnoreError();
  }

  std::string session_dir_;
  ToolOptions options_;
};

TEST_F(UnifiedFrameworkOpStatsProcessorTest, MinimalTest) {
  std::unique_ptr<UnifiedProfileProcessor> processor =
      UnifiedProfileProcessorFactory::GetInstance().Create(
          "framework_op_stats", options_);
  ASSERT_NE(processor, nullptr);

  std::string xspace_path =
      file::JoinPath(session_dir_, "test_host.xplane.pb");
  XSpace dummy_space;
  ASSERT_OK(WriteBinaryProto(xspace_path, dummy_space));

  std::vector<std::string> xspace_paths = {xspace_path};
  ASSERT_OK_AND_ASSIGN(
      SessionSnapshot session_snapshot,
      SessionSnapshot::Create(xspace_paths, /*xspaces=*/std::nullopt));

  EXPECT_OK(processor->ProcessSession(session_snapshot, options_));
}

}  // namespace
}  // namespace xprof
