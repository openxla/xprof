/* Copyright 2025 The OpenXLA Authors. All Rights Reserved.

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

#include "xprof/convert/memory_viewer_processor.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "file/base/filesystem.h"
#include "file/base/options.h"
#include "file/base/path.h"
#include "absl/status/status_matchers.h"
#include "<gtest/gtest.h>"
#include "absl/status/status.h"
#include "xla/service/hlo.pb.h"
#include "xla/tsl/platform/env.h"
#include "xprof/convert/repository.h"
#include "xprof/convert/tool_options.h"

namespace xprof {
namespace {

using ::tensorflow::profiler::SessionSnapshot;
using ::tensorflow::profiler::ToolOptions;
using ::tensorflow::profiler::XSpace;

class MemoryViewerProcessorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    test_dir_ =
        file::JoinPath(testing::TempDir(), "memory_viewer_processor_test");
    ASSERT_OK(file::CreateDir(test_dir_, file::Defaults()));

    // Create a dummy HloProto.
    xla::HloProto hlo_proto;
    hlo_proto.mutable_hlo_module()->set_name("test_module");
    std::string hlo_path =
        file::JoinPath(test_dir_, "test_module.hlo_proto.pb");
    ASSERT_OK(tsl::WriteBinaryProto(tsl::Env::Default(), hlo_path, hlo_proto));

    // Create a dummy XSpace file.
    XSpace xspace;
    xspace_path_ = file::JoinPath(test_dir_, "test.xplane.pb");
    ASSERT_OK(tsl::WriteBinaryProto(tsl::Env::Default(), xspace_path_, xspace));

    auto status_or_snapshot =
        SessionSnapshot::Create({xspace_path_}, std::nullopt);
    ASSERT_OK(status_or_snapshot);
    session_snapshot_ = std::make_unique<SessionSnapshot>(
        std::move(status_or_snapshot.value()));
  }

  void TearDown() override {
    file::RecursivelyDelete(test_dir_, file::Defaults()).IgnoreError();
  }

  std::string test_dir_;
  std::string xspace_path_;
  std::unique_ptr<SessionSnapshot> session_snapshot_;
};

TEST_F(MemoryViewerProcessorTest, RenderTimelineBool) {
  ToolOptions constructor_options;
  MemoryViewerProcessor processor(constructor_options);
  ToolOptions options;
  options["module_name"] = std::string("test_module");
  options["view_memory_allocation_timeline"] = true;

  ASSERT_OK(processor.ProcessSession(*session_snapshot_, options));
  EXPECT_EQ(processor.GetContentType(), "text/html");
}

TEST_F(MemoryViewerProcessorTest, RenderTimelineInt) {
  ToolOptions constructor_options;
  MemoryViewerProcessor processor(constructor_options);
  ToolOptions options;
  options["module_name"] = std::string("test_module");
  options["view_memory_allocation_timeline"] = 1;

  ASSERT_OK(processor.ProcessSession(*session_snapshot_, options));
  EXPECT_EQ(processor.GetContentType(), "text/html");
}

TEST_F(MemoryViewerProcessorTest, RenderTimelineFalse) {
  ToolOptions constructor_options;
  MemoryViewerProcessor processor(constructor_options);
  ToolOptions options;
  options["module_name"] = std::string("test_module");
  options["view_memory_allocation_timeline"] = false;

  ASSERT_OK(processor.ProcessSession(*session_snapshot_, options));
  EXPECT_EQ(processor.GetContentType(), "application/json");
}

TEST_F(MemoryViewerProcessorTest, RenderTimelineZero) {
  ToolOptions constructor_options;
  MemoryViewerProcessor processor(constructor_options);
  ToolOptions options;
  options["module_name"] = std::string("test_module");
  options["view_memory_allocation_timeline"] = 0;

  ASSERT_OK(processor.ProcessSession(*session_snapshot_, options));
  EXPECT_EQ(processor.GetContentType(), "application/json");
}

}  // namespace
}  // namespace xprof
