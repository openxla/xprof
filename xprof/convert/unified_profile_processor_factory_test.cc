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

#include "xprof/convert/unified_profile_processor_factory.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "<gtest/gtest.h>"
#include "testing/base/public/gmock.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/repository.h"
#include "xprof/convert/tool_options.h"
#include "xprof/convert/unified_profile_processor.h"

namespace xprof {
namespace {

class DummyProcessor : public UnifiedProfileProcessor {
 public:
  absl::StatusOr<std::string> Map(
      const tensorflow::profiler::SessionSnapshot& session_snapshot,
      absl::string_view hostname,
      const tensorflow::profiler::XSpace& xspace) override {
    return "";
  }
  absl::Status Reduce(
      const tensorflow::profiler::SessionSnapshot& session_snapshot,
      const std::vector<std::string>& map_output_files) override {
    return absl::OkStatus();
  }
};

class MacroRegisteredProcessor : public UnifiedProfileProcessor {
 public:
  explicit MacroRegisteredProcessor(
      const tensorflow::profiler::ToolOptions& /*options*/) {}
  absl::StatusOr<std::string> Map(
      const tensorflow::profiler::SessionSnapshot& session_snapshot,
      absl::string_view hostname,
      const tensorflow::profiler::XSpace& xspace) override {
    return "";
  }
  absl::Status Reduce(
      const tensorflow::profiler::SessionSnapshot& session_snapshot,
      const std::vector<std::string>& map_output_files) override {
    return absl::OkStatus();
  }
};

REGISTER_UNIFIED_PROFILE_PROCESSOR(macro_registered_tool,
                                   "macro_registered_tool",
                                   MacroRegisteredProcessor);

TEST(UnifiedProfileProcessorTest, BaseClassMethods) {
  DummyProcessor processor;
  processor.SetOutput("test_data", "test_content_type");
  EXPECT_EQ(processor.GetData(), "test_data");
  EXPECT_EQ(processor.GetContentType(), "test_content_type");

  ASSERT_OK_AND_ASSIGN(auto session_snapshot,
                       tensorflow::profiler::SessionSnapshot::Create(
                           /*xspace_paths=*/{"host1/trace.xspace"},
                           /*xspaces=*/std::nullopt));
  tensorflow::profiler::ToolOptions options;
  EXPECT_FALSE(processor.ShouldUseWorkerService(session_snapshot, options));
  EXPECT_THAT(processor.ProcessSession(session_snapshot, options),
              testing::status::StatusIs(absl::StatusCode::kUnimplemented));
}

TEST(UnifiedProfileProcessorFactoryTest, RegisterAndCreate) {
  UnifiedProfileProcessorFactory::GetInstance().Register(
      "dummy_tool",
      [](const tensorflow::profiler::ToolOptions& options)
          -> std::unique_ptr<UnifiedProfileProcessor> {
        return std::make_unique<DummyProcessor>();
      });

  tensorflow::profiler::ToolOptions options;
  auto processor = UnifiedProfileProcessorFactory::GetInstance().Create(
      "dummy_tool", options);
  EXPECT_NE(processor, nullptr);
}

TEST(UnifiedProfileProcessorFactoryTest, CreateNonExistent) {
  tensorflow::profiler::ToolOptions options;
  auto processor = UnifiedProfileProcessorFactory::GetInstance().Create(
      "non_existent_tool", options);
  EXPECT_EQ(processor, nullptr);
}

TEST(UnifiedProfileProcessorFactoryTest, RegisterMacro) {
  tensorflow::profiler::ToolOptions options;
  auto processor = UnifiedProfileProcessorFactory::GetInstance().Create(
      "macro_registered_tool", options);
  EXPECT_NE(processor, nullptr);
}

}  // namespace
}  // namespace xprof
