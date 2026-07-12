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

#include "xprof/convert/unified_tools_registration.h"

#include <memory>

#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xprof/convert/tool_options.h"
#include "xprof/convert/unified_profile_processor.h"
#include "xprof/convert/unified_profile_processor_factory.h"

namespace xprof {
namespace {

using ::tensorflow::profiler::ToolOptions;

// Expected unified tools that EnsureUnifiedToolsRegistered must provide.
constexpr absl::string_view kExpectedUnifiedTools[] = {
    "hlo_stats",
    "memory_viewer",
    "op_profile",
    "overview_page",
};

TEST(UnifiedToolsRegistrationTest, EnsureRegistersAllExpectedTools) {
  absl::Status status = EnsureUnifiedToolsRegistered();
  ASSERT_TRUE(status.ok()) << status;

  ToolOptions options;
  for (absl::string_view tool_name : kExpectedUnifiedTools) {
    std::unique_ptr<UnifiedProfileProcessor> processor =
        UnifiedProfileProcessorFactory::GetInstance().Create(tool_name,
                                                             options);
    EXPECT_NE(processor, nullptr)
        << "Missing processor for tool: " << tool_name;
  }
}

TEST(UnifiedToolsRegistrationTest, EnsureIsIdempotent) {
  ASSERT_TRUE(EnsureUnifiedToolsRegistered().ok());
  ASSERT_TRUE(EnsureUnifiedToolsRegistered().ok());

  ToolOptions options;
  for (absl::string_view tool_name : kExpectedUnifiedTools) {
    EXPECT_NE(UnifiedProfileProcessorFactory::GetInstance().Create(tool_name,
                                                                   options),
              nullptr);
  }
}

// Existing call sites that only Register must keep working.
TEST(UnifiedToolsRegistrationTest, RegisterStillCreatesExpectedTools) {
  RegisterUnifiedToolRegistrations();

  ToolOptions options;
  for (absl::string_view tool_name : kExpectedUnifiedTools) {
    EXPECT_NE(UnifiedProfileProcessorFactory::GetInstance().Create(tool_name,
                                                                   options),
              nullptr)
        << "Missing processor for tool: " << tool_name;
  }
}

}  // namespace
}  // namespace xprof
