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

#include <string>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "xprof/convert/tool_options.h"
#include "xprof/convert/unified_hlo_stats_processor.h"
#include "xprof/convert/unified_memory_viewer_processor.h"
#include "xprof/convert/unified_op_profile_processor.h"
#include "xprof/convert/unified_overview_page_processor.h"
#include "xprof/convert/unified_profile_processor_factory.h"

namespace xprof {
namespace {

// Tools that must be available after RegisterUnifiedToolRegistrations().
constexpr absl::string_view kExpectedUnifiedTools[] = {
    "hlo_stats",
    "memory_viewer",
    "op_profile",
    "overview_page",
};

}  // namespace

void RegisterUnifiedToolRegistrations() {
  REGISTER_UNIFIED_PROFILE_PROCESSOR("hlo_stats", UnifiedHloStatsProcessor);
  REGISTER_UNIFIED_PROFILE_PROCESSOR("memory_viewer",
                                     UnifiedMemoryViewerProcessor);
  REGISTER_UNIFIED_PROFILE_PROCESSOR("op_profile", UnifiedOpProfileProcessor);
  REGISTER_UNIFIED_PROFILE_PROCESSOR("overview_page",
                                     UnifiedOverviewPageProcessor);
}

absl::Status EnsureUnifiedToolsRegistered() {
  RegisterUnifiedToolRegistrations();

  tensorflow::profiler::ToolOptions options;
  std::vector<std::string> missing_tools;
  for (absl::string_view tool_name : kExpectedUnifiedTools) {
    auto processor = UnifiedProfileProcessorFactory::GetInstance().Create(
        tool_name, options);
    if (processor == nullptr) {
      missing_tools.emplace_back(tool_name);
    }
  }

  if (!missing_tools.empty()) {
    std::string message = absl::StrCat(
        "Expected unified tools failed to register: ",
        absl::StrJoin(missing_tools, ", "));
    LOG(ERROR) << message;
    return absl::FailedPreconditionError(message);
  }
  return absl::OkStatus();
}

}  // namespace xprof
