/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XPROF_PYWRAP_PROFILER_PLUGIN_IMPL_H_
#define XPROF_PYWRAP_PROFILER_PLUGIN_IMPL_H_

#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/tsl/platform/types.h"
#include "xprof/convert/tool_options.h"

namespace xprof {
namespace pywrap {

absl::Status Monitor(const char* service_addr, int duration_ms,
                     int monitoring_level, bool display_timestamp,
                     tsl::string* result);

absl::StatusOr<std::pair<std::string, bool>> XSpaceToToolsData(
    std::vector<std::string> xspace_paths, const std::string& tool_name,
    const tensorflow::profiler::ToolOptions& tool_options);

absl::StatusOr<std::pair<std::string, bool>> XSpaceToToolsDataFromByteString(
    std::vector<std::string> xspace_strings,
    std::vector<std::string> xspace_paths, const std::string& tool_name,
    const tensorflow::profiler::ToolOptions& tool_options);

}  // namespace pywrap
}  // namespace xprof

#endif  // XPROF_PYWRAP_PROFILER_PLUGIN_IMPL_H_
