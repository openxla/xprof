/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#ifndef THIRD_PARTY_XPROF_CONVERT_SMART_SUGGESTION_PROCESSOR_H_
#define THIRD_PARTY_XPROF_CONVERT_SMART_SUGGESTION_PROCESSOR_H_

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/profile_processor.h"
#include "xprof/convert/repository.h"
#include "xprof/convert/tool_options.h"

namespace xprof {

class SmartSuggestionProcessor : public ProfileProcessor {
 public:
  explicit SmartSuggestionProcessor(
      const tensorflow::profiler::ToolOptions& options) {}

  absl::StatusOr<std::string> Map(
      const tensorflow::profiler::SessionSnapshot& session_snapshot,
      const std::string& hostname,
      const tensorflow::profiler::XSpace& xspace) override;

  absl::Status Reduce(
      const tensorflow::profiler::SessionSnapshot& session_snapshot,
      const std::vector<std::string>& map_output_files) override;

  absl::Status ProcessSession(
      const tensorflow::profiler::SessionSnapshot& session_snapshot,
      const tensorflow::profiler::ToolOptions& options) override;
};

}  // namespace xprof

#endif  // THIRD_PARTY_XPROF_CONVERT_SMART_SUGGESTION_PROCESSOR_H_
