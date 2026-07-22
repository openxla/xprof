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

#ifndef THIRD_PARTY_XPROF_CONVERT_UNIFIED_MEMORY_PROFILE_PROCESSOR_H_
#define THIRD_PARTY_XPROF_CONVERT_UNIFIED_MEMORY_PROFILE_PROCESSOR_H_

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/tool_options.h"
#include "xprof/convert/unified_profile_processor.h"
#include "xprof/convert/unified_session_snapshot.h"

namespace xprof {

class UnifiedMemoryProfileProcessor : public virtual UnifiedProfileProcessor {
 public:
  explicit UnifiedMemoryProfileProcessor(
      const tensorflow::profiler::ToolOptions& options) {}
  ~UnifiedMemoryProfileProcessor() override = default;

  absl::Status ProcessSession(
      const XprofSessionSnapshot& session_snapshot,
      const tensorflow::profiler::ToolOptions& options) override;

  absl::StatusOr<std::string> Map(
      const XprofSessionSnapshot& session_snapshot, absl::string_view hostname,
      const tensorflow::profiler::XSpace& xspace) override {
    return absl::UnimplementedError(
        "Map not implemented for UnifiedMemoryProfileProcessor");
  }

  absl::Status Reduce(
      const XprofSessionSnapshot& session_snapshot,
      const std::vector<std::string>& map_output_files) override {
    return absl::UnimplementedError(
        "Reduce not implemented for UnifiedMemoryProfileProcessor");
  }
};

}  // namespace xprof

#endif  // THIRD_PARTY_XPROF_CONVERT_UNIFIED_MEMORY_PROFILE_PROCESSOR_H_
