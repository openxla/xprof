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

#ifndef THIRD_PARTY_XPROF_CONVERT_BASE_HLO_PROCESSOR_H_
#define THIRD_PARTY_XPROF_CONVERT_BASE_HLO_PROCESSOR_H_

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/service/hlo.pb.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/tool_options.h"
#include "xprof/convert/unified_profile_processor.h"
#include "xprof/convert/unified_session_snapshot.h"

namespace xprof {

// Unified base class for Hlo processors across 1P and 3P environments.
class BaseHloProcessor : public virtual UnifiedProfileProcessor {
 public:
  explicit BaseHloProcessor(const tensorflow::profiler::ToolOptions& options)
      : options_(options) {}

  virtual ~BaseHloProcessor() = default;

  absl::StatusOr<std::string> Map(
      const XprofSessionSnapshot& session_snapshot, absl::string_view hostname,
      const tensorflow::profiler::XSpace& xspace) override;

  absl::Status Reduce(
      const XprofSessionSnapshot& session_snapshot,
      const std::vector<std::string>& map_output_files) override;

  absl::Status ProcessSession(
      const XprofSessionSnapshot& session_snapshot,
      const tensorflow::profiler::ToolOptions& options) override;

  virtual absl::Status ProcessHlo(
      const XprofSessionSnapshot& session_snapshot,
      const xla::HloProto& hlo_proto,
      const tensorflow::profiler::ToolOptions& options) = 0;

 protected:
  tensorflow::profiler::ToolOptions options_;
};

}  // namespace xprof

#endif  // THIRD_PARTY_XPROF_CONVERT_BASE_HLO_PROCESSOR_H_
