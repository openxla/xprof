/* Copyright 2024 The OpenXLA Authors. All Rights Reserved.

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
#ifndef THIRD_PARTY_XPROF_CONVERT_OP_STATS_PROCESSOR_H_
#define THIRD_PARTY_XPROF_CONVERT_OP_STATS_PROCESSOR_H_

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/profile_processor.h"
#include "xprof/convert/repository.h"
#include "xprof/convert/tool_options.h"
#include "plugin/xprof/protobuf/op_stats.pb.h"

namespace xprof {

class OpStatsProcessor : public ProfileProcessor {
 public:
  // Converts XSpace to serialized OpStats.
  absl::StatusOr<std::string> Map(
      const tensorflow::profiler::SessionSnapshot& session_snapshot,
      const std::string& hostname,
      const tensorflow::profiler::XSpace& xspace) final;

  // Deserializes map_outputs, combines OpStats, and calls
  // ProcessCombinedOpStats.
  absl::Status Reduce(
      const tensorflow::profiler::SessionSnapshot& session_snapshot,
      const std::vector<std::string>& map_output_files) final;

  // Default implementation for tools that don't need a worker service.
  absl::Status ProcessSession(
      const tensorflow::profiler::SessionSnapshot& session_snapshot,
      const tensorflow::profiler::ToolOptions& options) override {
    return absl::UnimplementedError(
        "ProcessSession not implemented for OpStatsProcessor");
  }

  // Tool-specific processing using the combined OpStats.
  virtual absl::Status ProcessCombinedOpStats(
      const tensorflow::profiler::SessionSnapshot& session_snapshot,
      const tensorflow::profiler::OpStats& combined_op_stats) = 0;

 private:
  // Helper to get map output for a single host, with caching.
  absl::StatusOr<std::string> GetMapOutputForHost(
      const tensorflow::profiler::SessionSnapshot& session_snapshot,
      int host_index);
};

}  // namespace xprof

#endif  // THIRD_PARTY_XPROF_CONVERT_OP_STATS_PROCESSOR_H_
