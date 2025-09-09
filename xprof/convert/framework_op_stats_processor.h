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

#ifndef THIRD_PARTY_XPROF_CONVERT_FRAMEWORK_OP_STATS_PROCESSOR_H_
#define THIRD_PARTY_XPROF_CONVERT_FRAMEWORK_OP_STATS_PROCESSOR_H_

#include "absl/status/status.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/op_stats_processor.h"
#include "xprof/convert/profile_processor_factory.h"
#include "xprof/convert/tool_options.h"
#include "plugin/xprof/protobuf/op_stats.pb.h"

namespace xprof {

class FrameworkOpStatsProcessor : public OpStatsProcessor {
 public:
  explicit FrameworkOpStatsProcessor(
      const tensorflow::profiler::ToolOptions& options)
      : OpStatsProcessor(options) {}

  absl::Status ProcessSession(
      const tensorflow::profiler::SessionSnapshot& session_snapshot,
      const tensorflow::profiler::ToolOptions& options) override;

  absl::Status ProcessCombinedOpStats(
      const tensorflow::profiler::SessionSnapshot& session_snapshot,
      const tensorflow::profiler::OpStats& combined_op_stats,
      const tensorflow::profiler::ToolOptions& options) override;
};

REGISTER_PROFILE_PROCESSOR("framework_op_stats", FrameworkOpStatsProcessor);

}  // namespace xprof

#endif  // THIRD_PARTY_XPROF_CONVERT_FRAMEWORK_OP_STATS_PROCESSOR_H_
