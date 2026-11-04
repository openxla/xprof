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

#include "xprof/convert/op_profile_processor.h"

#include "absl/status/status.h"
#include "xla/tsl/platform/errors.h"
#include "xprof/convert/multi_xplanes_to_op_stats.h"
#include "xprof/convert/repository.h"
#include "xprof/convert/tool_options.h"
#include "xprof/convert/unified_op_profile_processor.h"
#include "plugin/xprof/protobuf/op_stats.pb.h"

namespace xprof {

using tensorflow::profiler::OpStats;
using tensorflow::profiler::SessionSnapshot;

absl::Status OpProfileProcessor::ProcessSession(
    const SessionSnapshot& session_snapshot,
    const tensorflow::profiler::ToolOptions& options) {
  OpStats combined_op_stats;
  TF_RETURN_IF_ERROR(ConvertMultiXSpaceToCombinedOpStatsWithCache(
      session_snapshot, &combined_op_stats));
  return ProcessCombinedOpStats(session_snapshot, combined_op_stats, options);
}

absl::Status OpProfileProcessor::ProcessCombinedOpStats(
    const SessionSnapshot& session_snapshot, const OpStats& combined_op_stats,
    const tensorflow::profiler::ToolOptions& options) {
  UnifiedOpProfileProcessor processor(options);
  TF_RETURN_IF_ERROR(processor.ProcessCombinedOpStats(
      session_snapshot, combined_op_stats, options));
  SetOutput(processor.GetData(), processor.GetContentType());
  return absl::OkStatus();
}

}  // namespace xprof
