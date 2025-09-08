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

#include "xprof/convert/kernel_stats_processor.h"

#include <string>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/errors.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/multi_xplanes_to_op_stats.h"
#include "xprof/convert/repository.h"
#include "xprof/convert/xplane_to_kernel_stats_db.h"
#include "plugin/xprof/protobuf/op_stats.pb.h"

namespace xprof {

using tensorflow::profiler::ConvertMultiXSpaceToCombinedOpStatsWithCache;
using tensorflow::profiler::KernelStatsToDataTableJson;
using tensorflow::profiler::OpStats;
using tensorflow::profiler::SessionSnapshot;

absl::Status KernelStatsProcessor::ProcessSession(
    const SessionSnapshot& session_snapshot,
    const tensorflow::profiler::ToolOptions& options) {
  OpStats combined_op_stats;
  TF_RETURN_IF_ERROR(ConvertMultiXSpaceToCombinedOpStatsWithCache(
      session_snapshot, &combined_op_stats));
  auto json_output =
      KernelStatsToDataTableJson(combined_op_stats.kernel_stats_db());
  SetOutput(json_output, "application/json");
  return absl::OkStatus();
}

absl::Status KernelStatsProcessor::ProcessCombinedOpStats(
    const SessionSnapshot& session_snapshot, const OpStats& combined_op_stats) {
  std::string kernel_stats_json =
      KernelStatsToDataTableJson(combined_op_stats.kernel_stats_db());
  SetOutput(kernel_stats_json, "application/json");
  return absl::OkStatus();
}

}  // namespace xprof
