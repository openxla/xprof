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

#include "xprof/convert/overview_page_processor.h"

#include <string>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/errors.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/compute_inference_latency.h"
#include "xprof/convert/multi_xplanes_to_op_stats.h"
#include "xprof/convert/multi_xspace_to_inference_stats.h"
#include "xprof/convert/op_stats_to_overview_page.h"
#include "xprof/convert/repository.h"
#include "xprof/convert/tool_options.h"
#include "plugin/xprof/protobuf/op_stats.pb.h"
#include "plugin/xprof/protobuf/overview_page.pb.h"

namespace xprof {

using tensorflow::profiler::ConvertMultiXSpaceToCombinedOpStatsWithCache;
using tensorflow::profiler::ConvertMultiXSpaceToInferenceStats;
using tensorflow::profiler::InferenceStats;
using tensorflow::profiler::OpStats;
using tensorflow::profiler::OverviewPage;
using tensorflow::profiler::SessionSnapshot;

absl::Status OverviewPageProcessor::ProcessCombinedOpStats(
    const SessionSnapshot& session_snapshot, const OpStats& combined_op_stats) {
  OverviewPage overview_page = ConvertOpStatsToOverviewPage(combined_op_stats);

  if (!combined_op_stats.run_environment().is_training()) {
    InferenceStats inference_stats;
    TF_RETURN_IF_ERROR(ConvertMultiXSpaceToInferenceStats(
        session_snapshot, "", "", &inference_stats));
    *overview_page.mutable_inference_latency() =
        tensorflow::profiler::ComputeInferenceLatencyResult(inference_stats);
  }

  std::string overview_page_json = OverviewPageToJson(overview_page);
  SetOutput(overview_page_json, "application/json");
  return absl::OkStatus();
}

absl::Status OverviewPageProcessor::ProcessSession(
    const SessionSnapshot& session_snapshot,
    const tensorflow::profiler::ToolOptions& options) {
  LOG(INFO) << "OverviewPageProcessor::ProcessSession";
  OpStats combined_op_stats;
  TF_RETURN_IF_ERROR(ConvertMultiXSpaceToCombinedOpStatsWithCache(
      session_snapshot, &combined_op_stats));
  OverviewPage overview_page = ConvertOpStatsToOverviewPage(combined_op_stats);
  if (!combined_op_stats.run_environment().is_training()) {
    InferenceStats inference_stats;
    TF_RETURN_IF_ERROR(ConvertMultiXSpaceToInferenceStats(
        session_snapshot, "", "", &inference_stats));
    *overview_page.mutable_inference_latency() =
        tensorflow::profiler::ComputeInferenceLatencyResult(inference_stats);
  }
  std::string overview_page_json = OverviewPageToJson(overview_page);
  SetOutput(overview_page_json, "application/json");
  return absl::OkStatus();
}

}  // namespace xprof
