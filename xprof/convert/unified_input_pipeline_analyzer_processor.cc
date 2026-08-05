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

#include "xprof/convert/unified_input_pipeline_analyzer_processor.h"

#include <string>

#include "absl/status/status.h"
#include "xprof/convert/op_stats_to_input_pipeline_analysis.h"
#include "xprof/convert/tool_options.h"
#include "xprof/convert/unified_session_snapshot.h"
#include "plugin/xprof/protobuf/input_pipeline.pb.h"
#include "plugin/xprof/protobuf/op_stats.pb.h"

namespace xprof {

using ::tensorflow::profiler::ConvertOpStatsToInputPipelineAnalysis;
using ::tensorflow::profiler::InputPipelineAnalysisResult;
using ::tensorflow::profiler::InputPipelineAnalysisResultToDataTableJson;
using ::tensorflow::profiler::OpStats;
using ::tensorflow::profiler::ToolOptions;

absl::Status UnifiedInputPipelineAnalyzerProcessor::ProcessCombinedOpStats(
    const XprofSessionSnapshot& session_snapshot,
    const OpStats& combined_op_stats, const ToolOptions& options) {
  InputPipelineAnalysisResult result =
      ConvertOpStatsToInputPipelineAnalysis(combined_op_stats);

  std::string input_pipeline_json =
      InputPipelineAnalysisResultToDataTableJson(result);
  SetOutput(input_pipeline_json, /*content_type=*/"application/json");
  return absl::OkStatus();
}

}  // namespace xprof
