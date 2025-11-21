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

#ifndef THIRD_PARTY_XPROF_CONVERT_SMART_SUGGESTION_TOOL_DATA_PROVIDER_IMPL_H_
#define THIRD_PARTY_XPROF_CONVERT_SMART_SUGGESTION_TOOL_DATA_PROVIDER_IMPL_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/tsl/platform/errors.h"
#include "xprof/convert/multi_xplanes_to_op_stats.h"
#include "xprof/convert/op_stats_to_input_pipeline_analysis.h"
#include "xprof/convert/op_stats_to_overview_page.h"
#include "xprof/convert/repository.h"
#include "xprof/convert/smart_suggestion/tool_data_provider.h"
#include "plugin/xprof/protobuf/event_time_fraction_analyzer.pb.h"
#include "plugin/xprof/protobuf/input_pipeline.pb.h"
#include "plugin/xprof/protobuf/op_stats.pb.h"
#include "plugin/xprof/protobuf/overview_page.pb.h"

namespace tensorflow {
namespace profiler {

// Concrete class to provide tool data from a SessionSnapshot.
class ToolDataProviderImpl : public ToolDataProvider {
 public:
  explicit ToolDataProviderImpl(const SessionSnapshot& session_snapshot)
      : session_snapshot_(session_snapshot) {}

  absl::StatusOr<const OverviewPage*> GetOverviewPage() override {
    if (!overview_page_cache_) {
      OpStats combined_op_stats;
      TF_RETURN_IF_ERROR(ConvertMultiXSpaceToCombinedOpStatsWithCache(
          session_snapshot_, &combined_op_stats));
      OverviewPage overview_page =
          ConvertOpStatsToOverviewPage(combined_op_stats);
      overview_page_cache_ =
          std::make_unique<OverviewPage>(std::move(overview_page));
    }
    return overview_page_cache_.get();
  }

  absl::StatusOr<const InputPipelineAnalysisResult*>
  GetInputPipelineAnalysisResult() override {
    if (!input_pipeline_analysis_cache_) {
      OpStats combined_op_stats;
      TF_RETURN_IF_ERROR(ConvertMultiXSpaceToCombinedOpStatsWithCache(
          session_snapshot_, &combined_op_stats));
      InputPipelineAnalysisResult input_pipeline_analysis =
          ConvertOpStatsToInputPipelineAnalysis(combined_op_stats);
      input_pipeline_analysis_cache_ =
          std::make_unique<InputPipelineAnalysisResult>(
              std::move(input_pipeline_analysis));
    }
    return input_pipeline_analysis_cache_.get();
  }

  absl::StatusOr<const EventTimeFractionAnalyzerResult*>
  GetEventTimeFractionAnalyzerResult(const std::string& target_event_name) {
    return absl::UnimplementedError("Not implemented yet.");
  }

 private:
  const SessionSnapshot& session_snapshot_;
  std::unique_ptr<OverviewPage> overview_page_cache_;
  std::unique_ptr<InputPipelineAnalysisResult> input_pipeline_analysis_cache_;
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // THIRD_PARTY_XPROF_CONVERT_SMART_SUGGESTION_TOOL_DATA_PROVIDER_IMPL_H_
