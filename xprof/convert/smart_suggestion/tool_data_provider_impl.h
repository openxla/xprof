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

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xprof/convert/multi_xplanes_to_op_stats.h"
#include "xprof/convert/op_stats_to_input_pipeline_analysis.h"
#include "xprof/convert/op_stats_to_op_profile.h"
#include "xprof/convert/op_stats_to_overview_page.h"
#include "xprof/convert/repository.h"
#include "xprof/convert/smart_suggestion/tool_data_provider.h"
#include "plugin/xprof/protobuf/event_time_fraction_analyzer.pb.h"
#include "plugin/xprof/protobuf/input_pipeline.pb.h"
#include "plugin/xprof/protobuf/op_profile.pb.h"
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
    absl::MutexLock lock(mutex_);
    if (!overview_page_cache_) {
      TF_ASSIGN_OR_RETURN(const OpStats* combined_op_stats, GetOpStatsLocked());
      OverviewPage overview_page =
          ConvertOpStatsToOverviewPage(*combined_op_stats);
      overview_page_cache_ =
          std::make_unique<OverviewPage>(std::move(overview_page));
    }
    return overview_page_cache_.get();
  }

  absl::StatusOr<const InputPipelineAnalysisResult*>
  GetInputPipelineAnalysisResult() override {
    absl::MutexLock lock(mutex_);
    if (!input_pipeline_analysis_cache_) {
      TF_ASSIGN_OR_RETURN(const OpStats* combined_op_stats, GetOpStatsLocked());
      InputPipelineAnalysisResult input_pipeline_analysis =
          ConvertOpStatsToInputPipelineAnalysis(*combined_op_stats);
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

  // Returns a non-owning pointer to OpStats. The lifetime of the returned
  // pointer is tied to the ToolDataProviderImpl instance.
  absl::StatusOr<const OpStats*> GetOpStats() override {
    absl::MutexLock lock(mutex_);
    return GetOpStatsLocked();
  }

  absl::StatusOr<const op_profile::Profile*> GetOpProfile() override {
    absl::MutexLock lock(mutex_);
    if (!op_profile_cache_) {
      TF_ASSIGN_OR_RETURN(const OpStats* combined_op_stats, GetOpStatsLocked());
      op_profile_cache_ = std::make_unique<op_profile::Profile>();
      ConvertOpStatsToOpProfile(
          *combined_op_stats,
          combined_op_stats->run_environment().hardware_type(),
          *op_profile_cache_);
    }
    return op_profile_cache_.get();
  }

  absl::StatusOr<const MemoryProfile*> GetMemoryProfile() override {
    return absl::UnimplementedError("Not implemented yet.");
  }

 private:
  // Returns OpStats, assumes mutex_ is already held.
  absl::StatusOr<const OpStats*> GetOpStatsLocked()
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
    if (!op_stats_cache_) {
      op_stats_cache_ = std::make_unique<OpStats>();
      TF_RETURN_IF_ERROR(ConvertMultiXSpaceToCombinedOpStatsWithCache(
          session_snapshot_, op_stats_cache_.get()));
    }
    return op_stats_cache_.get();
  }

  const SessionSnapshot& session_snapshot_;

  absl::Mutex mutex_;
  std::unique_ptr<OverviewPage> overview_page_cache_ ABSL_GUARDED_BY(mutex_);
  std::unique_ptr<InputPipelineAnalysisResult> input_pipeline_analysis_cache_
      ABSL_GUARDED_BY(mutex_);
  std::unique_ptr<OpStats> op_stats_cache_ ABSL_GUARDED_BY(mutex_);
  std::unique_ptr<op_profile::Profile> op_profile_cache_
      ABSL_GUARDED_BY(mutex_);
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // THIRD_PARTY_XPROF_CONVERT_SMART_SUGGESTION_TOOL_DATA_PROVIDER_IMPL_H_
