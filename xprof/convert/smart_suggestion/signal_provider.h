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

#ifndef THIRD_PARTY_XPROF_CONVERT_SMART_SUGGESTION_SIGNAL_PROVIDER_H_
#define THIRD_PARTY_XPROF_CONVERT_SMART_SUGGESTION_SIGNAL_PROVIDER_H_

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/tsl/platform/statusor.h"
#include "xprof/convert/smart_suggestion/constants.h"
#include "xprof/convert/smart_suggestion/tool_data_provider.h"
#include "xprof/utils/op_metrics_db_utils.h"
#include "plugin/xprof/protobuf/input_pipeline.pb.h"
#include "plugin/xprof/protobuf/overview_page.pb.h"
#include "plugin/xprof/protobuf/tpu_input_pipeline.pb.h"
#include "absl/strings/string_view.h"
#include "util/gtl/flat_set.h"
#include "plugin/xprof/protobuf/tf_op_stats.pb.h"
#include "tensorflow/core/profiler/protobuf/op_stats.pb.h"
#include "plugin/xprof/protobuf/op_metrics.pb.h"
#include "plugin/xprof/protobuf/steps_db.pb.h"

namespace tensorflow {
namespace profiler {

// SignalProvider is a helper class to provide signals for smart suggestion
// rules. It wraps a ToolDataProvider and provides methods to extract specific
// signals from the tool data.
class SignalProvider {
 public:
  explicit SignalProvider(std::unique_ptr<ToolDataProvider> tool_data_provider)
      : tool_data_provider_(std::move(tool_data_provider)) {}

  // Average HBM utilization from overview page.
  absl::StatusOr<double> GetHbmUtilization() const {
    TF_ASSIGN_OR_RETURN(const auto* overview_page,
                     tool_data_provider_->GetOverviewPage());
    return overview_page->analysis()
        .memory_bw_utilization_relative_to_hw_limit_percent();
  }

  // Average MXU utilization from overview page.
  absl::StatusOr<double> GetMxuUtilization() const {
    TF_ASSIGN_OR_RETURN(const auto* overview_page,
                     tool_data_provider_->GetOverviewPage());
    return overview_page->analysis().mxu_utilization_percent();
  }

  // Returns the input percentage of step time from input pipeline analysis.
  // Used for the input bound rule.
  absl::StatusOr<double> GetInputPercentOfStepTime() const {
    TF_ASSIGN_OR_RETURN(const auto* input_pipeline_analysis,
                     tool_data_provider_->GetInputPipelineAnalysisResult());
    return input_pipeline_analysis->input_percent();
  }

  // Returns the enqueue time (data transfer time: host-to-device) in
  // microseconds from host-side input analysis.
  // Used for the input bound rule.
  absl::StatusOr<double> GetEnqueueUs() const {
    TF_ASSIGN_OR_RETURN(const auto* input_pipeline_analysis,
                     tool_data_provider_->GetInputPipelineAnalysisResult());
    return input_pipeline_analysis->input_time_breakdown().enqueue_us();
  }

  // Returns the total non-enqueue time (non data transfer time) in microseconds
  // from host-side input analysis.
  // Used for the input bound rule.
  absl::StatusOr<double> GetNonEnqueueUs() const {
    TF_ASSIGN_OR_RETURN(const auto* input_pipeline_analysis,
                     tool_data_provider_->GetInputPipelineAnalysisResult());
    const auto& breakdown = input_pipeline_analysis->input_time_breakdown();
    return breakdown.demanded_file_read_us() +
           breakdown.advanced_file_read_us() + breakdown.preprocessing_us() +
           breakdown.unclassified_non_enqueue_us();
  }

  // Returns the percentage of input time that is due to enqueuing data.
  absl::StatusOr<double> GetEnqueuePercentOfInput() const {
    TF_ASSIGN_OR_RETURN(double enqueue_us, GetEnqueueUs());
    TF_ASSIGN_OR_RETURN(double non_enqueue_us, GetNonEnqueueUs());
    double total_input_time_us = enqueue_us + non_enqueue_us;
    if (total_input_time_us == 0) {
      return 0.0;
    }
    return (enqueue_us / total_input_time_us) * 100.0;
  }

  // Returns the percentage of input time that is due to non-enqueuing
  // activities.
  absl::StatusOr<double> GetNonEnqueuePercentOfInput() const {
    TF_ASSIGN_OR_RETURN(double non_enqueue_us, GetNonEnqueueUs());
    TF_ASSIGN_OR_RETURN(double enqueue_us, GetEnqueueUs());
    double total_input_time_us = non_enqueue_us + enqueue_us;
    if (total_input_time_us == 0) {
      return 0.0;
    }
    return (non_enqueue_us / total_input_time_us) * 100.0;
  }

  // Returns the collective time fraction of each step for collective ops.
  absl::StatusOr<std::vector<float>> GetCollectiveTimeFractionEachStep() const {
    TF_ASSIGN_OR_RETURN(const auto* tf_op_stats,
                        tool_data_provider_->GetTfOpStats());

    const auto& step_db = tf_op_stats->op_stats().step_db();
    std::vector<float> fractions;
    for (const auto& step : step_db.step_sequence()) {
      if (step.step_info_per_core().empty()) continue;
      uint64_t sum_duration_ps = 0;
      for (const auto& [core_id, step_info] : step.step_info_per_core()) {
        // Ignore sparse core steps as they do not contain collective ops.
        if (core_id >= kSparseCoreIndexStart) continue;
        sum_duration_ps += step_info.duration_ps();
      }
      uint64_t total_duration_ps = sum_duration_ps;
      if (total_duration_ps == 0) continue;

      uint64_t collective_time_ps = 0;
      for (const auto& it : step.hlo_metrics_db().metrics_db()) {
        if (IsCollective(it.category())) {
          collective_time_ps += it.self_time_ps();
        }
      }
      fractions.push_back(static_cast<float>(collective_time_ps) /
                          total_duration_ps);
    }
    return fractions;
  }

  // Returns the average percentage of step time for collective operations.
  absl::StatusOr<double> GetAvgCollectiveTimePercent() const {
    TF_ASSIGN_OR_RETURN(
        auto collective_fractions,
        GetCollectiveTimeFractionEachStep());
    if (collective_fractions.empty()) {
      return 0.0;
    }
    double sum = 0;
    for (float fraction : collective_fractions) {
      sum += fraction;
    }
    return sum / collective_fractions.size() * 100.0;
  }

  // Returns the percentage of time when the TensorCore is idle.
  absl::StatusOr<double> GetTensorCoreIdleTimePercent() const {
    TF_ASSIGN_OR_RETURN(const auto* input_pipeline_analysis,
                        tool_data_provider_->GetInputPipelineAnalysisResult());
    TpuStepTimeBreakdown step_time_breakdown;
    double tensor_core_idle_time_ms = 0.0;
    if (input_pipeline_analysis->step_time_breakdown().UnpackTo(
            &step_time_breakdown)) {
      tensor_core_idle_time_ms =
          step_time_breakdown.tc_idle_ms_summary().average();
    } else {
      return absl::NotFoundError("Failed to unpack TpuStepTimeBreakdown.");
    }
    double step_time_ms =
        input_pipeline_analysis->step_time_summary().average();
    if (step_time_ms == 0) {
      return 0.0;
    }
    return (tensor_core_idle_time_ms / step_time_ms) * 100.0;
  }

  // Returns the average percentage of step time for a given event name.
  absl::StatusOr<double> GetAvgEventTimePercent(
      const std::string& event_name) const {
    auto it = avg_event_time_percent_cache_.find(event_name);
    if (it != avg_event_time_percent_cache_.end()) {
      return it->second;
    }

    TF_ASSIGN_OR_RETURN(
        const auto* analyzer_result,
        tool_data_provider_->GetEventTimeFractionAnalyzerResult(event_name));

    double total_percent = 0;
    for (float event_percent : analyzer_result->event_time_fractions()) {
      total_percent += event_percent;
    }

    if (analyzer_result->event_time_fractions().empty()) {
      avg_event_time_percent_cache_[event_name] = 0.0;
      return 0.0;
    }
    double avg_percent =
        (total_percent / analyzer_result->event_time_fractions().size()) *
        100.0;
    avg_event_time_percent_cache_[event_name] = avg_percent;
    return avg_percent;
  }

  // Returns the percentage of time that is spent on SparseCore.
  absl::StatusOr<double> GetSparseCoreTimePercent() const {
    TF_ASSIGN_OR_RETURN(const auto* input_pipeline_analysis,
                        tool_data_provider_->GetInputPipelineAnalysisResult());
    TpuStepTimeBreakdown step_time_breakdown;
    double sparse_core_time_ms = 0.0;
    if (input_pipeline_analysis->step_time_breakdown().UnpackTo(
            &step_time_breakdown)) {
      sparse_core_time_ms = step_time_breakdown.sparse_core_step_summary()
                                     .sc_step_time_ms_summary()
                                     .average();
    } else {
      return absl::NotFoundError("Failed to unpack TpuStepTimeBreakdown.");
    }
    double step_time_ms =
        input_pipeline_analysis->step_time_summary().average();
    if (step_time_ms == 0) {
      return 0.0;
    }
    return (sparse_core_time_ms / step_time_ms) * 100.0;
  }

  // Returns true if the profile is latency bound, i.e. MXU and HBM utilization
  // are both below 50%.
  bool IsLatencyBound() const {
    absl::StatusOr<double> mxu_utilization = GetMxuUtilization();
    absl::StatusOr<double> hbm_utilization = GetHbmUtilization();
    if (!mxu_utilization.ok() || !hbm_utilization.ok()) {
      return false;
    }

    return *mxu_utilization < kMxuUtilizationLowThreshold &&
           *hbm_utilization < kHbmUtilizationLowThreshold;
  }

  // Returns true if the input percentage of step time is above the threshold.
  bool IsInputBound() const {
    absl::StatusOr<double> input_percent = GetInputPercentOfStepTime();
    if (!input_percent.ok()) {
      return false;
    }
    return *input_percent > kInfeedPercentageThreshold;
  }

 private:
  bool IsCollective(absl::string_view value) const {
    static constexpr auto kCollectives =
        (gtl::fixed_flat_set_of<absl::string_view>({
            "all-reduce",
            "all-reduce fusion",
            "all-reduce-scatter fusion",
            "all-to-all",
            "all-gather",
            "all-gather-start",
            "all-gather-done",
            "all-gather fusion",
            "reduce-scatter",
            "collective-permute",
            "collective-permute-done",
            "collective-permute-start",
            "megacore fusion",
            "host recv",
            "host recv-done",
            "host send",
            "host send-done",
        }));
    return kCollectives.contains(value);
  }

  std::unique_ptr<ToolDataProvider> tool_data_provider_;
  mutable absl::flat_hash_map<std::string, double>
      avg_event_time_percent_cache_;
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // THIRD_PARTY_XPROF_CONVERT_SMART_SUGGESTION_SIGNAL_PROVIDER_H_
