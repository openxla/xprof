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

#ifndef THIRD_PARTY_XPROF_CONVERT_SMART_SUGGESTION_SPARSE_CORE_OFFLOAD_RULE_H_
#define THIRD_PARTY_XPROF_CONVERT_SMART_SUGGESTION_SPARSE_CORE_OFFLOAD_RULE_H_

#include <optional>
#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "xla/tsl/platform/statusor.h"
#include "xprof/convert/smart_suggestion/constants.h"
#include "xprof/convert/smart_suggestion/signal_provider.h"
#include "xprof/convert/smart_suggestion/smart_suggestion_rule.h"
#include "plugin/xprof/protobuf/smart_suggestion.pb.h"

namespace tensorflow {
namespace profiler {

// Rule to detect potential SparseCore offload opportunities.
class SparseCoreOffloadRule : public SmartSuggestionRule {
 public:
  bool MeetsConditions(const SignalProvider& signal_provider) const override {
    absl::StatusOr<double> async_done_percent =
        signal_provider.GetAsyncDoneOpPercent();
    absl::StatusOr<double> memory_utilization =
        signal_provider.GetPeakMemoryUtilization();

    if (!async_done_percent.ok() || !memory_utilization.ok()) {
      return false;
    }

    // Trigger if async-done is high and memory BW utilization is low.
    return *async_done_percent > kAsyncDoneThresholdInPercent &&
           *memory_utilization < kMemoryUtilizationHighThreshold;
  }

  absl::StatusOr<std::optional<SmartSuggestion>> GenerateSuggestion(
      const SignalProvider& signal_provider) const override {
    SmartSuggestion suggestion;
    suggestion.set_rule_name("SparseCoreOffloadRule");

    TF_ASSIGN_OR_RETURN(double async_done_percent,
                        signal_provider.GetAsyncDoneOpPercent());
    TF_ASSIGN_OR_RETURN(double memory_utilization,
                        signal_provider.GetPeakMemoryUtilization());

    std::string suggestion_text = absl::StrCat(
        "<p>Your program might benefit from offloading work to the "
        "SparseCore. <b>",
        absl::StrFormat("%.1f", async_done_percent),
        "%</b> of time is spent on async-done operations with low memory "
        "utilization of <b>",
        absl::StrFormat("%.1f", memory_utilization),
        "%</b>. This suggests that TensorCores might be waiting for data "
        "transfers or other asynchronous tasks. Consider increasing the values "
        "following flags to offload computations to the SparseCore to improve "
        "overall performance:</p>",
        "<ul>"
        "<li><b>xla_tpu_sparse_core_all_gather_latency_multiplier</b>: "
        "Cost model scaling factor for SparseCore offloaded all-gather. 1.0 "
        "(schedule using XLA cost model) / 0.0 (synchronous scheduling) / inf "
        "(schedule on data dependencies).</li>"
        "<li><b>xla_tpu_sparse_core_all_reduce_latency_multiplier</b>: "
        "Cost model scaling factor for SparseCore offloaded all-reduce.</li>"
        "<li><b>xla_tpu_sparse_core_reduce_gather_latency_multiplier</b>: "
        "Cost model scaling factor for SparseCore offloaded reduce-gather.</li>"
        "</ul>");

    suggestion.set_suggestion_text(suggestion_text);
    return suggestion;
  }
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // THIRD_PARTY_XPROF_CONVERT_SMART_SUGGESTION_SPARSE_CORE_OFFLOAD_RULE_H_
