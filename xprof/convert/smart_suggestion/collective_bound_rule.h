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

#ifndef THIRD_PARTY_XPROF_CONVERT_SMART_SUGGESTION_COLLECTIVE_BOUND_RULE_H_
#define THIRD_PARTY_XPROF_CONVERT_SMART_SUGGESTION_COLLECTIVE_BOUND_RULE_H_

#include <optional>
#include <string>

#include "absl/strings/str_cat.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xla/tsl/platform/statusor.h"
#include "xprof/convert/smart_suggestion/signal_provider.h"
#include "xprof/convert/smart_suggestion/smart_suggestion_rule.h"
#include "plugin/xprof/protobuf/smart_suggestion.pb.h"

namespace tensorflow {
namespace profiler {

constexpr double kCollectiveBoundThresholdInPercent = 4;

// Rule to detect high collective op percentage bottleneck.
class CollectiveBoundRule : public SmartSuggestionRule {
 public:
    bool MeetsConditions(const SignalProvider& signal_provider) const override {
      absl::StatusOr<double> avg_collective_percent =
          signal_provider.GetAvgCollectiveTimePercent();
      if (!avg_collective_percent.ok()) {
        return false;
      }
      return *avg_collective_percent
          >= kCollectiveBoundThresholdInPercent;
    }

  absl::StatusOr<std::optional<SmartSuggestion>> GenerateSuggestion(
      const SignalProvider& signal_provider) const override {
    SmartSuggestion suggestion;
    suggestion.set_rule_name("CollectiveBoundRule");
    TF_ASSIGN_OR_RETURN(double avg_collective_percent,
                        signal_provider.GetAvgCollectiveTimePercent());
    std::string suggestion_text = absl::StrCat(
        "<p>Your program is likely bottlenecked by <b>Collective operations "
        "(e.g., AllReduce, AllGather)</b>. An average of <b>",
        absl::StrFormat("%.1f%%", avg_collective_percent),
        "</b> of each step is spent on these operations. This suggests that "
        "your model is collective communication-bound. Please consider the "
        "following optimizations:</p>"
        "<ul>"
        "<li><b>Overlap Communication with Computation:</b> Try to schedule "
        "collective operations to overlap with computation to hide "
        "communication latency.</li>"
        "<li><b>Investigate Workload Balance:</b> Check for stragglers, i.e., "
        "workers that are significantly slower than others. Uneven workloads "
        "can cause faster workers to wait during collective operations.</li>"
        "<li><b>Optimize Collective Operations:</b> Reduce the amount of data "
        "transferred between devices by using lower precision data formats for "
        "gradients (e.g., bfloat16).</li>"
        "<li><b>Check Network:</b> Network latency or bandwidth can be a "
        "bottleneck for distributed operations, causing workers to wait longer "
        "during collectives.</li>"
        "<li><b>Improve Data Input Pipeline:</b> Ensure your data loading and "
        "preprocessing pipeline is efficient and balanced across all workers. "
        "A slow input pipeline on one worker can stall all others.</li>"
        "</ul>");
    suggestion.set_suggestion_text(suggestion_text);
    return suggestion;
  }
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // THIRD_PARTY_XPROF_CONVERT_SMART_SUGGESTION_COLLECTIVE_BOUND_RULE_H_
