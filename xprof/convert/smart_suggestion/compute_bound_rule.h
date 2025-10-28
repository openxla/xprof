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

#ifndef THIRD_PARTY_XPROF_CONVERT_SMART_SUGGESTION_COMPUTE_BOUND_RULE_H_
#define THIRD_PARTY_XPROF_CONVERT_SMART_SUGGESTION_COMPUTE_BOUND_RULE_H_

#include <optional>
#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "xla/tsl/platform/statusor.h"
#include "xprof/convert/smart_suggestion/signal_provider.h"
#include "xprof/convert/smart_suggestion/smart_suggestion_rule.h"
#include "plugin/xprof/protobuf/smart_suggestion.pb.h"

namespace tensorflow {
namespace profiler {

// If MXU utilization is higher than kComputeBoundMxuUtilizationThreshold and
// HBM bandwidth utilization is lower than kComputeBoundHbmUtilizationThreshold,
// it is considered compute bound.
constexpr double kComputeBoundMxuUtilizationThreshold = 70;
constexpr double kComputeBoundHbmUtilizationThreshold = 50;

// Rule to detect compute-bound bottleneck.
class ComputeBoundRule : public SmartSuggestionRule {
 public:
  bool MeetsConditions(const SignalProvider& signal_provider) const override {
    absl::StatusOr<double> hbm_utilization_percent =
        signal_provider.GetHbmUtilization();
    absl::StatusOr<double> mxu_utilization_percent =
        signal_provider.GetMxuUtilization();
    if (!hbm_utilization_percent.ok() || !mxu_utilization_percent.ok()) {
      return false;
    }

    return *mxu_utilization_percent > kComputeBoundMxuUtilizationThreshold &&
           *hbm_utilization_percent < kComputeBoundHbmUtilizationThreshold;
  }

  absl::StatusOr<std::optional<SmartSuggestion>> GenerateSuggestion(
      const SignalProvider& signal_provider) const override {
    SmartSuggestion suggestion;
    suggestion.set_rule_name("ComputeBoundRule");

    TF_ASSIGN_OR_RETURN(double hbm_utilization_percent,
                        signal_provider.GetHbmUtilization());
    TF_ASSIGN_OR_RETURN(double mxu_utilization_percent,
                        signal_provider.GetMxuUtilization());

    std::string suggestion_text = absl::StrCat(
        "<p>Your program is likely bottlenecked by <b>Compute Operations</b>: "
        "High MXU utilization of <b>",
        absl::StrFormat("%.1f", mxu_utilization_percent),
        "%</b> and low HBM Bandwidth utilization of <b>",
        absl::StrFormat("%.1f", hbm_utilization_percent),
        "%</b> indicates that the primary bottleneck is the raw processing "
        "power of the hardware. Please consider the following optimizations: "
        "</p>",
        "<ul>"
        "<li><b>Use Mixed Precision:</b> Using bfloat16 for computations and "
        "storing weights can significantly speed up matrix multiplications "
        "and reduce memory usage. Ensure that this does not negatively impact "
        "your model's convergence.</li>"
        "<li><b>Optimize Your Kernels:</b> If you are using custom operations, "
        "profile them to identify any inefficiencies. For standard "
        "operations, ensure you are using the latest version of your framework "
        "and libraries (e.g., CUDA, cuDNN for GPUs) which often include "
        "optimized kernels.</li>"
        "<li><b>Experiment with Batch Size:</b> While a large batch size can "
        "improve hardware utilization, an excessively large batch size might "
        "not always be optimal. Experiment with different batch sizes to find "
        "the sweet spot for your specific model and hardware.</li>"
        "<li><b>Re-evaluate the model architecture:</b> Consider if there are "
        "more computationally efficient alternatives to your current model or "
        "its components. For example, some layers are inherently more "
        "computationally expensive than others. Research and experiment with "
        "newer, more efficient architectures that can achieve similar "
        "performance with fewer floating-point operations (FLOPs)."
        "</ul>");

    suggestion.set_suggestion_text(suggestion_text);
    return suggestion;
  }
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // THIRD_PARTY_XPROF_CONVERT_SMART_SUGGESTION_COMPUTE_BOUND_RULE_H_
