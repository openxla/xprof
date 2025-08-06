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

#ifndef THIRD_PARTY_XPROF_CONVERT_SMART_SUGGESTION_HOST_PROCESSING_BOUND_RULE_H_
#define THIRD_PARTY_XPROF_CONVERT_SMART_SUGGESTION_HOST_PROCESSING_BOUND_RULE_H_

#include <optional>
#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "xla/tsl/platform/statusor.h"
#include "xprof/convert/smart_suggestion/input_bound_rule.h"
#include "xprof/convert/smart_suggestion/signal_provider.h"
#include "xprof/convert/smart_suggestion/smart_suggestion_rule.h"
#include "plugin/xprof/protobuf/smart_suggestion.pb.h"

namespace tensorflow {
namespace profiler {

// If the percentage of input time that is due to host processing is high than
// HostProcessingBoundThresholdInPercent, it is considered
// host processing-bound.
constexpr double kHostProcessingBoundThresholdInPercent = 50;

// Rule to detect if the input bottleneck is primarily due to host-side
// processing.
class HostProcessingBoundRule : public SmartSuggestionRule {
 protected:
  bool MeetsConditions(const SignalProvider& signal_provider) const override {
    InputBoundRule input_bound_rule;
    if (!input_bound_rule.MeetsConditions(signal_provider)) {
      return false;
    }

    absl::StatusOr<double> non_enqueue_percent_of_input =
        signal_provider.GetNonEnqueuePercentOfInput();
    if (!non_enqueue_percent_of_input.ok()) {
      return false;
    }

    return *non_enqueue_percent_of_input >=
           kHostProcessingBoundThresholdInPercent;
  }

  absl::StatusOr<std::optional<SmartSuggestion>> GenerateSuggestion(
      const SignalProvider& signal_provider) const override {
    SmartSuggestion suggestion;
    suggestion.set_rule_name("HostProcessingBoundRule");

    TF_ASSIGN_OR_RETURN(double input_percent_of_step_time,
                     signal_provider.GetInputPercentOfStepTime());
    TF_ASSIGN_OR_RETURN(double non_enqueue_percent_of_input,
                     signal_provider.GetNonEnqueuePercentOfInput());

    std::string suggestion_text = absl::StrCat(
        "Your program is likely bottlenecked by Host-side Processing "
        "in the input pipeline.\n",
        absl::StrFormat("%.1f", input_percent_of_step_time *
                                    non_enqueue_percent_of_input / 100),
        "% of the total step time is spent on host-side input data "
        "processing.\n",
        "Recommendations:\n",
        "- Optimize Data Reading: Ensure efficient file reading "
        "patterns. Use prefetching and interleaving to load data in parallel "
        "and in advance.\n",
        "- Parallelize Data Preprocessing: Utilize parallel "
        "processing "
        "techniques for CPU-bound preprocessing steps.\n",
        "- Offline Preprocessing: For static datasets, consider "
        "performing expensive preprocessing steps offline and saving the "
        "results.\n",
        "- Tuning Parameters: Experiment with buffer sizes, the "
        "number "
        "of parallel threads, and prefetch distances in your input pipeline "
        "to find the best settings.\n");

    suggestion.set_suggestion_text(suggestion_text);
    return suggestion;
  }
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // THIRD_PARTY_XPROF_CONVERT_SMART_SUGGESTION_HOST_PROCESSING_BOUND_RULE_H_
