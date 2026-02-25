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

#ifndef THIRD_PARTY_XPROF_CONVERT_SMART_SUGGESTION_INPUT_BOUND_RULE_H_
#define THIRD_PARTY_XPROF_CONVERT_SMART_SUGGESTION_INPUT_BOUND_RULE_H_

#include <cmath>
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

// Rule to detect if the input bottleneck is primarily due to input
// processing.
class InputBoundRule : public SmartSuggestionRule {
 protected:
  bool MeetsConditions(const SignalProvider& signal_provider) const override {
    if (!signal_provider.IsInputBound()) {
      return false;
    }

    absl::StatusOr<double> non_enqueue_percent_of_input =
        signal_provider.GetNonEnqueuePercentOfInput();
    if (!non_enqueue_percent_of_input.ok()) {
      return false;
    }
    absl::StatusOr<double> enqueue_percent_of_input =
        signal_provider.GetEnqueuePercentOfInput();
    if (!enqueue_percent_of_input.ok()) {
      return false;
    }
    // This occurs when infeed ops are present, but the host input pipeline
    // events cannot be correctly categorized likely due to a custom input
    // pipeline framework.
    return std::abs(*non_enqueue_percent_of_input) < kZeroEpsilon &&
           std::abs(*enqueue_percent_of_input) < kZeroEpsilon;
  }

  absl::StatusOr<std::optional<SmartSuggestion>> GenerateSuggestion(
      const SignalProvider& signal_provider) const override {
    SmartSuggestion suggestion;
    suggestion.set_rule_name("InputBoundRule");

    TF_ASSIGN_OR_RETURN(double input_percent_of_step_time,
                        signal_provider.GetInputPercentOfStepTime());

    // TODO(pennyhui): Switch from HTML to supporting breakdowns in
    // SmartSuggestion proto, which will be easy to render in the frontend.
    std::string suggestion_text = absl::StrCat(
        "<p>Your program is likely <b>input-bound</b> "
        ": ",
        "<b>", absl::StrFormat("%.1f", input_percent_of_step_time),
        "% of step time</b> is spent on input operations."
        " Please consider the following documentations to optimize "
        "your input pipeline:</p><ul>");
    absl::StrAppend(&suggestion_text, "</ul>");

    suggestion.set_suggestion_text(suggestion_text);
    return suggestion;
  }
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // THIRD_PARTY_XPROF_CONVERT_SMART_SUGGESTION_INPUT_BOUND_RULE_H_
