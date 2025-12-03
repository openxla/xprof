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

#ifndef THIRD_PARTY_XPROF_CONVERT_SMART_SUGGESTION_DATA_SHUFFLE_BOUND_RULE_H_
#define THIRD_PARTY_XPROF_CONVERT_SMART_SUGGESTION_DATA_SHUFFLE_BOUND_RULE_H_

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

// Rule to detect high data shuffle percentage bottleneck.
class DataShuffleBoundRule : public SmartSuggestionRule {
 public:
  bool MeetsConditions(const SignalProvider& signal_provider) const override {
    absl::StatusOr<double> avg_data_shuffle_percent =
        signal_provider.GetAvgDataShuffleTimePercent();
    if (!avg_data_shuffle_percent.ok()) {
      return false;
    }
    return *avg_data_shuffle_percent >= kDataShuffleBoundThresholdInPercent;
  }

  absl::StatusOr<std::optional<SmartSuggestion>> GenerateSuggestion(
      const SignalProvider& signal_provider) const override {
    SmartSuggestion suggestion;
    suggestion.set_rule_name("DataShuffleBoundRule");
    TF_ASSIGN_OR_RETURN(double avg_data_shuffle_percent,
                        signal_provider.GetAvgDataShuffleTimePercent());
    std::string suggestion_text = absl::StrCat(
        "<p>Your program is likely bottlenecked by <b>Data Shuffle operations "
        "(e.g., sort, gather, scatter)</b>. An average of <b>",
        absl::StrFormat("%.1f%%", avg_data_shuffle_percent),
        "</b> of each step is spent on these operations. Please consider the "
        "following optimizations:</p>"
        "<ul>"
        "<li><b>Optimize Gather, Scatter Operations:</b> If gather, "
        "scatter operations are the bottleneck, review their "
        "implementation. Ensure that the dimensions of tensors involved in "
        "gather and scatter operations are multiples of 128 (common for most "
        "TPUs).</li>"
        "</ul>");
    suggestion.set_suggestion_text(suggestion_text);
    return suggestion;
  }
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // THIRD_PARTY_XPROF_CONVERT_SMART_SUGGESTION_DATA_SHUFFLE_BOUND_RULE_H_
