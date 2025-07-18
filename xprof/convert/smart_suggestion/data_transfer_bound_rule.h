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

#ifndef THIRD_PARTY_XPROF_CONVERT_SMART_SUGGESTION_DATA_TRANSFER_BOUND_RULE_H_
#define THIRD_PARTY_XPROF_CONVERT_SMART_SUGGESTION_DATA_TRANSFER_BOUND_RULE_H_

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

// If the percentage of input time that is due to data transfer is high than
// DataTransferBoundThresholdInPercent, it is considered
// data transfer-bound.
constexpr double kDataTransferBoundThresholdInPercent = 30;

// Rule to detect if the input bottleneck is primarily due to data transfer.
class DataTransferBoundRule : public SmartSuggestionRule {
 protected:
  bool MeetsConditions(const SignalProvider& signal_provider) const override {
    InputBoundRule input_bound_rule;
    if (!input_bound_rule.MeetsConditions(signal_provider)) {
      return false;
    }

    absl::StatusOr<double> enqueue_percent_of_input =
        signal_provider.GetEnqueuePercentOfInput();
    if (!enqueue_percent_of_input.ok()) {
      return false;
    }

    return *enqueue_percent_of_input >= kDataTransferBoundThresholdInPercent;
  }

  absl::StatusOr<std::optional<SmartSuggestion>> GenerateSuggestion(
      const SignalProvider& signal_provider) const override {
    SmartSuggestion suggestion;
    suggestion.set_rule_name("DataTransferBoundRule");

    TF_ASSIGN_OR_RETURN(double enqueue_percent_of_input,
                     signal_provider.GetEnqueuePercentOfInput());

    std::string suggestion_text = absl::StrCat(
        "<p>Your program is likely bottlenecked by <b>data transfer</b> "
        "between Host and Device.</p>",
        "<p>", absl::StrFormat("%.1f", enqueue_percent_of_input),
        "% of the total input time is spent on enqueuing data to the "
        "device.</p>",
        "<ul>"
        "<li><b>Combine small data chunks:</b> Transferring many small "
        "chunks of data can be inefficient. Try to batch them into fewer, "
        "larger transfers.</li>"
        "<li><b>Check transfer size:</b> Ensure the size of data being "
        "transferred in each batch is optimal for the hardware.</li>"
        "<li><b>Use prefetching:</b> Overlap data transfer with "
        "computation.</li>"
        "</ul>");

    suggestion.set_suggestion_text(suggestion_text);
    return suggestion;
  }
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // THIRD_PARTY_XPROF_CONVERT_SMART_SUGGESTION_DATA_TRANSFER_BOUND_RULE_H_
