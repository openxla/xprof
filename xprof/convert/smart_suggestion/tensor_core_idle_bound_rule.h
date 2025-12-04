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

#ifndef THIRD_PARTY_XPROF_CONVERT_SMART_SUGGESTION_TENSOR_CORE_IDLE_BOUND_RULE_H_
#define THIRD_PARTY_XPROF_CONVERT_SMART_SUGGESTION_TENSOR_CORE_IDLE_BOUND_RULE_H_

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


// Rule to detect high device idle time bottleneck.
class TensorCoreIdleBoundRule : public SmartSuggestionRule {
 public:
  bool MeetsConditions(const SignalProvider& signal_provider) const override {
    if (!signal_provider.IsLatencyBound()) {
      return false;
    }

    absl::StatusOr<double> tensor_core_idle_time_percent =
        signal_provider.GetTensorCoreIdleTimePercent();
    if (!tensor_core_idle_time_percent.ok()) {
      return false;
    }

    return *tensor_core_idle_time_percent >
           kTensorCoreIdleTimeThresholdInPercent;
  }

  absl::StatusOr<std::optional<SmartSuggestion>> GenerateSuggestion(
      const SignalProvider& signal_provider) const override {
    SmartSuggestion suggestion;
    suggestion.set_rule_name("TensorCoreIdleBoundRule");

    TF_ASSIGN_OR_RETURN(double tensor_core_idle_time_percent,
                        signal_provider.GetTensorCoreIdleTimePercent());
    std::string suggestion_text = absl::StrCat(
        "<p>Your program is likely bottlenecked by <b>TensorCore Idle Time</b>:"
        " High TensorCore idle time percentage of <b>",
        absl::StrFormat("%.1f", tensor_core_idle_time_percent),
        "%</b> indicates that TensorCores are spending a significant amount of "
        "time waiting, not working. Please consider the following "
        "optimizations: </p>",
        "<ul>"
        "<li><b>Reduce Kernel Launch Overhead:</b> Batch small operations into "
        "larger ones to reduce the number of kernel launches.</li>"
        "<li><b>Minimize Python Overhead:</b> Enclose more operations within "
        "compiled graphs or functions to reduce Python interpreter overhead "
        "between steps.</li>"
        "<li><b>Avoid Small CPU Ops:</b> Shift small, frequent operations from "
        "the CPU to the device if possible.</li>"
        "<li><b>Use Asynchronous Operations:</b> Employ asynchronous execution "
        "for tasks like checkpointing or metric logging to prevent blocking "
        "device execution.</li>"
        "</ul>");

    suggestion.set_suggestion_text(suggestion_text);
    return suggestion;
  }
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // THIRD_PARTY_XPROF_CONVERT_SMART_SUGGESTION_TENSOR_CORE_IDLE_BOUND_RULE_H_
