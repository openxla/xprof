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

#ifndef THIRD_PARTY_XPROF_CONVERT_SMART_SUGGESTION_BARRIER_CORES_RULE_H_
#define THIRD_PARTY_XPROF_CONVERT_SMART_SUGGESTION_BARRIER_CORES_RULE_H_

#include <optional>
#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "xprof/convert/smart_suggestion/signal_provider.h"
#include "xprof/convert/smart_suggestion/smart_suggestion_rule.h"
#include "plugin/xprof/protobuf/smart_suggestion.pb.h"

namespace tensorflow {
namespace profiler {

// The name of the special op we are interested in, by default barrier-cores.
// TODO(zhuruiyang): We will need to update it to support other special ops with
// a vector of op strings.
constexpr char kSpecialOpName[] = "barrier-cores";

// Rule to detect high special op percentage bottleneck.
class BarrierCoresRule : public SmartSuggestionRule {
 public:
  bool MeetsConditions(const SignalProvider& signal_provider) const override {
    absl::StatusOr<double> special_op_percent =
        signal_provider.GetAvgEventTimePercent(kSpecialOpName);
    if (!special_op_percent.ok()) {
      return false;
    }
    return *special_op_percent >= kSpecialOpBoundThresholdInPercent;
  }

  // Generates a suggestion if the special op percentage is above the threshold.
  // TODO(zhuruiyang): We will need to update it to support other special ops
  // with a vector of op strings. Currently the suggestion text only supports
  // barrier-cores.
  absl::StatusOr<std::optional<SmartSuggestion>> GenerateSuggestion(
      const SignalProvider& signal_provider) const override {
    SmartSuggestion suggestion;
    suggestion.set_rule_name("BarrierCoresRule");
    // GetAvgEventTimePercent is guaranteed to return an ok status if
    // MeetsConditions returned true.
    double special_op_percent =
        signal_provider.GetAvgEventTimePercent(kSpecialOpName).value();
    auto display_name = absl::StrCat("TPU ", kSpecialOpName);
    // Identify potential straggler hosts.
    std::string workload_balance_suggestion;
    auto stragglers = signal_provider.GetHostStragglers(kSpecialOpName);
    if (stragglers.ok() && !stragglers->empty()) {
      std::string stragglers_list_html = "<ul>";
      for (const auto& straggler : *stragglers) {
        absl::StrAppend(&stragglers_list_html, "<li>Host <b>",
                        straggler.hostname,
                        "</b> average barrier-cores time fraction: <b>",
                        absl::StrFormat("%.1f", straggler.avg_fraction_percent),
                        "%</b></li>");
      }
      absl::StrAppend(&stragglers_list_html, "</ul>");
      workload_balance_suggestion = absl::StrCat(
          "<li><b>Investigate Stragglers:</b> The following hosts are "
          "identified as potential stragglers with "
          "significantly different barrier time fraction compared to "
          "others:",
          stragglers_list_html, "</li>");
    } else {
      workload_balance_suggestion =
          "<li><b>Investigate Workload Balance:</b> Check for stragglers, "
          "i.e., "
          "workers that are significantly slower than others. Uneven workloads "
          "can cause faster workers to wait at the barrier.</li>";
    }

    std::string suggestion_text = absl::StrCat(
        "<p>Your program is likely bottlenecked by <b>", display_name,
        "</b> operations: <b> an average of ",
        absl::StrFormat("%.1f", special_op_percent),
        "% of each step time</b> is spent on these operations. This "
        "often indicates a synchronization issue between workers in a "
        "distributed training setup. Please consider the following "
        "optimizations:</p>",
        "<ul>", workload_balance_suggestion,
        "<li><b>Optimize Collective Operations:</b> Operations like AllReduce "
        "involve synchronization. Ensure they are used efficiently. Check "
        "the size of data being communicated.</li>"
        "<li><b>Check Network:</b> Network latency or bandwidth can be a "
        "bottleneck for distributed operations, causing workers to wait "
        "longer at barriers.</li>"
        "<li><b>Improve Data Input Pipeline:</b> Ensure your data loading and "
        "preprocessing pipeline is efficient and balanced across all "
        "workers. A slow input pipeline on one worker can stall all "
        "others.</li>"
        "</ul>");

    suggestion.set_suggestion_text(suggestion_text);
    return suggestion;
  }
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // THIRD_PARTY_XPROF_CONVERT_SMART_SUGGESTION_BARRIER_CORES_RULE_H_
