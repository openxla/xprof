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

#ifndef THIRD_PARTY_XPROF_CONVERT_SMART_SUGGESTION_DEBUG_PRINT_RULE_H_
#define THIRD_PARTY_XPROF_CONVERT_SMART_SUGGESTION_DEBUG_PRINT_RULE_H_

#include <optional>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "xprof/convert/smart_suggestion/constants.h"
#include "xprof/convert/smart_suggestion/signal_provider.h"
#include "xprof/convert/smart_suggestion/smart_suggestion_rule.h"
#include "plugin/xprof/protobuf/smart_suggestion.pb.h"

namespace tensorflow {
namespace profiler {

constexpr char kDebugPrintOpName[] = "debug_print";

// Rule to detect debug print percentage bottleneck.
class DebugPrintRule : public SmartSuggestionRule {
 public:
  bool MeetsConditions(const SignalProvider& signal_provider) const override {
    auto host_stats =
        signal_provider.GetPerHostAvgEventTimePercent(kDebugPrintOpName);
    if (!host_stats.ok() || host_stats->empty()) {
      return false;
    }
    for (const auto& host_stat : *host_stats) {
      if (host_stat.second >= kDebugPrintBoundThresholdInPercent) {
        return true;
      }
    }
    return false;
  }

  // Generates suggestions if the debug print percentage is above the threshold.
  absl::StatusOr<std::optional<SmartSuggestion>> GenerateSuggestion(
      const SignalProvider& signal_provider) const override {
    SmartSuggestion suggestion;
    suggestion.set_rule_name("DebugPrintRule");
    // If MeetsConditions passed, GetPerHostAvgEventTimePercent is ok and has
    // hosts with fractions >= kDebugPrintBoundThresholdInPercent.
    absl::flat_hash_map<std::string, double> high_debug_print_hosts;
    double max_percent = 0.0;
    auto host_stats =
        signal_provider.GetPerHostAvgEventTimePercent(kDebugPrintOpName);
    if (host_stats.ok()) {
      for (const auto& host_stat : *host_stats) {
        if (host_stat.second >= kDebugPrintBoundThresholdInPercent) {
          high_debug_print_hosts.insert(host_stat);
          if (host_stat.second > max_percent) {
            max_percent = host_stat.second;
          }
        }
      }
    }
    std::string debug_print_hosts_list_html = "<ul>";
    for (const auto& [hostname, avg_percent] : high_debug_print_hosts) {
      absl::StrAppend(&debug_print_hosts_list_html, "<li>Host <b>", hostname,
                      "</b> average debug_print time fraction: <b>",
                      absl::StrFormat("%.1f", avg_percent), "%</b></li>");
    }
    absl::StrAppend(&debug_print_hosts_list_html, "</ul>");
    std::string debug_print_suggestion = absl::StrCat(
        "<li><b>Investigate Hosts with High Debug Print Time"
        ":</b> The following hosts have "
        "high debug_print time fraction compared to "
        "others:",
        debug_print_hosts_list_html, "</li>");

    auto display_name = absl::StrCat(kDebugPrintOpName);
    std::string suggestion_text = absl::StrCat(
        "<p>Your program is likely bottlenecked by <b>", display_name,
        "</b> operations: <b> up to ",
        absl::StrFormat("%.1f", max_percent),
        "% of step time</b> is spent on these operations on some hosts. This "
        "often indicates the debug print operations causes "
        "busy waiting for other workers in a "
        "distributed training setup. Please consider the following "
        "optimizations:</p>",
        "<ul>", debug_print_suggestion,
        "<li><b>Check for debug print usage:</b> Check for "
        "<code>jax.debug.print</code> calls in your code.</li></ul>");

    suggestion.set_suggestion_text(suggestion_text);
    return suggestion;
  }
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // THIRD_PARTY_XPROF_CONVERT_SMART_SUGGESTION_DEBUG_PRINT_RULE_H_
