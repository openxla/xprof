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

#ifndef THIRD_PARTY_XPROF_CONVERT_SMART_SUGGESTION_INFEED_RULE_H_
#define THIRD_PARTY_XPROF_CONVERT_SMART_SUGGESTION_INFEED_RULE_H_

#include <algorithm>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "xprof/convert/smart_suggestion/constants.h"
#include "xprof/convert/smart_suggestion/signal_provider.h"
#include "xprof/convert/smart_suggestion/smart_suggestion_rule.h"
#include "plugin/xprof/protobuf/smart_suggestion.pb.h"

namespace tensorflow {
namespace profiler {

constexpr char kInfeedOpName[] = "infeed";

// Rule to detect infeed percentage bottleneck.
class InfeedRule : public SmartSuggestionRule {
 public:
  bool MeetsConditions(const SignalProvider& signal_provider) const override {
    auto host_stats =
        signal_provider.GetPerHostAvgEventTimePercent(kInfeedOpName);
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

  // Generates suggestions if the infeed percentage is above the threshold.
  absl::StatusOr<std::optional<SmartSuggestion>> GenerateSuggestion(
      const SignalProvider& signal_provider) const override {
    SmartSuggestion suggestion;
    suggestion.set_rule_name("InfeedRule");
    // If MeetsConditions passed, GetPerHostAvgEventTimePercent is ok and has
    // hosts with fractions >= kDebugPrintBoundThresholdInPercent.
    absl::flat_hash_map<std::string, double> high_infeed_hosts;
    double max_percent = 0.0;
    auto host_stats =
        signal_provider.GetPerHostAvgEventTimePercent(kInfeedOpName);
    if (host_stats.ok()) {
      for (const auto& host_stat : *host_stats) {
        if (host_stat.second >= kDebugPrintBoundThresholdInPercent) {
          high_infeed_hosts.insert(host_stat);
          if (host_stat.second > max_percent) {
            max_percent = host_stat.second;
          }
        }
      }
    }
    std::string infeed_hosts_list_html;
    std::string infeed_suggestion;
    if (high_infeed_hosts.size() > 5) {
      infeed_hosts_list_html = absl::StrCat(
          " <b>", high_infeed_hosts.size(),
          " hosts</b> have an average infeed time fraction above <b>",
          absl::StrFormat("%.1f", kDebugPrintBoundThresholdInPercent),
          "%</b>.");
      infeed_suggestion =
          absl::StrCat("<li><b>Investigate Hosts with High Infeed Time:",
                       infeed_hosts_list_html, "</li>");
    } else {
      std::vector<std::string> host_entries;
      for (const auto& [hostname, avg_percent] : high_infeed_hosts) {
        host_entries.push_back(absl::StrCat(
            "Host <b>", hostname, "</b> ",
            "average infeed time fraction: <b>",
            absl::StrFormat("%.1f", avg_percent), "%</b>"));
      }
      std::sort(host_entries.begin(), host_entries.end());
      infeed_hosts_list_html = absl::StrJoin(host_entries, ", ");
      infeed_suggestion = absl::StrCat(
          "<li><b>Investigate Hosts with High Infeed Time"
          ":</b> The following hosts have high infeed time fraction: ",
          infeed_hosts_list_html, "</li>");
    }

    auto display_name = absl::StrCat(kInfeedOpName);
    std::string suggestion_text = absl::StrCat(
        "<p>Your program is likely bottlenecked by <b>", display_name,
        "</b> operations: <b> up to ",
        absl::StrFormat("%.1f", max_percent),
        "% of step time</b> is spent on these operations on some hosts. "
        "Please consider the following "
        "optimizations:</p>",
        "<ul>", infeed_suggestion);
    absl::StrAppend(&suggestion_text, "</ul>");

    suggestion.set_suggestion_text(suggestion_text);
    return suggestion;
  }
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // THIRD_PARTY_XPROF_CONVERT_SMART_SUGGESTION_INFEED_RULE_H_
