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

#ifndef THIRD_PARTY_XPROF_CONVERT_SMART_SUGGESTION_HOST_CPU_BOUND_RULE_H_
#define THIRD_PARTY_XPROF_CONVERT_SMART_SUGGESTION_HOST_CPU_BOUND_RULE_H_

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

constexpr char kInfeedOpName[] = "infeed";
constexpr char kEnqueueDeviceOpName[] = "EnqueueDevice";

// Rule to detect host cpu bound bottlenecks (combining TPU infeed and
// CPU EnqueueDevice).
class HostCPUBoundRule : public SmartSuggestionRule {
 public:
  bool MeetsConditions(const SignalProvider& signal_provider) const override {
    auto infeed_stats =
        signal_provider.GetPerHostAvgEventTimePercent(kInfeedOpName);
    auto enqueue_stats =
        signal_provider.GetPerHostAvgEventTimePercent(kEnqueueDeviceOpName);

    if (!infeed_stats.ok() || infeed_stats->empty() || !enqueue_stats.ok() ||
        enqueue_stats->empty()) {
      return false;
    }

    bool meets_infeed = false;
    for (const auto& host_stat : *infeed_stats) {
      if (host_stat.second >= kInfeedOpBoundThresholdInPercent) {
        meets_infeed = true;
        break;
      }
    }

    bool meets_enqueue = false;
    for (const auto& host_stat : *enqueue_stats) {
      if (host_stat.second >= kEnqueueDeviceBoundThresholdInPercent) {
        meets_enqueue = true;
        break;
      }
    }

    return meets_infeed && meets_enqueue;
  }

  absl::StatusOr<std::optional<SmartSuggestion>> GenerateSuggestion(
      const SignalProvider& signal_provider) const override {
    SmartSuggestion suggestion;
    suggestion.set_rule_name("HostCPUBoundRule");

    absl::flat_hash_map<std::string, double> high_infeed_hosts;
    double max_infeed_percent = 0.0;
    auto infeed_stats =
        signal_provider.GetPerHostAvgEventTimePercent(kInfeedOpName);
    if (infeed_stats.ok()) {
      for (const auto& host_stat : *infeed_stats) {
        if (host_stat.second >= kInfeedOpBoundThresholdInPercent) {
          high_infeed_hosts.insert(host_stat);
          if (host_stat.second > max_infeed_percent) {
            max_infeed_percent = host_stat.second;
          }
        }
      }
    }

    absl::flat_hash_map<std::string, double> high_enqueue_hosts;
    double max_enqueue_percent = 0.0;
    auto enqueue_stats =
        signal_provider.GetPerHostAvgEventTimePercent(kEnqueueDeviceOpName);
    if (enqueue_stats.ok()) {
      for (const auto& host_stat : *enqueue_stats) {
        if (host_stat.second >= kEnqueueDeviceBoundThresholdInPercent) {
          high_enqueue_hosts.insert(host_stat);
          if (host_stat.second > max_enqueue_percent) {
            max_enqueue_percent = host_stat.second;
          }
        }
      }
    }

    std::string infeed_hosts_list_html = "<ul>";
    for (const auto& [hostname, avg_percent] : high_infeed_hosts) {
      absl::StrAppend(&infeed_hosts_list_html, "<li>Host <b>", hostname,
                      "</b> average infeed time fraction: <b>",
                      absl::StrFormat("%.1f", avg_percent), "%</b></li>");
    }
    absl::StrAppend(&infeed_hosts_list_html, "</ul>");

    std::string enqueue_hosts_list_html = "<ul>";
    for (const auto& [hostname, avg_percent] : high_enqueue_hosts) {
      absl::StrAppend(&enqueue_hosts_list_html, "<li>Host <b>", hostname,
                      "</b> average enqueue_device time fraction: <b>",
                      absl::StrFormat("%.1f", avg_percent), "%</b></li>");
    }
    absl::StrAppend(&enqueue_hosts_list_html, "</ul>");

    std::string suggestion_text = absl::StrCat(
        "<p>Your program is likely bottlenecked by EnqueueDevice ops on host "
        "CPU, an average of up to <b>",
        absl::StrFormat("%.1f%%", max_enqueue_percent),
        "</b> of time is spent on these operations. In addition, infeed op "
        "on TensorCore consumes an average of up to <b>",
        absl::StrFormat("%.1f%%", max_infeed_percent),
        "</b> of step time correlated with the host CPU bottleneck.</p>",
        "<p>Please consider the following optimizations:</p><ul>");


    // absl::StrAppend(&suggestion_text,
    //     "<li><b>Increase Host GCU:</b> provide more CPU resources to your "
    //     "TPU worker tasks.</li>");

    absl::StrAppend(
        &suggestion_text,
        "<li><b>(SparseCore)</b> Consider to improve input pipeline and "
        "apply sparsecore-specific offloads, e.g., embedding data formatting "
        "offload, gather offload.</li>");


    // absl::StrAppend(&suggestion_text,
    //     "<li><b>Overlap computation:</b> Confirm that "
    //     "<code>pipeline_execution_with_tensor_core</code> is set to "
    //     "<code>True</code> in your embedding layer configuration. This "
    //     "allows the TensorCore to potentially overlap its computation with "
    //     "the SparseCore operations for the next step.</li>");

    absl::StrAppend(
        &suggestion_text,
        "<li><b>Host Breakdown (Infeed):</b> The following hosts have high "
        "device infeed time fractions:",
        infeed_hosts_list_html,
        "</li><li><b>Host Breakdown (EnqueueDevice):</b> The following hosts "
        "have high host enqueue time fractions:",
        enqueue_hosts_list_html, "</li></ul>");

    suggestion.set_suggestion_text(suggestion_text);
    return suggestion;
  }
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // THIRD_PARTY_XPROF_CONVERT_SMART_SUGGESTION_HOST_CPU_BOUND_RULE_H_
