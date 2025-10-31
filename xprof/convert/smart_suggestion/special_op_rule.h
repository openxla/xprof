#ifndef THIRD_PARTY_XPROF_CONVERT_SMART_SUGGESTION_SPECIAL_OP_RULE_H_
#define THIRD_PARTY_XPROF_CONVERT_SMART_SUGGESTION_SPECIAL_OP_RULE_H_

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

// The name of the special op we are interested in, by default barrier-cores.
// TODO(zhuruiyang): We will need to update it to support other special ops with
// a vector of op strings.
constexpr char kSpecialOpName[] = "barrier-cores";
// If the percentage of step time that is due to the special op is higher than
// this threshold, it is considered a bottleneck.
constexpr double kSpecialOpBoundThresholdInPercent = 10;

// Rule to detect high special op percentage bottleneck.
class SpecialOpRule : public SmartSuggestionRule {
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
    suggestion.set_rule_name("SpecialOpRule");
    TF_ASSIGN_OR_RETURN(double special_op_percent,
                        signal_provider.GetAvgEventTimePercent(kSpecialOpName));
    auto display_name = absl::StrCat("TPU ", kSpecialOpName);
    // TODO(zhuruiyang): The current suggestion text is hard-coded for just
    // barrier-cores. We will need to update it to support other special ops.
    std::string suggestion_text = absl::StrCat(
        "<p>Your program is likely bottlenecked by <b>", display_name,
        "</b> operations: <b> an average of ",
        absl::StrFormat("%.1f", special_op_percent),
        "% of each step time</b> is spent on these operations. This "
        "often indicates a synchronization issue between workers in a "
        "distributed training setup. Please consider the following "
        "optimizations:</p>",
        "<ul>"
        "<li><b>Investigate Workload Balance:</b> Check for stragglers, i.e., "
        "workers that are significantly slower than others. Uneven workloads "
        "can cause faster workers to wait at the barrier.</li>"
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

#endif  // THIRD_PARTY_XPROF_CONVERT_SMART_SUGGESTION_SPECIAL_OP_RULE_H_
