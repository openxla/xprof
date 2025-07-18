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
#include "xprof/convert/smart_suggestion/smart_suggestion_engine.h"

#include <optional>

#include "absl/status/statusor.h"
#include "xla/tsl/platform/statusor.h"
#include "xprof/convert/smart_suggestion/signal_provider.h"
#include "xprof/convert/smart_suggestion/smart_suggestion_rule.h"
#include "xprof/convert/smart_suggestion/smart_suggestion_rule_factory.h"
#include "plugin/xprof/protobuf/smart_suggestion.pb.h"

namespace tensorflow {
namespace profiler {

absl::StatusOr<SmartSuggestionReport> SmartSuggestionEngine::Run(
    const SignalProvider& signal_provider,
    const SmartSuggestionRuleFactory& rule_factory) const {
  SmartSuggestionReport report;

  const auto& rules = rule_factory.CreateAllRules();
  for (const auto& rule : rules) {
    TF_ASSIGN_OR_RETURN(std::optional<SmartSuggestion> suggestion,
                     rule->Apply(signal_provider));
    if (suggestion.has_value()) {
      *report.add_suggestions() = *suggestion;
    }
  }
  return report;
}

}  // namespace profiler
}  // namespace tensorflow
