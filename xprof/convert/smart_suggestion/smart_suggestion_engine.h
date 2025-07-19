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

#ifndef THIRD_PARTY_XPROF_CONVERT_SMART_SUGGESTION_SMART_SUGGESTION_ENGINE_H_
#define THIRD_PARTY_XPROF_CONVERT_SMART_SUGGESTION_SMART_SUGGESTION_ENGINE_H_

#include "absl/status/statusor.h"
#include "xprof/convert/smart_suggestion/signal_provider.h"
#include "xprof/convert/smart_suggestion/smart_suggestion_rule_factory.h"
#include "plugin/xprof/protobuf/smart_suggestion.pb.h"

namespace tensorflow {
namespace profiler {

// Engine to generate smart suggestions.
class SmartSuggestionEngine {
 public:
  explicit SmartSuggestionEngine() = default;

  // Generates smart suggestions based on the provided signal provider and rule
  // factory.
  absl::StatusOr<SmartSuggestionReport> Run(
      const SignalProvider& signal_provider,
      const SmartSuggestionRuleFactory& rule_factory) const;
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // THIRD_PARTY_XPROF_CONVERT_SMART_SUGGESTION_SMART_SUGGESTION_ENGINE_H_
