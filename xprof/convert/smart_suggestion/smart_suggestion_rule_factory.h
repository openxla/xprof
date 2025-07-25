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

#ifndef THIRD_PARTY_XPROF_CONVERT_SMART_SUGGESTION_RULE_FACTORY_H_
#define THIRD_PARTY_XPROF_CONVERT_SMART_SUGGESTION_RULE_FACTORY_H_

#include <memory>
#include <vector>

#include "xprof/convert/smart_suggestion/smart_suggestion_rule.h"

namespace tensorflow {
namespace profiler {

// Factory class to manage smart suggestion rules.
class SmartSuggestionRuleFactory {
 public:
  // Registers a rule with the factory.
  template <typename RuleType>
  void Register() {
    rules_.push_back(std::make_unique<RuleType>());
  }

  const std::vector<std::unique_ptr<SmartSuggestionRule>>& GetAllRules() const {
    return rules_;
  }

 private:
  std::vector<std::unique_ptr<SmartSuggestionRule>> rules_;
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // THIRD_PARTY_XPROF_CONVERT_SMART_SUGGESTION_RULE_FACTORY_H_
