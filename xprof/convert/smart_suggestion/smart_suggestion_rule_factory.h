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

#include <functional>
#include <memory>
#include <vector>

#include "xprof/convert/smart_suggestion/smart_suggestion_rule.h"

namespace tensorflow {
namespace profiler {

// Factory class to manage smart suggestion rules.
class SmartSuggestionRuleFactory {
 public:
  // Registers a rule with the factory by storing a function that can create it.
  template <typename RuleType>
  void Register() {
    rule_creators_.push_back([] { return std::make_unique<RuleType>(); });
  }

  // Returns a vector of new instances of all registered rules.
  std::vector<std::unique_ptr<SmartSuggestionRule>> CreateAllRules() const {
    std::vector<std::unique_ptr<SmartSuggestionRule>> rules;
    rules.reserve(rule_creators_.size());
    for (const auto& creator : rule_creators_) {
      rules.push_back(creator());
    }
    return rules;
  }

 private:
  // Stores functions that can create instances of the rules.
  std::vector<std::function<std::unique_ptr<SmartSuggestionRule>()>>
      rule_creators_;
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // THIRD_PARTY_XPROF_CONVERT_SMART_SUGGESTION_RULE_FACTORY_H_
