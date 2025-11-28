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

#ifndef THIRD_PARTY_XPROF_CONVERT_SMART_SUGGESTION_ALL_RULES_H_
#define THIRD_PARTY_XPROF_CONVERT_SMART_SUGGESTION_ALL_RULES_H_

#include "xprof/convert/smart_suggestion/barrier_cores_rule.h"
#include "xprof/convert/smart_suggestion/collective_bound_rule.h"
#include "xprof/convert/smart_suggestion/compute_bound_rule.h"
#include "xprof/convert/smart_suggestion/data_transfer_bound_rule.h"
#include "xprof/convert/smart_suggestion/host_processing_bound_rule.h"
#include "xprof/convert/smart_suggestion/memory_bound_rule.h"
#include "xprof/convert/smart_suggestion/smart_suggestion_rule_factory.h"
#include "xprof/convert/smart_suggestion/tensor_core_idle_bound_rule.h"

namespace tensorflow {
namespace profiler {

// Registers all smart suggestion rules.
inline void RegisterAllRules(SmartSuggestionRuleFactory* f) {
  // go/keep-sorted start
  f->Register<BarrierCoresRule>();
  f->Register<CollectiveBoundRule>();
  f->Register<ComputeBoundRule>();
  f->Register<DataTransferBoundRule>();
  f->Register<HostProcessingBoundRule>();
  f->Register<MemoryBoundRule>();
  f->Register<TensorCoreIdleBoundRule>();
  // go/keep-sorted end
}

}  // namespace profiler
}  // namespace tensorflow

#endif  // THIRD_PARTY_XPROF_CONVERT_SMART_SUGGESTION_ALL_RULES_H_
