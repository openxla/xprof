/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef THIRD_PARTY_XPROF_UTILS_CUSTOM_CALL_COST_ESTIMATOR_H_
#define THIRD_PARTY_XPROF_UTILS_CUSTOM_CALL_COST_ESTIMATOR_H_

#include <cstdint>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "xla/hlo/ir/hlo_instruction.h"

namespace CustomCallCostEstimator {

// Represents the computational cost of an operation as well as group of
// operations.
struct OperationCost {
  uint64_t flops = 0;
  uint64_t bytes_consumed = 0;
  uint64_t hbm_rw_bytes = 0;
};

void calculateCustomCallCost(
    const xla::HloInstruction& hlo_instruction,
    absl::flat_hash_map<std::string, OperationCost>& custom_call_block_costs,
    OperationCost& custom_call_cost_);

}  // namespace CustomCallCostEstimator

#endif  // THIRD_PARTY_XPROF_UTILS_CUSTOM_CALL_COST_ESTIMATOR_H_
