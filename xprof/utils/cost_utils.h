/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#ifndef XPROF_UTILS_COST_UTILS_H_
#define XPROF_UTILS_COST_UTILS_H_

#include <algorithm>
#include <cstdint>
#include <string>

#include "absl/container/flat_hash_set.h"
#include "xla/tsl/profiler/utils/xplane_visitor.h"
#include "tensorflow/core/grappler/costs/cost_estimator.h"
#include "tensorflow/core/grappler/costs/op_level_cost_estimator.h"
#include "xla/tsl/platform/types.h"

namespace tensorflow {
namespace profiler {

using ::tsl::profiler::XEventVisitor;

// Returns 0 in case a cost returned by HloCostAnalysis is -1.
// HloCostAnalysis returns -1 if the instruction does not have a cost.
inline int64_t ValidHloCost(int64_t cost) { return std::max<int64_t>(0, cost); }

// This is a wrapper of tensorflow::grappler::OpLevelCostEstimator and use
// tracing time information to estimate the roof line stats for each traced
// tensorflow op.
class TfOpRoofLineCostEstimator
    : public tensorflow::grappler::OpLevelCostEstimator {
 public:
  TfOpRoofLineCostEstimator() = default;
  ~TfOpRoofLineCostEstimator() override;

  grappler::DeviceInfo GetDeviceInfo(
      const DeviceProperties& device) const override;

  struct OpRoofLineStats {
    uint64 flops = 0LL;
    uint64 bytes_accessed = 0LL;
    bool inaccurate = false;
  };
  OpRoofLineStats Predict(const XEventVisitor& event);

 private:
  absl::flat_hash_set<std::string>
      unsupported_ops_;  // summary for unsupported ops.

  TfOpRoofLineCostEstimator(const TfOpRoofLineCostEstimator&) = delete;
  void operator=(const TfOpRoofLineCostEstimator&) = delete;
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // XPROF_UTILS_COST_UTILS_H_
