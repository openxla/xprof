/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#ifndef THIRD_PARTY_XPROF_CONVERT_FLAT_OP_METRICS_DB_COMBINER_H_
#define THIRD_PARTY_XPROF_CONVERT_FLAT_OP_METRICS_DB_COMBINER_H_

#include "tsl/platform/protobuf.h"
#include "plugin/xprof/protobuf/flat_op_metrics.pb.h"
#include "xprof/utils/flat_op_metrics_db_utils.h"

namespace tensorflow {
namespace profiler {

// Helper to combine flat op metrics databases.
class FlatOpMetricsDbCombiner : public FlatOpMetricsDbBuilder {
 public:
  explicit FlatOpMetricsDbCombiner(FlatOpMetricsDb* dst)
      : FlatOpMetricsDbBuilder(dst) {}

  // Combine the FlatOpMetrics in FlatOpMetricsDb <src> to current
  // FlatOpMetricsDbCombiner. If <update_num_cores> is set to true, update the
  // FlatOpMetrics.num_cores to calculate the number of cores a certain op
  // occurs.
  void Combine(const FlatOpMetricsDb& src, bool update_num_cores = true);

 private:
  // Copies FlatOpMetrics metadata (e.g., category, provenance) from src to dst.
  static void CopyFlatOpMetricsMetadata(const FlatOpMetrics& src,
                                        FlatOpMetrics* dst);

  // Combines FlatOpMetrics data from src into dst.
  static void CombineFlatOpMetrics(const FlatOpMetrics& src, FlatOpMetrics* dst,
                                   bool update_num_cores);

  // Combines the memory access breakdown for FlatOpMetrics.
  static void CombineMemoryAccessedBreakdown(
      const tsl::protobuf::RepeatedPtrField<FlatOpMetrics::MemoryAccessed>& src,
      tsl::protobuf::RepeatedPtrField<FlatOpMetrics::MemoryAccessed>* dst);
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // THIRD_PARTY_XPROF_CONVERT_FLAT_OP_METRICS_DB_COMBINER_H_
