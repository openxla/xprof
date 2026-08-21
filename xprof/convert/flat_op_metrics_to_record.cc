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

#include "xprof/convert/flat_op_metrics_to_record.h"

#include <vector>

#include "absl/algorithm/container.h"

namespace xprof {

using ::tensorflow::profiler::FlatOpMetrics;
using ::tensorflow::profiler::FlatOpMetricsDb;


std::vector<const FlatOpMetrics*> SortedOpMetricsDb(
    const FlatOpMetricsDb& metrics_db, int max_records) {
  std::vector<const FlatOpMetrics*> result;
  // Exclude SparseCore ops to maintain parity with legacy OpMetricsDb behavior.
  // SparseCore operations are typically aggregated as children of TensorCore
  // operations and are not listed as independent top-level records in this
  // view.
  result.reserve(metrics_db.op_instances_size());
  for (const FlatOpMetrics& metrics : metrics_db.op_instances()) {
    if (metrics.core_type() == FlatOpMetrics::SPARSE_CORE) continue;
    result.push_back(&metrics);
  }

  auto comp = [](const FlatOpMetrics* a, const FlatOpMetrics* b) {
    if (a->self_time_ps() != b->self_time_ps()) {
      return a->self_time_ps() > b->self_time_ps();
    }
    return a->hlo_name() > b->hlo_name();
  };
  int result_size = result.size();
  if (max_records > 0 && result_size > max_records) {
    absl::c_partial_sort(result, result.begin() + max_records, comp);
    result.resize(max_records);
  } else {
    absl::c_sort(result, comp);
  }
  return result;
}

}  // namespace xprof
