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

#ifndef XPROF_CONVERT_FLAT_OP_METRICS_TO_RECORD_H_
#define XPROF_CONVERT_FLAT_OP_METRICS_TO_RECORD_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/tsl/profiler/utils/math_utils.h"
#include "tsl/platform/protobuf.h"
#include "xprof/convert/op_metrics_to_record.h"
#include "plugin/xprof/protobuf/flat_op_metrics.pb.h"
#include "plugin/xprof/protobuf/hardware_types.pb.h"
#include "plugin/xprof/protobuf/op_metrics.pb.h"
#include "plugin/xprof/protobuf/op_stats.pb.h"

namespace xprof {

using ::tensorflow::profiler::FlatOpMetrics;
using ::tensorflow::profiler::FlatOpMetricsDb;
using ::tensorflow::profiler::OpMetrics;

inline const tsl::protobuf::RepeatedPtrField<FlatOpMetrics>& GetMetricsList(
    const FlatOpMetricsDb& db) {
  return db.op_instances();
}

inline absl::string_view GetMetricsName(const FlatOpMetrics& metrics) {
  return metrics.hlo_name();
}

inline absl::string_view GetMetricsName(const OpMetrics& metrics) {
  return metrics.name();
}

inline absl::string_view GetMetricsCategory(const FlatOpMetrics& metrics) {
  constexpr absl::string_view kUnknownLower = "unknown";
  constexpr absl::string_view kUnknownCap = "Unknown";
  if (!metrics.category().empty() && metrics.category() != kUnknownLower &&
      metrics.category() != kUnknownCap) {
    return metrics.category();
  }
  if (xla::StringToHloOpcode(metrics.hlo_name()).ok()) {
    return metrics.hlo_name();
  }
  return metrics.category().empty() ? kUnknownLower : metrics.category();
}

inline absl::string_view GetMetricsCategory(const OpMetrics& metrics) {
  constexpr absl::string_view kUnknownLower = "unknown";
  constexpr absl::string_view kUnknownCap = "Unknown";
  if (!metrics.category().empty() && metrics.category() != kUnknownLower &&
      metrics.category() != kUnknownCap) {
    return metrics.category();
  }
  if (xla::StringToHloOpcode(metrics.name()).ok()) {
    return metrics.name();
  }
  return metrics.category().empty() ? kUnknownLower : metrics.category();
}


// Returns a sorted vector of pointers to FlatOpMetrics in the given database.
// The returned pointers are only valid as long as `metrics_db` exists and is
// not modified.
std::vector<const FlatOpMetrics*> SortedOpMetricsDb(
    const FlatOpMetricsDb& metrics_db, int max_records = -1);


}  // namespace xprof

#endif  // XPROF_CONVERT_FLAT_OP_METRICS_TO_RECORD_H_
