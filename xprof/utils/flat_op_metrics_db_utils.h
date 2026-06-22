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

#ifndef XPROF_UTILS_FLAT_OP_METRICS_DB_UTILS_H_
#define XPROF_UTILS_FLAT_OP_METRICS_DB_UTILS_H_

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/macros.h"
#include "xla/tsl/profiler/utils/xplane_visitor.h"
#include "plugin/xprof/protobuf/flat_op_metrics.pb.h"
#include "plugin/xprof/protobuf/op_metrics.pb.h"

namespace tensorflow {
namespace profiler {

TF_CONST_INIT extern const absl::string_view kIdle;

// Helps build a flat op metrics database (borrowed).
// Enables fast lookup of existing ops and prevents the creation of duplicate
// ops when filling the FlatOpMetricsDb proto.
class FlatOpMetricsDbBuilder {
 public:
  // Create with a borrowed flat op database.
  explicit FlatOpMetricsDbBuilder(FlatOpMetricsDb* db);

 protected:
  // Looks up the given hlo_module_id and hlo_name.
  // If it is already in the database,
  // return its FlatOpMetrics; otherwise, insert a new one.
  FlatOpMetrics* LookupOrInsertNewFlatOpMetrics(uint64_t hlo_module_id,
                                                absl::string_view hlo_name);

  FlatOpMetricsDb* db() { return db_; }

 private:
  FlatOpMetricsDb* db_;
  // Maps op (hlo_module_id, hlo_name) to the corresponding metrics in the flat
  // op database.
  absl::flat_hash_map<
      uint64_t /*hlo_module_id*/,
      absl::flat_hash_map<std::string /*hlo_name*/, FlatOpMetrics*>>
      op_metrics_map_;
};

// Helps build a flat op metrics database (borrowed) from XEvents.
class XEventsFlatOpMetricsDbBuilder {
 public:
  struct OpKey {
    std::optional<uint64_t> program_id;
    std::optional<uint64_t> symbol_id;
  };

  // Constructs a FlatOpMetrics from the provided XEventVisitor.
  static FlatOpMetrics FromXEvent(const tsl::profiler::XEventVisitor& xevent);

  // Returns the OpKey (Program ID and Symbol ID) for the provided
  // XEventVisitor.
  // This key is used to group and aggregate FlatOpMetrics in the database.
  static OpKey GetFlatOpKeyFromXEvent(
      const tsl::profiler::XEventVisitor& event);

  // Add FlatOpMetrics from XEventVisitor.
  void AddOpMetric(const tsl::profiler::XEventVisitor& xevent);

  // Add a FlatOpMetrics to the builder based on the provided key.
  void AddOpMetric(const FlatOpMetrics& op_metrics, const OpKey& key);

  // Finalize FlatOpMetricsDb and add total time and Idle op.
  FlatOpMetricsDb Finalize(uint64_t total_time);

  // Finalize FlatOpMetricsDb without setting total time or adding Idle op.
  FlatOpMetricsDb Finalize();

  // Finalize FlatOpMetricsDb with Topologically sorted (Parent before Child)
  // operations.
  FlatOpMetricsDb FinalizeSorted();

  // Finalize FlatOpMetricsDb with Topologically sorted (Parent before Child)
  // operations and add total time.
  FlatOpMetricsDb FinalizeSorted(uint64_t total_time);

 private:
  using FlatOpMetricBySymbol =
      absl::flat_hash_map</*symbol_id=*/uint64_t, FlatOpMetrics>;
  absl::flat_hash_map</*program_id=*/uint64_t, FlatOpMetricBySymbol>
      flat_op_metric_;
};

// Returns true if the given metrics represents idle time.
inline bool IsIdleOp(const FlatOpMetrics& metrics) {
  return metrics.category() == kIdle;
}

// Returns the time spent in children (nested) ops.
inline uint64_t ChildrenTimePs(const FlatOpMetrics& metrics) {
  return metrics.time_ps() - metrics.self_time_ps();
}

// A perfectly deterministic 64-bit hash function based on FNV-1a
inline uint64_t StableHash64(absl::string_view str) {
  uint64_t hash = 0xcbf29ce484222325ULL;
  for (char c : str) {
    hash ^= static_cast<uint8_t>(c);
    hash *= 0x100000001b3ULL;
  }
  return hash;
}

inline uint64_t StableOpId(uint64_t hlo_module_id, absl::string_view name) {
  uint64_t combined = StableHash64(name);  // Fingerprint 2011
  combined ^=
      hlo_module_id + 0x9e3779b97f4a7c15ULL + (combined << 6) + (combined >> 2);
  return combined;
}

struct FlatOpMetricMeta {
  uint64_t hlo_module_id;
  std::string hlo_name;
  uint64_t op_id;
  uint64_t parent_op_id;
  uint64_t num_cores;
  uint64_t occurrences;
};

// Sets the total time for OpMetricsDb, ensuring idle time is not negative.
inline void SetTotalTimePs(FlatOpMetricsDb& db, uint64_t total_time_ps) {
  db.set_total_time_ps(std::max(db.total_op_time_ps(), total_time_ps));
}

// Adds an FlatOpMetrics record representing idle time, i.e., the amount of time
// spent without any op execution.
// REQUIRED: All ops must have been added to the database and the total time
// must have been set.
void AddIdleOp(FlatOpMetricsDb& db);

// Returns the idle time in picoseconds.
uint64_t IdleTimePs(const FlatOpMetricsDb& db);

// Populates an FlatOpMetrics record representing idle time, i.e., the amount of
// time spent without any op execution.
void SetIdleOp(uint64_t idle_time_ps, FlatOpMetrics& idle_op);


// Converts from the device flat op metrics to Tf-op metrics.
FlatOpMetricsDb CreateTfMetricsDbFromDeviceOpMetricsDb(
    const FlatOpMetricsDb& device_op_metrics_db, bool with_idle = true);

}  // namespace profiler
}  // namespace tensorflow

#endif  // XPROF_UTILS_FLAT_OP_METRICS_DB_UTILS_H_
