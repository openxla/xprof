#include "xprof/convert/op_metrics_to_record.h"

#include <cstddef>
#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/profiler/convert/xla_op_utils.h"
#include "plugin/xprof/protobuf/op_metrics.pb.h"

namespace tensorflow {
namespace profiler {

std::vector<const OpMetrics*> ScSortedOpMetricsDb(const OpMetricsDb& metrics_db,
                                                  int max_records) {
  std::vector<const OpMetrics*> queue;
  queue.reserve(metrics_db.metrics_db_size());
  for (const OpMetrics& metrics : metrics_db.metrics_db()) {
    queue.push_back(&metrics);
  }
  std::vector<const OpMetrics*> result;
  result.reserve(metrics_db.metrics_db_size());
  // Deduplicate by (hlo_module_id, name) because both async-start and
  // async-done TensorCore wrappers can reference the same SparseCore child DB.
  absl::flat_hash_set<std::pair<uint64_t, absl::string_view>> seen_ops;
  size_t head = 0;
  while (head < queue.size()) {
    const OpMetrics* metric_node = queue[head++];
    const bool is_async_wrapper =
        metric_node->category() == tsl::profiler::kHloAsyncStart ||
        metric_node->category() == tsl::profiler::kHloAsyncUpdate ||
        metric_node->category() == tsl::profiler::kHloAsyncDone;
    if (metric_node->core_type() == OpMetrics::SPARSE_CORE &&
        !is_async_wrapper) {
      if (seen_ops.emplace(metric_node->hlo_module_id(), metric_node->name())
              .second) {
        result.push_back(metric_node);
      }
      continue;
    }
    for (const OpMetrics& child_metric : metric_node->children().metrics_db()) {
      queue.push_back(&child_metric);
    }
  }

  auto comp = [](const OpMetrics* a, const OpMetrics* b) {
    return std::forward_as_tuple(a->self_time_ps(), b->name()) >
           std::forward_as_tuple(b->self_time_ps(), a->name());
  };
  if (max_records >= 0 && result.size() > static_cast<size_t>(max_records)) {
    absl::c_partial_sort(result, result.begin() + max_records, comp);
    result.resize(max_records);
  } else {
    absl::c_sort(result, comp);
  }
  return result;
}

}  // namespace profiler
}  // namespace tensorflow
