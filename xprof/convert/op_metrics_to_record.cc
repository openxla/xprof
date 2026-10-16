#include "xprof/convert/op_metrics_to_record.h"
#include <cstddef>
#include <tuple>
#include <vector>

#include "absl/algorithm/container.h"
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
  size_t head = 0;
  while (head < queue.size()) {
    const OpMetrics* metric_node = queue[head++];
    if (metric_node->core_type() == OpMetrics::SPARSE_CORE) {
      result.push_back(metric_node);
    }
    for (const OpMetrics& child_metric : metric_node->children().metrics_db()) {
      queue.push_back(&child_metric);
    }
  }

  auto comp = [](const OpMetrics* a, const OpMetrics* b) {
    return std::make_tuple(a->self_time_ps(), b->name()) >
           std::make_tuple(b->self_time_ps(), a->name());
  };
  int result_size = result.size();
  if (max_records >= 0 && result_size > max_records) {
    absl::c_partial_sort(result, result.begin() + max_records, comp);
    result.resize(max_records);
  } else {
    absl::c_sort(result, comp);
  }
  return result;
}

}  // namespace profiler
}  // namespace tensorflow
