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

#include "xprof/convert/flat_op_metrics_db_combiner.h"

#include <algorithm>
#include <cstdint>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "xla/tsl/platform/logging.h"
#include "tsl/platform/protobuf.h"
#include "plugin/xprof/protobuf/op_stats.pb.h"
#include "plugin/xprof/protobuf/source_info.pb.h"
#include "xprof/utils/flat_op_metrics_db_utils.h"

namespace tensorflow {
namespace profiler {

void FlatOpMetricsDbCombiner::CopyFlatOpMetricsMetadata(
    const FlatOpMetrics& src, FlatOpMetrics* dst) {
  DCHECK(dst != nullptr);
  DCHECK_EQ(src.hlo_module_id(), dst->hlo_module_id());
  DCHECK_EQ(src.hlo_name(), dst->hlo_name());

  if (dst->op_id() == 0) {
    dst->set_op_id(src.op_id());
  }
  if (dst->parent_op_id() == 0) {
    dst->set_parent_op_id(src.parent_op_id());
  }

  if (dst->long_name().empty()) {
    dst->set_long_name(src.long_name());
  }
  if (dst->category().empty()) {
    dst->set_category(src.category());
  }
  if (dst->provenance().empty()) {
    dst->set_provenance(src.provenance());
  }
  if (dst->deduplicated_name().empty()) {
    dst->set_deduplicated_name(src.deduplicated_name());
  }
  if (dst->core_type() == FlatOpMetrics::UNKNOWN) {
    dst->set_core_type(src.core_type());
  }
  if (dst->symbol_id() == 0) {
    dst->set_symbol_id(src.symbol_id());
  }
  if (dst->children_count() == 0) {
    dst->set_children_count(src.children_count());
  }
  if (!dst->has_sc_children()) {
    dst->set_has_sc_children(src.has_sc_children());
  }
  if (!dst->is_eager()) {
    dst->set_is_eager(src.is_eager());
  }
  if (!dst->autotuned()) {
    dst->set_autotuned(src.autotuned());
  }
  if (!dst->has_source_info() && src.has_source_info()) {
    *dst->mutable_source_info() = src.source_info();
  }
}

void FlatOpMetricsDbCombiner::CombineFlatOpMetrics(const FlatOpMetrics& src,
                                                   FlatOpMetrics* dst,
                                                   bool update_num_cores) {
  DCHECK(dst != nullptr);
  if (dst->occurrences() == 0) {
    dst->set_min_time_ps(src.min_time_ps());
  } else {
    dst->set_min_time_ps(std::min(src.min_time_ps(), dst->min_time_ps()));
  }
  if (update_num_cores) {
    dst->set_num_cores(src.num_cores() + dst->num_cores());
  }
  dst->set_occurrences(src.occurrences() + dst->occurrences());
  dst->set_time_ps(src.time_ps() + dst->time_ps());
  dst->set_self_time_ps(src.self_time_ps() + dst->self_time_ps());
  dst->set_normalized_time_ps(src.normalized_time_ps() +
                              dst->normalized_time_ps());
  dst->set_flops(src.flops() + dst->flops());
  dst->set_model_flops(src.model_flops() + dst->model_flops());
  dst->set_flops_v2(src.flops_v2() + dst->flops_v2());
  dst->set_model_flops_v2(src.model_flops_v2() + dst->model_flops_v2());
  dst->set_bytes_accessed(src.bytes_accessed() + dst->bytes_accessed());
  dst->set_dma_stall_ps(src.dma_stall_ps() + dst->dma_stall_ps());
  dst->set_autotuned(dst->autotuned() || src.autotuned());
  CombineMemoryAccessedBreakdown(src.memory_accessed_breakdown(),
                                 dst->mutable_memory_accessed_breakdown());
  dst->set_is_eager(dst->is_eager() || src.is_eager());
}

void FlatOpMetricsDbCombiner::CombineMemoryAccessedBreakdown(
    const tsl::protobuf::RepeatedPtrField<FlatOpMetrics::MemoryAccessed>& src,
    tsl::protobuf::RepeatedPtrField<FlatOpMetrics::MemoryAccessed>* dst) {
  if (src.empty()) return;
  using FlatOperationType = FlatOpMetrics::MemoryAccessed::OperationType;
  absl::flat_hash_map<std::pair<uint64_t /*memory_space*/, FlatOperationType>,
                      FlatOpMetrics::MemoryAccessed*>
      dst_memory_accessed_map;
  for (auto& dst_memory_accessed : *dst) {
    dst_memory_accessed_map[{dst_memory_accessed.memory_space(),
                             dst_memory_accessed.operation_type()}] =
        &dst_memory_accessed;
  }
  for (const auto& src_memory_accessed : src) {
    uint64_t memory_space = src_memory_accessed.memory_space();
    FlatOperationType operation_type = src_memory_accessed.operation_type();
    auto*& dst_memory_accessed =
        dst_memory_accessed_map[{memory_space, operation_type}];
    if (dst_memory_accessed == nullptr) {
      dst_memory_accessed = dst->Add();
      dst_memory_accessed->set_memory_space(memory_space);
      dst_memory_accessed->set_operation_type(operation_type);
    }
    dst_memory_accessed->set_bytes_accessed(
        src_memory_accessed.bytes_accessed() +
        dst_memory_accessed->bytes_accessed());
  }
}

void FlatOpMetricsDbCombiner::Combine(const FlatOpMetricsDb& src,
                                      bool update_num_cores) {
  FlatOpMetricsDb* dst = db();
  dst->set_total_time_ps(src.total_time_ps() + dst->total_time_ps());
  dst->set_total_op_time_ps(src.total_op_time_ps() + dst->total_op_time_ps());
  dst->set_idle_time_ps(src.idle_time_ps() + dst->idle_time_ps());
  dst->set_busy_time_ps(src.busy_time_ps() + dst->busy_time_ps());
  dst->set_normalized_total_op_time_ps(src.normalized_total_op_time_ps() +
                                       dst->normalized_total_op_time_ps());
  dst->set_total_host_infeed_enq_duration_ps(
      src.total_host_infeed_enq_duration_ps() +
      dst->total_host_infeed_enq_duration_ps());
  dst->set_total_host_infeed_enq_start_timestamp_ps_diff(
      src.total_host_infeed_enq_start_timestamp_ps_diff() +
      dst->total_host_infeed_enq_start_timestamp_ps_diff());

  // Removed assignment of perf_env, device_type, and program_id_to_name_map
  // as they were removed from the proto definition.

  for (const auto& src_metrics : src.op_instances()) {
    if (IsIdleOp(src_metrics)) {
      if (src_metrics.parent_op_id() != 0) continue;
    }
    if (src_metrics.is_fusion_child()) {
      *dst->add_op_instances() = src_metrics;
    } else {
      auto* dst_metrics = LookupOrInsertNewFlatOpMetrics(
          src_metrics.hlo_module_id(), src_metrics.hlo_name());
      CopyFlatOpMetricsMetadata(src_metrics, dst_metrics);
      CombineFlatOpMetrics(src_metrics, dst_metrics, update_num_cores);
    }
  }
}

}  // namespace profiler
}  // namespace tensorflow
