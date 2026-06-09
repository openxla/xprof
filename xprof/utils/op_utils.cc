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

#include "xprof/utils/op_utils.h"

#include <cstdint>
#include <string>

#include "absl/log/check.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/tsl/profiler/convert/xla_op_utils.h"
#include "xla/tsl/profiler/utils/tf_op_utils.h"
#include "xla/tsl/profiler/utils/timespan.h"
#include "tsl/platform/protobuf.h"
#include "xprof/convert/op_metrics_db_combiner.h"
#include "plugin/xprof/protobuf/op_metrics.pb.h"
#include "plugin/xprof/protobuf/source_info.pb.h"
#include "xprof/utils/hlo_module_map.h"
#include "xprof/utils/performance_info_wrapper.h"

namespace tensorflow {
namespace profiler {

tsl::protobuf::RepeatedPtrField<OpMetrics::MemoryAccessed>
ConvertPerformanceInfo(
    const tsl::protobuf::RepeatedPtrField<
        tensorflow::profiler::PerformanceInfoWrapper::PerfInfoType::
            MemoryAccessed>& memory_accessed_breakdown,
    uint64_t occurrences) {
  tsl::protobuf::RepeatedPtrField<OpMetrics::MemoryAccessed>
      memory_access_breakdown;
  for (const auto& m : memory_accessed_breakdown) {
    auto* memory_access = memory_access_breakdown.Add();
    memory_access->set_operation_type(m.is_read()
                                          ? OpMetrics::MemoryAccessed::READ
                                          : OpMetrics::MemoryAccessed::WRITE);
    memory_access->set_memory_space(m.memory_space());
    memory_access->set_bytes_accessed(m.bytes_accessed() * occurrences);
  }
  return memory_access_breakdown;
}

// Annotate the op_metrics with the metadata from the instr_wrapper.
void EnterOpMetadata(OpMetrics* op_metrics,
                     const HloInstructionWrapper* instr_wrapper) {
  if (op_metrics->name().empty() && op_metrics->category().empty() &&
      op_metrics->provenance().empty()) {
    op_metrics->set_name(std::string(instr_wrapper->Name()));
    op_metrics->set_category(std::string(instr_wrapper->Category()));
    op_metrics->set_deduplicated_name(
        instr_wrapper->Metadata().deduplicated_name());
    op_metrics->set_provenance(std::string(instr_wrapper->op_full_name()));
    op_metrics->set_num_cores(1);
    op_metrics->set_occurrences(op_metrics->occurrences() + 1);
    op_metrics->set_flops(op_metrics->flops() + instr_wrapper->flops());
    op_metrics->set_flops_v2(op_metrics->flops_v2() +
                             static_cast<double>(instr_wrapper->flops()));
    op_metrics->set_bytes_accessed(op_metrics->bytes_accessed() +
                                   instr_wrapper->bytes_accessed());
    op_metrics->set_long_name(instr_wrapper->Expression());
  }
}

void AddFusionChildrenToOpMetricsFromHloInstruction(
    OpMetrics* op_metrics, const HloInstructionWrapper* instr_wrapper) {
  if (instr_wrapper->FusedChildren().empty()) return;
  for (const HloInstructionWrapper* child : instr_wrapper->FusedChildren()) {
    if (child->HloOpcode() == xla::HloOpcode::kParameter ||
        child->HloOpcode() == xla::HloOpcode::kTuple)
      continue;
    OpMetrics* child_op_metrics =
        op_metrics->mutable_children()->add_metrics_db();
    EnterOpMetadata(child_op_metrics, child);
    AddFusionChildrenToOpMetricsFromHloInstruction(child_op_metrics, child);
  }
}

void EnterOpMetadataFromHloModuleMap(OpMetrics* op_metrics,
                                     const HloModuleMap& hlo_module_map) {
  const HloInstructionWrapper* instr_wrapper = GetHloInstruction(
      hlo_module_map, op_metrics->hlo_module_id(), op_metrics->name());
  if (instr_wrapper != nullptr) {
    AddFusionChildrenToOpMetricsFromHloInstruction(op_metrics, instr_wrapper);
  }
}

void HostOpMetricsDbBuilder::EnterOp(absl::string_view name,
                                     absl::string_view category, bool is_eager,
                                     uint64_t time_ps,
                                     uint64_t children_time_ps, int64_t id) {
  uint64_t self_time_ps = time_ps - children_time_ps;
  DCHECK_GE(time_ps, self_time_ps);
  OpMetrics* op_metrics =
      LookupOrInsertNewOpMetrics(/*hlo_module_id=*/id, name);
  if (op_metrics->category().empty())
    op_metrics->set_category(category.data(), category.size());
  op_metrics->set_num_cores(1);
  op_metrics->set_is_eager(op_metrics->is_eager() || is_eager);
  op_metrics->set_occurrences(op_metrics->occurrences() + 1);
  op_metrics->set_time_ps(op_metrics->time_ps() + time_ps);
  op_metrics->set_self_time_ps(op_metrics->self_time_ps() + self_time_ps);
  db()->set_total_op_time_ps(db()->total_op_time_ps() + self_time_ps);
}

void HostOpMetricsDbBuilder::EnterHostInfeedEnqueue(
    tsl::profiler::Timespan host_infeed_enqueue) {
  if (!last_host_infeed_enqueue_.Empty()) {
    // Expect non-overlapping InfeedEnqueue timespans sorted by time.
    DCHECK_GE(host_infeed_enqueue.end_ps(),
              last_host_infeed_enqueue_.begin_ps());
    db()->set_total_host_infeed_enq_duration_ps(
        db()->total_host_infeed_enq_duration_ps() +
        last_host_infeed_enqueue_.duration_ps());
    db()->set_total_host_infeed_enq_start_timestamp_ps_diff(
        db()->total_host_infeed_enq_start_timestamp_ps_diff() +
        (host_infeed_enqueue.begin_ps() -
         last_host_infeed_enqueue_.begin_ps()));
  }
  last_host_infeed_enqueue_ = host_infeed_enqueue;
}

void DeviceOpMetricsDbBuilder::EnterOpMetadataFromHloModuleMap(
    uint64_t program_id, absl::string_view op_name,
    const HloModuleMap& hlo_module_map) {
  OpMetrics* op_metrics = LookupOrInsertNewOpMetrics(program_id, op_name);
  tensorflow::profiler::EnterOpMetadataFromHloModuleMap(op_metrics,
                                                        hlo_module_map);
}

void DeviceOpMetricsDbBuilder::EnterOpMetadata(const OpIdentifier& op_id,
                                               bool is_eager) {
  // We only need to add xla metadata once to each new op, as they are the
  // same across occurrences.
  OpMetrics* op_metrics =
      LookupOrInsertNewOpMetrics(op_id.program_id, op_id.name);
  if (op_metrics->occurrences() > 0 || !op_metrics->category().empty() ||
      !op_metrics->provenance().empty())
    return;
  op_metrics->set_category(op_id.category == tsl::profiler::kUnknownOp
                               ? "unknown"
                               : std::string(op_id.category));
  op_metrics->set_provenance(std::string(op_id.provenance));
  if (!op_id.deduplicated_name.empty()) {
    op_metrics->set_deduplicated_name(std::string(op_id.deduplicated_name));
  }
  if (!op_id.long_name.empty()) {
    op_metrics->set_long_name(std::string(op_id.long_name));
  }
  op_metrics->set_is_eager(op_metrics->is_eager() || is_eager);
  op_metrics->mutable_source_info()->set_file_name(
      op_id.op_source_info.source_file);
  op_metrics->mutable_source_info()->set_line_number(
      op_id.op_source_info.source_line);
  op_metrics->mutable_source_info()->set_stack_frame(
      op_id.op_source_info.stack_frame);
}

void DeviceOpMetricsDbBuilder::EnterOp(const OpIdentifier& op_id,
                                       const OpData& event_data) {
  EnterOpMetadata(op_id, event_data.is_eager);
  uint64_t self_time_ps = event_data.time_ps - event_data.children_time_ps;
  DCHECK_GE(event_data.time_ps, self_time_ps);
  OpMetrics* op_metrics =
      LookupOrInsertNewOpMetrics(op_id.program_id, op_id.name);
  op_metrics->set_num_cores(1);
  op_metrics->set_occurrences(op_metrics->occurrences() +
                              event_data.occurrences);
  op_metrics->set_time_ps(op_metrics->time_ps() + event_data.time_ps);
  op_metrics->set_self_time_ps(op_metrics->self_time_ps() + self_time_ps);

  // Populate metrics from PerformanceInfoWrapper if available (GPU/Symbol path)
  const auto* perf_info = event_data.perf_info;
  const int64_t flops =
      perf_info != nullptr ? perf_info->DeviceFlops() : event_data.flops;
  const int64_t bytes_accessed = perf_info != nullptr
                                     ? perf_info->bytes_accessed()
                                     : event_data.bytes_accessed;
  const tsl::protobuf::RepeatedPtrField<OpMetrics::MemoryAccessed>&
      memory_accessed_breakdown =
          perf_info != nullptr
              ? ConvertPerformanceInfo(perf_info->memory_accessed_breakdown(),
                                       event_data.occurrences)
              : event_data.memory_accessed_breakdown;
  const int64_t model_flops =
      perf_info != nullptr ? perf_info->ModelFlops() : event_data.model_flops;

  op_metrics->set_flops(op_metrics->flops() + flops * event_data.occurrences);
  op_metrics->set_flops_v2(op_metrics->flops_v2() +
                           static_cast<double>(flops) * event_data.occurrences);
  if (model_flops == 0) {
    // If ModelsFlops is 0, use the same value as device flops.
    op_metrics->set_model_flops(op_metrics->flops());
    op_metrics->set_model_flops_v2(op_metrics->flops_v2());
  } else {
    op_metrics->set_model_flops(op_metrics->model_flops() +
                                model_flops * event_data.occurrences);
    op_metrics->set_model_flops_v2(
        op_metrics->model_flops_v2() +
        static_cast<double>(model_flops) *
            static_cast<double>(event_data.occurrences));
  }
  op_metrics->set_bytes_accessed(op_metrics->bytes_accessed() +
                                 bytes_accessed * event_data.occurrences);
  CombineMemoryAccessedBreakdown(
      memory_accessed_breakdown,
      op_metrics->mutable_memory_accessed_breakdown());

  // Accumulate DMA stall time
  if (event_data.dma_stall_ps > 0) {
    op_metrics->set_dma_stall_ps(op_metrics->dma_stall_ps() +
                                 event_data.dma_stall_ps *
                                     event_data.occurrences);
  }

  db()->set_total_op_time_ps(db()->total_op_time_ps() + self_time_ps);
}

}  // namespace profiler
}  // namespace tensorflow
