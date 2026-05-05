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

// This file contains utilities for building and manipulating FlatOpMetricsDb,
// which is a flattened representation of operation metrics, used for profiling
// and performance analysis.

#include "xprof/utils/flat_op_metrics_db_utils.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/profiler/utils/math_utils.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_visitor.h"
#include "plugin/xprof/protobuf/flat_op_metrics.pb.h"
#include "xprof/utils/op_metrics_db_utils.h"

namespace tensorflow {
namespace profiler {

using tsl::profiler::StatType;
using tsl::profiler::XEventMetadataVisitor;
using tsl::profiler::XStatVisitor;

namespace {

const int64_t kSingleOccurrence = 1;
constexpr uint64_t kRootSymbolId = 0;

// Extracts metadata from an XEventMetadataVisitor and populates the
// corresponding fields in a FlatOpMetrics protobuf message.
// This function handles names, categories, and various statistics associated
// with the event.
void SetOpMetadataFromHloEventMetadata(
    const XEventMetadataVisitor& hlo_event_metadata,
    FlatOpMetrics* op_metrics) {
  // Fill Names & Categories group.
  // We prefer the display name if available, as it is usually more readable.
  if (hlo_event_metadata.HasDisplayName()) {
    op_metrics->set_hlo_name(std::string(hlo_event_metadata.DisplayName()));
    op_metrics->set_long_name(std::string(hlo_event_metadata.Name()));
  } else {
    op_metrics->set_hlo_name(std::string(hlo_event_metadata.Name()));
  }

  // Iterate through all statistics associated with the event metadata.
  hlo_event_metadata.ForEachStat([&](const XStatVisitor& stat) {
    if (stat.Type().has_value()) {
      switch (static_cast<StatType>(*stat.Type())) {
        case StatType::kProgramId:
          // Fill Hierarchy & Identification.
          // The program ID maps to the HLO module ID in FlatOpMetrics.
          op_metrics->set_hlo_module_id(stat.IntOrUintValue());
          break;

        case StatType::kSymbolId:
          // Fill Hierarchy & Identification.
          op_metrics->set_symbol_id(stat.IntOrUintValue());
          break;

        case StatType::kHloCategory:
          // Fill Names & Categories.
          op_metrics->set_category(std::string(stat.StrOrRefValue()));
          break;

        case StatType::kTfOp:
          // Fill Names & Categories.
          // Provenance indicates the source operation (e.g., TF or JAX op).
          op_metrics->set_provenance(std::string(stat.StrOrRefValue()));
          break;

        case StatType::kFlops:
          // Fill Performance Metrics.
          // We populate both deprecated and v2 fields for compatibility.
          op_metrics->set_flops(stat.IntOrUintValue());
          op_metrics->set_flops_v2(static_cast<double>(stat.IntOrUintValue()));
          break;

        case StatType::kModelFlops:
          // Fill Performance Metrics.
          op_metrics->set_model_flops(stat.IntOrUintValue());
          op_metrics->set_model_flops_v2(
              static_cast<double>(stat.IntOrUintValue()));
          break;

        case StatType::kBytesAccessed:
          // Fill Performance Metrics.
          op_metrics->set_bytes_accessed(stat.IntOrUintValue());
          break;

        case StatType::kDeduplicatedName:
          // Fill Names & Categories.
          op_metrics->set_deduplicated_name(std::string(stat.StrOrRefValue()));
          break;

        case StatType::kMemoryAccessBreakdown: {
          // Fill Memory Breakdown.
          // Parse the serialized MemoryAccessBreakdown proto.
          tensorflow::profiler::MemoryAccessBreakdown breakdown;
          const auto& value = stat.BytesValue();
          if (breakdown.ParseFromString(value)) {
            for (const auto& mem_accessed : breakdown.memory_accessed()) {
              auto* flat_mem_accessed =
                  op_metrics->add_memory_accessed_breakdown();
              flat_mem_accessed->set_operation_type(
                  static_cast<FlatOpMetrics::MemoryAccessed::OperationType>(
                      mem_accessed.operation_type()));
              flat_mem_accessed->set_memory_space(mem_accessed.memory_space());
              flat_mem_accessed->set_bytes_accessed(
                  mem_accessed.bytes_accessed());
            }
          }
          break;
        }

        default:
          break;
      }
    }
  });
}

// Extracts execution metrics from an XEventVisitor and populates or updates
// the corresponding fields in a FlatOpMetrics protobuf message.
// If the metrics already exist (occurrences > 0), the new metrics are
// aggregated.
void SetOpMetricsFromHloEvent(const tsl::profiler::XEventVisitor& hlo_event,
                              FlatOpMetrics* flat_op_metrics) {
  uint64_t duration_ps = hlo_event.DurationPs();
  uint64_t min_duration_ps = duration_ps;
  uint64_t self_duration_ps = duration_ps;
  uint64_t normalized_duration_ps = 0;
  uint64_t dma_stall_ps = 0;

  // Extract durations from stats if available.
  hlo_event.ForEachStat([&](const XStatVisitor& stat) {
    if (!stat.Type()) return;
    switch (static_cast<StatType>(*stat.Type())) {
      case StatType::kMinDurationPs:
        min_duration_ps = stat.IntValue();
        break;
      case StatType::kSelfDurationPs:
        self_duration_ps = stat.IntValue();
        break;
      case StatType::kDmaStallDurationPs:
        dma_stall_ps = stat.IntValue();
        break;
      default:
        break;
    }
  });

  // Calculate normalized duration if time scale multiplier is present.
  std::optional<XStatVisitor> time_scale_multiplier_stat =
      hlo_event.GetStat(StatType::kTimeScaleMultiplier);
  if (time_scale_multiplier_stat.has_value()) {
    normalized_duration_ps =
        duration_ps * time_scale_multiplier_stat->DoubleValue();
  }

  // If this is the first occurrence, we initialize the metrics.
  if (flat_op_metrics->occurrences() == 0) {
    // Fill Names & Categories.
    SetOpMetadataFromHloEventMetadata(hlo_event.Metadata(), flat_op_metrics);

    // Fill Performance Metrics.
    flat_op_metrics->set_occurrences(
        std::max(kSingleOccurrence, hlo_event.NumOccurrences()));

    // Fill Execution Context & Target.
    flat_op_metrics->set_num_cores(1);

    // Fill Time Metrics (in picoseconds).
    flat_op_metrics->set_time_ps(duration_ps);
    flat_op_metrics->set_min_time_ps(min_duration_ps);
    flat_op_metrics->set_self_time_ps(self_duration_ps);
    flat_op_metrics->set_normalized_time_ps(normalized_duration_ps);
    flat_op_metrics->set_dma_stall_ps(dma_stall_ps);

  } else {
    // If this operation has been seen before, we update the metrics by
    // aggregating them.

    // Update Performance Metrics.
    flat_op_metrics->set_occurrences(flat_op_metrics->occurrences() +
                                     hlo_event.NumOccurrences());

    // Update Time Metrics (in picoseconds).
    flat_op_metrics->set_time_ps(flat_op_metrics->time_ps() + duration_ps);
    flat_op_metrics->set_min_time_ps(
        std::min<uint64_t>(flat_op_metrics->min_time_ps(), min_duration_ps));
    flat_op_metrics->set_self_time_ps(flat_op_metrics->self_time_ps() +
                                      self_duration_ps);
    flat_op_metrics->set_normalized_time_ps(
        flat_op_metrics->normalized_time_ps() + normalized_duration_ps);
    flat_op_metrics->set_dma_stall_ps(flat_op_metrics->dma_stall_ps() +
                                      dma_stall_ps);
  }

  // Special handling for CustomCall ops to extract flops and bytes accessed
  // from the event stats, as they might not be in the metadata.
  if (flat_op_metrics->category() ==
      xla::HloOpcodeString(xla::HloOpcode::kCustomCall)) {
    const int64_t num_occurrences =
        std::max(kSingleOccurrence, hlo_event.NumOccurrences());
    hlo_event.ForEachStat([&](const XStatVisitor& stat) {
      if (!stat.Type()) return;
      switch (static_cast<StatType>(*stat.Type())) {
        case StatType::kBytesAccessed:
          flat_op_metrics->set_bytes_accessed(
              flat_op_metrics->bytes_accessed() +
              stat.IntOrUintValue() * num_occurrences);
          break;
        case StatType::kModelFlops:
          flat_op_metrics->set_model_flops(flat_op_metrics->model_flops() +
                                           stat.IntOrUintValue() *
                                               num_occurrences);
          flat_op_metrics->set_model_flops_v2(
              flat_op_metrics->model_flops_v2() +
              static_cast<double>(stat.IntOrUintValue()) * num_occurrences);
          break;
        case StatType::kFlops:
          flat_op_metrics->set_flops(flat_op_metrics->flops() +
                                     stat.IntOrUintValue() * num_occurrences);
          flat_op_metrics->set_flops_v2(
              flat_op_metrics->flops_v2() +
              static_cast<double>(stat.IntOrUintValue()) * num_occurrences);
          break;
        default:
          break;
      }
    });
  }
}

// Merges metrics from a source FlatOpMetrics into a destination FlatOpMetrics.
// This is used to aggregate metrics from multiple occurrences or different
// threads/cores.
void MergeOpMetrics(const FlatOpMetrics& src, FlatOpMetrics& dst) {
  // If the destination is empty, we can simply copy the source.
  if (dst.occurrences() == 0) {
    dst = src;
  } else {
    // Otherwise, we aggregate the metrics.

    // Merge Performance Metrics.
    dst.set_occurrences(src.occurrences() + dst.occurrences());

    // Merge Execution Context & Target.
    dst.set_core_type(src.core_type());

    // Merge Time Metrics (in picoseconds).
    dst.set_time_ps(src.time_ps() + dst.time_ps());

    // For min time, we take the minimum of both.
    dst.set_min_time_ps(
        std::min<uint64_t>(src.min_time_ps(), dst.min_time_ps()));

    dst.set_self_time_ps(src.self_time_ps() + dst.self_time_ps());
    dst.set_dma_stall_ps(src.dma_stall_ps() + dst.dma_stall_ps());
    dst.set_normalized_time_ps(src.normalized_time_ps() +
                               dst.normalized_time_ps());

    if (dst.category() == xla::HloOpcodeString(xla::HloOpcode::kCustomCall)) {
      // Merge Performance Metrics (Flops and Bytes) For CustomCall ops.
      // Their Cost is Dynamic in nature
      dst.set_flops(dst.flops() + src.flops());
      dst.set_flops_v2(dst.flops_v2() + src.flops_v2());
      dst.set_model_flops(dst.model_flops() + src.model_flops());
      dst.set_model_flops_v2(dst.model_flops_v2() + src.model_flops_v2());
      dst.set_bytes_accessed(dst.bytes_accessed() + src.bytes_accessed());
    }
  }
}

// Adjusts FLOPs and bytes accessed metrics by multiplying them by the number
// of occurrences. This is needed because the raw metrics from events are
// usually per-occurrence, but we want total metrics in the finalized database.
void AdjustFlopsAndBytesAccessed(FlatOpMetrics& op_metrics) {
  // CustomCall ops are excluded from this adjustment because their metrics
  // are already accumulated in SetOpMetricsFromHloEvent.
  if (op_metrics.category() !=
      xla::HloOpcodeString(xla::HloOpcode::kCustomCall)) {
    op_metrics.set_flops(op_metrics.flops() * op_metrics.occurrences());
    op_metrics.set_flops_v2(op_metrics.flops_v2() * op_metrics.occurrences());

    // If model_flops is explicitly set, we scale it.
    // Otherwise, we fallback to scaling the standard flops.
    if (op_metrics.model_flops() > 0) {
      op_metrics.set_model_flops(op_metrics.model_flops() *
                                 op_metrics.occurrences());
      op_metrics.set_model_flops_v2(op_metrics.model_flops_v2() *
                                    op_metrics.occurrences());
    } else {
      op_metrics.set_model_flops(op_metrics.flops());
      op_metrics.set_model_flops_v2(op_metrics.flops_v2());
    }

    op_metrics.set_bytes_accessed(op_metrics.bytes_accessed() *
                                  op_metrics.occurrences());
  }

  // Scale memory access breakdown by occurrences as well.
  for (auto& memory_access : *op_metrics.mutable_memory_accessed_breakdown()) {
    memory_access.set_bytes_accessed(memory_access.bytes_accessed() *
                                     op_metrics.occurrences());
  }
}

}  // namespace

// Constructor for FlatOpMetricsDbBuilder.
// Initializes the builder with a pointer to the FlatOpMetricsDb to be
// populated. It also builds a lookup map for fast access to existing metrics.
FlatOpMetricsDbBuilder::FlatOpMetricsDbBuilder(FlatOpMetricsDb* db) : db_(db) {
  DCHECK_NE(db_, nullptr);
  for (auto& op_metrics : *db_->mutable_op_instances()) {
    op_metrics_map_[op_metrics.hlo_module_id()][op_metrics.hlo_name()] =
        &op_metrics;
  }
}

// Looks up an existing FlatOpMetrics instance for the given HLO module ID and
// name. If it doesn't exist, a new one is inserted and initialized with the ID
// and name.
FlatOpMetrics* FlatOpMetricsDbBuilder::LookupOrInsertNewFlatOpMetrics(
    uint64_t hlo_module_id, absl::string_view hlo_name) {
  FlatOpMetrics*& op_metrics =
      op_metrics_map_[hlo_module_id][std::string(hlo_name)];
  if (op_metrics == nullptr) {
    op_metrics = db_->add_op_instances();
    op_metrics->set_hlo_module_id(hlo_module_id);
    op_metrics->set_hlo_name(std::string(hlo_name));
  }
  return op_metrics;
}

// Creates a FlatOpMetrics message from a tsl::profiler::XEventVisitor.
// This is a static helper method.
FlatOpMetrics XEventsFlatOpMetricsDbBuilder::FromXEvent(
    const tsl::profiler::XEventVisitor& xevent) {
  FlatOpMetrics op_metrics;
  SetOpMetricsFromHloEvent(xevent, &op_metrics);
  return op_metrics;
}

// Adds an operation metric from a tsl::profiler::XEventVisitor to the builder.
// It extracts the key and aggregates the metrics.
void XEventsFlatOpMetricsDbBuilder::AddOpMetric(
    const tsl::profiler::XEventVisitor& event) {
  auto key = GetOpKeyFromXEvent(event);
  XEventsFlatOpMetricsDbBuilder::OpKey flat_key;
  flat_key.program_id = key.program_id;
  flat_key.symbol_id = key.symbol_id;
  AddOpMetric(XEventsFlatOpMetricsDbBuilder::FromXEvent(event), flat_key);
}

// Adds a FlatOpMetrics message with a specific OpKey to the builder.
// This handles the actual aggregation logic.
void XEventsFlatOpMetricsDbBuilder::AddOpMetric(const FlatOpMetrics& op_metrics,
                                                const OpKey& key) {
  if (!key.program_id.has_value() || !key.symbol_id.has_value() ||
      key.symbol_id == kRootSymbolId)
    return;
  MergeOpMetrics(
      op_metrics,
      flat_op_metric_[key.program_id.value()][key.symbol_id.value()]);
}

// Finalizes the database and sets the total time.
// This overload accepts a total time in picoseconds.
// Currently, it just calls the parameterless Finalize() and returns the result.
// TODO: Implement SetTotalTimePs and AddIdleOp for FlatOpMetricsDb if needed.
FlatOpMetricsDb XEventsFlatOpMetricsDbBuilder::Finalize(
    uint64_t total_time_ps) {
  FlatOpMetricsDb db = Finalize();
  SetTotalTimePs(db, total_time_ps);
  AddIdleOp(db);
  return db;
}

// Finalizes the database by aggregating all collected metrics.
// It iterates through the internal map, adjusts flops and bytes accessed,
// and adds the metrics to the FlatOpMetricsDb proto message.
FlatOpMetricsDb XEventsFlatOpMetricsDbBuilder::Finalize() {
  FlatOpMetricsDb db;

  uint64_t total_op_time_ps = 0;
  uint64_t normalized_total_op_time_ps = 0;

  // Iterate over all programs and symbols to collect metrics.
  for (auto& [program_id, op_metric_by_symbol] : flat_op_metric_) {
    for (auto& [symbol_id, op_metrics] : op_metric_by_symbol) {
      // Scale metrics by occurrences before adding to the DB.
      AdjustFlopsAndBytesAccessed(op_metrics);
      total_op_time_ps += op_metrics.self_time_ps();
      normalized_total_op_time_ps +=
          op_metrics.self_time_ps() *
          tsl::profiler::SafeDivide(op_metrics.normalized_time_ps() * 1.0,
                                    op_metrics.time_ps());
      *db.add_op_instances() = std::move(op_metrics);
    }
  }
  db.set_total_op_time_ps(total_op_time_ps);
  db.set_normalized_total_op_time_ps(normalized_total_op_time_ps);
  return db;
}

// Finalizes the database by Topologically Sorting the operations.
FlatOpMetricsDb XEventsFlatOpMetricsDbBuilder::FinalizeSorted() {
  FlatOpMetricsDb db;
  uint64_t total_op_time_ps = 0;
  uint64_t normalized_total_op_time_ps = 0;

  absl::flat_hash_map<uint64_t, FlatOpMetrics> metrics_map;
  absl::flat_hash_map<uint64_t, std::vector<uint64_t>> children_map;
  std::queue<uint64_t> roots_queue;

  for (auto& [program_id, op_metric_by_symbol] : flat_op_metric_) {
    for (auto& [symbol_id, op_metrics] : op_metric_by_symbol) {
      AdjustFlopsAndBytesAccessed(op_metrics);

      uint64_t op_id = op_metrics.op_id();
      uint64_t parent_id = op_metrics.parent_op_id();

      metrics_map[op_id] = std::move(op_metrics);

      if (parent_id == 0) {
        roots_queue.push(op_id);
      } else {
        children_map[parent_id].push_back(op_id);
      }
    }
  }

  while (!roots_queue.empty()) {
    uint64_t curr_id = roots_queue.front();
    roots_queue.pop();

    auto it = metrics_map.find(curr_id);
    if (it != metrics_map.end()) {
      FlatOpMetrics& op_metrics = it->second;

      total_op_time_ps += op_metrics.self_time_ps();
      normalized_total_op_time_ps +=
          op_metrics.self_time_ps() *
          tsl::profiler::SafeDivide(op_metrics.normalized_time_ps() * 1.0,
                                    op_metrics.time_ps());

      *db.add_op_instances() = std::move(op_metrics);
      metrics_map.erase(it);

      for (uint64_t child_id : children_map[curr_id]) {
        roots_queue.push(child_id);
      }
    }
  }

  db.set_total_op_time_ps(total_op_time_ps);
  db.set_normalized_total_op_time_ps(normalized_total_op_time_ps);

  return db;
}

// Finalizes the database with Topologically sorted operations and add total
// time.
FlatOpMetricsDb XEventsFlatOpMetricsDbBuilder::FinalizeSorted(
    uint64_t total_time_ps) {
  FlatOpMetricsDb db = FinalizeSorted();
  SetTotalTimePs(db, total_time_ps);
  AddIdleOp(db);
  return db;
}

void AddIdleOp(FlatOpMetricsDb& db) {
  uint64_t idle_time_ps = IdleTimePs(db);
  SetIdleOp(idle_time_ps, *db.add_op_instances());
}

uint64_t IdleTimePs(const FlatOpMetricsDb& db) {
  DCHECK_GE(db.total_time_ps(), db.total_op_time_ps());
  return db.total_time_ps() - db.total_op_time_ps();
}

void SetIdleOp(uint64_t idle_time_ps, FlatOpMetrics& idle_op) {
  idle_op.set_hlo_name(kIdle);
  idle_op.set_category(kIdle);
  idle_op.set_occurrences(0);
  idle_op.set_time_ps(idle_time_ps);
  idle_op.set_self_time_ps(idle_time_ps);
}

}  // namespace profiler
}  // namespace tensorflow
