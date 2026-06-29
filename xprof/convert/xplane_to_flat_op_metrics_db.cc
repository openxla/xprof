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

#include "xprof/convert/xplane_to_flat_op_metrics_db.h"
#include "xprof/convert/xplane_to_op_metrics_db.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/profiler/utils/tf_xplane_visitor.h"
#include "xla/tsl/profiler/utils/trace_utils.h"
#include "xla/tsl/profiler/utils/timespan.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_utils.h"
#include "xla/tsl/profiler/utils/xplane_visitor.h"
#include "xprof/convert/flat_op_metrics_db_combiner.h"
#include "plugin/xprof/protobuf/flat_op_metrics.pb.h"
#include "xprof/utils/flat_op_metrics_db_utils.h"
#include "xprof/utils/op_metrics_db_utils.h"
#include "xprof/utils/op_utils.h"
#include "absl/strings/str_cat.h"
#include "absl/log/log.h"
#include "xla/tsl/profiler/utils/tf_op_utils.h"
#include "xprof/utils/gpu_event_stats.h"
#include "xprof/utils/hlo_module_map.h"

namespace tensorflow {
namespace profiler {

using ::tsl::profiler::GetDeviceEventTimespan;
using ::tsl::profiler::StatType;
using ::tsl::profiler::XEventVisitor;
using ::tsl::profiler::XLineVisitor;
using ::tsl::profiler::XPlaneVisitor;

void ConvertSparseCoreDeviceTraceXPlaneToFlatOpMetricsDb(
    const XPlane& device_trace,
    absl::flat_hash_map<std::pair<uint64_t, uint64_t>, FlatOpMetricsDb>&
        sparse_core_metrics_map) {
  XPlaneVisitor plane = tsl::profiler::CreateTfXPlaneVisitor(&device_trace);

  struct ParentReference {
    const XEventVisitor event;
    tsl::profiler::Timespan device_timespan;
    uint64_t offload_core_id;
    uint64_t tc_start_id;
    uint64_t children_duration_ps = 0;
    uint64_t op_id = 0;
    uint64_t parent_op_id = 0;
  };

  struct OpMetricsInProgress {
    std::unique_ptr<XEventsFlatOpMetricsDbBuilder> builder;
    tsl::profiler::AncestorStack<ParentReference> event_stack;
    OpMetricsInProgress()
        : builder(std::make_unique<XEventsFlatOpMetricsDbBuilder>()),
          event_stack(
              [builder_ptr = builder.get()](const ParentReference& parent) {
                FlatOpMetrics op_metrics =
                    XEventsFlatOpMetricsDbBuilder::FromXEvent(parent.event);
                op_metrics.set_time_ps(parent.device_timespan.duration_ps());
                op_metrics.set_self_time_ps(op_metrics.time_ps() -
                                            parent.children_duration_ps);
                op_metrics.set_core_type(FlatOpMetrics_TpuCoreType_SPARSE_CORE);
                op_metrics.set_op_id(parent.op_id);
                op_metrics.set_parent_op_id(parent.parent_op_id);
                std::optional<tsl::profiler::XStatVisitor>
                    time_scale_multiplier_stat =
                        parent.event.GetStat(StatType::kTimeScaleMultiplier);
                double factor = time_scale_multiplier_stat.has_value()
                                    ? time_scale_multiplier_stat->DoubleValue()
                                    : 1.0;
                op_metrics.set_normalized_time_ps(op_metrics.time_ps() *
                                                  factor);
                XEventsOpMetricsDbBuilder::OpKey key =
                    GetOpKeyFromXEvent(parent.event);
                XEventsFlatOpMetricsDbBuilder::OpKey flat_key;
                flat_key.program_id = key.program_id;
                flat_key.symbol_id = key.symbol_id;
                builder_ptr->AddOpMetric(op_metrics, flat_key);
              },
              [](const ParentReference& parent, const ParentReference& child) {
                return parent.device_timespan.Includes(child.device_timespan);
              },
              [](ParentReference& parent, ParentReference& child) {
                parent.children_duration_ps +=
                    child.device_timespan.duration_ps();
                child.parent_op_id = parent.op_id;
              }) {}
  };
  absl::flat_hash_map<std::pair<uint64_t, uint64_t>, OpMetricsInProgress>
      intermediate_op_metrics_map;

  struct ModuleReference {
    tsl::profiler::Timespan timespan;
    const uint64_t offload_core_id;
    const uint64_t tc_start_id;
    bool operator<(const ModuleReference& other) const {
      return timespan < other.timespan;
    }
  };
  absl::btree_set<ModuleReference> module_timespans;

  // Find the module timespan
  plane.ForEachLine([&](const XLineVisitor& line) {
    if (line.Name() == tsl::profiler::kSparseCoreModuleLineName) {
      line.ForEachEvent([&](const XEventVisitor& event) {
        const std::optional<tsl::profiler::XStatVisitor> offload_core_id =
            event.GetStat(StatType::kOffloadCoreId);
        const std::optional<tsl::profiler::XStatVisitor> tc_start_id =
            event.GetStat(StatType::kTcOffloadStartId);
        if (offload_core_id.has_value() && tc_start_id.has_value()) {
          module_timespans.insert(
              {.timespan = GetDeviceEventTimespan(event),
               .offload_core_id = offload_core_id->UintValue(),
               .tc_start_id = tc_start_id->UintValue()});
        }
      });
    }
  });
  // Now walk through the events and add them to their proper OpMetricsDb
  plane.ForEachLine([&](const XLineVisitor& line) {
    if (line.Name() == tsl::profiler::kSparseCoreOpLineName) {
      auto module_it = module_timespans.begin();
      line.ForEachEvent([&](const XEventVisitor& event) {
        const tsl::profiler::Timespan timespan = GetDeviceEventTimespan(event);
        // Advance module_it to skip modules that end before this event starts.
        while (module_it != module_timespans.end() &&
               module_it->timespan.end_ps() < timespan.begin_ps()) {
          ++module_it;
        }
        if (module_it == module_timespans.end()) {
          return;
        }
        // Check if the current module_it encapsulates the event.
        if (module_it->timespan.Includes(timespan)) {
          // Insert that event into the stack for that module.
          absl::string_view hlo_name = event.Metadata().HasDisplayName()
                                           ? event.Metadata().DisplayName()
                                           : event.Metadata().Name();
          XEventsOpMetricsDbBuilder::OpKey key = GetOpKeyFromXEvent(event);
          auto module_id = key.program_id.has_value() ? key.program_id.value()
                                                      : module_it->tc_start_id;
          uint64_t op_id = StableOpId(module_id, hlo_name);
          intermediate_op_metrics_map[{module_it->offload_core_id,
                                       module_it->tc_start_id}]
              .event_stack.Push({.event = event,
                                 .device_timespan = timespan,
                                 .offload_core_id = module_it->offload_core_id,
                                 .tc_start_id = module_it->tc_start_id,
                                 .op_id = op_id,
                                 .parent_op_id = 0});
        }
      });
    }
  });
  absl::flat_hash_map<std::pair<uint64_t, uint64_t>, uint64_t>
      total_duration_map;
  for (const auto& module : module_timespans) {
    const std::pair<uint64_t, uint64_t> key_id = {module.offload_core_id,
                                                  module.tc_start_id};
    if (auto it = intermediate_op_metrics_map.find(key_id);
        it != intermediate_op_metrics_map.end()) {
      it->second.event_stack.Flush();
      total_duration_map[key_id] += module.timespan.duration_ps();
    }
  }
  for (auto& [key, progress] : intermediate_op_metrics_map) {
    sparse_core_metrics_map[key] =
        progress.builder->FinalizeSorted(total_duration_map[key]);
  }
}

FlatOpMetricsDb ConvertTensorCoreDeviceTraceXPlaneToFlatOpMetricsDb(
    const XPlane& device_trace,
    const absl::flat_hash_map<std::pair<uint64_t, uint64_t>, FlatOpMetricsDb>&
        sparse_core_metrics_map) {
  XPlaneVisitor plane = tsl::profiler::CreateTfXPlaneVisitor(&device_trace);
  XEventsFlatOpMetricsDbBuilder builder;
  FlatOpMetricsDb sparse_core_metrics_db;
  FlatOpMetricsDbCombiner sparse_core_metrics_builder(&sparse_core_metrics_db);
  uint64_t first_op_timestamp_ps = std::numeric_limits<uint64_t>::max();
  uint64_t last_op_timestamp_ps = 0;

  absl::flat_hash_map<std::pair<uint64_t, uint64_t>, uint64_t>
      sc_offload_to_tc_caller_id_map;

  struct ParentReference {
    const XEventVisitor event;
    tsl::profiler::Timespan device_timespan;
    uint64_t children_duration_ps = 0;
    uint64_t op_id = 0;
    uint64_t parent_op_id = 0;
  };

  tsl::profiler::AncestorStack<ParentReference> event_stack(
      [&](const ParentReference& parent) {
        FlatOpMetrics op_metrics =
            XEventsFlatOpMetricsDbBuilder::FromXEvent(parent.event);
        op_metrics.set_time_ps(parent.device_timespan.duration_ps());
        op_metrics.set_self_time_ps(op_metrics.time_ps() -
                                    parent.children_duration_ps);
        op_metrics.set_op_id(parent.op_id);
        op_metrics.set_parent_op_id(parent.parent_op_id);
        std::optional<tsl::profiler::XStatVisitor> time_scale_multiplier_stat =
            parent.event.GetStat(StatType::kTimeScaleMultiplier);
        double factor = time_scale_multiplier_stat.has_value()
                            ? time_scale_multiplier_stat->DoubleValue()
                            : 1.0;
        op_metrics.set_normalized_time_ps(op_metrics.time_ps() * factor);
        op_metrics.set_core_type(FlatOpMetrics_TpuCoreType_TENSOR_CORE);
        std::optional<tsl::profiler::XStatVisitor> offload_core_id_stat =
            parent.event.GetStat(StatType::kOffloadCoreId);
        std::optional<tsl::profiler::XStatVisitor> offload_start_id_stat =
            parent.event.GetStat(StatType::kTcOffloadStartId);
        if (offload_core_id_stat.has_value() &&
            offload_start_id_stat.has_value()) {
          const std::pair<uint64_t, int64_t> offload_start_op_key = {
              offload_core_id_stat->UintValue(),
              offload_start_id_stat->IntValue()};
          if (auto it = sparse_core_metrics_map.find(offload_start_op_key);
              it != sparse_core_metrics_map.end()) {
            sc_offload_to_tc_caller_id_map[offload_start_op_key] = parent.op_id;
            op_metrics.set_has_sc_children(true);
          }
        }
        auto key = GetOpKeyFromXEvent(parent.event);
        XEventsFlatOpMetricsDbBuilder::OpKey flat_key;
        flat_key.program_id = key.program_id;
        flat_key.symbol_id = key.symbol_id;
        builder.AddOpMetric(op_metrics, flat_key);
      },
      [](const ParentReference& parent, const ParentReference& child) {
        return parent.device_timespan.Includes(child.device_timespan);
      },
      [](ParentReference& parent, ParentReference& child) {
        child.parent_op_id = parent.op_id;
        parent.children_duration_ps += child.device_timespan.duration_ps();
      });

  auto track_first_and_last_op_timestamps = [&](const XEventVisitor& event) {
    tsl::profiler::Timespan timespan = GetDeviceEventTimespan(event);
    first_op_timestamp_ps =
        std::min(first_op_timestamp_ps, timespan.begin_ps());
    last_op_timestamp_ps = std::max(last_op_timestamp_ps, timespan.end_ps());
  };

  plane.ForEachLine([&](const XLineVisitor& line) {
    if (line.Name() == tsl::profiler::kSparseCoreStepLineName ||
        line.Name() == tsl::profiler::kStepLineName) {
      line.ForEachEvent(track_first_and_last_op_timestamps);
    }
    if (!tsl::profiler::IsOpLineName(line.Name())) return;
    line.ForEachEvent([&](const XEventVisitor& event) {
      tsl::profiler::Timespan timespan = GetDeviceEventTimespan(event);
      track_first_and_last_op_timestamps(event);

      auto key = GetOpKeyFromXEvent(event);
      std::string hlo_name = event.Metadata().HasDisplayName()
                                 ? std::string(event.Metadata().DisplayName())
                                 : std::string(event.Metadata().Name());
      uint64_t op_id = StableOpId(key.program_id.value_or(0), hlo_name);
      event_stack.Push({.event = event,
                        .device_timespan = timespan,
                        .op_id = op_id,
                        .parent_op_id = 0});
    });
    event_stack.Flush();
  });

  for (auto [offload_start_op_key, parent_op_id] :
       sc_offload_to_tc_caller_id_map) {
    FlatOpMetricsDb sc_db = sparse_core_metrics_map.at(offload_start_op_key);
    for (auto& sc_op : *sc_db.mutable_op_instances()) {
      if (sc_op.parent_op_id() == 0) {
        sc_op.set_parent_op_id(parent_op_id);
      }
    }
    sparse_core_metrics_builder.Combine(sc_db);
  }

  uint64_t duration_ps = last_op_timestamp_ps > first_op_timestamp_ps
                             ? last_op_timestamp_ps - first_op_timestamp_ps
                             : 0;
  FlatOpMetricsDb tc_db = builder.Finalize(duration_ps);
  for (auto& op_instance : *sparse_core_metrics_db.mutable_op_instances()) {
    tc_db.add_op_instances()->Swap(&op_instance);
  }
  return tc_db;
}

}  // namespace profiler
}  // namespace tensorflow

namespace tensorflow {
namespace profiler {
namespace {

struct HLOTracker {
  uint64_t duration = 0;
  double vdd_energy_j = 0.0;
  uint64_t program_id = 0;
  uint64_t group_id = 0;
  bool is_eager;
  const HloInstructionWrapper* hlo_instruction = nullptr;
  std::string hlo_op_name;

  void Reset() {
    duration = program_id = group_id = 0;
    vdd_energy_j = 0.0;
    hlo_op_name.clear();
    hlo_instruction = nullptr;
  }
};

void AggregateHloFunc(HLOTracker& current,
                      DeviceFlatOpMetricsDbBuilder& builder) {
  if (current.hlo_instruction == nullptr) return;

  DeviceFlatOpMetricsDbBuilder::OpIdentifier op_id;
  op_id.program_id = current.program_id;
  op_id.name = current.hlo_op_name;
  op_id.category = current.hlo_instruction->Category();
  op_id.provenance = current.hlo_instruction->TfOpName();
  op_id.deduplicated_name = current.hlo_instruction->DeduplicatedName();
  op_id.long_name = current.hlo_instruction->Expression();
  op_id.op_source_info = current.hlo_instruction->SourceInfo();

  DeviceFlatOpMetricsDbBuilder::OpData op_data;
  op_data.is_eager = current.is_eager;
  op_data.occurrences = 1;
  op_data.time_ps = current.duration;
  op_data.children_time_ps = 0;
  op_data.vdd_energy_j = current.vdd_energy_j;
  op_data.perf_info = current.hlo_instruction->GetPerformanceInfoWrapper();

  builder.EnterOp(op_id, op_data);

  current.Reset();
}

}  // namespace

FlatOpMetricsDb ConvertDeviceTraceXPlaneToFlatOpMetricsDb(
    const XPlane& device_trace, const HloModuleMap& hlo_module_map) {
  FlatOpMetricsDb db;
  DeviceFlatOpMetricsDbBuilder builder(&db);

  int64_t first_op_offset_ps = std::numeric_limits<int64_t>::max();
  int64_t last_op_offset_ps = 0;
  int64_t num_tf_ops = 0;

  XPlaneVisitor plane = tsl::profiler::CreateTfXPlaneVisitor(&device_trace);
  HLOTracker current;

  plane.ForEachLine([&](const XLineVisitor& line) {
    if (tsl::profiler::IsDerivedThreadId(line.Id())) return;
    line.ForEachEvent([&](const XEventVisitor& event) {
      first_op_offset_ps = std::min(first_op_offset_ps, event.OffsetPs());
      last_op_offset_ps = std::max(last_op_offset_ps, event.EndOffsetPs());

      GpuEventStats stats(&event);
      if (stats.IsXlaOp()) {
        const auto* hlo_instruction = GetHloInstruction(
            hlo_module_map, stats.program_id, stats.hlo_op_names.back());
        if (hlo_instruction != nullptr) {
          if (stats.hlo_op_names.back() != current.hlo_op_name ||
              stats.group_id != current.group_id) {
            AggregateHloFunc(current, builder);
          }
          current.hlo_instruction = hlo_instruction;
          current.hlo_op_name = stats.hlo_op_names.back();
          current.duration += event.DurationPs();
          event.ForEachStat(
              [&current](const tsl::profiler::XStatVisitor& stat) {
                if (stat.Name() == "vdd_energy_j") {
                  current.vdd_energy_j += stat.DoubleValue();
                }
              });
          current.is_eager = stats.is_eager;
          current.program_id = *stats.program_id;
          if (stats.group_id.has_value()) {
            current.group_id = *stats.group_id;
          }
        }
      } else if (stats.IsTfOp()) {
        AggregateHloFunc(current, builder);
        tsl::profiler::TfOp tf_op =
            tsl::profiler::ParseTfOpFullname(stats.tf_op_fullname);
        if (tf_op.category != tsl::profiler::Category::kUnknown) {
          num_tf_ops++;
        }
        std::string name = absl::StrCat(tf_op.name, "/", event.Name());

        DeviceFlatOpMetricsDbBuilder::OpIdentifier op_id;
        op_id.program_id = 0;
        op_id.name = name;
        op_id.category = tf_op.type;
        op_id.provenance = stats.tf_op_fullname;
        op_id.deduplicated_name = "";

        DeviceFlatOpMetricsDbBuilder::OpData op_data;
        op_data.is_eager = stats.is_eager;
        op_data.occurrences = 1;
        op_data.time_ps = event.DurationPs();
        op_data.children_time_ps = 0;

        builder.EnterOp(op_id, op_data);
      }
    });
    if (num_tf_ops >= 5) {
      LOG(WARNING)
          << "TfOpRoofLineCostEstimator has been deprecated, but we "
          << "detected " << num_tf_ops << " TfOps. If you rely on "
          << "individual TfOp peak flops and bytes accessed estimates, "
          << "please open an issue on GitHub at openxla/xprof.";
    }
    AggregateHloFunc(current, builder);
  });

  uint64_t total_time_ps = last_op_offset_ps > first_op_offset_ps
                               ? last_op_offset_ps - first_op_offset_ps
                               : 0;
  SetTotalTimePs(db, total_time_ps);
  AddIdleOp(db);
  return db;
}

FlatOpMetricsDb ConvertHostThreadsXPlaneToFlatOpMetricsDb(
    const XPlane& host_trace) {
  FlatOpMetricsDb result;
  FlatOpMetricsDbCombiner combiner(&result);
  absl::flat_hash_map<int64_t, tsl::profiler::TfOp> tf_ops =
      CollectTfOpsFromHostThreadsXPlane(host_trace);
  XPlaneVisitor plane = tsl::profiler::CreateTfXPlaneVisitor(&host_trace);

  struct ParentReference {
    tsl::profiler::Timespan timespan;
    uint64_t children_duration_ps = 0;
    tsl::profiler::TfOp tf_op;
    bool is_eager = false;
  };

  FlatOpMetricsDb line_db;
  plane.ForEachLine([&](const XLineVisitor& line) {
    if (tsl::profiler::IsDerivedThreadId(line.Id())) return;

    line_db.Clear();
    HostFlatOpMetricsDbBuilder builder(&line_db);
    uint64_t first_op_timestamp_ps = std::numeric_limits<uint64_t>::max();
    uint64_t last_op_timestamp_ps = 0;

    tsl::profiler::AncestorStack<ParentReference> event_stack(
        [&](const ParentReference& parent) {
          builder.EnterOp(parent.tf_op.name, parent.tf_op.type, parent.is_eager,
                           parent.timespan.duration_ps(),
                           parent.children_duration_ps, parent.tf_op.id);
          if (tsl::profiler::IsInfeedEnqueueOp(parent.tf_op.type)) {
            builder.EnterHostInfeedEnqueue(parent.timespan);
          }
        },
        [](const ParentReference& parent, const ParentReference& child) {
          return parent.timespan.Includes(child.timespan);
        },
        [](ParentReference& parent, ParentReference& child) {
          parent.children_duration_ps += child.timespan.duration_ps();
        });

    line.ForEachEvent([&](const XEventVisitor& event) {
      if (event.Name().empty()) return;

      tsl::profiler::Timespan span = event.GetTimespan();
      first_op_timestamp_ps = std::min(first_op_timestamp_ps, span.begin_ps());
      last_op_timestamp_ps = std::max(last_op_timestamp_ps, span.end_ps());

      auto id = event.Id();
      if (const auto& stat = event.GetStat(StatType::kInputPipelineStageId);
          stat.has_value()) {
        id = stat->IntValue();
      }
      auto it = tf_ops.find(id);
      const tsl::profiler::TfOp* tf_op =
          (it != tf_ops.end()) ? &it->second : nullptr;

      tsl::profiler::TfOp parsed_tf_op;
      bool is_eager = false;

      if (tf_op != nullptr) {
        parsed_tf_op = *tf_op;
        if (std::optional<XStatVisitor> stat =
                event.GetStat(StatType::kIsEager)) {
          is_eager = stat->IntValue();
        }
      } else if (auto tf_op_stat = event.GetStat(StatType::kTfOp);
                 tf_op_stat.has_value()) {
        absl::string_view tf_op_fullname = tf_op_stat->StrOrRefValue();
        if (tf_op_fullname.empty()) return;
        parsed_tf_op = tsl::profiler::ParseTfOpFullname(tf_op_fullname);
      } else {
        return;
      }

      event_stack.Push({.timespan = span,
                        .tf_op = parsed_tf_op,
                        .is_eager = is_eager});
    });
    event_stack.Flush();

    uint64_t total_time_ps = last_op_timestamp_ps > first_op_timestamp_ps
                                 ? last_op_timestamp_ps - first_op_timestamp_ps
                                 : 0;
    SetTotalTimePs(line_db, total_time_ps);
    AddIdleOp(line_db);

    combiner.Combine(line_db, /*update_num_cores=*/false);
  });

  return result;
}

}  // namespace profiler
}  // namespace tensorflow
