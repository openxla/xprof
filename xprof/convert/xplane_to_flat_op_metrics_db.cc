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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/hash/hash.h"
#include "xla/tsl/profiler/utils/tf_xplane_visitor.h"
#include "xla/tsl/profiler/utils/timespan.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_utils.h"
#include "xla/tsl/profiler/utils/xplane_visitor.h"
#include "plugin/xprof/protobuf/flat_op_metrics.pb.h"
#include "plugin/xprof/protobuf/flat_op_metrics.pb.h"
#include "xprof/utils/flat_op_metrics_db_utils.h"
#include "xprof/utils/op_metrics_db_utils.h"

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
    std::shared_ptr<XEventsFlatOpMetricsDbBuilder> builder;
    tsl::profiler::AncestorStack<ParentReference> event_stack;
    OpMetricsInProgress()
        : builder(std::make_shared<XEventsFlatOpMetricsDbBuilder>()),
          event_stack(
              [builder = builder](const ParentReference& parent) {
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
                auto key = GetOpKeyFromXEvent(parent.event);
                XEventsFlatOpMetricsDbBuilder::OpKey flat_key;
                flat_key.program_id = key.program_id;
                flat_key.symbol_id = key.symbol_id;
                builder->AddOpMetric(op_metrics, flat_key);
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
          std::string hlo_name =
              event.Metadata().HasDisplayName()
                  ? std::string(event.Metadata().DisplayName())
                  : std::string(event.Metadata().Name());
          XEventsOpMetricsDbBuilder::OpKey key = GetOpKeyFromXEvent(event);
          auto module_id = key.program_id.has_value() ? key.program_id.value()
                                                      : module_it->tc_start_id;
          uint64_t op_id = absl::HashOf(module_id, hlo_name);
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

}  // namespace profiler
}  // namespace tensorflow
