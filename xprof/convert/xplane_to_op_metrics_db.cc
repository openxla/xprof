/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "xprof/convert/xplane_to_op_metrics_db.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/lib/gtl/map_util.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/profiler/utils/tf_op_utils.h"
#include "xla/tsl/profiler/utils/tf_xplane_visitor.h"
#include "xla/tsl/profiler/utils/timespan.h"
#include "xla/tsl/profiler/utils/trace_utils.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_utils.h"
#include "xla/tsl/profiler/utils/xplane_visitor.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/op_metrics_db_combiner.h"
#include "xprof/convert/op_stack.h"
#include "plugin/xprof/protobuf/op_metrics.pb.h"
#include "xprof/utils/cost_utils.h"
#include "xprof/utils/gpu_event_stats.h"
#include "xprof/utils/hlo_module_map.h"
#include "xprof/utils/op_metrics_db_utils.h"
#include "xprof/utils/op_utils.h"

namespace tensorflow {
namespace profiler {
namespace {

using ::tensorflow::profiler::GpuEventStats;
using ::tsl::profiler::GetDeviceEventTimespan;

struct HLOTracker {
  uint64_t duration = 0;
  uint64_t program_id = 0;
  uint64_t group_id = 0;
  bool is_eager;
  const HloInstructionWrapper* hlo_instruction = nullptr;
  std::string hlo_op_name;

  void Reset() {
    duration = program_id = group_id = 0;
    hlo_op_name.clear();
    hlo_instruction = nullptr;
  }
};

// Type of a TensorFlow Op activity, which is either beginning or ending an Op.
enum TfActivityType { kTfOpBegin, kTfOpEnd };

// Instant activity representing the begin or end of a host-side TF Op.
struct TfActivity {
  // The timestamp in picoseconds when this activity happened.
  uint64_t timestamp_ps;
  // The ID of this Op.
  uint32_t tf_op_id;
  // Type of this activity.
  TfActivityType activity_type;
  // Full TF op name and type of this activity (backed by XEvent::name).
  tsl::profiler::TfOp tf_op;
  // Whether it is eagerly executed.
  bool is_eager;
};

// TF Op metrics stored as element in OpStack.
struct TfOpInfo {
  explicit TfOpInfo(uint64_t ts) : start_timestamp_ps(ts) {}

  // Start timestamp in picoseconds.
  uint64_t start_timestamp_ps;
  // Children duration in picoseconds.
  uint64_t children_duration_ps = 0;
};

// Processes a TF-activity on particular core.
void ProcessOneTfActivity(const TfActivity& activity,
                          OpStack<TfOpInfo>* tf_op_stack,
                          TfMetricsDbData* tf_metrics_data) {
  uint32_t tf_op_id = activity.tf_op_id;
  switch (activity.activity_type) {
    case kTfOpBegin: {
      tf_op_stack->Push(tf_op_id,
                        std::make_unique<TfOpInfo>(activity.timestamp_ps));
      break;
    }
    case kTfOpEnd: {
      std::unique_ptr<TfOpInfo> info = tf_op_stack->Pop(tf_op_id);
      if (info == nullptr) {
        // This happens if TraceMes overlap.
        VLOG(1) << "No begin event found for TF activity id=" << tf_op_id
                << " name=" << activity.tf_op.name
                << " type=" << activity.tf_op.type;
        break;
      }
      tsl::profiler::Timespan tf_op_span = tsl::profiler::PicoSpan(
          info->start_timestamp_ps, activity.timestamp_ps);
      // Note the tf_op.id will be used as the hlo_module_id in EnterOp when
      // constructing the op metrics db.
      // - not set for legacy TfOp: behavior unchanged with hlo_module_id=0
      // - for input pipeline ops, this is the stage id.
      tf_metrics_data->tf_metrics_db_builder.EnterOp(
          activity.tf_op.name, activity.tf_op.type, activity.is_eager,
          tf_op_span.duration_ps(), info->children_duration_ps,
          activity.tf_op.id);
      TfOpInfo* parent_info = tf_op_stack->Top();
      if (parent_info != nullptr) {
        parent_info->children_duration_ps += tf_op_span.duration_ps();
      }
      if (tsl::profiler::IsInfeedEnqueueOp(activity.tf_op.type)) {
        tf_metrics_data->tf_metrics_db_builder.EnterHostInfeedEnqueue(
            tf_op_span);
      }
      break;
    }
  }
}

// Processes all TF-activities on the given core.
void ProcessTfActivities(std::vector<TfActivity>* tf_activities,
                         TfMetricsDbData* tf_metrics_db_data) {
  if (tf_activities->empty()) return;
  absl::c_stable_sort(*tf_activities,
                      [](const TfActivity& a, const TfActivity& b) {
                        return a.timestamp_ps < b.timestamp_ps;
                      });
  OpStack<TfOpInfo> tf_op_stack;
  for (const auto& tf_activity : *tf_activities) {
    ProcessOneTfActivity(tf_activity, &tf_op_stack, tf_metrics_db_data);
  }
  SetTotalTimePs(
      tf_metrics_db_data->tf_metrics_db,
      tf_activities->back().timestamp_ps - tf_activities->front().timestamp_ps);
}

void CollectTfActivities(
    const XLineVisitor& line,
    const absl::flat_hash_map<int64_t, tsl::profiler::TfOp>& tf_ops,
    std::vector<TfActivity>* tf_activities) {
  uint32_t tf_op_id = 0;
  if (tsl::profiler::IsDerivedThreadId(line.Id())) return;
  tf_activities->reserve(line.NumEvents() * 2);
  line.ForEachEvent(
      [&tf_ops, &tf_op_id, &tf_activities](const XEventVisitor& event) {
        auto id = event.Id();
        // Add id override for input pipeline ops.
        if (const auto& stat = event.GetStat(StatType::kInputPipelineStageId);
            stat.has_value()) {
          id = stat->IntValue();
        }
        const tsl::profiler::TfOp* tf_op = tsl::gtl::FindOrNull(tf_ops, id);
        if (tf_op != nullptr) {
          ++tf_op_id;
          bool is_eager = false;
          if (std::optional<XStatVisitor> stat =
                  event.GetStat(StatType::kIsEager)) {
            is_eager = stat->IntValue();
          }
          tsl::profiler::Timespan span = event.GetTimespan();
          tf_activities->push_back(
              {span.begin_ps(), tf_op_id, kTfOpBegin, *tf_op, is_eager});
          tf_activities->push_back(
              {span.end_ps(), tf_op_id, kTfOpEnd, *tf_op, is_eager});
        }
        if (auto tf_op_stat = event.GetStat(StatType::kTfOp);
            tf_op_stat.has_value()) {
          absl::string_view tf_op_fullname = tf_op_stat->StrOrRefValue();
          if (tf_op_fullname.empty()) {
            return;
          }
          ++tf_op_id;
          tsl::profiler::TfOp tf_op =
              tsl::profiler::ParseTfOpFullname(tf_op_fullname);
          tsl::profiler::Timespan span = event.GetTimespan();
          tf_activities->push_back(
              {span.begin_ps(), tf_op_id, kTfOpBegin, tf_op, false});
          tf_activities->push_back(
              {span.end_ps(), tf_op_id, kTfOpEnd, tf_op, false});
        }
      });
}

}  // namespace

TfMetricsDbData ConvertHostThreadsXLineToTfMetricsDbData(
    const XLineVisitor& line,
    const absl::flat_hash_map<int64_t, tsl::profiler::TfOp>& tf_ops) {
  TfMetricsDbData tf_metrics_db_data;
  std::vector<TfActivity> tf_activities;
  CollectTfActivities(line, tf_ops, &tf_activities);
  ProcessTfActivities(&tf_activities, &tf_metrics_db_data);
  return tf_metrics_db_data;
}

void ConsumeTfMetricsDbData(TfMetricsDbData src, OpMetricsDbCombiner* dst) {
  AddIdleOp(src.tf_metrics_db);
  // Host OpMetricsDb does not need to update the number of cores a certain op
  // occurs.
  dst->Combine(src.tf_metrics_db, /*update_num_cores=*/false);
  src.tf_metrics_db.Clear();
}

absl::flat_hash_map<int64_t, tsl::profiler::TfOp>
CollectTfOpsFromHostThreadsXPlane(const XPlane& host_trace) {
  absl::flat_hash_map<int64_t, tsl::profiler::TfOp> tf_ops;
  XPlaneVisitor plane = tsl::profiler::CreateTfXPlaneVisitor(&host_trace);
  plane.ForEachLine([&tf_ops](const XLineVisitor& line) {
    line.ForEachEvent([&tf_ops](const XEventVisitor& event) {
      // 0. Skip events that does not have valid name - which is necessary
      // for the host event parsing
      if (event.Name().empty()) return;

      // 1. Newly added input pipeline ops processing: identified by the
      // stage id and category.
      auto input_pipeline_stage_id =
          event.GetStat(StatType::kInputPipelineStageId);
      if (input_pipeline_stage_id.has_value()) {
        auto input_pipeline_stage_category =
            event.GetStat(StatType::kInputPipelineStageCategory);
        if (input_pipeline_stage_category.has_value()) {
          tsl::profiler::TfOp tf_op = tsl::profiler::ParseTfOpFullname(
              event.Name(), tsl::profiler::Category::kInputPipeline,
              input_pipeline_stage_category->StrOrRefValue(),
              input_pipeline_stage_id->IntValue());
          // Note using input pipeline stage id as unique identifier here
          // instead of events id, because event id's uniqueness is bind
          // with the event name string due to nature of xplane event
          // metadata creation, making it a non-sufficient identifier when
          // building an input pipeline event stack.
          tf_ops.try_emplace(input_pipeline_stage_id->IntValue(), tf_op);
        }
        return;
      }

      // 2. Fallback to legacy host ops processing.
      // On the host, we have added some user-specified TraceMe's in
      // addition to the TraceMe's added to every TensorFlow op by the
      // system. These user-inserted TraceMe's have "unknown" type. We don't
      // count them in Tf-stats.
      tsl::profiler::TfOp tf_op =
          tsl::profiler::ParseTfOpFullname(event.Name());
      if (tf_op.category != tsl::profiler::Category::kUnknown) {
        tf_ops.try_emplace(event.Id(), tf_op);
      }
    });
  });
  return tf_ops;
}

OpMetricsDb ConvertHostThreadsXPlaneToOpMetricsDb(const XPlane& host_trace) {
  OpMetricsDb result;
  OpMetricsDbCombiner combiner(&result);
  absl::flat_hash_map<int64_t, tsl::profiler::TfOp> tf_ops =
      CollectTfOpsFromHostThreadsXPlane(host_trace);
  XPlaneVisitor plane = tsl::profiler::CreateTfXPlaneVisitor(&host_trace);
  plane.ForEachLine([&tf_ops, &combiner](const XLineVisitor& line) {
    ConsumeTfMetricsDbData(
        ConvertHostThreadsXLineToTfMetricsDbData(line, tf_ops), &combiner);
  });
  return result;
}

OpMetricsDb ConvertTensorCoreDeviceTraceXPlaneToOpMetricsDb(
    const XPlane& device_trace,
    const absl::flat_hash_map<std::pair<uint64_t, uint64_t>, OpMetricsDb>&
        sparse_core_metrics_map) {
  XPlaneVisitor plane = tsl::profiler::CreateTfXPlaneVisitor(&device_trace);
  XEventsOpMetricsDbBuilder builder;
  uint64_t first_op_timestamp_ps = std::numeric_limits<uint64_t>::max();
  uint64_t last_op_timestamp_ps = 0;

  struct ParentReference {
    const XEventVisitor event;
    tsl::profiler::Timespan device_timespan;
    uint64_t children_duration_ps = 0;
  };

  tsl::profiler::AncestorStack<ParentReference> event_stack(
      [&](const ParentReference& parent) {
        OpMetrics op_metrics = FromXEvent(parent.event);
        op_metrics.set_time_ps(parent.device_timespan.duration_ps());
        op_metrics.set_self_time_ps(op_metrics.time_ps() -
                                    parent.children_duration_ps);
        std::optional<tsl::profiler::XStatVisitor> time_scale_multiplier_stat =
            parent.event.GetStat(StatType::kTimeScaleMultiplier);
        double factor = time_scale_multiplier_stat.has_value()
                            ? time_scale_multiplier_stat->DoubleValue()
                            : 1.0;
        op_metrics.set_normalized_time_ps(op_metrics.time_ps() * factor);
        op_metrics.set_core_type(OpMetrics_TpuCoreType_TENSOR_CORE);
        std::optional<tsl::profiler::XStatVisitor> offload_core_id_stat =
            parent.event.GetStat(StatType::kOffloadCoreId);
        std::optional<tsl::profiler::XStatVisitor> offload_start_id_stat =
            parent.event.GetStat(StatType::kTcOffloadStartId);
        if (offload_core_id_stat.has_value() &&
            offload_start_id_stat.has_value()) {
          const std::pair<uint64_t, uint64_t> offload_start_op_key = {
              offload_core_id_stat->UintValue(),
              offload_start_id_stat->IntValue()};
          if (auto it = sparse_core_metrics_map.find(offload_start_op_key);
              it != sparse_core_metrics_map.end()) {
            OpMetricsDbCombiner combiner(op_metrics.mutable_children());
            combiner.Combine(it->second, /*update_num_cores=*/false);
          }
        }
        builder.AddOpMetric(op_metrics, GetOpKeyFromXEvent(parent.event));
      },
      [](const ParentReference& parent, const ParentReference& child) {
        return parent.device_timespan.Includes(child.device_timespan);
      },
      [](ParentReference& parent, ParentReference& child) {
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

      event_stack.Push({.event = event, .device_timespan = timespan});
    });
    event_stack.Flush();
  });

  return builder.Finalize(last_op_timestamp_ps - first_op_timestamp_ps);
}

void ConvertSparseCoreDeviceTraceXPlaneToOpMetricsDb(
    const XPlane& device_trace,
    absl::flat_hash_map<std::pair<uint64_t, uint64_t>, OpMetricsDb>&
        sparse_core_metrics_map) {
  XPlaneVisitor plane = tsl::profiler::CreateTfXPlaneVisitor(&device_trace);

  struct ParentReference {
    const XEventVisitor event;
    tsl::profiler::Timespan device_timespan;
    uint64_t offload_core_id;
    uint64_t tc_start_id;
    uint64_t children_duration_ps = 0;
  };
  struct OpMetricsInProgress {
    std::shared_ptr<XEventsOpMetricsDbBuilder> builder;
    tsl::profiler::AncestorStack<ParentReference> event_stack;
    OpMetricsInProgress()
        : builder(std::make_shared<XEventsOpMetricsDbBuilder>()),
          event_stack(
              [builder = builder](const ParentReference& parent) {
                OpMetrics op_metrics = FromXEvent(parent.event);
                op_metrics.set_time_ps(parent.device_timespan.duration_ps());
                op_metrics.set_self_time_ps(op_metrics.time_ps() -
                                            parent.children_duration_ps);
                op_metrics.set_core_type(OpMetrics_TpuCoreType_SPARSE_CORE);
                XEventsOpMetricsDbBuilder::OpKey key =
                    GetOpKeyFromXEvent(parent.event);
                builder->AddOpMetric(op_metrics, key);
              },
              [](const ParentReference& parent, const ParentReference& child) {
                return parent.device_timespan.Includes(child.device_timespan);
              },
              [](ParentReference& parent, ParentReference& child) {
                parent.children_duration_ps +=
                    child.device_timespan.duration_ps();
              }) {};
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
        const std::optional<XStatVisitor> offload_core_id =
            event.GetStat(StatType::kOffloadCoreId);
        const std::optional<XStatVisitor> tc_start_id =
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
          // Insert that event into the stack for that module
          intermediate_op_metrics_map[{module_it->offload_core_id,
                                       module_it->tc_start_id}]
              .event_stack.Push({.event = event,
                                 .device_timespan = timespan,
                                 .offload_core_id = module_it->offload_core_id,
                                 .tc_start_id = module_it->tc_start_id});
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
        progress.builder->Finalize(total_duration_map[key]);
  }
}

void AggregateHloFunc(HLOTracker& current, DeviceOpMetricsDbBuilder& metricDb) {
  if (current.hlo_instruction == nullptr) return;

  DeviceOpMetricsDbBuilder::OpIdentifier op_id = {
      .program_id = current.program_id,
      .name = current.hlo_op_name,
      .category = current.hlo_instruction->Category(),
      .provenance = current.hlo_instruction->TfOpName(),
      .deduplicated_name = current.hlo_instruction->DeduplicatedName(),
      .long_name = current.hlo_instruction->Expression(),
      .op_source_info = current.hlo_instruction->SourceInfo(),
  };

  DeviceOpMetricsDbBuilder::OpData event_data = {
      .is_eager = current.is_eager,
      .occurrences = 1,
      .time_ps = current.duration,
      .children_time_ps = 0,
      .perf_info = current.hlo_instruction->GetPerformanceInfoWrapper(),
  };

  metricDb.EnterOp(op_id, event_data);
  current.Reset();
}

OpMetricsDb ConvertDeviceTraceXPlaneToOpMetricsDb(
    const XPlane& device_trace, const HloModuleMap& hlo_module_map) {
  OpMetricsDb result;
  DeviceOpMetricsDbBuilder device_op_metrics_db_builder(&result);

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
            AggregateHloFunc(current, device_op_metrics_db_builder);
          }
          // Merge identical and contiguous HLOs.
          current.hlo_instruction = hlo_instruction;
          current.hlo_op_name = stats.hlo_op_names.back();
          current.duration += event.DurationPs();
          current.is_eager = stats.is_eager;
          current.program_id = *stats.program_id;
          if (stats.group_id.has_value()) {
            current.group_id = *stats.group_id;
          }
        }
      } else if (stats.IsTfOp()) {
        AggregateHloFunc(current, device_op_metrics_db_builder);
        tsl::profiler::TfOp tf_op =
            tsl::profiler::ParseTfOpFullname(stats.tf_op_fullname);
        if (tf_op.category != tsl::profiler::Category::kUnknown) {
          num_tf_ops++;
        }
        std::string name = absl::StrCat(tf_op.name, "/", event.Name());
        DeviceOpMetricsDbBuilder::OpIdentifier op_id = {
            .program_id = 0,
            .name = name,
            .category = tf_op.type,
            .provenance = stats.tf_op_fullname,
            .deduplicated_name = "",
        };
        DeviceOpMetricsDbBuilder::OpData event_data = {
            .is_eager = stats.is_eager,
            .occurrences = 1,
            .time_ps = static_cast<uint64_t>(event.DurationPs()),
            .children_time_ps = 0,
            .flops = 0,
            .bytes_accessed = 0,
        };
        device_op_metrics_db_builder.EnterOp(op_id, event_data);
      }
    });
    if (num_tf_ops >= 5) {
      LOG(WARNING)
          << "TfOpRoofLineCostEstimator has been deprecated, but we"
          << "detected " << num_tf_ops << " TfOps. If you rely on "
          << "individual TfOp peak flops and bytes accessed estimates, "
          << "please open an issue on GitHub at openxla/xprof.";
    }
    AggregateHloFunc(current, device_op_metrics_db_builder);
  });
  SetTotalTimePs(
      result, last_op_offset_ps ? last_op_offset_ps - first_op_offset_ps : 0);
  AddIdleOp(result);
  return result;
}

}  // namespace profiler
}  // namespace tensorflow
