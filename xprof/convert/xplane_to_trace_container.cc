/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "xprof/convert/xplane_to_trace_container.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/optimization.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "re2/re2.h"
#include "xla/tsl/profiler/utils/tf_xplane_visitor.h"
#include "xla/tsl/profiler/utils/timespan.h"
#include "xla/tsl/profiler/utils/trace_utils.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_utils.h"
#include "xla/tsl/profiler/utils/xplane_visitor.h"
#include "xprof/convert/trace_viewer/trace_event_arguments_builder.h"
#include "xprof/convert/trace_viewer/trace_events_util.h"
#include "plugin/xprof/protobuf/trace_events.pb.h"
#include "plugin/xprof/protobuf/trace_events_raw.pb.h"

namespace tensorflow {
namespace profiler {
namespace {

using tsl::profiler::FindPlanesWithPrefix;
using tsl::profiler::HostEventType;
using tsl::profiler::StatType;
using tsl::profiler::XEventVisitor;
using tsl::profiler::XFlow;
using tsl::profiler::XLineVisitor;
using tsl::profiler::XPlaneVisitor;
using tsl::profiler::XStatVisitor;

std::string GetThreadPoolName(absl::string_view resource_name) {
  std::string thread_name;
  static LazyRE2 pattern = {R"((.*)\/\d+(:?-.*)?)"};
  if (!RE2::FullMatch(resource_name, *pattern, &thread_name)) return "";
  static LazyRE2 pattern2 = {R"((.*)[-_]\d+)"};
  std::string pool_name;
  if (RE2::FullMatch(thread_name, *pattern2, &pool_name)) return pool_name;
  return thread_name;
}

class ThreadPoolGrouper : public ResourceGrouperInterface {
 public:
  ThreadPoolGrouper(uint32_t start_device_id, uint32_t first_pool_device_id,
                    uint32_t max_pool_device_id,
                    const XPlaneVisitor& plane_visitor, int pid_offset = 0)
      : start_device_id_(start_device_id),
        first_pool_device_id_(first_pool_device_id),
        max_pool_device_id_(max_pool_device_id),
        pid_offset_(pid_offset) {
    GroupThreadsByPool(plane_visitor);
  }

  int pids_used() const { return thread_pools_.size() + pids_used_; }

  std::vector<std::pair<uint32_t, absl::string_view>> Devices() const override {
    std::vector<std::pair<uint32_t, absl::string_view>> devices;
    devices.push_back({start_device_id_, plane_name_});
    for (const auto& pool : thread_pools_) {
      if (pool->num_threads == 0) continue;
      devices.push_back({pool->device_id, pool->name});
    }
    return devices;
  }

  uint32_t GetDeviceId(uint32_t resource_id) const override {
    auto it = thread_pool_map_.find(resource_id);
    if (it == thread_pool_map_.end()) {
      return start_device_id_;
    }
    return it->second.pool->device_id;
  }

 private:
  struct ThreadPoolInfo {
    ThreadPoolInfo(uint32_t id, absl::string_view n) : device_id(id), name(n) {}
    uint32_t device_id;
    std::string name;
    uint32_t num_threads = 0;
  };
  struct ThreadInfo {
    ThreadPoolInfo* pool;
    absl::string_view name;
  };

  void GroupThreadsByPool(const XPlaneVisitor& plane_visitor) {
    plane_name_ = plane_visitor.Name();
    uint32_t initial_pid = start_device_id_;
    if (pid_offset_ > 0) {
      initial_pid = first_pool_device_id_ + pid_offset_ - 1;
    }
    if (initial_pid > max_pool_device_id_) {
      initial_pid = max_pool_device_id_;
    }

    auto default_pool =
        std::make_unique<ThreadPoolInfo>(initial_pid, plane_visitor.Name());
    ThreadPoolInfo* default_pool_ptr = default_pool.get();
    thread_pools_.push_back(std::move(default_pool));

    plane_visitor.ForEachLine([&](const XLineVisitor& line) {
      uint32_t tid = line.DisplayId();
      thread_pool_map_[tid].name = line.DisplayName();
      thread_pool_map_[tid].pool = default_pool_ptr;
      ++default_pool_ptr->num_threads;
    });

    absl::flat_hash_map<std::string, std::vector<ThreadInfo*>> groups;
    for (auto& [tid, info] : thread_pool_map_) {
      std::string pool_name = GetThreadPoolName(info.name);
      groups[pool_name].push_back(&info);
      VLOG(1) << info.name << " assigned to '" << pool_name << "'";
    }

    if (groups.size() <= 1 || initial_pid >= max_pool_device_id_) return;

    const size_t max_pools = max_pool_device_id_ - first_pool_device_id_ + 1;
    const int available_pools = max_pools - pid_offset_;
    if (available_pools <= 0) return;

    std::vector<std::pair<absl::string_view, size_t>> pool_name_and_size;
    for (const auto& [name, group] : groups) {
      if (name.empty()) continue;
      if (group.size() == 1) continue;
      pool_name_and_size.push_back(
          std::make_pair(absl::string_view(name), group.size()));
    }

    if (pool_name_and_size.size() > available_pools) {
      std::partial_sort(pool_name_and_size.begin(),
                        pool_name_and_size.begin() + available_pools,
                        pool_name_and_size.end(),
                        [](const std::pair<absl::string_view, size_t>& left,
                           const std::pair<absl::string_view, size_t>& right) {
                          return left.second > right.second;
                        });
      pool_name_and_size.resize(available_pools);
    }

    std::sort(pool_name_and_size.begin(), pool_name_and_size.end(),
              [](const std::pair<absl::string_view, size_t>& left,
                 const std::pair<absl::string_view, size_t>& right) {
                return left.first < right.first;
              });

    for (int i = 0; i < pool_name_and_size.size(); ++i) {
      const auto& [pool_name, num_threads] = pool_name_and_size[i];
      uint32_t pool_device_id = first_pool_device_id_ + pid_offset_ + i;
      if (pool_device_id > max_pool_device_id_) break;
      thread_pools_.push_back(
          std::make_unique<ThreadPoolInfo>(pool_device_id, pool_name));
      for (auto* thread : groups[pool_name]) {
        --thread->pool->num_threads;
        thread->pool = thread_pools_.back().get();
        ++thread->pool->num_threads;
      }
    }
  }

  uint32_t start_device_id_;
  uint32_t first_pool_device_id_;
  uint32_t max_pool_device_id_;
  int pid_offset_;
  std::string plane_name_;
  int pids_used_ = 0;
  std::vector<std::unique_ptr<ThreadPoolInfo>> thread_pools_;
  absl::flat_hash_map<uint32_t, ThreadInfo> thread_pool_map_;
};

struct SpecialArguments {
  std::optional<int64_t> group_id;
  absl::string_view step_name;
  bool is_async_event = false;
  // Both flow and async events share the flow specification.
  std::optional<XFlow> flow;
};

inline TraceEvent::FlowEntryType FlowEntryTypeFromDirection(
    XFlow::FlowDirection direction) {
  switch (direction) {
    case XFlow::kFlowUnspecified:
      return TraceEvent::FLOW_NONE;
    case XFlow::kFlowIn:
      return TraceEvent::FLOW_END;
    case XFlow::kFlowOut:
      return TraceEvent::FLOW_START;
    case XFlow::kFlowInOut:
      return TraceEvent::FLOW_MID;
  }
}

template <typename T>
void ConvertXStatToTraceEventArgument(const XStatVisitor& stat, T value,
                                      SpecialArguments& special_args,
                                      TraceEventArgumentsBuilder& args) {
  if (stat.Type() == StatType::kFlow) {
    special_args.flow = XFlow::FromStatValue(value);
  } else if (stat.Type() == StatType::kGroupId) {
    special_args.group_id = value;
  } else if (stat.Type() == StatType::kIsAsync) {
    special_args.is_async_event = true;
  } else {
    args.Append(stat.Name(), value);
  }
}

SpecialArguments ConvertXStatsToTraceEventArguments(
    const XEventVisitor& event, RawData* raw_data,
    TraceEventArguments* raw_args) {
  TraceEventArgumentsBuilder args(raw_args);
  SpecialArguments special_args;
  auto for_each_stat = [&special_args, &args](const XStatVisitor& stat) {
    if (tsl::profiler::IsInternalStat(stat.Type())) return;
    switch (stat.ValueCase()) {
      case XStat::kInt64Value:
        ConvertXStatToTraceEventArgument(stat, stat.IntValue(), special_args,
                                         args);
        break;
      case XStat::kUint64Value:
        ConvertXStatToTraceEventArgument(stat, stat.UintValue(), special_args,
                                         args);
        break;
      case XStat::kDoubleValue:
        args.Append(stat.Name(), stat.DoubleValue());
        break;
      case XStat::kStrValue:
      case XStat::kRefValue: {
        auto stat_value = stat.StrOrRefValue();
        if (stat.Type() == StatType::kStepName) {
          special_args.step_name = stat_value;
        }
        args.Append(stat.Name(), stat_value);
        break;
      }
      case XStat::kBytesValue:
        break;
      case XStat::VALUE_NOT_SET:
        break;
    }
  };
  // Ensure the metadata stats appear before the per-occurrence stats.
  event.Metadata().ForEachStat(for_each_stat);
  event.ForEachStat(for_each_stat);
  return special_args;
}

void ConvertXLineToTraceEventsContainer(uint32_t device_id,
                                        const XLineVisitor& line,
                                        TraceEventsContainer* container) {
  std::optional<uint32_t> resource_id;

  if (line.Name() != tsl::profiler::kCounterEventsLineName) {
    resource_id = line.DisplayId();
    Resource* resource = container->MutableResource(*resource_id, device_id);
    resource->set_resource_id(*resource_id);
    resource->set_name(std::string(line.DisplayName()));
    resource->set_num_events(line.NumEvents());
  }

  RawData raw_data;  // hoisted for performance
  line.ForEachEvent([device_id, resource_id, &raw_data,
                     container](const XEventVisitor& event) {
    int64_t event_type =
        event.Type().value_or(HostEventType::kUnknownHostEventType);
    if (tsl::profiler::IsInternalEvent(event_type)) return;
    TraceEventArguments* raw_args = raw_data.mutable_args();
    absl::string_view event_name;
    if (event.HasDisplayName()) {
      event_name = event.DisplayName();
      TraceEventArgumentsBuilder args(raw_args);
      constexpr size_t kMaxLongName = 10000;
      if (event.Name().size() > kMaxLongName) {
        args.Append("long_name",
                    absl::StrCat(event.Name().substr(0, kMaxLongName),
                                 "...<truncated>"));
      } else {
        args.Append("long_name", event.Name());
      }
    } else {
      event_name = event.Name();
    }
    SpecialArguments special_args =
        ConvertXStatsToTraceEventArguments(event, &raw_data, raw_args);
    if (!special_args.step_name.empty()) {
      event_name = special_args.step_name;
    }
    if (!resource_id) {
      container->AddCounterEvent(event_name, device_id, event.TimestampPs(),
                                 raw_data);
    } else if (special_args.flow) {
      tsl::profiler::Timespan span(event.TimestampPs(), event.DurationPs());
      if (special_args.is_async_event) {
        container->AddAsyncEvent(
            event_name, device_id, span, special_args.flow->Id(),
            FlowEntryTypeFromDirection(special_args.flow->Direction()),
            special_args.flow->Category(), &raw_data, special_args.group_id);
      } else {
        container->AddFlowEvent(
            event_name, *resource_id, device_id, span, special_args.flow->Id(),
            FlowEntryTypeFromDirection(special_args.flow->Direction()),
            special_args.flow->Category(), &raw_data, special_args.group_id);
      }
    } else {
      tsl::profiler::Timespan span(event.TimestampPs(), event.DurationPs());
      container->AddCompleteEvent(event_name, *resource_id, device_id, span,
                                  &raw_data, special_args.group_id);
    }
    // Cleanup hoisted structure for next event.
    if (raw_data.has_args()) raw_args->clear_arg();
  });
}

void ConvertXPlaneToTraceEventsContainer(
    uint64_t device_id, absl::string_view hostname, const XPlane& xplane,
    TraceEventsContainer* container,
    std::unique_ptr<ResourceGrouperInterface> resource_grouper = nullptr) {
  XPlaneVisitor plane = tsl::profiler::CreateTfXPlaneVisitor(&xplane);
  if (resource_grouper == nullptr) {
    resource_grouper = CreateDefaultResourceGrouper(device_id, plane.Name());
  }

  if (plane.NumLines() == 0) return;

  for (const auto& [device_id, name] : resource_grouper->Devices()) {
    Device* device = container->MutableDevice(device_id);
    device->set_device_id(device_id);
    device->set_name(absl::StrCat(hostname, " ", name));
  }

  plane.ForEachLine([&](const XLineVisitor& line) {
    if (line.NumEvents() == 0) return;
    if (absl::StartsWith(line.Name(), "counters_")) return;
    // Capture a copy of XLineVisitor because it will go out of scope.
    uint32_t device_id = resource_grouper->GetDeviceId(line.DisplayId());
    ConvertXLineToTraceEventsContainer(device_id, line, container);
  });
}

}  // namespace

void ConvertXSpaceToTraceEventsContainer(absl::string_view hostname,
                                         const XSpace& space,
                                         TraceEventsContainer* container) {
  std::vector<const XPlane*> host_planes =
      FindPlanesWithPrefix(space, tsl::profiler::kHostThreadsPlaneName);
  if (!host_planes.empty()) {
    uint32_t host_device_id = tsl::profiler::kHostThreadsDeviceId;
    absl::c_sort(host_planes, [](const XPlane* a, const XPlane* b) {
      return a->name() < b->name();
    });
    int pid_offset = 0;
    for (const XPlane* host_plane : host_planes) {
      XPlaneVisitor plane = tsl::profiler::CreateTfXPlaneVisitor(host_plane);
      uint32_t first_pool_device_id = tsl::profiler::kHostThreadsDeviceId + 1;
      uint32_t max_pool_device_id = tsl::profiler::kHostThreadsDeviceId + 1000;
      auto thread_pool_grouper = std::make_unique<ThreadPoolGrouper>(
          host_device_id, first_pool_device_id, max_pool_device_id, plane,
          pid_offset);
      int used = thread_pool_grouper->pids_used();
      ConvertXPlaneToTraceEventsContainer(host_device_id, hostname, *host_plane,
                                          container,
                                          std::move(thread_pool_grouper));
      pid_offset += used;
      host_device_id++;
    }
  }

  std::vector<const XPlane*> device_planes =
      FindPlanesWithPrefix(space, tsl::profiler::kGpuPlanePrefix);

  if (device_planes.empty()) {
    device_planes = FindPlanesWithPrefix(space, tsl::profiler::kTpuPlanePrefix);
  }

  for (const XPlane* device_plane : device_planes) {
    uint32_t device_pid = tsl::profiler::kFirstDeviceId + device_plane->id();
    if (ABSL_PREDICT_FALSE(device_pid > tsl::profiler::kLastDeviceId)) {
      device_pid = tsl::profiler::kFirstDeviceId;
    }
    ConvertXPlaneToTraceEventsContainer(device_pid, hostname, *device_plane,
                                        container);
  }
  for (const XPlane* custom_plane :
       FindPlanesWithPrefix(space, tsl::profiler::kCustomPlanePrefix)) {
    ConvertXPlaneToTraceEventsContainer(
        tsl::profiler::kFirstCustomPlaneDeviceId + custom_plane->id(), hostname,
        *custom_plane, container);
  }
}

}  // namespace profiler
}  // namespace tensorflow
