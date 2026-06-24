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

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/optimization.h"
#include "absl/strings/match.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/arena.h"
#include "xla/tsl/profiler/utils/tf_xplane_visitor.h"
#include "xla/tsl/profiler/utils/timespan.h"
#include "xla/tsl/profiler/utils/trace_utils.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_utils.h"
#include "xla/tsl/profiler/utils/xplane_visitor.h"
#include "xprof/convert/trace_viewer/lite_trace_events.h"
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

void ConvertXPlaneToTraceEventsContainer(uint64_t device_id,
                                         absl::string_view hostname,
                                         const XPlane& xplane,
                                         TraceEventsContainer* container) {
  XPlaneVisitor plane = tsl::profiler::CreateTfXPlaneVisitor(&xplane);
  std::unique_ptr<ResourceGrouperInterface> resource_grouper =
      CreateDefaultResourceGrouper(device_id, plane.Name());

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

void ConvertXPlaneToLiteTraceEventsContainer(
    uint32_t device_id, absl::string_view hostname, const XPlane& plane,
    TraceEventLiteContainer* container) {
  auto plane_visitor = std::make_shared<const tsl::profiler::XPlaneVisitor>(
      tsl::profiler::CreateTfXPlaneVisitor(&plane));
  std::unique_ptr<ResourceGrouperInterface> resource_grouper =
      CreateDefaultResourceGrouper(device_id, plane_visitor->Name());

  if (plane_visitor->NumLines() == 0) return;

  for (const auto& [dev_id, name] : resource_grouper->Devices()) {
    auto& device = (*container->trace.mutable_devices())[dev_id];
    device.set_device_id(dev_id);
    device.set_name(absl::StrCat(hostname, " ", name));
  }

  absl::flat_hash_map<TraceTrackMetadata, const TraceTrackMetadata*>
      metadata_ptr_map;
  auto get_or_create_metadata_ptr =
      [&](const TraceTrackMetadata& meta) -> const TraceTrackMetadata* {
    auto [it, inserted] = metadata_ptr_map.try_emplace(meta, nullptr);
    if (inserted) {
      auto new_meta = std::make_unique<TraceTrackMetadata>(meta);
      it->second = new_meta.get();
      container->tracks_metadata.push_back(std::move(new_meta));
    }
    return it->second;
  };

  for (size_t line_idx = 0; line_idx < plane.lines_size(); ++line_idx) {
    const auto& line = plane.lines(line_idx);
    if (line.events_size() == 0) continue;
    tsl::profiler::XLineVisitor line_visitor(plane_visitor.get(), &line);
    bool is_counter_line =
        (line_visitor.Name() == tsl::profiler::kCounterEventsLineName);
    uint32_t resource_id = static_cast<uint32_t>(line_visitor.DisplayId());
    uint32_t line_device_id =
        resource_grouper->GetDeviceId(resource_id);

    if (!is_counter_line) {
      auto& device = (*container->trace.mutable_devices())[line_device_id];
      auto& resource =
          (*device.mutable_resources())[resource_id];
      resource.set_resource_id(resource_id);
      resource.set_name(std::string(line_visitor.DisplayName()));
      resource.set_num_events(line_visitor.NumEvents());
    }

    for (size_t event_idx = 0; event_idx < line.events_size(); ++event_idx) {
      const auto& event = line.events(event_idx);
      tsl::profiler::XEventVisitor event_visitor(plane_visitor.get(), &line,
                                                 &event);
      int64_t event_type =
          event_visitor.Type().value_or(HostEventType::kUnknownHostEventType);
      if (tsl::profiler::IsInternalEvent(event_type)) continue;

      tensorflow::profiler::TraceEventLite lite_event;
      lite_event.timestamp_ps = event_visitor.TimestampPs();
      lite_event.duration_ps = event_visitor.DurationPs();
      lite_event.line = &line;
      lite_event.flow_id = UINT64_MAX;
      lite_event.event_idx = event_idx;

      ExpandTraceSpan(tsl::profiler::Timespan(lite_event.timestamp_ps,
                                              lite_event.duration_ps),
                      &container->trace);

      bool has_flow = false;
      bool is_async = false;
      std::optional<tsl::profiler::XFlow> parsed_flow;

      auto extract_special_args = [&](const tsl::profiler::XStatVisitor& stat) {
        if (stat.Type() == tsl::profiler::StatType::kFlow) {
          has_flow = true;
          parsed_flow =
              tsl::profiler::XFlow::FromStatValue(stat.IntOrUintValue());
        } else if (stat.Type() == tsl::profiler::StatType::kIsAsync) {
          is_async = true;
        }
      };

      event_visitor.Metadata().ForEachStat(extract_special_args);
      event_visitor.ForEachStat(extract_special_args);

      uint64_t name_ref = 0;
      if (is_counter_line || (is_async && has_flow)) {
        absl::string_view name =
            is_counter_line
                ? event_visitor.Name()
                : (event_visitor.HasDisplayName() ? event_visitor.DisplayName()
                                                  : event_visitor.Name());
        uint64_t fp = std::hash<absl::string_view>()(name);
        container->name_table[fp] = std::string(name);
        name_ref = fp;
      }

      TraceTrackMetadata meta;
      meta.device_id = line_device_id;
      meta.plane_visitor = plane_visitor;
      meta.name_ref = name_ref;

      if (is_counter_line) {
        const auto* meta_ptr = get_or_create_metadata_ptr(meta);
        lite_event.metadata = meta_ptr;
        container->counter_events[meta].push_back(lite_event);

      } else if (is_async && has_flow) {
        lite_event.is_async = true;
        lite_event.flow_id = parsed_flow->Id();
        lite_event.flow_entry_type =
            FlowEntryTypeFromDirection(parsed_flow->Direction());
        meta.resource_id = 0;

        const auto* meta_ptr = get_or_create_metadata_ptr(meta);
        lite_event.metadata = meta_ptr;
        container->counter_events[meta].push_back(lite_event);

      } else {
        meta.resource_id = static_cast<uint32_t>(line_visitor.DisplayId());
        if (has_flow) {
          lite_event.flow_id = parsed_flow->Id();
          lite_event.flow_entry_type =
              FlowEntryTypeFromDirection(parsed_flow->Direction());
        }

        const auto* meta_ptr = get_or_create_metadata_ptr(meta);
        lite_event.metadata = meta_ptr;
        container->complete_events[meta].push_back(lite_event);
      }
    }
  }
}

}  // namespace

void ConvertXSpaceToTraceEventsContainer(absl::string_view hostname,
                                         const XSpace& space,
                                         TraceEventsContainer* container) {
  std::vector<const XPlane*> host_planes =
      FindPlanesWithPrefix(space, tsl::profiler::kHostThreadsPlaneName);
  if (!host_planes.empty()) {
    int32_t host_device_id = tsl::profiler::kHostThreadsDeviceId;
    absl::c_sort(host_planes, [](const XPlane* a, const XPlane* b) {
      return a->name() < b->name();
    });
    for (const XPlane* host_plane : host_planes) {
      ConvertXPlaneToTraceEventsContainer(host_device_id++,
                                          hostname, *host_plane, container);
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

void ConvertXSpaceToLiteTraceEventsContainer(
    absl::string_view hostname, const XSpace& space,
    TraceEventLiteContainer* container) {
  std::vector<const XPlane*> host_planes =
      FindPlanesWithPrefix(space, tsl::profiler::kHostThreadsPlaneName);
  if (!host_planes.empty()) {
    int32_t host_device_id = tsl::profiler::kHostThreadsDeviceId;
    absl::c_sort(host_planes, [](const XPlane* a, const XPlane* b) {
      return a->name() < b->name();
    });
    for (const XPlane* host_plane : host_planes) {
      ConvertXPlaneToLiteTraceEventsContainer(host_device_id++, hostname,
                                              *host_plane, container);
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
    ConvertXPlaneToLiteTraceEventsContainer(device_pid, hostname, *device_plane,
                                            container);
  }
  for (const XPlane* custom_plane :
       FindPlanesWithPrefix(space, tsl::profiler::kCustomPlanePrefix)) {
    ConvertXPlaneToLiteTraceEventsContainer(
        tsl::profiler::kFirstCustomPlaneDeviceId + custom_plane->id(), hostname,
        *custom_plane, container);
  }
}

void ConvertLiteTraceEventToFullTraceEvent(
    const tsl::profiler::XEventVisitor& event_visitor,
    const TraceEventLite& lite_event, const TraceEventLiteContainer& container,
    absl::flat_hash_map<uint64_t, std::string>* local_name_table,
    TraceEvent* full_event, google::protobuf::Arena* arena) {
  const auto& metadata = *lite_event.metadata;

  RawData* raw_data = google::protobuf::Arena::Create<RawData>(arena);
  TraceEventArguments* raw_args = raw_data->mutable_args();
  absl::string_view event_name;

  if (event_visitor.HasDisplayName()) {
    event_name = event_visitor.DisplayName();
    TraceEventArgumentsBuilder args(raw_args);
    constexpr size_t kMaxLongName = 10000;
    if (event_visitor.Name().size() > kMaxLongName) {
      args.Append("long_name",
                  absl::StrCat(event_visitor.Name().substr(0, kMaxLongName),
                               "...<truncated>"));
    } else {
      args.Append("long_name", event_visitor.Name());
    }
  } else {
    event_name = event_visitor.Name();
  }

  // Stats are parsed for EVERY event contiguously, matching legacy!
  SpecialArguments special_args =
      ConvertXStatsToTraceEventArguments(event_visitor, raw_data, raw_args);
  if (!special_args.step_name.empty()) {
    event_name = special_args.step_name;
  }

  tsl::profiler::Timespan span(lite_event.timestamp_ps, lite_event.duration_ps);

  std::optional<int64_t> serial =
      lite_event.serial > 0 ? std::optional<int64_t>(lite_event.serial)
                            : std::nullopt;

  // 1:1 Flat Parity Routing Block
  if (metadata.resource_id == UINT32_MAX) {
    auto it = container.name_table.find(metadata.name_ref);
    if (it != container.name_table.end()) {
      event_name = it->second;
    }
    CreateCounterEvent(event_name, metadata.device_id, lite_event.timestamp_ps,
                       *raw_data, serial, full_event);
  } else if (special_args.flow) {
    if (special_args.is_async_event) {
      CreateAsyncEvent(
          event_name, metadata.device_id, span, lite_event.flow_id,
          static_cast<TraceEvent::FlowEntryType>(lite_event.flow_entry_type),
          special_args.flow->Category(), raw_data, special_args.group_id,
          serial, local_name_table, full_event);
    } else {
      CreateFlowEvent(
          event_name, metadata.resource_id, metadata.device_id, span,
          lite_event.flow_id,
          static_cast<TraceEvent::FlowEntryType>(lite_event.flow_entry_type),
          special_args.flow->Category(), raw_data,
          special_args.group_id, serial, local_name_table, full_event);
    }
  } else {
    CreateCompleteEvent(event_name, metadata.resource_id, metadata.device_id,
                        span, raw_data, special_args.group_id, serial,
                        local_name_table, full_event);
  }
}

}  // namespace profiler
}  // namespace tensorflow
