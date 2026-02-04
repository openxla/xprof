/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#include "xprof/convert/xspace_to_event_time_fraction_analyzer.h"

#include <numeric>
#include <string>
#include <cstdint>
#include <utility>

#include "absl/status/statusor.h"
#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/arena.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/profiler/utils/tf_xplane_visitor.h"
#include "xla/tsl/profiler/utils/timespan.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_visitor.h"
#include "xprof/convert/preprocess_single_host_xplane.h"
#include "xprof/convert/repository.h"
#include "xprof/convert/xplane_to_step_events.h"
#include "xprof/utils/event_span.h"
#include "plugin/xprof/protobuf/event_time_fraction_analyzer.pb.h"

namespace tensorflow {
namespace profiler {

// TODO(zhuruiyang): 1P SS also uses the same logic to process the Xspace to get
// the event time fraction. We will make 1P reuse this library in the future.
absl::StatusOr<EventTimeFractionAnalyzerResult>
ConvertXSpaceToEventTimeFractionAnalyzerResult(
    const XSpace& xspace, absl::string_view target_event_name) {
  EventTimeFractionAnalyzerResult result_proto;
  absl::btree_map<int64_t, absl::flat_hash_map<std::string, double>>
      step_id_to_plane_fractions;

  for (auto& plane : xspace.planes()) {
    StepEvents step_events = ConvertDeviceTraceXPlaneToStepEvents(plane);

    tsl::profiler::XPlaneVisitor plane_visitor =
        tsl::profiler::CreateTfXPlaneVisitor(&plane);
    plane_visitor.ForEachLine([&](const tsl::profiler::XLineVisitor& line) {
      line.ForEachEvent([&](const tsl::profiler::XEventVisitor& event) {
        if (event.Name() != target_event_name) return;

        int64_t step_id = -1;
        auto group_id_stat = event.GetStat(tsl::profiler::StatType::kGroupId);
        if (group_id_stat.has_value()) {
          step_id = group_id_stat->IntValue();
        }
        if (step_id == -1) return;
        const auto it = step_events.find(step_id);
        if (it == step_events.end()) return;

        const auto step_duration_ps = it->second.StepTime().duration_ps();
        if (step_duration_ps == 0) return;

        auto event_duration =
            event.GetStat(tsl::profiler::StatType::kDeviceDurationPs);
        if (!event_duration.has_value()) return;

        auto event_duration_ps = event_duration->UintValue();
        // TODO(zhuruiyang): Make this check more robust (megacore check).
        // Add a special check for barrier-cores events, skip when
        // event_duration_ps is 0 ns or 1.250 ns (dummy value).
        if (target_event_name == "barrier-cores" &&
            (event_duration_ps == 0 || event_duration_ps == 1250)) {
          return;
        }
        float portion_of_step = static_cast<double>(event_duration_ps) /
                                static_cast<double>(step_duration_ps);
        step_id_to_plane_fractions[step_id][plane.name()] += portion_of_step;
      });
    });
  }

  absl::flat_hash_map<std::string, EventTimeFractionPerChip> plane_to_fractions;
  for (const auto& [step_id, plane_fractions_map] :
       step_id_to_plane_fractions) {
    for (const auto& [plane_name, fraction] : plane_fractions_map) {
      plane_to_fractions[plane_name].set_id(plane_name);
      plane_to_fractions[plane_name].add_event_time_fractions(fraction);
    }
  }

  for (auto& [plane_name, fractions] : plane_to_fractions) {
    result_proto.mutable_chip_event_time_fractions()->insert(
        {plane_name, fractions});
  }
  if (!xspace.hostnames().empty()) {
    std::string hostname = xspace.hostnames(0);
    EventTimeFractionPerHost host_fractions;
    host_fractions.set_hostname(hostname);
    for (const auto& [step_id, plane_fractions_map] :
         step_id_to_plane_fractions) {
      if (plane_fractions_map.empty()) continue;
      double sum = std::accumulate(plane_fractions_map.begin(),
                                   plane_fractions_map.end(), 0.0,
                                   [](double acc, const auto& plane_fraction) {
                                     return acc + plane_fraction.second;
                                   });
      host_fractions.add_event_time_fractions(
          sum / static_cast<double>(plane_fractions_map.size()));
    }
    result_proto.mutable_host_event_time_fractions()->insert(
        {hostname, host_fractions});
  }
  return result_proto;
}

}  // namespace profiler
}  // namespace tensorflow
