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

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <limits>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "google/protobuf/arena.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/profiler/utils/tf_xplane_visitor.h"
#include "xla/tsl/profiler/utils/timespan.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_visitor.h"
#include "xprof/convert/repository.h"
#include "xprof/convert/xplane_to_step_events.h"
#include "plugin/xprof/protobuf/event_time_fraction_analyzer.pb.h"
#include "xprof/utils/event_span.h"

namespace tensorflow {
namespace profiler {

absl::StatusOr<EventTimeFractionAnalyzerResults>
ConvertXSpaceToHostEventTimeFractionAnalyzerResults(
    const XSpace& xspace, absl::Span<const std::string> target_event_names) {
  EventTimeFractionAnalyzerResults results_proto;

  for (const std::string& target_event_name : target_event_names) {
    EventTimeFractionAnalyzerResult result_proto;
    absl::flat_hash_map<std::string, std::vector<double>>
        hostname_to_plane_fractions;

    for (const auto& plane : xspace.planes()) {
      if (plane.name() != tsl::profiler::kHostThreadsPlaneName) {
        continue;
      }

      tsl::profiler::XPlaneVisitor plane_visitor =
          tsl::profiler::CreateTfXPlaneVisitor(&plane);

      absl::flat_hash_set<int64_t> target_event_metadata_ids;
      for (const auto& [metadata_id, metadata] : plane.event_metadata()) {
        if (absl::StrContains(metadata.name(), target_event_name)) {
          target_event_metadata_ids.insert(metadata_id);
        }
      }

      if (target_event_metadata_ids.empty()) continue;

      plane_visitor.ForEachLine([&](const tsl::profiler::XLineVisitor& line) {
        uint64_t xline_start_ps = std::numeric_limits<uint64_t>::max();
        uint64_t xline_end_ps = 0;
        uint64_t target_event_duration_ps = 0;
        line.ForEachEvent([&](const tsl::profiler::XEventVisitor& event) {
          xline_start_ps =
              std::min(xline_start_ps, event.GetTimespan().begin_ps());
          xline_end_ps = std::max(xline_end_ps, event.GetTimespan().end_ps());
          if (target_event_metadata_ids.contains(event.Id())) {
            target_event_duration_ps += event.GetTimespan().duration_ps();
          }
        });
        if (xline_start_ps < xline_end_ps && target_event_duration_ps > 0) {
          double fraction = static_cast<double>(target_event_duration_ps) /
                            (xline_end_ps - xline_start_ps);
          std::string hostname =
              xspace.hostnames().empty() ? "unknown" : xspace.hostnames(0);
          hostname_to_plane_fractions[hostname].push_back(fraction);
        }
      });
    }

    for (const auto& [hostname, host_plane_fractions] :
         hostname_to_plane_fractions) {
      EventTimeFractionPerHost host_fractions;
      host_fractions.set_hostname(hostname);
      if (!host_plane_fractions.empty()) {
        double sum = std::accumulate(host_plane_fractions.begin(),
                                     host_plane_fractions.end(), 0.0);
        host_fractions.add_event_time_fractions(
            sum / static_cast<double>(host_plane_fractions.size()));
      }
      if (host_fractions.event_time_fractions_size() > 0) {
        result_proto.mutable_host_event_time_fractions()->insert(
            {hostname, host_fractions});
      }
    }
    results_proto.mutable_results()->insert({target_event_name, result_proto});
  }
  return results_proto;
}

// TODO(zhuruiyang): 1P SS also uses the same logic to process the Xspace to get
// the event time fraction. We will make 1P reuse this library in the future.
absl::StatusOr<EventTimeFractionAnalyzerResults>
ConvertXSpaceToEventTimeFractionAnalyzerResults(
    const XSpace& xspace, absl::Span<const std::string> target_event_names) {
  if (target_event_names.empty()) {
    std::vector<std::string> wildcard = {""};
    return ConvertXSpaceToEventTimeFractionAnalyzerResults(xspace, wildcard);
  }

  EventTimeFractionAnalyzerResults results_proto;

  absl::flat_hash_map<std::string, tensorflow::profiler::StepEvents>
      plane_name_to_step_events;
  for (const auto& plane : xspace.planes()) {
    if (plane.name() != tsl::profiler::kHostThreadsPlaneName) {
      plane_name_to_step_events[plane.name()] =
          ConvertDeviceTraceXPlaneToStepEvents(plane);
    }
  }

  for (const std::string& target_event_name : target_event_names) {
    EventTimeFractionAnalyzerResult result_proto;
    absl::btree_map<int64_t, absl::flat_hash_map<std::string, double>>
        step_id_to_plane_fractions;
    absl::flat_hash_map<int64_t, uint64_t> step_id_to_duration_ps;

    for (const auto& plane : xspace.planes()) {
      if (plane.name() == tsl::profiler::kHostThreadsPlaneName) {
        continue;
      }
      const StepEvents& step_events =
          plane_name_to_step_events.at(plane.name());

      tsl::profiler::XPlaneVisitor plane_visitor =
          tsl::profiler::CreateTfXPlaneVisitor(&plane);

      absl::flat_hash_set<int64_t> target_event_metadata_ids;
      for (const auto& [metadata_id, metadata] : plane.event_metadata()) {
        if (absl::StrContains(metadata.name(), target_event_name)) {
          target_event_metadata_ids.insert(metadata_id);
        }
      }

      if (target_event_metadata_ids.empty()) continue;

      plane_visitor.ForEachLine([&](const tsl::profiler::XLineVisitor& line) {
        line.ForEachEvent([&](const tsl::profiler::XEventVisitor& event) {
          if (!target_event_metadata_ids.contains(event.Id())) {
            return;
          }

          auto group_id_stat = event.GetStat(tsl::profiler::StatType::kGroupId);
          if (!group_id_stat.has_value()) return;

          int64_t step_id = group_id_stat->IntValue();
          const auto it = step_events.find(step_id);
          if (it == step_events.end()) return;

          const auto step_duration_ps = it->second.StepTime().duration_ps();
          if (step_duration_ps == 0) return;

          uint64_t event_duration_ps = 0;
          if (auto event_duration =
                  event.GetStat(tsl::profiler::StatType::kDeviceDurationPs)) {
            event_duration_ps = event_duration->UintValue();
          }
          if (event_duration_ps == 0) return;
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
          step_id_to_duration_ps[step_id] = step_duration_ps;
        });
      });
    }

    // Heuristic to remove incomplete steps from the analysis.
    // If there are at least 3 steps, remove the first and last steps. If there
    // are exactly 2 steps, remove the shorter step if its duration is less than
    // step_duration_ratio of the longer step.
    constexpr double kDefaultStepDurationRatioThreshold = 0.01;
    if (step_id_to_plane_fractions.size() >= 3) {
      step_id_to_plane_fractions.erase(step_id_to_plane_fractions.begin());
      step_id_to_plane_fractions.erase(
          std::prev(step_id_to_plane_fractions.end()));
    } else if (step_id_to_plane_fractions.size() == 2) {
      auto it1 = step_id_to_plane_fractions.begin();
      auto it2 = std::next(it1);
      uint64_t duration1 = step_id_to_duration_ps[it1->first];
      uint64_t duration2 = step_id_to_duration_ps[it2->first];
      if (duration2 < kDefaultStepDurationRatioThreshold * duration1) {
        step_id_to_plane_fractions.erase(it2);
      } else if (duration1 < kDefaultStepDurationRatioThreshold * duration2) {
        step_id_to_plane_fractions.erase(it1);
      }
    }

    absl::flat_hash_map<std::string, EventTimeFractionPerChip>
        plane_to_fractions;
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
        double sum = std::accumulate(
            plane_fractions_map.begin(), plane_fractions_map.end(), 0.0,
            [](double acc, const auto& plane_fraction) {
              return acc + plane_fraction.second;
            });
        host_fractions.add_event_time_fractions(
            sum / static_cast<double>(plane_fractions_map.size()));
      }
      if (host_fractions.event_time_fractions_size() > 0) {
        result_proto.mutable_host_event_time_fractions()->insert(
            {hostname, host_fractions});
      }
    }
    results_proto.mutable_results()->insert({target_event_name, result_proto});
  }
  return results_proto;
}

absl::StatusOr<EventTimeFractionAnalyzerResults>
ConvertMultiXSpacesToEventTimeFractionAnalyzerResults(
    const SessionSnapshot& session_snapshot,
    absl::Span<const std::string> target_event_names) {
  EventTimeFractionAnalyzerResults combined_results;
  for (int i = 0; i < session_snapshot.XSpaceSize(); ++i) {
    google::protobuf::Arena arena;
    TF_ASSIGN_OR_RETURN(XSpace * xspace, session_snapshot.GetXSpace(i, &arena));
    TF_ASSIGN_OR_RETURN(EventTimeFractionAnalyzerResults results,
                        ConvertXSpaceToEventTimeFractionAnalyzerResults(
                            *xspace, target_event_names));
    for (const auto& [target_event_name, result] : results.results()) {
      auto* combined_result =
          &(*combined_results.mutable_results())[target_event_name];
      for (const auto& [chip_id, fractions] :
           result.chip_event_time_fractions()) {
        auto& combined_fractions =
            (*combined_result->mutable_chip_event_time_fractions())[chip_id];
        combined_fractions.set_id(chip_id);
        combined_fractions.mutable_event_time_fractions()->Add(
            fractions.event_time_fractions().begin(),
            fractions.event_time_fractions().end());
      }
      for (const auto& [host_name, fractions] :
           result.host_event_time_fractions()) {
        auto& combined_fractions =
            (*combined_result->mutable_host_event_time_fractions())[host_name];
        combined_fractions.set_hostname(host_name);
        combined_fractions.mutable_event_time_fractions()->Add(
            fractions.event_time_fractions().begin(),
            fractions.event_time_fractions().end());
      }
    }
  }
  return combined_results;
}

}  // namespace profiler
}  // namespace tensorflow
