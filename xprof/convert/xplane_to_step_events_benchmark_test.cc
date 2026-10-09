#include "xprof/convert/xplane_to_step_events.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <string>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/tsl/profiler/utils/tf_xplane_visitor.h"
#include "xla/tsl/profiler/utils/timespan.h"
#include "xla/tsl/profiler/utils/xplane_builder.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_visitor.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/file_utils.h"
#include "xprof/convert/xspace_to_event_time_fraction_analyzer.h"
#include "plugin/xprof/protobuf/event_time_fraction_analyzer.pb.h"
#include "xprof/utils/event_span.h"

namespace tensorflow {
namespace profiler {
namespace {

using ::tsl::profiler::StatType;
using ::tsl::profiler::XEventBuilder;
using ::tsl::profiler::XLineBuilder;
using ::tsl::profiler::XPlaneBuilder;
using ::tsl::profiler::XStatsBuilder;

// Helper to create a synthetic TPU plane with a step line and an op line.
XPlane CreateSyntheticTpuPlane(int64_t num_ops, int64_t num_steps = 10) {
  XPlane raw_plane;
  XPlaneBuilder plane(&raw_plane);
  int64_t device_id = 0;
  plane.SetId(device_id);
  plane.SetName("/device:TPU:0");

  XLineBuilder step_line = plane.GetOrCreateLine(0);
  step_line.SetName(tsl::profiler::kStepLineName);

  XLineBuilder op_line = plane.GetOrCreateLine(1);
  op_line.SetName(tsl::profiler::kXlaOpLineName);

  const XStatMetadata& program_id_stat =
      *plane.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kProgramId));
  const XStatMetadata& symbol_id_stat =
      *plane.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kSymbolId));
  const XStatMetadata& group_id_stat =
      *plane.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kGroupId));
  const XStatMetadata& duration_stat = *plane.GetOrCreateStatMetadata(
      GetStatTypeStr(StatType::kDeviceDurationPs));

  constexpr uint64_t kStepDurationPs = 100000000ULL;  // 100 us
  constexpr int kNumDistinctOps = 100;
  std::vector<XEventMetadata*> op_metadata(kNumDistinctOps);
  for (int i = 0; i < kNumDistinctOps; ++i) {
    op_metadata[i] =
        plane.GetOrCreateEventMetadata(absl::StrCat("op_symbol_", i));
    op_metadata[i]->set_display_name(absl::StrCat("op_", i));
    XStatsBuilder<XEventMetadata> stats(op_metadata[i], &plane);
    stats.AddStatValue(program_id_stat, 1);
    stats.AddStatValue(symbol_id_stat, i);
  }

  int64_t ops_per_step = std::max<int64_t>(1, num_ops / num_steps);
  uint64_t op_duration_ps = kStepDurationPs / ops_per_step;

  for (int64_t step = 1; step <= num_steps; ++step) {
    uint64_t step_offset_ps = (step - 1) * kStepDurationPs;
    {
      XEventMetadata* step_meta = plane.CreateEventMetadata();
      XEventBuilder step_event = step_line.AddEvent(*step_meta);
      step_event.SetOffsetPs(step_offset_ps);
      step_event.SetDurationPs(kStepDurationPs);
      step_event.AddStatValue(group_id_stat, step);
    }

    for (int64_t op_idx = 0; op_idx < ops_per_step; ++op_idx) {
      int meta_idx = op_idx % kNumDistinctOps;
      XEventBuilder op_event = op_line.AddEvent(*op_metadata[meta_idx]);
      op_event.SetOffsetPs(step_offset_ps + op_idx * op_duration_ps);
      op_event.SetDurationPs(op_duration_ps);
      op_event.AddStatValue(group_id_stat, step);
      op_event.AddStatValue(duration_stat, op_duration_ps);
    }
  }

  return raw_plane;
}

// Helper to create a synthetic TPU XSpace with multiple TPU cores.
XSpace CreateSyntheticTpuSpace(int64_t num_ops_per_core, int num_cores = 4) {
  XSpace space;
  for (int core = 0; core < num_cores; ++core) {
    XPlane* plane = space.add_planes();
    *plane = CreateSyntheticTpuPlane(num_ops_per_core);
    plane->set_id(core);
    plane->set_name(absl::StrCat("/device:TPU:", core));
  }
  return space;
}

// Reimplementation of ConvertXSpaceToEventTimeFractionAnalyzerResults before
// CL 992579135 which parsed full device step events (collect_op_metrics=true).
absl::StatusOr<EventTimeFractionAnalyzerResults>
ConvertXSpaceToEventTimeFractionAnalyzerResultsBefore(
    const XSpace& xspace, absl::Span<const std::string> target_event_names) {
  if (target_event_names.empty()) {
    std::vector<std::string> wildcard = {""};
    return ConvertXSpaceToEventTimeFractionAnalyzerResultsBefore(xspace,
                                                                 wildcard);
  }

  EventTimeFractionAnalyzerResults results_proto;

  absl::flat_hash_map<std::string, tensorflow::profiler::StepEvents>
      plane_name_to_step_events;
  for (const auto& plane : xspace.planes()) {
    plane_name_to_step_events[plane.name()] =
        ConvertDeviceTraceXPlaneToStepEvents(plane);
  }

  for (const std::string& target_event_name : target_event_names) {
    EventTimeFractionAnalyzerResult result_proto;
    absl::btree_map<int64_t, absl::flat_hash_map<std::string, double>>
        step_id_to_plane_fractions;
    absl::flat_hash_map<int64_t, uint64_t> step_id_to_duration_ps;

    for (const auto& plane : xspace.planes()) {
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

          auto event_duration =
              event.GetStat(tsl::profiler::StatType::kDeviceDurationPs);
          if (!event_duration.has_value()) return;

          auto event_duration_ps = event_duration->UintValue();
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
    results_proto.mutable_results()->insert({target_event_name, result_proto});
  }
  return results_proto;
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
