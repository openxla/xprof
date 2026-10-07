// Resolves the HLO operands and consumers of the selected event to timeline
// events, so that HLO dependency arrows can be drawn between them.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "frontend/app/components/trace_viewer_v2/timeline/timeline.h"
#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_event.h"

namespace traceviewer {
namespace {

// The maximum number of events checked per level on each side of the selected
// event when resolving its HLO dependencies. Bounds the cost on large traces.
// An operand or consumer that is further away on its level is not found.
constexpr size_t kMaxDependencySearchEventsPerLevel = 10000;

// For each of `names`, finds the event with that name in levels
// [`first_level`, `end_level`) that starts closest to `time`, among the events
// that start before `time` if `before` is true, or after it otherwise. Returns
// the indices of the events found in the order of `names`, without duplicates.
//
// For example, if the selected `fusion.7` starts at `time`, its operand
// `fusion.3` resolves to the closer `fusion.3` before it, and its consumer
// `add.2` to the closer `add.2` after it:
//
//   | fusion.3 | mul.1 | fusion.3 | fusion.7 | exp.4 | add.2 | add.2 |
//                      ^ operand  ^ time           ^ consumer
//
// Events on a level are sorted by start time, so a binary search finds where
// `time` splits each level, and only events on one side of it are checked.
std::vector<int> FindClosestEventsByName(const FlameChartTimelineData& data,
                                         int first_level, int end_level,
                                         absl::Span<const std::string> names,
                                         Microseconds time, bool before) {
  const std::vector<Microseconds>& starts = data.entry_start_times;
  absl::flat_hash_map<absl::string_view, int> closest_by_name;
  for (const std::string& name : names) closest_by_name.try_emplace(name, -1);
  for (int level = first_level; level < end_level; ++level) {
    // Events on a level are sorted by start time.
    const absl::Span<const int> events = data.level_events(level);
    const size_t split = std::distance(
        events.begin(),
        std::partition_point(events.begin(), events.end(), [&](int i) {
          return before ? starts[i] < time : starts[i] <= time;
        }));
    const absl::Span<const int> candidates =
        before ? events.first(split).last(
                     std::min(split, kMaxDependencySearchEventsPerLevel))
               : events.subspan(split, kMaxDependencySearchEventsPerLevel);
    for (const int event : candidates) {
      const auto it = closest_by_name.find(data.entry_names[event]);
      if (it != closest_by_name.end() &&
          (it->second == -1 || std::abs(starts[event] - time) <
                                   std::abs(starts[it->second] - time))) {
        it->second = event;
      }
    }
  }
  std::vector<int> closest_events;
  for (const std::string& name : names) {
    // Resetting the entry adds each event only once, even for repeated names.
    if (const int event = std::exchange(closest_by_name[name], -1);
        event != -1) {
      closest_events.push_back(event);
    }
  }
  return closest_events;
}

}  // namespace

void Timeline::SetSelectedEventDependencies(
    absl::Span<const std::string> operand_names,
    absl::Span<const std::string> consumer_names) {
  if (!hlo_dependency_arrows_enabled_) return;
  dependencies_ = {.event_index = selected_event_index_};
  if (selected_event_index_ < 0 ||
      selected_event_index_ >=
          static_cast<int>(timeline_data_.entry_levels.size()) ||
      selected_event_index_ >=
          static_cast<int>(timeline_data_.entry_start_times.size())) {
    return;
  }
  // Only the track of the selected event is searched, across all of its
  // levels, e.g. the `XLA Ops` line of a TPU. Operands and consumers on other
  // tracks are not found.
  const Group* group = timeline_data_.FindGroupForLevel(
      timeline_data_.entry_levels[selected_event_index_]);
  if (group == nullptr) return;
  const int end_level = group->start_level + group->level_count;
  const Microseconds start_time =
      timeline_data_.entry_start_times[selected_event_index_];
  // An op runs after its operands and before its consumers.
  dependencies_.producer_indices =
      FindClosestEventsByName(timeline_data_, group->start_level, end_level,
                              operand_names, start_time, /*before=*/true);
  dependencies_.consumer_indices =
      FindClosestEventsByName(timeline_data_, group->start_level, end_level,
                              consumer_names, start_time, /*before=*/false);
}

}  // namespace traceviewer
