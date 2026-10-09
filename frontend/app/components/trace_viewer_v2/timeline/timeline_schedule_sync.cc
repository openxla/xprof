#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "frontend/app/components/trace_viewer_v2/event_data.h"
#include "frontend/app/components/trace_viewer_v2/timeline/constants.h"
#include "frontend/app/components/trace_viewer_v2/timeline/timeline.h"
#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_event.h"

namespace traceviewer {
namespace {

// Returns true for non-instruction scope/region flame tracks in Static Kernel
// Viewer ("Regions", "Named Scope", and "Pallas Primitives").
bool IsScopeOrRegionFlameGroup(absl::string_view group_name) {
  return group_name == "Regions" || group_name == "Named Scope" ||
         group_name == "Pallas Primitives";
}

void PopulateMeanCounterUtilization(const FlameChartTimelineData& timeline_data,
                                    Microseconds start, Microseconds end,
                                    EventData& event_data) {
  const Microseconds dur = end - start;
  if (dur <= 0.0) return;
  EventData mean_util;
  for (const auto& [g_idx, counter_data] :
       timeline_data.counter_data_by_group_index) {
    if (g_idx < 0 || g_idx >= static_cast<int>(timeline_data.groups.size()) ||
        counter_data.timestamps.empty() || counter_data.values.empty()) {
      continue;
    }
    const auto& ts = counter_data.timestamps;
    const auto& vals = counter_data.values;
    double weighted_sum = 0.0;
    auto it = std::upper_bound(ts.begin(), ts.end(), start);
    size_t i =
        (it == ts.begin()) ? 0 : std::distance(ts.begin(), std::prev(it));
    for (; i < ts.size() && i < vals.size(); ++i) {
      const Microseconds seg_start = std::max(start, ts[i]);
      if (seg_start >= end) break;
      const Microseconds seg_end =
          (i + 1 < ts.size()) ? std::min(end, ts[i + 1]) : end;
      if (seg_end > seg_start) {
        weighted_sum += vals[i] * (seg_end - seg_start);
      }
    }
    mean_util.try_emplace(timeline_data.groups[g_idx].name, weighted_sum / dur);
  }
  if (!mean_util.empty()) {
    event_data.try_emplace("utilization", std::move(mean_util));
  }
}

}  // namespace

Microseconds Timeline::GetScheduleEndBundle() const {
  Microseconds end_bundle =
      std::max(data_time_range_.end(), fetched_data_time_range_.end());
  if (end_bundle > 0.0) {
    return end_bundle;
  }
  const size_t n = std::min(timeline_data_.entry_start_times.size(),
                            timeline_data_.entry_total_times.size());
  for (size_t i = 0; i < n; ++i) {
    end_bundle = std::max(end_bundle, timeline_data_.entry_start_times[i] +
                                          timeline_data_.entry_total_times[i]);
  }
  return end_bundle;
}

void Timeline::PopulateEventHoveredScheduleDetails(
    int event_index, EventData& event_data) const {
  if (event_index >= 0 &&
      static_cast<size_t>(event_index) < timeline_data_.entry_names.size() &&
      static_cast<size_t>(event_index) < timeline_data_.entry_levels.size()) {
    const int level = timeline_data_.entry_levels[event_index];
    int group_index = -1;
    for (size_t i = 0; i < timeline_data_.groups.size(); ++i) {
      const int next_group_start_level =
          GetNextGroupStartLevel(timeline_data_, i);
      if (level >= timeline_data_.groups[i].start_level &&
          level < next_group_start_level) {
        group_index = static_cast<int>(i);
        break;
      }
    }
    if (group_index != -1) {
      event_data.try_emplace("trackName",
                             timeline_data_.groups[group_index].name);
      PopulateHoverScheduleDetails(group_index, event_index, event_data);
    }
  } else if (hovered_group_index_ >= 0 &&
             hovered_group_index_ <
                 static_cast<int>(timeline_data_.groups.size()) &&
             hovered_bundle_ >= 0) {
    const Group& group = timeline_data_.groups[hovered_group_index_];
    event_data.try_emplace("trackName", group.name);
    event_data.try_emplace("trackType", group.type == Group::Type::kCounter
                                            ? std::string("counter")
                                            : std::string("process"));
    event_data.try_emplace(kEventSelectedName, group.name);
    event_data.try_emplace(kEventSelectedStart,
                           static_cast<double>(hovered_bundle_));
    event_data.try_emplace(kEventSelectedDuration, 1.0);
    event_data.try_emplace("counterValue", hovered_counter_value_);
    const Microseconds total_bundles = GetScheduleEndBundle();
    if (total_bundles > 0.0) {
      event_data.try_emplace("totalBundles", total_bundles);
    }
    PopulateBundleScheduleDetails(static_cast<Microseconds>(hovered_bundle_),
                                  event_data);
  }
}

void Timeline::PopulateBundleScheduleDetails(Microseconds bundle,
                                             EventData& event_data) const {
  EventData bundle_counts;
  std::vector<std::string> regions;
  std::vector<EventData> region_ancestors;
  const int total_levels = timeline_data_.total_levels();
  const size_t num_entries =
      std::min({timeline_data_.entry_names.size(),
                timeline_data_.entry_start_times.size(),
                timeline_data_.entry_total_times.size()});

  for (size_t g_idx = 0; g_idx < timeline_data_.groups.size(); ++g_idx) {
    const Group& g = timeline_data_.groups[g_idx];
    if (g.type != Group::Type::kFlame ||
        (g.nesting_level == kProcessNestingLevel && g.has_children)) {
      continue;
    }
    const int start_lvl = std::max(0, g.start_level);
    const int end_lvl =
        std::min(GetNextGroupStartLevel(timeline_data_, g_idx), total_levels);
    if (g.name == "Regions") {
      for (int lvl = start_lvl; lvl < end_lvl; ++lvl) {
        absl::Span<const int> indices = timeline_data_.level_events(lvl);
        auto it = std::upper_bound(
            indices.begin(), indices.end(), bundle,
            [this, num_entries](Microseconds t, int idx) {
              if (idx < 0 || static_cast<size_t>(idx) >= num_entries) {
                return false;
              }
              return t < timeline_data_.entry_start_times[idx];
            });
        if (it != indices.begin()) {
          const int idx = *std::prev(it);
          if (idx >= 0 && static_cast<size_t>(idx) < num_entries) {
            const Microseconds s = timeline_data_.entry_start_times[idx];
            const Microseconds d = timeline_data_.entry_total_times[idx];
            if (s <= bundle && s + d > bundle) {
              regions.push_back(timeline_data_.entry_names[idx]);
              EventData ancestor;
              ancestor.try_emplace(kEventSelectedIndex, idx);
              ancestor.try_emplace(kEventSelectedName,
                                   timeline_data_.entry_names[idx]);
              ancestor.try_emplace(kEventSelectedStart, s);
              ancestor.try_emplace(kEventSelectedDuration, d);
              region_ancestors.push_back(std::move(ancestor));
            }
          }
        }
      }
      continue;
    }
    if (IsScopeOrRegionFlameGroup(g.name)) {
      continue;
    }

    int count = 0;
    for (int lvl = start_lvl; lvl < end_lvl; ++lvl) {
      absl::Span<const int> indices = timeline_data_.level_events(lvl);
      auto it = std::upper_bound(
          indices.begin(), indices.end(), bundle,
          [this, num_entries](Microseconds t, int idx) {
            if (idx < 0 || static_cast<size_t>(idx) >= num_entries) {
              return false;
            }
            return t < timeline_data_.entry_start_times[idx];
          });
      if (it != indices.begin()) {
        const int idx = *std::prev(it);
        if (idx >= 0 && static_cast<size_t>(idx) < num_entries) {
          const Microseconds s = timeline_data_.entry_start_times[idx];
          const Microseconds e = s + timeline_data_.entry_total_times[idx];
          if (s <= bundle && e > bundle) {
            ++count;
          }
        }
      }
    }
    if (count > 0) {
      bundle_counts.try_emplace(g.name, count);
    }
  }

  event_data.try_emplace("bundleCounts", std::move(bundle_counts));
  event_data.try_emplace("regions", std::move(regions));
  if (!region_ancestors.empty()) {
    event_data.try_emplace("regionAncestors", std::move(region_ancestors));
  }

  EventData utilization;
  for (const auto& [g_idx, counter_data] :
       timeline_data_.counter_data_by_group_index) {
    if (g_idx < 0 || g_idx >= static_cast<int>(timeline_data_.groups.size()) ||
        counter_data.timestamps.empty() || counter_data.values.empty()) {
      continue;
    }
    auto it = std::upper_bound(counter_data.timestamps.begin(),
                               counter_data.timestamps.end(), bundle);
    double val = 0.0;
    if (it != counter_data.timestamps.begin()) {
      const size_t idx =
          std::distance(counter_data.timestamps.begin(), std::prev(it));
      if (idx < counter_data.values.size()) {
        val = counter_data.values[idx];
      }
    }
    utilization.try_emplace(timeline_data_.groups[g_idx].name, val);
  }
  if (!utilization.empty()) {
    event_data.try_emplace("utilization", std::move(utilization));
  }
}

void Timeline::PopulateHoverScheduleDetails(int group_index, int event_index,
                                            EventData& event_data) const {
  const Microseconds total_bundles = GetScheduleEndBundle();
  if (total_bundles > 0.0) {
    event_data.try_emplace("totalBundles", total_bundles);
  }
  if (event_index < 0 ||
      static_cast<size_t>(event_index) >=
          timeline_data_.entry_start_times.size() ||
      static_cast<size_t>(event_index) >=
          timeline_data_.entry_total_times.size()) {
    return;
  }

  const size_t event_idx = static_cast<size_t>(event_index);
  if (timeline_data_.HasEntryArgs(event_idx)) {
    const auto raw_args = timeline_data_.GetEntryArgs(event_idx);
    EventData args_data;
    for (const auto& [key, val] : raw_args) {
      if (!val.empty() && !(key == kHloModule && val == kHloModuleDefault)) {
        args_data.try_emplace(key, val);
      }
    }
    if (!args_data.empty()) {
      event_data.try_emplace("args", std::move(args_data));
    }
  }

  const Microseconds start = timeline_data_.entry_start_times[event_index];
  const Microseconds dur = timeline_data_.entry_total_times[event_index];
  const Microseconds end = start + dur;
  const Group& hovered_group = timeline_data_.groups[group_index];

  if (!IsScopeOrRegionFlameGroup(hovered_group.name)) {
    PopulateBundleScheduleDetails(std::floor(start), event_data);
    return;
  }

  const int total_levels = timeline_data_.total_levels();
  const size_t num_entries =
      std::min({timeline_data_.entry_names.size(),
                timeline_data_.entry_start_times.size(),
                timeline_data_.entry_total_times.size()});
  const int event_level =
      (static_cast<size_t>(event_index) < timeline_data_.entry_levels.size())
          ? timeline_data_.entry_levels[event_index]
          : hovered_group.start_level;
  const int depth = std::max(0, event_level - hovered_group.start_level);
  event_data.try_emplace("regionDepth", depth);

  std::vector<std::string> parent_regions;
  std::vector<EventData> region_ancestors;
  const int region_start_lvl = std::max(0, hovered_group.start_level);
  const int region_end_lvl = std::min(event_level, total_levels);
  for (int lvl = region_start_lvl; lvl < region_end_lvl; ++lvl) {
    absl::Span<const int> indices = timeline_data_.level_events(lvl);
    auto it = std::upper_bound(
        indices.begin(), indices.end(), start,
        [this, num_entries](Microseconds t, int idx) {
          if (idx < 0 || static_cast<size_t>(idx) >= num_entries) {
            return false;
          }
          return t < timeline_data_.entry_start_times[idx];
        });
    if (it != indices.begin()) {
      const int idx = *std::prev(it);
      if (idx >= 0 && static_cast<size_t>(idx) < num_entries) {
        const Microseconds s = timeline_data_.entry_start_times[idx];
        const Microseconds d = timeline_data_.entry_total_times[idx];
        if (s <= start && s + d > start) {
          parent_regions.push_back(timeline_data_.entry_names[idx]);
          EventData ancestor;
          ancestor.try_emplace(kEventSelectedIndex, idx);
          ancestor.try_emplace(kEventSelectedName,
                               timeline_data_.entry_names[idx]);
          ancestor.try_emplace(kEventSelectedStart, s);
          ancestor.try_emplace(kEventSelectedDuration, d);
          region_ancestors.push_back(std::move(ancestor));
        }
      }
    }
  }
  event_data.try_emplace("regions", std::move(parent_regions));
  if (!region_ancestors.empty()) {
    event_data.try_emplace("regionAncestors", std::move(region_ancestors));
  }

  EventData mix_counts;
  for (size_t g_idx = 0; g_idx < timeline_data_.groups.size(); ++g_idx) {
    const Group& g = timeline_data_.groups[g_idx];
    if (g.type != Group::Type::kFlame ||
        (g.nesting_level == kProcessNestingLevel && g.has_children) ||
        IsScopeOrRegionFlameGroup(g.name)) {
      continue;
    }
    const int start_lvl = std::max(0, g.start_level);
    const int end_lvl =
        std::min(GetNextGroupStartLevel(timeline_data_, g_idx), total_levels);
    int count = 0;
    for (int lvl = start_lvl; lvl < end_lvl; ++lvl) {
      absl::Span<const int> indices = timeline_data_.level_events(lvl);
      auto first_it = std::lower_bound(
          indices.begin(), indices.end(), start,
          [this, num_entries](int idx, Microseconds t) {
            if (idx < 0 || static_cast<size_t>(idx) >= num_entries) {
              return false;
            }
            return timeline_data_.entry_start_times[idx] < t;
          });
      auto last_it = std::lower_bound(
          first_it, indices.end(), end,
          [this, num_entries](int idx, Microseconds t) {
            if (idx < 0 || static_cast<size_t>(idx) >= num_entries) {
              return false;
            }
            return timeline_data_.entry_start_times[idx] < t;
          });
      count += static_cast<int>(std::distance(first_it, last_it));
    }
    if (count > 0) {
      mix_counts.try_emplace(g.name, count);
    }
  }
  event_data.try_emplace("bundleCounts", std::move(mix_counts));

  PopulateMeanCounterUtilization(timeline_data_, start, end, event_data);
}

}  // namespace traceviewer
