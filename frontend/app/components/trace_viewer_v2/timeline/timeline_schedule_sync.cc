#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "imgui.h"
#include "frontend/app/components/trace_viewer_v2/color/colors.h"
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

// Formats an ImU32 color (0xAABBGGRR) as a lowercase CSS #rrggbb hex string.
std::string FormatCssHexColor(ImU32 color) {
  return absl::StrFormat("#%02x%02x%02x", color & 0xFF, (color >> 8) & 0xFF,
                         (color >> 16) & 0xFF);
}

struct InstructionSummary {
  std::string track_name;
  std::string name;
  int count = 0;
  Microseconds total_bundles = 0.0;
  int first_event_index = -1;
};

std::vector<EventData> BuildTopInstructionEventData(
    absl::flat_hash_map<std::pair<std::string, std::string>,
                        InstructionSummary>& instruction_map) {
  std::vector<InstructionSummary> sorted_instructions;
  sorted_instructions.reserve(instruction_map.size());
  for (auto& [key, summary] : instruction_map) {
    sorted_instructions.push_back(std::move(summary));
  }
  std::sort(sorted_instructions.begin(), sorted_instructions.end(),
            [](const InstructionSummary& a, const InstructionSummary& b) {
              if (a.count != b.count) return a.count > b.count;
              if (a.total_bundles != b.total_bundles) {
                return a.total_bundles > b.total_bundles;
              }
              return a.name < b.name;
            });

  constexpr size_t kMaxTopInstructions = 24;
  std::vector<EventData> top_instructions;
  const size_t limit =
      std::min(sorted_instructions.size(), kMaxTopInstructions);
  top_instructions.reserve(limit);
  for (size_t i = 0; i < limit; ++i) {
    const auto& s = sorted_instructions[i];
    EventData item;
    item.try_emplace(kEventSelectedName, s.name);
    item.try_emplace("trackName", s.track_name);
    item.try_emplace("count", s.count);
    item.try_emplace("totalBundles", s.total_bundles);
    item.try_emplace("firstEventIndex", s.first_event_index);
    top_instructions.push_back(std::move(item));
  }
  return top_instructions;
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

void Timeline::PopulateEventSelectedScheduleDetails(
    int event_index, EventData& event_data) const {
  if (event_index < 0 ||
      static_cast<size_t>(event_index) >= timeline_data_.entry_names.size() ||
      static_cast<size_t>(event_index) >= timeline_data_.entry_levels.size()) {
    return;
  }
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
    PopulateSelectedScheduleDetails(group_index, event_index, event_data);
  }
}

void Timeline::PopulateCounterSelectedScheduleDetails(
    int group_index, size_t counter_index, EventData& event_data) const {
  if (group_index < 0 ||
      static_cast<size_t>(group_index) >= timeline_data_.groups.size()) {
    return;
  }
  const auto it = timeline_data_.counter_data_by_group_index.find(group_index);
  if (it == timeline_data_.counter_data_by_group_index.end()) return;
  const CounterData& data = it->second;
  if (counter_index >= data.timestamps.size() ||
      counter_index >= data.values.size()) {
    return;
  }
  const std::string& name = timeline_data_.groups[group_index].name;
  const Microseconds bundle = std::floor(data.timestamps[counter_index]);
  event_data.try_emplace("trackName", name);
  event_data.try_emplace("trackType", std::string("counter"));
  event_data.try_emplace(kEventSelectedStart, bundle);
  event_data.try_emplace(kEventSelectedDuration, 1.0);
  event_data.try_emplace("counterValue", data.values[counter_index]);
  const Microseconds total_bundles = GetScheduleEndBundle();
  if (total_bundles > 0.0) {
    event_data.try_emplace("totalBundles", total_bundles);
  }
  PopulateBundleScheduleDetails(bundle, event_data);
  PopulateBundleEvents(bundle, -1, event_data);
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

void Timeline::PopulateBundleEvents(Microseconds bundle,
                                    int selected_event_index,
                                    EventData& event_data) const {
  std::vector<EventData> bundle_events;
  const int total_levels = timeline_data_.total_levels();
  const size_t num_entries =
      std::min({timeline_data_.entry_names.size(),
                timeline_data_.entry_start_times.size(),
                timeline_data_.entry_total_times.size()});

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
            EventData item;
            item.try_emplace(kEventSelectedIndex, idx);
            item.try_emplace(kEventSelectedName,
                             timeline_data_.entry_names[idx]);
            item.try_emplace("trackName", g.name);
            item.try_emplace(kEventSelectedStart, s);
            item.try_emplace(kEventSelectedDuration, d);
            item.try_emplace("selected", idx == selected_event_index);
            if (timeline_data_.HasEntryArgs(static_cast<size_t>(idx))) {
              const auto raw_args =
                  timeline_data_.GetEntryArgs(static_cast<size_t>(idx));
              for (const char* key : {"produces", "dest", "result"}) {
                if (auto arg_it = raw_args.find(key);
                    arg_it != raw_args.end() && !arg_it->second.empty()) {
                  item.try_emplace("produces", arg_it->second);
                  break;
                }
              }
              for (const char* key :
                   {"ordinal", "program_order", "instruction_ordinal"}) {
                if (auto arg_it = raw_args.find(key);
                    arg_it != raw_args.end() && !arg_it->second.empty()) {
                  item.try_emplace("ordinal", arg_it->second);
                  break;
                }
              }
            }
            bundle_events.push_back(std::move(item));
          }
        }
      }
    }
  }
  event_data.try_emplace("bundleEvents", std::move(bundle_events));
}

void Timeline::PopulateSelectedScheduleDetails(int group_index, int event_index,
                                               EventData& event_data) const {
  if (event_index < 0 ||
      static_cast<size_t>(event_index) >=
          timeline_data_.entry_start_times.size() ||
      static_cast<size_t>(event_index) >=
          timeline_data_.entry_total_times.size() ||
      group_index < 0 ||
      static_cast<size_t>(group_index) >= timeline_data_.groups.size()) {
    return;
  }

  const int total_levels = timeline_data_.total_levels();
  const size_t num_entries =
      std::min({timeline_data_.entry_names.size(),
                timeline_data_.entry_start_times.size(),
                timeline_data_.entry_total_times.size()});
  const Microseconds start = timeline_data_.entry_start_times[event_index];
  const Microseconds dur = timeline_data_.entry_total_times[event_index];
  const Microseconds end = start + dur;
  const Group& selected_group = timeline_data_.groups[group_index];

  const int event_level =
      (static_cast<size_t>(event_index) < timeline_data_.entry_levels.size())
          ? timeline_data_.entry_levels[event_index]
          : selected_group.start_level;

  int prev_event_index = -1;
  int next_event_index = -1;
  if (event_level >= 0 && event_level < total_levels) {
    absl::Span<const int> level_indices =
        timeline_data_.level_events(event_level);
    auto it = std::lower_bound(
        level_indices.begin(), level_indices.end(), start,
        [this, num_entries](int idx, Microseconds t) {
          if (idx < 0 || static_cast<size_t>(idx) >= num_entries) {
            return false;
          }
          return timeline_data_.entry_start_times[idx] < t;
        });
    while (it != level_indices.end() && *it != event_index && *it >= 0 &&
           static_cast<size_t>(*it) < num_entries &&
           timeline_data_.entry_start_times[*it] <= start) {
      ++it;
    }
    if (it != level_indices.end() && *it == event_index) {
      if (it != level_indices.begin()) {
        prev_event_index = *std::prev(it);
      }
      if (std::next(it) != level_indices.end()) {
        next_event_index = *std::next(it);
      }
    }
  }
  event_data.try_emplace("prevEventIndex", prev_event_index);
  event_data.try_emplace("nextEventIndex", next_event_index);

  if (!IsScopeOrRegionFlameGroup(selected_group.name)) {
    PopulateBundleEvents(std::floor(start), event_index, event_data);
    return;
  }

  // Populate direct child regions on (event_level + 1) inside [start, end).
  const int group_end_lvl = std::min(
      GetNextGroupStartLevel(timeline_data_, group_index), total_levels);
  const int child_lvl = event_level + 1;
  if (child_lvl >= selected_group.start_level && child_lvl < group_end_lvl) {
    std::vector<EventData> child_regions;
    constexpr int kMaxChildRegions = 24;
    absl::Span<const int> child_indices =
        timeline_data_.level_events(child_lvl);
    auto first_it = std::lower_bound(
        child_indices.begin(), child_indices.end(), start,
        [this, num_entries](int idx, Microseconds t) {
          if (idx < 0 || static_cast<size_t>(idx) >= num_entries) {
            return false;
          }
          return timeline_data_.entry_start_times[idx] < t;
        });
    for (auto it = first_it;
         it != child_indices.end() &&
         static_cast<int>(child_regions.size()) < kMaxChildRegions;
         ++it) {
      const int idx = *it;
      if (idx < 0 || static_cast<size_t>(idx) >= num_entries) continue;
      const Microseconds child_start = timeline_data_.entry_start_times[idx];
      if (child_start >= end) break;
      const Microseconds child_dur = timeline_data_.entry_total_times[idx];
      EventData child_item;
      child_item.try_emplace(kEventSelectedIndex, idx);
      child_item.try_emplace(kEventSelectedName,
                             timeline_data_.entry_names[idx]);
      child_item.try_emplace(kEventSelectedStart, child_start);
      child_item.try_emplace(kEventSelectedDuration, child_dur);
      child_regions.push_back(std::move(child_item));
    }
    if (!child_regions.empty()) {
      event_data.try_emplace("childRegions", std::move(child_regions));
    }
  }

  // Aggregate top instructions scheduled inside [start, end).
  absl::flat_hash_map<std::pair<std::string, std::string>, InstructionSummary>
      instruction_map;
  int total_instructions = 0;

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
      for (auto it = first_it; it != last_it; ++it) {
        const int idx = *it;
        if (idx < 0 || static_cast<size_t>(idx) >= num_entries) continue;
        const std::string& instr_name = timeline_data_.entry_names[idx];
        auto& summary = instruction_map[{g.name, instr_name}];
        if (summary.count == 0) {
          summary.track_name = g.name;
          summary.name = instr_name;
          summary.first_event_index = idx;
        }
        summary.count++;
        summary.total_bundles += timeline_data_.entry_total_times[idx];
        total_instructions++;
      }
    }
  }

  event_data.try_emplace("topInstructions",
                         BuildTopInstructionEventData(instruction_map));
  event_data.try_emplace("totalInstructions", total_instructions);
}

void Timeline::PopulateMultiSelectionScheduleDetails(
    Microseconds selection_start_us, Microseconds selection_extent_us,
    EventData& event_data) const {
  const size_t num_entries = std::min({timeline_data_.entry_names.size(),
                                       timeline_data_.entry_start_times.size(),
                                       timeline_data_.entry_total_times.size(),
                                       timeline_data_.entry_levels.size()});

  Microseconds min_start = std::numeric_limits<Microseconds>::max();
  Microseconds max_end = 0.0;
  for (const int idx : selected_event_indices_) {
    if (idx < 0 || static_cast<size_t>(idx) >= num_entries) continue;
    const Microseconds s = timeline_data_.entry_start_times[idx];
    const Microseconds e = s + timeline_data_.entry_total_times[idx];
    min_start = std::min(min_start, s);
    max_end = std::max(max_end, e);
  }
  for (const auto& [g_idx, pt_idx] : selected_counter_points_) {
    const auto it = timeline_data_.counter_data_by_group_index.find(g_idx);
    if (it == timeline_data_.counter_data_by_group_index.end()) continue;
    if (pt_idx < it->second.timestamps.size()) {
      const Microseconds ts = std::floor(it->second.timestamps[pt_idx]);
      min_start = std::min(min_start, ts);
      max_end = std::max(max_end, ts + 1.0);
    }
  }

  const Microseconds range_start =
      selection_extent_us > 0.0
          ? std::max(0.0, std::floor(selection_start_us))
          : (min_start < std::numeric_limits<Microseconds>::max()
                 ? std::max(0.0, std::floor(min_start))
                 : 0.0);
  const Microseconds range_end =
      selection_extent_us > 0.0
          ? std::ceil(selection_start_us + selection_extent_us)
          : std::max(range_start, std::ceil(max_end));
  const Microseconds range_dur = std::max(0.0, range_end - range_start);

  event_data.try_emplace("selectionStart", range_start);
  event_data.try_emplace("selectionExtent", range_dur);
  event_data.try_emplace("totalSelectedEvents",
                         static_cast<int>(selected_event_indices_.size()));
  const Microseconds total_bundles = GetScheduleEndBundle();
  if (total_bundles > 0.0) {
    event_data.try_emplace("totalBundles", total_bundles);
  }

  // Precompute level -> group name once (O(G + L)) instead of scanning groups
  // per selected event.
  const int total_levels = timeline_data_.total_levels();
  std::vector<int> level_to_group_index(std::max(0, total_levels), -1);
  for (size_t i = 0; i < timeline_data_.groups.size(); ++i) {
    const int start_lvl = std::max(0, timeline_data_.groups[i].start_level);
    const int end_lvl =
        std::min(GetNextGroupStartLevel(timeline_data_, i), total_levels);
    for (int lvl = start_lvl; lvl < end_lvl; ++lvl) {
      if (level_to_group_index[lvl] == -1) {
        level_to_group_index[lvl] = static_cast<int>(i);
      }
    }
  }
  auto find_group_name = [&](int level) -> std::string {
    if (level >= 0 &&
        static_cast<size_t>(level) < level_to_group_index.size()) {
      const int g_idx = level_to_group_index[level];
      if (g_idx >= 0) {
        return timeline_data_.groups[g_idx].name;
      }
    }
    return "";
  };

  absl::flat_hash_map<std::string, int> unit_counts;
  absl::flat_hash_map<std::pair<std::string, std::string>, InstructionSummary>
      instruction_map;
  std::vector<int> sorted_indices;
  sorted_indices.reserve(selected_event_indices_.size());
  int total_instructions = 0;

  for (const int idx : selected_event_indices_) {
    if (idx < 0 || static_cast<size_t>(idx) >= num_entries) continue;
    sorted_indices.push_back(idx);
    const std::string track_name =
        find_group_name(timeline_data_.entry_levels[idx]);
    if (track_name.empty() || IsScopeOrRegionFlameGroup(track_name)) continue;

    unit_counts[track_name]++;
    total_instructions++;
    const std::string& instr_name = timeline_data_.entry_names[idx];
    auto& summary = instruction_map[{track_name, instr_name}];
    if (summary.count == 0) {
      summary.track_name = track_name;
      summary.name = instr_name;
      summary.first_event_index = idx;
    }
    summary.count++;
    summary.total_bundles += timeline_data_.entry_total_times[idx];
  }

  EventData bundle_counts;
  for (const auto& [unit, count] : unit_counts) {
    bundle_counts.try_emplace(unit, count);
  }
  event_data.try_emplace("bundleCounts", std::move(bundle_counts));
  event_data.try_emplace("totalInstructions", total_instructions);
  event_data.try_emplace("topInstructions",
                         BuildTopInstructionEventData(instruction_map));

  std::sort(sorted_indices.begin(), sorted_indices.end(), [this](int a, int b) {
    const Microseconds sa = timeline_data_.entry_start_times[a];
    const Microseconds sb = timeline_data_.entry_start_times[b];
    if (sa != sb) return sa < sb;
    return a < b;
  });
  constexpr size_t kMaxSelectedEvents = 100;
  std::vector<EventData> selected_events;
  const size_t ev_limit = std::min(sorted_indices.size(), kMaxSelectedEvents);
  selected_events.reserve(ev_limit);
  for (size_t i = 0; i < ev_limit; ++i) {
    const int idx = sorted_indices[i];
    EventData item;
    item.try_emplace(kEventSelectedIndex, idx);
    item.try_emplace(kEventSelectedName, timeline_data_.entry_names[idx]);
    item.try_emplace("trackName",
                     find_group_name(timeline_data_.entry_levels[idx]));
    item.try_emplace(kEventSelectedStart,
                     timeline_data_.entry_start_times[idx]);
    item.try_emplace(kEventSelectedDuration,
                     timeline_data_.entry_total_times[idx]);
    selected_events.push_back(std::move(item));
  }
  event_data.try_emplace("selectedEvents", std::move(selected_events));

  PopulateMeanCounterUtilization(timeline_data_, range_start, range_end,
                                 event_data);
}

void Timeline::SyncMinimapState() {
  if ((event_tooltip_enabled_ && !minimap_enabled_) || !event_callback_) {
    return;
  }
  if (data_time_range_.duration() <= 0.0 &&
      fetched_data_time_range_.duration() <= 0.0) {
    return;
  }
  const Microseconds data_start = data_time_range_.duration() > 0.0
                                      ? data_time_range_.start()
                                      : fetched_data_time_range_.start();
  const Microseconds data_end = GetScheduleEndBundle();
  if (data_end <= data_start) {
    return;
  }
  const Microseconds visible_start = visible_range().start();
  const Microseconds visible_end = visible_range().end();

  const bool data_bounds_changed =
      std::abs(data_start - last_minimap_data_start_) > 1e-6 ||
      std::abs(data_end - last_minimap_data_end_) > 1e-6;
  const bool palette_changed =
      palette_.GetVersion() != last_minimap_palette_version_ ||
      palette_.GetTraceVersion() != last_minimap_trace_version_;
  const bool fetched_covers_full_range =
      fetched_data_time_range_.duration() <= 0.0 ||
      fetched_data_time_range_.duration() >= (data_end - data_start) * 0.8;
  const bool include_schedule =
      !has_minimap_schedule_ || data_bounds_changed || palette_changed ||
      (minimap_schedule_dirty_ && fetched_covers_full_range);
  const bool viewport_changed =
      std::abs(visible_start - last_minimap_visible_start_) > 1e-3 ||
      std::abs(visible_end - last_minimap_visible_end_) > 1e-3 ||
      std::abs(label_width_ - last_minimap_label_width_) > 0.5f;

  if (!include_schedule && !viewport_changed) {
    return;
  }

  last_minimap_data_start_ = data_start;
  last_minimap_data_end_ = data_end;
  last_minimap_visible_start_ = visible_start;
  last_minimap_visible_end_ = visible_end;
  last_minimap_label_width_ = label_width_;
  last_minimap_palette_version_ = palette_.GetVersion();
  last_minimap_trace_version_ = palette_.GetTraceVersion();

  EventData payload;
  payload.try_emplace("dataStartUs", data_start);
  payload.try_emplace("dataEndUs", data_end);
  payload.try_emplace("visibleStartUs", visible_start);
  payload.try_emplace("visibleEndUs", visible_end);
  payload.try_emplace("labelWidthPx", static_cast<double>(label_width_));
  payload.try_emplace(
      "timeAxisUnit",
      (time_axis_unit_ == TimeAxisUnit::kUnitless || !event_tooltip_enabled_)
          ? std::string("unitless")
          : std::string("time"));

  if (include_schedule) {
    minimap_schedule_dirty_ = false;
    has_minimap_schedule_ = true;
    PopulateMinimapScheduleSummary(data_start, data_end, payload);
  }

  event_callback_(kMinimapUpdated, payload);
}

void Timeline::PopulateMinimapScheduleSummary(Microseconds data_start,
                                              Microseconds data_end,
                                              EventData& payload) const {
  constexpr int kNumMinimapBins = 160;
  constexpr size_t kMaxMinimapRegions = 24;
  constexpr size_t kMaxMinimapUnits = 32;
  constexpr int kMaxLevelsPerUnitGroup = 8;
  const Microseconds total_span = data_end - data_start;
  if (total_span <= 0.0) {
    return;
  }
  const ImU32 on_surface_color =
      palette_.GetColor(ColorPalette::Key::kOnSurface)
          .value_or(kOnSurfaceColor);
  const ImU32 inverse_on_surface_color =
      palette_.GetColor(ColorPalette::Key::kInverseOnSurface)
          .value_or(kInverseOnSurfaceColor);

  auto get_hex_color = [&](ColorPalette::Key key, ImU32 fallback) {
    return FormatCssHexColor(palette_.GetColor(key).value_or(fallback));
  };
  EventData theme;
  theme.try_emplace("paletteName", palette_.GetCurrentPresetName());
  theme.try_emplace("background",
                    get_hex_color(ColorPalette::Key::kBackground, 0xFFFFFFFF));
  theme.try_emplace("foreground",
                    get_hex_color(ColorPalette::Key::kForeground, 0xFF000000));
  theme.try_emplace("midtone", get_hex_color(ColorPalette::Key::kMidtone,
                                             kInverseOnSurfaceColor));
  theme.try_emplace("flameHeader",
                    get_hex_color(ColorPalette::Key::kFlameHeader, kBlue70));
  theme.try_emplace("collapsedHeader",
                    get_hex_color(ColorPalette::Key::kCollapsedHeader,
                                  kInverseOnSurfaceColor));
  theme.try_emplace("expandedHeader",
                    get_hex_color(ColorPalette::Key::kExpandedHeader,
                                  kSecondaryContainerColor));
  theme.try_emplace("subtitle", get_hex_color(ColorPalette::Key::kSubtitle,
                                              kOnSecondaryFixedVariantColor));
  theme.try_emplace(
      "rulerText", get_hex_color(ColorPalette::Key::kRulerText, kOutlineColor));
  theme.try_emplace("rulerLine", get_hex_color(ColorPalette::Key::kRulerLine,
                                               kOutlineVariantColor));
  theme.try_emplace("selection",
                    get_hex_color(ColorPalette::Key::kSelection, 0xFFFFC9A1));
  theme.try_emplace("onSurface", FormatCssHexColor(on_surface_color));
  theme.try_emplace("inverseOnSurface",
                    FormatCssHexColor(inverse_on_surface_color));
  std::vector<std::string> trace_colors;
  const absl::Span<const ImU32> raw_trace_colors = palette_.GetTraceColors();
  trace_colors.reserve(raw_trace_colors.size());
  for (ImU32 tc : raw_trace_colors) {
    trace_colors.push_back(FormatCssHexColor(tc));
  }
  theme.try_emplace("traceColors", std::move(trace_colors));
  payload.try_emplace("theme", std::move(theme));

  const Microseconds bin_width = total_span / kNumMinimapBins;
  const int total_levels = timeline_data_.total_levels();
  const size_t num_entries =
      std::min({timeline_data_.entry_names.size(),
                timeline_data_.entry_start_times.size(),
                timeline_data_.entry_total_times.size()});

  std::vector<std::string> unit_names;
  std::vector<int> unit_group_indices;
  std::vector<int> all_flame_group_indices;
  int regions_group_index = -1;
  int pallas_group_index = -1;
  int named_scope_group_index = -1;
  int steps_group_index = -1;
  int xla_modules_group_index = -1;
  int framework_scope_group_index = -1;
  for (size_t g_idx = 0; g_idx < timeline_data_.groups.size(); ++g_idx) {
    const Group& g = timeline_data_.groups[g_idx];
    if (g.type != Group::Type::kFlame ||
        (g.nesting_level == kProcessNestingLevel && g.has_children)) {
      continue;
    }
    all_flame_group_indices.push_back(static_cast<int>(g_idx));
    if (g.name == "Regions") {
      regions_group_index = static_cast<int>(g_idx);
      continue;
    }
    if (g.name == "Pallas Primitives") {
      pallas_group_index = static_cast<int>(g_idx);
      continue;
    }
    if (g.name == "Named Scope") {
      named_scope_group_index = static_cast<int>(g_idx);
      continue;
    }
    if (g.name == "Steps" || g.name == "StepMarker") {
      if (steps_group_index < 0) {
        steps_group_index = static_cast<int>(g_idx);
      }
      continue;
    }
    if (g.name == "XLA Modules" || g.name == "HLO Modules") {
      if (xla_modules_group_index < 0) {
        xla_modules_group_index = static_cast<int>(g_idx);
      }
      continue;
    }
    if (g.name == "Framework Name Scope" || g.name == "Name Scope") {
      if (framework_scope_group_index < 0) {
        framework_scope_group_index = static_cast<int>(g_idx);
      }
      continue;
    }
    if (unit_names.size() < kMaxMinimapUnits) {
      unit_names.push_back(g.name);
      unit_group_indices.push_back(static_cast<int>(g_idx));
    }
  }
  if (unit_names.empty()) {
    for (int g_idx : all_flame_group_indices) {
      if (unit_names.size() >= kMaxMinimapUnits) break;
      unit_names.push_back(timeline_data_.groups[g_idx].name);
      unit_group_indices.push_back(g_idx);
    }
  }

  const size_t num_units = unit_names.size();
  std::vector<double> unit_bin_overlap(num_units * kNumMinimapBins, 0.0);
  std::vector<int> unit_bin_best_event(num_units * kNumMinimapBins, -1);
  std::vector<double> unit_bin_best_overlap(num_units * kNumMinimapBins, 0.0);
  std::vector<double> bin_total_overlap(kNumMinimapBins, 0.0);

  for (size_t u = 0; u < num_units; ++u) {
    const int g_idx = unit_group_indices[u];
    const Group& g = timeline_data_.groups[g_idx];
    const int start_lvl = std::max(0, g.start_level);
    const int end_lvl =
        std::min({GetNextGroupStartLevel(timeline_data_, g_idx), total_levels,
                  start_lvl + kMaxLevelsPerUnitGroup});
    for (int lvl = start_lvl; lvl < end_lvl; ++lvl) {
      for (int idx : timeline_data_.level_events(lvl)) {
        if (idx < 0 || static_cast<size_t>(idx) >= num_entries) continue;
        const Microseconds s =
            std::max(data_start, timeline_data_.entry_start_times[idx]);
        const Microseconds e =
            std::min(data_end, timeline_data_.entry_start_times[idx] +
                                   timeline_data_.entry_total_times[idx]);
        if (e <= s) continue;
        const int b0 =
            std::clamp(static_cast<int>((s - data_start) / bin_width), 0,
                       kNumMinimapBins - 1);
        const int b1 =
            std::clamp(static_cast<int>((e - data_start - 1e-9) / bin_width), 0,
                       kNumMinimapBins - 1);
        for (int b = b0; b <= b1; ++b) {
          const Microseconds bin_s = data_start + b * bin_width;
          const Microseconds bin_e = bin_s + bin_width;
          const Microseconds overlap = std::min(e, bin_e) - std::max(s, bin_s);
          if (overlap > 0.0) {
            const size_t cell = u * kNumMinimapBins + b;
            unit_bin_overlap[cell] += overlap;
            if (overlap > unit_bin_best_overlap[cell]) {
              unit_bin_best_overlap[cell] = overlap;
              unit_bin_best_event[cell] = idx;
            }
            bin_total_overlap[b] += overlap;
          }
        }
      }
    }
  }

  struct RegionCandidate {
    int event_index;
    std::string name;
    Microseconds start_us;
    Microseconds duration_us;
    int depth;
  };
  auto extract_scope_group = [&](int group_index, int max_depth,
                                 double min_span_fraction,
                                 bool prefer_innermost,
                                 std::vector<std::string>& bin_names,
                                 std::vector<EventData>& out_markers) {
    if (group_index < 0) return;
    const Group& grp = timeline_data_.groups[group_index];
    const int start_lvl = std::max(0, grp.start_level);
    const int end_lvl = std::min(
        GetNextGroupStartLevel(timeline_data_, group_index), total_levels);
    std::vector<RegionCandidate> candidates;
    for (int lvl = start_lvl; lvl < end_lvl; ++lvl) {
      const int depth = lvl - start_lvl;
      for (int idx : timeline_data_.level_events(lvl)) {
        if (idx < 0 || static_cast<size_t>(idx) >= num_entries) continue;
        const Microseconds s = timeline_data_.entry_start_times[idx];
        const Microseconds d = timeline_data_.entry_total_times[idx];
        const Microseconds e = s + d;
        if (e <= data_start || s >= data_end || d <= 0.0) continue;
        const int b0 =
            std::clamp(static_cast<int>((std::max(data_start, s) - data_start) /
                                        bin_width),
                       0, kNumMinimapBins - 1);
        const int b1 = std::clamp(
            static_cast<int>((std::min(data_end, e) - data_start - 1e-9) /
                             bin_width),
            0, kNumMinimapBins - 1);
        for (int b = b0; b <= b1; ++b) {
          if (prefer_innermost || bin_names[b].empty()) {
            bin_names[b] = timeline_data_.entry_names[idx];
          }
        }
        if (depth <= max_depth && d >= total_span * min_span_fraction) {
          candidates.push_back(
              {idx, timeline_data_.entry_names[idx], s, d, depth});
        }
      }
    }
    if (candidates.size() > kMaxMinimapRegions) {
      std::sort(candidates.begin(), candidates.end(),
                [](const RegionCandidate& a, const RegionCandidate& b) {
                  if (a.depth != b.depth) return a.depth < b.depth;
                  if (a.duration_us != b.duration_us) {
                    return a.duration_us > b.duration_us;
                  }
                  return a.start_us < b.start_us;
                });
      candidates.resize(kMaxMinimapRegions);
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const RegionCandidate& a, const RegionCandidate& b) {
                if (a.start_us != b.start_us) return a.start_us < b.start_us;
                return a.depth < b.depth;
              });
    out_markers.reserve(candidates.size());
    for (const RegionCandidate& c : candidates) {
      EventData reg;
      reg.try_emplace(kEventSelectedIndex, c.event_index);
      reg.try_emplace(kEventSelectedName, c.name);
      reg.try_emplace(kEventSelectedStart, c.start_us);
      reg.try_emplace(kEventSelectedDuration, c.duration_us);
      reg.try_emplace("depth", c.depth);
      const ImU32 event_color = GetEventColor(c.event_index);
      const ImU32 text_color = GetTextColorForContrast(
          event_color, on_surface_color, inverse_on_surface_color);
      reg.try_emplace("color", FormatCssHexColor(event_color));
      reg.try_emplace("textColor", FormatCssHexColor(text_color));
      if (timeline_data_.HasEntryArgs(static_cast<size_t>(c.event_index))) {
        const auto raw_args =
            timeline_data_.GetEntryArgs(static_cast<size_t>(c.event_index));
        if (auto it = raw_args.find("source");
            it != raw_args.end() && !it->second.empty()) {
          reg.try_emplace("source", it->second);
        }
      }
      out_markers.push_back(std::move(reg));
    }
  };

  const int primary_region_group =
      regions_group_index >= 0 ? regions_group_index : steps_group_index;
  const int primary_pallas_group =
      pallas_group_index >= 0 ? pallas_group_index : xla_modules_group_index;
  const int primary_named_scope_group = named_scope_group_index >= 0
                                            ? named_scope_group_index
                                            : framework_scope_group_index;

  std::vector<std::string> bin_region_names(kNumMinimapBins);
  std::vector<EventData> regions;
  extract_scope_group(primary_region_group, /*max_depth=*/2,
                      /*min_span_fraction=*/0.015, /*prefer_innermost=*/true,
                      bin_region_names, regions);

  std::vector<std::string> bin_pallas_names(kNumMinimapBins);
  std::vector<EventData> pallas_primitives;
  extract_scope_group(primary_pallas_group, /*max_depth=*/4,
                      /*min_span_fraction=*/0.01, /*prefer_innermost=*/false,
                      bin_pallas_names, pallas_primitives);

  std::vector<std::string> bin_named_scope_names(kNumMinimapBins);
  std::vector<EventData> named_scopes;
  extract_scope_group(primary_named_scope_group, /*max_depth=*/2,
                      /*min_span_fraction=*/0.015, /*prefer_innermost=*/true,
                      bin_named_scope_names, named_scopes);
  if (pallas_primitives.empty() && !named_scopes.empty()) {
    pallas_primitives = std::move(named_scopes);
  }
  if (regions.empty() && !pallas_primitives.empty() &&
      primary_region_group < 0 && pallas_group_index < 0) {
    regions = std::move(pallas_primitives);
    pallas_primitives.clear();
  }
  if (regions.empty() && pallas_primitives.empty() &&
      !unit_group_indices.empty()) {
    extract_scope_group(unit_group_indices.front(), /*max_depth=*/0,
                        /*min_span_fraction=*/0.02, /*prefer_innermost=*/false,
                        bin_region_names, regions);
  }

  double max_bin_total = bin_width;
  for (double val : bin_total_overlap) {
    max_bin_total = std::max(max_bin_total, val);
  }

  std::vector<EventData> bins;
  bins.reserve(kNumMinimapBins);
  for (int b = 0; b < kNumMinimapBins; ++b) {
    int best_u = -1;
    int second_u = -1;
    double best_overlap = 0.0;
    double second_overlap = 0.0;
    for (size_t u = 0; u < num_units; ++u) {
      const double ov = unit_bin_overlap[u * kNumMinimapBins + b];
      if (ov > best_overlap) {
        second_overlap = best_overlap;
        second_u = best_u;
        best_overlap = ov;
        best_u = static_cast<int>(u);
      } else if (ov > second_overlap) {
        second_overlap = ov;
        second_u = static_cast<int>(u);
      }
    }
    EventData bin_data;
    bin_data.try_emplace("unit",
                         best_u >= 0 ? unit_names[best_u] : std::string());
    if (best_u >= 0) {
      const int best_evt = unit_bin_best_event[best_u * kNumMinimapBins + b];
      if (best_evt >= 0) {
        bin_data.try_emplace("unitColor",
                             FormatCssHexColor(GetEventColor(best_evt)));
      }
    }
    if (second_u >= 0 && best_overlap > 0.0 &&
        second_overlap >= best_overlap * 0.25) {
      bin_data.try_emplace("secondaryUnit", unit_names[second_u]);
      const int sec_evt = unit_bin_best_event[second_u * kNumMinimapBins + b];
      if (sec_evt >= 0) {
        bin_data.try_emplace("secondaryUnitColor",
                             FormatCssHexColor(GetEventColor(sec_evt)));
      }
    }
    const double density =
        (max_bin_total > 0.0 && bin_total_overlap[b] > 0.0)
            ? std::min(1.0, bin_total_overlap[b] / max_bin_total)
            : 0.0;
    bin_data.try_emplace("density", density);
    if (!bin_region_names[b].empty()) {
      bin_data.try_emplace("region", bin_region_names[b]);
    } else if (!bin_named_scope_names[b].empty()) {
      bin_data.try_emplace("region", bin_named_scope_names[b]);
    }
    if (!bin_pallas_names[b].empty()) {
      bin_data.try_emplace("pallasPrimitive", bin_pallas_names[b]);
    }
    if (!bin_named_scope_names[b].empty()) {
      bin_data.try_emplace("namedScope", bin_named_scope_names[b]);
    }
    bins.push_back(std::move(bin_data));
  }

  payload.try_emplace("bins", std::move(bins));
  payload.try_emplace("regions", std::move(regions));
  payload.try_emplace("pallasPrimitives", std::move(pallas_primitives));
}

}  // namespace traceviewer
