#include "frontend/app/components/trace_viewer_v2/timeline/timeline.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "imgui.h"
#include "imgui_internal.h"
#include "frontend/app/components/trace_viewer_v2/color/color_generator.h"
#include "frontend/app/components/trace_viewer_v2/color/colors.h"
#include "frontend/app/components/trace_viewer_v2/event_data.h"
#include "frontend/app/components/trace_viewer_v2/fonts/fonts.h"
#include "frontend/app/components/trace_viewer_v2/helper/clipboard.h"
#include "frontend/app/components/trace_viewer_v2/helper/time_formatter.h"
#include "frontend/app/components/trace_viewer_v2/timeline/constants.h"
#include "frontend/app/components/trace_viewer_v2/timeline/time_range.h"
#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_event.h"

namespace traceviewer {
namespace {

using FallbackKey = std::tuple<ProcessId, ThreadId, absl::string_view>;

absl::flat_hash_map<FallbackKey, std::vector<int>> BuildFallbackMap(
    const FlameChartTimelineData& timeline_data) {
  const int num_loaded = timeline_data.entry_event_ids.size();
  absl::flat_hash_map<FallbackKey, std::vector<int>> fallback_map;
  fallback_map.reserve(num_loaded);
  for (int i = 0; i < num_loaded; ++i) {
    const ProcessId loaded_pid =
        (i < timeline_data.entry_pids.size()) ? timeline_data.entry_pids[i] : 0;
    const ThreadId loaded_tid =
        (i < timeline_data.entry_tids.size()) ? timeline_data.entry_tids[i] : 0;
    const absl::string_view loaded_name =
        (i < timeline_data.entry_names.size())
            ? absl::string_view(timeline_data.entry_names[i])
            : absl::string_view();
    fallback_map[FallbackKey(loaded_pid, loaded_tid, loaded_name)].push_back(i);
  }
  return fallback_map;
}

int FindFallbackMatchedIndex(
    const absl::flat_hash_map<FallbackKey, std::vector<int>>& fallback_map,
    const FlameChartTimelineData& timeline_data, ProcessId pid, ThreadId tid,
    absl::string_view name, Microseconds start_time, Microseconds duration) {
  auto bucket_it = fallback_map.find(FallbackKey(pid, tid, name));
  if (bucket_it == fallback_map.end()) {
    return -1;
  }
  for (int i : bucket_it->second) {
    const Microseconds loaded_start =
        (i < timeline_data.entry_start_times.size())
            ? timeline_data.entry_start_times[i]
            : 0.0;
    const Microseconds loaded_dur = (i < timeline_data.entry_total_times.size())
                                        ? timeline_data.entry_total_times[i]
                                        : 0.0;
    if (std::abs(loaded_start - start_time) < 1.0 &&
        std::abs(loaded_dur - duration) < 1.0) {
      return i;
    }
  }
  return -1;
}

void ApplySnappingToEdge(Microseconds time, Microseconds threshold,
                         const std::vector<Microseconds>& candidates,
                         Microseconds& best_diff, Microseconds& snapped_edge,
                         bool& snapped) {
  auto it =
      std::lower_bound(candidates.begin(), candidates.end(), time - threshold);
  for (; it != candidates.end() && *it <= time + threshold; ++it) {
    Microseconds diff = std::abs(time - *it);
    if (diff < best_diff) {
      best_diff = diff;
      snapped_edge = *it;
      snapped = true;
    }
  }
}
// Calculates a speed multiplier based on how long a key has been held down.
// Provides acceleration for continuous actions like panning and zooming.
float GetSpeedMultiplier(const ImGuiIO& io, ImGuiKey key) {
  const float down_duration = ImGui::GetKeyData(key)->DownDuration;
  if (down_duration < kAccelerateThreshold) {
    return 1.0f;
  }

  const float accelerated_time = down_duration - kAccelerateThreshold;

  const float multiplier =
      std::min(accelerated_time * kAccelerateRate, kMaxAccelerateFactor);

  return 1.0f + multiplier;
}

// Extracts process sort indices from metadata events in search results.
absl::flat_hash_map<ProcessId, uint32_t> GetProcessSortIndices(
    const ParsedTraceEvents& search_results) {
  absl::flat_hash_map<ProcessId, uint32_t> process_sort_indices;
  for (const TraceEvent& event : search_results.flame_events) {
    if (event.ph == Phase::kMetadata && event.name == kProcessSortIndex) {
      if (const auto it = event.args.find(kSortIndex); it != event.args.end()) {
        double sort_index_double;
        if (absl::SimpleAtod(it->second, &sort_index_double)) {
          process_sort_indices[event.pid] =
              static_cast<uint32_t>(sort_index_double);
        }
      }
    }
  }
  return process_sort_indices;
}

// Draws an expand/collapse button for a group.
bool DrawExpandCollapseButton(Group& group, int group_index, Pixel height) {
  bool toggled = false;
  // Always show the expand/collapse button.
  ImGui::PushID(group_index);
  // Draw a smaller arrow button.
  const Pixel kArrowSize = ImGui::GetFontSize() * kIconSizeScale;
  const Pixel kButtonHeight = height;
  ImVec2 p = ImGui::GetCursorScreenPos();
  // Center the arrow in the button area.
  Pixel center_y = p.y + kButtonHeight * 0.5f;
  Pixel center_x = p.x + kArrowSize * 0.5f;

  // Invisible button for interaction
  if (ImGui::InvisibleButton("##expand_collapse",
                             ImVec2(kArrowSize, kButtonHeight))) {
    group.expanded = !group.expanded;
    toggled = true;
  }

  // Draw the arrow
  ImDrawList* draw_list = ImGui::GetWindowDrawList();
  ImU32 arrow_col = ImGui::GetColorU32(ImGuiCol_Text);
  if (ImGui::IsItemHovered()) {
    ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
    arrow_col = ImGui::GetColorU32(ImGuiCol_ButtonHovered);
  }

  Pixel h = kArrowSize * 0.4f;
  Pixel w = kArrowSize * 0.2f;

  if (group.expanded) {
    // Down arrow, like v
    draw_list->AddLine(ImVec2(center_x - h, center_y - w),
                       ImVec2(center_x, center_y + w), arrow_col, 1.2f);
    draw_list->AddLine(ImVec2(center_x, center_y + w),
                       ImVec2(center_x + h, center_y - w), arrow_col, 1.2f);
  } else {
    // Right arrow, like >
    draw_list->AddLine(ImVec2(center_x - w, center_y - h),
                       ImVec2(center_x + w, center_y), arrow_col, 1.2f);
    draw_list->AddLine(ImVec2(center_x + w, center_y),
                       ImVec2(center_x - w, center_y + h), arrow_col, 1.2f);
  }
  ImGui::PopID();

  return toggled;
}

// Gets the starting level index of the group immediately following the group
// at the given index. If the given group is the last one, returns the total
// number of levels.
int GetNextGroupStartLevel(const FlameChartTimelineData& data,
                           int group_index) {
  if (group_index + 1 < data.groups.size()) {
    return data.groups[group_index + 1].start_level;
  }
  return static_cast<int>(data.events_by_level.size());
}

}  // namespace

// Finds the index of the first visible ancestor (or the group itself if it is
// visible) for a given group index.
//
// During layout pre-computation (`UpdateLevelPositions`), any child group
// nesting under a collapsed (collapsed == !expanded) parent will have its
// visibility flag (`group_visible_[index]`) set to false. All of its
// descendants form a contiguous range of invisible groups because of the
// parent's collapsed state.
//
// If the queried group index points to an invisible group (for instance, when
// locating groups from physical screen coordinates or user interactions), we
// backtrack (decrement index) to find the nearest ancestor group that is
// currently visible.
int Timeline::FindFirstVisibleAncestorIndex(int start_idx) const {
  while (start_idx > 0 && start_idx < group_visible_.size() &&
         !group_visible_[start_idx]) {
    start_idx--;
  }
  return start_idx;
}

void Timeline::UpdateLevelPositions(const FlameChartTimelineData& data) {
  const int level_count = data.events_by_level.size();
  const int group_count = data.groups.size();

  std::vector<Pixel> new_visible_level_offsets(level_count, 0.0f);
  std::vector<Pixel> new_group_offsets(group_count + 1, 0.0f);
  std::vector<bool> new_group_visible(group_count, true);

  Pixel current_offset =
      ImGui::GetCurrentContext() ? ImGui::GetStyle().CellPadding.y : 0.0f;
  int hidden_nesting_level = std::numeric_limits<int>::max();
  Pixel hidden_group_center_y = 0.0f;
  bool in_hidden_process = false;
  bool has_visible_group = false;

  for (int group_index = 0; group_index < group_count; ++group_index) {
    const Group& group = data.groups[group_index];

    if (group.nesting_level == kProcessNestingLevel) {
      in_hidden_process = track_management_enabled_ &&
                          hidden_track_names_.contains(group.name);
    }

    const int next_group_start_level =
        GetNextGroupStartLevel(data, group_index);

    if (in_hidden_process) {
      new_group_offsets[group_index] = current_offset;
      new_group_visible[group_index] = false;
      for (int level = group.start_level; level < next_group_start_level;
           ++level) {
        if (level < level_count) {
          new_visible_level_offsets[level] = current_offset;
        }
      }
      continue;
    }

    if (group.nesting_level <= hidden_nesting_level) {
      hidden_nesting_level = std::numeric_limits<int>::max();
    }

    if (hidden_nesting_level != std::numeric_limits<int>::max()) {
      new_group_offsets[group_index] = current_offset;
      new_group_visible[group_index] = false;
      for (int level = group.start_level; level < next_group_start_level;
           ++level) {
        if (level < level_count) {
          new_visible_level_offsets[level] = hidden_group_center_y;
        }
      }
      continue;
    }

    if (has_visible_group) {
      current_offset += (group.nesting_level == kProcessNestingLevel)
                            ? kProcessTrackGap
                            : kThreadTrackGap;
    }

    new_group_offsets[group_index] = current_offset;
    has_visible_group = true;

    const bool has_children =
        group_index + 1 < data.groups.size() &&
        data.groups[group_index + 1].nesting_level > group.nesting_level;
    const bool has_multiple_levels =
        next_group_start_level - group.start_level > 1;

    const bool expandable = group.type == Group::Type::kFlame &&
                            (has_children || has_multiple_levels);

    const bool is_collapsed = expandable && !group.expanded;

    Pixel group_height = kEventHeight;
    if (group.nesting_level == kProcessNestingLevel) {
      group_height = kProcessTrackHeight;
    } else if (!is_collapsed) {
      if (group.type == Group::Type::kCounter) {
        group_height = kCounterTrackHeight;
      } else if (group.type == Group::Type::kFlame) {
        group_height = std::max(1, next_group_start_level - group.start_level) *
                       (kEventHeight + kEventPaddingBottom);
      }
    }

    if (is_collapsed &&
        hidden_nesting_level == std::numeric_limits<int>::max()) {
      hidden_nesting_level = group.nesting_level;
      hidden_group_center_y = current_offset + group_height * 0.5f;
    }

    const int start_level = group.start_level;

    if (is_collapsed) {
      for (int level = start_level; level < next_group_start_level; ++level) {
        if (level < level_count) {
          new_visible_level_offsets[level] =
              current_offset + group_height * 0.5f;
        }
      }
    } else {
      for (int level = start_level; level < next_group_start_level; ++level) {
        if (level < level_count) {
          new_visible_level_offsets[level] =
              current_offset +
              (level - start_level) * (kEventHeight + kEventPaddingBottom) +
              kEventHeight * 0.5f;
        }
      }
    }

    current_offset += group_height;
  }

  new_group_offsets[group_count] = current_offset;

  group_offsets_ = std::move(new_group_offsets);
  visible_level_offsets_ = std::move(new_visible_level_offsets);
  group_visible_ = std::move(new_group_visible);
}

void Timeline::SetVisibleRange(const TimeRange& range, bool animate) {
  if (animate) {
    visible_range_ = range;
  } else {
    visible_range_.snap_to(range);
  }
  if (redraw_callback_) redraw_callback_();
}

void Timeline::SetTimelineData(FlameChartTimelineData data) {
  // Pre-calculate the level positions to avoid partial state and per-frame
  // layout recalculations before saving the newly arrived timeline_data.
  UpdateLevelPositions(data);
  timeline_data_ = std::move(data);

  // Reconcile loaded_index for all search results against the new
  // timeline_data_.
  const int num_loaded = timeline_data_.entry_event_ids.size();
  absl::flat_hash_map<EventId, int> event_id_to_loaded_index;
  event_id_to_loaded_index.reserve(num_loaded);
  for (int i = 0; i < num_loaded; ++i) {
    event_id_to_loaded_index.try_emplace(timeline_data_.entry_event_ids[i], i);
  }

  std::optional<absl::flat_hash_map<FallbackKey, std::vector<int>>>
      lazy_loaded_events;

  matching_event_indices_.clear();
  for (SearchResult& result : search_results_) {
    if (const auto it = event_id_to_loaded_index.find(result.event_id);
        it != event_id_to_loaded_index.end()) {
      result.loaded_index = it->second;
    } else {
      if (!lazy_loaded_events.has_value()) {
        lazy_loaded_events = BuildFallbackMap(timeline_data_);
      }
      result.loaded_index = FindFallbackMatchedIndex(
          *lazy_loaded_events, timeline_data_, result.pid, result.tid,
          result.name, result.start_time, result.duration);
      if (result.loaded_index != -1) {
        result.event_id = timeline_data_.entry_event_ids[result.loaded_index];
      }
    }
    if (result.loaded_index != -1) {
      matching_event_indices_.insert(result.loaded_index);
    }
  }

  if (pending_navigation_event_id_.has_value()) {
    int matched_index = -1;
    if (current_search_result_index_ >= 0 &&
        current_search_result_index_ < search_results_.size()) {
      matched_index =
          search_results_.at(current_search_result_index_).loaded_index;
    }

    if (matched_index != -1) {
      ZoomEvent(matched_index);
      pending_navigation_event_id_.reset();
    }
  } else if (current_search_result_index_ >= 0 &&
             current_search_result_index_ < search_results_.size()) {
    const SearchResult& active =
        search_results_.at(current_search_result_index_);
    selected_event_index_ = active.loaded_index;
  }

  if (is_incremental_loading_) {
    should_restore_scroll_ = true;
  }

  if (redraw_callback_) redraw_callback_();
}

void Timeline::Draw() {
  hovered_event_index_ = -1;
  event_clicked_this_frame_ = false;

  const ImGuiViewport* viewport = ImGui::GetMainViewport();
  ImGui::SetNextWindowPos(viewport->Pos);
  ImGui::SetNextWindowSize(viewport->Size);

  ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.f);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
  ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, ImVec2(0.0f, 0.0f));
  ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0.0f, 0.0f));

  ImGui::Begin("Timeline viewer", nullptr, kImGuiWindowFlags);

  // Calculate the available width for the timeline before entering the table.
  // This ensures we get the correct width even if the table layout hasn't
  // finished or if GetContentRegionAvail behaves differently inside the table.
  // Reserve space for the vertical scrollbar so that the timeline layout
  // doesn't shift horizontally when the scrollbar appears/disappears due to
  // drawer resizing. We calculate the fixed table width based on the parent
  // window's available width, minus the scrollbar size.
  const Pixel content_region_avail_width =
      ImGui::GetWindowWidth() - ImGui::GetStyle().ScrollbarSize;

  current_timeline_width_ =
      content_region_avail_width - label_width_ - kTimelinePaddingRight;

  const double px_per_time_unit_val = px_per_time_unit(current_timeline_width_);
  const TickInfo tick_info = CalculateTickInfo(px_per_time_unit_val);

  const ImVec2 ruler_start_pos = ImGui::GetCursorPos();
  const ImVec2 ruler_start_screen_pos = ImGui::GetCursorScreenPos();
  ruler_screen_y_ = ruler_start_screen_pos.y;

  // Draw Ruler background anchored to the top (outside the scrollable child
  // window). The background starts *after* the left label area according to
  // user request.
  ImGui::GetWindowDrawList()->AddRectFilled(
      ImVec2(ruler_start_screen_pos.x + label_width_, ruler_start_screen_pos.y),
      ImVec2(ruler_start_screen_pos.x + content_region_avail_width,
             ruler_start_screen_pos.y + kRulerHeight),
      palette_.GetColor(ColorPalette::Key::kBackground).value_or(kWhiteColor));

  ImGui::SetCursorPos(ruler_start_pos);
  DrawRulerUI(tick_info, current_timeline_width_);

  // Now move the cursor below the Ruler to start the Tracks child
  ImGui::SetCursorPos(
      ImVec2(ruler_start_pos.x, ruler_start_pos.y + kRulerHeight));

  // The tracks are in a child window to allow scrolling independently of the
  // ruler.
  // Keep the NoScrollWithMouse flag to disable the default scroll behavior
  // of ImGui, and use the custom scroll handler defined in `HandleWheel`
  // instead.
  ImGui::BeginChild("Tracks", ImVec2(0, 0), 0,
                    ImGuiWindowFlags_NoScrollWithMouse);

  if (should_restore_scroll_) {
    ImGui::SetScrollY(last_scroll_y_);
    should_restore_scroll_ = false;
  }

  // We set cursor to 0,0 locally
  const ImVec2 tracks_start_pos = ImGui::GetCursorPos();
  const ImVec2 tracks_start_screen_pos = ImGui::GetCursorScreenPos();
  tracks_start_screen_pos_ = tracks_start_screen_pos;

  DrawVerticalGridLines(tick_info, current_timeline_width_,
                        viewport->Pos.y + viewport->Size.y);

  const Pixel scroll_y = ImGui::GetScrollY();
  const Pixel window_height = ImGui::GetWindowHeight();

  // Find the first group that is visible.
  auto start_it = std::upper_bound(group_offsets_.begin() + 1,
                                   group_offsets_.end(), scroll_y);
  int start_idx = std::distance(group_offsets_.begin(), start_it) - 1;
  start_idx = std::max(
      0, std::min(start_idx, static_cast<int>(timeline_data_.groups.size())));

  // If the start_idx lands on a child of a collapsed group, scan backwards
  // to find the first visible ancestor using our pre-computed visibility map.
  start_idx = FindFirstVisibleAncestorIndex(start_idx);

  // Find the last group that is visible.
  auto end_it = std::upper_bound(group_offsets_.begin(), group_offsets_.end(),
                                 scroll_y + window_height);
  int end_idx = timeline_data_.groups.size();
  if (end_it != group_offsets_.end()) {
    end_idx = std::distance(group_offsets_.begin(), end_it);
  }

  for (int group_index = start_idx; group_index < end_idx; ++group_index) {
    if (!group_visible_[group_index]) continue;
    Group& group = timeline_data_.groups[group_index];
    ImGui::PushID(group_index);

    // Set cursor to draw the label
    ImGui::SetCursorPos(ImVec2(
        tracks_start_pos.x, tracks_start_pos.y + group_offsets_[group_index]));

    ImDrawList* draw_list = ImGui::GetWindowDrawList();

    if (group.nesting_level == kProcessNestingLevel) {
      if (track_management_enabled_ &&
          hidden_track_names_.contains(group.name)) {
        ImGui::PopID();
        continue;
      }
      ImU32 bg_color =
          group.expanded
              ? palette_.GetColor(ColorPalette::Key::kExpandedHeader)
                    .value_or(kProcessTrackExpandedColor)
              : palette_.GetColor(ColorPalette::Key::kCollapsedHeader)
                    .value_or(kProcessTrackCollapsedColor);
      draw_list->AddRectFilled(
          ImVec2(tracks_start_screen_pos.x,
                 tracks_start_screen_pos.y + group_offsets_[group_index]),
          ImVec2(tracks_start_screen_pos.x + content_region_avail_width,
                 tracks_start_screen_pos.y + group_offsets_[group_index + 1]),
          bg_color);
    }

    // Push clip rect to prevent label text from bleeding into the track area
    ImGui::PushClipRect(
        ImVec2(tracks_start_screen_pos.x,
               tracks_start_screen_pos.y + group_offsets_[group_index]),
        ImVec2(tracks_start_screen_pos.x + label_width_ - kSplitterOffset,
               tracks_start_screen_pos.y + group_offsets_[group_index + 1]),
        true);

    const bool has_children =
        group_index + 1 < timeline_data_.groups.size() &&
        timeline_data_.groups[group_index + 1].nesting_level >
            group.nesting_level;
    const int next_group_start_level =
        GetNextGroupStartLevel(timeline_data_, group_index);
    const bool has_multiple_levels =
        next_group_start_level - group.start_level > 1;

    const bool expandable = group.type == Group::Type::kFlame &&
                            (has_children || has_multiple_levels);

    const bool is_collapsed = expandable && !group.expanded;
    Pixel group_height = kEventHeight;
    if (group.nesting_level == kProcessNestingLevel) {
      group_height = kProcessTrackHeight;
    } else if (!is_collapsed) {
      if (group.type == Group::Type::kCounter) {
        group_height = kCounterTrackHeight;
      } else if (group.type == Group::Type::kFlame) {
        const int end_level =
            GetNextGroupStartLevel(timeline_data_, group_index);
        group_height = std::max(1, end_level - group.start_level) *
                       (kEventHeight + kEventPaddingBottom);
      }
    }

    const Pixel kArrowSize = ImGui::GetFontSize() * kIconSizeScale;
    // We add 1 to the nesting level because ImGui::Indent(0) results in a
    // default, potentially large indentation. By adding 1, even top-level
    // groups (nesting_level 0) receive a base indentation of `kIndentSize`,
    // ensuring consistent and controlled visual separation from the left edge.
    Pixel indent_amount = (group.nesting_level + 1) * kIndentSize;
    if (!expandable && group.nesting_level > 0) {
      // If a group is not expandable (e.g., a thread with no sub-threads or
      // only one level), align its label indentation with top-level groups
      // (nesting_level 0) for better visual consistency.
      indent_amount = kIndentSize;
    }

    const Pixel label_start_y = ImGui::GetCursorPosY();
    // Use the first level's height for centering if the track has multiple
    // levels, to keep the label near the top. Process track uses its height.
    const Pixel centereable_height =
        group.nesting_level == kProcessNestingLevel
            ? kProcessTrackHeight
            : (group.type == Group::Type::kFlame ? kEventHeight : group_height);

    ImGui::Indent(indent_amount);

    if (expandable) {
      if (DrawExpandCollapseButton(group, group_index, centereable_height)) {
        UpdateLevelPositions(timeline_data_);
        if (redraw_callback_) redraw_callback_();
      }
    } else {
      ImGui::Dummy(ImVec2(kArrowSize, centereable_height));
    }

    ImGui::SameLine();
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + kLabelPaddingLeft);

    ImGui::PushFont(traceviewer::fonts::label_large);
    const Pixel text_height_large = ImGui::GetTextLineHeight();
    ImGui::PopFont();

    ImGui::PushFont(traceviewer::fonts::label_medium);
    const Pixel text_height_medium = ImGui::GetTextLineHeight();
    ImGui::PopFont();

    const bool has_subtitle = !group.subtitle.empty();

    // We let ImGui native stacking handle the gap.
    const Pixel spacing = ImGui::GetStyle().ItemSpacing.y;
    const Pixel total_text_height =
        has_subtitle ? (text_height_large + spacing + text_height_medium)
                     : text_height_large;

    // Perfectly vertically center the text sequence within the track's height.
    const Pixel vertical_offset =
        (centereable_height - total_text_height) * 0.5f;
    ImGui::SetCursorPosY(label_start_y + std::max(0.0f, vertical_offset));

    // Begin a Group so both texts form a SINGLE item on the SameLine as the
    // toggle arrow! This perfectly isolates the text so they stack normally
    // without being pushed below the 50px Dummy.
    ImGui::BeginGroup();

    absl::string_view display_name = group.name;
    if (has_subtitle) {
      display_name.remove_prefix(group.subtitle.size() + 1);
      if (!display_name.empty() && display_name.front() == '/') {
        display_name.remove_prefix(1);
      }
    }

    ImGui::PushFont(traceviewer::fonts::label_large);
    ImGui::TextUnformatted(display_name.data(),
                           display_name.data() + display_name.size());
    ImGui::PopFont();

    if (has_subtitle) {
      ImGui::PushFont(traceviewer::fonts::label_medium);
      ImGui::PushStyleColor(ImGuiCol_Text,
                            palette_.GetColor(ColorPalette::Key::kSubtitle)
                                .value_or(kOnSecondaryFixedVariantColor));
      ImGui::TextUnformatted(group.subtitle.data());
      ImGui::PopStyleColor();
      ImGui::PopFont();
    }

    ImGui::EndGroup();

    if (ImGui::IsItemHovered()) {
      ImGui::SetMouseCursor(ImGuiMouseCursor_TextInput);
      if (ImGui::IsMouseClicked(0)) {
        ImGui::SetClipboardText(group.name.c_str());
        traceviewer::CopyToClipboard(group.name);

        copied_track_name_ = group.name;
        copy_notification_timer_ = 2.0f;
      }
    }

    // "Selecting text" visual effect when clicked/timer active
    if (copy_notification_timer_ > 1.8f && copied_track_name_ == group.name) {
      ImVec2 group_min = ImGui::GetItemRectMin();
      ImVec2 group_max = ImGui::GetItemRectMax();
      // Draw a blue highlight over the text to simulate browser selection
      ImGui::GetWindowDrawList()->AddRectFilled(group_min, group_max,
                                                IM_COL32(66, 133, 244, 128));
    }

    ImGui::Unindent(indent_amount);
    ImGui::SetCursorPosY(label_start_y);
    ImGui::PopClipRect();

    if (track_management_enabled_) {
      if (group.nesting_level == kProcessNestingLevel) {
        ImGui::SameLine();
        const Pixel kArrowSize = ImGui::GetFontSize() * kIconSizeScale;
        ImGui::SetCursorPosX(tracks_start_pos.x + label_width_ -
                            kSplitterOffset - kArrowSize);
        const bool is_track_hidden = hidden_track_names_.contains(group.name);
        DrawHideButton(
            group_index, centereable_height, is_track_hidden);
      }
    }

    ImGui::SetCursorPos(
        ImVec2(tracks_start_pos.x + label_width_,
               tracks_start_pos.y + group_offsets_[group_index]));

    if (is_collapsed) {
      DrawGroupPreview(group_index, px_per_time_unit_val);
      int current_nesting_level = group.nesting_level;
      while (group_index + 1 < timeline_data_.groups.size() &&
             timeline_data_.groups[group_index + 1].nesting_level >
                 current_nesting_level) {
        group_index++;
      }
    } else {
      DrawGroup(group_index, px_per_time_unit_val, scroll_y, window_height);
    }
    ImGui::PopID();
  }

  // Create a dummy at the end to ensure the Tracks child has the right
  // scrolling height. We position the top of a 1px high dummy at
  // `group_offsets_.back() - 1.0f`. This means the bottom of the dummy is
  // exactly at `group_offsets_.back()`, ensuring the content size of the
  // child window matches the total calculated height from group offsets.
  // Using a 1px dummy instead of a 0-height dummy helps avoid potential issues
  // with ImGui's `ItemSpacing` adding extra space around zero-sized items.
  ImGui::SetCursorPos(
      ImVec2(0, tracks_start_pos.y + group_offsets_.back() - 1.0f));
  ImGui::Dummy(ImVec2(content_region_avail_width, 1.0f));

  // Handle label resizing manually since we removed the table
  ImGui::SetCursorPos(ImVec2(
      tracks_start_pos.x + label_width_ - kSplitterOffset, tracks_start_pos.y));
  ImGui::InvisibleButton("##LabelResizer",
                         ImVec2(kSplitterWidth, group_offsets_.back()));
  if (ImGui::IsItemActive()) {
    label_width_ += ImGui::GetIO().MouseDelta.x;
    label_width_ = std::max(10.0f, label_width_);
    is_resizing_label_column_ = true;
  } else {
    is_resizing_label_column_ = false;
  }
  if (ImGui::IsItemHovered()) {
    ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
  }

  HandleEventDeselection();

  // Handle continuous keyboard and mouse wheel input for timeline navigation.
  // These functions are called every frame to ensure smooth and responsive
  // interaction.
  // The performance impact is fine because HandleKeyboard/HandleWheel() only
  // performs lightweight checks and calculations.
  bool is_interacting = is_resizing_label_column_;
  if (!is_resizing_label_column_) {
    is_interacting |= HandleKeyboard();
    is_interacting |= HandleWheel();
    is_interacting |= HandleMouse();
  }

  // We call MaybeRequestData() to check if we need to fetch more data.
  // We do this here instead of in SetVisibleRange because we want to debounce
  // the request during continuous interaction (like panning/zooming).
  // The check is skipped if the user is currently interacting with
  // the timeline to avoid sending requests during interaction.
  if (!is_interacting) {
    MaybeRequestData();
  }

  ProcessPendingScroll();

  last_scroll_y_ = ImGui::GetScrollY();
  ImGui::EndChild();

  // Draw the selected time range in a separate overlay child window.
  // This ensures it is drawn on top of the "Tracks" child window (because it's
  // declared after) but below tooltips (because it's a child window, not
  // in the foreground draw list).
  ImGui::SetCursorPos(ImVec2(0, 0));
  ImGui::BeginChild("SelectionOverlay", ImVec2(0, 0), 0,
                    ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                        ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar |
                        ImGuiWindowFlags_NoSavedSettings |
                        ImGuiWindowFlags_NoInputs |
                        ImGuiWindowFlags_NoBackground);
  // `DrawFlows` and `DrawSelectedTimeRanges` should be called after all other
  // timeline content (events, ruler, etc.) has been drawn. This ensures that
  // flow lines and selected time ranges are rendered on top of everything
  // else within the current ImGui window, without affecting global foreground
  // elements like tooltips.
  DrawFlows(current_timeline_width_, tracks_start_screen_pos.y);
  DrawSelectedTimeRanges(current_timeline_width_, px_per_time_unit_val);
  DrawBookmarks(current_timeline_width_, px_per_time_unit_val);
  DrawSelectionRectangle();

  // Draw vertical split line between sidebar and tracks
  // Drawn last inside SelectionOverlay so it sits on top of other elements,
  // and extends upwards to the beginning of the ruler.
  Pixel split_x = std::floor(ruler_start_screen_pos.x + label_width_) + 0.5f;
  ImGui::GetWindowDrawList()->AddLine(
      ImVec2(split_x, ruler_start_screen_pos.y),
      ImVec2(split_x, ruler_start_screen_pos.y + ImGui::GetWindowHeight()),
      ImGui::GetColorU32(ImGuiCol_TableBorderLight), 1.0f);

  ImGui::EndChild();

  DrawToast(absl::StrCat("Copied track name: ", copied_track_name_),
            copy_notification_timer_, 32.0f);

  float bounds_toast_offset = 32.0f;
  if (copy_notification_timer_ > 0.0f) {
    bounds_toast_offset += 48.0f;
  }
  DrawToast(bounds_notification_message_, bounds_notification_timer_,
            bounds_toast_offset);

  ImGui::PopStyleVar();  // ItemSpacing
  ImGui::PopStyleVar();  // CellPadding
  ImGui::PopStyleVar();  // WindowPadding
  ImGui::PopStyleVar();  // WindowBorderSize
  ImGui::PopStyleVar();  // WindowRounding
  ImGui::End();          // Timeline viewer
}

EventRect Timeline::CalculateEventRect(
    Microseconds start, Microseconds end, Pixel screen_x_offset,
    Pixel screen_y_offset, double px_per_time_unit, int level_in_group,
    Pixel timeline_width, Pixel event_height, Pixel padding_bottom) const {
  const Pixel left = TimeToScreenX(start, screen_x_offset, px_per_time_unit);
  Pixel right = TimeToScreenX(end, screen_x_offset, px_per_time_unit);

  // Ensure minimum width for visibility.
  right = std::max(right, left + kEventMinimumDrawWidth);
  // Add a small gap to the right of the event for visual separation.
  // This is done here instead of in the Draw function to ensure the gap is
  // visible even if the event name overflows the right edge of the event. We
  // only adjust `right` to ensure the `left` boundary accurately reflects the
  // event's start time.
  right -= kEventPaddingRight;

  const Pixel top =
      screen_y_offset + level_in_group * (event_height + padding_bottom);
  const Pixel bottom = top + event_height;

  const Pixel timeline_right_boundary = screen_x_offset + timeline_width;

  // If the event ends before the visible area starts, return a zero-width
  // rectangle at the left boundary.
  if (right < screen_x_offset) {
    return {screen_x_offset, top, screen_x_offset, bottom};
  }
  // If the event starts after the visible area ends, return a zero-width
  // rectangle at the right boundary.
  if (left > timeline_right_boundary) {
    return {timeline_right_boundary, top, timeline_right_boundary, bottom};
  }

  // Clip the event rectangle to the visible window bounds.
  const Pixel clipped_left = std::max(left, screen_x_offset);
  const Pixel clipped_right = std::min(right, timeline_right_boundary);

  return {clipped_left, top, clipped_right, bottom};
}

ImVec2 Timeline::CalculateEventTextRect(absl::string_view event_name,
                                        const EventRect& event_rect) const {
  const ImVec2 text_size = GetTextSize(event_name);

  // Center the text within the clipped visible portion of the event.
  const Pixel clipped_width = event_rect.right - event_rect.left;
  const Pixel text_x = event_rect.left + (clipped_width - text_size.x) * 0.5f;
  const Pixel event_height = event_rect.bottom - event_rect.top;
  const Pixel text_y = event_rect.top + (event_height - text_size.y) * 0.5f;

  // Ensure the text starts at least at the left boundary of the event rect.
  // ImGui's PushClipRect in DrawEventName will handle the right boundary
  // clipping.
  const Pixel text_x_clipped = std::max(text_x, event_rect.left);

  return ImVec2(text_x_clipped, text_y);
}

std::string Timeline::GetTextForDisplay(absl::string_view event_name,
                                        Pixel available_width) const {
  const Pixel text_width = GetTextSize(event_name).x;

  if (text_width > available_width) {
    // Truncate text with "..." at the end
    const Pixel ellipsis_width = GetTextSize("...").x;
    if (available_width <= ellipsis_width) {
      return "";
    }

    // Binary search for the longest prefix that fits within the available
    // width.
    int low = 0, high = event_name.length(), fit_len = 0;
    while (low <= high) {
      const int mid = low + (high - low) / 2;
      if (GetTextSize(absl::string_view(event_name.data(), mid)).x +
              ellipsis_width <=
          available_width) {
        fit_len = mid;
        low = mid + 1;
      } else {
        high = mid - 1;
      }
    }

    if (fit_len == 0) {
      return "";
    }

    return absl::StrCat(event_name.substr(0, fit_len), "...");
  }
  return std::string(event_name);
}

Microseconds Timeline::PixelToTime(Pixel pixel_offset,
                                   double px_per_time_unit) const {
  if (px_per_time_unit <= 0) return visible_range().start();
  return visible_range().start() +
         (static_cast<Microseconds>(pixel_offset) / px_per_time_unit);
}

Pixel Timeline::TimeToPixel(Microseconds time, double px_per_time_unit) const {
  if (px_per_time_unit <= 0) return 0;
  return static_cast<Pixel>((time - visible_range_->start()) *
                            px_per_time_unit);
}

Pixel Timeline::TimeToScreenX(Microseconds time, Pixel screen_x_offset,
                              double px_per_time_unit) const {
  return screen_x_offset + TimeToPixel(time, px_per_time_unit);
}

void Timeline::ConstrainTimeRange(TimeRange& range) {
  if (range.duration() < kMinDurationMicros) {
    double center = range.center();
    range = {center - kMinDurationMicros / 2.0,
             center + kMinDurationMicros / 2.0};
  }
  if (range.start() < data_time_range_.start()) {
    // When shifting the start to data_time_range_.start(), ensure the
    // new end does not exceed data_time_range_.end().
    range = {data_time_range_.start(),
             std::min(range.end() + data_time_range_.start() - range.start(),
                      data_time_range_.end())};
  } else if (range.end() > data_time_range_.end()) {
    // When shifting the end to data_time_range_.end(), ensure the new
    // start does not go before data_time_range_.start() by taking the
    // maximum.
    range = {std::max(range.start() - range.end() + data_time_range_.end(),
                      data_time_range_.start()),
             data_time_range_.end()};
  }
}

void Timeline::EmitEventSelected(int event_index) {
  EventData event_data;
  event_data.try_emplace(kEventSelectedIndex, event_index);
  event_data.try_emplace(kEventSelectedName,
                         timeline_data_.entry_names[event_index]);
  event_data.try_emplace(kEventSelectedStart,
                         timeline_data_.entry_start_times[event_index]);
  event_data.try_emplace(kEventSelectedDuration,
                         timeline_data_.entry_total_times[event_index]);
  event_data.try_emplace(
      kEventSelectedStartFormatted,
      FormatTime(timeline_data_.entry_start_times[event_index]));
  event_data.try_emplace(
      kEventSelectedDurationFormatted,
      FormatTime(timeline_data_.entry_total_times[event_index]));
  event_data.try_emplace(
      kEventSelectedPid,
      static_cast<double>(timeline_data_.entry_pids[event_index]));
  const absl::flat_hash_map<std::string, std::string>& args =
      timeline_data_.entry_args[event_index];
  if (const auto it = args.find("uid"); it != args.end()) {
    event_data.try_emplace(kEventSelectedUid, it->second);
  }
  if (const auto it = args.find(kHloModule); it != args.end()) {
    event_data.try_emplace(kEventSelectedHloModuleName, it->second);
  }
  if (const auto it = args.find(kHloOp); it != args.end()) {
    event_data.try_emplace(kEventSelectedHloOpName, it->second);
  }
  event_callback_(kEventSelected, event_data);
}

void Timeline::EmitViewportChanged(const TimeRange& range) {
  EventData range_obj;
  range_obj[kViewportChangedMin] = range.start();
  range_obj[kViewportChangedMax] = range.end();
  EventData detail_obj;
  detail_obj[kViewportChangedRange] = range_obj;
  event_callback_(kViewportChanged, detail_obj);
}

void Timeline::EmitMouseModeChanged() {
  if (!event_callback_) return;
  EventData event_data;
  event_data.try_emplace(kMouseModeKey, static_cast<int>(mouse_mode_));
  event_callback_(kMouseModeChanged, event_data);
}

void Timeline::ShowBoundsNotification(const std::string& message) {
  bounds_notification_message_ = message;
  bounds_notification_timer_ = 2.0f;
  if (redraw_callback_) redraw_callback_();
}

void Timeline::RevealEvent(int event_index) {
  if (event_index < 0 ||
      event_index >= timeline_data_.entry_start_times.size() ||
      event_index >= timeline_data_.entry_total_times.size()) {
    LOG(ERROR) << "Invalid event index: " << event_index;
    return;
  }

  selected_event_index_ = event_index;
  event_index_to_scroll_to_ = event_index;
  ExpandRelatedTracks(event_index);

  const Microseconds start = timeline_data_.entry_start_times[event_index];
  Microseconds event_duration = timeline_data_.entry_total_times[event_index];
  // Marker entries or zero duration events.
  if (std::isnan(event_duration) || event_duration <= 0) {
    event_duration = kMinVisibleEventDuration;
  }
  const Microseconds end = start + event_duration;
  const Microseconds time_left = visible_range().start();
  const Microseconds time_right = visible_range().end();
  const Microseconds current_duration = visible_range().duration();

  Microseconds min_entry_time_window =
      std::min(event_duration, current_duration);

  // Ensure at least 30 pixels are visible.
  double min_visible_width_px = 30.0;
  double px_per_time = px_per_time_unit();
  if (px_per_time > 0) {
    double time_per_px = 1.0 / px_per_time;
    min_entry_time_window =
        std::max(min_entry_time_window, time_per_px * min_visible_width_px);
  }

  if (time_left > end) {
    double delta = time_left - end + min_entry_time_window;
    SetVisibleRange({time_left - delta, time_right - delta}, /*animate=*/true);
  } else if (time_right < start) {
    double delta = start - time_right + min_entry_time_window;
    SetVisibleRange({time_left + delta, time_right + delta}, /*animate=*/true);
  }

  EmitEventSelected(event_index);
}

void Timeline::ZoomEvent(int event_index) {
  if (event_index < 0 ||
      event_index >= timeline_data_.entry_start_times.size() ||
      event_index >= timeline_data_.entry_total_times.size()) {
    LOG(ERROR) << "Invalid event index: " << event_index;
    return;
  }

  selected_event_index_ = event_index;
  event_index_to_scroll_to_ = event_index;
  ExpandRelatedTracks(event_index);

  const Microseconds start = timeline_data_.entry_start_times[event_index];
  const Microseconds event_duration =
      timeline_data_.entry_total_times[event_index];
  // When navigating to an event, set the visible duration to 2.5 times the
  // event's duration to provide context around the event. Clamp the
  // duration between 10us and 5s to prevent zooming in too far on
  // short events or zooming out too far on long events.
  const Microseconds duration =
      std::max(kEventNavigationMinDurationMicros,
               std::min(event_duration * kEventNavigationZoomFactor,
                        kEventNavigationMaxDurationMicros));
  const Microseconds center = start + event_duration / 2.0;
  TimeRange new_range = {center - duration / 2.0, center + duration / 2.0};
  ConstrainTimeRange(new_range);

  SetVisibleRange(new_range, /*animate=*/true);

  EmitEventSelected(event_index);
}

void Timeline::ExpandRelatedTracks(int event_index) {
  int level = timeline_data_.entry_levels[event_index];
  int group_index = -1;
  for (size_t i = 0; i < timeline_data_.groups.size(); ++i) {
    int next_group_start_level = GetNextGroupStartLevel(timeline_data_, i);
    if (level >= timeline_data_.groups[i].start_level &&
        level < next_group_start_level) {
      group_index = i;
      break;
    }
  }

  if (group_index != -1) {
    bool changed = false;
    if (!timeline_data_.groups[group_index].expanded) {
      timeline_data_.groups[group_index].expanded = true;
      changed = true;
    }
    int current_nesting = timeline_data_.groups[group_index].nesting_level;
    for (int i = group_index - 1; i >= 0 && current_nesting > 0; --i) {
      if (timeline_data_.groups[i].nesting_level < current_nesting) {
        if (!timeline_data_.groups[i].expanded) {
          timeline_data_.groups[i].expanded = true;
          changed = true;
        }
        current_nesting = timeline_data_.groups[i].nesting_level;
      }
    }
    if (changed) {
      UpdateLevelPositions(timeline_data_);
      if (redraw_callback_) redraw_callback_();
    }
  }
}

void Timeline::HideTrack(absl::string_view name) {
  hidden_track_names_.insert(std::string(name));
  UpdateLevelPositions(timeline_data_);
  if (redraw_callback_) redraw_callback_();
}


void Timeline::CalculateBezierControlPoints(float start_x, float start_y,
                                            float end_x, float end_y,
                                            ImVec2& cp0, ImVec2& cp1) {
  const Pixel dist = std::abs(end_x - start_x) * 0.5f;
  cp0 = ImVec2(start_x + dist, start_y);
  cp1 = ImVec2(end_x - dist, end_y);
}

void Timeline::Pan(Pixel pixel_amount) {
  // If the pixel amount is 0, we don't need to pan.
  if (pixel_amount == 0.0) return;

  const double px_per_time_unit_val = px_per_time_unit();
  // This should never happen, but we check it to avoid division by zero.
  if (px_per_time_unit_val <= 0.0) return;

  const double time_offset = pixel_amount / px_per_time_unit_val;
  TimeRange new_range = visible_range_.target() + time_offset;

  const bool showing_entire_trace =
      visible_range_.target().start() <= data_time_range_.start() &&
      visible_range_.target().end() >= data_time_range_.end();

  if (!showing_entire_trace) {
    if (pixel_amount < 0.0 && new_range.start() < data_time_range_.start()) {
      ShowBoundsNotification(
          "Cannot pan further left: reached the beginning of the trace.");
    } else if (pixel_amount > 0.0 && new_range.end() > data_time_range_.end()) {
      ShowBoundsNotification(
          "Cannot pan further right: reached the end of the trace.");
    }
  }

  ConstrainTimeRange(new_range);

  // Update the target of the animated visible range. The timeline will animate
  // towards this new time.
  SetVisibleRange(new_range, /*animate=*/true);
  EmitViewportChanged(new_range);
}

void Timeline::Scroll(Pixel pixel_amount) {
  // If the pixel amount is 0, we don't need to scroll.
  if (pixel_amount == 0.0) return;

  ImGui::SetScrollY(ImGui::GetScrollY() + pixel_amount);
}

void Timeline::Zoom(float zoom_factor) {
  const ImRect timeline_area = GetTimelineArea();
  // Use the latest mouse position as the zoom pivot. However, if the mouse is
  // outside the timeline area, we fallback to using the current visible range
  // center.
  // We use IsMouseHoveringRect solely to check if the mouse position is within
  // the timeline bounds, as we don't rely on the window's hover state.
  if (ImGui::IsMouseHoveringRect(timeline_area.Min, timeline_area.Max)) {
    const double px_per_time = px_per_time_unit();
    const Microseconds pivot =
        PixelToTime(ImGui::GetMousePos().x - timeline_area.Min.x, px_per_time);

    Zoom(zoom_factor, pivot);
  } else {
    Zoom(zoom_factor, visible_range_->center());
  }
}

void Timeline::Zoom(float zoom_factor, Microseconds pivot) {
  // If the zoom factor is 1, we don't need to zoom.
  if (zoom_factor == 1.0) return;

  // Clamp the zoom factor to the minimum value. This prevents the time
  // durations (mathmatically) become zero or negative.
  zoom_factor = std::max(zoom_factor, kMinZoomFactor);

  TimeRange new_range = visible_range_.target();
  new_range.Zoom(zoom_factor, pivot);

  if (zoom_factor < 1.0) {
    if (new_range.duration() < kMinDurationMicros) {
      ShowBoundsNotification(
          "Cannot zoom in further: minimum zoom duration reached.");
    }
  } else if (zoom_factor > 1.0) {
    TimeRange constrained_range = new_range;
    ConstrainTimeRange(constrained_range);
    if (constrained_range.duration() < new_range.duration() ||
        (visible_range_.target().start() <= data_time_range_.start() &&
         visible_range_.target().end() >= data_time_range_.end())) {
      ShowBoundsNotification(
          "Cannot zoom out further: showing the entire trace.");
    }
  }

  ConstrainTimeRange(new_range);

  // Update the target of the animated visible range. The timeline will animate
  // towards this new zoom level.
  SetVisibleRange(new_range, /*animate=*/true);
  EmitViewportChanged(new_range);
}

void Timeline::ApplySnapping(TimeRange& range) {
  // Global flag to enable or disable the snapping feature.
  if (!snap_to_time_range_enabled_) {
    return;
  }
  // Snapping only applies when in measuring mode (kTiming) or when Shift+Drag
  // is used to select a time range.
  if (mouse_mode_ != MouseMode::kTiming && !ImGui::GetIO().KeyShift) {
    return;
  }
  const double px_per_time = px_per_time_unit();
  if (px_per_time <= 0) return;

  const Microseconds threshold = 16.0 / px_per_time;
  const Microseconds original_duration = visible_range_.target().duration();
  const bool is_pan = std::abs(range.duration() - original_duration) < 1e-6;

  Microseconds best_diff_start = threshold;
  Microseconds best_diff_end = threshold;
  Microseconds snapped_start_time = range.start();
  Microseconds snapped_end_time = range.end();
  bool snapped_start = false;
  bool snapped_end = false;

  // 1. Check selected time ranges
  for (const auto& sel_range : selected_time_ranges_) {
    ApplySnappingToEdge(range.start(), threshold,
                        {sel_range.start(), sel_range.end()}, best_diff_start,
                        snapped_start_time, snapped_start);
    ApplySnappingToEdge(range.end(), threshold,
                        {sel_range.start(), sel_range.end()}, best_diff_end,
                        snapped_end_time, snapped_end);
  }

  // 2. Check all visible events
  FindNearestEventEdge(range.start(), threshold, best_diff_start,
                       snapped_start_time, snapped_start);
  FindNearestEventEdge(range.end(), threshold, best_diff_end, snapped_end_time,
                       snapped_end);

  if (is_pan) {
    if (snapped_start) {
      range = {snapped_start_time, snapped_start_time + original_duration};
    } else if (snapped_end) {
      range = {snapped_end_time - original_duration, snapped_end_time};
    }
  } else {
    Microseconds final_start =
        snapped_start ? snapped_start_time : range.start();
    Microseconds final_end = snapped_end ? snapped_end_time : range.end();
    range = TimeRange(std::min(final_start, final_end),
                      std::max(final_start, final_end));
  }
}

void Timeline::FindNearestEventEdge(Microseconds time, Microseconds threshold,
                                    Microseconds& best_diff,
                                    Microseconds& snapped_time,
                                    bool& snapped) const {
  const double px_per_time = px_per_time_unit();
  if (px_per_time <= 0) return;

  Pixel current_scroll_y = ImGui::GetScrollY();
  Pixel window_height = ImGui::GetWindowHeight();

  for (int group_index = 0; group_index < timeline_data_.groups.size();
       ++group_index) {
    if (group_index >= group_visible_.size() || !group_visible_[group_index]) {
      continue;
    }

    const Group& group = timeline_data_.groups[group_index];
    int next_group_start_level =
        GetNextGroupStartLevel(timeline_data_, group_index);

    const bool has_children =
        group_index + 1 < timeline_data_.groups.size() &&
        timeline_data_.groups[group_index + 1].nesting_level >
            group.nesting_level;
    const bool has_multiple_levels =
        next_group_start_level - group.start_level > 1;
    const bool expandable = group.type == Group::Type::kFlame &&
                            (has_children || has_multiple_levels);
    const bool is_collapsed = expandable && !group.expanded;

    if (is_collapsed) {
      continue;
    }

    for (int level = group.start_level; level < next_group_start_level;
         ++level) {
      if (level < 0 || level >= timeline_data_.events_by_level.size() ||
          level >= visible_level_offsets_.size()) {
        continue;
      }

      Pixel y_center = visible_level_offsets_[level];
      Pixel y_top = y_center - kEventHeight * 0.5f;
      Pixel y_bottom = y_center + kEventHeight * 0.5f;

      if (y_bottom < current_scroll_y ||
          y_top > current_scroll_y + window_height) {
        continue;
      }

      const auto& indices = timeline_data_.events_by_level[level];
      if (indices.empty()) continue;

      auto it = std::lower_bound(
          indices.begin(), indices.end(), time - threshold,
          [this](int event_index, Microseconds t) {
            return timeline_data_.entry_start_times[event_index] +
                       timeline_data_.entry_total_times[event_index] <
                   t;
          });

      for (; it != indices.end(); ++it) {
        const int event_index = *it;
        if (event_index < 0 ||
            event_index >= timeline_data_.entry_start_times.size() ||
            event_index >= timeline_data_.entry_total_times.size()) {
          continue;
        }

        const Microseconds start =
            timeline_data_.entry_start_times[event_index];
        if (start > time + threshold) {
          break;
        }

        const Microseconds duration =
            timeline_data_.entry_total_times[event_index];
        if (duration * px_per_time >= 2.0) {
          ApplySnappingToEdge(time, threshold, {start, start + duration},
                              best_diff, snapped_time, snapped);
        }
      }
    }
  }
}

double Timeline::px_per_time_unit() const {
  return px_per_time_unit(current_timeline_width_);
}

double Timeline::px_per_time_unit(Pixel timeline_width) const {
  const Microseconds view_duration = visible_range_->duration();
  if (view_duration > 0 && timeline_width > 0) {
    return static_cast<double>(timeline_width) / view_duration;
  } else {
    return 0.0;
  }
}

// Calculates the timing and pixel spacing for timeline ticks.
// This function determines a "nice" interval (e.g., 1, 2, 5, 10...) such that
// ticks are neither too crowded nor too sparse on screen.
Timeline::TickInfo Timeline::CalculateTickInfo(
    double px_per_time_unit_val) const {
  const Microseconds min_time_interval =
      kMinTickDistancePx / px_per_time_unit_val;
  const Microseconds tick_interval = CalculateNiceInterval(min_time_interval);
  const Pixel major_tick_dist_px = tick_interval * px_per_time_unit_val;

  const Microseconds view_start = visible_range().start();
  const Microseconds trace_start = data_time_range_.start();

  const Microseconds view_start_relative = view_start - trace_start;
  const Microseconds first_tick_time_relative =
      std::floor(view_start_relative / tick_interval) * tick_interval;

  return {tick_interval, major_tick_dist_px, first_tick_time_relative};
}

// Renders the ruler UI element at the top of the timeline.
// This is drawn as a table row and includes the background, the main horizontal
// line, major/minor tick marks, and time labels.
void Timeline::DrawRulerUI(const TickInfo& info, Pixel timeline_width) {
  const ImVec2 pos = ImGui::GetCursorScreenPos();
  const ImU32 ruler_text_color =
      palette_.GetColor(ColorPalette::Key::kRulerText)
          .value_or(kRulerTextColor);
  const ImU32 ruler_line_color =
      palette_.GetColor(ColorPalette::Key::kRulerLine)
          .value_or(kRulerLineColor);

  ImGui::SetCursorScreenPos(ImVec2(pos.x + label_width_, pos.y));

  ImDrawList* const draw_list = ImGui::GetWindowDrawList();

  const double px_per_time_unit_val = px_per_time_unit(timeline_width);
  if (px_per_time_unit_val > 0) {
    // Draw horizontal line
    const Pixel line_y = pos.y + kRulerHeight;
    draw_list->AddLine(
        ImVec2(pos.x + label_width_, line_y),
        ImVec2(pos.x + label_width_ + timeline_width + kTimelinePaddingRight,
               line_y),
        ruler_line_color);

    const Microseconds tick_interval = info.tick_interval;
    const Pixel major_tick_dist_px = info.major_tick_dist_px;
    const Microseconds first_tick_time_relative = info.first_tick_time_relative;
    const Microseconds trace_start = data_time_range_.start();

    const Pixel minor_tick_dist_px =
        major_tick_dist_px / static_cast<float>(kMinorTickDivisions);

    Microseconds t_relative = first_tick_time_relative;
    Pixel x = TimeToScreenX(t_relative + trace_start, pos.x + label_width_,
                            px_per_time_unit_val);

    for (;; t_relative += tick_interval, x += major_tick_dist_px) {
      if (x > pos.x + label_width_ + timeline_width + kRulerScreenBuffer) {
        break;
      }

      if (x >= pos.x + label_width_ - kRulerScreenBuffer) {
        // Draw major tick marks on the ruler.
        draw_list->AddLine(ImVec2(x, pos.y), ImVec2(x, line_y),
                           ruler_line_color);

        const std::string time_label_text = FormatTime(t_relative);
        ImGui::PushFont(fonts::label_small);
        draw_list->AddText(ImVec2(x + kRulerTextPadding, pos.y),
                           ruler_text_color, time_label_text.c_str());
        ImGui::PopFont();
      }

      // Draw minor ticks for the current interval.
      for (int i = 1; i < kMinorTickDivisions; ++i) {
        const Pixel minor_x = x + i * minor_tick_dist_px;
        if (minor_x >
            pos.x + label_width_ + timeline_width + kRulerScreenBuffer) {
          break;
        }
        if (minor_x >= pos.x + label_width_ - kRulerScreenBuffer) {
          draw_list->AddLine(ImVec2(minor_x, line_y - kRulerMinorTickHeight),
                             ImVec2(minor_x, line_y), ruler_line_color);
        }
      }
    }
  }
}

// Draws vertical grid lines that extend from the ruler down across all tracks.
// These lines are typically drawn behind the tracks in the background layer
// to provide a visual time reference without obscuring track content.
void Timeline::DrawVerticalGridLines(const TickInfo& info, Pixel timeline_width,
                                     Pixel viewport_bottom) {
  const ImVec2 pos = ImGui::GetCursorScreenPos();
  ImDrawList* const draw_list = ImGui::GetWindowDrawList();

  const double px_per_time_unit_val = px_per_time_unit(timeline_width);
  if (px_per_time_unit_val <= 0) return;

  const Pixel timeline_x_start = pos.x + label_width_;
  const Pixel line_y_top = pos.y;

  const Microseconds tick_interval = info.tick_interval;
  const Pixel major_tick_dist_px = info.major_tick_dist_px;
  const Microseconds first_tick_time_relative = info.first_tick_time_relative;

  const Microseconds trace_start = data_time_range_.start();

  Microseconds t_relative = first_tick_time_relative;
  Pixel x = TimeToScreenX(t_relative + trace_start, timeline_x_start,
                          px_per_time_unit_val);

  for (;; t_relative += tick_interval, x += major_tick_dist_px) {
    if (x > timeline_x_start + timeline_width + kRulerScreenBuffer) {
      break;
    }

    if (x >= timeline_x_start - kRulerScreenBuffer) {
      // Draw vertical line across the tracks.
      draw_list->AddLine(ImVec2(x, line_y_top), ImVec2(x, viewport_bottom),
                         palette_.GetColor(ColorPalette::Key::kMidtone)
                             .value_or(kTraceVerticalLineColor));
    }
  }
}

void Timeline::DrawEventName(absl::string_view event_name,
                             const EventRect& event_rect,
                             ImDrawList* absl_nonnull draw_list,
                             ImU32 text_color) const {
  // Cull text if the event overlaps with the ruler.
  if (event_rect.top < ruler_screen_y_ + kRulerHeight) {
    return;
  }

  const Pixel available_width = event_rect.right - event_rect.left;

  if (available_width >= kMinTextWidth) {
    const std::string text_display =
        GetTextForDisplay(event_name, available_width);

    if (!text_display.empty()) {
      const ImVec2 text_pos = CalculateEventTextRect(text_display, event_rect);

      // Push a clipping rectangle to ensure the text is only drawn within the
      // bounds of the event_rect. This prevents text from overflowing visually.
      draw_list->PushClipRect(ImVec2(event_rect.left, event_rect.top),
                              ImVec2(event_rect.right, event_rect.bottom));
      draw_list->AddText(text_pos, text_color, text_display.c_str());
      draw_list->PopClipRect();
    }
  }
}

void Timeline::DrawEvent(int group_index, int event_index,
                         const EventRect& rect,
                         ImDrawList* absl_nonnull draw_list) {
  // Only draw the rectangle if it has a positive width after clipping.
  // TODO: b/453676716 - Add ImGUI test for this function, including condition
  // rect.right > rect.left.
  if (rect.right > rect.left) {
    const std::string& event_name = timeline_data_.entry_names[event_index];

    const bool is_instant =
        timeline_data_.entry_total_times[event_index] <= 1e-06;
    bool is_hovered = false;
    // As the shapes for instant events are triangles, we need to check for
    // hover in a different way than regular events.
    if (is_instant) {
      // Even if the instant event is triangular, to solve the "fat finger"
      // problem, we still want to check for hover in a rectangle around the
      // triangle.
      is_hovered = ImGui::IsMouseHoveringRect(
          ImVec2(rect.left - kInstantEventChevronHalfWidth, rect.top),
          ImVec2(rect.left + kInstantEventChevronHalfWidth,
                 rect.top + kInstantEventChevronHeight));
    } else {
      is_hovered = ImGui::IsMouseHoveringRect(
          ImVec2(rect.left, rect.top), ImVec2(rect.right, rect.bottom));
    }

    const Pixel corner_rounding =
        is_hovered ? kHoverCornerRounding : kCornerRounding;

    ImU32 event_color = GetColorForId(event_name, palette_.GetTraceColors());

    bool matches_search = false;
    if (!search_query_lower_.empty()) {
      matches_search = matching_event_indices_.contains(event_index);
      if (!matches_search) {
        // Gray out non-matching events with muted grayscale and lower opacity
        const uint32_t r = (event_color >> IM_COL32_R_SHIFT) & 0xFF;
        const uint32_t g = (event_color >> IM_COL32_G_SHIFT) & 0xFF;
        const uint32_t b = (event_color >> IM_COL32_B_SHIFT) & 0xFF;
        const uint32_t luminance = (r * 299 + g * 587 + b * 114) / 1000;
        event_color = IM_COL32(luminance, luminance, luminance, 102);
      }
    }

    ImVec2 top, left_bottom, right_bottom;
    if (is_instant) {
      const Pixel centerX = rect.left;
      const Pixel chevron_half_width =
          is_hovered ? kInstantEventHoverChevronHalfWidth
                     : kInstantEventChevronHalfWidth;
      const Pixel chevron_height = is_hovered ? kInstantEventHoverChevronHeight
                                              : kInstantEventChevronHeight;

      top = ImVec2(centerX, rect.top);
      left_bottom = ImVec2(centerX - chevron_half_width,
                           rect.top + chevron_height);
      right_bottom = ImVec2(centerX + chevron_half_width,
                            rect.top + chevron_height);

      // Add transparency (0.6 opacity) to the event color of instant events
      // to help distinguish overlapping ones. Use full opacity when hovered.
      const ImU32 transparent_color =
          (event_color & ~IM_COL32_A_MASK) |
          (static_cast<ImU32>(0.6f * 255.0f) << IM_COL32_A_SHIFT);
      const ImU32 color =
          (is_hovered || matches_search) ? event_color : transparent_color;

      draw_list->AddTriangleFilled(top, left_bottom, right_bottom, color);
    } else {
      draw_list->AddRectFilled(ImVec2(rect.left, rect.top),
                               ImVec2(rect.right, rect.bottom), event_color,
                               corner_rounding, kImDrawFlags);
    }
    if (is_hovered) {
      hovered_event_index_ = event_index;
      // Draw a semi-transparent overlay when the event is hovered.
      if (is_instant) {
        draw_list->AddTriangleFilled(top, left_bottom, right_bottom,
                                     kHoverMaskColor);
        draw_list->AddTriangle(top, left_bottom, right_bottom, event_color,
                               2.0f);
      } else {
        draw_list->AddRectFilled(
            ImVec2(rect.left, rect.top), ImVec2(rect.right, rect.bottom),
            kHoverMaskColor, corner_rounding, kImDrawFlags);
      }

      ImGui::SetTooltip(
          "%s (%s)", event_name.c_str(),
          FormatTime(timeline_data_.entry_total_times[event_index]).c_str());

      // ImGui uses 0 to represent the left mouse button, as defined in the
      // ImGuiMouseButton enum. We check if the left mouse button was clicked.
      if (ImGui::IsMouseReleased(0)) {
        bool is_click = true;
        if (selection_start_pos_) {
          const float dx = ImGui::GetIO().MousePos.x - selection_start_pos_->x;
          const float dy = ImGui::GetIO().MousePos.y - selection_start_pos_->y;
          const float distance_squared = dx * dx + dy * dy;
          if (distance_squared > kClickDistanceThresholdSquared) {
            is_click = false;
          }
        }
        if (is_click) {
          event_clicked_this_frame_ = true;

          // If shift is held down, select/deselect the time range of the event.
          if (ImGui::GetIO().KeyShift) {
            const Microseconds start =
                timeline_data_.entry_start_times[event_index];
            const Microseconds end =
                start + timeline_data_.entry_total_times[event_index];
            TimeRange selected_time_range(start, end);
            auto it = absl::c_find(selected_time_ranges_, selected_time_range);
            // Click on the event to select, and click on the same event to
            // de-select.
            if (it != selected_time_ranges_.end()) {
              selected_time_ranges_.erase(it);
            } else {
              selected_time_ranges_.push_back(selected_time_range);
            }
          }

          if (selected_event_index_ != event_index) {
            selected_group_index_ = group_index;
            selected_event_index_ = event_index;
            // Deselect any selected counter event.
            selected_counter_index_ = -1;

            EmitEventSelected(event_index);
          }
        }  // close if (is_click)
      }  // close if (IsMouseReleased)
    }

    if (selected_event_index_ == event_index) {
      if (is_instant) {
        const Pixel centerX = rect.left;
        const Pixel chevron_half_width =
            is_hovered ? kInstantEventHoverChevronHalfWidth
                       : kInstantEventChevronHalfWidth;
        const Pixel chevron_height =
            is_hovered ? kInstantEventHoverChevronHeight
                       : kInstantEventChevronHeight;

        ImVec2 top(centerX, rect.top);
        ImVec2 left_bottom(centerX - chevron_half_width,
                           rect.top + chevron_height);
        ImVec2 right_bottom(centerX + chevron_half_width,
                            rect.top + chevron_height);

        draw_list->AddTriangle(top, left_bottom, right_bottom,
                               kSelectedBorderColor, kSelectedBorderThickness);
      } else {
        // Draw a border around the selected event.
        draw_list->AddRect(ImVec2(rect.left, rect.top),
                           ImVec2(rect.right, rect.bottom),
                           kSelectedBorderColor, corner_rounding, kImDrawFlags,
                           kSelectedBorderThickness);
      }
    }

    const ImU32 on_surface_color =
        palette_.GetColor(ColorPalette::Key::kOnSurface)
            .value_or(kOnSurfaceColor);
    const ImU32 inverse_on_surface_color =
        palette_.GetColor(ColorPalette::Key::kInverseOnSurface)
            .value_or(kInverseOnSurfaceColor);
    const ImU32 text_color = GetTextColorForContrast(
        event_color, on_surface_color, inverse_on_surface_color);
    DrawEventName(event_name, rect, draw_list, text_color);
  }
}

void Timeline::DrawEventsForLevel(int group_index,
                                  absl::Span<const int> event_indices,
                                  double px_per_time_unit, int level_in_group,
                                  const ImVec2& pos, const ImVec2& max,
                                  Pixel event_height, Pixel padding_bottom) {
  ImDrawList* const draw_list = ImGui::GetWindowDrawList();
  if (!draw_list) {
    return;
  }

  const Microseconds visible_start_time = PixelToTime(0, px_per_time_unit);
  const Microseconds visible_end_time = PixelToTime(max.x, px_per_time_unit);

  auto first_visible_it = std::lower_bound(
      event_indices.begin(), event_indices.end(), visible_start_time,
      [&](int event_index, Microseconds time) {
        // Defensive bounds check for event_index. While event_indices is
        // expected to contain valid indices, this prevents crashes if an
        // invalid index somehow gets included.
        if (event_index < 0 ||
            event_index >= timeline_data_.entry_start_times.size() ||
            event_index >= timeline_data_.entry_total_times.size()) {
          return false;  // Treat out-of-bounds as not ending before 'time'.
        }
        return timeline_data_.entry_start_times[event_index] +
                   timeline_data_.entry_total_times[event_index] <
               time;
      });

  auto last_visible_it = std::upper_bound(
      first_visible_it, event_indices.end(), visible_end_time,
      [&](Microseconds time, int event_index) {
        // Defensive bounds check for event_index.
        if (event_index < 0 ||
            event_index >= timeline_data_.entry_start_times.size()) {
          return false;  // Treat out-of-bounds as not starting after 'time'.
        }
        return time < timeline_data_.entry_start_times[event_index];
      });

  for (auto it = first_visible_it; it != last_visible_it; ++it) {
    int event_index = *it;
    if (event_index < 0 ||
        event_index >= timeline_data_.entry_start_times.size() ||
        event_index >= timeline_data_.entry_total_times.size()) {
      // Should not happen if data is well-formed, but good to be safe.
      continue;
    }
    const Microseconds start = timeline_data_.entry_start_times[event_index];
    const Microseconds end =
        start + timeline_data_.entry_total_times[event_index];

    const EventRect rect =
        CalculateEventRect(start, end, pos.x, pos.y, px_per_time_unit,
                           level_in_group, max.x, event_height, padding_bottom);

    DrawEvent(group_index, event_index, rect, draw_list);
  }
}

void Timeline::DrawCounterTooltip(int group_index, const CounterData& data,
                                  double px_per_time_unit_val,
                                  const ImVec2& pos, Pixel height,
                                  float y_ratio, ImDrawList* draw_list) {
  const ImVec2 mouse_pos = ImGui::GetMousePos();
  const double mouse_time =
      PixelToTime(mouse_pos.x - pos.x, px_per_time_unit_val);

  // Find the interval [t_i, t_{i+1}) containing mouse_time for sample-and-hold
  // (step) interpolation.
  // We use upper_bound to find the first timestamp strictly greater than
  // mouse_time. This ensures that std::prev(it) always points to t_i (the
  // start of the interval), even if mouse_time exactly equals t_i.
  // Using lower_bound would be incorrect for exact matches, as it would return
  // t_i, causing std::prev(it) to point to t_{i-1}.
  auto it = std::upper_bound(data.timestamps.begin(), data.timestamps.end(),
                             mouse_time);

  // Ensure we are not before the first timestamp.
  if (it != data.timestamps.begin()) {
    size_t index = std::distance(data.timestamps.begin(), std::prev(it));
    double val = data.values[index];

    const Pixel y_base = pos.y + height;

    if (index + 1 < data.timestamps.size()) {
      const double t1 = data.timestamps[index];
      const double t2 = data.timestamps[index + 1];
      const double v1 = data.values[index];

      // Sample-and-hold interpolation to match the bars drawn in
      // DrawCounterTrack.
      val = v1;

      // Highlight the entire bar under hover.
      const Pixel x1 = TimeToScreenX(t1, pos.x, px_per_time_unit_val);
      const Pixel x2 = TimeToScreenX(t2, pos.x, px_per_time_unit_val);
      const Pixel y = y_base - (v1 - data.min_value) * y_ratio;
      draw_list->AddRect(ImVec2(x1, y), ImVec2(x2, y_base), kCounterHoverColor,
                         0.0f, 0, kCounterHoverThickness);
    }

    const Pixel x = mouse_pos.x;
    const Pixel y = y_base - (val - data.min_value) * y_ratio;

    // Draw circle
    draw_list->AddCircleFilled(ImVec2(x, y), kPointRadius, kWhiteColor);
    draw_list->AddCircle(ImVec2(x, y), kPointRadius, kBlackColor);

    // Draw tooltip for current counter point's value and timestamp
    ImGui::SetTooltip(kCounterTooltipFormat, FormatTime(mouse_time).c_str(),
                      val);

    // ImGui uses 0 to represent the left mouse button, as defined in the
    // ImGuiMouseButton enum. We check if the left mouse button was clicked.
    if (ImGui::IsMouseReleased(0) && mouse_mode_ != MouseMode::kSelect) {
      bool is_click = true;
      if (selection_start_pos_) {
        const float dx = ImGui::GetIO().MousePos.x - selection_start_pos_->x;
        const float dy = ImGui::GetIO().MousePos.y - selection_start_pos_->y;
        const float distance_squared = dx * dx + dy * dy;
        if (distance_squared > kClickDistanceThresholdSquared) {
          is_click = false;
        }
      }
      if (is_click) {
        event_clicked_this_frame_ = true;
        if (selected_group_index_ != group_index ||
            selected_counter_index_ != index) {
          selected_group_index_ = group_index;
          selected_counter_index_ = index;
          // Deselect any selected flame event.
          selected_event_index_ = -1;

          // Emit an event to notify the application that a counter event was
          // selected.
          const std::string& name = timeline_data_.groups[group_index].name;
          EventData event_data;
          // We pass -1 for the event index to indicate that no flame event is
          // selected.
          event_data.try_emplace(kEventSelectedIndex, -1);
          event_data.try_emplace(kEventSelectedName, name);

          event_callback_(kEventSelected, event_data);
        }
      }
    }
  }
}

void Timeline::DrawCounterTrack(int group_index, const CounterData& data,
                                double px_per_time_unit_val, const ImVec2& pos,
                                Pixel height) {
  // At least two timestamps are required to draw a line segment.
  if (data.timestamps.size() < 2) return;

  ImDrawList* const draw_list = ImGui::GetWindowDrawList();

  if (!draw_list) return;
  const double value_range = data.max_value - data.min_value;

  // This should not happen with valid data where max_value >= min_value.
  if (value_range < 0) {
    LOG(ERROR) << "Invalid counter data: max_value " << data.max_value
               << " is less than min_value " << data.min_value;
    return;
  }

  const Pixel y_base = pos.y + height;

  // If all counter values are the same, draw a filled rectangle filling
  // the bottom half of the track to avoid division by zero.
  if (value_range == 0) {
    const Pixel y = pos.y + height / 2.0f;
    const Pixel x_start =
        TimeToScreenX(data.timestamps.front(), pos.x, px_per_time_unit_val);
    const Pixel x_end =
        TimeToScreenX(data.timestamps.back(), pos.x, px_per_time_unit_val);
    draw_list->AddRectFilled(ImVec2(x_start, y), ImVec2(x_end, y_base),
                             kCounterTrackColor);
    return;
  }

  const float y_ratio = height / value_range;

  for (size_t i = 0; i < data.timestamps.size() - 1; ++i) {
    Pixel x1 = TimeToScreenX(data.timestamps[i], pos.x, px_per_time_unit_val);
    Pixel x2 =
        TimeToScreenX(data.timestamps[i + 1], pos.x, px_per_time_unit_val);
    Pixel y = y_base - (data.values[i] - data.min_value) * y_ratio;

    // Add a minimum 1px height so that a value equal to min_value is still
    // visible as a thin line instead of completely disappearing.
    y = std::min(y, y_base - 1.0f);

    draw_list->AddRectFilled(ImVec2(x1, y), ImVec2(x2, y_base),
                             kCounterTrackColor);
  }

  // For the last point, draw a 1px wide bar to show its value.
  if (!data.timestamps.empty()) {
    Pixel x =
        TimeToScreenX(data.timestamps.back(), pos.x, px_per_time_unit_val);
    Pixel y = y_base - (data.values.back() - data.min_value) * y_ratio;
    draw_list->AddRectFilled(ImVec2(x, y), ImVec2(x + 1.0f, y_base),
                             kCounterTrackColor);
  }

  // Draw selected points from rectangle selection.
  auto it_pair = std::equal_range(
      selected_counter_points_.begin(), selected_counter_points_.end(),
      std::make_pair(group_index, 0),
      [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
        return a.first < b.first;
      });

  for (auto it = it_pair.first; it != it_pair.second; ++it) {
    size_t p_idx = it->second;
    if (p_idx < data.timestamps.size()) {
      if (selected_group_index_ == group_index &&
          selected_counter_index_ == p_idx) {
        continue;  // Handled by single selection below.
      }
      Microseconds ts = data.timestamps[p_idx];
      double val = data.values[p_idx];
      Pixel x = TimeToScreenX(ts, pos.x, px_per_time_unit_val);
      Pixel y = pos.y + height - (val - data.min_value) * y_ratio;

      draw_list->AddCircleFilled(ImVec2(x, y), kSelectedDataPointRadius,
                                 kBlue60);
      draw_list->AddCircle(ImVec2(x, y), kSelectedDataPointRadius, kWhiteColor,
                           /*num_segments=*/0,
                           /*thickness=*/kSelectedBorderThickness);
    }
  }

  if (selected_group_index_ == group_index && selected_counter_index_ != -1 &&
      selected_counter_index_ < data.timestamps.size()) {
    Microseconds ts = data.timestamps[selected_counter_index_];
    double val = data.values[selected_counter_index_];
    Pixel x = TimeToScreenX(ts, pos.x, px_per_time_unit_val);
    Pixel y = pos.y + height - (val - data.min_value) * y_ratio;

    draw_list->AddCircleFilled(ImVec2(x, y), kPointRadius, kWhiteColor);
    draw_list->AddCircle(ImVec2(x, y), kPointRadius, kSelectedBorderColor,
                         /*num_segments=*/0,
                         /*thickness=*/kSelectedBorderThickness);
  }

  if (ImGui::IsWindowHovered()) {
    DrawCounterTooltip(group_index, data, px_per_time_unit_val, pos, height,
                       y_ratio, draw_list);
  }
}

void Timeline::DrawGroup(int group_index, double px_per_time_unit_val,
                         Pixel scroll_y, Pixel window_height) {
  const Group& group = timeline_data_.groups[group_index];
  const int start_level = group.start_level;
  int end_level = GetNextGroupStartLevel(timeline_data_, group_index);
  if (group.type == Group::Type::kFlame && !group.expanded) {
    end_level = start_level;
  }
  // Ensure end_level is not less than start_level, to avoid negative height.
  end_level = std::max(start_level, end_level);

  // Calculate group height. Ensure a minimum height of one level to prevent
  // ImGui::BeginChild from auto-resizing, even if a group contains no levels.
  // This is important for parent groups (e.g., a process) that might not
  // contain any event levels directly.
  // TODO: b/453676716 - Add tests for group height calculation.
  const Pixel group_height =
      group.type == Group::Type::kCounter
          ? kCounterTrackHeight
          : (group.nesting_level == kProcessNestingLevel
                 ? kProcessTrackHeight
                 : std::max(1, end_level - start_level) *
                       (kEventHeight + kEventPaddingBottom));
  // Groups might have the same name. We add the index of the group to the ID
  // to ensure each ImGui::BeginChild call has a unique ID, otherwise ImGui
  // might ignore later calls with the same name.
  const std::string timeline_child_id =
      absl::StrCat("TimelineChild_", group.name, "_", group_index);

  const ImVec2 pos = ImGui::GetCursorScreenPos();

  if (ImGui::BeginChild(timeline_child_id.c_str(), ImVec2(0, group_height), 0,
                        kTrackFlags)) {
    const ImVec2 max = ImGui::GetContentRegionMax();

    if (group.type == Group::Type::kCounter) {
      const auto it =
          timeline_data_.counter_data_by_group_index.find(group_index);
      if (it != timeline_data_.counter_data_by_group_index.end()) {
        DrawCounterTrack(group_index, it->second, px_per_time_unit_val, pos,
                         group_height);
      }
    } else if (group.type == Group::Type::kFlame) {
      if (group.nesting_level == kProcessNestingLevel) {
        ImDrawList* const draw_list = ImGui::GetWindowDrawList();
        if (draw_list) {
          // Find the next group that is NOT a child of the current group to
          // determine the end level for the utilization chart.
          int proc_end_level = timeline_data_.events_by_level.size();
          for (size_t i = group_index + 1; i < timeline_data_.groups.size();
               ++i) {
            if (timeline_data_.groups[i].nesting_level <= group.nesting_level) {
              proc_end_level = timeline_data_.groups[i].start_level;
              break;
            }
          }
          DrawUtilizationAreaChart(start_level, proc_end_level,
                                   px_per_time_unit_val, pos, group_height,
                                   draw_list);
        }
      }
      const Pixel level_stride = kEventHeight + kEventPaddingBottom;
      const Pixel group_offset = group_offsets_[group_index];

      int first_visible_level = start_level;
      Pixel relative_scroll_y = scroll_y - group_offset;
      if (relative_scroll_y > 0) {
        first_visible_level =
            start_level + std::floor(relative_scroll_y / level_stride);
      }

      int last_visible_level = end_level;
      Pixel relative_scroll_end = scroll_y + window_height - group_offset;
      if (relative_scroll_end > 0) {
        last_visible_level =
            start_level + std::ceil(relative_scroll_end / level_stride);
      }

      first_visible_level = std::max(start_level, first_visible_level);
      last_visible_level = std::min(end_level, last_visible_level);

      for (int level = first_visible_level; level < last_visible_level;
           ++level) {
        // This is a sanity check to ensure the level is within the bounds of
        // events_by_level.
        if (level < timeline_data_.events_by_level.size()) {
          // TODO: b/453676716 - Add boundary test cases for this function.
          DrawEventsForLevel(group_index, timeline_data_.events_by_level[level],
                             px_per_time_unit_val,
                             /*level_in_group=*/level - start_level, pos, max,
                             kEventHeight, kEventPaddingBottom);
        }
      }
    }
  }
  ImGui::EndChild();
}

void Timeline::DrawGroupPreview(int group_index, double px_per_time_unit_val) {
  const Group& group = timeline_data_.groups[group_index];
  const std::string timeline_child_id =
      absl::StrCat("TimelineChildPreview_", group.name, "_", group_index);

  // Process tracks have a fixed height, other tracks use a single event height
  // for the preview.
  const Pixel group_height = group.nesting_level == kProcessNestingLevel
                                 ? kProcessTrackHeight
                                 : kEventHeight;

  // Calculate level Y positions for the preview.
  const ImVec2 pos = ImGui::GetCursorScreenPos();
  const int start_level = group.start_level;
  int end_level = timeline_data_.events_by_level.size();
  // Find the next group that is NOT a child of the current group.
  for (size_t i = group_index + 1; i < timeline_data_.groups.size(); ++i) {
    if (timeline_data_.groups[i].nesting_level <= group.nesting_level) {
      end_level = timeline_data_.groups[i].start_level;
      break;
    }
  }
  end_level = std::max(start_level, end_level);

  if (ImGui::BeginChild(timeline_child_id.c_str(), ImVec2(0, group_height), 0,
                        kTrackFlags)) {
    ImDrawList* draw_list = ImGui::GetWindowDrawList();

    if (group.type == Group::Type::kCounter) {
      const auto it =
          timeline_data_.counter_data_by_group_index.find(group_index);
      if (it != timeline_data_.counter_data_by_group_index.end()) {
        DrawCounterTrack(group_index, it->second, px_per_time_unit_val, pos,
                         group_height);
      }
    } else if (group.type == Group::Type::kFlame) {
      if (group.nesting_level == kProcessNestingLevel) {
        DrawUtilizationAreaChart(start_level, end_level, px_per_time_unit_val,
                                 pos, group_height, draw_list);
      } else {
        DrawFlameGroupPreview(start_level, end_level, px_per_time_unit_val, pos,
                              group_height, draw_list);
      }
    }
  }
  ImGui::EndChild();
}

void Timeline::DrawFlameGroupPreview(int start_level, int end_level,
                                     double px_per_time_unit_val,
                                     const ImVec2& pos, Pixel group_height,
                                     ImDrawList* draw_list) {
  // Aggregated view: Flatten all levels into one track with reduced
  // opacity.
  const Microseconds visible_start = visible_range().start();
  const Microseconds visible_end = visible_range().end();

  absl::string_view last_name;
  ImU32 last_color = 0;

  for (int level = start_level; level < end_level; ++level) {
    if (level >= timeline_data_.events_by_level.size()) continue;
    const auto& indices = timeline_data_.events_by_level[level];

    // Find the first event that ends after the visible start.
    // Since events in the same level are non-overlapping and sorted by
    // start time, they are effectively sorted by end time as well.
    auto it = std::lower_bound(
        indices.begin(), indices.end(), visible_start,
        [&](int event_idx, Microseconds t) {
          const Microseconds end = timeline_data_.entry_start_times[event_idx] +
                                   timeline_data_.entry_total_times[event_idx];
          return end <= t;
        });

    for (; it != indices.end(); ++it) {
      int event_index = *it;
      const Microseconds start = timeline_data_.entry_start_times[event_index];
      if (start >= visible_end) break;

      const Microseconds end =
          start + timeline_data_.entry_total_times[event_index];

      Pixel x_start = TimeToScreenX(start, pos.x, px_per_time_unit_val);
      Pixel x_end = TimeToScreenX(end, pos.x, px_per_time_unit_val);

      // Draw Logic
      const std::string& name = timeline_data_.entry_names[event_index];
      ImU32 color;
      if (name == last_name) {
        color = last_color;
      } else {
        last_name = name;
        color = GetColorForId(name, palette_.GetTraceColors());
        // Render with reduced opacity to show density.
        color = (color & ~IM_COL32_A_MASK) |
                (static_cast<ImU32>(kGroupPreviewOpacity * 255.0f)
                 << IM_COL32_A_SHIFT);
        last_color = color;
      }

      if (x_end < x_start) std::swap(x_start, x_end);
      x_end = std::max(x_end, x_start + kEventMinimumDrawWidth);

      draw_list->AddRectFilled(ImVec2(x_start, pos.y),
                               ImVec2(x_end, pos.y + group_height), color);
    }
  }
}

void Timeline::DrawUtilizationAreaChart(int start_level, int end_level,
                                        double px_per_time_unit_val,
                                        const ImVec2& pos, Pixel group_height,
                                        ImDrawList* draw_list) {
  const Microseconds visible_start = visible_range().start();
  const Microseconds visible_end = visible_range().end();
  const Pixel timeline_width = current_timeline_width_;
  if (timeline_width <= 0) return;

  const int num_bins = static_cast<int>(std::ceil(timeline_width));
  if (num_bins <= 0) return;

  if (utilization_bins_.size() < num_bins) utilization_bins_.resize(num_bins);
  std::fill(utilization_bins_.begin(), utilization_bins_.begin() + num_bins,
            0.0f);

  for (int level = start_level; level < end_level; ++level) {
    if (level >= timeline_data_.events_by_level.size()) continue;
    const auto& indices = timeline_data_.events_by_level[level];

    auto it = std::lower_bound(
        indices.begin(), indices.end(), visible_start,
        [&](int event_idx, Microseconds t) {
          const Microseconds end = timeline_data_.entry_start_times[event_idx] +
                                   timeline_data_.entry_total_times[event_idx];
          return end <= t;
        });

    for (; it != indices.end(); ++it) {
      int event_index = *it;
      const Microseconds start = timeline_data_.entry_start_times[event_index];
      if (start >= visible_end) break;
      const Microseconds end =
          start + timeline_data_.entry_total_times[event_index];

      // Calculate pixel coordinates relative to the start of the visible range.
      Pixel x_start = TimeToPixel(start, px_per_time_unit_val);
      Pixel x_end = TimeToPixel(end, px_per_time_unit_val);

      // Clip events that are partially outside the visible range.
      // Offset by 0.5 to center the bins on pixels? No, ImGui uses screen
      // coords.
      int bin_start = std::max(0, static_cast<int>(std::floor(x_start)));
      int bin_end =
          std::min(num_bins - 1, static_cast<int>(std::ceil(x_end - kEpsilon)));

      for (int i = bin_start; i <= bin_end; ++i) {
        Pixel overlap = std::min(x_end, static_cast<Pixel>(i + 1)) -
                        std::max(x_start, static_cast<float>(i));
        if (overlap > 0) {
          utilization_bins_[i] += overlap;
        }
      }
    }
  }

  Pixel max_util = 0.0f;
  for (Pixel val : utilization_bins_) {
    max_util = std::max(max_util, val);
  }

  // Normalize by at least one full track of activity.
  max_util = std::max(kMinUtilizationNormalization, max_util);

  // Draw each bin as a bar.
  for (size_t i = 0; i < utilization_bins_.size(); ++i) {
    if (utilization_bins_[i] > 0.0f) {
      Pixel h = (utilization_bins_[i] / max_util) * group_height;
      draw_list->AddRectFilled(
          ImVec2(pos.x + i, pos.y + group_height - h),
          ImVec2(pos.x + i + 1, pos.y + group_height),
          palette_.GetColor(ColorPalette::Key::kFlameHeader).value_or(kBlue70));
    }
  }

  // Draw tooltips when hovering over the chart.
  if (ImGui::IsWindowHovered()) {
    const ImVec2 mouse_pos = ImGui::GetMousePos();
    if (mouse_pos.x >= pos.x && mouse_pos.x < pos.x + timeline_width &&
        mouse_pos.y >= pos.y && mouse_pos.y < pos.y + group_height) {
      const int bin_idx = static_cast<int>(mouse_pos.x - pos.x);
      if (bin_idx >= 0 && bin_idx < num_bins) {
        float val = utilization_bins_[bin_idx];
        ImGui::SetTooltip(
            "Utilization: %.2f\n(Chart height represents event density)", val);
      }
    }
  }
}

void Timeline::DrawSingleFlow(const FlowLine& flow, Pixel timeline_x_start,
                              Pixel timeline_y_start, double px_per_time,
                              ImDrawList* draw_list) {
  if (flow.source_level >= visible_level_offsets_.size() ||
      flow.target_level >= visible_level_offsets_.size()) {
    return;
  }

  const Pixel start_y =
      timeline_y_start + visible_level_offsets_[flow.source_level];
  const Pixel end_y =
      timeline_y_start + visible_level_offsets_[flow.target_level];

  const Pixel start_x =
      TimeToScreenX(flow.source_ts, timeline_x_start, px_per_time);
  const Pixel end_x =
      TimeToScreenX(flow.target_ts, timeline_x_start, px_per_time);

  const ImVec2 p0(start_x, start_y);
  const ImVec2 p1(end_x, end_y);

  ImVec2 cp0, cp1;
  Timeline::CalculateBezierControlPoints(start_x, start_y, end_x, end_y, cp0,
                                         cp1);

  draw_list->AddBezierCubic(p0, cp0, cp1, p1, flow.color, 1.0f);
  draw_list->AddCircleFilled(p0, kPointRadius, flow.color);
  draw_list->AddCircleFilled(p1, kPointRadius, flow.color);
}

void Timeline::SetVisibleFlowCategories(const std::vector<int>& category_ids) {
  visible_flow_categories_.clear();
  visible_flow_categories_.insert(category_ids.begin(), category_ids.end());
  if (redraw_callback_) redraw_callback_();
}

void Timeline::DrawFlows(Pixel timeline_width, Pixel timeline_y_start) {
  const bool has_selected_event =
      selected_event_index_ != -1 &&
      selected_event_index_ < timeline_data_.entry_event_ids.size();

  if ((visible_flow_categories_.empty() && !has_selected_event) ||
      timeline_data_.flow_lines.empty()) {
    return;
  }

  const ImVec2 table_rect_min = ImGui::GetItemRectMin();
  const Pixel timeline_x_start = table_rect_min.x + label_width_;

  // Use ForegroundDrawList to draw on top of opaque child windows (groups).
  ImDrawList* const draw_list = ImGui::GetForegroundDrawList();

  // Clip to the current window ("Tracks") to avoid drawing outside the timeline
  // area.
  const ImVec2 window_pos = ImGui::GetWindowPos();
  const ImVec2 clip_min = ImVec2(window_pos.x + label_width_, window_pos.y);
  const ImVec2 clip_max = ImVec2(window_pos.x + ImGui::GetWindowWidth(),
                                 window_pos.y + ImGui::GetWindowHeight());
  draw_list->PushClipRect(clip_min, clip_max,
                          /*intersect_with_current_clip_rect=*/true);

  const double px_per_time = px_per_time_unit(timeline_width);
  // If px_per_time is non-positive, it indicates an invalid or zero-width
  // timeline or view duration, so we cannot draw flows.
  if (px_per_time <= 0) {
    draw_list->PopClipRect();
    return;
  }

  if (has_selected_event) {
    EventId selected_event_id =
        timeline_data_.entry_event_ids[selected_event_index_];
    auto it_ids = timeline_data_.flow_ids_by_event_id.find(selected_event_id);
    if (it_ids != timeline_data_.flow_ids_by_event_id.end()) {
      for (const std::string& flow_id : it_ids->second) {
        auto it_lines = timeline_data_.flow_lines_by_flow_id.find(flow_id);
        if (it_lines != timeline_data_.flow_lines_by_flow_id.end()) {
          for (const auto& flow : it_lines->second) {
            DrawSingleFlow(flow, timeline_x_start, timeline_y_start,
                           px_per_time, draw_list);
          }
        }
      }
    }
  } else {
    for (const auto& flow : timeline_data_.flow_lines) {
      if (visible_flow_categories_.contains(static_cast<int>(flow.category))) {
        DrawSingleFlow(flow, timeline_x_start, timeline_y_start, px_per_time,
                       draw_list);
      }
    }
  }

  draw_list->PopClipRect();
}

void Timeline::DrawSelectedTimeRange(const TimeRange& range,
                                     Pixel timeline_width,
                                     double px_per_time_unit_val,
                                     bool show_delete_button) {
  const ImGuiViewport* viewport = ImGui::GetMainViewport();
  const Pixel timeline_x_start = viewport->Pos.x + label_width_;
  const ImU32 color = palette_.GetColor(ColorPalette::Key::kSelection)
                          .value_or(kSelectedTimeRangeColor);

  const Pixel time_range_x_start =
      TimeToScreenX(range.start(), timeline_x_start, px_per_time_unit_val);
  const Pixel time_range_x_end =
      TimeToScreenX(range.end(), timeline_x_start, px_per_time_unit_val) -
      kEventPaddingRight;
  // Clip the selection rectangle to the visible timeline bounds.
  // If the selection starts before the timeline's visible area,
  // clipped_x_start ensures we only start drawing from timeline_x_start.
  const Pixel clipped_x_start = std::max(time_range_x_start, timeline_x_start);
  // If the selection ends after the timeline's visible area, clipped_x_end
  // ensures we stop drawing at the right edge of the timeline.
  const Pixel clipped_x_end =
      std::min(time_range_x_end, timeline_x_start + timeline_width);

  if (clipped_x_end > clipped_x_start) {
    // Use the window draw list to render over all other timeline content.
    ImDrawList* const draw_list = ImGui::GetWindowDrawList();

    const Pixel rect_y_min = ruler_screen_y_ + kRulerHeight;
    const Pixel rect_y_max = viewport->Pos.y + viewport->Size.y;

    draw_list->AddRectFilledMultiColor(ImVec2(clipped_x_start, rect_y_min),
                                       ImVec2(clipped_x_end, rect_y_max),
                                       (color & 0x00FFFFFF) | (0x99 << 24),
                                       (color & 0x00FFFFFF) | (0x99 << 24),
                                       (color & 0x00FFFFFF) | (0x1A << 24),
                                       (color & 0x00FFFFFF) | (0x1A << 24));

    // Only draw the border if the edge of the time range is visible.
    if (time_range_x_start >= timeline_x_start) {
      draw_list->AddLine(ImVec2(time_range_x_start, rect_y_min),
                         ImVec2(time_range_x_start, rect_y_max), color);
      // Check for hovering on the start edge.
      if (!is_dragging_ &&
          std::abs(ImGui::GetMousePos().x - time_range_x_start) <=
              kSelectionEdgeThreshold &&
          ImGui::IsMouseHoveringRect(
              ImVec2(time_range_x_start - kSelectionEdgeThreshold, rect_y_min),
              ImVec2(time_range_x_start + kSelectionEdgeThreshold,
                     rect_y_max))) {
        ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
      }
    }
    if (time_range_x_end <= timeline_x_start + timeline_width) {
      draw_list->AddLine(ImVec2(time_range_x_end, rect_y_min),
                         ImVec2(time_range_x_end, rect_y_max), color);
      // Check for hovering on the end edge.
      if (!is_dragging_ &&
          std::abs(ImGui::GetMousePos().x - time_range_x_end) <=
              kSelectionEdgeThreshold &&
          ImGui::IsMouseHoveringRect(
              ImVec2(time_range_x_end - kSelectionEdgeThreshold, rect_y_min),
              ImVec2(time_range_x_end + kSelectionEdgeThreshold, rect_y_max))) {
        ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
      }
    }

    const std::string text = FormatTime(range.duration());
    const ImVec2 text_size = ImGui::CalcTextSize(text.c_str());
    const ImU32 kTextColor =
        palette_.GetColor(ColorPalette::Key::kForeground).value_or(kBlackColor);

    // Move the text to the top, below the ruler.
    const Pixel text_y = rect_y_min + kSelectedTimeRangeTextTopPadding;
    const Pixel text_x = clipped_x_start +
                         (clipped_x_end - clipped_x_start - text_size.x) / 2.0f;
    const ImVec2 text_pos(text_x, text_y);

    const ImRect visible_range_rect(ImVec2(clipped_x_start, rect_y_min),
                                    ImVec2(clipped_x_end, rect_y_max));
    const ImRect full_range_rect(ImVec2(time_range_x_start, rect_y_min),
                                 ImVec2(time_range_x_end, rect_y_max));

    if (show_delete_button) {
      const DeleteButtonLayout layout = GetDeleteButtonLayout(
          text_size, text_pos, visible_range_rect, full_range_rect);

      if (layout.text_fits) {
        draw_list->AddText(text_pos, kTextColor, text.c_str());
      }

      DrawDeleteButton(draw_list, layout.button_pos, layout.hover_rect, range);
    } else {
      bool text_fits = visible_range_rect.GetWidth() > text_size.x;
      if (text_fits) {
        draw_list->AddText(text_pos, kTextColor, text.c_str());
      }
    }
  }
}

DeleteButtonLayout Timeline::GetDeleteButtonLayout(
    const ImVec2& text_size, const ImVec2& text_pos,
    const ImRect& visible_range_rect, const ImRect& full_range_rect) const {
  ImVec2 button_pos;
  ImRect hover_rect;
  bool text_fits = false;

  // Check if text fits in the visible range.
  if (visible_range_rect.GetWidth() > text_size.x) {
    text_fits = true;
    // Position the button to the right of the text.
    button_pos = ImVec2(text_pos.x + text_size.x + kCloseButtonPadding,
                        text_pos.y + (text_size.y - kCloseButtonSize) / 2.0f);

    // Expand the hover area to include both the text and the button, with a
    // small margin.
    ImVec2 hover_min(std::min(text_pos.x, button_pos.x),
                     std::min(text_pos.y, button_pos.y));
    ImVec2 hover_max(
        std::max(text_pos.x + text_size.x, button_pos.x + kCloseButtonSize),
        std::max(text_pos.y + text_size.y, button_pos.y + kCloseButtonSize));

    hover_min.x -= kHoverPadding;
    hover_min.y -= kHoverPadding;
    hover_max.x += kHoverPadding;
    hover_max.y += kHoverPadding;
    hover_rect = ImRect(hover_min, hover_max);
  } else {
    // If text doesn't fit, center the button in the visible range.
    const Pixel center_x = visible_range_rect.GetCenter().x;
    button_pos = ImVec2(center_x - kCloseButtonSize / 2.0f,
                        text_pos.y + (text_size.y - kCloseButtonSize) / 2.0f);

    // Set the hover area to the visible selected time range, expanded to
    // include the button.
    hover_rect = visible_range_rect;
    const ImRect button_rect(button_pos,
                             ImVec2(button_pos.x + kCloseButtonSize,
                                    button_pos.y + kCloseButtonSize));
    hover_rect.Add(button_rect);
  }

  return {button_pos, hover_rect, text_fits};
}

void Timeline::DrawDeleteButton(ImDrawList* draw_list, const ImVec2& button_pos,
                                const ImRect& hover_rect,
                                const TimeRange& range) {
  if (DrawCloseButton(draw_list, button_pos, hover_rect)) {
    auto it = absl::c_find(selected_time_ranges_, range);
    if (it != selected_time_ranges_.end()) {
      selected_time_ranges_.erase(it);
    }
    if (current_selected_time_range_ &&
        *current_selected_time_range_ == range) {
      current_selected_time_range_.reset();
    }
  }
}

bool Timeline::DrawCloseButton(ImDrawList* draw_list, const ImVec2& button_pos,
                               const ImRect& hover_rect) {
  const Pixel button_size = kCloseButtonSize;
  const ImVec2 button_min = button_pos;
  const ImVec2 button_max(button_pos.x + button_size,
                          button_pos.y + button_size);

  bool clicked = false;
  if (ImGui::IsMouseHoveringRect(hover_rect.Min, hover_rect.Max)) {
    ImU32 button_color = kCloseButtonColor;
    if (ImGui::IsMouseHoveringRect(button_min, button_max)) {
      button_color = kCloseButtonHoverColor;
      ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
      if (ImGui::IsMouseClicked(0)) {
        clicked = true;
      }
    }

    const ImVec2 center(button_min.x + button_size / 2.0f,
                        button_min.y + button_size / 2.0f);
    draw_list->AddCircleFilled(center, button_size / 2.0f, button_color);

    const Pixel x_radius = button_size * 0.25f;
    draw_list->AddLine(ImVec2(center.x - x_radius, center.y - x_radius),
                       ImVec2(center.x + x_radius, center.y + x_radius),
                       kWhiteColor);
    draw_list->AddLine(ImVec2(center.x - x_radius, center.y + x_radius),
                       ImVec2(center.x + x_radius, center.y - x_radius),
                       kWhiteColor);
  }
  return clicked;
}

void Timeline::DrawHideButton(int group_index,
                                Pixel height, bool is_track_hidden) {
  const Group& group = timeline_data_.groups[group_index];

  // Base size to determine the icon's drawing area and the button's width.
  // The button size (especially width) directly relates to the icon draw size
  // so that the hit target matches the visual boundary of the hide icon.
  const Pixel kIconDrawSize = ImGui::GetFontSize() * kIconSizeScale;
  const Pixel kButtonVisibleHeight = height;

  ImVec2 p = ImGui::GetCursorScreenPos();
  // The icon will be drawn within a kIconDrawSize * kIconDrawSize area.
  // The button width matches the icon draw width.
  const ImVec2 buttonSize(kIconDrawSize, kButtonVisibleHeight);

  if (ImGui::InvisibleButton("##hide", buttonSize)) {
    auto it = hidden_track_names_.find(group.name);
    if (it != hidden_track_names_.end()) {
      hidden_track_names_.erase(it);
    } else {
      hidden_track_names_.insert(group.name);
    }
    UpdateLevelPositions(timeline_data_);
    if (redraw_callback_) redraw_callback_();
  }

  // Calculate the center point for drawing the icon.
  Pixel center_x = p.x + kIconDrawSize * 0.5f;
  Pixel center_y = p.y + kButtonVisibleHeight * 0.5f;

  ImDrawList* draw_list = ImGui::GetWindowDrawList();
  ImU32 icon_col = ImGui::GetColorU32(ImGuiCol_Text);
  if (ImGui::IsItemHovered()) {
    ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
    icon_col = ImGui::GetColorU32(ImGuiCol_ButtonHovered);
    ImGui::SetTooltip(is_track_hidden ? kUnhideTrackTooltip
                                      : kHideTrackTooltip);
  }

  const Pixel content_region_avail_width =
      ImGui::GetWindowWidth() - ImGui::GetStyle().ScrollbarSize;
  const bool is_row_hovered = ImGui::IsMouseHoveringRect(
      ImVec2(tracks_start_screen_pos_.x,
             tracks_start_screen_pos_.y + group_offsets_[group_index]),
      ImVec2(tracks_start_screen_pos_.x + content_region_avail_width,
             tracks_start_screen_pos_.y + group_offsets_[group_index + 1]));

  if (is_row_hovered) {
    DrawHideIcon(draw_list, center_x, center_y, kIconDrawSize, icon_col,
                 is_track_hidden);
  }
}

// Draws the hide/unhide icon in a kIconDrawSize * kIconDrawSize square area.
// The icon's drawing coordinates are relative to the center and will not
// exceed the boundaries defined by kIconDrawSize.
void Timeline::DrawHideIcon(ImDrawList* draw_list, Pixel center_x,
                            Pixel center_y, Pixel kIconDrawSize, ImU32 icon_col,
                            bool is_track_hidden) {
  float r = kIconDrawSize * 0.5f;

  // Curve approximation using segments
  ImVec2 p0(center_x - r, center_y);
  ImVec2 p1(center_x - r * 0.5f, center_y - r * 0.45f);
  ImVec2 p2(center_x, center_y - r * 0.6f);
  ImVec2 p3(center_x + r * 0.5f, center_y - r * 0.45f);
  ImVec2 p4(center_x + r, center_y);

  // Top eye curve
  draw_list->AddLine(p0, p1, icon_col, 1.0f);
  draw_list->AddLine(p1, p2, icon_col, 1.0f);
  draw_list->AddLine(p2, p3, icon_col, 1.0f);
  draw_list->AddLine(p3, p4, icon_col, 1.0f);

  // Bottom eye curve
  ImVec2 p5(center_x - r * 0.5f, center_y + r * 0.45f);
  ImVec2 p6(center_x, center_y + r * 0.6f);
  ImVec2 p7(center_x + r * 0.5f, center_y + r * 0.45f);

  draw_list->AddLine(p0, p5, icon_col, 1.0f);
  draw_list->AddLine(p5, p6, icon_col, 1.0f);
  draw_list->AddLine(p6, p7, icon_col, 1.0f);
  draw_list->AddLine(p7, p4, icon_col, 1.0f);

  // Pupil (center)
  draw_list->AddCircleFilled(ImVec2(center_x, center_y), r * 0.25f, icon_col);

  // Slashed line for crossed eye
  if (is_track_hidden) {
    draw_list->AddLine(ImVec2(center_x - r * 0.9f, center_y - r * 0.6f),
                       ImVec2(center_x + r * 0.9f, center_y + r * 0.6f),
                       icon_col, 1.0f);
  }
}

void Timeline::DrawSelectedTimeRanges(Pixel timeline_width,
                                      double px_per_time_unit_val) {
  // We make a copy of the selected time ranges because DrawSelectedTimeRange
  // might modify `selected_time_ranges_` (e.g. by deleting a range), which
  // would invalidate iterators if we were iterating over the original vector.
  // We want to iterate over the ranges as they were at the start of the frame.
  // Note that `selected_time_ranges_` contains only user-created selections
  // (typically very few), not the trace events, so this copy is negligible in
  // terms of memory and performance.
  std::vector<TimeRange> ranges_to_draw = selected_time_ranges_;
  for (const auto& range : ranges_to_draw) {
    DrawSelectedTimeRange(range, timeline_width, px_per_time_unit_val);
  }

  if (current_selected_time_range_) {
    DrawSelectedTimeRange(*current_selected_time_range_, timeline_width,
                          px_per_time_unit_val, /*show_delete_button=*/false);
  }
}

bool Timeline::HandleKeyboard() {
  const ImGuiIO& io = ImGui::GetIO();
  bool is_interacting = false;

  // Pan left
  if (ImGui::IsKeyDown(ImGuiKey_A)) {
    float multiplier = GetSpeedMultiplier(io, ImGuiKey_A);
    Pan(-kPanningSpeed * io.DeltaTime * multiplier);
    is_interacting = true;
  }
  // Pan right
  if (ImGui::IsKeyDown(ImGuiKey_D)) {
    float multiplier = GetSpeedMultiplier(io, ImGuiKey_D);
    Pan(kPanningSpeed * io.DeltaTime * multiplier);
    is_interacting = true;
  }

  // Unlike panning and zooming, vertical scrolling does not affect the visible
  // time range, so it doesn't need to pause data requests by setting
  // is_interacting = true.
  // Scroll up
  if (ImGui::IsKeyDown(ImGuiKey_UpArrow)) {
    Scroll(-kScrollSpeed * io.DeltaTime);
  }
  // Scroll down
  if (ImGui::IsKeyDown(ImGuiKey_DownArrow)) {
    Scroll(kScrollSpeed * io.DeltaTime);
  }

  // Zoom in
  if (ImGui::IsKeyDown(ImGuiKey_W)) {
    float multiplier = GetSpeedMultiplier(io, ImGuiKey_W);
    Zoom(1.0f - kZoomSpeed * io.DeltaTime * multiplier);
    is_interacting = true;
  }
  // Zoom out
  if (ImGui::IsKeyDown(ImGuiKey_S)) {
    float multiplier = GetSpeedMultiplier(io, ImGuiKey_S);
    Zoom(1.0f + kZoomSpeed * io.DeltaTime * multiplier);
    is_interacting = true;
  }

  // Cancel selection or resize
  if (ImGui::IsKeyPressed(ImGuiKey_Escape)) {
    if (is_selecting_ || time_range_resizing_state_.has_value()) {
      if (is_selecting_) {
        is_selecting_ = false;
      }
      if (time_range_resizing_state_.has_value()) {
        time_range_resizing_state_.reset();
      }

      // Common cancel actions
      is_dragging_ = false;
      current_selected_time_range_.reset();
    }
  }

  // Mouse Mode shortcuts
  if (ImGui::IsKeyPressed(ImGuiKey_1)) {
    mouse_mode_ = MouseMode::kSelect;
    EmitMouseModeChanged();
  }
  if (ImGui::IsKeyPressed(ImGuiKey_2)) {
    mouse_mode_ = MouseMode::kPan;
    EmitMouseModeChanged();
  }
  if (ImGui::IsKeyPressed(ImGuiKey_3)) {
    mouse_mode_ = MouseMode::kZoom;
    EmitMouseModeChanged();
  }
  if (ImGui::IsKeyPressed(ImGuiKey_4)) {
    mouse_mode_ = MouseMode::kTiming;
    EmitMouseModeChanged();
  }

  return is_interacting;
}

bool Timeline::HandleWheel() {
  const ImGuiIO& io = ImGui::GetIO();

  if (io.MouseWheel == 0.0f && io.MouseWheelH == 0.0f) {
    return false;
  }

  if (io.KeyCtrl || io.KeySuper) {
    // If the mouse wheel is being used with the control or command key, zoom
    // in or out.
    const float zoom_factor = 1.0f + io.MouseWheel * kMouseWheelZoomSpeed;
    Zoom(zoom_factor);
    return true;
  }

  const Pixel horizontal_pan_delta =
      io.KeyShift ? io.MouseWheel : io.MouseWheelH;
  const Pixel vertical_scroll_delta =
      io.KeyShift ? io.MouseWheelH : io.MouseWheel;

  if (horizontal_pan_delta != 0.0f) Pan(horizontal_pan_delta);
  if (vertical_scroll_delta != 0.0f) Scroll(vertical_scroll_delta);

  return true;
}

// Checks if there is a pending request to vertically scroll to a specific
// event, and sets next window scroll to make it visible if it's out of view.
void Timeline::ProcessPendingScroll() {
  // Check if there is a pending request to scroll to an event.
  if (event_index_to_scroll_to_ < 0) return;

  int level = timeline_data_.entry_levels[event_index_to_scroll_to_];
  if (level < 0 || level >= visible_level_offsets_.size()) return;

  Pixel y_center = visible_level_offsets_[level];
  Pixel y_top = y_center - kEventHeight * 0.5f;
  Pixel y_bottom = y_center + kEventHeight * 0.5f;
  Pixel window_height = ImGui::GetWindowHeight();
  Pixel current_scroll_y = ImGui::GetScrollY();

  // Default target scroll position is current scroll position.
  Pixel target_scroll_y = current_scroll_y;

  // Check if the event is already fully visible.
  bool is_fully_visible = (y_top >= current_scroll_y) &&
                          (y_bottom <= current_scroll_y + window_height);

  if (is_fully_visible) {
    // Event is fully visible, no need to scroll.
  } else if (y_top < current_scroll_y) {
    // Case A: Event is above the current viewport, scroll up until top is
    // visible.
    target_scroll_y = y_top;
  } else if (y_bottom > current_scroll_y + window_height) {
    // Case B: Event is below the current viewport, scroll down until bottom
    // is visible.
    target_scroll_y = y_bottom - window_height;
  }

  // Ensure scroll value is not negative.
  target_scroll_y = std::max(0.0f, target_scroll_y);

  // Trigger scroll if target position changed.
  if (target_scroll_y != current_scroll_y) {
    ImGui::SetScrollY(target_scroll_y);
    // Request a redraw to ensure the UI updates immediately after scrolling.
    if (redraw_callback_) redraw_callback_();
  }

  // Reset request flag to prevent repeated scrolling.
  event_index_to_scroll_to_ = -1;
}

void Timeline::HandleEventDeselection() {
  const bool has_single_selection =
      (selected_event_index_ != -1 || selected_group_index_ != -1);
  const bool has_rectangle_selection =
      (!selected_event_indices_.empty() || !selected_counter_points_.empty());

  // If an event was selected, and the user clicks on an empty area
  // (i.e., not on any event), deselect the event.
  if ((has_single_selection || has_rectangle_selection) &&
      ImGui::IsMouseReleased(0) &&
      ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows) &&
      !event_clicked_this_frame_) {
    bool is_click = true;
    if (selection_start_pos_) {
      const float dx = ImGui::GetIO().MousePos.x - selection_start_pos_->x;
      const float dy = ImGui::GetIO().MousePos.y - selection_start_pos_->y;
      const float distance_squared = dx * dx + dy * dy;
      if (distance_squared > kClickDistanceThresholdSquared) {
        is_click = false;
      }
    }
    if (is_click) {
      if (has_single_selection) {
        selected_event_index_ = -1;
        selected_group_index_ = -1;
        selected_counter_index_ = -1;

        EventData event_data;
        event_data.try_emplace(kEventSelectedIndex, -1);
        event_data.try_emplace(kEventSelectedName, std::string());
        event_data.try_emplace(kEventSelectedStart, 0.0);
        event_data.try_emplace(kEventSelectedDuration, 0.0);
        event_data.try_emplace(kEventSelectedStartFormatted, std::string());
        event_data.try_emplace(kEventSelectedDurationFormatted, std::string());

        event_callback_(kEventSelected, event_data);
      }

      if (has_rectangle_selection) {
        selected_event_indices_.clear();
        selected_counter_points_.clear();
        CalculateAndEmitMetrics();
      }

      if (redraw_callback_) redraw_callback_();
    }
  }
}

bool Timeline::HandleMouse() {
  const ImRect timeline_area = GetTimelineArea();
  const bool is_mouse_over_timeline =
      ImGui::IsMouseHoveringRect(timeline_area.Min, timeline_area.Max);

  if (time_range_resizing_state_.has_value()) {
    ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
  } else if (is_mouse_over_timeline) {
    switch (mouse_mode_) {
      case MouseMode::kSelect:
        ImGui::SetMouseCursor(ImGuiMouseCursor_Arrow);
        break;
      case MouseMode::kTiming:
        ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
        break;
      case MouseMode::kPan:
        if (ImGui::IsMouseDown(0)) {
          ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeAll);
        } else {
          ImGui::SetMouseCursor(ImGuiMouseCursor_Arrow);
        }
        break;
      case MouseMode::kZoom:
        ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);
        break;
      default:
        break;
    }
  }

  if (!is_mouse_over_timeline && !is_dragging_) {
    return false;
  }

  if (is_mouse_over_timeline) {
    HandleMouseDown(timeline_area.Min.x);
  }

  if (is_dragging_) {
    HandleMouseDrag(timeline_area.Min.x);
    HandleMouseRelease();
    return true;
  }

  return false;
}

void Timeline::HandleMouseDown(Pixel timeline_origin_x) {
  // ImGui uses 0 to represent the left mouse button, as defined in the
  // ImGuiMouseButton enum. We check if the left mouse button was clicked.
  if (ImGui::IsMouseClicked(0) && !event_clicked_this_frame_) {
    is_dragging_ = true;
    ImGuiIO& io = ImGui::GetIO();
    selection_start_pos_ = io.MousePos;

    // Check if we are starting a resize operation on an existing range.
    const double px_per_time = px_per_time_unit();
    if (px_per_time > 0) {
      for (size_t i = 0; i < selected_time_ranges_.size(); ++i) {
        const auto& range = selected_time_ranges_[i];
        const Pixel x_start =
            TimeToScreenX(range.start(), timeline_origin_x, px_per_time);
        const Pixel x_end =
            TimeToScreenX(range.end(), timeline_origin_x, px_per_time);

        if (std::abs(io.MousePos.x - x_start) <= kSelectionEdgeThreshold) {
          time_range_resizing_state_ = {i, /*is_start_edge=*/true};
          return;
        } else if (std::abs(io.MousePos.x - x_end) <= kSelectionEdgeThreshold) {
          time_range_resizing_state_ = {i, /*is_start_edge=*/false};
          return;
        }
      }
    }

    if (mouse_mode_ == MouseMode::kSelect && !is_selecting_) {
      is_selecting_ = true;
      selection_end_pos_ = io.MousePos;
      selected_event_indices_.clear();
      selected_counter_points_.clear();
    } else if (mouse_mode_ != MouseMode::kSelect) {
      is_selecting_ = io.KeyShift || mouse_mode_ == MouseMode::kTiming;
      if (is_selecting_) {
        const double px_per_time = px_per_time_unit();
        drag_start_time_ =
            PixelToTime(io.MousePos.x - timeline_origin_x, px_per_time);
        current_selected_time_range_ =
            TimeRange(drag_start_time_, drag_start_time_);
      }
    }
  }
}

void Timeline::HandleMouseDrag(Pixel timeline_origin_x) {
  // ImGui uses 0 to represent the left mouse button, as defined in the
  // ImGuiMouseButton enum. We check if the left mouse button was clicked.
  if (ImGui::IsMouseDown(0)) {
    ImGuiIO& io = ImGui::GetIO();
    if (time_range_resizing_state_.has_value() &&
        time_range_resizing_state_->range_index <
            selected_time_ranges_.size()) {
      const double px_per_time = px_per_time_unit();
      Microseconds current_time =
          PixelToTime(io.MousePos.x - timeline_origin_x, px_per_time);

      auto& range =
          selected_time_ranges_[time_range_resizing_state_->range_index];
      Microseconds start = range.start();
      Microseconds end = range.end();

      if (time_range_resizing_state_->is_start_edge) {
        start = current_time;
      } else {
        end = current_time;
      }

      TimeRange new_range(std::min(start, end), std::max(start, end));
      ApplySnapping(new_range);
      range = new_range;
    } else if (is_selecting_) {
      if (mouse_mode_ == MouseMode::kSelect) {
        selection_end_pos_ = io.MousePos;
      } else {
        const double px_per_time = px_per_time_unit();
        Microseconds current_time =
            PixelToTime(io.MousePos.x - timeline_origin_x, px_per_time);
        current_selected_time_range_ =
            TimeRange(std::min(drag_start_time_, current_time),
                      std::max(drag_start_time_, current_time));
        ApplySnapping(*current_selected_time_range_);
      }
    } else if (mouse_mode_ == MouseMode::kZoom) {
      Zoom(1.0f + io.MouseDelta.y * 0.01f);
    } else {
      Pan(-io.MouseDelta.x);
      Scroll(-io.MouseDelta.y);
    }
  }
}

void Timeline::DrawToast(absl::string_view message, float& timer,
                         float base_y_offset) {
  if (timer <= 0.0f) return;

  timer -= ImGui::GetIO().DeltaTime;
  if (redraw_callback_) redraw_callback_();

  const ImGuiViewport* main_viewport = ImGui::GetMainViewport();
  ImDrawList* draw_list = ImGui::GetForegroundDrawList();

  const ImVec2 text_size =
      ImGui::CalcTextSize(message.data(), message.data() + message.size());
  const ImVec2 padding(16.0f, 8.0f);
  const ImVec2 toast_size(text_size.x + padding.x * 2.0f,
                          text_size.y + padding.y * 2.0f);

  const ImVec2 toast_pos(
      main_viewport->Pos.x + (main_viewport->Size.x - toast_size.x) * 0.5f,
      main_viewport->Pos.y + main_viewport->Size.y - toast_size.y -
          base_y_offset);

  // Fade out at the end.
  const float alpha = std::max(0.0f, std::min(1.0f, timer * 2.0f));
  const ImU32 bg_color =
      IM_COL32(32, 33, 36, (int)(230.0f * alpha));  // Dark grey
  const ImU32 text_color = IM_COL32(255, 255, 255, (int)(255.0f * alpha));

  draw_list->AddRectFilled(
      toast_pos, ImVec2(toast_pos.x + toast_size.x, toast_pos.y + toast_size.y),
      bg_color, kToastCornerRounding);
  draw_list->AddText(ImVec2(toast_pos.x + padding.x, toast_pos.y + padding.y),
                     text_color, message.data(),
                     message.data() + message.size());
}

void Timeline::HandleMouseRelease() {
  if (ImGui::IsMouseReleased(0)) {
    is_dragging_ = false;
    is_selecting_ = false;
    time_range_resizing_state_.reset();

    bool is_click = false;
    if (selection_start_pos_) {
      const float dx = ImGui::GetIO().MousePos.x - selection_start_pos_->x;
      const float dy = ImGui::GetIO().MousePos.y - selection_start_pos_->y;
      const float distance_squared = dx * dx + dy * dy;
      if (distance_squared <= kClickDistanceThresholdSquared) {
        is_click = true;
      }
    }

    // If a bookmark was handled (e.g., added via Ctrl+Click), bypass
    // selection or time range additions. This prevents accidentally adding
    // tiny time ranges if the user slightly moved the mouse (within
    // threshold) during the click.
    if (!HandleBookmarkAddition(is_click)) {
      HandleSelectionOrTimeRangeAddition();
      if (redraw_callback_)
          redraw_callback_();  // Trigger redraw to show points
    }

    selection_start_pos_.reset();
    selection_end_pos_.reset();
    current_selected_time_range_.reset();
  }
}

bool Timeline::HandleBookmarkAddition(bool is_click) {
  if (bookmarks_enabled_ && is_click &&
      (ImGui::GetIO().KeyCtrl || ImGui::GetIO().KeySuper)) {
    const ImRect timeline_area = GetTimelineArea();
    if (ImGui::IsMouseHoveringRect(timeline_area.Min, timeline_area.Max)) {
      const Microseconds time = PixelToTime(
          ImGui::GetMousePos().x - timeline_area.Min.x, px_per_time_unit());
      AddBookmark(time);
      return true;
    }
  }
  return false;
}

bool Timeline::HandleSelectionOrTimeRangeAddition() {
  if (mouse_mode_ == MouseMode::kSelect && selection_start_pos_ &&
      selection_end_pos_) {
    const float dx = selection_end_pos_->x - selection_start_pos_->x;
    const float dy = selection_end_pos_->y - selection_start_pos_->y;
    const float distance_squared = dx * dx + dy * dy;
    if (distance_squared > kClickDistanceThresholdSquared) {
      ImRect selection_rect =
          ImRect(std::min(selection_start_pos_->x, selection_end_pos_->x),
                 std::min(selection_start_pos_->y, selection_end_pos_->y),
                 std::max(selection_start_pos_->x, selection_end_pos_->x),
                 std::max(selection_start_pos_->y, selection_end_pos_->y));
      FindSelectedEvents(selection_rect);
      CalculateAndEmitMetrics();
      if (redraw_callback_)
        redraw_callback_();  // Trigger redraw to show points
      return true;
      } else {
        // Simple click in Select mode clears rectangle selection in UI.
        CalculateAndEmitMetrics();
    }
  } else if (current_selected_time_range_ &&
             current_selected_time_range_->duration() > 0) {
    selected_time_ranges_.push_back(*current_selected_time_range_);
    return true;
  }
  return false;
}

ImRect Timeline::GetTimelineArea() const {
  const ImVec2 window_pos = ImGui::GetWindowPos();
  const ImVec2 content_min = ImGui::GetWindowContentRegionMin();
  const Pixel start_x = window_pos.x + content_min.x + label_width_;
  const Pixel start_y = window_pos.y + content_min.y;

  const Pixel end_x = start_x + current_timeline_width_;
  const Pixel end_y = window_pos.y + ImGui::GetWindowHeight();

  return {start_x, start_y, end_x, end_y};
}

void Timeline::InitializeLastFetchRequestRange(const TimeRange& visible_range) {
  TimeRange fetch = visible_range.Scale(kFetchRatio);

  if (fetch.duration() < kMinFetchDurationMicros) {
    Microseconds center = fetch.center();
    fetch = {center - kMinFetchDurationMicros / 2.0,
             center + kMinFetchDurationMicros / 2.0};
  }

  ConstrainTimeRange(fetch);
  last_fetch_request_range_ = fetch;
}

void Timeline::MaybeRequestData() {
  // Don't request more data if a request is already in flight.
  if (is_incremental_loading_) return;

  // We have several ranges of interest for incremental loading:
  //
  // |-----------data_time_range_---------|                  Full trace duration
  //     |---------fetch----------|         Amount to fetch on load (viewport*3)
  //         |----preserve----|               Buffer to keep loaded (viewport*2)
  //             |viewport|              aka current_visible: On-screen viewport
  //
  // If 'preserve' isn't contained in 'last_fetch_request_range_', or resolution
  // is too coarse, a new load of 'fetch' range is triggered.
  // - last_fetch_request_range_: The time range covered by data currently
  // loaded.
  const TimeRange current_visible = visible_range();

  TimeRange preserve = current_visible.Scale(kPreserveRatio);
  TimeRange fetch = current_visible.Scale(kFetchRatio);

  if (fetch.duration() < kMinFetchDurationMicros) {
    Microseconds center = fetch.center();
    fetch = {center - kMinFetchDurationMicros / 2.0,
             center + kMinFetchDurationMicros / 2.0};
  }

  // Constrain the ranges to the valid data range. This ensures that we don't
  // try to fetch data outside the available trace duration (e.g. negative time
  // or future time), preventing infinite refetch loops at the boundaries.
  ConstrainTimeRange(preserve);
  ConstrainTimeRange(fetch);

  // Refetch data if user zoomed in significantly, making resolution too coarse.
  // We use last_fetch_request_range_ here because it reflects the density of
  // the data we hold (asked for), whereas fetched_data_time_range_ might be
  // smaller/sparse depending on actual event content.
  const bool zoomed_in_too_much =
      (last_fetch_request_range_.duration() / fetch.duration() >
       kRefetchZoomRatio);

  // Utilize the last fetch request range as a guard to prevent redundant
  // fetches only if we don't need higher resolution data.
  //
  // If `last_fetch_request_range_` contains `preserve`, it means we've already
  // successfully requested a superset of the required data.
  // EXCEPT IF `zoomed_in_too_much` is true: in that case, even if we have the
  // range, the data density might be too low, so we MUST refetch to get
  // higher-resolution data.
  if (!zoomed_in_too_much && last_fetch_request_range_.Contains(preserve)) {
    return;
  }

  EventData event_data;
  event_data.try_emplace(kFetchDataStart, MicrosToMillis(fetch.start()));
  event_data.try_emplace(kFetchDataEnd, MicrosToMillis(fetch.end()));

  event_callback_(kFetchData, event_data);

  last_fetch_request_range_ = fetch;

  // We set is_incremental_loading_ to true to prevent sending duplicate
  // requests. The flag will be reset to false when the data is received and
  // processed.
  is_incremental_loading_ = true;
}

void Timeline::SetSearchQuery(absl::string_view query) {
  const std::string new_query_lower = absl::AsciiStrToLower(query);
  if (search_query_lower_ == new_query_lower) {
    if (redraw_callback_) redraw_callback_();
    return;
  }
  search_query_lower_ = new_query_lower;
  pending_navigation_event_id_.reset();
  search_results_.clear();
  matching_event_indices_.clear();
  current_search_result_index_ = -1;

  if (query.empty()) {
    if (redraw_callback_) redraw_callback_();
    return;
  }

  RecomputeSearchResults();
}

void Timeline::RecomputeSearchResults() {
  const int num_entries = timeline_data_.entry_names.size();
  for (int i = 0; i < num_entries; ++i) {
    const std::string& name = timeline_data_.entry_names[i];
    if (absl::StartsWithIgnoreCase(name, search_query_lower_)) {
      const EventId event_id = timeline_data_.entry_event_ids[i];
      const int level = timeline_data_.entry_levels[i];
      const Microseconds start_time = timeline_data_.entry_start_times[i];
      const Microseconds total_time = timeline_data_.entry_total_times[i];
      const ProcessId pid = timeline_data_.entry_pids[i];
      const ThreadId tid = timeline_data_.entry_tids[i];
      search_results_.push_back(SearchResult{.event_id = event_id,
                                             .level = level,
                                             .start_time = start_time,
                                             .duration = total_time,
                                             .pid = pid,
                                             .tid = tid,
                                             .name = name,
                                             .loaded_index = i});
      matching_event_indices_.insert(i);
    }
  }

  // Sort results horizontally by track hierarchy and start time.
  absl::c_sort(search_results_,
               [](const SearchResult& a, const SearchResult& b) {
                 return std::tie(a.level, a.start_time, a.loaded_index) <
                        std::tie(b.level, b.start_time, b.loaded_index);
               });

  if (redraw_callback_) redraw_callback_();
}

void Timeline::SetSearchResults(const ParsedTraceEvents& search_results) {
  const bool navigation_active = (current_search_result_index_ >= 0);

  search_results_.clear();
  matching_event_indices_.clear();
  current_search_result_index_ = -1;

  if (search_query_lower_.empty()) {
    if (redraw_callback_) redraw_callback_();
    return;
  }

  absl::flat_hash_map<EventId, int> event_id_to_loaded_index;
  absl::flat_hash_map<EventId, int> event_id_to_level;
  const int num_loaded = timeline_data_.entry_event_ids.size();
  event_id_to_loaded_index.reserve(num_loaded);
  event_id_to_level.reserve(num_loaded);
  for (int i = 0; i < num_loaded; ++i) {
    const EventId ev_id = timeline_data_.entry_event_ids[i];
    event_id_to_loaded_index.try_emplace(ev_id, i);
    const int lvl = (i < timeline_data_.entry_levels.size())
                        ? timeline_data_.entry_levels[i]
                        : -1;
    event_id_to_level.try_emplace(ev_id, lvl);
  }

  std::optional<absl::flat_hash_map<FallbackKey, std::vector<int>>>
      lazy_loaded_events;

  for (const TraceEvent& event : search_results.flame_events) {
    if (event.ph != Phase::kComplete && event.ph != Phase::kInstant &&
        event.ph != Phase::kInstantDeprecated) {
      continue;
    }
    int level = -1;
    if (const auto it = event_id_to_level.find(event.event_id);
        it != event_id_to_level.end()) {
      level = it->second;
    }
    EventId event_id = event.event_id;
    int loaded_index = -1;
    if (const auto it = event_id_to_loaded_index.find(event.event_id);
        it != event_id_to_loaded_index.end()) {
      loaded_index = it->second;
    } else {
      if (!lazy_loaded_events.has_value()) {
        lazy_loaded_events = BuildFallbackMap(timeline_data_);
      }
      loaded_index = FindFallbackMatchedIndex(
          *lazy_loaded_events, timeline_data_, event.pid, event.tid, event.name,
          event.ts, event.dur);
      if (loaded_index != -1) {
        event_id = timeline_data_.entry_event_ids[loaded_index];
      }
    }

    search_results_.push_back(SearchResult{.event_id = event_id,
                                           .level = level,
                                           .start_time = event.ts,
                                           .duration = event.dur,
                                           .pid = event.pid,
                                           .tid = event.tid,
                                           .name = event.name,
                                           .loaded_index = loaded_index});
    if (loaded_index != -1) {
      matching_event_indices_.insert(loaded_index);
    }
  }

  if (search_results_.empty()) {
    if (redraw_callback_) redraw_callback_();
    return;
  }

  absl::flat_hash_map<ProcessId, uint32_t> process_sort_indices =
      GetProcessSortIndices(search_results);
  absl::flat_hash_map<std::pair<ProcessId, ThreadId>, uint32_t>
      thread_sort_indices;
  for (const TraceEvent& event : search_results.flame_events) {
    if (event.ph == Phase::kMetadata && event.name == kThreadSortIndex) {
      if (const auto it = event.args.find(kSortIndex); it != event.args.end()) {
        double sort_index_double;
        if (absl::SimpleAtod(it->second, &sort_index_double)) {
          thread_sort_indices[{event.pid, event.tid}] =
              static_cast<uint32_t>(sort_index_double);
        }
      }
    }
  }

  absl::flat_hash_map<std::pair<ProcessId, ThreadId>, int> pid_tid_to_min_level;
  for (int i = 0; i < num_loaded; ++i) {
    const ProcessId pid = (i < timeline_data_.entry_pids.size())
                              ? timeline_data_.entry_pids[i]
                              : 0;
    const ThreadId tid = (i < timeline_data_.entry_tids.size())
                             ? timeline_data_.entry_tids[i]
                             : 0;
    const int level = (i < timeline_data_.entry_levels.size())
                          ? timeline_data_.entry_levels[i]
                          : -1;
    if (level != -1) {
      auto [it, inserted] = pid_tid_to_min_level.try_emplace({pid, tid}, level);
      if (level < it->second) {
        it->second = level;
      }
    }
  }

  absl::c_sort(
      search_results_, [&](const SearchResult& a, const SearchResult& b) {
        if (a.pid != b.pid) {
          const auto it_a = process_sort_indices.find(a.pid);
          uint32_t sort_index_a = (it_a != process_sort_indices.end())
                                      ? it_a->second
                                      : std::numeric_limits<uint32_t>::max();
          const auto it_b = process_sort_indices.find(b.pid);
          uint32_t sort_index_b = (it_b != process_sort_indices.end())
                                      ? it_b->second
                                      : std::numeric_limits<uint32_t>::max();
          if (sort_index_a != sort_index_b) {
            return sort_index_a < sort_index_b;
          }
          return a.pid < b.pid;
        }

        if (a.tid != b.tid) {
          int min_level_a = -1;
          if (const auto it_min_a = pid_tid_to_min_level.find({a.pid, a.tid});
              it_min_a != pid_tid_to_min_level.end()) {
            min_level_a = it_min_a->second;
          }
          int min_level_b = -1;
          if (const auto it_min_b = pid_tid_to_min_level.find({b.pid, b.tid});
              it_min_b != pid_tid_to_min_level.end()) {
            min_level_b = it_min_b->second;
          }

          if (min_level_a != -1 && min_level_b != -1) {
            if (min_level_a != min_level_b) {
              return min_level_a < min_level_b;
            }
          }

          uint32_t thread_sort_a = std::numeric_limits<uint32_t>::max();
          if (const auto it_thread_a = thread_sort_indices.find({a.pid, a.tid});
              it_thread_a != thread_sort_indices.end()) {
            thread_sort_a = it_thread_a->second;
          }
          uint32_t thread_sort_b = std::numeric_limits<uint32_t>::max();
          if (const auto it_thread_b = thread_sort_indices.find({b.pid, b.tid});
              it_thread_b != thread_sort_indices.end()) {
            thread_sort_b = it_thread_b->second;
          }

          if (thread_sort_a != thread_sort_b) {
            return thread_sort_a < thread_sort_b;
          }
          return a.tid < b.tid;
        }

        int effective_level_a = a.level;
        if (effective_level_a == -1) {
          const auto it_min = pid_tid_to_min_level.find({a.pid, a.tid});
          effective_level_a =
              (it_min != pid_tid_to_min_level.end()) ? it_min->second : 0;
        }
        int effective_level_b = b.level;
        if (effective_level_b == -1) {
          const auto it_min = pid_tid_to_min_level.find({b.pid, b.tid});
          effective_level_b =
              (it_min != pid_tid_to_min_level.end()) ? it_min->second : 0;
        }
        if (effective_level_a != effective_level_b) {
          return effective_level_a < effective_level_b;
        }
        return a.start_time < b.start_time;
      });

  if (navigation_active) {
    current_search_result_index_ = 0;
    NavigateToSearchResult(search_results_[0]);
  }

  if (redraw_callback_) redraw_callback_();
}

void Timeline::NavigateToSearchResult(const SearchResult& result) {
  if (result.loaded_index != -1) {
    ZoomEvent(result.loaded_index);
  } else {
    pending_navigation_event_id_ = result.event_id;
    const Microseconds start = result.start_time;
    const Microseconds event_duration = result.duration;
    const Microseconds duration =
        std::max(kEventNavigationMinDurationMicros,
                 std::min(event_duration * kEventNavigationZoomFactor,
                          kEventNavigationMaxDurationMicros));
    const Microseconds center = start + event_duration / 2.0;
    TimeRange new_range = {center - duration / 2.0, center + duration / 2.0};
    ConstrainTimeRange(new_range);
    SetVisibleRange(new_range, /*animate=*/true);
  }
}

void Timeline::NavigateToNextSearchResult() {
  if (search_results_.empty()) return;
  current_search_result_index_++;
  if (current_search_result_index_ >= search_results_.size()) {
    current_search_result_index_ = 0;
  }
  NavigateToSearchResult(search_results_[current_search_result_index_]);
  if (redraw_callback_) redraw_callback_();
}

void Timeline::NavigateToPrevSearchResult() {
  if (search_results_.empty()) return;
  current_search_result_index_--;
  if (current_search_result_index_ < 0) {
    current_search_result_index_ = search_results_.size() - 1;
  }
  NavigateToSearchResult(search_results_[current_search_result_index_]);
  if (redraw_callback_) redraw_callback_();
}

void Timeline::FindSelectedEvents(const ImRect& selection_rect) {
  selected_event_indices_.clear();
  selected_counter_points_.clear();

  const ImRect timeline_area = GetTimelineArea();
  const Pixel screen_x_offset = timeline_area.Min.x;
  const double px_per_time = px_per_time_unit();

  for (size_t group_index = 0; group_index < timeline_data_.groups.size();
       ++group_index) {
    const auto& group = timeline_data_.groups[group_index];
    if (!group.expanded) continue;

    if (group.type == Group::Type::kFlame) {
      const int start_level = group.start_level;
      int end_level = GetNextGroupStartLevel(timeline_data_, group_index);

      for (int level = start_level; level < end_level; ++level) {
        if (level >= timeline_data_.events_by_level.size()) continue;

        const auto& events = timeline_data_.events_by_level[level];
        const Pixel y_top = tracks_start_screen_pos_.y +
                            visible_level_offsets_[level] -
                            kEventHeight * 0.5f;
        const Pixel y_bottom = y_top + kEventHeight;

        if (y_bottom < selection_rect.Min.y || y_top > selection_rect.Max.y) {
          continue;
        }

        Microseconds selection_start_time =
            PixelToTime(selection_rect.Min.x - screen_x_offset, px_per_time);
        Microseconds selection_end_time =
            PixelToTime(selection_rect.Max.x - screen_x_offset, px_per_time);

        auto it =
            std::lower_bound(events.begin(), events.end(), selection_start_time,
                             [&](int event_idx, Microseconds t) {
                               const Microseconds end =
                                   timeline_data_.entry_start_times[event_idx] +
                                   timeline_data_.entry_total_times[event_idx];
                               return end <= t;
                             });

        for (; it != events.end(); ++it) {
          const int event_index = *it;
          const Microseconds start =
              timeline_data_.entry_start_times[event_index];
          if (start > selection_end_time) {
            break;
          }

          const Microseconds end =
              start + timeline_data_.entry_total_times[event_index];

          const Pixel left = TimeToScreenX(start, screen_x_offset, px_per_time);
          Pixel right = TimeToScreenX(end, screen_x_offset, px_per_time);
          right = std::max(right, left + kEventMinimumDrawWidth);

          if (right < selection_rect.Min.x || left > selection_rect.Max.x) {
            continue;
          }

          selected_event_indices_.push_back(event_index);
        }
      }
    } else if (group.type == Group::Type::kCounter) {
      Pixel y_top =
          tracks_start_screen_pos_.y + group_offsets_[group_index];
      Pixel group_height = kCounterTrackHeight;
      Pixel y_bottom = y_top + group_height;

      if (y_bottom < selection_rect.Min.y || y_top > selection_rect.Max.y) {
        continue;
      }

      const auto it =
          timeline_data_.counter_data_by_group_index.find(group_index);
      if (it == timeline_data_.counter_data_by_group_index.end()) continue;
      const auto& counter_data = it->second;

      const double value_range =
          counter_data.max_value - counter_data.min_value;
      const float y_ratio = value_range > 0 ? group_height / value_range : 0.0f;

      Microseconds selection_start_time =
          PixelToTime(selection_rect.Min.x - screen_x_offset, px_per_time);
      Microseconds selection_end_time =
          PixelToTime(selection_rect.Max.x - screen_x_offset, px_per_time);

      auto it_start =
          std::lower_bound(counter_data.timestamps.begin(),
                           counter_data.timestamps.end(), selection_start_time);
      auto it_end =
          std::upper_bound(counter_data.timestamps.begin(),
                           counter_data.timestamps.end(), selection_end_time);

      for (auto it = it_start; it != it_end; ++it) {
        size_t i = std::distance(counter_data.timestamps.begin(), it);
        double val = counter_data.values[i];

        Pixel x = TimeToScreenX(*it, screen_x_offset, px_per_time);
        Pixel y = (value_range > 0)
                      ? y_top + group_height -
                            (val - counter_data.min_value) * y_ratio
                      : y_top + group_height / 2.0f;

        if (selection_rect.Contains(ImVec2(x, y))) {
          selected_counter_points_.push_back({group_index, i});
        }
      }
    }
  }
}

void Timeline::CalculateAndEmitMetrics() {
  if (selected_event_indices_.empty() && selected_counter_points_.empty()) {
    event_callback_(kEventsSelected, EventData());
    return;
  }

  Microseconds selection_start_us = 0;
  Microseconds selection_extent_us = 0;
  if (selection_start_pos_ && selection_end_pos_ &&
      visible_range_->duration() > 0) {
    ImRect timeline_area = GetTimelineArea();
    double px_per_time = current_timeline_width_ / visible_range_->duration();
    Microseconds start_us =
        PixelToTime(std::min(selection_start_pos_->x, selection_end_pos_->x) -
                        timeline_area.Min.x,
                    px_per_time);
    Microseconds end_us =
        PixelToTime(std::max(selection_start_pos_->x, selection_end_pos_->x) -
                        timeline_area.Min.x,
                    px_per_time);
    selection_start_us = start_us;
    selection_extent_us = end_us - start_us;
  }

  std::string json =
      absl::StrFormat(R"({"selectionStartUs":%.1f,"selectionExtentUs":%.1f)",
                      selection_start_us, selection_extent_us);

  if (!selected_event_indices_.empty()) {
    struct Metrics {
      int count = 0;
      Microseconds wall_time = 0;
      Microseconds self_time = 0;
    };

    absl::flat_hash_map<std::string, Metrics> aggregated_metrics;

    for (const int event_index : selected_event_indices_) {
      const std::string& name = timeline_data_.entry_names[event_index];
      Microseconds wall = timeline_data_.entry_total_times[event_index];
      Microseconds self = timeline_data_.entry_self_times[event_index];

      Metrics& m = aggregated_metrics[name];
      m.count++;
      m.wall_time += wall;
      m.self_time += self;
    }

    std::string metrics_json = "[";
    bool first = true;
    for (const auto& [name, metrics] : aggregated_metrics) {
      if (!first) absl::StrAppend(&metrics_json, ",");
      first = false;
      absl::StrAppendFormat(
          &metrics_json,
          R"({"name":"%s","count":%d,"wallTimeUs":%.1f,"selfTimeUs":%.1f,"avgWallDurationUs":%.1f})",
          name, metrics.count, metrics.wall_time, metrics.self_time,
          metrics.wall_time / metrics.count);
    }
    absl::StrAppend(&metrics_json, "]");

    absl::StrAppend(&json, R"(,"metrics":)", metrics_json);
  }

  if (!selected_counter_points_.empty()) {
    std::string counters_json = "[";
    bool first = true;
    for (const auto& [group_index, point_index] : selected_counter_points_) {
      const auto& group = timeline_data_.groups[group_index];
      const auto it =
          timeline_data_.counter_data_by_group_index.find(group_index);
      if (it == timeline_data_.counter_data_by_group_index.end()) continue;
      const auto& counter_data = it->second;

      Microseconds ts = counter_data.timestamps[point_index];
      double val = counter_data.values[point_index];

      if (!first) absl::StrAppend(&counters_json, ",");
      first = false;
      absl::StrAppendFormat(
          &counters_json,
          R"({"counter":"%s","series":"%s","time":%.1f,"value":%.1f})",
          group.name, counter_data.event_stats, ts, val);
    }
    absl::StrAppend(&counters_json, "]");

    absl::StrAppend(&json, R"(,"counters":)", counters_json);
  }

  absl::StrAppend(&json, "}");

  EventData event_data;
  event_data.try_emplace(kEventsSelectedData, json);
  event_callback_(kEventsSelected, event_data);
}

void Timeline::DrawSelectionRectangle() {
  if (mouse_mode_ != MouseMode::kSelect || !is_selecting_ ||
      !selection_start_pos_ || !selection_end_pos_) {
    return;
  }

  ImDrawList* draw_list = ImGui::GetForegroundDrawList();
  const ImU32 color = IM_COL32(0, 120, 215, 64);
  const ImU32 border_color = IM_COL32(0, 120, 215, 255);

  ImRect rect =
      ImRect(std::min(selection_start_pos_->x, selection_end_pos_->x),
             std::min(selection_start_pos_->y, selection_end_pos_->y),
             std::max(selection_start_pos_->x, selection_end_pos_->x),
             std::max(selection_start_pos_->y, selection_end_pos_->y));

  draw_list->AddRectFilled(rect.Min, rect.Max, color);
  draw_list->AddRect(rect.Min, rect.Max, border_color, 0.0f, 0, 2.0f);
}

void Timeline::DrawBookmarks(Pixel timeline_width,
                             double px_per_time_unit_val) {
  if (!bookmarks_enabled_ || bookmarks_.empty() || px_per_time_unit_val <= 0)
    return;

  const ImRect timeline_area = GetTimelineArea();
  const Pixel screen_x_offset = timeline_area.Min.x;
  ImDrawList* draw_list = ImGui::GetForegroundDrawList();
  std::optional<Microseconds> bookmark_to_delete;

  for (const Microseconds time : bookmarks_) {
    const Pixel x = TimeToScreenX(time, screen_x_offset, px_per_time_unit_val);

    if (x < timeline_area.Min.x || x > timeline_area.Max.x) continue;

    // Draw vertical line
    draw_list->AddLine(ImVec2(x, timeline_area.Min.y),
                       ImVec2(x, timeline_area.Max.y), kBookmarkColor,
                       kBookmarkThickness);

    // Draw label (timestamp)
    const std::string label = FormatTime(time);
    ImGui::PushFont(fonts::label_small);
    const ImVec2 label_size = ImGui::CalcTextSize(label.c_str());
    const ImVec2 label_pos = ImVec2(
        x + kBookmarkLabelPadding, timeline_area.Min.y + kBookmarkLabelPadding);
    draw_list->AddText(label_pos, kBookmarkColor, label.c_str());

    // Unified Delete Button logic (matches TimeRange selection behavior)
    const ImVec2 button_pos =
        ImVec2(label_pos.x + label_size.x + kBookmarkLabelPadding, label_pos.y);
    const ImRect button_rect(button_pos,
                             ImVec2(button_pos.x + kCloseButtonSize,
                                    button_pos.y + kCloseButtonSize));

    // Hover area includes label and button
    const ImRect hover_rect(label_pos, button_rect.Max);

    if (DrawCloseButton(draw_list, button_pos, hover_rect)) {
      bookmark_to_delete = time;
    }

    ImGui::PopFont();
  }

  if (bookmark_to_delete) {
    RemoveBookmark(*bookmark_to_delete);
  }
}

void Timeline::AddBookmark(Microseconds time) {
  if (!bookmarks_enabled_) return;
  const double px_per_time = px_per_time_unit();
  const Microseconds threshold = px_per_time > 0 ? 5.0 / px_per_time : 1e-9;

  auto it = absl::c_find_if(bookmarks_, [time, threshold](Microseconds b) {
    return std::abs(b - time) < threshold;
  });

  if (it == bookmarks_.end()) {
    bookmarks_.push_back(time);
    absl::c_sort(bookmarks_);
    if (redraw_callback_) redraw_callback_();
  }
}

void Timeline::RemoveBookmark(Microseconds time) {
  if (!bookmarks_enabled_) return;
  auto it = absl::c_find(bookmarks_, time);
  if (it != bookmarks_.end()) {
    bookmarks_.erase(it);
    if (redraw_callback_) redraw_callback_();
  }
}

}  // namespace traceviewer
