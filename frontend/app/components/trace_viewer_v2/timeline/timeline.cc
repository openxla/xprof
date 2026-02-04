#include "xprof/frontend/app/components/trace_viewer_v2/timeline/timeline.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <numeric>
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
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "third_party/dear_imgui/imgui.h"
#include "third_party/dear_imgui/imgui_internal.h"
#include "xprof/frontend/app/components/trace_viewer_v2/color/color_generator.h"
#include "xprof/frontend/app/components/trace_viewer_v2/event_data.h"
#include "xprof/frontend/app/components/trace_viewer_v2/fonts/fonts.h"
#include "xprof/frontend/app/components/trace_viewer_v2/helper/time_formatter.h"
#include "xprof/frontend/app/components/trace_viewer_v2/timeline/constants.h"
#include "xprof/frontend/app/components/trace_viewer_v2/timeline/time_range.h"
#include "xprof/frontend/app/components/trace_viewer_v2/trace_helper/trace_event.h"

namespace traceviewer {
namespace {

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

// The argument name for sort index in process_sort_index and
// thread_sort_index metadata events.
constexpr absl::string_view kSortIndex = "sort_index";
constexpr absl::string_view kProcessSortIndex = "process_sort_index";

// Extracts process sort indices from metadata events in search results.
absl::flat_hash_map<ProcessId, uint32_t> GetProcessSortIndices(
    const ParsedTraceEvents& search_results) {
  absl::flat_hash_map<ProcessId, uint32_t> process_sort_indices;
  for (const auto& event : search_results.flame_events) {
    if (event.ph == Phase::kMetadata && event.name == kProcessSortIndex) {
      if (auto it = event.args.find(std::string(kSortIndex));
          it != event.args.end()) {
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
}  // namespace

void Timeline::SetSearchQuery(const std::string& query) {
  search_query_lower_ = absl::AsciiStrToLower(query);
  pending_navigation_event_id_.reset();
  RecomputeSearchResults();
}

void Timeline::SetVisibleRange(const TimeRange& range, bool animate) {
  if (animate) {
    visible_range_ = range;
  } else {
    visible_range_.snap_to(range);
  }
}

void Timeline::SetSearchResults(const ParsedTraceEvents& search_results) {
  // If a search result is currently selected, save its event id for reference,
  // so we can find and focus on the same event, and keep the selection view
  // persistent while search results being updated on the background.
  EventId selected_event_id = -1;
  if (current_search_result_index_ >= 0 &&
      current_search_result_index_ < sorted_search_results_.size()) {
    selected_event_id =
        sorted_search_results_[current_search_result_index_].event_id;
  }

  // Clear previous search results and reset the index.
  sorted_search_results_.clear();
  current_search_result_index_ = -1;

  // If the search query is empty, there are no results to process.
  if (search_query_lower_.empty()) return;

  // Build a map of event IDs to their levels in the timeline for quick lookup.
  // This helps in assigning levels to search results for sorting.
  // Level information is only available for events in timeline_data_,
  // search results not in timeline_data_ will be assigned level -1.
  absl::flat_hash_map<EventId, int> event_id_to_level;
  for (int i = 0; i < timeline_data_.entry_event_ids.size(); ++i) {
    event_id_to_level.try_emplace(timeline_data_.entry_event_ids[i],
                                  timeline_data_.entry_levels[i]);
  }

  // Filter for complete events from the search results and populate the
  // sorted_search_results_ vector with relevant event data.
  for (const auto& event : search_results.flame_events) {
    if (event.ph != Phase::kComplete) continue;
    // TODO: jonahweaver - Get level information for search results
    // for proper navigation.
    int level = -1;
    if (auto it = event_id_to_level.find(event.event_id);
        it != event_id_to_level.end()) {
      level = it->second;
    }
    sorted_search_results_.push_back({event.event_id, level, event.ts,
                                    event.dur, event.pid, event.tid});
  }

  // If no complete events were found, there's nothing more to do.
  if (sorted_search_results_.empty()) return;

  // Extract process sort indices from metadata events. These indices are used
  // to sort search results in an order consistent with the timeline display.
  absl::flat_hash_map<ProcessId, uint32_t> process_sort_indices =
      GetProcessSortIndices(search_results);
  // Sort the search results. The sorting order is primarily based on
  // process sort index, then by process ID, thread ID, level, and finally
  // event start time. This ensures a stable and intuitive navigation order.

  // PID and TID are used as a fallback sorting criteria
  // to best maintain order by level while levels are not available.
  absl::c_sort(sorted_search_results_, [&](const auto& a, const auto& b) {
    auto it_a = process_sort_indices.find(a.pid);
    uint32_t sort_index_a =
        (it_a != process_sort_indices.end()) ? it_a->second : a.pid;
    auto it_b = process_sort_indices.find(b.pid);
    uint32_t sort_index_b =
        (it_b != process_sort_indices.end()) ? it_b->second : b.pid;
    return std::tie(sort_index_a, a.pid, a.tid, a.level, a.start_time) <
           std::tie(sort_index_b, b.pid, b.tid, b.level, b.start_time);
  });

  // If an event was selected before the update, try to find it in the new
  // sorted list and restore the selection index.
  if (selected_event_id != -1) {
    auto it = absl::c_find_if(sorted_search_results_,
                            [&](const auto& result) {
                              return result.event_id == selected_event_id;
                            });
    if (it != sorted_search_results_.end()) {
      current_search_result_index_ =
          std::distance(sorted_search_results_.begin(), it);
    }
  }
}

void Timeline::set_timeline_data(FlameChartTimelineData data) {
  timeline_data_ = std::move(data);
  if (pending_navigation_event_id_.has_value()) {
    EventId event_id = pending_navigation_event_id_.value();
    auto it = absl::c_find(timeline_data_.entry_event_ids, event_id);
    if (it != timeline_data_.entry_event_ids.end()) {
      NavigateToEvent(
          std::distance(timeline_data_.entry_event_ids.begin(), it));
      pending_navigation_event_id_.reset();
    }
  }
}

void Timeline::Draw() {
  event_clicked_this_frame_ = false;
  level_y_positions_.assign(timeline_data_.events_by_level.size(), -FLT_MAX);

  const ImGuiViewport* viewport = ImGui::GetMainViewport();
  ImGui::SetNextWindowPos(viewport->Pos);
  ImGui::SetNextWindowSize(viewport->Size);
  ImGui::SetNextWindowViewport(viewport->ID);

  ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.f);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
  ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, ImVec2(0.0f, 0.0f));
  ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0.0f, 0.0f));

  ImGui::Begin("Timeline viewer", nullptr, kImGuiWindowFlags);

  const Pixel timeline_width =
      ImGui::GetContentRegionAvail().x - label_width_ - kTimelinePaddingRight;
  const double px_per_time_unit_val = px_per_time_unit(timeline_width);

  DrawRuler(timeline_width, viewport->Pos.y + viewport->Size.y);

  // The tracks are in a child window to allow scrolling independently of the
  // ruler.
  // Keep the NoScrollWithMouse flag to disable the default scroll behavior
  // of ImGui, and use the custom scroll handler defined in `HandleWheel`
  // instead.
  ImGui::BeginChild("Tracks", ImVec2(0, 0), ImGuiChildFlags_None,
                    ImGuiWindowFlags_NoScrollWithMouse);

  ImGui::BeginTable("Timeline", 2, kImGuiTableFlags, ImVec2(0.0f, -FLT_MIN));
  ImGui::TableSetupColumn("Labels", ImGuiTableColumnFlags_WidthFixed,
                          label_width_);
  ImGui::TableSetupColumn("Timeline", ImGuiTableColumnFlags_WidthStretch);

  for (int group_index = 0; group_index < timeline_data_.groups.size();
       ++group_index) {
    Group& group = timeline_data_.groups[group_index];
    group.has_children = true;
    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    // Indent the group name. We add 1 to the nesting level because
    // ImGui::Indent(0) results in a default, potentially large indentation.
    // By adding 1, even top-level groups (nesting_level 0) receive a base
    // indentation of `kIndentSize`, ensuring consistent and controlled visual
    // separation from the left edge of the table column.
    ImGui::Indent((group.nesting_level + 1) * kIndentSize);
    if (group.has_children) {
      if (ImGui::ArrowButton(std::to_string(group_index).c_str(),
                             group.expanded ? ImGuiDir_Down : ImGuiDir_Right)) {
        group.expanded = !group.expanded;
      }
      ImGui::SameLine();
    }
    ImGui::TextUnformatted(group.name.c_str());
    ImGui::Unindent((group.nesting_level + 1) * kIndentSize);

    ImGui::TableNextColumn();

    DrawGroup(group_index, px_per_time_unit_val);

    if (group.has_children && !group.expanded && group.nesting_level == 0) {
      int current_nesting_level = group.nesting_level;
      while (group_index + 1 < timeline_data_.groups.size() &&
             timeline_data_.groups[group_index + 1].nesting_level >
                 current_nesting_level) {
        group_index++;
      }
    }
  }

  ImGui::EndTable();

  HandleEventDeselection();

  // Handle continuous keyboard and mouse wheel input for timeline navigation.
  // These functions are called every frame to ensure smooth and responsive
  // interaction.
  // The performance impact is fine because HandleKeyboard/HandleWheel() only
  // performs lightweight checks and calculations.
  bool is_interacting = false;
  is_interacting |= HandleKeyboard();
  is_interacting |= HandleWheel();
  is_interacting |= HandleMouse();

  // We call MaybeRequestData() to check if we need to fetch more data.
  // We do this here instead of in SetVisibleRange because we want to debounce
  // the request during continuous interaction (like panning/zooming).
  // The check is skipped if the user is currently interacting with
  // the timeline to avoid sending requests during interaction.
  if (!is_interacting) {
    MaybeRequestData();
  }

  ImGui::EndChild();

  // Draw the selected time range in a separate overlay child window.
  // This ensures it is drawn on top of the "Tracks" child window (because it's
  // declared after) but below tooltips (because it's a child window, not
  // in the foreground draw list).
  ImGui::SetCursorPos(ImVec2(0, 0));
  ImGui::BeginChild("SelectionOverlay", ImVec2(0, 0), ImGuiChildFlags_None,
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
  DrawFlows(timeline_width);
  DrawSelectedTimeRanges(timeline_width, px_per_time_unit_val);
  ImGui::EndChild();

  ImGui::PopStyleVar();  // ItemSpacing
  ImGui::PopStyleVar();  // CellPadding
  ImGui::PopStyleVar();  // WindowPadding
  ImGui::PopStyleVar();  // WindowBorderSize
  ImGui::PopStyleVar();  // WindowRounding
  ImGui::End();          // Timeline viewer
}

EventRect Timeline::CalculateEventRect(Microseconds start, Microseconds end,
                                       Pixel screen_x_offset,
                                       Pixel screen_y_offset,
                                       double px_per_time_unit,
                                       int level_in_group,
                                       Pixel timeline_width) const {
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
      screen_y_offset + level_in_group * (kEventHeight + kEventPaddingBottom);
  const Pixel bottom = top + kEventHeight;

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
                                        float available_width) const {
  const ImVec2 text_size = GetTextSize(event_name);

  if (text_size.x > available_width) {
    // Truncate text with "..." at the end
    const float ellipsis_width = GetTextSize("...").x;
    if (available_width <= ellipsis_width) {
      return "";
    }

    // Binary search for the longest prefix that fits within the available
    // width.
    int low = 0, high = event_name.length(), fit_len = 0;
    while (low <= high) {
      const int mid = std::midpoint(low, high);
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
  event_data.try_emplace(kEventSelectedPid,
                         timeline_data_.entry_pids[event_index]);
  auto& args = timeline_data_.entry_args[event_index];
  if (auto it = args.find("uid"); it != args.end()) {
    event_data.try_emplace(kEventSelectedUid, it->second);
  }
  event_callback_(kEventSelected, event_data);
}

void Timeline::NavigateToEvent(int event_index) {
  if (event_index < 0 ||
      event_index >= timeline_data_.entry_start_times.size() ||
      event_index >= timeline_data_.entry_total_times.size()) {
    LOG(ERROR) << "Invalid event index: " << event_index;
    return;
  }

  selected_event_index_ = event_index;

  const Microseconds start = timeline_data_.entry_start_times[event_index];
  const Microseconds end =
      start + timeline_data_.entry_total_times[event_index];
  const Microseconds event_duration =
      timeline_data_.entry_total_times[event_index];
  // When navigating to an event, set the visible duration to 20 times the
  // event's duration to provide context around the event. Clamp the
  // duration between 10ms and 5s to prevent zooming in too far on
  // short events or zooming out too far on long events.
  const Microseconds duration =
      std::clamp(event_duration * kEventNavigationZoomFactor,
                 kEventNavigationMinDurationMicros,
                 kEventNavigationMaxDurationMicros);
  const Microseconds center = std::midpoint(start, end);
  TimeRange new_range = {center - duration / 2.0, center + duration / 2.0};
  ConstrainTimeRange(new_range);

  SetVisibleRange(new_range, /*animate=*/true);

  EmitEventSelected(event_index);
}

void Timeline::CalculateBezierControlPoints(float start_x, float start_y,
                                            float end_x, float end_y,
                                            ImVec2& cp0, ImVec2& cp1) {
  const float dist = std::abs(end_x - start_x) * 0.5f;
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
  ConstrainTimeRange(new_range);

  // Update the target of the animated visible range. The timeline will animate
  // towards this new time.
  SetVisibleRange(new_range, /*animate=*/true);
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
  ConstrainTimeRange(new_range);

  // Update the target of the animated visible range. The timeline will animate
  // towards this new zoom level.
  SetVisibleRange(new_range, /*animate=*/true);
}

double Timeline::px_per_time_unit() const {
  const Pixel timeline_width =
      ImGui::GetContentRegionAvail().x - label_width_ - kTimelinePaddingRight;
  return px_per_time_unit(timeline_width);
}

double Timeline::px_per_time_unit(Pixel timeline_width) const {
  const Microseconds view_duration = visible_range_->duration();
  if (view_duration > 0 && timeline_width > 0) {
    return static_cast<double>(timeline_width) / view_duration;
  } else {
    return 0.0;
  }
}

// Draws the timeline ruler. This includes the main horizontal line,
// vertical tick marks indicating time intervals, and their corresponding time
// labels.
void Timeline::DrawRuler(Pixel timeline_width, Pixel viewport_bottom) {
  if (ImGui::BeginTable("Ruler", 2, kImGuiTableFlags)) {
    ImGui::TableSetupColumn("Labels", ImGuiTableColumnFlags_WidthFixed,
                            label_width_);
    ImGui::TableSetupColumn("Timeline", ImGuiTableColumnFlags_WidthStretch);
    ImGui::TableNextRow();
    ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0,
                           ImGui::GetColorU32(ImGuiCol_WindowBg));

    ImGui::TableNextColumn();
    ImGui::TableNextColumn();

    const ImVec2 pos = ImGui::GetCursorScreenPos();
    ImDrawList* const draw_list = ImGui::GetWindowDrawList();

    const double px_per_time_unit_val = px_per_time_unit(timeline_width);
    if (px_per_time_unit_val > 0) {
      // Draw horizontal line
      const Pixel line_y = pos.y + kRulerHeight;
      draw_list->AddLine(ImVec2(pos.x, line_y),
                         ImVec2(pos.x + timeline_width, line_y),
                         kRulerLineColor);

      const Microseconds min_time_interval =
          kMinTickDistancePx / px_per_time_unit_val;
      const Microseconds tick_interval =
          CalculateNiceInterval(min_time_interval);
      const Pixel major_tick_dist_px = tick_interval * px_per_time_unit_val;

      const Microseconds view_start = visible_range().start();
      const Microseconds trace_start = data_time_range_.start();

      const Microseconds view_start_relative = view_start - trace_start;
      const Microseconds first_tick_time_relative =
          std::floor(view_start_relative / tick_interval) * tick_interval;

      const Pixel minor_tick_dist_px =
          major_tick_dist_px / static_cast<float>(kMinorTickDivisions);

      Microseconds t_relative = first_tick_time_relative;
      Pixel x =
          TimeToScreenX(t_relative + trace_start, pos.x, px_per_time_unit_val);

      for (;; t_relative += tick_interval, x += major_tick_dist_px) {
        if (x > pos.x + timeline_width + kRulerScreenBuffer) {
          break;
        }

        // Draw major tick.
        if (x >= pos.x - kRulerScreenBuffer) {
          // Draw major tick.
          draw_list->AddLine(ImVec2(x, pos.y), ImVec2(x, line_y),
                             kRulerLineColor);

          // Draw vertical line across the tracks.
          draw_list->AddLine(ImVec2(x, line_y), ImVec2(x, viewport_bottom),
                             kTraceVerticalLineColor);

          const std::string time_label_text = FormatTime(t_relative);
          ImGui::PushFont(fonts::label_small);
          draw_list->AddText(ImVec2(x + kRulerTextPadding, pos.y),
                             kRulerTextColor, time_label_text.c_str());
          ImGui::PopFont();
        }

        // Draw minor ticks for the current interval.
        for (int i = 1; i < kMinorTickDivisions; ++i) {
          const Pixel minor_x = x + i * minor_tick_dist_px;
          if (minor_x > pos.x + timeline_width + kRulerScreenBuffer) {
            break;
          }
          if (minor_x >= pos.x - kRulerScreenBuffer) {
            draw_list->AddLine(ImVec2(minor_x, line_y - kRulerMinorTickHeight),
                               ImVec2(minor_x, line_y), kRulerLineColor);
          }
        }
      }
    }

    // Reserve space for the ruler
    ImGui::Dummy(ImVec2(0.0f, kRulerHeight + ImGui::GetStyle().CellPadding.y));
    ImGui::EndTable();
  }
}

void Timeline::DrawEventName(absl::string_view event_name,
                             const EventRect& event_rect,
                             ImDrawList* absl_nonnull draw_list) const {
  const float available_width = event_rect.right - event_rect.left;

  if (available_width >= kMinTextWidth) {
    const std::string text_display =
        GetTextForDisplay(event_name, available_width);

    if (!text_display.empty()) {
      const ImVec2 text_pos = CalculateEventTextRect(text_display, event_rect);

      // Push a clipping rectangle to ensure the text is only drawn within the
      // bounds of the event_rect. This prevents text from overflowing visually.
      draw_list->PushClipRect(ImVec2(event_rect.left, event_rect.top),
                              ImVec2(event_rect.right, event_rect.bottom));
      draw_list->AddText(text_pos, kDefaultTextColor, text_display.c_str());
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

    const bool is_hovered = ImGui::IsMouseHoveringRect(
        ImVec2(rect.left, rect.top), ImVec2(rect.right, rect.bottom));

    const float corner_rounding =
        is_hovered ? kHoverCornerRounding : kCornerRounding;

    const ImU32 event_color = GetColorForId(event_name);
    draw_list->AddRectFilled(ImVec2(rect.left, rect.top),
                             ImVec2(rect.right, rect.bottom), event_color,
                             corner_rounding, kImDrawFlags);
    if (is_hovered) {
      // Draw a semi-transparent overlay when the event is hovered.
      draw_list->AddRectFilled(ImVec2(rect.left, rect.top),
                               ImVec2(rect.right, rect.bottom), kHoverMaskColor,
                               corner_rounding, kImDrawFlags);

      ImGui::SetTooltip(
          "%s (%s)", event_name.c_str(),
          FormatTime(timeline_data_.entry_total_times[event_index]).c_str());

      // ImGui uses 0 to represent the left mouse button, as defined in the
      // ImGuiMouseButton enum. We check if the left mouse button was clicked.
      if (ImGui::IsMouseClicked(0)) {
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
      }
    }

    if (selected_event_index_ == event_index) {
      // Draw a border around the selected event.
      draw_list->AddRect(ImVec2(rect.left, rect.top),
                         ImVec2(rect.right, rect.bottom), kSelectedBorderColor,
                         corner_rounding, kImDrawFlags,
                         kSelectedBorderThickness);
    }

    DrawEventName(event_name, rect, draw_list);
  }
}

void Timeline::DrawEventsForLevel(int group_index,
                                  absl::Span<const int> event_indices,
                                  double px_per_time_unit, int level_in_group,
                                  const ImVec2& pos, const ImVec2& max) {
  ImDrawList* const draw_list = ImGui::GetWindowDrawList();
  if (!draw_list) {
    return;
  }

  for (int event_index : event_indices) {
    if (event_index < 0 ||
        event_index >= timeline_data_.entry_start_times.size() ||
        event_index >= timeline_data_.entry_total_times.size()) {
      // Should not happen if data is well-formed, but good to be safe.
      continue;
    }
    const Microseconds start = timeline_data_.entry_start_times[event_index];
    const Microseconds end =
        start + timeline_data_.entry_total_times[event_index];

    const EventRect rect = CalculateEventRect(
        start, end, pos.x, pos.y, px_per_time_unit, level_in_group, max.x);

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
    const double val = data.values[index];

    const Pixel x = mouse_pos.x;
    const Pixel y = pos.y + height - (val - data.min_value) * y_ratio;

    // Draw circle
    draw_list->AddCircleFilled(ImVec2(x, y), kPointRadius, kWhiteColor);
    draw_list->AddCircle(ImVec2(x, y), kPointRadius, kBlackColor);

    // Draw tooltip for current counter point's value and timestamp
    ImGui::SetTooltip(kCounterTooltipFormat, FormatTime(mouse_time).c_str(),
                      val);

    // ImGui uses 0 to represent the left mouse button, as defined in the
    // ImGuiMouseButton enum. We check if the left mouse button was clicked.
    if (ImGui::IsMouseClicked(0)) {
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

  // If all counter values are the same, draw a single horizontal line
  // vertically centered in the track.
  // Also avoid division by zero.
  if (value_range == 0) {
    const Pixel y = pos.y + height / 2.0f;
    const Pixel x_start =
        TimeToScreenX(data.timestamps.front(), pos.x, px_per_time_unit_val);
    const Pixel x_end =
        TimeToScreenX(data.timestamps.back(), pos.x, px_per_time_unit_val);
    draw_list->AddLine(ImVec2(x_start, y), ImVec2(x_end, y),
                       kCounterTrackColor);
    return;
  }

  const float y_ratio = height / value_range;
  const Pixel y_base = pos.y + height;

  // Calculate the coordinates of the first point.
  ImVec2 p1(TimeToScreenX(data.timestamps[0], pos.x, px_per_time_unit_val),
            y_base - (data.values[0] - data.min_value) * y_ratio);

  for (size_t i = 1; i < data.timestamps.size(); ++i) {
    // Calculate the coordinates of the next point.
    ImVec2 p2(TimeToScreenX(data.timestamps[i], pos.x, px_per_time_unit_val),
              y_base - (data.values[i] - data.min_value) * y_ratio);

    draw_list->AddLine(p1, p2, kCounterTrackColor);
    // Reuse p2 as the start point for the next segment to avoid re-calculation.
    p1 = p2;
  }

  if (selected_group_index_ == group_index && selected_counter_index_ != -1 &&
      selected_counter_index_ < data.timestamps.size()) {
    Microseconds ts = data.timestamps[selected_counter_index_];
    double val = data.values[selected_counter_index_];
    Pixel x = TimeToScreenX(ts, pos.x, px_per_time_unit_val);
    Pixel y = pos.y + height - (val - data.min_value) * y_ratio;

    draw_list->AddCircleFilled(ImVec2(x, y), kPointRadius, kWhiteColor);
    draw_list->AddCircle(ImVec2(x, y), kPointRadius, kSelectedBorderColor,
                         /*num_segments=*/0, /*thickness=*/2.0f);
  }

  if (ImGui::IsWindowHovered()) {
    DrawCounterTooltip(group_index, data, px_per_time_unit_val, pos, height,
                       y_ratio, draw_list);
  }
}

void Timeline::DrawGroup(int group_index, double px_per_time_unit_val) {
  const Group& group = timeline_data_.groups[group_index];
  const int start_level = group.start_level;
  int end_level = (group_index + 1 < timeline_data_.groups.size())
                      ? timeline_data_.groups[group_index + 1].start_level
                      // If this is the last group, the end level is the total
                      // number of levels.
                      : timeline_data_.events_by_level.size();
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
  const Pixel group_height = group.type == Group::Type::kCounter
                                 ? kCounterTrackHeight
                                 : std::max(1, end_level - start_level) *
                                       (kEventHeight + kEventPaddingBottom);
  // Groups might have the same name. We add the index of the group to the ID
  // to ensure each ImGui::BeginChild call has a unique ID, otherwise ImGui
  // might ignore later calls with the same name.
  const std::string timeline_child_id =
      absl::StrCat("TimelineChild_", group.name, "_", group_index);

  // Calculate level Y positions regardless of whether the child window is
  // visible. This ensures that flow lines connecting to off-screen groups are
  // drawn correctly.
  const ImVec2 pos = ImGui::GetCursorScreenPos();
  for (int level = start_level; level < end_level; ++level) {
    if (level < level_y_positions_.size()) {
      level_y_positions_[level] =
          pos.y + (level - start_level) * (kEventHeight + kEventPaddingBottom) +
          kEventHeight * 0.5f;
    }
  }

  if (ImGui::BeginChild(timeline_child_id.c_str(), ImVec2(0, group_height),
                        ImGuiChildFlags_None, kLaneFlags)) {
    const ImVec2 max = ImGui::GetContentRegionMax();

    if (group.type == Group::Type::kCounter) {
      const auto it =
          timeline_data_.counter_data_by_group_index.find(group_index);
      if (it != timeline_data_.counter_data_by_group_index.end()) {
        DrawCounterTrack(group_index, it->second, px_per_time_unit_val, pos,
                         group_height);
      }
    } else if (group.type == Group::Type::kFlame) {
      for (int level = start_level; level < end_level; ++level) {
        // This is a sanity check to ensure the level is within the bounds of
        // events_by_level.
        if (level < timeline_data_.events_by_level.size()) {
          // TODO: b/453676716 - Add boundary test cases for this function.
          DrawEventsForLevel(group_index, timeline_data_.events_by_level[level],
                             px_per_time_unit_val,
                             /*level_in_group=*/level - start_level, pos, max);
        }
      }
    }
  }
  ImGui::EndChild();

  if (group_index < timeline_data_.groups.size() - 1) {
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    float line_y = ImGui::GetItemRectMax().y + ImGui::GetStyle().CellPadding.y;
    draw_list->AddLine(ImVec2(viewport->Pos.x + label_width_, line_y),
                       ImVec2(viewport->Pos.x + viewport->Size.x, line_y),
                       kLightGrayColor);
  }
}

void Timeline::DrawSingleFlow(const FlowLine& flow, Pixel timeline_x_start,
                              double px_per_time, ImDrawList* draw_list) {
  if (flow.source_level >= level_y_positions_.size() ||
      flow.target_level >= level_y_positions_.size()) {
    return;
  }

  const float start_y = level_y_positions_[flow.source_level];
  const float end_y = level_y_positions_[flow.target_level];

  if (start_y == -FLT_MAX || end_y == -FLT_MAX) {
    return;
  }

  const float start_x =
      TimeToScreenX(flow.source_ts, timeline_x_start, px_per_time);
  const float end_x =
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
}

void Timeline::DrawFlows(Pixel timeline_width) {
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
            DrawSingleFlow(flow, timeline_x_start, px_per_time, draw_list);
          }
        }
      }
    }
  } else {
    for (const auto& flow : timeline_data_.flow_lines) {
      if (visible_flow_categories_.contains(static_cast<int>(flow.category))) {
        DrawSingleFlow(flow, timeline_x_start, px_per_time, draw_list);
      }
    }
  }

  draw_list->PopClipRect();
}

void Timeline::DrawSelectedTimeRange(const TimeRange& range,
                                     Pixel timeline_width,
                                     double px_per_time_unit_val) {
  const ImGuiViewport* viewport = ImGui::GetMainViewport();
  const Pixel timeline_x_start = viewport->Pos.x + label_width_;

  const Pixel time_range_x_start =
      TimeToScreenX(range.start(), timeline_x_start, px_per_time_unit_val);
  const Pixel time_range_x_end =
      TimeToScreenX(range.end(), timeline_x_start, px_per_time_unit_val);
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

    const float rect_y_min = viewport->Pos.y;
    const float rect_y_max = viewport->Pos.y + viewport->Size.y;
    const float rect_y_mid = (rect_y_min + rect_y_max) * 0.5f;

    // Draw the top half with a lighter color to keep the timeline content
    // visible.
    draw_list->AddRectFilled(ImVec2(clipped_x_start, rect_y_min),
                             ImVec2(clipped_x_end, rect_y_mid),
                             kSelectedTimeRangeTopColor);

    // Apply the gradient only to the bottom half of the timeline.
    // Increase the opacity of the bottom part to make the text area less
    // transparent and the text more visible.
    draw_list->AddRectFilledMultiColor(
        ImVec2(clipped_x_start, rect_y_mid), ImVec2(clipped_x_end, rect_y_max),
        kSelectedTimeRangeTopColor, kSelectedTimeRangeTopColor,
        kSelectedTimeRangeBottomColor, kSelectedTimeRangeBottomColor);

    // Only draw the border if the edge of the time range is visible.
    if (time_range_x_start >= timeline_x_start) {
      draw_list->AddLine(ImVec2(time_range_x_start, rect_y_min),
                         ImVec2(time_range_x_start, rect_y_max),
                         kSelectedTimeRangeBorderColor);
    }
    if (time_range_x_end <= timeline_x_start + timeline_width) {
      draw_list->AddLine(ImVec2(time_range_x_end, rect_y_min),
                         ImVec2(time_range_x_end, rect_y_max),
                         kSelectedTimeRangeBorderColor);
    }

    const std::string text = FormatTime(range.duration());
    const ImVec2 text_size = ImGui::CalcTextSize(text.c_str());
    // Only draw the text if the text fits within the selected time range.
    if (clipped_x_end - clipped_x_start > text_size.x) {
      const float text_x =
          clipped_x_start + (clipped_x_end - clipped_x_start - text_size.x) / 2;
      // Move the text up a little bit to avoid being too close to the bottom
      // edge.
      const float text_y =
          rect_y_max - text_size.y - kSelectedTimeRangeTextBottomPadding;

      DrawDeleteButton(draw_list, ImVec2(text_x, text_y), text_size, range);
      draw_list->AddText(ImVec2(text_x, text_y), kBlackColor, text.c_str());
    }
  }
}

void Timeline::DrawDeleteButton(ImDrawList* draw_list, const ImVec2& text_pos,
                                const ImVec2& text_size,
                                const TimeRange& range) {
  const float button_size = kCloseButtonSize;
  const float padding = kCloseButtonPadding;

  const ImVec2 button_pos(text_pos.x + text_size.x + padding,
                          text_pos.y + (text_size.y - button_size) / 2.0f);

  const ImVec2 text_min = text_pos;
  const ImVec2 text_max(text_pos.x + text_size.x, text_pos.y + text_size.y);

  const ImVec2 button_min = button_pos;
  const ImVec2 button_max(button_pos.x + button_size,
                          button_pos.y + button_size);

  // Expand the hover area to include both the text and the button, with a
  // small margin.
  ImVec2 hover_min(std::min(text_min.x, button_min.x),
                   std::min(text_min.y, button_min.y));
  ImVec2 hover_max(std::max(text_max.x, button_max.x),
                   std::max(text_max.y, button_max.y));

  hover_min.x -= 2.0f;
  hover_min.y -= 2.0f;
  hover_max.x += 2.0f;
  hover_max.y += 2.0f;

  // If the mouse is hovering over the text area, draw the button.
  if (ImGui::IsMouseHoveringRect(hover_min, hover_max)) {
    ImU32 button_color = kCloseButtonColor;

    // If the mouse is hovering over the button, change the color to the
    // hover color. Also, if the mouse is clicked on the button, remove the
    // range from the list of selected time ranges.
    if (ImGui::IsMouseHoveringRect(button_min, button_max)) {
      button_color = kCloseButtonHoverColor;
      ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
      // ImGui uses 0 to represent the left mouse button.
      // If the mouse is clicked on the button, remove the range from the list
      // of selected time ranges.
      if (ImGui::IsMouseClicked(0)) {
        auto it = absl::c_find(selected_time_ranges_, range);
        if (it != selected_time_ranges_.end()) {
          selected_time_ranges_.erase(it);
        }
      }
    }

    const ImVec2 center(button_min.x + button_size / 2.0f,
                        button_min.y + button_size / 2.0f);
    draw_list->AddCircleFilled(center, button_size / 2.0f, button_color);

    const float x_radius = button_size * 0.25f;
    draw_list->AddLine(ImVec2(center.x - x_radius, center.y - x_radius),
                       ImVec2(center.x + x_radius, center.y + x_radius),
                       kWhiteColor);
    draw_list->AddLine(ImVec2(center.x - x_radius, center.y + x_radius),
                       ImVec2(center.x + x_radius, center.y - x_radius),
                       kWhiteColor);
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
                          px_per_time_unit_val);
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

  // Cancel selection
  if (ImGui::IsKeyPressed(ImGuiKey_Escape)) {
    if (is_selecting_) {
      is_selecting_ = false;
      is_dragging_ = false;
      current_selected_time_range_.reset();
    }
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

  const float horizontal_pan_delta =
      io.KeyShift ? io.MouseWheel : io.MouseWheelH;
  const float vertical_scroll_delta =
      io.KeyShift ? io.MouseWheelH : io.MouseWheel;

  if (horizontal_pan_delta != 0.0f) Pan(horizontal_pan_delta);
  if (vertical_scroll_delta != 0.0f) Scroll(vertical_scroll_delta);

  return true;
}

void Timeline::HandleEventDeselection() {
  // If an event was selected, and the user clicks on an empty area
  // (i.e., not on any event), deselect the event.
  if ((selected_event_index_ != -1 || selected_group_index_ != -1) &&
      ImGui::IsMouseClicked(0) &&
      ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows) &&
      !event_clicked_this_frame_) {
    selected_event_index_ = -1;
    selected_group_index_ = -1;
    selected_counter_index_ = -1;

    EventData event_data;
    event_data[std::string(kEventSelectedIndex)] = -1;
    event_data[std::string(kEventSelectedName)] = std::string("");
    event_data[std::string(kEventSelectedStart)] = 0.0;
    event_data[std::string(kEventSelectedDuration)] = 0.0;
    event_data[std::string(kEventSelectedStartFormatted)] = std::string("");
    event_data[std::string(kEventSelectedDurationFormatted)] = std::string("");

    event_callback_(kEventSelected, event_data);
  }
}

bool Timeline::HandleMouse() {
  const ImRect timeline_area = GetTimelineArea();
  const bool is_mouse_over_timeline =
      ImGui::IsMouseHoveringRect(timeline_area.Min, timeline_area.Max);

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

void Timeline::HandleMouseDown(float timeline_origin_x) {
  // ImGui uses 0 to represent the left mouse button, as defined in the
  // ImGuiMouseButton enum. We check if the left mouse button was clicked.
  if (ImGui::IsMouseClicked(0) && !event_clicked_this_frame_) {
    is_dragging_ = true;
    ImGuiIO& io = ImGui::GetIO();
    is_selecting_ = io.KeyShift;
    if (is_selecting_) {
      const double px_per_time = px_per_time_unit();
      drag_start_time_ =
          PixelToTime(io.MousePos.x - timeline_origin_x, px_per_time);
      current_selected_time_range_ =
          TimeRange(drag_start_time_, drag_start_time_);
    }
  }
}

void Timeline::HandleMouseDrag(float timeline_origin_x) {
  // ImGui uses 0 to represent the left mouse button, as defined in the
  // ImGuiMouseButton enum. We check if the left mouse button was clicked.
  if (ImGui::IsMouseDown(0)) {
    ImGuiIO& io = ImGui::GetIO();
    if (is_selecting_) {
      const double px_per_time = px_per_time_unit();
      Microseconds current_time =
          PixelToTime(io.MousePos.x - timeline_origin_x, px_per_time);
      current_selected_time_range_ =
          TimeRange(std::min(drag_start_time_, current_time),
                    std::max(drag_start_time_, current_time));
    } else {
      Pan(-io.MouseDelta.x);
      Scroll(-io.MouseDelta.y);
    }
  }
}

void Timeline::HandleMouseRelease() {
  if (ImGui::IsMouseReleased(0)) {
    is_dragging_ = false;
    is_selecting_ = false;
    if (current_selected_time_range_ &&
        current_selected_time_range_->duration() > 0) {
      selected_time_ranges_.push_back(*current_selected_time_range_);
    }
    current_selected_time_range_.reset();
  }
}

ImRect Timeline::GetTimelineArea() const {
  const ImVec2 window_pos = ImGui::GetWindowPos();
  const ImVec2 content_min = ImGui::GetWindowContentRegionMin();
  const Pixel start_x = window_pos.x + content_min.x + label_width_;
  const Pixel start_y = window_pos.y + content_min.y;

  const Pixel timeline_width =
      ImGui::GetContentRegionAvail().x - label_width_ - kTimelinePaddingRight;

  const Pixel end_x = start_x + timeline_width;
  const Pixel end_y = window_pos.y + ImGui::GetWindowHeight();

  return {start_x, start_y, end_x, end_y};
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

// This function is called when the search query changes. It re-filters and
// sorts the event indices based on the current search query.
void Timeline::RecomputeSearchResults() {
  sorted_search_results_.clear();
  current_search_result_index_ = -1;
  if (search_query_lower_.empty()) {
    return;
  }

  for (int i = 0; i < timeline_data_.entry_names.size(); ++i) {
    if (absl::StrContains(absl::AsciiStrToLower(timeline_data_.entry_names[i]),
                          search_query_lower_)) {
      EventId event_id = timeline_data_.entry_event_ids[i];
      sorted_search_results_.push_back(
          {event_id,
            timeline_data_.entry_levels[i],
            timeline_data_.entry_start_times[i],
            timeline_data_.entry_total_times[i],
            timeline_data_.entry_pids[i],
            timeline_data_.entry_tids[i]});
    }
  }
  // Sort shallow results by start time, to have some order.
  absl::c_sort(sorted_search_results_, [&](const auto& a, const auto& b) {
    return std::tie(a.pid, a.tid, a.level, a.start_time) <
           std::tie(b.pid, b.tid, b.level, b.start_time);
  });

  EventData event_data;
  event_data.try_emplace(kSearchEventsQuery, search_query_lower_);
  event_callback_(kSearchEvents, event_data);
  if (!sorted_search_results_.empty()) {
    NavigateToNextSearchResult();
  }
}

void Timeline::NavigateToNextSearchResult() {
  if (sorted_search_results_.empty()) return;
  current_search_result_index_++;
  if (current_search_result_index_ >= sorted_search_results_.size()) {
    current_search_result_index_ = 0;
  }
  const auto& result = sorted_search_results_[current_search_result_index_];
  EventId event_id = result.event_id;
  auto it = absl::c_find(timeline_data_.entry_event_ids, event_id);
  if (it != timeline_data_.entry_event_ids.end()) {
    NavigateToEvent(std::distance(timeline_data_.entry_event_ids.begin(), it));
  } else {
    pending_navigation_event_id_ = event_id;
    // If event is not in current data, zoom to its time range to trigger load.
    const Microseconds start = result.start_time;
    const Microseconds event_duration = result.duration;
    const Microseconds end = start + event_duration;
    const Microseconds duration =
        std::clamp(event_duration * kEventNavigationZoomFactor,
                   kEventNavigationMinDurationMicros,
                   kEventNavigationMaxDurationMicros);
    const Microseconds center = std::midpoint(start, end);
    TimeRange new_range = {center - duration / 2.0, center + duration / 2.0};
    ConstrainTimeRange(new_range);
    SetVisibleRange(new_range, /*animate=*/true);
  }
}

void Timeline::NavigateToPrevSearchResult() {
  if (sorted_search_results_.empty()) return;
  current_search_result_index_--;
  if (current_search_result_index_ < 0) {
    current_search_result_index_ = sorted_search_results_.size() - 1;
  }
  const auto& result = sorted_search_results_[current_search_result_index_];
  EventId event_id = result.event_id;
  auto it = absl::c_find(timeline_data_.entry_event_ids, event_id);
  if (it != timeline_data_.entry_event_ids.end()) {
    NavigateToEvent(std::distance(timeline_data_.entry_event_ids.begin(), it));
  } else {
    // TODO(jonahweaver): Remove this section once deep search is implemented.
    // Expected behavior is that the event might not be loaded yet, and might
    // still be loading once navigated to.
    pending_navigation_event_id_ = event_id;
    // If event is not in current data, zoom to its time range to trigger load.
    const Microseconds start = result.start_time;
    const Microseconds event_duration = result.duration;
    const Microseconds end = start + event_duration;
    const Microseconds duration =
        std::clamp(event_duration * kEventNavigationZoomFactor,
                   kEventNavigationMinDurationMicros,
                   kEventNavigationMaxDurationMicros);
    const Microseconds center = std::midpoint(start, end);
    TimeRange new_range = {center - duration / 2.0, center + duration / 2.0};
    ConstrainTimeRange(new_range);
    SetVisibleRange(new_range, /*animate=*/true);
  }
}

}  // namespace traceviewer
