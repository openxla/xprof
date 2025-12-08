#include "xprof/frontend/app/components/trace_viewer_v2/timeline/timeline.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <numeric>
#include <string>

#include "absl/base/nullability.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "third_party/dear_imgui/imgui.h"
#include "third_party/dear_imgui/imgui_internal.h"
#include "xprof/frontend/app/components/trace_viewer_v2/color/color_generator.h"
#include "xprof/frontend/app/components/trace_viewer_v2/event_data.h"
#include "xprof/frontend/app/components/trace_viewer_v2/helper/time_formatter.h"
#include "xprof/frontend/app/components/trace_viewer_v2/timeline/constants.h"
#include "xprof/frontend/app/components/trace_viewer_v2/timeline/draw_utils.h"
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
}  // namespace

void Timeline::SetVisibleRange(const TimeRange& range, bool animate) {
  if (animate) {
    visible_range_ = range;
  } else {
    visible_range_.snap_to(range);
  }
  if (animate && viewport_changed_callback_) {
    viewport_changed_callback_(visible_range_->start(), visible_range_->end());
  }
}

void Timeline::Draw() {
  event_clicked_this_frame_ = false;

  const ImGuiViewport* viewport = ImGui::GetMainViewport();
  ImGui::SetNextWindowPos(viewport->Pos);
  ImGui::SetNextWindowSize(viewport->Size);
  ImGui::SetNextWindowViewport(viewport->ID);

  ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.f);
  ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, ImVec2(0.0f, 0.0f));
  ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0.0f, 0.0f));

  ImGui::Begin("Timeline viewer", nullptr, kImGuiWindowFlags);

  if (timeline_data_.groups.empty()) {
    DrawLoadingIndicator(viewport);
  }

  const Pixel timeline_width =
      ImGui::GetContentRegionAvail().x - label_width_ - kTimelinePaddingRight;
  const double px_per_time_unit_val = px_per_time_unit(timeline_width);

  DrawRuler(timeline_width, viewport->Pos.y + viewport->Size.y);

  // The tracks are in a child window to allow scrolling independently of the
  // ruler.
  ImGui::BeginChild(
      "Tracks", ImVec2(0, 0), ImGuiChildFlags_None,
      ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoScrollbar);

  ImGui::BeginTable("Timeline", 2, kImGuiTableFlags, ImVec2(0.0f, -FLT_MIN));
  ImGui::TableSetupColumn("Labels", ImGuiTableColumnFlags_WidthFixed,
                          label_width_);
  ImGui::TableSetupColumn("Timeline", ImGuiTableColumnFlags_WidthStretch);

  for (int group_index = 0; group_index < timeline_data_.groups.size();
       ++group_index) {
    const Group& group = timeline_data_.groups[group_index];
    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    // Indent the group name. We add 1 to the nesting level because
    // ImGui::Indent(0) results in a default, potentially large indentation.
    // By adding 1, even top-level groups (nesting_level 0) receive a base
    // indentation of `kIndentSize`, ensuring consistent and controlled visual
    // separation from the left edge of the table column.
    ImGui::Indent((group.nesting_level + 1) * kIndentSize);
    ImGui::TextUnformatted(group.name.c_str());
    ImGui::Unindent((group.nesting_level + 1) * kIndentSize);

    ImGui::TableNextColumn();

    DrawGroup(group_index, px_per_time_unit_val);
  }

  ImGui::EndTable();

  HandleEventDeselection();

  // Handle continuous keyboard and mouse wheel input for timeline navigation.
  // These functions are called every frame to ensure smooth and responsive
  // interaction.
  // The performance impact is fine because HandleKeyboard/HandleWheel() only
  // performs lightweight checks and calculations.
  HandleKeyboard();
  HandleWheel();
  HandleMouse();

  // Keep this at the end.
  // `DrawSelectedTimeRanges` should be called after all other timeline content
  // (events, ruler, etc.) has been drawn. This ensures that the selected time
  // range is rendered on top of everything else within the current ImGui
  // window, without affecting global foreground elements like tooltips.
  DrawSelectedTimeRanges(timeline_width, px_per_time_unit_val);

  ImGui::EndChild();
  ImGui::PopStyleVar();  // ItemSpacing
  ImGui::PopStyleVar();  // CellPadding
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
    // When shifting the start to data_time_range_.start(), ensure the new end
    // does not exceed data_time_range_.end().
    range = {data_time_range_.start(),
             std::min(range.end() + data_time_range_.start() - range.start(),
                      data_time_range_.end())};
  } else if (range.end() > data_time_range_.end()) {
    // When shifting the end to data_time_range_.end(), ensure the new start
    // does not go before data_time_range_.start() by taking the maximum.
    range = {std::max(range.start() - range.end() + data_time_range_.end(),
                      data_time_range_.start()),
             data_time_range_.end()};
  }
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
  const Microseconds duration = visible_range_->duration();
  const Microseconds center = std::midpoint(start, end);
  TimeRange new_range = {center - duration / 2.0, center + duration / 2.0};
  ConstrainTimeRange(new_range);

  SetVisibleRange(new_range, /*animate=*/true);
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
  // If the zoom factor is 1, we don't need to zoom.
  if (zoom_factor == 1.0) return;

  // Clamp the zoom factor to the minimum value. This prevents the time
  // durations (mathmatically) become zero or negative.
  zoom_factor = std::max(zoom_factor, kMinZoomFactor);

  TimeRange new_range = visible_range_.target();
  new_range.Zoom(zoom_factor);
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
          draw_list->AddLine(ImVec2(x, line_y - kRulerTickHeight),
                             ImVec2(x, line_y), kRulerLineColor);
          draw_list->AddLine(ImVec2(x, line_y), ImVec2(x, viewport_bottom),
                             kLightGrayColor);

          const std::string text = FormatTime(t_relative);
          draw_list->AddText(ImVec2(x + kRulerTextPadding, pos.y),
                             kRulerTextColor, text.c_str());
        }

        // Draw minor ticks for the current interval.
        for (int i = 1; i < kMinorTickDivisions; ++i) {
          const Pixel minor_x = x + i * minor_tick_dist_px;
          if (minor_x > pos.x + timeline_width + kRulerScreenBuffer) {
            break;
          }
          if (minor_x >= pos.x - kRulerScreenBuffer) {
            draw_list->AddLine(
                ImVec2(minor_x, line_y - kRulerTickHeight / 2.0f),
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

void Timeline::DrawEvent(int event_index, const EventRect& rect,
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
          auto it = std::find(selected_time_ranges_.begin(),
                              selected_time_ranges_.end(), selected_time_range);
          // Click on the event to select, and click on the same event to
          // de-select.
          if (it != selected_time_ranges_.end()) {
            selected_time_ranges_.erase(it);
          } else {
            selected_time_ranges_.push_back(selected_time_range);
          }
        }

        if (selected_event_index_ != event_index) {
          selected_event_index_ = event_index;

          EventData event_data;

          event_data.insert(
              {std::string(kEventSelectedIndex), selected_event_index_});
          event_data.insert({std::string(kEventSelectedName), event_name});
          event_data.insert(
              {std::string(kEventSelectedStart),
               timeline_data_.entry_start_times[event_index]});
          event_data.insert(
              {std::string(kEventSelectedDuration),
               timeline_data_.entry_total_times[event_index]});
          event_callback_(kEventSelected, event_data);
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

void Timeline::DrawEventsForLevel(absl::Span<const int> event_indices,
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

    DrawEvent(event_index, rect, draw_list);
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
  // Ensure end_level is not less than start_level, to avoid negative height.
  end_level = std::max(start_level, end_level);

  // Calculate group height. Ensure a minimum height of one level to prevent
  // ImGui::BeginChild from auto-resizing, even if a group contains no levels.
  // This is important for parent groups (e.g., a process) that might not
  // contain any event levels directly.
  // TODO: b/453676716 - Add tests for group height calculation.
  const Pixel group_height = std::max(1, end_level - start_level) *
                             (kEventHeight + kEventPaddingBottom);
  // Groups might have the same name. We add the index of the group to the ID
  // to ensure each ImGui::BeginChild call has a unique ID, otherwise ImGui
  // might ignore later calls with the same name.
  const std::string timeline_child_id =
      absl::StrCat("TimelineChild_", group.name, "_", group_index);

  if (ImGui::BeginChild(timeline_child_id.c_str(), ImVec2(0, group_height),
                        ImGuiChildFlags_None, kLaneFlags)) {
    const ImVec2 pos = ImGui::GetCursorScreenPos();
    const ImVec2 max = ImGui::GetContentRegionMax();

    for (int level = start_level; level < end_level; ++level) {
      // This is a sanity check to ensure the level is within the bounds of
      // events_by_level.
      if (level < timeline_data_.events_by_level.size()) {
        // TODO: b/453676716 - Add boundary test cases for this function.
        DrawEventsForLevel(timeline_data_.events_by_level[level],
                           px_per_time_unit_val,
                           /*level_in_group=*/level - start_level, pos, max);
      }
    }
  }
  ImGui::EndChild();

  if (group_index < timeline_data_.groups.size() - 1) {
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    float line_y = ImGui::GetItemRectMax().y + ImGui::GetStyle().CellPadding.y;
    draw_list->AddLine(ImVec2(viewport->Pos.x + label_width_ + 15, line_y),
                       ImVec2(viewport->Pos.x + viewport->Size.x, line_y),
                       kLightGrayColor);
  }
}

void Timeline::DrawSelectedTimeRange(const TimeRange& range,
                                     Pixel timeline_width,
                                     double px_per_time_unit_val) {
  const ImVec2 table_rect_min = ImGui::GetItemRectMin();
  const ImVec2 table_rect_max = ImGui::GetItemRectMax();
  const Pixel timeline_x_start = table_rect_min.x + label_width_;

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
    draw_list->AddRectFilled(ImVec2(clipped_x_start, table_rect_min.y),
                             ImVec2(clipped_x_end, table_rect_max.y),
                             kSelectedTimeRangeColor);
    // Only draw the border if the edge of the time range is visible.
    if (time_range_x_start >= timeline_x_start) {
      draw_list->AddLine(ImVec2(time_range_x_start, table_rect_min.y),
                         ImVec2(time_range_x_start, table_rect_max.y),
                         kSelectedTimeRangeBorderColor);
    }
    if (time_range_x_end <= timeline_x_start + timeline_width) {
      draw_list->AddLine(ImVec2(time_range_x_end, table_rect_min.y),
                         ImVec2(time_range_x_end, table_rect_max.y),
                         kSelectedTimeRangeBorderColor);
    }

    const std::string text = FormatTime(range.duration());
    const ImVec2 text_size = ImGui::CalcTextSize(text.c_str());
    // Only draw the text if the text fits within the selected time range.
    if (clipped_x_end - clipped_x_start > text_size.x) {
      const float text_x =
          clipped_x_start + (clipped_x_end - clipped_x_start - text_size.x) / 2;
      const ImVec2 window_pos = ImGui::GetWindowPos();
      const ImVec2 window_size = ImGui::GetWindowSize();
      const float text_y =
          window_pos.y + window_size.y - text_size.y - kRulerTextPadding;
      draw_list->AddText(ImVec2(text_x, text_y), kRulerTextColor, text.c_str());
    }
  }
}

void Timeline::DrawSelectedTimeRanges(Pixel timeline_width,
                                      double px_per_time_unit_val) {
  for (const TimeRange& selected_time_range : selected_time_ranges_) {
    DrawSelectedTimeRange(selected_time_range, timeline_width,
                          px_per_time_unit_val);
  }

  if (current_selected_time_range_) {
    DrawSelectedTimeRange(*current_selected_time_range_, timeline_width,
                          px_per_time_unit_val);
  }
}

void Timeline::HandleKeyboard() {
  const ImGuiIO& io = ImGui::GetIO();

  // Pan left
  if (ImGui::IsKeyDown(ImGuiKey_A)) {
    float multiplier = GetSpeedMultiplier(io, ImGuiKey_A);
    Pan(-kPanningSpeed * io.DeltaTime * multiplier);
  }
  // Pan right
  if (ImGui::IsKeyDown(ImGuiKey_D)) {
    float multiplier = GetSpeedMultiplier(io, ImGuiKey_D);
    Pan(kPanningSpeed * io.DeltaTime * multiplier);
  }

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
  }
  // Zoom out
  if (ImGui::IsKeyDown(ImGuiKey_S)) {
    float multiplier = GetSpeedMultiplier(io, ImGuiKey_S);
    Zoom(1.0f + kZoomSpeed * io.DeltaTime * multiplier);
  }
}

void Timeline::HandleMouse() {
  // Determine the bounding box for the timeline area.
  const ImVec2 main_window_pos = ImGui::GetWindowPos();
  const ImVec2 content_min = ImGui::GetWindowContentRegionMin();
  const ImVec2 timeline_area_pos(
      main_window_pos.x + content_min.x + label_width_,
      main_window_pos.y + content_min.y);
  const Pixel timeline_width =
      ImGui::GetContentRegionAvail().x - label_width_ - kTimelinePaddingRight;
  const ImRect timeline_area(
      timeline_area_pos, ImVec2(timeline_area_pos.x + timeline_width,
                                main_window_pos.y + ImGui::GetWindowHeight()));

  ImGuiIO& io = ImGui::GetIO();
  const bool is_mouse_over_timeline =
      ImGui::IsMouseHoveringRect(timeline_area.Min, timeline_area.Max);

  if (!is_mouse_over_timeline && !is_dragging_) {
    return;
  }

  if (is_mouse_over_timeline && ImGui::IsMouseClicked(0) && io.KeyShift &&
      !event_clicked_this_frame_) {
    is_dragging_ = true;
    const double px_per_time = px_per_time_unit();
    drag_start_time_ =
        PixelToTime(io.MousePos.x - timeline_area.Min.x, px_per_time);
    current_selected_time_range_ =
        TimeRange(drag_start_time_, drag_start_time_);
  }

  if (is_dragging_) {
    if (ImGui::IsMouseDown(0)) {
      const double px_per_time = px_per_time_unit();
      Microseconds current_time =
          PixelToTime(io.MousePos.x - timeline_area.Min.x, px_per_time);
      current_selected_time_range_ =
          TimeRange(std::min(drag_start_time_, current_time),
                    std::max(drag_start_time_, current_time));
    }

    if (ImGui::IsMouseReleased(0)) {
      is_dragging_ = false;
      if (current_selected_time_range_ &&
          current_selected_time_range_->duration() > 0) {
        selected_time_ranges_.push_back(*current_selected_time_range_);
      }
      current_selected_time_range_.reset();
    }
  }
}

void Timeline::HandleWheel() {
  const ImGuiIO& io = ImGui::GetIO();

  if (io.MouseWheel == 0.0f && io.MouseWheelH == 0.0f) {
    return;
  }

  if (io.KeyCtrl || io.KeySuper) {
    // If the mouse wheel is being used with the control or command key, zoom
    // in or out.
    const float zoom_factor = 1.0f + io.MouseWheel * kMouseWheelZoomSpeed;
    Zoom(zoom_factor);
  } else if (io.MouseWheelH != 0.0f) {
    // Handle horizontal scrolling (e.g., with a trackpad or mouse with
    // horizontal wheel).
    Pan(io.MouseWheelH);
  } else if (io.KeyShift) {
    // If the mouse wheel is being used with the shift key, pan the timeline
    // horizontally.
    Pan(io.MouseWheel);
  } else {
    // Otherwise, scroll the timeline vertically.
    Scroll(io.MouseWheel);
  }
}

void Timeline::HandleEventDeselection() {
  // If an event was selected, and the user clicks on an empty area
  // (i.e., not on any event), deselect the event.
  if (selected_event_index_ != -1 && ImGui::IsMouseClicked(0) &&
      ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows) &&
      !event_clicked_this_frame_) {
    selected_event_index_ = -1;

    EventData event_data;
    event_data[std::string(kEventSelectedIndex)] = -1;
    event_data[std::string(kEventSelectedName)] = std::string("");
    event_data[std::string(kEventSelectedStart)] = 0.0;
    event_data[std::string(kEventSelectedDuration)] = 0.0;

    event_callback_(kEventSelected, event_data);
  }
}

}  // namespace traceviewer
