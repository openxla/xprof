#include "frontend/app/components/trace_viewer_v2/timeline/timeline.h"

#include <algorithm>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "imgui.h"

namespace traceviewer {

void Timeline::DrawTimelinePlayerSync() {
#ifdef __EMSCRIPTEN__
  if (!timeline_player_enabled_) return;

  if (!timeline_data_.groups.empty()) {
    Microseconds visible_duration = visible_range().duration();

    if (current_play_time_ < visible_range().start()) {
      current_play_time_ = visible_range().start();
    }

    if (is_playing_) {
      double delta =
          ImGui::GetIO().DeltaTime * play_speed_ * visible_duration * 0.1;
      current_play_time_ += delta;

      if (current_play_time_ >= visible_range().end()) {
        current_play_time_ = visible_range().end();
        is_playing_ = false;
      }
      if (redraw_callback_) redraw_callback_();
    }

    double current_progress_us = current_play_time_ - visible_range().start();
    if (current_progress_us < 0) current_progress_us = 0;
    if (current_progress_us > visible_duration)
      current_progress_us = visible_duration;

    ImDrawList* fg_list = ImGui::GetForegroundDrawList();
    const ImRect timeline_area = GetTimelineArea();

    // Broadcast stats via event_callback_

    // Only trigger event listener conditionally to avoid JS spam, but always
    // draw tooltips
    bool should_broadcast =
        (current_play_time_ != previous_play_time_) ||
        (visible_range().start() != previous_visible_start_) ||
        (visible_duration != previous_visible_duration_);
    previous_play_time_ = current_play_time_;
    previous_visible_start_ = visible_range().start();
    previous_visible_duration_ = visible_duration;

    EventData payload;
    payload["time"] = current_play_time_;
    payload["currentTime"] = current_progress_us;
    payload["duration"] = (double)visible_duration;
    payload["isPlaying"] = is_playing_;

    std::vector<int> level_to_group(timeline_data_.events_by_level.size(), -1);
    for (size_t i = 0; i < timeline_data_.groups.size(); ++i) {
      for (int l = timeline_data_.groups[i].start_level;
           l < level_to_group.size(); ++l) {
        if (l >= 0) level_to_group[l] = i;
      }
    }

    std::vector<EventData> active_events;
    for (int level = 0; level < timeline_data_.events_by_level.size();
         ++level) {
      for (int event_idx : timeline_data_.events_by_level[level]) {
        double start = timeline_data_.entry_start_times[event_idx];
        double duration = timeline_data_.entry_total_times[event_idx];
        if (current_play_time_ >= start &&
            current_play_time_ <= start + duration) {
          int group_index = level_to_group[level];
          std::string g_name = group_index != -1
                                   ? timeline_data_.groups[group_index].name
                                   : "Unknown";
          if (should_broadcast) {
            EventData ev;
            ev["level"] = level;
            ev["group"] = g_name;
            std::string event_name = timeline_data_.entry_names[event_idx];
            ev["name"] = event_name;
            ev["duration"] = duration;
            active_events.push_back(ev);
          }
        } else if (start > current_play_time_) {
          break;
        }
      }
    }
    payload["events"] = active_events;

    std::vector<EventData> active_counters;
    for (const auto& [idx, counter] :
         timeline_data_.counter_data_by_group_index) {
      if (counter.timestamps.empty()) continue;

      // Ensure playhead intersects the valid drawn bounding area of the counter
      // chart.
      if (current_play_time_ < counter.timestamps.front() ||
          current_play_time_ > counter.timestamps.back()) {
        continue;
      }

      double val = 0.0;
      auto it = std::upper_bound(counter.timestamps.begin(),
                                 counter.timestamps.end(), current_play_time_);
      if (it != counter.timestamps.begin()) {
        size_t index = std::distance(counter.timestamps.begin(), it) - 1;
        val = counter.values[index];
      } else {
        val = counter.values.front();
      }

      if (should_broadcast) {
        EventData c_data;
        c_data["name"] = timeline_data_.groups[idx].name;
        c_data["min"] = counter.min_value;
        c_data["max"] = counter.max_value;
        c_data["value"] = val;
        active_counters.push_back(c_data);
      }

      if (idx < group_visible_.size() && group_visible_[idx]) {
        const double value_range = counter.max_value - counter.min_value;
        double y_ratio =
            value_range == 0 ? 0 : kCounterTrackHeight / value_range;

        Pixel group_y = tracks_start_screen_pos_.y + group_offsets_[idx];
        Pixel y_base = group_y + kCounterTrackHeight;

        Pixel counter_y = y_base - (val - counter.min_value) * y_ratio;
        // Match DrawCounterTrack minimum 1px height
        counter_y = std::min(counter_y, y_base - 1.0f);

        if (group_y > ruler_screen_y_ + kRulerHeight &&
            group_y < timeline_area.Max.y) {
          const Pixel px_per_time = px_per_time_unit();
          Pixel x = TimeToScreenX(current_play_time_, timeline_area.Min.x,
                                  px_per_time);

          // Render actual value precisely formatted
          std::string text =
              absl::StrCat(timeline_data_.groups[idx].name, ": ", val);
          ImVec2 text_size = ImGui::CalcTextSize(text.c_str());

          ImVec2 rect_min(x + 8, counter_y - text_size.y / 2.0f - 2);
          ImVec2 rect_max(x + 8 + text_size.x + 8,
                          counter_y + text_size.y / 2.0f + 2);
          fg_list->AddRectFilled(rect_min, rect_max, IM_COL32(30, 30, 30, 240),
                                 2.0f);
          fg_list->AddRect(rect_min, rect_max, IM_COL32(200, 200, 200, 100),
                           2.0f);
          fg_list->AddText(ImVec2(x + 12, counter_y - text_size.y / 2.0f),
                           IM_COL32(255, 255, 255, 255), text.c_str());
          fg_list->AddCircleFilled(ImVec2(x, counter_y), 3.0f,
                                   IM_COL32(255, 0, 0, 255));
        }
      }
    }
    payload["counters"] = active_counters;

    if (event_callback_ && should_broadcast) {
      event_callback_("timeline-player-sync-backend", payload);
    }
  }
#endif
}

}  // namespace traceviewer
