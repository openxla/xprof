#ifndef PERFTOOLS_ACCELERATORS_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TIMELINE_TIMELINE_H_
#define PERFTOOLS_ACCELERATORS_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TIMELINE_TIMELINE_H_

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "xprof/frontend/app/components/trace_viewer_v2/animation.h"
#include "xprof/frontend/app/components/trace_viewer_v2/event_data.h"
#include "xprof/frontend/app/components/trace_viewer_v2/timeline/constants.h"
#include "xprof/frontend/app/components/trace_viewer_v2/timeline/time_range.h"
#include "xprof/frontend/app/components/trace_viewer_v2/trace_helper/trace_event.h"
#include "absl/base/nullability.h"
#include "absl/functional/any_invocable.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "third_party/dear_imgui/imgui.h"

namespace traceviewer {

// Represents a rectangle on the screen.
struct EventRect {
  Pixel left = 0.0f;
  Pixel top = 0.0f;
  Pixel right = 0.0f;
  Pixel bottom = 0.0f;
};

// Represents a grouping of timeline tracks, such as threads or processes.
struct Group {
  std::string name;
  int start_level = 0;
  int nesting_level = 0;
  // TODO - b/444029726: Add other fields like expanded, hidden
};

// Holds all the data required to render a flame chart, including event timing,
// grouping information, and mappings between levels and events.
struct FlameChartTimelineData {
  std::vector<int> entry_levels;
  std::vector<Microseconds> entry_total_times;
  std::vector<Microseconds> entry_start_times;
  std::vector<std::string> entry_names;
  std::vector<Group> groups;
  std::vector<std::vector<int>> events_by_level;
};

// Renders an interactive timeline visualization for trace events, handling
// zooming, panning, and rendering of events grouped into lanes.
class Timeline {
 public:
  // A callback function to handle events from the timeline. The first argument
  // is the event type string. The second argument, EventData, is the payload
  // dispatched as the `detail` of a `CustomEvent` on the `window` object.
  // The callback is expected to be lightweight and non-blocking, as it will be
  // called on the main thread.
  using EventCallback =
      absl::AnyInvocable<void(absl::string_view, const EventData&) const>;

  Timeline() = default;
  // This is necessary because MockTimeline in the tests inherits from Timeline.
  virtual ~Timeline() = default;

  // The provided callback is stored and invoked during the lifetime of this
  // `Timeline` instance. Any captured references must outlive the `Timeline`
  // instance.
  void set_event_callback(EventCallback callback) {
    event_callback_ = std::move(callback);
  }

  // Sets the visible time range. If animate is true, the transition to the
  // new range will be animated, otherwise it will snap to the new time range.
  // Animation is useful for smoothing out transitions caused by user actions
  // like zooming to a selection.
  void SetVisibleRange(const TimeRange& range, bool animate = false);
  const TimeRange& visible_range() const { return *visible_range_; }

  const std::optional<TimeRange>& selected_time_range() const {
    return selected_time_range_;
  }

  void set_data_time_range(const TimeRange& range) { data_time_range_ = range; }

  void set_timeline_data(FlameChartTimelineData data) {
    timeline_data_ = std::move(data);
  }
  const FlameChartTimelineData& timeline_data() const { return timeline_data_; }

  void Draw();

  // Calculates the screen coordinates of the rectangle for an event.
  EventRect CalculateEventRect(Microseconds start, Microseconds end,
                               Pixel screen_x_offset, Pixel screen_y_offset,
                               double px_per_time_unit, int level_in_group,
                               Pixel timeline_width) const;

  // Calculates the top-left screen coordinates for the event name text.
  ImVec2 CalculateEventTextRect(absl::string_view event_name,
                                const EventRect& event_rect) const;

  // Returns text truncated with ellipsis if it's wider than available_width.
  std::string GetTextForDisplay(absl::string_view event_name,
                                float available_width) const;

  // Converts a pixel offset relative to the start of the visible range to a
  // time.
  Microseconds PixelToTime(Pixel pixel_offset, double px_per_time_unit) const;

  // Converts a time to a pixel offset relative to the start of the visible
  // range.
  Pixel TimeToPixel(Microseconds time, double px_per_time_unit) const;

  // Converts a time value to an absolute screen X coordinate.
  Pixel TimeToScreenX(Microseconds time, Pixel screen_x_offset,
                      double px_per_time_unit) const;

  void ConstrainTimeRange(TimeRange& range);

  // Navigates to and selects the event with the given index.
  void NavigateToEvent(int event_index);

 protected:
  // Virtual method to allow mocking in tests.
  virtual ImVec2 GetTextSize(absl::string_view text) const {
    return ImGui::CalcTextSize(text.data(), text.data() + text.size());
  }

  // Pans the visible time range by the given pixel amount.
  // This method is virtual to allow derived classes to customize or extend
  // panning behavior.
  virtual void Pan(Pixel pixel_amount);

  // Scrolls the visible time range by the given pixel amount.
  // This method is virtual to allow derived classes to customize or extend
  // panning behavior.
  virtual void Scroll(Pixel pixel_amount);

  // Zooms the visible time range by the given zoom factor.
  // This method is virtual to allow derived classes to customize or extend
  // zooming behavior.
  virtual void Zoom(float zoom_factor);

 private:
  double px_per_time_unit() const;
  double px_per_time_unit(Pixel timeline_width) const;

  // Draws the timeline ruler. `viewport_bottom` is the y-coordinate of the
  // bottom of the viewport, used to draw vertical grid lines across the tracks.
  void DrawRuler(Pixel timeline_width, Pixel viewport_bottom);

  void DrawEventName(absl::string_view event_name, const EventRect& rect,
                     ImDrawList* absl_nonnull draw_list) const;

  void DrawEvent(int event_index, const EventRect& rect,
                 ImDrawList* absl_nonnull draw_list);

  void DrawEventsForLevel(absl::Span<const int> event_indices,
                          double px_per_time_unit, int level_in_group,
                          const ImVec2& pos, const ImVec2& max);

  void DrawGroup(int group_index, double px_per_time_unit_val);

  // Draws the selected time range.
  void DrawSelectedTimeRange(Pixel timeline_width, double px_per_time_unit_val);

  // Handles keyboard input for panning and zooming.
  void HandleKeyboard();

  // Handles mouse wheel input for scrolling.
  void HandleWheel();

  // Handles deselection of events when clicking on an empty area.
  void HandleEventDeselection();

  // Private static constants.
  static constexpr ImGuiWindowFlags kImGuiWindowFlags =
      ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoCollapse |
      ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
      ImGuiWindowFlags_NoMove;
  static constexpr ImGuiTableFlags kImGuiTableFlags =
      ImGuiTableFlags_NoPadOuterX | ImGuiTableFlags_BordersInnerV;
  static constexpr ImGuiWindowFlags kLaneFlags =
      ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse;

  FlameChartTimelineData timeline_data_;
  // TODO - b/444026851: Set the label width based on the real screen width.
  Pixel label_width_ = 250.0f;

  // The visible time range in microseconds in the timeline. It is initialized
  // to {0, 0} by the `TimeRange` default constructor.
  // This range is updated through `SetVisibleRange`.
  // User interactions like panning and zooming also cause updates to this
  // range.
  Animated<TimeRange> visible_range_;
  // The total time range [min_time, max_time] in microseconds of the loaded
  // trace data. This range is set when trace data is processed and used as the
  // boundaries for constraining panning and zooming. It does not change during
  // user interactions like pan or zoom.
  TimeRange data_time_range_ = TimeRange::Zero();

  // The index of the currently selected event, or -1 if no event is selected.
  int selected_event_index_ = -1;
  EventCallback event_callback_ = [](absl::string_view, const EventData&) {};
  // Flag to track if an event was clicked in the current frame. This is used
  // to detect clicks in empty areas for deselection logic.
  bool event_clicked_this_frame_ = false;

  std::optional<TimeRange> selected_time_range_;
};

}  // namespace traceviewer
#endif  // PERFTOOLS_ACCELERATORS_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TIMELINE_TIMELINE_H_
