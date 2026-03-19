#ifndef THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TIMELINE_TIMELINE_H_
#define THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TIMELINE_TIMELINE_H_

#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/any_invocable.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "imgui.h"
#include "imgui_internal.h"
#include "tsl/profiler/lib/context_types.h"
#include "frontend/app/components/trace_viewer_v2/animation.h"
#include "frontend/app/components/trace_viewer_v2/event_data.h"
#include "frontend/app/components/trace_viewer_v2/timeline/constants.h"
#include "frontend/app/components/trace_viewer_v2/timeline/time_range.h"
#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_event.h"

namespace traceviewer {

namespace FlowCategoryFilter {
static constexpr int kAll = -1;
static constexpr int kNone = -2;
}  // namespace FlowCategoryFilter

// Renders an interactive timeline visualization for trace events, handling
// zooming, panning, and rendering of events grouped into lanes.
class Timeline {
 public:
  // Represents a rectangle on the screen.
  struct EventRect {
    Pixel left = 0.0f;
    Pixel top = 0.0f;
    Pixel right = 0.0f;
    Pixel bottom = 0.0f;
  };

  // Layout information for the time range delete button.
  struct DeleteButtonLayout {
    // Screen position where the button should be drawn.
    ImVec2 button_pos;
    // Area that triggers hover effects for the delete button.
    ImRect hover_rect;
    // Whether the button text fits within the available space.
    bool text_fits = false;
  };

  // Holds timing and values for a counter track.
  struct CounterData {
    std::vector<Microseconds> timestamps;
    std::vector<double> values;
    // Minimum value among all data points for scaling.
    double min_value = std::numeric_limits<double>::max();
    // Maximum value among all data points for scaling.
    double max_value = std::numeric_limits<double>::lowest();
  };

  // Represents a grouping of timeline tracks, such as processes, threads, or
  // counters.
  struct Group {
    enum class Type { kFlame, kCounter };
    Type type = Type::kFlame;
    std::string name;
    // The start level of the groups of complete events.
    // For flame groups, we increment the group level by real events' levels.
    // For counter groups, we increment the group level by 1.
    int start_level = 0;
    int nesting_level = 0;
    bool expanded = false;
  };

  struct FlowLine {
    Microseconds source_ts = 0.0;

    Microseconds target_ts = 0.0;

    int source_level = 0;

    int target_level = 0;

    uint32_t color = traceviewer::kBlackColor;
    tsl::profiler::ContextType category = tsl::profiler::ContextType::kGeneric;
  };

  // Holds all the data required to render a flame chart and counter lines,
  // including event timing, grouping information, and mappings between levels
  // and events.
  struct FlameChartTimelineData {
    std::vector<int> entry_levels;
    std::vector<Microseconds> entry_total_times;
    std::vector<Microseconds> entry_start_times;
    std::vector<std::string> entry_names;
    std::vector<EventId> entry_event_ids;
    // TODO: b/474668991 - Check if we can fetch PID and entry args from backend
    // instead of storing them here, to reduce memory usage.
    // Compare latency from network to memory-heavy local storage.
    std::vector<ProcessId> entry_pids;
    std::vector<ThreadId> entry_tids;
    std::vector<std::map<std::string, std::string>> entry_args;
    std::vector<Group> groups;
    // A map from level to a list of event indices at that level.
    // This is used to quickly draw events at a given level.
    // Technically, we can calculate this in the Timeline class, but doing it
    // here saves us from traversing all the events 2 times, though the time
    // complexity are the same. But given there might be tens of thousands
    // events, this optimization is worth it.
    std::vector<std::vector<int>> events_by_level;
    std::vector<FlowLine> flow_lines;
    // Map from event_id to list of flow ids that connect to this event.
    absl::flat_hash_map<EventId, std::vector<std::string>> flow_ids_by_event_id;
    // Map from flow_id to list of flow lines that belong to this flow.
    absl::flat_hash_map<std::string, std::vector<FlowLine>>
        flow_lines_by_flow_id;
    // A map from group index to counter data.
    // We use group index instead of PID as the key because a process (PID) can
    // have multiple counter tracks associated with it. The group index uniquely
    // identifies each track within the `groups` vector.
    std::map<int, CounterData> counter_data_by_group_index;
  };

  // Information about timeline ticks for drawing ruler and grid lines.
  struct TickInfo {
    // Time duration between major ticks.
    Microseconds tick_interval;
    // Pixel distance between major ticks.
    Pixel major_tick_dist_px;
    // Time of the first major tick relative to trace start.
    Microseconds first_tick_time_relative;
  };

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

  // --- Search and Selection ---

  // Sets the search query to highlight events matching the query. Search is
  // case-insensitive.
  void SetSearchQuery(const std::string& query);

  // Sets the search results from the given parsed trace events.
  void SetSearchResults(const ParsedTraceEvents& search_results);

  // Returns search result context.
  size_t get_search_results_count() const {
    return sorted_search_results_.size();
  }
  int get_current_search_result_index() const {
    return current_search_result_index_;
  }

  // Navigation through search results.
  void NavigateToEvent(int event_index);
  void NavigateToNextSearchResult();
  void NavigateToPrevSearchResult();

  // Selection states.
  int selected_event_index() const { return selected_event_index_; }
  int selected_group_index() const { return selected_group_index_; }
  int selected_counter_index() const { return selected_counter_index_; }

  const std::vector<TimeRange>& selected_time_ranges() const {
    return selected_time_ranges_;
  }

  const std::optional<TimeRange>& current_selected_time_range() const {
    return current_selected_time_range_;
  }

  // --- Time Ranges and Viewport ---

  // Sets the visible time range. If animate is true, the transition to the
  // new range will be animated.
  void SetVisibleRange(const TimeRange& range, bool animate = false);
  const TimeRange& visible_range() const { return *visible_range_; }

  // fetched_data_time_range is the range of currently loaded trace data.
  void set_fetched_data_time_range(const TimeRange& range) {
    fetched_data_time_range_ = range;
    // If the last fetch request range is empty, it means we haven't made any
    // incremental loading yet. In this case, we initialize it to the fetched
    // data range to prevent immediate redundant fetches upon the first update.
    if (last_fetch_request_range_.duration() == 0) {
      last_fetch_request_range_ = range;
    }
  }
  const TimeRange& fetched_data_time_range() const {
    return fetched_data_time_range_;
  }

  // data_time_range is the total range of the entire trace.
  void set_data_time_range(const TimeRange& range) { data_time_range_ = range; }
  const TimeRange& data_time_range() const { return data_time_range_; }

  void set_is_incremental_loading(bool is_incremental_loading) {
    is_incremental_loading_ = is_incremental_loading;
  }

  // --- Data and Layout ---

  void set_timeline_data(FlameChartTimelineData data);
  const FlameChartTimelineData& timeline_data() const { return timeline_data_; }

  const std::vector<float>& GetLevelYPositions() const {
    return level_y_positions_;
  }

  void set_mpmd_pipeline_view_enabled(bool enabled) {
    mpmd_pipeline_view_enabled_ = enabled;
  }
  bool mpmd_pipeline_view_enabled() const {
    return mpmd_pipeline_view_enabled_;
  }

  Pixel GetLabelWidth() const { return label_width_; }

  void SetVisibleFlowCategory(int category_id) {
    flow_category_filter_ = category_id;
  }
  void SetVisibleFlowCategories(const std::vector<int>& category_ids);

  // --- Coordinates and Rendering ---

  // Renders the timeline. Should be called every frame.
  void Draw();

  // Calculates the screen coordinates of the rectangle for an event.
  EventRect CalculateEventRect(Microseconds start, Microseconds end,
                               Pixel screen_x_offset, Pixel screen_y_offset,
                               double px_per_time_unit, int level_in_group,
                               Pixel timeline_width, Pixel event_height,
                               Pixel padding_bottom) const;

  // Calculates the top-left screen coordinates for the event name text.
  ImVec2 CalculateEventTextRect(absl::string_view event_name,
                                const EventRect& event_rect) const;

  // Returns text truncated with ellipsis if it's wider than available_width.
  std::string GetTextForDisplay(absl::string_view event_name,
                                float available_width) const;

  // Coordinate conversion helpers.
  Microseconds PixelToTime(Pixel pixel_offset, double px_per_time_unit) const;
  Pixel TimeToPixel(Microseconds time, double px_per_time_unit) const;
  Pixel TimeToScreenX(Microseconds time, Pixel screen_x_offset,
                      double px_per_time_unit) const;

  // Forces the range to stay within data_time_range_.
  void ConstrainTimeRange(TimeRange& range);

  // Calculates tick information based on current zoom level (px_per_time_unit).
  TickInfo CalculateTickInfo(double px_per_time_unit_val) const;

  // Calculates the control points for a cubic Bezier curve used to draw flows.
  static void CalculateBezierControlPoints(float start_x, float start_y,
                                           float end_x, float end_y,
                                           ImVec2& cp0, ImVec2& cp1);

  // Methods exposed for testing.

  // Checks if the visible time range is close to the edge of the loaded data
  // range. If the user pans or zooms to an area where data might soon be
  // needed (i.e., outside the `preserve` range), this function triggers a data
  // fetch request for a larger range (`fetch` range) to ensure data is
  // available before it becomes visible, providing a smoother user experience.
  void MaybeRequestData();

  // Calculates the layout for the delete button and its hover area.
  DeleteButtonLayout GetDeleteButtonLayout(const ImVec2& text_size,
                                           const ImVec2& text_pos,
                                           const ImRect& visible_range_rect,
                                           const ImRect& full_range_rect) const;

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

  // Zooms the visible time range by the given zoom factor, centered around the
  // mouse position, or the center of the visible range if the mouse is outside
  // the trace events area.
  // These methods are virtual to allow derived classes to customize or extend
  // zooming behavior.
  virtual void Zoom(float zoom_factor);
  virtual void Zoom(float zoom_factor, Microseconds pivot);

  // --- Interaction Helpers ---

  // Handles keyboard input for panning and zooming. Returns true if any
  // interaction occurred.
  bool HandleKeyboard();

  // Handles mouse wheel input for scrolling. Returns true if any interaction
  // occurred.
  bool HandleWheel();

  // Handles mouse input for selection and panning. Returns true if any
  // interaction occurred.
  bool HandleMouse();

  void HandleMouseDown(float timeline_origin_x);
  void HandleMouseDrag(float timeline_origin_x);
  void HandleMouseRelease();

  // Handles deselection of events when clicking on an empty area.
  void HandleEventDeselection();

  // --- Rendering Helpers ---

  // Draws the timeline ruler UI (background, horizontal line, labels, ticks).
  void DrawRulerUI(const TickInfo& info, Pixel timeline_width);

  // Draws vertical grid lines across the background of the tracks.
  void DrawVerticalGridLines(const TickInfo& info, Pixel timeline_width,
                             Pixel viewport_bottom);

  void DrawGroup(int group_index, double px_per_time_unit_val);
  void DrawGroupPreview(int group_index, double px_per_time_unit_val);
  void DrawFlameGroupPreview(int start_level, int end_level,
                             double px_per_time_unit_val, const ImVec2& pos,
                             Pixel group_height, ImDrawList* draw_list);

  void DrawEvent(int group_index, int event_index, const EventRect& rect,
                 ImDrawList* absl_nonnull draw_list);

  void DrawEventName(absl::string_view event_name, const EventRect& rect,
                     ImDrawList* absl_nonnull draw_list) const;

  void DrawEventsForLevel(int group_index, absl::Span<const int> event_indices,
                          double px_per_time_unit, int level_in_group,
                          const ImVec2& pos, const ImVec2& max,
                          Pixel event_height, Pixel padding_bottom);

  void DrawCounterTrack(int group_index, const CounterData& counter_data,
                        double px_per_time_unit_val, const ImVec2& pos,
                        Pixel height);

  void DrawCounterTooltip(int group_index, const CounterData& counter_data,
                          double px_per_time_unit_val, const ImVec2& pos,
                          Pixel height, float y_ratio, ImDrawList* draw_list);

  // Draws flow lines connecting events.
  void DrawFlows(Pixel timeline_width);

  // Draws a single flow line.
  void DrawSingleFlow(const FlowLine& flow, Pixel timeline_x_start,
                      double px_per_time, ImDrawList* draw_list);

  // Selection overlay rendering.
  void DrawSelectedTimeRanges(Pixel timeline_width,
                              double px_per_time_unit_val);

  void DrawSelectedTimeRange(const TimeRange& range, Pixel timeline_width,
                             double px_per_time_unit_val,
                             bool show_delete_button = true);

  void DrawDeleteButton(ImDrawList* draw_list, const ImVec2& button_pos,
                        const ImRect& hover_rect, const TimeRange& range);

  // --- Internal Utilities ---

  double px_per_time_unit() const;
  double px_per_time_unit(Pixel timeline_width) const;

  void EmitEventSelected(int event_index);
  void EmitViewportChanged(const TimeRange& range);

  // Helper to calculate the timeline area.
  ImRect GetTimelineArea() const;

  // Updates the search results based on the current search query.
  void RecomputeSearchResults();

  // Private structs and constants.

  struct SearchResult {
    EventId event_id;
    int level;
    Microseconds start_time;
    Microseconds duration;
    ProcessId pid;
    ThreadId tid;
  };

  static constexpr ImGuiWindowFlags kImGuiWindowFlags =
      ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoCollapse |
      ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
      ImGuiWindowFlags_NoMove;
  static constexpr ImGuiTableFlags kImGuiTableFlags =
      ImGuiTableFlags_NoPadOuterX | ImGuiTableFlags_BordersInnerV |
      ImGuiTableFlags_Resizable | ImGuiTableFlags_NoSavedSettings;
  static constexpr ImGuiWindowFlags kLaneFlags =
      ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse;

  // --- Data and State Members ---

  FlameChartTimelineData timeline_data_;

  // TODO - b/444026851: Set the label width based on the real screen width.
  Pixel label_width_ = kDefaultLabelWidth;
  // The width of the timeline track area in pixels.
  Pixel current_timeline_width_ = 0.0f;
  bool is_resizing_label_column_ = false;

  std::vector<float> level_y_positions_;

  Animated<TimeRange> visible_range_;
  TimeRange fetched_data_time_range_ = TimeRange::Zero();
  TimeRange data_time_range_ = TimeRange::Zero();

  int selected_group_index_ = -1;
  int selected_event_index_ = -1;
  int selected_counter_index_ = -1;

  EventCallback event_callback_ = [](absl::string_view, const EventData&) {};
  bool event_clicked_this_frame_ = false;

  bool is_dragging_ = false;
  int flow_category_filter_ = FlowCategoryFilter::kNone;
  absl::flat_hash_set<int> visible_flow_categories_;
  bool is_selecting_ = false;

  bool mpmd_pipeline_view_enabled_ = false;

  std::vector<TimeRange> selected_time_ranges_;
  Microseconds drag_start_time_ = 0.0;
  std::optional<TimeRange> current_selected_time_range_;

  bool is_incremental_loading_ = true;
  TimeRange last_fetch_request_range_ = TimeRange::Zero();

  std::string search_query_lower_ = "";
  std::vector<SearchResult> sorted_search_results_;
  int current_search_result_index_ = -1;
  std::optional<EventId> pending_navigation_event_id_;
};

}  // namespace traceviewer
#endif  // THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TIMELINE_TIMELINE_H_
