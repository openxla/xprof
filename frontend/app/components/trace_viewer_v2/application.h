#ifndef THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_APPLICATION_H_
#define THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_APPLICATION_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>

#include "emscripten/val.h"
#include "absl/base/no_destructor.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "imgui.h"
#include "frontend/app/components/trace_viewer_v2/canvas_state.h"
#include "frontend/app/components/trace_viewer_v2/color/colors.h"
#include "frontend/app/components/trace_viewer_v2/scheduler.h"
#include "frontend/app/components/trace_viewer_v2/timeline/data_provider.h"
#include "frontend/app/components/trace_viewer_v2/timeline/time_range.h"
#include "frontend/app/components/trace_viewer_v2/timeline/timeline.h"
#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_event.h"
#include "frontend/app/components/trace_viewer_v2/webgpu_render_platform.h"

namespace traceviewer {

class Application {
 public:
  // Application is implemented as a singleton because it represents the entire
  // program's state and main control flow. This ensures that there is only one
  // central control object for the application, managing the lifecycle of core
  // components like the renderer and the timeline view. This provides a single
  // point of entry and control for the whole application.
  static Application& Instance() {
    static absl::NoDestructor<Application> instance;
    return *instance;
  }

  // Application is a singleton, so it should not be copyable or movable.
  Application(const Application&) = delete;
  Application& operator=(const Application&) = delete;
  Application(Application&&) = delete;
  Application& operator=(Application&&) = delete;

  ~Application() { ImGui::DestroyContext(); }

  void Initialize();
  void Shutdown();
  void Main() { RequestRedraw(); }
  void RequestRedraw() { Scheduler::Instance().RequestRedraw(); }

  Timeline& timeline() { return *timeline_; };
  DataProvider& data_provider() { return data_provider_; };

  ColorPalette& GetPalette() { return palette_; };

  bool IsFeatureEnabled(const std::string& name);

  bool IsInitialized() const { return timeline_ != nullptr; }

  void SetVisibleFlowCategory(int category_id) {
    timeline_->SetVisibleFlowCategory(category_id);
  }

  void SetVisibleFlowCategories(const emscripten::val& category_ids);

  void Resize(float dpr, int width, int height);

  void SetVisibleRange(Microseconds start, Microseconds end) {
    if (timeline_) {
      TimeRange range(start, end);
      timeline_->ConstrainTimeRange(range);
      timeline_->SetVisibleRange(range);
      RequestRedraw();
    }
  }

  void SetSearchQuery(const std::string& query) {
    if (timeline_) {
      timeline_->SetSearchQuery(query);
    }
  }

  void SetMouseMode(int mode) {
    if (timeline_) {
      timeline_->set_mouse_mode(static_cast<MouseMode>(mode));
    }
  }

  void SetPanningSpeed(float speed) {
    if (timeline_) {
      timeline_->set_panning_speed(speed);
    }
  }

  void SetZoomSpeed(float speed) {
    if (timeline_) {
      timeline_->set_zoom_speed(speed);
    }
  }

  void SetPlaybackState(bool is_playing, double current_progress_us,
                        double play_speed) {
    if (timeline_) {
      timeline_->SetPlaybackState(is_playing, current_progress_us, play_speed);
    }
  }

  void SetMouseWheelZoomSpeed(float speed) {
    if (timeline_) {
      timeline_->set_mouse_wheel_zoom_speed(speed);
    }
  }

  // See Timeline::set_incremental_fetch_enabled().
  void SetIncrementalFetchEnabled(bool enabled) {
    if (timeline_) {
      timeline_->set_incremental_fetch_enabled(enabled);
    }
  }

  // See Timeline::set_event_tooltip_enabled().
  void SetEventTooltipEnabled(bool enabled) {
    if (timeline_) {
      timeline_->set_event_tooltip_enabled(enabled);
    }
  }

  // Clears all per-dataset state so a different dataset can be loaded into
  // this instance. Call before loadTraceData() when switching datasets.
  void ResetDataset() {
    if (!timeline_) return;
    data_provider_.Reset();
    timeline_->ResetForNewDataset();
    RequestRedraw();
  }

  // Enables or disables reordering tracks by their drag handle. See
  // `Timeline::set_track_reordering_enabled()`.
  void SetTrackReorderingEnabled(bool enabled) {
    if (timeline_) {
      timeline_->set_track_reordering_enabled(enabled);
      RequestRedraw();
    }
  }

  // Sets the track ordering mode from its `TrackOrder::ModeName()`, such as
  // "alphabetical_asc". While reordering is disabled, the mode is stored and
  // saved to local storage, but the tracks stay in natural order. Returns false
  // and changes nothing if `mode_name` is not a known mode or the application
  // is not initialized.
  bool SetTrackOrderMode(absl::string_view mode_name);

  // Clears the custom track order and resets the ordering mode to `custom`,
  // restoring the trace order. See `Timeline::ResetTrackOrder()`.
  void ResetTrackOrder();

  // Returns the track order state as a JavaScript object
  // `{enabled: boolean, mode: string, hasCustomOrder: boolean}`, where `mode`
  // is a `TrackOrder::ModeName()`. The `kTrackOrderChanged` event, which is
  // dispatched after a frame in which the track order, its mode, or the track
  // list changed, carries the same object as its `detail`.
  emscripten::val GetTrackOrderState() const;

  // Returns all tracks, in the order of `Timeline::GetTrackList()`, as a
  // JavaScript array of `{id: number, parentId: number, name: string,
  // subtitle: string, kind: 'process' | 'thread' | 'counter', section:
  // 'hidden' | 'pinned' | 'all'}`. `id` and `parentId` (-1 for root tracks)
  // are valid until the timeline data changes, which is followed by a
  // `kTrackOrderChanged` event after the next frame.
  emscripten::val GetTrackList() const;

  // Moves the track `source` right before (or, if `after`, right after) its
  // sibling `target`, where both are `id`s from `GetTrackList()`. Returns
  // false if the move is invalid. See `Timeline::MoveTrack()`.
  bool MoveTrack(int source, int target, bool after);

  // Re-applies the track order saved in `localStorage`, or resets the track
  // order if none (or a malformed one) is saved. The saved order is shared by
  // all sessions and browser tabs under a last-writer-wins policy, so this is
  // called before a trace is loaded to pick up the latest layout.
  void RestoreSavedTrackOrder();

  // Saves `serialized_order` (see `TrackOrder::Serialize()`) in
  // `localStorage`, or removes the saved order if `serialized_order` is empty.
  static void SaveTrackOrderToLocalStorage(absl::string_view serialized_order);

  // Returns the track order saved in `localStorage`, or an empty string if
  // none is saved.
  static std::string LoadTrackOrderFromLocalStorage();

  void NavigateToNextSearchResult() {
    if (timeline_) {
      timeline_->NavigateToNextSearchResult();
    }
  }

  void NavigateToPrevSearchResult() {
    if (timeline_) {
      timeline_->NavigateToPrevSearchResult();
    }
  }

  size_t GetSearchResultsCount() const {
    if (timeline_) {
      return timeline_->get_search_results_count();
    }
    return 0;
  }

  int GetCurrentSearchResultIndex() const {
    if (!timeline_) {
      return -1;
    }
    return timeline_->get_current_search_result_index();
  }

 private:
  friend class absl::NoDestructor<Application>;

  // Members are initialized to nullptr here and will be properly allocated in
  // the Initialize() method.
  Application() = default;

  void MainLoop();
  // Draws a single frame. Should ONLY be called from MainLoop() to ensure
  // DeltaTime and Animations are updated correctly. Other methods should
  // use RequestRedraw() instead of calling Draw() directly.
  void Draw();

  float GetDeltaTime();

  void UpdateImGuiDisplaySize(const CanvasState& canvas_state);

  void UpdateMouseCursor();

  void UpdateApplicationColors();

  // Returns the track order state as EventData for dispatching or Embind.
  EventData TrackOrderStateData() const;

  // Dispatches the `kTrackOrderChanged` event with `GetTrackOrderState()` if
  // the track order or the track list changed since the last one.
  void DispatchTrackOrderChangedIfNeeded();

  std::unique_ptr<WGPURenderPlatform> platform_;
  std::unique_ptr<Timeline> timeline_;
  // The data provider for trace events.
  DataProvider data_provider_;

  // Color theme for timeline.
  ColorPalette palette_ = ColorPalette::Default();
  uint64_t palette_version_ = 0xDEADBEEF;

  std::unordered_map<std::string, bool> feature_flags_cache_;

  // Whether the track order or the track list changed since the last
  // `kTrackOrderChanged` event.
  bool track_order_event_pending_ = false;

  absl::Time last_frame_time_ = absl::Now();
  ImGuiMouseCursor last_cursor_ = ImGuiMouseCursor_Arrow;
  bool frame_active_ = false;
};

}  // namespace traceviewer

#endif  // THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_APPLICATION_H_
