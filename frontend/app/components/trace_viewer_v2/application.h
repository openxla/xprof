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

  // Default `localStorage` key of the saved track order (used by the Trace
  // Viewer). Hosts such as the Kernel Viewer override it with
  // `SetTrackOrderStorageKey()`.
  static constexpr absl::string_view kDefaultTrackOrderStorageKey =
      "xprof_trace_viewer_track_order";

  // Enables or disables reordering tracks by their drag handle. See
  // `Timeline::set_track_reordering_enabled()`.
  void SetTrackReorderingEnabled(bool enabled) {
    if (timeline_) {
      timeline_->set_track_reordering_enabled(enabled);
    }
  }

  // Sets the `localStorage` key used by `SaveTrackOrderToLocalStorage()`,
  // `LoadTrackOrderFromLocalStorage()`, and `RestoreSavedTrackOrder()`, so
  // that hosts (such as the Trace Viewer and the Kernel Viewer) keep separate
  // track orders. If `key` is empty, `kDefaultTrackOrderStorageKey` is used.
  // Immediately restores the order saved under the new key if the application
  // is initialized.
  void SetTrackOrderStorageKey(absl::string_view key);

  const std::string& track_order_storage_key() const {
    return track_order_storage_key_;
  }

  // Re-applies the track order saved in `localStorage` under
  // `track_order_storage_key_`, or resets the track order if none (or a
  // malformed one) is saved. Within each viewer, all sessions and browser tabs
  // share the key under a last-writer-wins policy, so this is called before a
  // trace is loaded to pick up the latest layout.
  void RestoreSavedTrackOrder();

  // Saves `serialized_order` (see `TrackOrder::Serialize()`) in
  // `localStorage` under `track_order_storage_key_`, or removes the saved
  // order if `serialized_order` is empty.
  void SaveTrackOrderToLocalStorage(absl::string_view serialized_order) const;

  // Returns the track order saved in `localStorage` under
  // `track_order_storage_key_`, or an empty string if none is saved.
  std::string LoadTrackOrderFromLocalStorage() const;

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

  std::unique_ptr<WGPURenderPlatform> platform_;
  std::unique_ptr<Timeline> timeline_;
  // The data provider for trace events.
  DataProvider data_provider_;

  // Color theme for timeline.
  ColorPalette palette_ = ColorPalette::Default();
  uint64_t palette_version_ = 0xDEADBEEF;

  std::unordered_map<std::string, bool> feature_flags_cache_;

  std::string track_order_storage_key_ =
      std::string(kDefaultTrackOrderStorageKey);

  void MainLoop();
  // Draws a single frame. Should ONLY be called from MainLoop() to ensure
  // DeltaTime and Animations are updated correctly. Other methods should
  // use RequestRedraw() instead of calling Draw() directly.
  void Draw();

  absl::Time last_frame_time_ = absl::Now();
  float GetDeltaTime();

  void UpdateImGuiDisplaySize(const CanvasState& canvas_state);

  void UpdateMouseCursor();

  void UpdateApplicationColors();

  ImGuiMouseCursor last_cursor_ = ImGuiMouseCursor_Arrow;
  bool frame_active_ = false;
};

}  // namespace traceviewer

#endif  // THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_APPLICATION_H_
