#ifndef THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_APPLICATION_H_
#define THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_APPLICATION_H_
#include <cstddef>
#include <memory>

#include "emscripten/val.h"
#include "absl/base/no_destructor.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "third_party/dear_imgui/imgui.h"
#include "xprof/frontend/app/components/trace_viewer_v2/timeline/data_provider.h"
#include "xprof/frontend/app/components/trace_viewer_v2/timeline/timeline.h"
#include "xprof/frontend/app/components/trace_viewer_v2/webgpu_render_platform.h"

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
  void Main();

  Timeline& timeline() { return *timeline_; };
  DataProvider& data_provider() { return data_provider_; };

  void SetVisibleFlowCategory(int category_id) {
    timeline_->SetVisibleFlowCategory(category_id);
  }

  void SetVisibleFlowCategories(const emscripten::val& category_ids);

  void Resize(float dpr, int width, int height);

  void SetVisibleRange(Microseconds start, Microseconds end) {
    timeline_->SetVisibleRange(TimeRange(start, end));
  }

  void SetSearchQuery(const std::string& query) {
    if (timeline_) {
      timeline_->SetSearchQuery(query);
    }
  }

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

  void MainLoop();
  void Draw();

  absl::Time last_frame_time_ = absl::Now();
  float GetDeltaTime();

  void UpdateImGuiDisplaySize(const CanvasState& canvas_state);

  void UpdateMouseCursor();
  ImGuiMouseCursor last_cursor_ = ImGuiMouseCursor_Arrow;
};

}  // namespace traceviewer

#endif  // THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_APPLICATION_H_
