#ifndef PERFTOOLS_ACCELERATORS_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_APPLICATION_H_
#define PERFTOOLS_ACCELERATORS_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_APPLICATION_H_
#include <stdlib.h>

#include <memory>

#include "xprof/frontend/app/components/trace_viewer_v2/timeline/data_provider.h"
#include "xprof/frontend/app/components/trace_viewer_v2/timeline/timeline.h"
#include "xprof/frontend/app/components/trace_viewer_v2/webgpu_render_platform.h"
#include "absl/base/no_destructor.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "third_party/dear_imgui/imgui.h"

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
  const std::vector<std::string> process_list() {
    return data_provider_.GetProcessList();
  };
  DataProvider& data_provider() { return data_provider_; };

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

  absl::Time last_frame_time_ = absl::Now();
  float GetDeltaTime();

  void UpdateImGuiDisplaySize(const CanvasState& canvas_state);
};

}  // namespace traceviewer

#endif  // PERFTOOLS_ACCELERATORS_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_APPLICATION_H_
