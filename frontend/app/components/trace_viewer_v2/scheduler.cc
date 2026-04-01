#include "frontend/app/components/trace_viewer_v2/scheduler.h"

#include <emscripten/em_types.h>
#include <emscripten/html5.h>

#include <functional>
#include <utility>

#include "absl/base/no_destructor.h"

namespace traceviewer {

Scheduler& Scheduler::Instance() {
  static absl::NoDestructor<Scheduler> instance;
  return *instance;
}

void Scheduler::SetMainLoopCallback(std::function<void()> callback) {
  callback_ = std::move(callback);
}

void Scheduler::Reset() {
  frame_scheduled_ = false;
  callback_ = {};
}

// Requests a redraw on the next animation frame.
// Leverages emscripten_request_animation_frame (browser's
// requestAnimationFrame) to align redraws with the screen refresh rate. It uses
// a throttle lock (frame_scheduled_) to prevent duplicate requests.
void Scheduler::RequestRedraw() {
  if (!frame_scheduled_) {
    emscripten_request_animation_frame(LoopOnce, this);
    frame_scheduled_ = true;
  }
}

// Static callback triggered by emscripten_request_animation_frame.
// Resets the throttle lock and runs the registered callback (usually
// Application::MainLoop). Returns EM_FALSE to notify the browser that this is a
// demand-driven one-off callback and should not automatically repeat.
EM_BOOL Scheduler::LoopOnce(double time, void* user_data) {
  auto* scheduler = static_cast<Scheduler*>(user_data);

  scheduler->frame_scheduled_ = false;

  if (scheduler->callback_) {
    scheduler->callback_();
  }

  // Return EM_FALSE because this is a demand-driven system. Redraws should
  // only happen when explicitly requested via RequestRedraw().
  return EM_FALSE;
}

}  // namespace traceviewer
