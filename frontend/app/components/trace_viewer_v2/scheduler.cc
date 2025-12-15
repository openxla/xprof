#include "xprof/frontend/app/components/trace_viewer_v2/scheduler.h"

#include <emscripten/html5.h>

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

void Scheduler::RequestRedraw() {
  if (!frame_scheduled_) {
    emscripten_request_animation_frame(LoopOnce, this);
    frame_scheduled_ = true;
  }
}

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
