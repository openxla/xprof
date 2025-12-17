#ifndef THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_SCHEDULER_H_
#define THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_SCHEDULER_H_

#include <emscripten/html5.h>

#include <functional>

#include "absl/base/no_destructor.h"

namespace traceviewer {

class Scheduler {
 public:
  static Scheduler& Instance();

  Scheduler(const Scheduler&) = delete;
  Scheduler& operator=(const Scheduler&) = delete;
  Scheduler(Scheduler&&) = delete;
  Scheduler& operator=(Scheduler&&) = delete;

  void RequestRedraw();

  void SetMainLoopCallback(std::function<void()> callback);

  // Resets the scheduler state.
  void Reset();

 private:
  friend class absl::NoDestructor<Scheduler>;
  Scheduler() = default;

  static EM_BOOL LoopOnce(double time, void* user_data);

  bool frame_scheduled_ = false;
  std::function<void()> callback_;
};

}  // namespace traceviewer

#endif  // THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_SCHEDULER_H_
