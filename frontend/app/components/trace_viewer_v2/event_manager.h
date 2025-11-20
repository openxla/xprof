#ifndef PERFTOOLS_ACCELERATORS_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_EVENT_MANAGER_H_
#define PERFTOOLS_ACCELERATORS_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_EVENT_MANAGER_H_

#include "xprof/frontend/app/components/trace_viewer_v2/event_data.h"
#include "absl/base/no_destructor.h"
#include "absl/strings/string_view.h"

namespace traceviewer {

class EventManager {
 public:
  static EventManager& Instance() {
    static absl::NoDestructor<EventManager> instance;
    return *instance;
  }

  EventManager(const EventManager&) = delete;
  EventManager& operator=(const EventManager&) = delete;
  EventManager(EventManager&&) = delete;
  EventManager& operator=(EventManager&&) = delete;

  // Dispatches a CustomEvent to the global window object. This is a C++ analog
  // to the CustomEvent constructor in TypeScript/JavaScript. The provided
  // `detail` is wrapped in an `event_init_dict` as required by the spec.
  // See:
  // https://developer.mozilla.org/en-US/docs/Web/API/CustomEvent/CustomEvent
  void DispatchEvent(absl::string_view type, const EventData& event_data);

 private:
  friend class absl::NoDestructor<EventManager>;
  EventManager() = default;
};

}  // namespace traceviewer

#endif  // PERFTOOLS_ACCELERATORS_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_EVENT_MANAGER_H_
