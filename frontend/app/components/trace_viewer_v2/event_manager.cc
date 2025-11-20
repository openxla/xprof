#include "xprof/frontend/app/components/trace_viewer_v2/event_manager.h"

#include <emscripten/val.h>

#include <string>
#include <utility>

#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "absl/types/any.h"

namespace traceviewer {

namespace {
emscripten::val AnyToVal(const absl::any& any_val);

emscripten::val EventDataToVal(const EventData& event_data) {
  emscripten::val val = emscripten::val::object();
  for (const std::pair<const std::string, absl::any>& item : event_data) {
    val.set(item.first, AnyToVal(item.second));
  }
  return val;
}

emscripten::val AnyToVal(const absl::any& any_val) {
  if (any_val.type() == typeid(int)) {
    return emscripten::val(absl::any_cast<int>(any_val));
  }
  if (any_val.type() == typeid(float)) {
    return emscripten::val(absl::any_cast<float>(any_val));
  }
  if (any_val.type() == typeid(bool)) {
    return emscripten::val(absl::any_cast<bool>(any_val));
  }
  if (any_val.type() == typeid(double)) {
    return emscripten::val(absl::any_cast<double>(any_val));
  }
  if (any_val.type() == typeid(std::string)) {
    return emscripten::val(absl::any_cast<std::string>(any_val));
  }
  if (any_val.type() == typeid(EventData)) {
    return EventDataToVal(absl::any_cast<EventData>(any_val));
  }
  // Add more types in the future if needed.

  // Log a warning for unsupported types.
  LOG(WARNING) << "Unsupported absl::any type encountered: "
               << any_val.type().name();
  return emscripten::val::undefined();
}
}  // namespace

void EventManager::DispatchEvent(absl::string_view type,
                                 const EventData& event_data) {
  emscripten::val detail_val = EventDataToVal(event_data);

  // The CustomEvent constructor expects an options object where the custom data
  // is passed under the "detail" key.
  emscripten::val options = emscripten::val::object();
  options.set("detail", detail_val);

  emscripten::val custom_event =
      emscripten::val::global("CustomEvent").new_(std::string(type), options);
  emscripten::val::global("window").call<void>("dispatchEvent", custom_event);
}

}  // namespace traceviewer
