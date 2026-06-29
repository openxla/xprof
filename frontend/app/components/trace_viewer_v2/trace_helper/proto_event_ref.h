#ifndef THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TRACE_HELPER_PROTO_EVENT_REF_H_
#define THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TRACE_HELPER_PROTO_EVENT_REF_H_

#include <cstdint>
#include <limits>
#include <map>
#include <string>

#include "absl/strings/string_view.h"
#include "tsl/profiler/lib/context_types.h"
#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_event.h"

namespace traceviewer {

struct ProtoEventRef {
  Phase ph = Phase::kUnknown;
  ProcessId pid = 0;
  ThreadId tid = 0;
  absl::string_view name;
  uint32_t name_index = 0;
  Microseconds ts = 0.0;
  Microseconds dur = 0.0;
  uint64_t flow_id = 0;
  tsl::profiler::ContextType category = tsl::profiler::ContextType::kGeneric;
  bool is_async = false;
  uint64_t uid = std::numeric_limits<uint64_t>::max();
  uint64_t group_id = 0;
  absl::string_view metadata_name;
  uint32_t sort_index = 0;
  const std::map<std::string, std::string, std::less<>>* args_map = nullptr;

  absl::string_view GetArg(absl::string_view key) const {
    if (args_map != nullptr) {
      auto it = args_map->find(key);
      if (it != args_map->end()) return it->second;
    }
    return "";
  }
};

}  // namespace traceviewer

#endif  // THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TRACE_HELPER_PROTO_EVENT_REF_H_
