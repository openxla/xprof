#ifndef THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TRACE_HELPER_PROTO_EVENT_REF_H_
#define THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TRACE_HELPER_PROTO_EVENT_REF_H_

#include <cstdint>

#include "absl/strings/string_view.h"
#include "tsl/profiler/lib/context_types.h"
#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_event.h"

namespace traceviewer {

struct ProtoEventRef {
  ProcessId pid = 0;
  ThreadId tid = 0;
  Microseconds ts = 0.0;
  Microseconds dur = 0.0;
  uint64_t flow_id = 0;
  absl::string_view name;
  uint32_t name_index = 0;
  uint32_t uid = 0;
  tsl::profiler::ContextType category = tsl::profiler::ContextType::kGeneric;
  Phase ph = Phase::kUnknown;
  bool is_async = false;
};

}  // namespace traceviewer

#endif  // THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TRACE_HELPER_PROTO_EVENT_REF_H_
