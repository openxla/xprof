#ifndef THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TRACE_HELPER_PERFETTO_CONSTANTS_H_
#define THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TRACE_HELPER_PERFETTO_CONSTANTS_H_

#include <cstdint>

namespace traceviewer {

// Constants for Perfetto parser.

// Offset added to track UUIDs when generating fallback TIDs for tracks that do
// not have a thread ID. This helps avoid conflicts with actual thread IDs,
// assuming they are typically smaller than this offset.
inline constexpr uint64_t kPerfettoFallbackTidOffset = 1000000;

// Fallback PID used when no process descriptor is found in the trace.
inline constexpr uint32_t kPerfettoFallbackPid = 1;

}  // namespace traceviewer

#endif  // THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TRACE_HELPER_PERFETTO_CONSTANTS_H_
