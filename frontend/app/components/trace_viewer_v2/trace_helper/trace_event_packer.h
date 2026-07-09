#ifndef THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TRACE_HELPER_TRACE_EVENT_PACKER_H_
#define THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TRACE_HELPER_TRACE_EVENT_PACKER_H_

#include <vector>

#include "absl/types/span.h"
#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_event.h"

namespace traceviewer {

// Represents a trace event and its assigned visual level (row) after packing.
struct PackedEvent {
  // Pointer to the original trace event. Must outlive PackedEvent and is
  // assumed to be non-null.
  const TraceEvent* event;
  // The 0-indexed level (row) assigned to event to avoid visual overlaps.
  int level;
};

// Packs trace events into visual levels such that overlapping events are
// assigned to different levels, while nested events are assigned to consecutive
// levels to represent parent-child relationships visually.
//
// events is a span of pointers to trace events.Returns a vector of
// `PackedEvent` entries ordered chronologically by event start time
// (ascending), with tie-breaking by duration (descending).
std::vector<PackedEvent> PackTraceEvents(
    absl::Span<const TraceEvent* const> events);

}  // namespace traceviewer

#endif  // THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TRACE_HELPER_TRACE_EVENT_PACKER_H_
