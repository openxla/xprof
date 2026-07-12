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
// Endpoint / overlap policy (half-open intervals):
//   Each event occupies the half-open range [ts, ts + dur). Two events
//   overlap (and therefore cannot share a visual level) iff their ranges
//   intersect. In particular:
//     - An event that ends at time T and another that starts at time T do
//       NOT overlap: the first event's end is exclusive, so the second may
//       reuse the same level (row). Implementation: a row is free when
//       row_end_time <= candidate_start.
//     - Zero-duration events occupy an empty range [ts, ts) and never block
//       later events that start at ts.
//   Nested containment uses the same half-open model: event B is nested
//   under A when A.ts <= B.ts and B.ts + B.dur <= A.ts + A.dur. Partial
//   (non-nested) overlaps are stacked on different levels rather than merged
//   or dropped — every input event appears exactly once in the output.
//
// events is a span of pointers to trace events. Null pointers are ignored.
// Returns a vector of `PackedEvent` entries ordered chronologically by event
// start time (ascending), with tie-breaking by duration (descending) so that
// longer parent events are packed before shorter children that start at the
// same timestamp.
std::vector<PackedEvent> PackTraceEvents(
    absl::Span<const TraceEvent* const> events);

}  // namespace traceviewer

#endif  // THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TRACE_HELPER_TRACE_EVENT_PACKER_H_
