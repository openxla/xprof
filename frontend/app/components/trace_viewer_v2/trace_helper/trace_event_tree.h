#ifndef PERFTOOLS_ACCELERATORS_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TRACE_HELPER_TRACE_EVENT_TREE_H_
#define PERFTOOLS_ACCELERATORS_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TRACE_HELPER_TRACE_EVENT_TREE_H_

#include <memory>
#include <vector>

#include "xprof/frontend/app/components/trace_viewer_v2/trace_helper/trace_event.h"
#include "absl/types/span.h"

namespace traceviewer {

// Represents a node in the trace event tree.
struct TraceEventNode {
  explicit TraceEventNode(const TraceEvent* e) : event(e) {}

  // The TraceEvent object pointed to must outlive this TraceEventNode instance.
  const TraceEvent* event;
  int depth = 0;
  TraceEventNode* parent = nullptr;
  std::vector<std::unique_ptr<TraceEventNode>> children;
  // Time spent in this event, excluding time in child events.
  Microseconds self_time = 0.0;
};

// Represents the tree structure for a thread's events.
struct TraceEventTree {
  std::vector<std::unique_ptr<TraceEventNode>> roots;
  // Maximum depth of any node in the tree.
  int max_depth = 0;
};

// Builds a tree from a vector of events.
TraceEventTree BuildTree(absl::Span<const TraceEvent* const> events);

}  // namespace traceviewer

#endif  // PERFTOOLS_ACCELERATORS_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TRACE_HELPER_TRACE_EVENT_TREE_H_
