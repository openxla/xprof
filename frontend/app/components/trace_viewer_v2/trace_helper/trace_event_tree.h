#ifndef THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TRACE_HELPER_TRACE_EVENT_TREE_H_
#define THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TRACE_HELPER_TRACE_EVENT_TREE_H_

#include <memory>
#include <vector>

#include "absl/types/span.h"
#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_event.h"

namespace traceviewer {

// Represents a node in the trace event tree.
template <typename EventT>
struct TraceEventNodeT {
  explicit TraceEventNodeT(const EventT* e) : event(e) {}

  // The EventT object pointed to must outlive this TraceEventNodeT instance.
  const EventT* event;
  int depth = 0;
  TraceEventNodeT<EventT>* parent = nullptr;
  std::vector<std::unique_ptr<TraceEventNodeT<EventT>>> children;
  // Time spent in this event, excluding time in child events.
  Microseconds self_time = 0.0;
};

// Represents the tree structure for a thread's events.
template <typename EventT>
struct TraceEventTreeT {
  std::vector<std::unique_ptr<TraceEventNodeT<EventT>>> roots;
  // Maximum depth of any node in the tree.
  int max_depth = 0;
};

using TraceEventNode = TraceEventNodeT<TraceEvent>;
using TraceEventTree = TraceEventTreeT<TraceEvent>;

// Builds a tree from a span of events.
template <typename EventT>
TraceEventTreeT<EventT> BuildTree(absl::Span<const EventT* const> events);

// Overload for std::vector to enable automatic template argument deduction.
template <typename EventT>
TraceEventTreeT<EventT> BuildTree(const std::vector<const EventT*>& events) {
  return BuildTree<EventT>(absl::Span<const EventT* const>(events));
}

}  // namespace traceviewer

#endif  // THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TRACE_HELPER_TRACE_EVENT_TREE_H_
