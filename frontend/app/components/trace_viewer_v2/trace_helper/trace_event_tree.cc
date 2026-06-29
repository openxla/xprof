#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_event_tree.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "frontend/app/components/trace_viewer_v2/trace_helper/proto_event_ref.h"
#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_event.h"
#include "absl/log/log.h"
#include "absl/types/span.h"

namespace traceviewer {

template <typename EventT>
TraceEventTreeT<EventT> BuildTree(absl::Span<const EventT* const> events) {
  TraceEventTreeT<EventT> tree;
  std::vector<TraceEventNodeT<EventT>*> stack;

  // Events are expected to be sorted by timestamp (ascending), then by
  // duration (descending).
  for (const EventT* event : events) {
    const Microseconds start_time = event->ts;
    const Microseconds duration = event->dur;
    const Microseconds end_time = start_time + duration;

    auto node = std::make_unique<TraceEventNodeT<EventT>>(event);
    node->self_time = duration;
    bool node_processed = false;

    while (!stack.empty()) {
      TraceEventNodeT<EventT>* parent_node = stack.back();
      const EventT* parent_event = parent_node->event;
      const Microseconds parent_start_time = parent_event->ts;
      const Microseconds parent_duration = parent_event->dur;
      const Microseconds parent_end_time = parent_start_time + parent_duration;

      if (start_time < parent_start_time) {
        // This block should be unreachable. The 'sorted_events' vector is
        // sorted primarily by event start time ('ts') in ascending order.
        // The stack contains parent events, and any new event being processed
        // must have a start time greater than or equal to the event on top
        // of the stack, which was added in a previous iteration.
        LOG(ERROR) << "Unreachable code reached: current event start time is "
                      "less than parent event start time.";
        break;
      }

      if (start_time >= parent_end_time) {
        // Current event starts after parent ends, pop parent.
        stack.pop_back();
      } else if (end_time <= parent_end_time) {
        // Current event is a child of the parent.
        node->parent = parent_node;
        node->depth = stack.size();
        parent_node->self_time -= duration;
        tree.max_depth = std::max(tree.max_depth, node->depth + 1);
        parent_node->children.push_back(std::move(node));
        stack.push_back(parent_node->children.back().get());
        node_processed = true;
        break;
      } else {
        // The current event overlaps with the parent but is not fully
        // contained within it. For building a strict parent-child tree
        // structure, such events are skipped as they don't fit the
        // hierarchical model where children must be fully within their parent's
        // time range.
        node_processed = true;
        break;
      }
    }

    if (node_processed) {
      continue;
    }

    // If stack is empty, it's a root node.
    tree.roots.push_back(std::move(node));
    stack.push_back(tree.roots.back().get());
    tree.max_depth = std::max(tree.max_depth, 1);
  }
  return tree;
}

template TraceEventTreeT<TraceEvent> BuildTree<TraceEvent>(
    absl::Span<const TraceEvent* const> events);
template TraceEventTreeT<ProtoEventRef> BuildTree<ProtoEventRef>(
    absl::Span<const ProtoEventRef* const> events);

}  // namespace traceviewer
