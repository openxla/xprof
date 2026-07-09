#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_event_packer.h"

#include <cstddef>
#include <optional>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/types/span.h"
#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_event.h"

namespace traceviewer {

std::vector<PackedEvent> PackTraceEvents(
    absl::Span<const TraceEvent* const> events) {
  std::vector<const TraceEvent*> sorted_events;
  sorted_events.reserve(events.size());
  for (const TraceEvent* event : events) {
    if (event != nullptr) {
      sorted_events.push_back(event);
    }
  }

  // Sort events by start time (ascending), then by duration (descending) as a
  // tie-breaker. This ensures that parent events (longer duration) are
  // processed before their children (shorter duration) when they start at the
  // same time.
  absl::c_sort(sorted_events, [](const TraceEvent* a, const TraceEvent* b) {
    if (a->ts != b->ts) {
      return a->ts < b->ts;
    }
    return a->dur > b->dur;
  });

  std::vector<PackedEvent> packed_events;
  packed_events.reserve(sorted_events.size());

  std::vector<Microseconds> row_end_times;
  for (const TraceEvent* event : sorted_events) {
    const Microseconds start = event->ts;
    const Microseconds duration = event->dur;
    const Microseconds end = start + duration;

    std::optional<int> selected_row = std::nullopt;
    for (size_t i = 0; i < row_end_times.size(); ++i) {
      if (row_end_times[i] <= start) {
        selected_row = static_cast<int>(i);
        break;
      }
    }

    if (!selected_row.has_value()) {
      row_end_times.push_back(end);
      selected_row = static_cast<int>(row_end_times.size() - 1);
    } else {
      row_end_times[*selected_row] = end;
    }

    packed_events.push_back({event, *selected_row});
  }

  return packed_events;
}

}  // namespace traceviewer
