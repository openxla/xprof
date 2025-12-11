#include "xprof/frontend/app/components/trace_viewer_v2/timeline/time_range.h"

#include <algorithm>

#include "xprof/frontend/app/components/trace_viewer_v2/trace_helper/trace_event.h"
#include "absl/log/log.h"

namespace traceviewer {

TimeRange::TimeRange(Microseconds start, Microseconds end) {
  if (start > end) {
    LOG(WARNING) << "Invalid TimeRange created with end (" << end
                 << ") < start (" << start << ").";
  }
  start_ = start;
  end_ = std::max(start_, end);
}

void TimeRange::Zoom(double zoom_factor) {
  if (zoom_factor <= 0) {
    // Zoom factor must be positive. This should not happen.
    return;
  }

  const Microseconds current_duration = duration();

  const double delta = current_duration * zoom_factor / 2.0;
  const Microseconds new_start = center() - delta;
  const Microseconds new_end = center() + delta;

  start_ = new_start;
  end_ = new_end;
}

}  // namespace traceviewer
