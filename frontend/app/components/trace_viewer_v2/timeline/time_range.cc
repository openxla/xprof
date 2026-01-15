#include "xprof/frontend/app/components/trace_viewer_v2/timeline/time_range.h"

#include <algorithm>

#include "absl/log/log.h"
#include "xprof/frontend/app/components/trace_viewer_v2/trace_helper/trace_event.h"

namespace traceviewer {

TimeRange::TimeRange(Microseconds start, Microseconds end) {
  if (start > end) {
    LOG(WARNING) << "Invalid TimeRange created with end (" << end
                 << ") < start (" << start << ").";
  }
  start_ = start;
  end_ = std::max(start_, end);
}

TimeRange TimeRange::Scale(double ratio) const {
  const double delta = duration() * ratio / 2.0;
  return {center() - delta, center() + delta};
}

void TimeRange::Zoom(double zoom_factor) { Zoom(zoom_factor, center()); }

void TimeRange::Zoom(double zoom_factor, Microseconds pivot) {
  if (zoom_factor <= 0) {
    // Zoom factor must be positive. This should not happen.
    return;
  }

  const Microseconds current_duration = duration();

  Microseconds new_start = pivot - (pivot - start_) * zoom_factor;
  Microseconds new_end = pivot + (end_ - pivot) * zoom_factor;

  if (new_start < 0) {
    // This condition occurs when the calculated `new_start` falls below zero.
    // While often caused by zooming out with a pivot near the start, it can
    // also happen in other scenarios. If this happens, clamp start to 0.0 and
    // set end to `current_duration * zoom_factor` to maintain the correct
    // zoomed duration.
    start_ = 0.0;
    end_ = current_duration * zoom_factor;
  } else {
    start_ = new_start;
    end_ = new_end;
  }
}

}  // namespace traceviewer
