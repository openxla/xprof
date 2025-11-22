#ifndef PERFTOOLS_ACCELERATORS_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TIMELINE_TIME_RANGE_H_
#define PERFTOOLS_ACCELERATORS_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TIMELINE_TIME_RANGE_H_

#include <cmath>

#include "xprof/frontend/app/components/trace_viewer_v2/trace_helper/trace_event.h"

namespace traceviewer {

// Represents a time interval [start, end].
class TimeRange {
 public:
  TimeRange() = default;

  // Initializes a TimeRange. If end is less than start, it is clamped to start.
  TimeRange(Microseconds start, Microseconds end);

  static TimeRange Zero() { return {0.0, 0.0}; }

  Microseconds start() const { return start_; }
  Microseconds end() const { return end_; }

  Microseconds duration() const { return end_ - start_; }

  Microseconds center() const { return start_ + duration() / 2.0; }

  // Expands this time range to include the given time range.
  void Encompass(const TimeRange& other) {
    start_ = std::fmin(start_, other.start_);
    end_ = std::fmax(end_, other.end_);
  }

  // Zooms in or out around the center of the time range by zoom_factor.
  // If zoom_factor > 1, it zooms out, if zoom_factor < 1, it zooms in.
  void Zoom(double zoom_factor);

  // Adds two TimeRanges. While not representing a real-world time range
  // operation, this is used by `Animated<TimeRange>` for linear interpolation
  // in its `Update` method.
  TimeRange operator+(const TimeRange& other) const {
    return {start_ + other.start_, end_ + other.end_};
  }

  // Subtracts two TimeRanges. While not representing a real-world time range
  // operation, this is used by `Animated<TimeRange>` to calculate the
  // difference between two TimeRanges, for example, to check if an animation
  // has completed.
  TimeRange operator-(const TimeRange& other) const {
    return {start_ - other.start_, end_ - other.end_};
  }

  TimeRange operator+(Microseconds val) const {
    return {start_ + val, end_ + val};
  }

  TimeRange operator-(Microseconds val) const {
    return {start_ - val, end_ - val};
  }

  TimeRange operator*(double val) const { return {start_ * val, end_ * val}; }

  TimeRange& operator+=(Microseconds val) {
    start_ += val;
    end_ += val;
    return *this;
  }

  bool operator==(const TimeRange& other) const {
    return start_ == other.start_ && end_ == other.end_;
  }

 private:
  Microseconds start_ = 0.0, end_ = 0.0;
};

// Defines an abs() operation for TimeRange. This is used by
// `Animated<TimeRange>::Update()` to check for convergence. The input `range`
// is typically the result of `current_ - target_`. The sum of the absolute
// values of `range.start()` and `range.end()` provides a metric for the total
// magnitude of the difference between the two TimeRanges, considering both
// their start and end points. Use lowercase to be found by Argument-Dependent
// Lookup (ADL).
// Defined as inline in the header to allow template instantiation
// (e.g. Animated<TimeRange>) and prevent multiple definition errors.
inline Microseconds abs(const TimeRange& range) {
  return std::fabs(range.start()) + std::fabs(range.end());
}

}  // namespace traceviewer

#endif  // PERFTOOLS_ACCELERATORS_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TIMELINE_TIME_RANGE_H_
