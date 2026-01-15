#ifndef THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TIMELINE_TIME_RANGE_H_
#define THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TIMELINE_TIME_RANGE_H_

#include <algorithm>
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

  // Returns the intersection of this time range and the other.
  // If the ranges do not overlap, returns a range with start > end (empty).
  TimeRange Intersect(const TimeRange& other) const {
    return {std::max(start_, other.start_), std::min(end_, other.end_)};
  }

  // Returns true if this time range fully contains the other (considering
  // floating point tolerances).
  bool Contains(const TimeRange& other) const {
    // We use a tolerance to handle floating point inaccuracies, similar to
    // Animation::Converged().

    auto almost_leq = [&](double a, double b) {
      double diff = a - b;
      // If a <= b, diff <= 0.
      if (diff <= 0) return true;
      // If a > b, check if diff is within tolerance.
      return diff < kAbsoluteTolerance;
    };

    auto almost_geq = [&](double a, double b) {
      double diff = b - a;
      if (diff <= 0) return true;
      return diff < kAbsoluteTolerance;
    };

    return almost_leq(start_, other.start_) && almost_geq(end_, other.end_);
  }

  // Scales the time range around its center by the given ratio.
  // Returns a new TimeRange and does not modify the current instance.
  // This is useful for calculating derived ranges (e.g., for data re-fetching)
  // without altering the current visible range.
  TimeRange Scale(double ratio) const;

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

  static constexpr Microseconds kAbsoluteTolerance = 1e-4;
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

#endif  // THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TIMELINE_TIME_RANGE_H_
