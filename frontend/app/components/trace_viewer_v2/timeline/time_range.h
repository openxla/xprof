#ifndef THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TIMELINE_TIME_RANGE_H_
#define THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TIMELINE_TIME_RANGE_H_

#include <algorithm>
#include <cmath>
#include <functional>
#include <utility>

#include "frontend/app/components/trace_viewer_v2/animation.h"
#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_event.h"

namespace traceviewer {

// Represents a 2D displacement/delta (start_delta, end_delta) between two
// TimeRanges. Unlike TimeRange, start_delta and end_delta can have independent
// signs and magnitudes.
struct TimeRangeDiff {
  Microseconds start_delta = 0.0;
  Microseconds end_delta = 0.0;

  TimeRangeDiff operator*(double factor) const {
    return {start_delta * factor, end_delta * factor};
  }

  TimeRangeDiff operator+(const TimeRangeDiff& other) const {
    return {start_delta + other.start_delta, end_delta + other.end_delta};
  }

  TimeRangeDiff operator-(const TimeRangeDiff& other) const {
    return {start_delta - other.start_delta, end_delta - other.end_delta};
  }

  bool operator==(const TimeRangeDiff& other) const {
    return start_delta == other.start_delta && end_delta == other.end_delta;
  }

  bool operator!=(const TimeRangeDiff& other) const {
    return !(*this == other);
  }
};

inline Microseconds abs(const TimeRangeDiff& diff) {
  return std::fabs(diff.start_delta) + std::fabs(diff.end_delta);
}

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
  // If the ranges do not overlap, returns a zero-duration range.
  TimeRange Intersect(const TimeRange& other) const {
    const Microseconds new_start = std::max(start_, other.start_);
    const Microseconds new_end = std::min(end_, other.end_);
    if (new_start > new_end) {
      return {new_start, new_start};
    }
    return {new_start, new_end};
  }

  // Returns true if this time range fully contains the other (considering
  // floating point tolerances).
  bool Contains(const TimeRange& other) const {
    auto almost_leq = [&](double a, double b) {
      double diff = a - b;
      if (diff <= 0) return true;
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
  // Zooms in or out around the pivot of the time range by zoom_factor.
  // If zoom_factor > 1, it zooms out, if zoom_factor < 1, it zooms in.
  void Zoom(double zoom_factor, Microseconds pivot);

  // Computes the displacement vector from `other` to `this` (this - other).
  TimeRangeDiff operator-(const TimeRange& other) const {
    return {start_ - other.start_, end_ - other.end_};
  }

  // Translates this TimeRange by a TimeRangeDiff displacement vector.
  TimeRange operator+(const TimeRangeDiff& diff) const {
    return {start_ + diff.start_delta, end_ + diff.end_delta};
  }

  TimeRange& operator+=(const TimeRangeDiff& diff) {
    start_ += diff.start_delta;
    end_ += diff.end_delta;
    return *this;
  }

  // Shifts the time range by a scalar time offset.
  TimeRange operator+(Microseconds val) const {
    return {start_ + val, end_ + val};
  }

  TimeRange operator-(Microseconds val) const {
    return {start_ - val, end_ - val};
  }

  TimeRange& operator+=(Microseconds val) {
    start_ += val;
    end_ += val;
    return *this;
  }

  bool operator==(const TimeRange& other) const {
    return start_ == other.start_ && end_ == other.end_;
  }

  bool operator!=(const TimeRange& other) const { return !(*this == other); }

 private:
  Microseconds start_ = 0.0, end_ = 0.0;

  static constexpr Microseconds kAbsoluteTolerance = 1e-4;
};

// Returns the magnitude of a TimeRange from origin, used by
// Animated<T>::Converged() to compute relative tolerance thresholds.
inline Microseconds abs(const TimeRange& range) {
  return std::fabs(range.start()) + std::fabs(range.end());
}

}  // namespace traceviewer

#endif  // THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TIMELINE_TIME_RANGE_H_
