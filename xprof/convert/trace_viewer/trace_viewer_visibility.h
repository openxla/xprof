/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef THIRD_PARTY_XPROF_CONVERT_TRACE_VIEWER_TRACE_VIEWER_VISIBILITY_H_
#define THIRD_PARTY_XPROF_CONVERT_TRACE_VIEWER_TRACE_VIEWER_VISIBILITY_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "xla/tsl/profiler/utils/timespan.h"
#include "xprof/convert/trace_viewer/trace_events_filter_interface.h"
#include "xprof/convert/trace_viewer/trace_options.h"
#include "plugin/xprof/protobuf/trace_events.pb.h"

namespace tensorflow {
namespace profiler {

// Determines whether an event will be visible in trace viewer within a visible
// tsl::profiler::Timespan at a certain resolution.
// Events must be evaluated in order by timestamp, because when an event is
// determined to be visible, the internal state of this class is updated.
// TODO(b/522205373): Cleanup legacy trace_events implementation once lite trace
// events is fully enabled.
class TraceViewerVisibility {
 public:
  // Immutable time window with four explicit states (`kUnconstrained`,
  // `kEmpty`, `kBounded`, `kOpenEnded`) so callers cannot misinterpret a
  // zero-duration `Timespan`.
  class Window {
   public:
    // Creates an unconstrained window matching all events.
    static Window Unconstrained() {
      return Window(Kind::kUnconstrained, tsl::profiler::Timespan(0, 0));
    }

    // Creates an empty window matching no events.
    static Window Empty() {
      return Window(Kind::kEmpty, tsl::profiler::Timespan(0, 0));
    }

    // Creates a closed window `[start_ps, *end_ps]` when `end_ps` is provided
    // (or `Empty()` if `start_ps > *end_ps`), `Unconstrained()` when `end_ps`
    // is `std::nullopt` and `start_ps == 0`, or an open-ended window
    // `[start_ps, +inf)` when `end_ps` is `std::nullopt` and `start_ps > 0`.
    static Window FromEndPoints(uint64_t start_ps,
                                std::optional<uint64_t> end_ps) {
      if (end_ps.has_value()) {
        if (start_ps > *end_ps) {
          return Empty();
        }
        return Window(Kind::kBounded, tsl::profiler::Timespan::FromEndPoints(
                                          start_ps, *end_ps));
      }
      if (start_ps == 0) {
        return Unconstrained();
      }
      return Window(Kind::kOpenEnded, tsl::profiler::Timespan(start_ps, 0));
    }

    // Interprets a legacy `Timespan`: zero-duration spans become open-ended
    // `[begin_ps(), +inf)` (`Unconstrained()` when `begin_ps() == 0`), while
    // non-zero spans become closed `[begin_ps(), end_ps()]`.
    static Window FromLegacySpan(tsl::profiler::Timespan span) {
      if (span.Instant()) {
        return FromEndPoints(span.begin_ps(), std::nullopt);
      }
      return Window(Kind::kBounded, span);
    }

    // Returns true if this window matches no events.
    bool IsEmpty() const { return kind_ == Kind::kEmpty; }

    // Returns true if this window disables time-window filtering.
    bool IsUnconstrained() const { return kind_ == Kind::kUnconstrained; }

    // Returns the lower timestamp bound in picoseconds (0 when unconstrained or
    // empty).
    uint64_t BeginPs() const { return span_.begin_ps(); }

    // Returns the inclusive upper timestamp bound in picoseconds when bounded,
    // or `std::nullopt` for unconstrained, open-ended, or empty windows.
    std::optional<uint64_t> EndPs() const {
      if (kind_ != Kind::kBounded) return std::nullopt;
      return span_.end_ps();
    }

    // Returns true if `span` overlaps this window.
    bool Overlaps(tsl::profiler::Timespan span) const {
      switch (kind_) {
        case Kind::kEmpty:
          return false;
        case Kind::kUnconstrained:
          return true;
        case Kind::kOpenEnded:
          return span.end_ps() >= span_.begin_ps();
        case Kind::kBounded:
          return span_.Overlaps(span);
      }
    }

    // Clamps this window against any `min_timestamp_ps` / `max_timestamp_ps`
    // bounds present on `trace`, returning `Empty()` if clamping yields an
    // inverted interval.
    Window ClampToTrace(const Trace& trace) const;

   private:
    enum class Kind {
      kUnconstrained,
      kEmpty,
      kBounded,
      kOpenEnded,
    };

    Window(Kind kind, tsl::profiler::Timespan span)
        : kind_(kind), span_(span) {}

    Kind kind_;
    tsl::profiler::Timespan span_;
  };

  // Creates a visibility filter over `window` at `resolution_ps` (in
  // picoseconds). When `window.IsEmpty()`, `resolution_ps` is normalized to
  // `0`. If `resolution_ps` is zero, no events are downsampled.
  explicit TraceViewerVisibility(Window window, uint64_t resolution_ps = 0);

  // Creates a visibility filter from a legacy `Timespan` interpreted via
  // `Window::FromLegacySpan(visible_span)`. If `resolution_ps` is zero, no
  // events are downsampled.
  explicit TraceViewerVisibility(tsl::profiler::Timespan visible_span,
                                 uint64_t resolution_ps = 0);

  // Returns true if the event overlaps the visible window and is
  // distinguishable at resolution_ps.
  bool Visible(const TraceEvent& event);

  // Returns true if the event is distinguishable at resolution_ps.
  bool VisibleAtResolution(const TraceEvent& event);

  // Records that event is distinguishable at resolution_ps.
  void SetVisibleAtResolution(const TraceEvent& event);

  // Returns the active visibility window.
  const Window& VisibleWindow() const { return window_; }

  // TODO(tf-profiler) Rename ResolutionPs and resolution_ps to be more
  // self-explanatory (eg. MinDurationPs)
  uint64_t ResolutionPs() const { return resolution_ps_; }

  const absl::flat_hash_map<uint64_t, bool>& Flows() const { return flows_; }

 private:
  // Identifier for one Trace Viewer row.
  using RowId = std::pair<uint32_t /*device_id*/, uint32_t /*resource_id*/>;
  using CounterRowId = std::pair<uint32_t /*device_id*/, std::string /*name*/>;

  // Visibility for one Trace Viewer row.
  class RowVisibility {
   public:
    // Returns the nesting depth for an event at begin_timestamp_ps.
    size_t Depth(uint64_t begin_timestamp_ps) const;

    // Returns the end_timestamp_ps of the last visible event at the given
    // nesting depth.
    std::optional<uint64_t> LastEndTimestampPs(size_t depth) const {
      std::optional<uint64_t> result;
      if (depth < last_end_timestamp_ps_.size()) {
        result = last_end_timestamp_ps_[depth];
      }
      return result;
    }

    // Returns the arrow timestamp of the last visible flow event.
    std::optional<uint64_t> LastFlowTimestampPs() const {
      return last_flow_timestamp_ps_;
    }

    // Sets the last visible timestamp at the given nesting depth.
    void SetLastEndTimestampPs(size_t depth, uint64_t timestamp_ps) {
      last_end_timestamp_ps_.resize(depth);
      last_end_timestamp_ps_.push_back(timestamp_ps);
    }

    // Sets the last visible arrow timestamp.
    void SetLastFlowTimestampPs(uint64_t timestamp_ps) {
      last_flow_timestamp_ps_ = timestamp_ps;
    }

   private:
    // Stack of most recently visible event end times. A stack is used to handle
    // nested events.
    std::vector<uint64_t> last_end_timestamp_ps_;

    // Timestamp of the arrow binding point of the last visible flow event.
    std::optional<uint64_t> last_flow_timestamp_ps_;
  };

  // Constructor arguments.
  Window window_;
  uint64_t resolution_ps_;

  // Visibility data for all rows.
  absl::flat_hash_map<RowId, RowVisibility> rows_;

  // Visibility of flows.
  absl::flat_hash_map<uint64_t /*flow_id*/, bool> flows_;

  // Visibility data for counter events.
  absl::flat_hash_map<CounterRowId, uint64_t> last_counter_timestamp_ps_;
};

class TraceVisibilityFilter : public TraceEventsFilterInterface {
 public:
  // Creates a visibility filter over `window` at `resolution`, using `options`
  // to configure trace filtering (such as `full_dma`). If `resolution` is
  // <= 0.0 or non-finite, events are not downsampled.
  explicit TraceVisibilityFilter(TraceViewerVisibility::Window window,
                                 double resolution,
                                 const TraceOptions& options = TraceOptions());

  // Creates a visibility filter over `visible_span` (interpreted via
  // `TraceViewerVisibility::Window::FromLegacySpan`) at `resolution`.
  // If `visible_span.Empty()`, all events are visible before `SetUp()`.
  // If `visible_span.Instant()` with non-zero `begin_ps()`, treats the upper
  // bound as open-ended `[begin_ps(), +inf)` until `SetUp()` clamps to trace
  // bounds. If `resolution` is <= 0.0 or non-finite, events are not
  // downsampled.
  TraceVisibilityFilter(tsl::profiler::Timespan visible_span,
                        double resolution);

  // Same as above, using `options` to configure trace filtering (such as
  // `full_dma`).
  explicit TraceVisibilityFilter(tsl::profiler::Timespan visible_span,
                                 double resolution,
                                 const TraceOptions& options);

  // Returns the active visibility window.
  const TraceViewerVisibility::Window& VisibleWindow() const {
    return visibility_.VisibleWindow();
  }

  uint64_t ResolutionPs() const { return visibility_.ResolutionPs(); }

  void SetUp(const Trace& trace) override;

  // Updates the visibility based on `resolution`.
  void UpdateVisibility(double resolution);

  bool Filter(const TraceEvent& event) override;

 private:
  // Returns the minimum duration in picoseconds that an event must have in
  // order to be visible within `window`, or `0` when downsampling is disabled
  // or `window` has no upper bound.
  uint64_t ResolutionPs(const TraceViewerVisibility::Window& window) const;

  const TraceOptions options_;
  const TraceViewerVisibility::Window configured_window_;
  double resolution_;  // number of visible events per row
  TraceViewerVisibility visibility_;
  absl::flat_hash_set<uint32_t /*device_id*/> tpu_noncore_devices_;
  absl::flat_hash_set<uint32_t /*device_id*/> tpu_core_devices_;
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // THIRD_PARTY_XPROF_CONVERT_TRACE_VIEWER_TRACE_VIEWER_VISIBILITY_H_
