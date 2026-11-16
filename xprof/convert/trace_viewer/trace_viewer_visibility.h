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

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "xla/tsl/profiler/utils/timespan.h"
#include "xprof/convert/trace_viewer/trace_events_filter_interface.h"
#include "xprof/convert/trace_viewer/trace_options.h"
#include "xprof/convert/trace_viewer/trace_utils.h"
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
  enum class InstantSpanPolicy { kUnconstrained, kExactInstant };

  // Create with visible timespan and resolution (in picoseconds).
  // The visible timespan must have non-zero duration unless
  // `instant_span_policy` is `InstantSpanPolicy::kExactInstant`.
  // If resolution is zero, no events are downsampled.
  explicit TraceViewerVisibility(tsl::profiler::Timespan visible_span,
                                 uint64_t resolution_ps = 0,
                                 InstantSpanPolicy instant_span_policy =
                                     InstantSpanPolicy::kUnconstrained);

  // Returns true if the event overlaps the visible span and is distinguishable
  // at resolution_ps.
  bool Visible(const TraceEvent& event);

  // Returns true if the event is distinguishable at resolution_ps.
  bool VisibleAtResolution(const TraceEvent& event);

  // Records that event is distinguishable at resolution_ps.
  void SetVisibleAtResolution(const TraceEvent& event);

  tsl::profiler::Timespan VisibleSpan() const { return visible_span_; }
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
  tsl::profiler::Timespan visible_span_;
  uint64_t resolution_ps_;
  InstantSpanPolicy instant_span_policy_ = InstantSpanPolicy::kUnconstrained;

  // Visibility data for all rows.
  absl::flat_hash_map<RowId, RowVisibility> rows_;

  // Visibility of flows.
  absl::flat_hash_map<uint64_t /*flow_id*/, bool> flows_;

  // Visibility data for counter events.
  absl::flat_hash_map<CounterRowId, uint64_t> last_counter_timestamp_ps_;
};

struct VisibilityWindowOptions {
  std::optional<bool> has_end_time = std::nullopt;
  bool empty_window = false;
};

class TraceVisibilityFilter : public TraceEventsFilterInterface {
 public:
  // If visible_span.Empty(), all events are visible before SetUp().
  // If resolution is 0.0, events aren't downsampled.
  TraceVisibilityFilter(tsl::profiler::Timespan visible_span, double resolution)
      : TraceVisibilityFilter(visible_span, resolution, TraceOptions()) {}

  tsl::profiler::Timespan VisibleSpan() const {
    return visibility_.VisibleSpan();
  }
  // Creates a visibility filter over `visible_span` at `resolution`.
  // If `window_options.has_end_time` is omitted, defaults to
  // `!visible_span.Instant()`, so zero-duration spans expand `end_time_ps` to
  // `trace.max_timestamp_ps()` in `SetUp(trace)` unless `has_end_time` is
  // explicitly true. If `window_options.empty_window` is true, all events are
  // filtered out.
  explicit TraceVisibilityFilter(tsl::profiler::Timespan visible_span,
                                 double resolution, const TraceOptions& options,
                                 VisibilityWindowOptions window_options = {})
      : options_(options),
        resolution_(resolution),
        has_end_time_(
            window_options.has_end_time.value_or(!visible_span.Instant())),
        empty_window_(window_options.empty_window),
        visibility_(visible_span, ResolutionPs(visible_span.duration_ps()),
                    GetInstantSpanPolicy(visible_span)) {}

  uint64_t ResolutionPs() const { return visibility_.ResolutionPs(); }

  void SetUp(const Trace& trace) override {
    // Update visible_span with trace bounds and recompute the resolution in
    // picoseconds.
    tsl::profiler::Timespan visible_span = VisibleSpan();
    uint64_t start_time_ps = visible_span.begin_ps();
    uint64_t end_time_ps = visible_span.end_ps();
    if (trace.has_min_timestamp_ps() &&
        start_time_ps < trace.min_timestamp_ps()) {
      VLOG(1) << "Adjusting start_time_ps from " << start_time_ps << " to "
              << trace.min_timestamp_ps();
      start_time_ps = trace.min_timestamp_ps();
    }
    if (trace.has_max_timestamp_ps() &&
        (!has_end_time_ || end_time_ps > trace.max_timestamp_ps())) {
      VLOG(1) << "Adjusting end_time_ps from " << end_time_ps << " to "
              << trace.max_timestamp_ps();
      end_time_ps = trace.max_timestamp_ps();
    }
    if (empty_window_ || start_time_ps > end_time_ps) {
      empty_window_ = true;
      visible_span = tsl::profiler::Timespan(0, 0);
    } else {
      visible_span =
          tsl::profiler::Timespan::FromEndPoints(start_time_ps, end_time_ps);
    }
    visibility_ = TraceViewerVisibility(
        visible_span, ResolutionPs(visible_span.duration_ps()),
        GetInstantSpanPolicy(visible_span));
    for (const auto& [device_id, device] : trace.devices()) {
      if (IsTpuCoreDeviceName(device.name())) {
        tpu_core_devices_.insert(device_id);
      } else if (MaybeTpuNonCoreDeviceName(device.name())) {
        tpu_noncore_devices_.insert(device_id);
      }
    }
  }

  // Updates the visibility based on `resolution`.
  void UpdateVisibility(double resolution) {
    resolution_ = resolution;
    const tsl::profiler::Timespan visible_span = visibility_.VisibleSpan();
    visibility_ = TraceViewerVisibility(
        visible_span, ResolutionPs(visible_span.duration_ps()),
        GetInstantSpanPolicy(visible_span));
  }

  bool Filter(const TraceEvent& event) override {
    if (empty_window_) {
      return true;
    }
    // Filter intermediate DMA flow events unless "Full DMA" is checked.
    if (IsFlowMid(event) &&
        (tpu_core_devices_.contains(event.device_id()) ||
         tpu_noncore_devices_.contains(event.device_id()))) {
      return !options_.full_dma;
    }
    return !visibility_.Visible(event);
  }

 private:
  const TraceOptions options_;
  // Returns the minimum duration in picoseconds that an event must have in
  // order to be visible.
  uint64_t ResolutionPs(uint64_t duration_ps) {
    return (resolution_ == 0.0) ? 0 : std::llround(duration_ps / resolution_);
  }

  TraceViewerVisibility::InstantSpanPolicy GetInstantSpanPolicy(
      const tsl::profiler::Timespan& visible_span) const {
    return (has_end_time_ || visible_span.begin_ps() > 0)
               ? TraceViewerVisibility::InstantSpanPolicy::kExactInstant
               : TraceViewerVisibility::InstantSpanPolicy::kUnconstrained;
  }

  double resolution_;  // number of visible events per row
  bool has_end_time_ = true;
  bool empty_window_ = false;
  TraceViewerVisibility visibility_;
  absl::flat_hash_set<uint32_t /*device_id*/> tpu_noncore_devices_;
  absl::flat_hash_set<uint32_t /*device_id*/> tpu_core_devices_;
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // THIRD_PARTY_XPROF_CONVERT_TRACE_VIEWER_TRACE_VIEWER_VISIBILITY_H_
