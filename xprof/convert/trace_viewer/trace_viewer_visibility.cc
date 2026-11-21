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
#include "xprof/convert/trace_viewer/trace_viewer_visibility.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "xla/tsl/profiler/utils/timespan.h"
#include "xprof/convert/trace_viewer/trace_options.h"
#include "xprof/convert/trace_viewer/trace_utils.h"
#include "plugin/xprof/protobuf/trace_events.pb.h"

namespace tensorflow {
namespace profiler {

TraceViewerVisibility::Window TraceViewerVisibility::Window::ClampToTrace(
    const Trace& trace) const {
  if (kind_ == Kind::kEmpty) {
    return Empty();
  }

  uint64_t start_time_ps = span_.begin_ps();
  std::optional<uint64_t> end_time_ps = EndPs();

  if (trace.has_min_timestamp_ps() &&
      start_time_ps < trace.min_timestamp_ps()) {
    VLOG(1) << "Adjusting start_time_ps from " << start_time_ps << " to "
            << trace.min_timestamp_ps();
    start_time_ps = trace.min_timestamp_ps();
  }
  if (trace.has_max_timestamp_ps()) {
    if (!end_time_ps.has_value()) {
      VLOG(1) << "Clamping open-ended end_time_ps to "
              << trace.max_timestamp_ps();
      end_time_ps = trace.max_timestamp_ps();
    } else if (*end_time_ps > trace.max_timestamp_ps()) {
      VLOG(1) << "Adjusting end_time_ps from " << *end_time_ps << " to "
              << trace.max_timestamp_ps();
      end_time_ps = trace.max_timestamp_ps();
    }
  }
  return FromEndPoints(start_time_ps, end_time_ps);
}

TraceViewerVisibility::TraceViewerVisibility(Window window,
                                             uint64_t resolution_ps)
    : window_(window), resolution_ps_(window.IsEmpty() ? 0 : resolution_ps) {}

TraceViewerVisibility::TraceViewerVisibility(
    tsl::profiler::Timespan visible_span, uint64_t resolution_ps)
    : TraceViewerVisibility(Window::FromLegacySpan(visible_span),
                            resolution_ps) {}

bool TraceViewerVisibility::Visible(const TraceEvent& event) {
  // Events outside the visible window (including empty windows) are not
  // visible.
  tsl::profiler::Timespan span(event.timestamp_ps(), event.duration_ps());
  if (!window_.Overlaps(span)) return false;

  // If resolution is zero, no downsampling.
  if (resolution_ps_ == 0) return true;

  return VisibleAtResolution(event);
}

bool TraceViewerVisibility::VisibleAtResolution(const TraceEvent& event) {
  DCHECK_NE(resolution_ps_, 0);
  // A counter event is visible if its distance from the last visible counter
  // event in the same device is large enough. The first counter event in a
  // row is always visible.
  if (!event.has_resource_id()) {
    CounterRowId counter_row_id(event.device_id(), event.name());
    auto iter = last_counter_timestamp_ps_.find(counter_row_id);
    bool found = (iter != last_counter_timestamp_ps_.end());
    bool visible =
        !found || ((event.timestamp_ps() - iter->second) >= resolution_ps_);
    if (visible) {
      if (found) {
        iter->second = event.timestamp_ps();
      } else {
        last_counter_timestamp_ps_.emplace(counter_row_id,
                                           event.timestamp_ps());
      }
    }
    return visible;
  }

  // An event is visible if its duration is large enough.
  tsl::profiler::Timespan span(event.timestamp_ps(), event.duration_ps());
  bool visible = (span.duration_ps() >= resolution_ps_);

  auto& row = rows_[RowId(event.device_id(), event.resource_id())];

  // An event is visible if it is the first event at its nesting depth, or its
  // distance from the last visible event at the same depth is large enough.
  size_t depth = row.Depth(span.begin_ps());
  if (!visible) {
    auto last_end_timestamp_ps = row.LastEndTimestampPs(depth);
    visible = !last_end_timestamp_ps ||
              (span.begin_ps() - *last_end_timestamp_ps >= resolution_ps_);
  }

  // A flow event is visible if the first event in the flow is visible.
  // The first event in the flow is visible if the distance between its arrow
  // binding point and the previous visible arrow binding point is large enough.
  // The arrow binds to the end time of the complete event.
  if (event.has_flow_id()) {
    // Only compute visibility for the first event in the flow.
    auto result = flows_.try_emplace(event.flow_id(), visible);
    if (!visible) {
      if (result.second) {
        auto last_flow_timestamp_ps = row.LastFlowTimestampPs();
        result.first->second =
            !last_flow_timestamp_ps ||
            (span.end_ps() - *last_flow_timestamp_ps >= resolution_ps_);
      }
      visible = result.first->second;
    }
    // If we see the last event in the flow, remove it from the map. We don't
    // use flow_entry_type for determining the first event in the flow because
    // for cross-host flows it won't be FLOW_START.
    // This removal prevents the map from growing too large.
    if (event.flow_entry_type() == TraceEvent::FLOW_END) {
      flows_.erase(result.first);
    }
    if (visible) {
      row.SetLastFlowTimestampPs(span.end_ps());
    }
  }

  if (visible) {
    row.SetLastEndTimestampPs(depth, span.end_ps());
  }
  return visible;
}

void TraceViewerVisibility::SetVisibleAtResolution(const TraceEvent& event) {
  DCHECK_NE(resolution_ps_, 0);
  if (!event.has_resource_id()) {
    CounterRowId counter_row_id(event.device_id(), event.name());
    last_counter_timestamp_ps_.insert_or_assign(counter_row_id,
                                                event.timestamp_ps());

  } else {
    tsl::profiler::Timespan span(event.timestamp_ps(), event.duration_ps());
    auto& row = rows_[RowId(event.device_id(), event.resource_id())];
    if (event.has_flow_id()) {
      if (event.flow_entry_type() == TraceEvent::FLOW_END) {
        flows_.erase(event.flow_id());
      } else {
        flows_.try_emplace(event.flow_id(), true);
      }
      row.SetLastFlowTimestampPs(span.end_ps());
    }
    size_t depth = row.Depth(span.begin_ps());
    row.SetLastEndTimestampPs(depth, span.end_ps());
  }
}

size_t TraceViewerVisibility::RowVisibility::Depth(
    uint64_t begin_timestamp_ps) const {
  size_t depth = 0;
  for (; depth < last_end_timestamp_ps_.size(); ++depth) {
    if (last_end_timestamp_ps_[depth] <= begin_timestamp_ps) break;
  }
  return depth;
}

TraceVisibilityFilter::TraceVisibilityFilter(
    TraceViewerVisibility::Window window, double resolution,
    const TraceOptions& options)
    : options_(options),
      configured_window_(window),
      resolution_(resolution),
      visibility_(configured_window_, ResolutionPs(configured_window_)) {}

TraceVisibilityFilter::TraceVisibilityFilter(
    tsl::profiler::Timespan visible_span, double resolution)
    : TraceVisibilityFilter(visible_span, resolution, TraceOptions()) {}

TraceVisibilityFilter::TraceVisibilityFilter(
    tsl::profiler::Timespan visible_span, double resolution,
    const TraceOptions& options)
    : TraceVisibilityFilter(
          TraceViewerVisibility::Window::FromLegacySpan(visible_span),
          resolution, options) {}

void TraceVisibilityFilter::SetUp(const Trace& trace) {
  tpu_core_devices_.clear();
  tpu_noncore_devices_.clear();
  for (const auto& [device_id, device] : trace.devices()) {
    if (IsTpuCoreDeviceName(device.name())) {
      tpu_core_devices_.insert(device_id);
    } else if (MaybeTpuNonCoreDeviceName(device.name())) {
      tpu_noncore_devices_.insert(device_id);
    }
  }
  // Recompute the active window from the immutable configured window so
  // repeated SetUp() calls do not latch prior trace bounds.
  const TraceViewerVisibility::Window active_window =
      configured_window_.ClampToTrace(trace);
  visibility_ =
      TraceViewerVisibility(active_window, ResolutionPs(active_window));
}

void TraceVisibilityFilter::UpdateVisibility(double resolution) {
  resolution_ = resolution;
  const TraceViewerVisibility::Window active_window =
      visibility_.VisibleWindow();
  visibility_ =
      TraceViewerVisibility(active_window, ResolutionPs(active_window));
}

bool TraceVisibilityFilter::Filter(const TraceEvent& event) {
  const TraceViewerVisibility::Window& window = visibility_.VisibleWindow();
  if (window.IsEmpty()) {
    return true;
  }
  // Filter intermediate DMA flow events unless "Full DMA" is checked and the
  // event overlaps the visible window.
  if (IsFlowMid(event) && (tpu_core_devices_.contains(event.device_id()) ||
                           tpu_noncore_devices_.contains(event.device_id()))) {
    return !options_.full_dma ||
           !window.Overlaps(tsl::profiler::Timespan(event.timestamp_ps(),
                                                    event.duration_ps()));
  }
  return !visibility_.Visible(event);
}

uint64_t TraceVisibilityFilter::ResolutionPs(
    const TraceViewerVisibility::Window& window) const {
  if (!std::isfinite(resolution_) || resolution_ <= 0.0) {
    return 0;
  }
  const std::optional<uint64_t> end_ps = window.EndPs();
  if (!end_ps.has_value()) {
    return 0;
  }
  const uint64_t duration_ps = *end_ps - window.BeginPs();
  if (resolution_ == 1.0) {
    return duration_ps;
  }
  // `uint64_t::max()` rounds up to 2^64 in IEEE-754 double, so `>=` clamps any
  // value that would overflow `uint64_t` (including `+inf`) before casting.
  constexpr double kUint64Limit =
      static_cast<double>(std::numeric_limits<uint64_t>::max());
  const double scaled =
      std::round(static_cast<double>(duration_ps) / resolution_);
  if (scaled >= kUint64Limit) {
    return std::numeric_limits<uint64_t>::max();
  }
  return static_cast<uint64_t>(scaled);
}

}  // namespace profiler
}  // namespace tensorflow
