/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XPROF_CONVERT_XPLANE_TO_STEP_EVENTS_H_
#define XPROF_CONVERT_XPLANE_TO_STEP_EVENTS_H_

#include <cstdint>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_visitor.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/utils/event_span.h"

namespace tensorflow {
namespace profiler {

using tsl::profiler::StatType;
using tsl::profiler::XEventVisitor;
using tsl::profiler::XLineVisitor;
using tsl::profiler::XPlaneVisitor;
using tsl::profiler::XStatVisitor;

// Aggregates the events in the line by group_id. If the device_step_events is
// non-null, CPU-only events will be filtered out (e.g. no matching group on the
// device). The async_input_pipeline_parents is used to propagate parent
// metadata to children in other XLines.
StepEvents ConvertHostThreadsXLineToStepEvents(
    const XLineVisitor& line, const StepEvents* device_step_events,
    const absl::flat_hash_set<std::pair<int64_t, int64_t>>&
        async_input_pipeline_parents = {});

// Convert the host threads in XPlane format to StepEvents format. If
// device_step_events is non-null, we will filter out events that only happens
// on CPU.
StepEvents ConvertHostThreadsXPlaneToStepEvents(
    const XPlane& host_trace, const StepEvents* device_step_events);

// Options for converting device trace XPlane to StepEvents.
struct DeviceStepEventsOptions {
  // Whether to collect per-core op metrics (e.g. from TPU OpLines) and GPU
  // stream events. When false, only step markers and step times are collected,
  // avoiding heavy OpMetricsDb construction and copying.
  bool collect_op_metrics = true;
};

// Convert the device trace in XLine format to StepEvents.
StepEvents ConvertDeviceTraceXLineToStepEvents(const XLineVisitor& line);

// Convert the device trace in XPlane format to StepEvents.
StepEvents ConvertDeviceTraceXPlaneToStepEvents(const XPlane& device_trace);

// Convert the device trace in XPlane format to StepEvents with custom options.
StepEvents ConvertDeviceTraceXPlaneToStepEventsWithOptions(
    const XPlane& device_trace, const DeviceStepEventsOptions& options);

// Convenience function to extract only step markers and step times from
// device trace.
inline StepEvents ConvertDeviceTraceXPlaneToStepMarkers(
    const XPlane& device_trace) {
  return ConvertDeviceTraceXPlaneToStepEventsWithOptions(
      device_trace, DeviceStepEventsOptions{.collect_op_metrics = false});
}

}  // namespace profiler
}  // namespace tensorflow

#endif  // XPROF_CONVERT_XPLANE_TO_STEP_EVENTS_H_
