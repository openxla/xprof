#include "xprof/convert/trace_viewer/trace_options.h"

#include <memory>

#include "absl/container/flat_hash_set.h"
#include "xprof/convert/tool_options.h"
#include "xprof/convert/trace_viewer/trace_events_filter_interface.h"
#include "xprof/convert/trace_viewer/trace_events_to_json.h"
#include "xprof/convert/trace_viewer/trace_utils.h"

namespace tensorflow {
namespace profiler {

void TraceEventsFilter::SetUp(const Trace& trace) {
  for (const auto& [device_id, device] : trace.devices()) {
    if (IsTpuCoreDeviceName(device.name())) {
      device_type_ = TraceDeviceType::kTpu;
      tpu_core_devices_.insert(device_id);
    } else if (MaybeTpuNonCoreDeviceName(device.name())) {
      tpu_noncore_devices_.insert(device_id);
    }
  }
}

bool TraceEventsFilter::Filter(const TraceEvent& event) {
  switch (device_type_) {
    case TraceDeviceType::kUnknownDevice:
      break;
    case TraceDeviceType::kTpu:
      if (tpu_noncore_devices_.contains(
              event.device_id()) ||
              tpu_core_devices_.contains(event.device_id())) {
        // Filter intermediate DMA flow events unless "Full DMA" is checked.
        if (IsFlowMid(event)) return !options_.full_dma;
      }
      break;
    case TraceDeviceType::kGpu:
      break;
  }
  return false;
}

TraceOptions TraceOptionsFromToolOptions(const ToolOptions& tool_options) {
  TraceOptions options;
  options.full_dma =
      GetParamWithDefault<bool>(tool_options, kFullDma, options.full_dma);
  return options;
}

JsonTraceOptions::Details TraceOptionsToDetails(TraceDeviceType device_type,
                                                const TraceOptions& options) {
  switch (device_type) {
    case TraceDeviceType::kUnknownDevice:
      return {};
    case TraceDeviceType::kTpu:
      return {
          {kFullDma, options.full_dma},
      };
    case TraceDeviceType::kGpu:
      return {};
  }
}

std::unique_ptr<tensorflow ::profiler::TraceEventsFilterInterface>
TraceEventsFilterFromTraceOptions(const TraceOptions& options) {
  return std::make_unique<TraceEventsFilter>(options);
}

}  // namespace profiler
}  // namespace tensorflow
