#ifndef THIRD_PARTY_XPROF_CONVERT_TRACE_VIEWER_TRACE_UTILS_H
#define THIRD_PARTY_XPROF_CONVERT_TRACE_VIEWER_TRACE_UTILS_H

#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"

namespace tensorflow {
namespace profiler {

inline bool IsTpuCoreDeviceName(absl::string_view device_name) {
  constexpr absl::string_view kDeprecatedCoreName = "TensorNode";
  constexpr absl::string_view kDeprecatedCoreNameForCloud = "TPU Core";
  return absl::StrContains(device_name, ::tsl::profiler::kTpuPlanePrefix) ||
         absl::StrContains(device_name, kDeprecatedCoreName) ||
         absl::StrContains(device_name, kDeprecatedCoreNameForCloud);
}

inline bool MaybeTpuHostInterfaceDeviceName(absl::string_view device_name) {
  return absl::StrContains(device_name, "Host Interface");
}

inline bool IsTpuHbmDeviceName(absl::string_view device_name) {
  return absl::StrContains(device_name, "HBM");
}

inline bool IsTpuIciRouterDeviceName(absl::string_view device_name) {
  return absl::StrContains(device_name, "ICI Router");
}

inline bool MaybeTpuNonCoreDeviceName(absl::string_view device_name) {
  return absl::StrContains(device_name,
                           ::tsl::profiler::kTpuNonCorePlaneNamePrefix) &&
         (MaybeTpuHostInterfaceDeviceName(device_name) ||
          IsTpuHbmDeviceName(device_name) ||
          IsTpuIciRouterDeviceName(device_name));
}

static constexpr int kMaxDevicesPerHost = 1000;

}  // namespace profiler
}  // namespace tensorflow

#endif  // THIRD_PARTY_XPROF_CONVERT_TRACE_VIEWER_TRACE_UTILS_H
