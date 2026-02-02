#ifndef THIRD_PARTY_XPROF_CONVERT_XPLANE_TO_UTILIZATION_VIEWER_H_
#define THIRD_PARTY_XPROF_CONVERT_XPLANE_TO_UTILIZATION_VIEWER_H_

#include <string>

#include "absl/status/statusor.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xprof {

// Converts an XSpace with performance counters to a JSON string formatted for
// the Utilization Viewer.
absl::StatusOr<std::string> ConvertXSpaceToUtilizationViewer(
    const tensorflow::profiler::XSpace& space);

}  // namespace xprof

#endif  // THIRD_PARTY_XPROF_CONVERT_XPLANE_TO_UTILIZATION_VIEWER_H_
