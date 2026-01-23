#ifndef THIRD_PARTY_XPROF_CONVERT_MEGASCALE_PERFETTO_XSPACE_LOADER_H_
#define THIRD_PARTY_XPROF_CONVERT_MEGASCALE_PERFETTO_XSPACE_LOADER_H_

#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/megascale_perfetto/xprof_trace.h"

// XSpaceLoader is a utility class for loading XSpace protos and converting
// them to XprofTrace objects. It does some additional processing:
// - Filters out non-TPU and non-Megascale planes.
// - Maps the Megascale events to the corresponding TPU device.
// - Reorders the megascale tracks such that there is one track per collective.

namespace xprof::megascale {

class XSpaceLoader {
 public:
  // Converts an XSpace proto to an XprofTrace.
  static XprofTrace Load(const tensorflow::profiler::XSpace& space);
  // Loads an XSpace proto from a file and converts it to an XprofTrace.
  static absl::StatusOr<XprofTrace> LoadFromFile(const std::string& file_path);
};

}  // namespace xprof::megascale

#endif  // THIRD_PARTY_XPROF_CONVERT_MEGASCALE_PERFETTO_XSPACE_LOADER_H_
