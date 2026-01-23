#ifndef THIRD_PARTY_XPROF_CONVERT_MEGASCALE_PERFETTO_PERFETTO_WRITER_H_
#define THIRD_PARTY_XPROF_CONVERT_MEGASCALE_PERFETTO_PERFETTO_WRITER_H_

#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "xprof/convert/megascale_perfetto/xprof_trace.h"

namespace xprof::megascale {

class PerfettoWriter {
 public:
  // Converts `trace` to a perfetto::protos::Trace proto, serializes it,
  // optionally compresses it, then writes it to `output`.
  static absl::Status WriteToCord(const XprofTrace& trace, absl::Cord* output,
                                  bool compressed_output = true);
  // Same as above but returns the serialized (and optionally compressed) proto
  // as a string.
  static absl::StatusOr<std::string> WriteToString(
      const XprofTrace& trace, bool compressed_output = true);
};

}  // namespace xprof::megascale

#endif  // THIRD_PARTY_XPROF_CONVERT_MEGASCALE_PERFETTO_PERFETTO_WRITER_H_
