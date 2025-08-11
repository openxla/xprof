#ifndef THIRD_PARTY_XPROF_CONVERT_MEGASCALE_STATS_PROCESSOR_H_
#define THIRD_PARTY_XPROF_CONVERT_MEGASCALE_STATS_PROCESSOR_H_

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/profile_processor.h"
#include "xprof/convert/repository.h"
#include "xprof/convert/tool_options.h"
#include "plugin/xprof/protobuf/op_stats.pb.h"
namespace xprof {

class MegascaleStatsProcessor : public ProfileProcessor {
 public:
  explicit MegascaleStatsProcessor(const tensorflow::profiler::ToolOptions&) {}

  absl::Status ProcessSession(
      const tensorflow::profiler::SessionSnapshot& session_snapshot,
      const tensorflow::profiler::ToolOptions& options) final;

  absl::StatusOr<std::string> Map(
      const tensorflow::profiler::SessionSnapshot& session_snapshot,
      const std::string& hostname,
      const tensorflow::profiler::XSpace& xspace) override {
    return absl::UnimplementedError(
        "Map not implemented for MegascaleStatsProcessor");
  }

  absl::Status Reduce(
      const tensorflow::profiler::SessionSnapshot& session_snapshot,
      const std::vector<std::string>& map_output_files) override {
    return absl::UnimplementedError(
        "Reduce not implemented for MegascaleStatsProcessor");
  }
};

}  // namespace xprof

#endif  // THIRD_PARTY_XPROF_CONVERT_MEGASCALE_STATS_PROCESSOR_H_
