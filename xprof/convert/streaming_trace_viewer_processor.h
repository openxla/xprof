#ifndef THIRD_PARTY_XPROF_CONVERT_STREAMING_TRACE_VIEWER_PROCESSOR_H_
#define THIRD_PARTY_XPROF_CONVERT_STREAMING_TRACE_VIEWER_PROCESSOR_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/errors.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/profile_processor.h"
#include "xprof/convert/repository.h"
#include "xprof/convert/tool_options.h"
#include "xprof/convert/trace_view_options.h"
#include "xprof/convert/trace_viewer/trace_options.h"
#include "xprof/convert/xplane_to_trace_container.h"
#include "plugin/xprof/protobuf/op_stats.pb.h"

namespace xprof {

// A profile processor that generates trace viewer events from XSpace data
// and supports streaming them (e.g., via LevelDB tables).
class StreamingTraceViewerProcessor : public ProfileProcessor {
 public:
  explicit StreamingTraceViewerProcessor(
      const tensorflow::profiler::ToolOptions& options)
      : options_(options) {}

  absl::Status ProcessSession(
      const tensorflow::profiler::SessionSnapshot& session_snapshot,
      const tensorflow::profiler::ToolOptions& options) final;

  absl::StatusOr<std::string> Map(const std::string& xspace_path) override;

  absl::StatusOr<std::string> Map(
      const tensorflow::profiler::SessionSnapshot& session_snapshot,
      const std::string& hostname,
      const tensorflow::profiler::XSpace& xspace) override;

  absl::Status Reduce(
      const tensorflow::profiler::SessionSnapshot& session_snapshot,
      const std::vector<std::string>& map_output_files) override;

  bool ShouldUseWorkerService(
      const tensorflow::profiler::SessionSnapshot& session_snapshot,
      const tensorflow::profiler::ToolOptions& options) const override {
    return session_snapshot.XSpaceSize() > 1;
  }

 private:
  absl::Status SerializeAndSetOutput(
      const tensorflow::profiler::TraceEventsContainer& merged_trace_container,
      const tensorflow::profiler::TraceViewOption& trace_option,
      const tensorflow::profiler::TraceOptions& profiler_trace_options,
      absl::string_view session_id);

  tensorflow::profiler::ToolOptions options_;
};

}  // namespace xprof

#endif  // THIRD_PARTY_XPROF_CONVERT_STREAMING_TRACE_VIEWER_PROCESSOR_H_
