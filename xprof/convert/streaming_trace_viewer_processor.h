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
#include "plugin/xprof/protobuf/op_stats.pb.h"
#include "absl/strings/numbers.h"
#include "xprof/convert/xplane_to_trace_container.h"
#include "xprof/convert/trace_viewer/trace_options.h"

namespace xprof {

namespace internal {
// Options for trace viewer used for testing.
struct TraceViewOption {
  uint64_t resolution = 0;
  double start_time_ms = 0.0;
  double end_time_ms = 0.0;
  std::string event_name = "";
  std::string search_prefix = "";
  double duration_ms = 0.0;
  uint64_t unique_id = 0;
  std::string format = "json";
};

inline absl::StatusOr<TraceViewOption> GetTraceViewOption(
    const tensorflow::profiler::ToolOptions& options) {
  TraceViewOption trace_options;
  auto start_time_ms_opt =
      tensorflow::profiler::GetParamWithDefault<std::string>(
          options, "start_time_ms", "0.0");
  auto end_time_ms_opt =
      tensorflow::profiler::GetParamWithDefault<std::string>(
          options, "end_time_ms", "0.0");
  auto resolution_opt =
      tensorflow::profiler::GetParamWithDefault<std::string>(
          options, "resolution", "0");
  trace_options.event_name =
      tensorflow::profiler::GetParamWithDefault<std::string>(
          options, "event_name", "");
  trace_options.search_prefix =
      tensorflow::profiler::GetParamWithDefault<std::string>(
          options, "search_prefix", "");
  auto duration_ms_opt =
      tensorflow::profiler::GetParamWithDefault<std::string>(
          options, "duration_ms", "0.0");
  auto unique_id_opt =
      tensorflow::profiler::GetParamWithDefault<std::string>(
          options, "unique_id", "0");
  trace_options.format =
      tensorflow::profiler::GetParamWithDefault<std::string>(
          options, "format", "json");


  if (!absl::SimpleAtod(start_time_ms_opt, &trace_options.start_time_ms) ||
      !absl::SimpleAtod(end_time_ms_opt, &trace_options.end_time_ms) ||
      !absl::SimpleAtod(duration_ms_opt, &trace_options.duration_ms)) {
    return tsl::errors::InvalidArgument("wrong arguments");
  }

  if (!absl::SimpleAtoi(resolution_opt, &trace_options.resolution)) {
    double resolution_double;
    if (absl::SimpleAtod(resolution_opt, &resolution_double)) {
      trace_options.resolution = static_cast<uint64_t>(resolution_double);
    } else {
      return tsl::errors::InvalidArgument("resolution must be a number");
    }
  }

  if (!absl::SimpleAtoi(unique_id_opt, &trace_options.unique_id)) {
    double unique_id_double;
    if (absl::SimpleAtod(unique_id_opt, &unique_id_double)) {
      trace_options.unique_id = static_cast<uint64_t>(unique_id_double);
    } else {
      return tsl::errors::InvalidArgument("unique_id must be a number");
    }
  }
  return trace_options;
}
}  // namespace internal

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
      const internal::TraceViewOption& trace_option,
      const tensorflow::profiler::TraceOptions& profiler_trace_options,
      absl::string_view session_id);

  tensorflow::profiler::ToolOptions options_;
};

}  // namespace xprof

#endif  // THIRD_PARTY_XPROF_CONVERT_STREAMING_TRACE_VIEWER_PROCESSOR_H_
