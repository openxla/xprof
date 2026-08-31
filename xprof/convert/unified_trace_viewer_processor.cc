/* Copyright 2026 The OpenXLA Authors. All Rights Reserved.

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

#include "xprof/convert/unified_trace_viewer_processor.h"

#include <string>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "google/protobuf/arena.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/preprocess_single_host_xplane.h"
#include "xprof/convert/tool_options.h"
#include "xprof/convert/trace_viewer/delta_series/trace_data_to_compressed_delta_series_proto.h"
#include "xprof/convert/trace_viewer/trace_events_to_json.h"
#include "xprof/convert/trace_viewer/trace_options.h"
#include "xprof/convert/unified_session_snapshot.h"
#include "xprof/convert/xplane_to_trace_container.h"

namespace xprof {

absl::Status UnifiedTraceViewerProcessor::ProcessSession(
    const XprofSessionSnapshot& session_snapshot,
    const tensorflow::profiler::ToolOptions& options) {
  absl::string_view session_id = session_snapshot.GetSessionRunDir();
  LOG(INFO)
      << "UnifiedTraceViewerProcessor::ProcessSession started session_id: "
      << session_id;
  absl::Time start_time = absl::Now();
  if (session_snapshot.XSpaceSize() != 1) {
    return absl::InvalidArgumentError(
        absl::StrCat("Trace events tool expects only 1 XSpace path but gets ",
                     session_snapshot.XSpaceSize()));
  }

  google::protobuf::Arena arena;
  TF_ASSIGN_OR_RETURN(tensorflow::profiler::XSpace * xspace,
                      session_snapshot.GetXSpace(0, &arena));
  PreprocessSingleHostXSpace(xspace, /*step_grouping=*/true,
                             /*derived_timeline=*/true);
  LOG(INFO) << "PreprocessSingleHostXSpace done. Duration: "
            << absl::Now() - start_time << " session_id: " << session_id;

  tensorflow::profiler::TraceEventsContainer trace_container;
  std::string hostname = session_snapshot.GetHostname(0);
  tensorflow::profiler::ConvertXSpaceToTraceEventsContainer(hostname, *xspace,
                                                            &trace_container);

  std::string format = tensorflow::profiler::GetParamWithDefault<std::string>(
      options, "format", "json");
  if (format == "pb") {
    tensorflow::profiler::DeltaSeriesProtoConversionOptions proto_options;
    absl::StatusOr<std::string> compressed_result =
        tensorflow::profiler::ConvertTraceDataToCompressedDeltaSeriesProto(
            proto_options, trace_container);
    if (compressed_result.ok()) {
      SetOutput(*compressed_result, "application/octet-stream");
    } else {
      LOG(ERROR) << "Failed to convert trace data: "
                 << compressed_result.status();
      return compressed_result.status();
    }
  } else {
    std::string trace_viewer_json;
    absl::Time convert_start_time = absl::Now();
    tensorflow::profiler::JsonTraceOptions json_trace_options;
    tensorflow::profiler::TraceDeviceType device_type =
        tensorflow::profiler::TraceDeviceType::kUnknownDevice;

    if (tensorflow::profiler::IsTpuTrace(trace_container.trace())) {
      device_type = tensorflow::profiler::TraceDeviceType::kTpu;
    }
    tensorflow::profiler::TraceOptions profiler_trace_options =
        tensorflow::profiler::TraceOptionsFromToolOptions(options);
    json_trace_options.details = tensorflow::profiler::TraceOptionsToDetails(
        device_type, profiler_trace_options);
    tensorflow::profiler::IOBufferAdapter adapter(&trace_viewer_json);
    tensorflow::profiler::TraceEventsToJson<
        tensorflow::profiler::IOBufferAdapter,
        tensorflow::profiler::TraceEventsContainer,
        tensorflow::profiler::RawData>(json_trace_options, trace_container,
                                       &adapter);
    LOG(INFO) << "ConvertXSpaceToTraceEventsString done. Duration: "
              << absl::Now() - convert_start_time
              << " session_id: " << session_id;

    SetOutput(trace_viewer_json, "application/json");
  }

  LOG(INFO)
      << "UnifiedTraceViewerProcessor::ProcessSession done. Total Duration: "
      << absl::Now() - start_time << " session_id: " << session_id;
  return absl::OkStatus();
}

}  // namespace xprof
