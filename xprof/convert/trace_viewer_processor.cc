/* Copyright 2025 The OpenXLA Authors. All Rights Reserved.

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

#include "xprof/convert/trace_viewer_processor.h"

#include <string>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "google/protobuf/arena.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/profiler/convert/xplane_to_trace_events.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/preprocess_single_host_xplane.h"
#include "xprof/convert/profile_processor_factory.h"
#include "xprof/convert/repository.h"
#include "xprof/convert/tool_options.h"

namespace xprof {

using ::tensorflow::profiler::SessionSnapshot;
using ::tensorflow::profiler::ToolOptions;
using ::tensorflow::profiler::XSpace;
using ::tsl::profiler::ConvertXSpaceToTraceEventsString;

absl::Status TraceViewerProcessor::ProcessSession(
    const SessionSnapshot& session_snapshot, const ToolOptions& options) {
  absl::string_view session_id = session_snapshot.GetSessionRunDir();
  LOG(INFO) << "TraceViewerProcessor::ProcessSession started session_id: "
            << session_id;
  absl::Time start_time = absl::Now();
  if (session_snapshot.XSpaceSize() != 1) {
    return tsl::errors::InvalidArgument(
        "Trace events tool expects only 1 XSpace path but gets ",
        session_snapshot.XSpaceSize());
  }

  google::protobuf::Arena arena;
  TF_ASSIGN_OR_RETURN(XSpace * xspace, session_snapshot.GetXSpace(0, &arena));
  PreprocessSingleHostXSpace(xspace, /*step_grouping=*/true,
                             /*derived_timeline=*/true);
  LOG(INFO) << "PreprocessSingleHostXSpace done. Duration: "
            << absl::Now() - start_time << " session_id: " << session_id;

  std::string trace_viewer_json;
  absl::Time convert_start_time = absl::Now();
  ConvertXSpaceToTraceEventsString(*xspace, &trace_viewer_json);
  LOG(INFO) << "ConvertXSpaceToTraceEventsString done. Duration: "
            << absl::Now() - convert_start_time
            << " session_id: " << session_id;

  absl::Time set_output_start_time = absl::Now();
  SetOutput(trace_viewer_json, "application/json");
  LOG(INFO) << "SetOutput done. Duration: "
            << absl::Now() - set_output_start_time
            << " session_id: " << session_id;

  LOG(INFO) << "TraceViewerProcessor::ProcessSession done. Total Duration: "
            << absl::Now() - start_time << " session_id: " << session_id;
  return absl::OkStatus();
}

REGISTER_PROFILE_PROCESSOR("trace_viewer", TraceViewerProcessor);

}  // namespace xprof
