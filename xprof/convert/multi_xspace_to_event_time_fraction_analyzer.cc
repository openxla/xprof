/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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


#include <string>

#include "xprof/convert/xspace_to_event_time_fraction_analyzer.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/statusor.h"
#include "xprof/convert/repository.h"
#include "xprof/convert/tool_options.h"
#include "xprof/convert/xplane_to_tools_data_with_profile_processor.h"
#include "plugin/xprof/protobuf/event_time_fraction_analyzer.pb.h"

namespace tensorflow {
namespace profiler {

absl::StatusOr<EventTimeFractionAnalyzerResult>
ConvertMultiXSpacesToEventTimeFractionAnalyzerResult(
    const SessionSnapshot& session_snapshot,
    absl::string_view target_event_name) {
  ToolOptions options;
  options["event_name"] = std::string(target_event_name);

  TF_ASSIGN_OR_RETURN(std::string serialized_result,
                      ConvertMultiXSpacesToToolDataWithProfileProcessor(
                          session_snapshot, "event_time_fraction_analyzer",
                          options));

  EventTimeFractionAnalyzerResult result;
  if (!result.ParseFromString(serialized_result)) {
    return absl::InternalError(
        "Failed to parse EventTimeFractionAnalyzerResult.");
  }
  return result;
}

}  // namespace profiler
}  // namespace tensorflow
