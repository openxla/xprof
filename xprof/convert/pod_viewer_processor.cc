/* Copyright 2024 The OpenXLA Authors. All Rights Reserved.
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

#include "xprof/convert/pod_viewer_processor.h"

#include <string>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/log/log.h"
#include "xprof/convert/op_stats_to_pod_viewer.h"
#include "tsl/platform/protobuf.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/repository.h"

namespace xprof {

using tensorflow::profiler::OpStats;
using tensorflow::profiler::SessionSnapshot;

absl::Status PodViewerProcessor::ProcessCombinedOpStats(
    const SessionSnapshot& session_snapshot, const OpStats& combined_op_stats) {
  std::string json_output;
  tsl::protobuf::util::JsonPrintOptions opts;
  opts.always_print_primitive_fields = true;

  auto encode_status = tsl::protobuf::util::MessageToJsonString(
      ConvertOpStatsToPodViewer(combined_op_stats), &json_output, opts);

  SetOutput(json_output, "application/json");
  return absl::OkStatus();
}

}  // namespace xprof
