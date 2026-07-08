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

#include "xprof/convert/unified_op_profile_processor.h"

#include <string>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "tsl/platform/protobuf.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/op_stats_to_op_profile.h"
#include "xprof/convert/tool_options.h"
#include "xprof/convert/unified_session_snapshot.h"
#include "plugin/xprof/protobuf/op_profile.pb.h"
#include "plugin/xprof/protobuf/op_stats.pb.h"
#include "xprof/utils/hardware_type_utils.h"

namespace xprof {

using tensorflow::profiler::OpStats;
using tensorflow::profiler::ParseHardwareType;
using tensorflow::profiler::op_profile::Profile;
using tsl::protobuf::util::JsonPrintOptions;

absl::Status UnifiedOpProfileProcessor::ProcessCombinedOpStats(
    const XprofSessionSnapshot& session_snapshot,
    const OpStats& combined_op_stats,
    const tensorflow::profiler::ToolOptions& options) {
  Profile profile;
  tensorflow::profiler::ConvertOpStatsToOpProfile(
      combined_op_stats,
      ParseHardwareType(combined_op_stats.run_environment().device_type()),
      profile, /*op_profile_limit=*/100,
      tensorflow::profiler::GetOpProfileGrouping(options));

  std::string op_profile_json;
  JsonPrintOptions opts;
  opts.always_print_fields_with_no_presence = true;

  if (auto encode_status = tsl::protobuf::util::MessageToJsonString(
          profile, &op_profile_json, opts);
      !encode_status.ok()) {
    return absl::InternalError(absl::StrCat(
        "Could not convert op profile proto to json. Error: ",
        encode_status.message()));
  }

  SetOutput(op_profile_json, "application/json");
  return absl::OkStatus();
}

}  // namespace xprof
