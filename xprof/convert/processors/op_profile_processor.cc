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

#include "xprof/convert/processors/op_profile_processor.h"

#include <string>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/errors.h"
#include "tsl/platform/protobuf.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/multi_xplanes_to_op_stats.h"
#include "xprof/convert/op_stats_to_op_profile.h"
#include "xprof/convert/repository.h"
#include "xprof/convert/tool_options.h"
#include "plugin/xprof/protobuf/op_profile.pb.h"
#include "plugin/xprof/protobuf/op_stats.pb.h"
#include "xprof/utils/hardware_type_utils.h"

namespace xprof {

using tensorflow::profiler::ConvertMultiXSpaceToCombinedOpStatsWithCache;
using tensorflow::profiler::OpStats;
using tensorflow::profiler::ParseHardwareType;
using tensorflow::profiler::SessionSnapshot;
using tensorflow::profiler::op_profile::Profile;
using tsl::protobuf::util::JsonPrintOptions;

absl::Status OpProfileProcessor::ProcessSession(
    const SessionSnapshot& session_snapshot,
    const tensorflow::profiler::ToolOptions& options) {
  OpStats combined_op_stats;
  TF_RETURN_IF_ERROR(ConvertMultiXSpaceToCombinedOpStatsWithCache(
      session_snapshot, &combined_op_stats));

  tensorflow::profiler::op_profile::Profile profile;
  ConvertOpStatsToOpProfile(
      combined_op_stats,
      ParseHardwareType(combined_op_stats.run_environment().device_type()),
      profile);
  std::string json_output;
  tsl::protobuf::util::JsonPrintOptions opts;
  opts.always_print_fields_with_no_presence = true;

  auto encode_status =
      tsl::protobuf::util::MessageToJsonString(profile, &json_output, opts);
  if (!encode_status.ok()) {
    const auto& error_message = encode_status.message();
    return tsl::errors::Internal(
        "Could not convert op profile proto to json. Error: ", error_message);
  }
  SetOutput(json_output, "application/json");
  return absl::OkStatus();
}

absl::Status OpProfileProcessor::ProcessCombinedOpStats(
    const SessionSnapshot& session_snapshot, const OpStats& combined_op_stats) {
  Profile profile;
  ConvertOpStatsToOpProfile(
      combined_op_stats,
      ParseHardwareType(combined_op_stats.run_environment().device_type()),
      profile);
  std::string op_profile_json;
  JsonPrintOptions opts;
  opts.always_print_fields_with_no_presence = true;

  auto encode_status =
      tsl::protobuf::util::MessageToJsonString(profile, &op_profile_json, opts);
  if (!encode_status.ok()) {
    const auto& error_message = encode_status.message();
    return tsl::errors::Internal(
        "Could not convert op profile proto to json. Error: ", error_message);
  }

  SetOutput(op_profile_json, "application/json");
  return absl::OkStatus();
}

}  // namespace xprof
