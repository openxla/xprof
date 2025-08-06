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

#include "xprof/convert/roofline_model_processor.h"

#include <string>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/op_stats_to_roofline_model.h"
#include "xprof/convert/repository.h"
#include "plugin/xprof/protobuf/roofline_model.pb.h"
#include "plugin/xprof/protobuf/op_stats.pb.h"

namespace xprof {

using tensorflow::profiler::RooflineModelDatabase;
using tensorflow::profiler::RooflineModelToDataTableJson;
using tensorflow::profiler::OpStats;
using tensorflow::profiler::SessionSnapshot;

absl::Status RooflineModelProcessor::ProcessCombinedOpStats(
    const SessionSnapshot& session_snapshot, const OpStats& combined_op_stats) {

  RooflineModelDatabase result =
      ConvertOpStatsToRooflineModel(combined_op_stats, true);
  RooflineModelDatabase result_without_infeed_outfeed =
      ConvertOpStatsToRooflineModel(combined_op_stats, false);

  result.mutable_roofline_model_record()->MergeFrom(
      result_without_infeed_outfeed.roofline_model_record());

  std::string roofline_model_json = RooflineModelToDataTableJson(result);

  SetOutput(roofline_model_json, "application/json");
  return absl::OkStatus();
}

}  // namespace xprof
