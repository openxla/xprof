/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XPROF_CONVERT_OP_STATS_TO_OP_PROFILE_H_
#define XPROF_CONVERT_OP_STATS_TO_OP_PROFILE_H_

#include "xprof/convert/op_profile_builder.h"
#include "xprof/convert/tool_options.h"
#include "plugin/xprof/protobuf/hardware_types.pb.h"
#include "plugin/xprof/protobuf/op_profile.pb.h"
#include "plugin/xprof/protobuf/op_stats.pb.h"

namespace tensorflow {
namespace profiler {

// Assembles a hierarchical performance profile based on HLOs in the op metrics
// db.
// The node hierarchy is as following:
//    by_category
//      - combined_root
//          - category 1
//          - category 2
//          - ...
//      - idle
//    by_program
//      - program_1_root
//          - category 1
//          - category 2
//          - ...
//      - program_2_root
//          - category 1
//          - ...
//      - idle
// The nodes in the profile are sorted by time in decreasing order and pruned
// to reduce the profile size. Only 100 nodes are kept for level >= 3.
// See op_profile.proto for the detailed semantics of the returned profile.
void ConvertOpStatsToOpProfile(
    const tensorflow::profiler::OpStats& op_stats,
    tensorflow::profiler::HardwareType hardware_type,
    tensorflow::profiler::op_profile::Profile& profile,
    int op_profile_limit = 100,
    OpProfileGrouping group_by = OpProfileGrouping::kByProgram);

// Parses the "group_by" option and returns the corresponding OpProfileGrouping.
inline OpProfileGrouping GetOpProfileGrouping(
    const tensorflow::profiler::ToolOptions& options) {
  if (auto it = options.find("group_by");
      it != options.end() && std::holds_alternative<std::string>(it->second)) {
    const std::string& group_by_str = std::get<std::string>(it->second);
    if (group_by_str == "category") {
      return OpProfileGrouping::kByCategory;
    }
    if (group_by_str == "provenance") {
      return OpProfileGrouping::kByProvenance;
    }
  }
  return OpProfileGrouping::kByProgram;
}

}  // namespace profiler
}  // namespace tensorflow

#endif  // THIRD_PARTY_XPROF_CONVERT_OP_STATS_TO_OP_PROFILE_H_
