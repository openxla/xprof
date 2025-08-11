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
#include "xprof/convert/op_stats_processor.h"

#include <string>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/types.h"
#include "tsl/platform/path.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/op_stats_combiner.h"
#include "xprof/convert/preprocess_single_host_xplane.h"
#include "xprof/convert/repository.h"
#include "xprof/convert/xplane_to_op_stats.h"
#include "plugin/xprof/protobuf/op_stats.pb.h"
#include "xprof/utils/hardware_type_utils.h"
#include "xprof/utils/step_intersection.h"

namespace xprof {

using ::tensorflow::profiler::CombineAllOpStats;
using ::tensorflow::profiler::ComputeStepIntersectionToMergeOpStats;
using ::tensorflow::profiler::ConvertXSpaceToOpStats;
using ::tensorflow::profiler::OpStats;
using ::tensorflow::profiler::OpStatsInfo;
using ::tensorflow::profiler::OpStatsOptions;
using ::tensorflow::profiler::ParseHardwareType;
using ::tensorflow::profiler::PreprocessSingleHostXSpace;
using ::tensorflow::profiler::SessionSnapshot;
using ::tensorflow::profiler::StepIntersection;
using ::tensorflow::profiler::StoredDataType;
using ::tensorflow::profiler::WriteBinaryProto;
using ::tensorflow::profiler::XSpace;
using tsl::kuint32max;

absl::StatusOr<std::string> OpStatsProcessor::Map(
    const SessionSnapshot& session_snapshot, const std::string& hostname,
    const XSpace& xspace) {
  StoredDataType cache_type = StoredDataType::OP_STATS;
  TF_ASSIGN_OR_RETURN(
      std::string filename,
      session_snapshot.GetHostDataFileName(cache_type, hostname));
  std::string cache_file_path =
      tsl::io::JoinPath(session_snapshot.GetSessionRunDir(), filename);

  // TODO: Check if use_saved_result is true before using cache.
  if (tsl::Env::Default()->FileExists(cache_file_path).ok()) {
    VLOG(1) << "Map output cache hit for host: " << hostname;
    return cache_file_path;
  }

  VLOG(1) << "Map output cache miss for host: " << hostname;
  // TODO : Avoid copying XSpace here.
  XSpace temp_xspace = xspace;
  PreprocessSingleHostXSpace(&temp_xspace, /*step_grouping=*/true,
                             /*derived_timeline=*/true);
  OpStatsOptions options;
  options.generate_op_metrics_db = true;
  options.generate_step_db = true;
  options.generate_kernel_stats_db = true;
  OpStats op_stats = ConvertXSpaceToOpStats(temp_xspace, options);
  TF_RETURN_IF_ERROR(
      WriteBinaryProto(session_snapshot, cache_type, hostname, op_stats));
  return cache_file_path;
}

absl::StatusOr<std::string> OpStatsProcessor::Map(
    const std::string& xspace_path) {
  return absl::UnimplementedError("Map not implemented");
}

absl::Status OpStatsProcessor::Reduce(
    const SessionSnapshot& session_snapshot,
    const std::vector<std::string>& map_output_files) {
  if (map_output_files.empty()) {
    return absl::InvalidArgumentError("map_output_files cannot be empty");
  }

  std::vector<OpStats> all_op_stats;
  all_op_stats.reserve(map_output_files.size());
  for (const auto& map_output_file : map_output_files) {
    OpStats op_stats;
    TF_RETURN_IF_ERROR(
        tsl::ReadBinaryProto(tsl::Env::Default(), map_output_file, &op_stats));
    all_op_stats.push_back(op_stats);
  }

  std::vector<OpStatsInfo> all_op_stats_info;
  all_op_stats_info.reserve(all_op_stats.size());
  // Create a modifiable copy of OpStats for all_op_stats_info
  std::vector<OpStats> op_stats_copy = all_op_stats;
  for (int i = 0; i < op_stats_copy.size(); i++) {
    all_op_stats_info.emplace_back(
        &op_stats_copy[i],
        ParseHardwareType(op_stats_copy[i].run_environment().device_type()), i);
  }

  StepIntersection step_intersection =
      ComputeStepIntersectionToMergeOpStats(all_op_stats_info, kuint32max);
  OpStats combined_op_stats;
  CombineAllOpStats(all_op_stats_info, step_intersection, &combined_op_stats);

  TF_RETURN_IF_ERROR(WriteBinaryProto(session_snapshot,
                                      StoredDataType::OP_STATS, "all_hosts",
                                      combined_op_stats));

  return ProcessCombinedOpStats(session_snapshot, combined_op_stats);
}

}  // namespace xprof
