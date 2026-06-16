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

#include "xprof/convert/base_op_stats_processor.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/platform/path.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/file_utils.h"
#include "xprof/convert/multi_xplanes_to_op_stats.h"
#include "xprof/convert/op_stats_combiner.h"
#include "xprof/convert/preprocess_single_host_xplane.h"
#include "xprof/convert/repository.h"
#include "xprof/convert/tool_options.h"
#include "xprof/convert/unified_session_snapshot.h"
#include "xprof/convert/xplane_to_op_stats.h"
#include "plugin/xprof/protobuf/op_stats.pb.h"
#include "xprof/utils/hardware_type_utils.h"
#include "xprof/utils/step_intersection.h"

namespace xprof {
namespace {

using ::tensorflow::profiler::CombineAllOpStats;
using ::tensorflow::profiler::ComputeStepIntersectionToMergeOpStats;
using ::tensorflow::profiler::ConvertMultiXSpaceToCombinedOpStatsWithCache;
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

absl::StatusOr<std::string> GetCacheFilePath(
    const XprofSessionSnapshot& session_snapshot, absl::string_view hostname) {
  StoredDataType cache_type = StoredDataType::OP_STATS;
  TF_ASSIGN_OR_RETURN(
      std::string filename,
      session_snapshot.GetHostDataFileName(cache_type, hostname));
  return tsl::io::JoinPath(session_snapshot.GetSessionRunDir(), filename);
}

bool GetUseSavedResult(const tensorflow::profiler::ToolOptions& options) {
  if (auto it = options.find("use_saved_result"); it != options.end()) {
    if (std::holds_alternative<bool>(it->second)) {
      return std::get<bool>(it->second);
    }
  }
  return false;
}

bool AreAllOpStatsCached(const XprofSessionSnapshot& session_snapshot) {
  for (size_t i = 0; i < session_snapshot.XSpaceSize(); ++i) {
    std::string hostname = session_snapshot.GetHostname(i);
    auto cache_file_path = GetCacheFilePath(session_snapshot, hostname);
    if (!cache_file_path.ok() ||
        !tsl::Env::Default()->FileExists(*cache_file_path).ok()) {
      return false;
    }
  }
  return true;
}

}  // namespace



absl::StatusOr<std::string> BaseOpStatsProcessor::Map(
    const XprofSessionSnapshot& session_snapshot, absl::string_view hostname,
    const XSpace& xspace) {
  TF_ASSIGN_OR_RETURN(std::string cache_file_path,
                      GetCacheFilePath(session_snapshot, hostname));

  if (tsl::Env::Default()->FileExists(cache_file_path).ok()) {
    return cache_file_path;
  }

  XSpace temp_xspace = xspace;
  PreprocessSingleHostXSpace(&temp_xspace, /*step_grouping=*/true,
                             /*derived_timeline=*/true);
  OpStatsOptions options = {
      .generate_op_metrics_db = true,
      .generate_step_db = true,
      .generate_kernel_stats_db = true,
  };

  TF_ASSIGN_OR_RETURN(OpStats op_stats,
                      ConvertXSpaceToOpStats(temp_xspace, options));
  TF_RETURN_IF_ERROR(WriteBinaryProto(
      session_snapshot, StoredDataType::OP_STATS, hostname, op_stats));
  return cache_file_path;
}

absl::Status BaseOpStatsProcessor::Reduce(
    const XprofSessionSnapshot& session_snapshot,
    const std::vector<std::string>& map_output_files) {
  if (map_output_files.empty()) {
    return absl::InvalidArgumentError("map_output_files cannot be empty");
  }

  std::vector<OpStats> all_op_stats;
  all_op_stats.reserve(map_output_files.size());

  for (const auto& map_output_file : map_output_files) {
    OpStats op_stats;
    TF_RETURN_IF_ERROR(xprof::ReadBinaryProto(map_output_file, &op_stats));
    all_op_stats.push_back(std::move(op_stats));
  }

  std::vector<OpStatsInfo> all_op_stats_info;
  all_op_stats_info.reserve(all_op_stats.size());
  // Note: all_op_stats is fully populated and its size is stable. It will not
  // reallocate, making it safe to hold pointers to its elements within
  // all_op_stats_info.
  for (int i = 0; i < all_op_stats.size(); i++) {
    all_op_stats_info.emplace_back(
        &all_op_stats[i],
        ParseHardwareType(all_op_stats[i].run_environment().device_type()), i);
  }

  StepIntersection step_intersection = ComputeStepIntersectionToMergeOpStats(
      all_op_stats_info, std::numeric_limits<uint32_t>::max());

  OpStats combined_op_stats;
  CombineAllOpStats(all_op_stats_info, step_intersection, &combined_op_stats);

  TF_RETURN_IF_ERROR(WriteBinaryProto(
      session_snapshot, StoredDataType::OP_STATS,
      tensorflow::profiler::kAllHostsIdentifier, combined_op_stats));

  return ProcessCombinedOpStats(session_snapshot, combined_op_stats, options_);
}

absl::Status BaseOpStatsProcessor::ProcessSession(
    const XprofSessionSnapshot& session_snapshot,
    const tensorflow::profiler::ToolOptions& options) {
  OpStats combined_op_stats;
  TF_RETURN_IF_ERROR(ConvertMultiXSpaceToCombinedOpStatsWithCache(
      session_snapshot, &combined_op_stats));

  return ProcessCombinedOpStats(session_snapshot, combined_op_stats, options);
}

bool BaseOpStatsProcessor::ShouldUseWorkerService(
    const XprofSessionSnapshot& session_snapshot,
    const tensorflow::profiler::ToolOptions& options) const {
  if (session_snapshot.XSpaceSize() == 1) {
    return false;
  }

  bool use_saved_result = GetUseSavedResult(options);
  if (!use_saved_result) {
    return true;
  }

  return !AreAllOpStatsCached(session_snapshot);
}

}  // namespace xprof
