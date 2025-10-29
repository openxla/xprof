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

#include "xprof/convert/op_stats_processor.h"

#include <optional>
#include <string>
#include <variant>  // Required for std::holds_alternative and std::get
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "google/protobuf/arena.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/types.h"
#include "tsl/platform/path.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/op_stats_combiner.h"
#include "xprof/convert/preprocess_single_host_xplane.h"
#include "xprof/convert/repository.h"
#include "xprof/convert/tool_options.h"
#include "xprof/convert/xplane_to_op_stats.h"
#include "plugin/xprof/protobuf/op_stats.pb.h"
#include "xprof/utils/hardware_type_utils.h"
#include "xprof/utils/step_intersection.h"

namespace xprof {
namespace {

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

std::string GetCacheFilePath(const SessionSnapshot& session_snapshot,
                             const std::string& hostname) {
  StoredDataType cache_type = StoredDataType::OP_STATS;
  std::string filename =
      session_snapshot.GetHostDataFileName(cache_type, hostname).value_or("");
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

// Checks if the OpStats cache files exist for all hosts.
bool AreAllOpStatsCached(const SessionSnapshot& session_snapshot) {
  for (int i = 0; i < session_snapshot.XSpaceSize(); ++i) {
    std::string hostname = session_snapshot.GetHostname(i);
    std::string cache_file_path = GetCacheFilePath(session_snapshot, hostname);
    if (!tsl::Env::Default()->FileExists(cache_file_path).ok()) {
      LOG(INFO) << "OpStats cache miss for host: " << hostname;
      return false;
    }
    LOG(INFO) << "OpStats cache hit for host: " << hostname
              << " with path: " << cache_file_path;
  }
  LOG(INFO) << "OpStats cache hit for all hosts.";
  return true;
}

}  // namespace

// This overload of Map is provided to conform to the ProfileProcessor
// interface. It creates a temporary SessionSnapshot from the given xspace_path
// to be able to call the other Map overload, which requires metadata from the
// SessionSnapshot for caching and processing.
absl::StatusOr<std::string> OpStatsProcessor::Map(
    const std::string& xspace_path) {
  std::vector<std::string> xspace_paths = {xspace_path};
  TF_ASSIGN_OR_RETURN(
      SessionSnapshot session_snapshot,
      SessionSnapshot::Create(xspace_paths, /*xspaces=*/std::nullopt));
  // get xspace from session snapshot
  std::string hostname = session_snapshot.GetHostname(0);
  google::protobuf::Arena arena;
  TF_ASSIGN_OR_RETURN(XSpace * xspace, session_snapshot.GetXSpace(0, &arena));

  return Map(session_snapshot, hostname, *xspace);
}

absl::StatusOr<std::string> OpStatsProcessor::Map(
    const SessionSnapshot& session_snapshot, const std::string& hostname,
    const XSpace& xspace) {
  std::string cache_file_path = GetCacheFilePath(session_snapshot, hostname);

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
  TF_RETURN_IF_ERROR(WriteBinaryProto(
      session_snapshot, StoredDataType::OP_STATS, hostname, op_stats));
  return cache_file_path;
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

  StepIntersection step_intersection = ComputeStepIntersectionToMergeOpStats(
      all_op_stats_info, std::numeric_limits<uint32_t>::max());
  OpStats combined_op_stats;
  CombineAllOpStats(all_op_stats_info, step_intersection, &combined_op_stats);

  TF_RETURN_IF_ERROR(WriteBinaryProto(
      session_snapshot, StoredDataType::OP_STATS,
      tensorflow::profiler::kAllHostsIdentifier, combined_op_stats));

  return ProcessCombinedOpStats(session_snapshot, combined_op_stats, options_);
}

bool OpStatsProcessor::ShouldUseWorkerService(
    const SessionSnapshot& session_snapshot,
    const tensorflow::profiler::ToolOptions& options) const {
  // TODO: b/442493266 - Support sharding a large single-host trace for
  // distributed processing.
  if (session_snapshot.XSpaceSize() == 1) {
    // If there is only one host, we don't need to use the worker service.
    // This is to avoid unnecessary overhead for single host processing.
    return false;
  }

  // TODO(b/441223611): Performance test between single host with and without
  //                    distributed processing.
  bool use_saved_result = GetUseSavedResult(options);
  LOG(INFO) << "use_saved_result: " << use_saved_result;

  // If not using saved results, always use the worker service for map/reduce.
  if (!use_saved_result) {
    return true;
  }

  // If using saved results, check if all OpStats are already cached.
  // If not, we need to run the Map phase on the worker service.
  return !AreAllOpStatsCached(session_snapshot);
}

absl::Status OpStatsProcessor::ProcessSession(
    const SessionSnapshot& session_snapshot,
    const tensorflow::profiler::ToolOptions& options) {
  return absl::OkStatus();
}

}  // namespace xprof
