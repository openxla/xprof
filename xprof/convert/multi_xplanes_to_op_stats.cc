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

#include "xprof/convert/multi_xplanes_to_op_stats.h"

#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "google/protobuf/arena.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/types.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/op_stats_combiner.h"
#include "xprof/convert/preprocess_single_host_xplane.h"
#include "xprof/convert/repository.h"
#include "xprof/convert/xplane_to_op_stats.h"
#include "plugin/xprof/protobuf/op_stats.pb.h"
#include "xprof/utils/hardware_type_utils.h"
#include "xprof/utils/step_intersection.h"

namespace tensorflow {
namespace profiler {
using tsl::kuint32max;

absl::Status ConvertMultiXSpacesToCombinedOpStats(
    const SessionSnapshot& session_snapshot, const OpStatsOptions& options,
    OpStats* combined_op_stats) {
  // Read multiple XSpaces and convert to multiple OpStats.
  // TODO(profiler): Change the combiner to convert and combine one OpStats at a
  // time, to reduce peak memory usage.
  std::vector<OpStats> all_op_stats;
  all_op_stats.reserve(session_snapshot.XSpaceSize());
  for (int i = 0; i < session_snapshot.XSpaceSize(); i++) {
    google::protobuf::Arena arena;
    TF_ASSIGN_OR_RETURN(XSpace* xspace, session_snapshot.GetXSpace(i, &arena));
    PreprocessSingleHostXSpace(xspace, /*step_grouping=*/true,
                               /*derived_timeline=*/true);
    all_op_stats.push_back(ConvertXSpaceToOpStats(*xspace, options));
  }

  // Combine OpStats.
  std::vector<OpStatsInfo> all_op_stats_info;
  all_op_stats_info.reserve(all_op_stats.size());
  for (int i = 0; i < all_op_stats.size(); i++) {
    all_op_stats_info.emplace_back(
        &all_op_stats[i],
        ParseHardwareType(all_op_stats[i].run_environment().device_type()), i);
  }

  // Do not limit the maximum number of steps during the merge of OpStats.
  StepIntersection step_intersection = ComputeStepIntersectionToMergeOpStats(
      all_op_stats_info, std::numeric_limits<uint32_t>::max());
  CombineAllOpStats(all_op_stats_info, step_intersection, combined_op_stats);

  return absl::OkStatus();
}

absl::Status ConvertMultiXSpaceToCombinedOpStatsWithCache(
    const SessionSnapshot& session_snapshot, OpStats* combined_op_stats) {
  OpStatsOptions options;
  options.generate_op_metrics_db = true;
  options.generate_step_db = true;
  options.generate_kernel_stats_db = true;
  TF_ASSIGN_OR_RETURN(auto has_cache,
                      session_snapshot.HasCacheFile(StoredDataType::OP_STATS));
  if (has_cache.first) {
    TF_RETURN_IF_ERROR(ReadBinaryProto(session_snapshot,
                                       StoredDataType::OP_STATS,
                                       kAllHostsIdentifier, combined_op_stats));

  } else {
    TF_RETURN_IF_ERROR(ConvertMultiXSpacesToCombinedOpStats(
        session_snapshot, options, combined_op_stats));
    if (!WriteBinaryProto(session_snapshot, StoredDataType::OP_STATS,
                          kAllHostsIdentifier, *combined_op_stats)
             .ok()) {
      LOG(WARNING) << "Failed to write op stats cache file.";
    };
  }
  return absl::OkStatus();
}

}  // namespace profiler
}  // namespace tensorflow
