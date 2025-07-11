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

#ifndef XPROF_CONVERT_MULTI_XPLANES_TO_OP_STATS_H_
#define XPROF_CONVERT_MULTI_XPLANES_TO_OP_STATS_H_

#include "absl/status/status.h"
#include "xprof/convert/repository.h"
#include "xprof/convert/xplane_to_op_stats.h"
#include "plugin/xprof/protobuf/op_stats.pb.h"

namespace tensorflow {
namespace profiler {

// Converts and combines multiple XSpace protos into a single OpStats
// <combined_op_stats>.
// Return the first error status during conversion, or return OkStatus() if
// there is no error.
absl::Status ConvertMultiXSpacesToCombinedOpStats(
    const SessionSnapshot& session_snapshot, const OpStatsOptions& options,
    OpStats* combined_op_stats);

// Converts multiple XSpaces to a combined OpStats, using cache if available.
absl::Status ConvertMultiXSpaceToCombinedOpStatsWithCache(
    const SessionSnapshot& session_snapshot, OpStats* combined_op_stats);

}  // namespace profiler
}  // namespace tensorflow

#endif  // XPROF_CONVERT_MULTI_XPLANES_TO_OP_STATS_H_
