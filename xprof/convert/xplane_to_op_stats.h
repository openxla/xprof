/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XPROF_CONVERT_XPLANE_TO_OP_STATS_H_
#define XPROF_CONVERT_XPLANE_TO_OP_STATS_H_

#include <optional>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xla/tsl/profiler/utils/xplane_visitor.h"
#include "xprof/convert/duty_cycle_tracker.h"
#include "plugin/xprof/protobuf/flat_op_metrics.pb.h"
#include "plugin/xprof/protobuf/op_stats.pb.h"
#include "xprof/utils/hlo_proto_map.h"

namespace tensorflow {
namespace profiler {

using ::tsl::profiler::XPlaneVisitor;

struct OpStatsOptions {
  bool maybe_drop_incomplete_steps = false;
  bool generate_op_metrics_db = false;
  bool generate_step_db = false;
  bool generate_kernel_stats_db = false;
};

// How CustomCall ops contribute to TPU duty-cycle / chip utilization.
// Default is kLegacy (post-#2942): only IsOffDutyOp categories are off-duty;
// CustomCall is not special-cased.
enum class CustomCallDutyCycleMode {
  kLegacy = 0,      // IsOffDutyOp only (default; env: "legacy")
  kFlopsModel = 1,  // CustomCall off-duty iff flops==0 && model_flops==0
  kIciAware = 2,    // flops_model + force on-duty when uses_ici != 0
};

// Parses a mode name. Empty / unknown values map to kLegacy.
// Recognized: "legacy", "flops_model", "ici_aware".
CustomCallDutyCycleMode ParseCustomCallDutyCycleMode(absl::string_view mode);

// Reads XPROF_CUSTOM_CALL_DUTY_CYCLE_MODE (unset/empty/unknown -> kLegacy).
CustomCallDutyCycleMode GetCustomCallDutyCycleModeFromEnv();

// Converts XSpace to FlatOpMetricsDb.
absl::StatusOr<OpStats> ConvertXSpaceToFlatOpMetricsDb(
    const XSpace& space, const OpStatsOptions& options);

// NOTE: call GroupTfEvents before if OpStats.step_db needs to be generated.
absl::StatusOr<OpStats> ConvertXSpaceToOpStats(const XSpace& space,
                                               const OpStatsOptions& options);

// Populates the program_id_to_name map in OpStats.
void SetProgramIdToNameMap(const HloProtoMap& hlo_proto_map,
                           tensorflow::profiler::OpStats& op_stats);

// Populates the given RunEnvironment with data from XSpace.
void SetRunEnvironment(const XSpace& space, RunEnvironment* env);

// Propagate and dedup the diagnostics in XSpace and add to OpStats.
void PropagateXSpaceDiagnosticsToOpStats(const XSpace& space,
                                         OpStats* op_stats);

// Populates PerfEnv.
PerfEnv MakePerfEnv(double peak_tera_flops_per_second,
                    std::vector<double> peak_bws);

// Extracts PerfEnv from XPlane stats.
PerfEnv GetPerfEnvFromXPlane(const XPlane& device_plane);

// Constructs a DutyCycleTracker from the given XPlaneVisitor.
// When mode_override is nullopt, uses GetCustomCallDutyCycleModeFromEnv().
// Default env mode is kLegacy (bit-identical to post-#2942 ConstructDutyCycleTracker).
DutyCycleTracker ConstructDutyCycleTracker(
    XPlaneVisitor& visitor,
    std::optional<CustomCallDutyCycleMode> mode_override = std::nullopt);

}  // namespace profiler
}  // namespace tensorflow

#endif  // XPROF_CONVERT_XPLANE_TO_OP_STATS_H_
