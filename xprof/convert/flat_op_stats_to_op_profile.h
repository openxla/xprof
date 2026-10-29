#ifndef THIRD_PARTY_XPROF_CONVERT_FLAT_OP_STATS_TO_OP_PROFILE_H_
#define THIRD_PARTY_XPROF_CONVERT_FLAT_OP_STATS_TO_OP_PROFILE_H_

#include "absl/base/nullability.h"
#include "xprof/convert/op_profile_builder.h"
#include "plugin/xprof/protobuf/flat_op_metrics.pb.h"
#include "plugin/xprof/protobuf/hardware_types.pb.h"
#include "plugin/xprof/protobuf/op_profile.pb.h"
#include "plugin/xprof/protobuf/op_stats.pb.h"

namespace tensorflow {
namespace profiler {

// Same as ConvertOpStatsToOpProfile, but builds the profile from
// op_stats.flat_device_op_metrics_db(). If program_serving_metadata is
// non-null, its entries are copied onto the matching program nodes.
void ConvertFlatOpStatsToOpProfile(
    const OpStats& op_stats, tensorflow::profiler::HardwareType hardware_type,
    tensorflow::profiler::op_profile::Profile& profile,
    int op_profile_limit = 100,
    OpProfileGrouping group_by = OpProfileGrouping::kByProgram,
    const ProgramServingMetadataMap* absl_nullable program_serving_metadata =
        nullptr);

}  // namespace profiler
}  // namespace tensorflow

#endif  // THIRD_PARTY_XPROF_CONVERT_FLAT_OP_STATS_TO_OP_PROFILE_H_
