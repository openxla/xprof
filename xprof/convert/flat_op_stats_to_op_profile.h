#ifndef THIRD_PARTY_XPROF_CONVERT_FLAT_OP_STATS_TO_OP_PROFILE_H_
#define THIRD_PARTY_XPROF_CONVERT_FLAT_OP_STATS_TO_OP_PROFILE_H_

#include "xprof/convert/op_profile_builder.h"
#include "plugin/xprof/protobuf/flat_op_metrics.pb.h"
#include "plugin/xprof/protobuf/hardware_types.pb.h"
#include "plugin/xprof/protobuf/op_profile.pb.h"
#include "plugin/xprof/protobuf/op_stats.pb.h"

namespace tensorflow {
namespace profiler {

void ConvertFlatOpStatsToOpProfile(
    const OpStats& op_stats, tensorflow::profiler::HardwareType hardware_type,
    tensorflow::profiler::op_profile::Profile& profile, int op_profile_limit,
    OpProfileGrouping group_by);

}  // namespace profiler
}  // namespace tensorflow

#endif  // THIRD_PARTY_XPROF_CONVERT_FLAT_OP_STATS_TO_OP_PROFILE_H_
