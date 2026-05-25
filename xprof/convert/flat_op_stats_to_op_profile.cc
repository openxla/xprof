#include "xprof/convert/flat_op_stats_to_op_profile.h"

#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/strings/match.h"
#include "xla/tsl/profiler/utils/math_utils.h"
#include "xprof/convert/op_profile_builder.h"
#include "plugin/xprof/protobuf/flat_op_metrics.pb.h"
#include "plugin/xprof/protobuf/hardware_types.pb.h"
#include "plugin/xprof/protobuf/op_metrics.pb.h"
#include "plugin/xprof/protobuf/op_profile.pb.h"
#include "plugin/xprof/protobuf/op_stats.pb.h"
#include "xprof/utils/flat_op_metrics_db_utils.h"
#include "xprof/utils/op_metrics_db_utils.h"

namespace tensorflow {
namespace profiler {
namespace {

using ::tensorflow::profiler::IsIdleOp;
using ::tensorflow::profiler::OpProfileBuilder;
using ::tensorflow::profiler::OpProfileGrouping;
using ::tensorflow::profiler::OpProfileOptions;
using ::tensorflow::profiler::op_profile::Node;

// Modified to take OpStats for program_id_to_name_map and perf_env
void BuildOpProfileNodeTreeAlter(const OpStats& op_stats,
                                 OpProfileGrouping group_by,
                                 bool exclude_idle_ops, int op_profile_limit,
                                 Node* root) {
  if (op_stats.flat_device_op_metrics_db().op_instances().empty()) return;

  OpProfileOptions options = {group_by,
                              /*group_by_deduplicated_name=*/true,
                              /*children_per_node=*/op_profile_limit};
  // Use op_stats.program_id_to_name_map()
  OpProfileBuilder builder(options, root, &op_stats.program_id_to_name_map());
  for (const FlatOpMetrics& op_metrics :
       op_stats.flat_device_op_metrics_db().op_instances()) {
    DCHECK(!op_metrics.hlo_name().empty());
    // Don't add ops that cannot be symbolized.
    if (absl::StartsWith(op_metrics.hlo_name(), "region")) continue;
    if (exclude_idle_ops && IsIdleOp(op_metrics)) continue;
    if ((op_metrics.core_type() == FlatOpMetrics::SPARSE_CORE) ||
        op_metrics.is_fusion_child()) {
      builder.AddFusionAndSparseCoreOp(op_metrics);
    } else {
      builder.AddOp(op_metrics);
    }
  }

  // Use op_stats.perf_env()
  const auto& perf_env = op_stats.perf_env();
  double max_gigaflops_per_second_per_core =
      tsl::profiler::TeraToGiga(perf_env.peak_tera_flops_per_second());
  std::vector<double> peak_bws;
  for (auto bw : perf_env.peak_bws_giga_bytes_per_second()) {
    peak_bws.push_back(tsl::profiler::GigaToGibi(bw));
  }
  builder.Finalize(max_gigaflops_per_second_per_core, peak_bws,
                   exclude_idle_ops
                       ? op_stats.flat_device_op_metrics_db().total_op_time_ps()
                       : op_stats.flat_device_op_metrics_db().total_time_ps());
}

}  // namespace

void ConvertFlatOpStatsToOpProfile(
    const OpStats& op_stats, tensorflow::profiler::HardwareType hardware_type,
    tensorflow::profiler::op_profile::Profile& profile, int op_profile_limit,
    OpProfileGrouping group_by) {
  profile.set_device_type(HardwareType_Name(hardware_type));

  profile.set_agg_dvfs_time_scale_multiplier(tsl::profiler::SafeDivide(
      op_stats.flat_device_op_metrics_db().normalized_total_op_time_ps(),
      op_stats.flat_device_op_metrics_db().total_op_time_ps()));

  switch (group_by) {
    case OpProfileGrouping::kByCategory:
      BuildOpProfileNodeTreeAlter(op_stats, OpProfileGrouping::kByCategory,
                                  /*exclude_idle_ops=*/false, op_profile_limit,
                                  profile.mutable_by_category());
      BuildOpProfileNodeTreeAlter(op_stats, OpProfileGrouping::kByCategory,
                                  /*exclude_idle_ops=*/true, op_profile_limit,
                                  profile.mutable_by_category_exclude_idle());
      break;
    case OpProfileGrouping::kByProgram:
      BuildOpProfileNodeTreeAlter(op_stats, OpProfileGrouping::kByProgram,
                                  /*exclude_idle_ops=*/false, op_profile_limit,
                                  profile.mutable_by_program());
      BuildOpProfileNodeTreeAlter(op_stats, OpProfileGrouping::kByProgram,
                                  /*exclude_idle_ops=*/true, op_profile_limit,
                                  profile.mutable_by_program_exclude_idle());
      break;
    case OpProfileGrouping::kByProvenance:
      BuildOpProfileNodeTreeAlter(op_stats, OpProfileGrouping::kByProvenance,
                                  /*exclude_idle_ops=*/false, op_profile_limit,
                                  profile.mutable_by_provenance());
      BuildOpProfileNodeTreeAlter(op_stats, OpProfileGrouping::kByProvenance,
                                  /*exclude_idle_ops=*/true, op_profile_limit,
                                  profile.mutable_by_provenance_exclude_idle());
      break;
  }
}

}  // namespace profiler
}  // namespace tensorflow
