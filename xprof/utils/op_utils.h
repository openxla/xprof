/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XPROF_UTILS_OP_UTILS_H_
#define XPROF_UTILS_OP_UTILS_H_

#include <cstdint>

#include "absl/strings/string_view.h"
#include "xla/tsl/platform/types.h"
#include "xla/tsl/profiler/convert/xla_op_utils.h"
#include "xla/tsl/profiler/utils/timespan.h"
#include "tsl/platform/protobuf.h"
#include "plugin/tensorboard_plugin_profile/protobuf/op_metrics.pb.h"
#include "xprof/utils/hlo_module_map.h"
#include "xprof/utils/op_metrics_db_utils.h"

namespace tensorflow {
namespace profiler {
using ::tensorflow::profiler::OpMetrics;
using ::tensorflow::profiler::OpMetrics_MemoryAccessed;
using tsl::uint64;

// Converts the memory access breakdown into OpMetrics's format.
tsl::protobuf::RepeatedPtrField<OpMetrics::MemoryAccessed>
ConvertPerformanceInfo(
    const tsl::protobuf::RepeatedPtrField<
        tensorflow::profiler::PerformanceInfoWrapper::PerfInfoType::
            MemoryAccessed>& memory_accessed_breakdown,
    uint64_t occurrences);

// Annotate the op_metrics with the metadata from the instr_wrapper.
void EnterOpMetadata(OpMetrics* op_metrics,
                     const HloInstructionWrapper* instr_wrapper);
void EnterOpMetadataFromHloModuleMap(OpMetrics* op_metrics,
                                     const HloModuleMap& hlo_module_map);

void AddFusionChildrenToOpMetricsFromHloInstruction(
    OpMetrics* op_metrics, const HloInstructionWrapper* instr_wrapper);

class HostOpMetricsDbBuilder : public OpMetricsDbBuilder {
 public:
  explicit HostOpMetricsDbBuilder(OpMetricsDb* db) : OpMetricsDbBuilder(db) {}

  // A function that will be called when the end of an OP is
  // observed on a trace, where:
  //   name = the OP name.
  //   category = the OP category.
  //   is_eager = whether this OP is eagerly executed.
  //   time_ps = the total execution time of the OP in picoseconds, including
  //             the execution time of its children.
  //   children_time_ps = the execution time of the children of this OP in
  //                      picoseconds
  void EnterOp(absl::string_view name, absl::string_view category,
               bool is_eager, uint64 time_ps, uint64 children_time_ps);

  // Updates total_host_infeed_enq_duration_ps_ and
  // total_host_infeed_enq_duration_ps_.
  void EnterHostInfeedEnqueue(tsl::profiler::Timespan host_infeed_enqueue);

 private:
  // The tsl::profiler::Timespan of the last InfeedEnqueue op on this thread.
  tsl::profiler::Timespan last_host_infeed_enqueue_;
};

class DeviceOpMetricsDbBuilder : public OpMetricsDbBuilder {
 public:
  explicit DeviceOpMetricsDbBuilder(OpMetricsDb* db) : OpMetricsDbBuilder(db) {}

  // A function that will be called when the end of an OP is
  // observed on a trace, where:
  //   program_id = the ID of the program that contains this OP.
  //   name = the OP name.
  //   category = the OP category.
  //   provenance = the provenance of this OP (e.g. original TF OP).
  //   is_eager = whether this OP is eagerly executed.
  //   occurrences = the number of occurrences of this OP.
  //   time_ps = the total execution time of the OP in picoseconds, including
  //             the execution time of its children.
  //   children_time_ps = the execution time of the children of this OP in
  //                      picoseconds.
  //   flops = the number of floating-point operations computed.
  //   bytes_accessed = the sum of bytes read and bytes written by this OP.
  //   memory_accessed_breakdown = the breakdown of memory accessed by operation
  //                               type and memory space.
  void EnterOp(uint64 program_id, absl::string_view name,
               absl::string_view category, absl::string_view provenance,
               absl::string_view deduplicated_name, bool is_eager,
               uint64 occurrences, uint64 time_ps, uint64 children_time_ps,
               int64_t flops, int64_t bytes_accessed,
               const tsl::protobuf::RepeatedPtrField<OpMetrics::MemoryAccessed>&
                   memory_accessed_breakdown = {},
               int64_t model_flops = 0, absl::string_view long_name = "",
               const tsl::profiler::OpSourceInfo& op_source_info = {});

  void EnterOpMetadata(uint64 program_id, absl::string_view program_name,
                       absl::string_view category, absl::string_view provenance,
                       absl::string_view deduplicated_name, bool is_eager,
                       absl::string_view long_name = "",
                       const tsl::profiler::OpSourceInfo& op_source_info = {});

  void EnterOpMetadataFromHloModuleMap(uint64 program_id,
                                       absl::string_view op_name,
                                       const HloModuleMap& hlo_module_map);
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // XPROF_UTILS_OP_UTILS_H_
