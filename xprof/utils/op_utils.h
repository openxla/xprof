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
#include "plugin/xprof/protobuf/op_metrics.pb.h"
#include "xprof/utils/hlo_module_map.h"
#include "xprof/utils/op_metrics_db_utils.h"
#include "xprof/utils/performance_info_wrapper.h"

namespace tensorflow {
namespace profiler {
using ::tensorflow::profiler::OpMetrics;
using ::tensorflow::profiler::OpMetrics_MemoryAccessed;

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
  //   id = host op uniqueness identifier. For input pipeline ops, this is the
  //        stage id. By default is 0 if not needed.
  void EnterOp(absl::string_view name, absl::string_view category,
               bool is_eager, uint64_t time_ps, uint64_t children_time_ps,
               int64_t id = 0);

  // Updates total_host_infeed_enq_duration_ps_ and
  // total_host_infeed_enq_duration_ps_.
  void EnterHostInfeedEnqueue(tsl::profiler::Timespan host_infeed_enqueue);

 private:
  // The tsl::profiler::Timespan of the last InfeedEnqueue op on this thread.
  tsl::profiler::Timespan last_host_infeed_enqueue_;
};

class DeviceOpMetricsDbBuilder : public OpMetricsDbBuilder {
 public:
  // Structure to hold identifying information and metadata about an Operation.
  //
  // This struct groups parameters that describe the static properties of the
  // operation itself, such as its name, category, origin, and source code
  // location information. It serves as the single source of truth for
  // identifying operations across both 1P and 3P profiling paths.
  struct OpIdentifier {
    // ID of the program (e.g., HLO module) containing the OP.
    uint64_t program_id;
    // The specific OP name.
    absl::string_view name;
    // The OP category (e.g., "Convolution", "MatMul").
    absl::string_view category;
    // Origin info, like the corresponding TensorFlow OP name.
    absl::string_view provenance;
    // Deduplicated HLO name for this OP.
    absl::string_view deduplicated_name;
    // Optional longer descriptive name for the OP.
    absl::string_view long_name = "";
    // Detailed source information (e.g., file, line).
    tsl::profiler::OpSourceInfo op_source_info;
  };

  // Structure to hold data specific to the traced event of an Operation.
  //
  // This struct groups parameters related to a specific instance or observation
  // of the operation within a trace, including timing, execution context, and
  // performance counters. It serves as the single source of truth for metrics
  // accumulation across both 1P and 3P profiling paths.
  struct OpData {
    // Flag indicating if the OP was eagerly executed.
    bool is_eager = false;
    // The number of occurrences of this OP.
    uint64_t occurrences = 1;
    // Total execution time (inclusive of children) in picoseconds.
    uint64_t time_ps = 0;
    // Execution time solely of children OPs in picoseconds.
    uint64_t children_time_ps = 0;

    // GPU-specific: Time spent stalled on DMA in picoseconds.
    uint64_t dma_stall_ps = 0;

    // GPU-specific / Symbol-specific: Optional pointer to detailed performance
    // analysis info. If provided, flops, bytes_accessed, etc. will be populated
    // from it.
    const tensorflow::profiler::PerformanceInfoWrapper* perf_info = nullptr;

    // TPU-specific / General performance counters:
    // The number of floating-point operations computed.
    int64_t flops = 0;
    // The sum of bytes read and bytes written by this OP.
    int64_t bytes_accessed = 0;
    // The breakdown of memory accessed by operation type and memory space.
    tsl::protobuf::RepeatedPtrField<OpMetrics::MemoryAccessed>
        memory_accessed_breakdown;
    // The number of floating-point operations computed by the model.
    int64_t model_flops = 0;
    // The total VDD energy tracked in Joules.
    double vdd_energy_j = 0.0;
  };

  explicit DeviceOpMetricsDbBuilder(OpMetricsDb* db) : OpMetricsDbBuilder(db) {}

  // Updates the metrics database with the provided `op_id` and `event_data`
  // when an end-of-OP event is observed.
  void EnterOp(const OpIdentifier& op_id, const OpData& event_data);

  // Updates the metadata of the OP in the database.
  void EnterOpMetadata(const OpIdentifier& op_id, bool is_eager);

  void EnterOpMetadataFromHloModuleMap(uint64_t program_id,
                                       absl::string_view op_name,
                                       const HloModuleMap& hlo_module_map);
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // XPROF_UTILS_OP_UTILS_H_
