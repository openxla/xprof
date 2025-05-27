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

#include "xprof/convert/xplane_to_kernel_stats_db.h"

#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/profiler/utils/math_utils.h"
#include "xla/tsl/profiler/utils/tf_op_utils.h"
#include "xla/tsl/profiler/utils/tf_xplane_visitor.h"
#include "xla/tsl/profiler/utils/trace_utils.h"
#include "xla/tsl/profiler/utils/xplane_visitor.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/data_table_utils.h"
#include "plugin/xprof/protobuf/kernel_stats.pb.h"
#include "xprof/utils/gpu_event_stats.h"
#include "xprof/utils/kernel_stats_utils.h"

namespace tensorflow {
namespace profiler {

void ConvertDeviceTraceXPlaneToKernelReports(
    const XPlane& device_trace,
    const std::function<void(const GpuEventStats&, KernelReport*)>&
        on_kernel_fn,
    KernelReportMap* reports) {
  tsl::profiler::XPlaneVisitor plane =
      tsl::profiler::CreateTfXPlaneVisitor(&device_trace);
  plane.ForEachLine([&](const tsl::profiler::XLineVisitor& line) {
    if (tsl::profiler::IsDerivedThreadId(line.Id())) {
      return;
    }
    line.ForEachEvent([&](const tsl::profiler::XEventVisitor& event) {
      if (event.DurationNs() == 0) return;
      KernelReport kernel;
      GpuEventStats stats(&event);
      if (!stats.IsKernel()) return;

      kernel.set_name(std::string(event.Name()));
      kernel.set_is_kernel_using_tensor_core(
          IsKernelUsingTensorCore(event.Name()));
      kernel.set_total_duration_ns(event.DurationNs());
      kernel.set_min_duration_ns(event.DurationNs());
      kernel.set_max_duration_ns(event.DurationNs());
      ParseKernelLaunchParams(stats.kernel_details, &kernel);

      if (stats.IsTfOp()) {
        tsl::profiler::TfOp tf_op =
            tsl::profiler::ParseTfOpFullname(stats.tf_op_fullname);
        kernel.set_op_name(std::string(tf_op.name));
        bool tensor_core_eligible =
            IsEinsumTensorCoreEligible(stats.equation) ||
            IsOpTensorCoreEligible(kernel.op_name());
        if (!tensor_core_eligible && kernel.is_kernel_using_tensor_core()) {
          VLOG(1) << "Detected new Op using TensorCores: " << kernel.op_name()
                  << std::endl;
          tensor_core_eligible = true;
        }
        kernel.set_is_op_tensor_core_eligible(tensor_core_eligible);
      }

      if (on_kernel_fn) {
        on_kernel_fn(stats, &kernel);
      }

      KernelReportValue value;
      value.total_duration_ns = event.DurationNs();
      value.min_duration_ns = event.DurationNs();
      value.max_duration_ns = event.DurationNs();
      value.occurrences = 1;
      InsertOrUpdateKernelReport(kernel, value, reports);
    });
  });
}

std::unique_ptr<DataTable> GenerateKernelStatsDataTable(
    const KernelStatsDb& kernel_stats_db) {
  std::vector<std::vector<std::string>> kColumns = {
      {"rank", "number", "Rank"},
      {"kernel_name", "string", "Kernel Name"},
      {"registers_per_thread", "number", "Registers per thread"},
      {"shmem_bytes", "number", "Shared Mem bytes"},
      {"block_dim", "string", "Block dim"},
      {"grid_dim", "string", "Grid dim"},
      {"occupancy_pct", "number", "Occupancy %"},
      {"is_op_tensor_core_eligible", "boolean", "Op is TensorCore eligible"},
      {"is_kernel_using_tensor_core", "boolean", "Kernel uses TensorCore"},
      {"op_name", "string", "Op Name"},
      {"occurrences", "number", "Occurrences"},
      {"total_duration_us", "number", "Total Duration (μs)"},
      {"avg_duration_us", "number", "Avg Duration (μs)"},
      {"min_duration_us", "number", "Min Duration (μs)"},
      {"max_duration_us", "number", "Max Duration (μs)"},
  };

  auto data_table = std::make_unique<DataTable>();
  for (const std::vector<std::string>& col : kColumns) {
    data_table->AddColumn(TableColumn(col[0], col[1], col[2]));
  }

  uint64_t prev_duration = std::numeric_limits<uint64_t>::max();
  uint32_t rank = 1;
  for (const auto& report : kernel_stats_db.reports()) {
    TableRow* row = data_table->AddRow();
    auto grid_dim = absl::StrJoin(report.grid_dim(), ",");
    auto block_dim = absl::StrJoin(report.block_dim(), ",");
    row->AddNumberCell(rank);
    row->AddTextCell(report.name());
    row->AddNumberCell(report.registers_per_thread());
    row->AddNumberCell(report.static_shmem_bytes() +
                       report.dynamic_shmem_bytes());
    row->AddTextCell(block_dim);
    row->AddTextCell(grid_dim);
    row->AddNumberCell(report.occupancy_pct());
    row->AddBooleanCell(report.is_op_tensor_core_eligible());
    row->AddBooleanCell(report.is_kernel_using_tensor_core());
    row->AddTextCell(report.op_name());
    row->AddNumberCell(report.occurrences());
    row->AddNumberCell(tsl::profiler::NanoToMicro(report.total_duration_ns()));
    row->AddNumberCell(tsl::profiler::NanoToMicro(report.total_duration_ns() /
                                                  report.occurrences()));
    row->AddNumberCell(tsl::profiler::NanoToMicro(report.min_duration_ns()))
        .AddNumberCell(tsl::profiler::NanoToMicro(report.max_duration_ns()));
    // Check that rows are sorted by total_duration descendingly.
    DCHECK_LE(report.total_duration_ns(), prev_duration);
    prev_duration = report.total_duration_ns();
    rank += 1;
  }
  return data_table;
}

std::string KernelStatsToDataTableJson(const KernelStatsDb& kernel_stats_db) {
  return GenerateKernelStatsDataTable(kernel_stats_db)->ToJson();
}

}  // namespace profiler
}  // namespace tensorflow
