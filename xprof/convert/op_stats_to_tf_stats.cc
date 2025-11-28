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

#include "xprof/convert/op_stats_to_tf_stats.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/types.h"
#include "xla/tsl/profiler/utils/math_utils.h"
#include "xprof/convert/data_table_utils.h"
#include "xprof/convert/model_tracker.h"
#include "xprof/convert/op_metrics_to_record.h"
#include "plugin/xprof/protobuf/op_metrics.pb.h"
#include "plugin/xprof/protobuf/op_stats.pb.h"
#include "plugin/xprof/protobuf/tf_stats.pb.h"
#include "xprof/utils/kernel_stats_utils.h"
#include "xprof/utils/op_metrics_db_utils.h"

namespace tensorflow {
namespace profiler {

namespace {
using ::tensorflow::profiler::TfStatsTable;

// The maximum number of Tensorflow Ops displayed on Tensorflow Stats page.
// 500 device side ops and 500 host side ops.
const int kMaxNumOfOps = 500;

TfStatsRecord ConvertOpMetricsToTfStatsRecord(bool on_device,
                                              const OpMetrics& metrics,
                                              const PerfEnv& perf_env,
                                              const RunEnvironment& run_env) {
  TfStatsRecord record;
  record.set_host_or_device(on_device ? "Device" : "Host");
  record.set_is_eager(metrics.is_eager());
  record.set_op_type(metrics.category());
  record.set_op_name(metrics.name());
  SetExecutionTimes(metrics, &record);
  SetRooflineMetrics(metrics, perf_env, run_env, &record);
  return record;
}

TfStatsTable GenerateTfStatsTable(
    const OpMetricsDb& host_tf_metrics_db,
    const OpMetricsDb& device_tf_metrics_db,
    const KernelStatsByOpName& kernel_stats_by_op_name, const PerfEnv& perf_env,
    const RunEnvironment& run_env, bool exclude_idle) {
  TfStatsTable tf_stats_table;
  TfStatsRecord sentinel;
  sentinel.set_rank(0);
  sentinel.set_device_cumulative_total_self_time_as_fraction(0.0);
  sentinel.set_host_cumulative_total_self_time_as_fraction(0.0);
  const TfStatsRecord* prev_record = &sentinel;

  // Sets device-side TF stats.
  uint64_t total_device_time_ps =
      TotalTimePs(device_tf_metrics_db, exclude_idle);
  double total_device_time_us =
      tsl::profiler::PicoToMicro(total_device_time_ps);
  for (const OpMetrics* metrics :
       SortedOpMetricsDb(device_tf_metrics_db, kMaxNumOfOps)) {
    if (exclude_idle && IsIdleOp(*metrics)) continue;
    TfStatsRecord* record = tf_stats_table.add_tf_stats_record();
    *record = ConvertOpMetricsToTfStatsRecord(
        /*on_device=*/true, *metrics, perf_env, run_env);
    // Compute TensorCore utilization only on device side.
    auto iter = kernel_stats_by_op_name.find(record->op_name());
    if (iter != kernel_stats_by_op_name.end()) {
      record->set_gpu_tensorcore_utilization(
          tsl::profiler::SafeDivide(iter->second.tensor_core_duration_ns,
                                    iter->second.total_duration_ns));
    } else {
      record->set_gpu_tensorcore_utilization(0.0);
    }
    SetRankAndDeviceTimeFractions(total_device_time_us, *prev_record, record);
    prev_record = record;
  }

  // Sets host-side TF stats.
  uint64_t total_host_time_ps = TotalTimePs(host_tf_metrics_db, exclude_idle);
  double total_host_time_us = tsl::profiler::PicoToMicro(total_host_time_ps);
  for (const OpMetrics* metrics : tensorflow::profiler::SortedOpMetricsDb(
           host_tf_metrics_db, kMaxNumOfOps)) {
    if (exclude_idle && IsIdleOp(*metrics)) continue;
    TfStatsRecord* record = tf_stats_table.add_tf_stats_record();
    *record = ConvertOpMetricsToTfStatsRecord(
        /*on_device=*/false, *metrics, perf_env, run_env);
    // Host side TensorCore utilization is always 0.0
    record->set_gpu_tensorcore_utilization(0.0);
    SetRankAndHostTimeFractions(total_host_time_us, *prev_record, record);
    prev_record = record;
  }
  return tf_stats_table;
}

}  // namespace

TfStatsDatabase ConvertOpStatsToTfStats(const OpStats& op_stats) {
  const OpMetricsDb& host_tf_metrics_db = op_stats.host_op_metrics_db();
  OpMetricsDb device_tf_metrics_db =
      CreateTfMetricsDbFromDeviceOpMetricsDb(op_stats.device_op_metrics_db());
  const PerfEnv perf_env = op_stats.perf_env();
  const RunEnvironment run_env = op_stats.run_environment();
  KernelStatsByOpName kernel_stats_by_op_name =
      GroupKernelReportsByOpName(op_stats.kernel_stats_db());
  TfStatsDatabase tf_stats_db;
  *tf_stats_db.mutable_with_idle() = GenerateTfStatsTable(
      host_tf_metrics_db, device_tf_metrics_db, kernel_stats_by_op_name,
      perf_env, run_env, /*exclude_idle=*/false);
  *tf_stats_db.mutable_without_idle() = GenerateTfStatsTable(
      host_tf_metrics_db, device_tf_metrics_db, kernel_stats_by_op_name,
      perf_env, run_env, /*exclude_idle=*/true);
  tf_stats_db.set_device_type(op_stats.run_environment().device_type());
  return tf_stats_db;
}

std::unique_ptr<DataTable> TfStatsToDataTable(const TfStatsTable& table,
                                              absl::string_view device_type) {
  std::vector<std::vector<std::string>> kColumns = {
      {"rank", "number", "Rank"},
      {"host_or_device", "string", "Host/device"},
      {"type", "string", "Operation Type"},
      {"operation", "string", "Operation Name"},
      {"occurrences", "number", "#Occurrences"},
      {"total_time", "number", "Total time (us)"},
      {"avg_time", "number", "Avg. time (us)"},
      {"total_self_time", "number", "Total self-time (us)"},
      {"avg_self_time", "number", "Avg. self-time (us)"},
      {"device_total_self_time_percent", "number",
       "Total self-time on Device (%)"},
      {"device_cumulative_total_self_time_percent", "number",
       "Cumulative total-self time on Device (%)"},
      {"host_total_self_time_percent", "number", "Total self-time on Host (%)"},
      {"Host_cumulative_total_self_time_percent", "number",
       "Cumulative total-self time on Host (%)"},
      {"measured_flop_rate", "number", "Normalized FLOP Rate (FLOPs/s)"},
      {"model_flop_rate", "number", "Model FLOP Rate (GFLOP/s)"},
      {"measured_memory_bw", "number", "Measured Memory BW (GBytes/Sec)"},
      {"operational_intensity", "number", "Operational Intensity (FLOPs/Byte)"},
  };
  auto data_table = std::make_unique<DataTable>();
  for (const std::vector<std::string>& col : kColumns) {
    data_table->AddColumn(TableColumn(col[0], col[1], col[2]));
  }
  // Add GPU TensorCore utilization only when hardware device type is GPU, for
  // other devices, the utilization number will always be 0
  if (absl::StrContains(device_type, "GPU")) {
    data_table->AddColumn(TableColumn("gpu_tensorcore_utilization", "number",
                                      "GPU TensorCore utilization"));
  }
  data_table->AddColumn(TableColumn("bound_by", "string", "Bound by"));
  data_table->AddColumn(TableColumn("eager", "string", "Execution mode"));

  ModelTracker model;

  for (const auto& record : table.tf_stats_record()) {
    TableRow* row = data_table->AddRow();
    row->AddNumberCell(record.rank());
    row->AddTextCell(record.host_or_device());
    row->AddTextCell(record.op_type());
    row->AddTextCell(record.op_name());
    row->AddNumberCell(record.occurrences());
    row->AddNumberCell(record.total_time_in_us());
    row->AddNumberCell(record.avg_time_in_us());
    row->AddNumberCell(record.total_self_time_in_us());
    row->AddNumberCell(record.avg_self_time_in_us());
    row->AddNumberCell(record.device_total_self_time_as_fraction());
    row->AddNumberCell(record.device_cumulative_total_self_time_as_fraction());
    row->AddNumberCell(record.host_total_self_time_as_fraction());
    row->AddNumberCell(record.host_cumulative_total_self_time_as_fraction());
    row->AddNumberCell(record.measured_flop_rate());
    row->AddNumberCell(record.model_flop_rate());
    row->AddNumberCell(record.measured_memory_bw());
    row->AddNumberCell(record.operational_intensity());
    if (absl::StrContains(device_type, "GPU")) {
      row->AddNumberCell(record.gpu_tensorcore_utilization());
    }
    row->AddTextCell(record.bound_by());
    row->AddTextCell(record.is_eager() ? "Eager" : "Function");

    model.ProcessOp(record.op_name(), record.op_type());
  }

  data_table->AddCustomProperty("task_type", model.GetTaskType());
  data_table->AddCustomProperty("architecture_type",
                                model.GetArchitectureType());
  return data_table;
}

std::string TfStatsToDataTableJson(const TfStatsDatabase& tf_stats_db) {
  std::unique_ptr<DataTable> data_table_with_idle =
      TfStatsToDataTable(tf_stats_db.with_idle(), tf_stats_db.device_type());
  std::unique_ptr<DataTable> data_table_without_idle =
      TfStatsToDataTable(tf_stats_db.without_idle(), tf_stats_db.device_type());
  return absl::StrCat("[", data_table_with_idle->ToJson(), ",",
                      data_table_without_idle->ToJson(), "]");
}

}  // namespace profiler
}  // namespace tensorflow
