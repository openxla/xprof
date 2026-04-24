/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "xprof/convert/op_stats_to_hlo_stats.h"

#include <memory>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "xla/tsl/profiler/convert/xla_op_utils.h"
#include "xla/tsl/profiler/utils/math_utils.h"
#include "xla/tsl/profiler/utils/tf_op_utils.h"
#include "xprof/convert/data_table_utils.h"
#include "xprof/convert/op_metrics_to_record.h"
#include "xprof/convert/source_info_utils.h"
#include "plugin/xprof/protobuf/hlo_stats.pb.h"
#include "plugin/xprof/protobuf/op_metrics.pb.h"
#include "plugin/xprof/protobuf/op_stats.pb.h"
#include "plugin/xprof/protobuf/source_info.pb.h"

namespace tensorflow {
namespace profiler {
namespace {

using tensorflow::profiler::OpMetrics;
using tensorflow::profiler::OpMetricsDb;
using ::tensorflow::profiler::OpStats;
using ::tensorflow::profiler::PerfEnv;
using ::tensorflow::profiler::RunEnvironment;
using tensorflow::profiler::hlo_stats::HloStatsDatabase;
using tensorflow::profiler::hlo_stats::HloStatsRecord;
using tsl::profiler::IsOutsideCompilationOp;

std::string GetTpuCoreType(OpMetrics::TpuCoreType core_type) {
  switch (core_type) {
    case OpMetrics::TENSOR_CORE:
      return "TensorCore";
    case OpMetrics::SPARSE_CORE:
      return "SparseCore";
    default:
      return "";
  }
}

HloStatsRecord ConvertOpMetricsToHloStatsRecord(
    const OpMetrics& metrics, const PerfEnv& perf_env,
    const RunEnvironment& run_env, const std::string& parent_op_name) {
  HloStatsRecord record;
  record.set_program_id(metrics.hlo_module_id());
  // If long_name is empty, use the name instead for the HLO expression.
  // This is because some ops don't have a long_name due to the event metadata
  // not having a display name
  // (op_metrics_db_utils.cc:SetOpMetadataFromHloEventMetadata).
  record.set_hlo_expression(metrics.long_name().empty() ? metrics.name()
                                                        : metrics.long_name());
  record.set_tf_op_name(metrics.provenance());
  record.set_hlo_category(metrics.category());
  record.set_autotuned(metrics.autotuned());
  tensorflow::profiler::SetExecutionTimes(metrics, &record);
  tensorflow::profiler::SetTpuUnitFractions(metrics, &record);
  SetRooflineMetrics(metrics, perf_env, run_env, &record);
  record.set_rematerialization(tsl::profiler::IsRematerialization(
      /*hlo_expression=*/metrics.long_name(),
      /*framework_op_name=*/metrics.provenance()));
  record.set_outside_compilation(
      IsOutsideCompilationOp(metrics.provenance(), metrics.long_name()));
  record.set_core_type(GetTpuCoreType(metrics.core_type()));
  record.set_parent_op_name(parent_op_name);
  *record.mutable_source_info() = metrics.source_info();
  return record;
}

void AddHloStatsRecordsRecursively(HloStatsDatabase& hlo_stats_db,
                                   const OpMetrics* metrics,
                                   const OpStats& op_stats,
                                   const HloStatsRecord*& prev_record,
                                   double total_device_time_us,
                                   const std::string& parent_op_name) {
  // Skip empty metrics.
  if (metrics->occurrences() == 0) return;
  // For child ops, we only want the SparseCore ops.
  if (parent_op_name.empty() ||
      metrics->core_type() == OpMetrics::SPARSE_CORE) {
    HloStatsRecord* record = hlo_stats_db.add_hlo_stats_record();
    *record = ConvertOpMetricsToHloStatsRecord(*metrics, op_stats.perf_env(),
                                               op_stats.run_environment(),
                                               parent_op_name);
    // Only set the rank and time fractions if there is a previous record.
    if (prev_record != nullptr) {
      tensorflow::profiler::SetRankAndTimeFractions(total_device_time_us,
                                                    *prev_record, record);
      prev_record = record;
    }
  }
  // Recursively add records for ops from child metrics dbs.
  if (metrics->children().metrics_db_size() > 0) {
    // Do not accumulate running totals for children ops.
    const HloStatsRecord* child_prev_record = nullptr;
    double child_total_device_time_us =
        tsl::profiler::PicoToMicro(metrics->children().total_time_ps());
    for (const OpMetrics* child_metrics :
         tensorflow::profiler::SortedOpMetricsDb(metrics->children())) {
      AddHloStatsRecordsRecursively(
          hlo_stats_db, child_metrics, op_stats, child_prev_record,
          child_total_device_time_us, metrics->name());
    }
  }
}

}  // namespace

HloStatsDatabase ConvertOpStatsToHloStats(const OpStats& op_stats) {
  HloStatsDatabase hlo_stats_db;
  const OpMetricsDb& hlo_metrics_db = op_stats.device_op_metrics_db();
  double total_device_time_us =
      tsl::profiler::PicoToMicro(hlo_metrics_db.total_time_ps());
  HloStatsRecord sentinel;
  sentinel.set_rank(0);
  sentinel.set_cumulative_total_self_time_as_fraction(0.0);
  const HloStatsRecord* prev_record = &sentinel;
  for (const OpMetrics* metrics :
       tensorflow::profiler::SortedOpMetricsDb(hlo_metrics_db)) {
    AddHloStatsRecordsRecursively(hlo_stats_db, metrics, op_stats, prev_record,
                                  total_device_time_us, /*parent_op_name=*/"");
  }
  return hlo_stats_db;
}

// The parse logic based on the assumption that the hlo op text is in format of
// '%op_name = <long name>'
std::string GetHloOpNameFromExpression(std::string expression) {
  std::vector<::std::string> parts = absl::StrSplit(expression, " = ");
  std::string hlo_op_name = parts[0];
  if (hlo_op_name[0] == '%') {
    hlo_op_name = hlo_op_name.substr(1);
  }
  return hlo_op_name;
}

std::vector<std::vector<std::string>> HloStatsDataTableColumns() {
  const std::vector<std::vector<std::string>> kColumns = {
      {"rank", "number", "Rank"},
      {"program_id", "string", "Program id"},
      {"category", "string", "HLO op category"},
      {"hlo_op_name", "string", "HLO op name"},
      {"hlo_op_expression", "string", "HLO op text"},
      {"tf_op_name", "string", "Framework op name"},
      {"occurrences", "number", "#Occurrences"},
      {"total_time", "number", "Total time (us)"},
      {"avg_time", "number", "Avg. time (us)"},
      {"total_self_time", "number", "Total self time (us)"},
      {"avg_self_time", "number", "Avg. self time (us)"},
      {"total_self_time_percent", "number", "Total self time (%)"},
      {
          "cumulative_total_self_time_percent",
          "number",
          "Cumulative total self time (%)",
      },
      {"dma_stall_percent", "number", "%time stalled by DMA"},
      {"model_flop_rate", "number", "Model GFLOP/s"},
      {"normalized_flop_rate", "number", "Normalized GFLOP/s"},
      {"measured_memory_bw", "number", "Measured memory BW (GiB/s)"},
      {"hbm_bw", "number", "HBM BW (GiB/s)"},
      {"cmem_read_bw", "number", "CMEM Read BW (GiB/s)"},
      {"cmem_write_bw", "number", "CMEM Write BW (GiB/s)"},
      {"operational_intensity", "number", "Operational intensity (FLOPS/Byte)"},
      {"bound_by", "string", "Bound by"},
      {"hlo_rematerialization", "string", "Rematerialization"},
      {"outside_compilation", "string", "Outside Compilation"},
      {"autotuned", "string", "Autotuned"},
      {"source_info", "string", "Source Info"},
      {"core_type", "string", "TPU core type"},
      {"parent_op_name", "string", "Parent op name"},
  };
  return kColumns;
}

std::unique_ptr<tensorflow::profiler::DataTable> CreateHloStatsDataTable(
    const HloStatsDatabase& hlo_stats_db) {
  auto data_table = std::make_unique<tensorflow::profiler::DataTable>();
  for (const std::vector<std::string>& col : HloStatsDataTableColumns()) {
    data_table->AddColumn(TableColumn(col[0], col[1], col[2]));
  }
  for (const HloStatsRecord& record : hlo_stats_db.hlo_stats_record()) {
    TableRow* row = data_table->AddRow();
    row->AddNumberCell(record.rank());
    row->AddTextCell(absl::StrCat(record.program_id()));
    row->AddTextCell(record.hlo_category());
    row->AddTextCell(GetHloOpNameFromExpression(record.hlo_expression()));
    row->AddTextCell(record.hlo_expression());
    row->AddTextCell(record.tf_op_name());
    row->AddNumberCell(record.occurrences());
    row->AddNumberCell(record.total_time_in_us());
    row->AddNumberCell(record.avg_time_in_us());
    row->AddNumberCell(record.total_self_time_in_us());
    row->AddNumberCell(record.avg_self_time_in_us());
    row->AddNumberCell(record.total_self_time_as_fraction() * 100.0);
    row->AddNumberCell(record.cumulative_total_self_time_as_fraction() * 100.0);
    row->AddNumberCell(record.dma_stall_fraction());
    row->AddNumberCell(record.model_flop_rate());
    row->AddNumberCell(record.measured_flop_rate());
    row->AddNumberCell(record.measured_memory_bw());
    row->AddNumberCell(record.hbm_bw());
    row->AddNumberCell(record.cmem_read_bw());
    row->AddNumberCell(record.cmem_write_bw());
    row->AddNumberCell(record.operational_intensity());
    row->AddTextCell(record.bound_by());
    row->AddTextCell(record.rematerialization() ? "Yes" : "No");
    row->AddTextCell(record.outside_compilation() ? "Yes" : "No");
    row->AddTextCell(record.autotuned() ? "Yes" : "No");
    row->AddTextCell(SourceInfoFormattedText(record.source_info()));
    row->AddTextCell(record.core_type());
    row->AddTextCell(record.parent_op_name());
  }
  return data_table;
}

std::string HloStatsToDataTableJson(const HloStatsDatabase& hlo_stats_db) {
  return CreateHloStatsDataTable(hlo_stats_db)->ToJson();
}

}  // namespace profiler
}  // namespace tensorflow
