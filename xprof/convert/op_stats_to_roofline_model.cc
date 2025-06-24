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

#include "xprof/convert/op_stats_to_roofline_model.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "xla/tsl/profiler/convert/xla_op_utils.h"
#include "xla/tsl/profiler/utils/math_utils.h"
#include "tsl/platform/protobuf.h"
#include "xprof/convert/data_table_utils.h"
#include "xprof/convert/op_metrics_db_combiner.h"
#include "xprof/convert/op_metrics_to_record.h"
#include "plugin/xprof/protobuf/hardware_types.pb.h"
#include "plugin/xprof/protobuf/op_metrics.pb.h"
#include "plugin/xprof/protobuf/op_stats.pb.h"
#include "plugin/xprof/protobuf/roofline_model.pb.h"
#include "plugin/xprof/protobuf/source_info.pb.h"
#include "plugin/xprof/protobuf/steps_db.pb.h"
#include "xprof/utils/diagnostics.h"
#include "xprof/utils/roofline_model_utils.h"

namespace tensorflow {
namespace profiler {
namespace {

using tensorflow::profiler::DataTable;
using tensorflow::profiler::OpMetrics;
using tensorflow::profiler::OpMetricsDb;
using tensorflow::profiler::PerfEnv;
using ::tensorflow::profiler::RecordType;
using ::tensorflow::profiler::RooflineModelDatabase;
using ::tensorflow::profiler::RooflineModelRecord;
using tensorflow::profiler::roofline_model::RecordType;
using tensorflow::profiler::roofline_model::RooflineModelDatabase;
using tensorflow::profiler::roofline_model::RooflineModelRecord;

// The maximum number of records to generate.
const uint32_t kMaxNumRecords = 1000;
}  // namespace

RooflineModelRecord ConvertOpMetricsToRooflineModelRecord(
    const OpStats& op_stats, const OpMetrics& metrics, RecordType record_type,
    uint32_t step_num, uint64_t total_time_ps,
    const RooflineModelDatabase& roofline_model_db,
    bool include_infeed_outfeed) {
  RooflineModelRecord record;
  record.set_hlo_name(metrics.name());
  record.set_hlo_category(metrics.category());
  record.set_hlo_module_id(metrics.hlo_module_id());
  record.set_record_type(record_type);
  record.set_step_num(step_num);
  *record.mutable_source_info() = metrics.source_info();
  SetExecutionTimes(metrics, &record);
  if (record_type == RecordType::AVERAGE_STEP) {
    // For RecordType::AVERAGE_STEP, divide by num_steps to show per-step
    // numbers when appropriate.
    int num_steps = op_stats.step_db().step_sequence_size();
    record.set_total_time_in_us(
        tsl::profiler::SafeDivide(record.total_time_in_us(), num_steps));
    record.set_total_self_time_in_us(
        tsl::profiler::SafeDivide(record.total_self_time_in_us(), num_steps));
  }
  record.set_total_time_per_core_in_us(tsl::profiler::SafeDivide(
      record.total_time_in_us(),
      op_stats.run_environment().device_core_count()));
  record.set_total_time_in_percentage(
      tsl::profiler::SafeDivide(metrics.time_ps(), total_time_ps));

  tensorflow::profiler::SetTpuUnitFractions(metrics, &record);

  // Set the roofline-specific fields.
  SetRooflineMetrics(metrics, op_stats.perf_env(), op_stats.run_environment(),
                     &record);
  const double cmem_wr_utilization =
      roofline_model_db.has_cmem()
          ? tsl::profiler::SafeDivide(record.cmem_write_bw(),
                                      roofline_model_db.peak_cmem_write_bw())
          : 0;
  const double cmem_rd_utilization =
      roofline_model_db.has_cmem()
          ? tsl::profiler::SafeDivide(record.cmem_read_bw(),
                                      roofline_model_db.peak_cmem_read_bw())
          : 0;
  const double vmem_rd_utilization =
      roofline_model_db.has_merged_vmem()
          ? tsl::profiler::SafeDivide(record.vmem_read_bw(),
                                      roofline_model_db.peak_vmem_read_bw())
          : 0;
  const double vmem_wr_utilization =
      roofline_model_db.has_merged_vmem()
          ? tsl::profiler::SafeDivide(record.vmem_write_bw(),
                                      roofline_model_db.peak_vmem_write_bw())
          : 0;
  const double flops_utilization = tsl::profiler::SafeDivide(
      record.measured_flop_rate(), roofline_model_db.peak_flop_rate());
  const double hbm_utilization = tsl::profiler::SafeDivide(
      record.hbm_bw(), roofline_model_db.peak_hbm_bw());

  const double max_mem_utilization =
      std::max({cmem_wr_utilization, cmem_rd_utilization, hbm_utilization,
                vmem_wr_utilization, vmem_rd_utilization});
  const double roofline_efficiency =
      std::max({max_mem_utilization, flops_utilization});
  // Note, copy-start/done can have utilizations above 1.0 since their
  // bytes/time are not accurate as they are asynchronous.
  record.set_optimal_flop_rate(tsl::profiler::SafeDivide(
      record.measured_flop_rate(), roofline_efficiency));
  record.set_roofline_efficiency(roofline_efficiency);
  record.set_flop_rate_relative_to_hw_limit(flops_utilization);
  record.set_memory_bw_relative_to_hw_limit(max_mem_utilization);

  record.set_include_infeed_outfeed(include_infeed_outfeed);

  return record;
}

RooflineModelRecord GenerateRooflineModelProgramRecord(
    const OpStats& op_stats, const OpMetricsDb& db, RecordType record_type,
    uint32_t step_num, const RooflineModelDatabase& roofline_model_db,
    bool include_infeed_outfeed) {
  OpMetrics program_metrics;
  program_metrics.set_name("Program");
  program_metrics.set_category("Program");
  program_metrics.set_occurrences(1);
  uint64_t infeed_outfeed_time = 0;
  for (const OpMetrics& metrics : db.metrics_db()) {
    // Aggregate innermost ops only to avoid redundant counting.
    if (tsl::profiler::MayHaveInnerOps(metrics.category())) continue;
    if (!include_infeed_outfeed &&
        tsl::profiler::IsInfeedOrOutfeed(metrics.category())) {
      infeed_outfeed_time += metrics.time_ps();
      continue;
    }
    program_metrics.set_flops(program_metrics.flops() + metrics.flops());
    program_metrics.set_model_flops(program_metrics.model_flops() +
                                    metrics.model_flops());
    program_metrics.set_bytes_accessed(program_metrics.bytes_accessed() +
                                       metrics.bytes_accessed());
    CombineMemoryAccessedBreakdown(
        metrics.memory_accessed_breakdown(),
        program_metrics.mutable_memory_accessed_breakdown());
  }
  uint64_t total_time_ps = db.total_time_ps();
  if (!include_infeed_outfeed) total_time_ps -= infeed_outfeed_time;
  program_metrics.set_time_ps(total_time_ps);
  RooflineModelRecord program_record = ConvertOpMetricsToRooflineModelRecord(
      op_stats, program_metrics, record_type, step_num, total_time_ps,
      roofline_model_db, include_infeed_outfeed);
  program_record.set_rank(0);
  program_record.set_total_self_time_as_fraction(0.0);
  program_record.set_cumulative_total_self_time_as_fraction(0.0);
  return program_record;
}

tsl::protobuf::RepeatedPtrField<RooflineModelRecord>
ConvertOpMetricsDbToRooflineModelRecords(
    const OpStats& op_stats, const OpMetricsDb& db, RecordType record_type,
    uint32_t step_num, const RooflineModelDatabase& roofline_model_db,
    bool include_infeed_outfeed) {
  tsl::protobuf::RepeatedPtrField<RooflineModelRecord> roofline_model_records;
  RooflineModelRecord* program_record = roofline_model_records.Add();
  *program_record = GenerateRooflineModelProgramRecord(
      op_stats, db, record_type, step_num, roofline_model_db,
      include_infeed_outfeed);
  const RooflineModelRecord* prev_record = program_record;
  uint64_t infeed_outfeed_time = 0;
  if (!include_infeed_outfeed) {
    // Calculate the total time spent on infeed and outfeed ops.
    for (const OpMetrics& metrics : db.metrics_db()) {
      if (tsl::profiler::IsInfeedOrOutfeed(metrics.category())) {
        infeed_outfeed_time += metrics.time_ps();
      }
    }
  }
  uint64_t total_time_ps = db.total_time_ps() - infeed_outfeed_time;
  double total_time_us = tsl::profiler::PicoToMicro(total_time_ps);
  for (const auto* metrics : SortedOpMetricsDb(db, kMaxNumRecords)) {
    if (metrics->occurrences() == 0) continue;
    if (!include_infeed_outfeed &&
        tsl::profiler::IsInfeedOrOutfeed(metrics->category())) {
      continue;
    }
    RooflineModelRecord* record = roofline_model_records.Add();
    *record = ConvertOpMetricsToRooflineModelRecord(
        op_stats, *metrics, record_type, step_num, total_time_ps,
        roofline_model_db, include_infeed_outfeed);
    SetRankAndTimeFractions(total_time_us, *prev_record, record);
    prev_record = record;
  }
  return roofline_model_records;
}

RooflineModelDatabase InitializeRooflineModelDatabaseFromOpStats(
    const OpStats& op_stats, bool include_infeed_outfeed) {
  tensorflow::profiler::HardwareType hardware_type =
      op_stats.run_environment().hardware_type();
  DCHECK(hardware_type == GPU || hardware_type == TPU);

  RooflineModelDatabase roofline_model_db;
  const PerfEnv& perf_env = op_stats.perf_env();
  roofline_model_db.set_device_type(op_stats.run_environment().device_type());

  // Set peak flop rate in GFLOPs/s.
  roofline_model_db.set_peak_flop_rate(
      tsl::profiler::TeraToGiga((perf_env.peak_tera_flops_per_second())));
  roofline_model_db.set_peak_hbm_bw(
      tsl::profiler::GigaToGibi(GetMemoryPeakBandwidth(perf_env, 0)));

  if (hardware_type == HardwareType::TPU) {
    roofline_model_db.set_megacore(perf_env.has_megacore());

    roofline_model_db.set_has_cmem(perf_env.has_cmem());
    roofline_model_db.set_has_merged_vmem(perf_env.has_merged_vmem());
    if (roofline_model_db.has_cmem()) {
      roofline_model_db.set_peak_cmem_read_bw(
          tsl::profiler::GigaToGibi(GetMemoryPeakBandwidth(perf_env, 3)));
      roofline_model_db.set_peak_cmem_write_bw(
          tsl::profiler::GigaToGibi(GetMemoryPeakBandwidth(perf_env, 4)));
    } else if (roofline_model_db.has_merged_vmem()) {
      roofline_model_db.set_peak_vmem_read_bw(
          tsl::profiler::GigaToGibi(GetMemoryPeakBandwidth(perf_env, 5)));
      roofline_model_db.set_peak_vmem_write_bw(
          tsl::profiler::GigaToGibi(GetMemoryPeakBandwidth(perf_env, 6)));
    }
  } else if (hardware_type == HardwareType::GPU) {
    roofline_model_db.set_megacore(false);
    roofline_model_db.set_has_cmem(false);
    roofline_model_db.set_has_merged_vmem(true);
    roofline_model_db.set_peak_vmem_read_bw(
        tsl::profiler::GigaToGibi(GetMemoryPeakBandwidth(perf_env, 1)));
    roofline_model_db.set_peak_vmem_write_bw(
        tsl::profiler::GigaToGibi(GetMemoryPeakBandwidth(perf_env, 2)));
  }

  return roofline_model_db;
}

RooflineModelDatabase ConvertOpStatsToRooflineModel(
    const OpStats& op_stats, bool include_infeed_outfeed) {
  HardwareType hardware_type = op_stats.run_environment().hardware_type();
  if (hardware_type != GPU && hardware_type != TPU) {
    return RooflineModelDatabase();
  }

  RooflineModelDatabase roofline_model_db =
      InitializeRooflineModelDatabaseFromOpStats(op_stats,
                                                 include_infeed_outfeed);

  AddRooflineModelRecordForProfileDuration(op_stats, roofline_model_db,
                                           include_infeed_outfeed);
  AddRooflineModelRecordsForCompleteSteps(op_stats, roofline_model_db,
                                          include_infeed_outfeed);
  AddRooflineModelRecordsPerStep(op_stats, roofline_model_db,
                                 include_infeed_outfeed);
  PopulateStepDiagnostics(op_stats, roofline_model_db.mutable_diagnostics());
  return roofline_model_db;
}

// Helper function to format source info
std::string SourceInfoFormattedText(
    const tensorflow::profiler::SourceInfo& source_info) {
  if (source_info.file_name().empty() || source_info.line_number() == -1)
    return "";
  // `title` attribute is used to show the full stack trace in the tooltip.
  // We assume that the interpolated strings do not contain any HTML tags. In
  // other words, we assume that they don't need to be escaped.
  return absl::StrCat("<div class='source-info-cell' title='",
                      source_info.stack_frame(), "'>", source_info.file_name(),
                      ":", source_info.line_number(), "</div>");
}

// Helper function to get step string
std::string GetStepString(RecordType record_type, int64_t step_num) {
  switch (record_type) {
    case RecordType::INVALID_RECORD_TYPE:
      return "Invalid";
    case RecordType::ALL:
      return "Total";
    case RecordType::ALL_HW:
      return "Total (HW)";
    case RecordType::AVERAGE_STEP:
      return "Average";
    case RecordType::PER_STEP:
      return absl::StrCat(step_num);
    default:
      return "Unknown";
  }
}

// Function to get roofline model table for GPU
std::unique_ptr<DataTable> GetRooflineModelDataTableForGpu(
    const RooflineModelDatabase& roofline_model_db) {
  std::vector<std::vector<std::string>> kColumns = {
      {"step", "string", "Step"},
      {"rank", "number", "Rank"},
      {"category", "string", "Category"},
      {"operation", "string", "Operation"},
      {"occurrences", "number", "# Occurrences"},
      {"total_time", "number", "Total Time (us)"},
      {"avg_time", "number", "Avg. time (us)"},
      {"total_self_time", "number", "Total self time (us)"},
      {"avg_self_time", "number", "Avg. self time (us)"},
      {"total_self_time_percent", "number", "Total self time (%)"},
      {
          "cumulative_total_self_time_percent",
          "number",
          "Cumulative total self time (%)",
      },
      {"measured_flop_rate", "number", "Normalized FLOP Rate (GFLOP/s)"},
      {"model_flop_rate", "number", "Model FLOP Rate (GFLOP/s)"},
      {"measured_memory_bw", "number", "Memory BW (GiB/s)"},
      {"hbm_bw", "number", "HBM BW (GiB/s)"},
      // For nvidia gpu, currently no vmem_read_bw field, and
      // vmem_write_bw is used for SHM/L1.
      {"vmem_write_bw", "number", "SHM/L1 BW (GiB/s)"},
      {"operational_intensity", "number", "Operational Intensity (FLOP/Byte)"},
      {
          "hbm_operational_intensity",
          "number",
          "HBM Operational Intensity (FLOP/Byte)",
      },
      // for nvidia gpu, currently novmem_read_operational_intensity field,
      // and vmem_write_operational_intensity used for SHM/L1.
      {
          "vmem_write_operational_intensity",
          "number",
          "SHM/L1 Operational Intensity (FLOP/Byte)",
      },
      {
          "bottleneck_operational_intensity",
          "number",
          "Bottleneck Operational Intensity (FLOP/Byte)",
      },
      {"bound_by", "string", "Bound by"},
      {"total_time_per_core", "number", "Total Time per core (us)"},
      {"total_time_in_percentage", "number", "Total Time (%)"},
      {"optimal_flop_rate", "number", "Optimal FLOP Rate (GFLOP/s)"},
      {"roofline_efficiency", "number", "Roofline efficiency (%)"},
      {"compute_efficiency", "number", "FLOP Rate / Peak (%)"},
      {
          "max_mem_bw_utilization",
          "number",
          "Max memory BW utilization (among supported memories) (%)",
      },
      {"include_infeed_outfeed", "boolean", "Include Infeed/Outfeed"},
      {"hlo_module_id", "string", "Program ID"},
      {"source_info", "string", "Source Info"},
  };

  auto data_table = std::make_unique<DataTable>();
  for (const std::vector<std::string>& col : kColumns) {
    data_table->AddColumn(TableColumn(col[0], col[1], col[2]));
  }

  for (const RooflineModelRecord& record :
       roofline_model_db.roofline_model_record()) {
    TableRow* row = data_table->AddRow();
    row->AddTextCell(GetStepString(record.record_type(), record.step_num()));
    row->AddNumberCell(record.rank());
    row->AddTextCell(record.hlo_category());
    row->AddTextCell(record.hlo_name());
    row->AddNumberCell(record.occurrences());
    row->AddNumberCell(record.total_time_in_us());
    row->AddNumberCell(record.avg_time_in_us());
    row->AddNumberCell(record.total_self_time_in_us());
    row->AddNumberCell(record.avg_self_time_in_us());
    row->AddNumberCell(record.total_self_time_as_fraction());
    row->AddNumberCell(record.cumulative_total_self_time_as_fraction());
    row->AddNumberCell(record.measured_flop_rate());
    row->AddNumberCell(record.model_flop_rate());
    row->AddNumberCell(record.measured_memory_bw());
    row->AddNumberCell(record.hbm_bw());
    row->AddNumberCell(record.vmem_write_bw());
    row->AddNumberCell(record.operational_intensity());
    row->AddNumberCell(record.hbm_operational_intensity());
    row->AddNumberCell(record.vmem_write_operational_intensity());
    row->AddNumberCell(record.bottleneck_operational_intensity());
    row->AddTextCell(record.bound_by());
    row->AddNumberCell(record.total_time_per_core_in_us());
    row->AddNumberCell(record.total_time_in_percentage());
    row->AddNumberCell(record.optimal_flop_rate());
    row->AddNumberCell(record.roofline_efficiency());
    row->AddNumberCell(record.flop_rate_relative_to_hw_limit());
    row->AddNumberCell(record.memory_bw_relative_to_hw_limit());
    row->AddBooleanCell(record.include_infeed_outfeed());
    row->AddTextCell(absl::StrCat(record.hlo_module_id()));
    row->AddTextCell(SourceInfoFormattedText(record.source_info()));
  }

  std::vector<std::vector<std::string>> kCustomProperties = {
      {"device_type", roofline_model_db.device_type()},
      {"peak_flop_rate", absl::StrCat(roofline_model_db.peak_flop_rate())},
      {"peak_hbm_bw", absl::StrCat(roofline_model_db.peak_hbm_bw())},
      {"peak_vmem_write_bw",
       absl::StrCat(roofline_model_db.peak_vmem_write_bw())},
      {"hbm_ridge_point",
       absl::StrCat(RidgePoint(roofline_model_db.peak_flop_rate(),
                               roofline_model_db.peak_hbm_bw()))},
      {"vmem_write_ridge_point",
       absl::StrCat(RidgePoint(roofline_model_db.peak_flop_rate(),
                               roofline_model_db.peak_vmem_write_bw()))}};

  for (const std::vector<std::string>& property : kCustomProperties) {
    data_table->AddCustomProperty(property[0], property[1]);
  }

  return data_table;
}

// Function to get roofline model table
std::unique_ptr<DataTable> GetRooflineModelDataTable(
    const RooflineModelDatabase& roofline_model_db) {
  std::vector<std::vector<std::string>> kColumns = {
      {"step", "string", "Step"},
      {"rank", "number", "Rank"},
      {"category", "string", "Category"},
      {"operation", "string", "Operation"},
      {"occurrences", "number", "# Occurrences"},
      {"total_time", "number", "Total Time (us)"},
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
      {"measured_flop_rate", "number", "Normalized FLOP Rate (GFLOP/s)"},
      {"model_flop_rate", "number", "Model FLOP Rate (GFLOP/s)"},
      {"measured_memory_bw", "number", "Memory BW (GiB/s)"},
      {"hbm_bw", "number", "HBM BW (GiB/s)"},
      {"cmem_read_bw", "number", "CMEM Read BW (GiB/s)"},
      {"cmem_write_bw", "number", "CMEM Write BW (GiB/s)"},
      {"vmem_read_bw", "number", "VMEM Read BW (GiB/s)"},
      {"vmem_write_bw", "number", "VMEM Write BW (GiB/s)"},
      {"operational_intensity", "number", "Operational Intensity (FLOP/Byte)"},
      {
          "hbm_operational_intensity",
          "number",
          "HBM Operational Intensity (FLOP/Byte)",
      },
      {
          "cmem_read_operational_intensity",
          "number",
          "CMEM Read Operational Intensity (FLOP/Byte)",
      },
      {
          "cmem_write_operational_intensity",
          "number",
          "CMEM Write Operational Intensity (FLOP/Byte)",
      },
      {
          "vmem_read_operational_intensity",
          "number",
          "VMEM Read Operational Intensity (FLOP/Byte)",
      },
      {
          "vmem_write_operational_intensity",
          "number",
          "VMEM Write Operational Intensity (FLOP/Byte)",
      },
      {
          "bottleneck_operational_intensity",
          "number",
          "Bottleneck Operational Intensity (FLOP/Byte)",
      },
      {"bound_by", "string", "Bound by"},
      {"total_time_per_core", "number", "Total Time per core (us)"},
      {"total_time_in_percentage", "number", "Total Time (%)"},
      {"optimal_flop_rate", "number", "Optimal FLOP Rate (GFLOP/s)"},
      {"roofline_efficiency", "number", "Roofline efficiency (%)"},
      {"compute_efficiency", "number", "FLOP Rate / Peak (%)"},
      {
          "max_mem_bw_utilization",
          "number",
          "Max memory BW utilization (among supported memories) (%)",
      },
      {"include_infeed_outfeed", "boolean", "Include Infeed/Outfeed"},
      {"hlo_module_id", "string", "Program ID"},
      {"source_info", "string", "Source Info"},
  };

  auto data_table = std::make_unique<DataTable>();
  for (const std::vector<std::string>& col : kColumns) {
    data_table->AddColumn(TableColumn(col[0], col[1], col[2]));
  }

  for (const RooflineModelRecord& record :
       roofline_model_db.roofline_model_record()) {
    TableRow* row = data_table->AddRow();
    row->AddTextCell(GetStepString(record.record_type(), record.step_num()));
    row->AddNumberCell(record.rank());
    row->AddTextCell(record.hlo_category());
    row->AddTextCell(record.hlo_name());
    row->AddNumberCell(record.occurrences());
    row->AddNumberCell(record.total_time_in_us());
    row->AddNumberCell(record.avg_time_in_us());
    row->AddNumberCell(record.total_self_time_in_us());
    row->AddNumberCell(record.avg_self_time_in_us());
    row->AddNumberCell(record.total_self_time_as_fraction());
    row->AddNumberCell(record.cumulative_total_self_time_as_fraction());
    row->AddNumberCell(record.dma_stall_fraction());
    row->AddNumberCell(record.measured_flop_rate());
    row->AddNumberCell(record.model_flop_rate());
    row->AddNumberCell(record.measured_memory_bw());
    row->AddNumberCell(record.hbm_bw());
    row->AddNumberCell(record.cmem_read_bw());
    row->AddNumberCell(record.cmem_write_bw());
    row->AddNumberCell(record.vmem_read_bw());
    row->AddNumberCell(record.vmem_write_bw());
    row->AddNumberCell(record.operational_intensity());
    row->AddNumberCell(record.hbm_operational_intensity());
    row->AddNumberCell(record.cmem_read_operational_intensity());
    row->AddNumberCell(record.cmem_write_operational_intensity());
    row->AddNumberCell(record.vmem_read_operational_intensity());
    row->AddNumberCell(record.vmem_write_operational_intensity());
    row->AddNumberCell(record.bottleneck_operational_intensity());
    row->AddTextCell(record.bound_by());
    row->AddNumberCell(record.total_time_per_core_in_us());
    row->AddNumberCell(record.total_time_in_percentage());
    row->AddNumberCell(record.optimal_flop_rate());
    row->AddNumberCell(record.roofline_efficiency());
    row->AddNumberCell(record.flop_rate_relative_to_hw_limit());
    row->AddNumberCell(record.memory_bw_relative_to_hw_limit());
    row->AddBooleanCell(record.include_infeed_outfeed());
    row->AddTextCell(absl::StrCat(record.hlo_module_id()));
    row->AddTextCell(SourceInfoFormattedText(record.source_info()));
  }

  std::vector<std::vector<std::string>> kCustomProperties = {
      {"device_type", roofline_model_db.device_type()},
      {"megacore",
       absl::StrCat(static_cast<int>(roofline_model_db.megacore()))},
      {"has_cmem",
       absl::StrCat(static_cast<int>(roofline_model_db.has_cmem()))},
      {"has_merged_vmem",
       absl::StrCat(static_cast<int>(roofline_model_db.has_merged_vmem()))},
      {"peak_flop_rate", absl::StrCat(roofline_model_db.peak_flop_rate())},
      {"peak_hbm_bw", absl::StrCat(roofline_model_db.peak_hbm_bw())},
      {"peak_cmem_read_bw",
       absl::StrCat(roofline_model_db.peak_cmem_read_bw())},
      {"peak_cmem_write_bw",
       absl::StrCat(roofline_model_db.peak_cmem_write_bw())},
      {"peak_vmem_read_bw",
       absl::StrCat(roofline_model_db.peak_vmem_read_bw())},
      {"peak_vmem_write_bw",
       absl::StrCat(roofline_model_db.peak_vmem_write_bw())},
      {"hbm_ridge_point",
       absl::StrCat(RidgePoint(roofline_model_db.peak_flop_rate(),
                               roofline_model_db.peak_hbm_bw()))},
      {"cmem_read_ridge_point",
       absl::StrCat(RidgePoint(roofline_model_db.peak_flop_rate(),
                               roofline_model_db.peak_cmem_read_bw()))},
      {"cmem_write_ridge_point",
       absl::StrCat(RidgePoint(roofline_model_db.peak_flop_rate(),
                               roofline_model_db.peak_cmem_write_bw()))},
      {"vmem_read_ridge_point",
       absl::StrCat(RidgePoint(roofline_model_db.peak_flop_rate(),
                               roofline_model_db.peak_vmem_read_bw()))},
      {"vmem_write_ridge_point",
       absl::StrCat(RidgePoint(roofline_model_db.peak_flop_rate(),
                               roofline_model_db.peak_vmem_write_bw()))},
  };

  for (const std::vector<std::string>& property : kCustomProperties) {
    data_table->AddCustomProperty(property[0], property[1]);
  }

  return data_table;
}

// Function to generate roofline model table
std::unique_ptr<DataTable> GenerateRooflineModelDataTable(
    const RooflineModelDatabase& roofline_model_db) {
  std::unique_ptr<DataTable> data_table = nullptr;
  if (absl::StrContains(roofline_model_db.device_type(), "GPU")) {
    data_table = GetRooflineModelDataTableForGpu(roofline_model_db);
  } else {
    data_table = GetRooflineModelDataTable(roofline_model_db);
  }
  return data_table;
}

std::unique_ptr<DataTable> GenerateDiagnosticsDataTable(
    const RooflineModelDatabase& roofline_model_db) {
  std::vector<std::vector<std::string>> kColumns = {
      {"severity", "string", "Severity"}, {"message", "string", "Message"}};
  auto data_table = std::make_unique<DataTable>();
  for (const std::vector<std::string>& col : kColumns) {
    data_table->AddColumn(TableColumn(col[0], col[1], col[2]));
  }
  for (const auto& info : roofline_model_db.diagnostics().info()) {
    TableRow* row = data_table->AddRow();
    row->AddTextCell("INFO");
    row->AddTextCell(info);
  }
  for (const auto& warning : roofline_model_db.diagnostics().warnings()) {
    TableRow* row = data_table->AddRow();
    row->AddTextCell("WARNING");
    row->AddTextCell(warning);
  }
  for (const auto& error : roofline_model_db.diagnostics().errors()) {
    TableRow* row = data_table->AddRow();
    row->AddTextCell("ERROR");
    row->AddTextCell(error);
  }
  return data_table;
}

std::string RooflineModelToDataTableJson(
    const RooflineModelDatabase& roofline_model_db) {
  std::string roofline_json =
      GenerateRooflineModelDataTable(roofline_model_db)->ToJson();
  std::string diagnostics_json =
      GenerateDiagnosticsDataTable(roofline_model_db)->ToJson();
  return absl::StrCat("[", roofline_json, ",", diagnostics_json, "]");
}

}  // namespace profiler
}  // namespace tensorflow
