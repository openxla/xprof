/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include <cstdint>
#include <memory>
#include <string>

#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "xla/tsl/profiler/utils/math_utils.h"
#include "xprof/convert/data_table_utils.h"
#include "xprof/convert/op_stats_to_roofline_model.h"
#include "plugin/xprof/protobuf/hardware_types.pb.h"
#include "plugin/xprof/protobuf/op_metrics.pb.h"
#include "plugin/xprof/protobuf/op_stats.pb.h"
#include "plugin/xprof/protobuf/roofline_model.pb.h"

namespace tensorflow {
namespace profiler {
namespace {

using ::tensorflow::profiler::roofline_model::RecordType;
using ::tensorflow::profiler::roofline_model::RooflineModelDatabase;
using ::tensorflow::profiler::roofline_model::ScRooflineModelRecord;

TEST(OpStatsToRooflineModelSparseCoreTest, SparseCoreHbmBoundRecord) {
  RooflineModelDatabase db;
  db.set_num_sparse_core_tiles(64);
  db.set_peak_sc_flop_rate(1525.0);
  db.set_peak_hbm_bw(1287.45);
  db.set_peak_sc_hbm_bw(643.725);

  OpMetrics metrics;
  metrics.set_name("embedding_lookup_hbm");
  metrics.set_category("sparse_core");
  metrics.set_core_type(OpMetrics_TpuCoreType_SPARSE_CORE);
  metrics.set_time_ps(10000000);       // 10 us = 10,000 ns
  metrics.set_self_time_ps(10000000);  // 10 us
  metrics.set_flops_v2(100000);        // 0.01 GFLOP/s
  metrics.set_bytes_accessed(5000000);
  auto* mem = metrics.add_memory_accessed_breakdown();
  mem->set_memory_space(PerformanceInfo::MemoryAccessed::HBM);
  mem->set_bytes_accessed(5000000);  // 5 MB in 10 us -> ~465.66 GiB/s

  OpStats op_stats;
  ScRooflineModelRecord record = ConvertOpMetricsToScRooflineModelRecord(
      op_stats, metrics, RecordType::ALL, /*step_num=*/0,
      /*total_time_ps=*/10000000, db, RooflineModelOptions{});

  EXPECT_EQ(record.hlo_name(), "embedding_lookup_hbm");
  EXPECT_EQ(record.sc_bound_by(), "HBM");
  const double expected_hbm_bw =
      ::tsl::profiler::GibibytesPerSecond(5000000, 10000.0);
  EXPECT_DOUBLE_EQ(record.hbm_bw(), expected_hbm_bw);
  EXPECT_NEAR(record.sc_roofline_efficiency(), expected_hbm_bw / 643.725, 1e-6);
  EXPECT_DOUBLE_EQ(record.bottleneck_operational_intensity(),
                   record.hbm_operational_intensity());
}

TEST(OpStatsToRooflineModelSparseCoreTest,
     SparseCoreVmemAndUnknownAreNotAttributedToSpmemOrHbm) {
  RooflineModelDatabase db;
  db.set_peak_sc_flop_rate(1525.0);
  db.set_peak_sc_hbm_bw(643.725);
  db.set_peak_spmem_read_bw(607.9);
  db.set_peak_spmem_write_bw(607.9);

  OpMetrics metrics;
  metrics.set_name("mixed_memory_spaces");
  metrics.set_core_type(OpMetrics_TpuCoreType_SPARSE_CORE);
  metrics.set_time_ps(1000000);  // 1 us = 1000 ns
  metrics.set_bytes_accessed(7000);

  auto add_access = [&](PerformanceInfo::MemoryAccessed::MemorySpace space,
                        OpMetrics::MemoryAccessed::OperationType op_type,
                        int64_t bytes) {
    auto* mem = metrics.add_memory_accessed_breakdown();
    mem->set_memory_space(space);
    mem->set_operation_type(op_type);
    mem->set_bytes_accessed(bytes);
  };
  add_access(PerformanceInfo::MemoryAccessed::HBM,
             OpMetrics::MemoryAccessed::READ, 1000);
  add_access(PerformanceInfo::MemoryAccessed::VMEM,
             OpMetrics::MemoryAccessed::READ, 2000);
  add_access(PerformanceInfo::MemoryAccessed::VMEM,
             OpMetrics::MemoryAccessed::WRITE, 3000);
  add_access(PerformanceInfo::MemoryAccessed::UNKNOWN,
             OpMetrics::MemoryAccessed::READ, 500);
  add_access(PerformanceInfo::MemoryAccessed::CMEM,
             OpMetrics::MemoryAccessed::WRITE, 500);

  OpStats op_stats;
  ScRooflineModelRecord record = ConvertOpMetricsToScRooflineModelRecord(
      op_stats, metrics, RecordType::ALL, /*step_num=*/0,
      /*total_time_ps=*/1000000, db, RooflineModelOptions{});

  EXPECT_DOUBLE_EQ(record.hbm_bw(),
                   ::tsl::profiler::GibibytesPerSecond(1000, 1000.0));
  EXPECT_DOUBLE_EQ(record.spmem_read_bw(), 0.0);
  EXPECT_DOUBLE_EQ(record.spmem_write_bw(), 0.0);
}

TEST(OpStatsToRooflineModelSparseCoreTest,
     SparseCoreWithoutBreakdownTreatsAllBytesAsHbm) {
  RooflineModelDatabase db;
  db.set_peak_sc_flop_rate(1525.0);
  db.set_peak_sc_hbm_bw(643.725);

  OpMetrics metrics;
  metrics.set_name("legacy_op");
  metrics.set_core_type(OpMetrics_TpuCoreType_SPARSE_CORE);
  metrics.set_time_ps(1000000);  // 1 us
  metrics.set_bytes_accessed(4000);

  OpStats op_stats;
  ScRooflineModelRecord record = ConvertOpMetricsToScRooflineModelRecord(
      op_stats, metrics, RecordType::ALL, /*step_num=*/0,
      /*total_time_ps=*/1000000, db, RooflineModelOptions{});

  EXPECT_DOUBLE_EQ(record.hbm_bw(),
                   ::tsl::profiler::GibibytesPerSecond(4000, 1000.0));
  EXPECT_DOUBLE_EQ(record.spmem_read_bw(), 0.0);
  EXPECT_DOUBLE_EQ(record.spmem_write_bw(), 0.0);
}

TEST(OpStatsToRooflineModelSparseCoreTest, SparseCoreZeroUtilizationIsUnknown) {
  RooflineModelDatabase db;
  db.set_peak_sc_flop_rate(1525.0);
  db.set_peak_sc_hbm_bw(643.725);

  OpMetrics metrics;
  metrics.set_name("idle_op");
  metrics.set_core_type(OpMetrics_TpuCoreType_SPARSE_CORE);
  metrics.set_time_ps(1000000);

  OpStats op_stats;
  ScRooflineModelRecord record = ConvertOpMetricsToScRooflineModelRecord(
      op_stats, metrics, RecordType::ALL, /*step_num=*/0,
      /*total_time_ps=*/1000000, db, RooflineModelOptions{});

  EXPECT_EQ(record.sc_bound_by(), "Unknown");
  EXPECT_DOUBLE_EQ(record.sc_roofline_efficiency(), 0.0);
}

TEST(OpStatsToRooflineModelSparseCoreTest,
     SparseCoreNoPeakFlopRateStillClassifiesHbm) {
  RooflineModelDatabase db;
  db.set_peak_sc_hbm_bw(643.725);

  OpMetrics metrics;
  metrics.set_name("hbm_only_peaks");
  metrics.set_core_type(OpMetrics_TpuCoreType_SPARSE_CORE);
  metrics.set_time_ps(1000000);  // 1 us = 1000 ns
  metrics.set_bytes_accessed(100000);

  OpStats op_stats;
  ScRooflineModelRecord record = ConvertOpMetricsToScRooflineModelRecord(
      op_stats, metrics, RecordType::ALL, /*step_num=*/0,
      /*total_time_ps=*/1000000, db, RooflineModelOptions{});

  EXPECT_EQ(record.sc_bound_by(), "HBM");
  EXPECT_GT(record.sc_roofline_efficiency(), 0.0);
}

TEST(OpStatsToRooflineModelSparseCoreTest, InvalidWrappedFlopsAreIgnored) {
  RooflineModelDatabase db;
  db.set_peak_sc_flop_rate(1525.0);
  db.set_peak_sc_hbm_bw(643.725);

  OpMetrics metrics;
  metrics.set_name("custom_fusion_unknown_cost");
  metrics.set_core_type(OpMetrics_TpuCoreType_SPARSE_CORE);
  metrics.set_occurrences(2);
  metrics.set_time_ps(1000000);
  // Unknown (-1) FLOPs wrapped to 2^64 per occurrence.
  metrics.set_flops_v2(2.0 * 18446744073709551616.0);
  metrics.set_model_flops_v2(2.0 * 18446744073709551616.0);
  metrics.set_bytes_accessed(1000);

  OpStats op_stats;
  ScRooflineModelRecord record = ConvertOpMetricsToScRooflineModelRecord(
      op_stats, metrics, RecordType::ALL, /*step_num=*/0,
      /*total_time_ps=*/1000000, db, RooflineModelOptions{});

  EXPECT_DOUBLE_EQ(record.measured_flop_rate(), 0.0);
  EXPECT_DOUBLE_EQ(record.model_flop_rate(), 0.0);
  EXPECT_DOUBLE_EQ(record.operational_intensity(), 0.0);
  EXPECT_LE(record.sc_roofline_efficiency(), 1.0);
}

TEST(OpStatsToRooflineModelSparseCoreTest, SparseCoreTecComputeBoundRecord) {
  RooflineModelDatabase db;
  db.set_num_sparse_core_tiles(64);
  db.set_peak_sc_flop_rate(1525.0);  // 1525 GFLOP/s
  db.set_peak_sc_hbm_bw(643.725);

  OpMetrics metrics;
  metrics.set_name("tec_compute_heavy_op");
  metrics.set_category("sparse_core");
  metrics.set_core_type(OpMetrics_TpuCoreType_SPARSE_CORE);
  metrics.set_time_ps(1000000);      // 1 us = 1000 ns
  metrics.set_flops_v2(762500.0);    // 762.5 GFLOP/s measured (50% of peak)
  metrics.set_bytes_accessed(1000);  // ~0.93 GiB/s

  OpStats op_stats;
  ScRooflineModelRecord record = ConvertOpMetricsToScRooflineModelRecord(
      op_stats, metrics, RecordType::ALL, /*step_num=*/0,
      /*total_time_ps=*/1000000, db, RooflineModelOptions{});

  EXPECT_EQ(record.hlo_name(), "tec_compute_heavy_op");
  EXPECT_EQ(record.sc_bound_by(), "TEC Compute");
  EXPECT_DOUBLE_EQ(record.measured_flop_rate(), 762.5);
  EXPECT_NEAR(record.sc_roofline_efficiency(), 0.5, 1e-6);
}

TEST(OpStatsToRooflineModelSparseCoreTest, ScNegativeFlopsSanitized) {
  RooflineModelDatabase db;
  db.set_num_sparse_core_tiles(64);
  db.set_peak_sc_flop_rate(1525.0);
  db.set_peak_sc_hbm_bw(643.725);

  OpMetrics metrics;
  metrics.set_name("sc_op_negative_flops");
  metrics.set_category("sparse_core");
  metrics.set_core_type(OpMetrics_TpuCoreType_SPARSE_CORE);
  metrics.set_time_ps(1000000);  // 1 us
  metrics.set_flops(-1);
  metrics.set_flops_v2(-1.0);
  metrics.set_model_flops_v2(-1.0);
  metrics.set_bytes_accessed(1668);

  OpStats op_stats;
  ScRooflineModelRecord record = ConvertOpMetricsToScRooflineModelRecord(
      op_stats, metrics, RecordType::ALL, /*step_num=*/0,
      /*total_time_ps=*/1000000, db, RooflineModelOptions{});

  EXPECT_DOUBLE_EQ(record.measured_flop_rate(), 0.0);
  EXPECT_DOUBLE_EQ(record.model_flop_rate(), 0.0);
  EXPECT_DOUBLE_EQ(record.operational_intensity(), 0.0);
}

TEST(OpStatsToRooflineModelSparseCoreTest, DataTableScRowsAndCustomProperties) {
  RooflineModelDatabase db;
  db.set_peak_flop_rate(1000.0);
  db.set_peak_hbm_bw(1287.45);
  db.set_num_sparse_core_tiles(64);
  db.set_peak_sc_flop_rate(1525.0);
  db.set_peak_sc_hbm_bw(643.725);
  db.set_peak_spmem_read_bw(607.9);
  db.set_peak_spmem_write_bw(607.9);

  auto* tc_rec = db.add_roofline_model_record();
  tc_rec->set_record_type(RecordType::ALL);
  tc_rec->set_rank(1);
  tc_rec->set_hlo_name("tc_matmul");
  tc_rec->set_bound_by("HBM");
  tc_rec->set_roofline_efficiency(0.42);
  tc_rec->set_vmem_read_bw(123.0);

  auto* sc_rec = db.add_sc_roofline_model_record();
  sc_rec->set_record_type(RecordType::ALL);
  sc_rec->set_rank(1);
  sc_rec->set_hlo_name("sc_gather");
  sc_rec->set_sc_bound_by("TEC Compute");
  sc_rec->set_sc_roofline_efficiency(0.75);
  sc_rec->set_hbm_bw(321.0);

  std::unique_ptr<DataTable> data_table = GenerateRooflineModelDataTable(db);
  ASSERT_NE(data_table, nullptr);
  ASSERT_EQ(data_table->GetRows().size(), 2);

  absl::flat_hash_map<std::string, int> col_idx;
  const auto& cols = data_table->GetColumns();
  for (int i = 0; i < cols.size(); ++i) {
    col_idx[cols[i].id] = i;
  }
  ASSERT_TRUE(col_idx.contains("is_sparse_core"));
  ASSERT_TRUE(col_idx.contains("bound_by"));
  ASSERT_TRUE(col_idx.contains("roofline_efficiency"));

  const auto& tc_cells = data_table->GetRows()[0]->GetCells();
  const auto& sc_cells = data_table->GetRows()[1]->GetCells();
  EXPECT_EQ(tc_cells.size(), cols.size());
  EXPECT_EQ(sc_cells.size(), cols.size());

  EXPECT_EQ(tc_cells[col_idx["is_sparse_core"]]->GetCellValueStr(), "FALSE");
  EXPECT_EQ(tc_cells[col_idx["bound_by"]]->GetCellValueStr(), "HBM");
  EXPECT_DOUBLE_EQ(
      std::stod(tc_cells[col_idx["roofline_efficiency"]]->GetCellValueStr()),
      0.42);
  EXPECT_DOUBLE_EQ(
      std::stod(tc_cells[col_idx["vmem_read_bw"]]->GetCellValueStr()), 123.0);

  EXPECT_EQ(sc_cells[col_idx["is_sparse_core"]]->GetCellValueStr(), "TRUE");
  EXPECT_EQ(sc_cells[col_idx["bound_by"]]->GetCellValueStr(), "TEC Compute");
  EXPECT_DOUBLE_EQ(
      std::stod(sc_cells[col_idx["roofline_efficiency"]]->GetCellValueStr()),
      0.75);
  EXPECT_DOUBLE_EQ(std::stod(sc_cells[col_idx["hbm_bw"]]->GetCellValueStr()),
                   321.0);

  const auto& props = data_table->GetCustomProperties();
  EXPECT_EQ(props.at("num_sparse_core_tiles"), "64");
  EXPECT_EQ(props.at("peak_sc_flop_rate"), "1525");
  EXPECT_EQ(props.at("peak_sc_hbm_bw"), "643.725");
  EXPECT_NEAR(std::stod(props.at("sc_hbm_ridge_point")),
              1525.0 / ::tsl::profiler::GibiToGiga(643.725), 1e-5);

  // Non-SC database must omit SC custom properties.
  RooflineModelDatabase non_sc_db;
  non_sc_db.set_peak_flop_rate(1000.0);
  std::unique_ptr<DataTable> non_sc_table =
      GenerateRooflineModelDataTable(non_sc_db);
  EXPECT_FALSE(
      non_sc_table->GetCustomProperties().contains("num_sparse_core_tiles"));
  EXPECT_FALSE(non_sc_table->GetCustomProperties().contains("peak_sc_hbm_bw"));
}

TEST(OpStatsToRooflineModelSparseCoreTest,
     EndToEndInitializeRooflineModelDatabaseFromOpStats) {
  OpStats op_stats;
  op_stats.mutable_run_environment()->set_hardware_type(
      tensorflow::profiler::HardwareType::TPU);
  op_stats.mutable_run_environment()->set_device_type("TPU v5");
  auto* perf_env = op_stats.mutable_perf_env();
  perf_env->set_peak_tera_flops_per_second(48.0);
  perf_env->set_peak_hbm_bw_giga_bytes_per_second(600.0);
  perf_env->set_has_megacore(true);
  // Chip-level SC peaks: 4 SCs, 2 TCs per chip.
  perf_env->set_peak_sc_tera_flops_per_second(6.1);
  perf_env->set_num_sparse_core_tiles(64);
  while (perf_env->peak_bws_giga_bytes_per_second_size() <= 8) {
    perf_env->add_peak_bws_giga_bytes_per_second(0.0);
  }
  perf_env->set_peak_bws_giga_bytes_per_second(0, 600.0);
  perf_env->set_peak_bws_giga_bytes_per_second(7, 2610.91);
  perf_env->set_peak_bws_giga_bytes_per_second(8, 2610.91);

  auto* core_details = op_stats.mutable_core_id_to_details();
  (*core_details)[0].set_is_sparse_core(false);
  (*core_details)[1].set_is_sparse_core(false);
  for (int i = 0; i < 4; ++i) {
    (*core_details)[100 + i].set_is_sparse_core(true);
  }

  RooflineModelDatabase db =
      InitializeRooflineModelDatabaseFromOpStats(op_stats);

  EXPECT_EQ(db.num_sparse_core_tiles(), 64);
  EXPECT_DOUBLE_EQ(db.peak_sc_flop_rate(), 1525.0);
  EXPECT_NEAR(db.peak_spmem_read_bw(), 607.9, 0.05);
  EXPECT_NEAR(db.peak_spmem_write_bw(), 607.9, 0.05);
  // 2 TCs / 4 SCs = 0.5 * peak_hbm_bw.
  EXPECT_NEAR(db.peak_sc_hbm_bw(), 0.5 * db.peak_hbm_bw(), 0.001);
}

TEST(OpStatsToRooflineModelSparseCoreTest,
     ScChipWithoutSpmemStillPopulatesScPeaks) {
  OpStats op_stats;
  op_stats.mutable_run_environment()->set_hardware_type(
      tensorflow::profiler::HardwareType::TPU);
  auto* perf_env = op_stats.mutable_perf_env();
  perf_env->set_peak_hbm_bw_giga_bytes_per_second(600.0);
  perf_env->set_peak_sc_tera_flops_per_second(10.0);

  RooflineModelDatabase db =
      InitializeRooflineModelDatabaseFromOpStats(op_stats);

  EXPECT_DOUBLE_EQ(db.peak_sc_flop_rate(), 10000.0);
  EXPECT_FALSE(db.has_num_sparse_core_tiles());
  EXPECT_DOUBLE_EQ(db.peak_sc_hbm_bw(), db.peak_hbm_bw());
}

TEST(OpStatsToRooflineModelSparseCoreTest,
     ProgramRecordFlatWhileAndInvalidFlops) {
  OpStats op_stats;
  op_stats.mutable_run_environment()->set_hardware_type(
      tensorflow::profiler::HardwareType::TPU);
  auto* perf_env = op_stats.mutable_perf_env();
  perf_env->set_peak_sc_tera_flops_per_second(1.525);
  perf_env->set_peak_hbm_bw_giga_bytes_per_second(600.0);

  auto* db = op_stats.mutable_device_op_metrics_db();
  db->set_total_time_ps(1000000);
  auto* tc_op = db->add_metrics_db();
  tc_op->set_name("offload");
  tc_op->set_core_type(OpMetrics_TpuCoreType_TENSOR_CORE);
  tc_op->set_occurrences(1);
  tc_op->set_time_ps(1000000);

  // Flat SC DB: a `while` wrapper and its body fusion are siblings.
  auto* sc_while = tc_op->mutable_children()->add_metrics_db();
  sc_while->set_name("sc_while");
  sc_while->set_category("while");
  sc_while->set_core_type(OpMetrics_TpuCoreType_SPARSE_CORE);
  sc_while->set_occurrences(1);
  sc_while->set_time_ps(1000000);
  sc_while->set_self_time_ps(100000);
  sc_while->set_flops_v2(1000);

  auto* sc_body = tc_op->mutable_children()->add_metrics_db();
  sc_body->set_name("sc_body");
  sc_body->set_category("custom fusion");
  sc_body->set_core_type(OpMetrics_TpuCoreType_SPARSE_CORE);
  sc_body->set_occurrences(1);
  sc_body->set_time_ps(900000);
  sc_body->set_self_time_ps(900000);
  sc_body->set_flops_v2(900000);

  // Op with invalid wrapped FLOPs: still contributes self_time_ps.
  auto* sc_invalid = tc_op->mutable_children()->add_metrics_db();
  sc_invalid->set_name("sc_invalid");
  sc_invalid->set_category("custom fusion");
  sc_invalid->set_core_type(OpMetrics_TpuCoreType_SPARSE_CORE);
  sc_invalid->set_occurrences(1);
  sc_invalid->set_time_ps(100000);
  sc_invalid->set_self_time_ps(100000);
  sc_invalid->set_flops_v2(18446744073709551616.0);

  RooflineModelDatabase result_db = ConvertOpStatsToRooflineModel(op_stats);

  int program_records = 0;
  for (const auto& record : result_db.sc_roofline_model_record()) {
    if (record.hlo_name() != "Program") continue;
    ++program_records;
    // `while` is excluded from both FLOPs and ScTotalTimePs.
    // Remaining: 900,000 FLOPs / (900 ns + 100 ns) = 900 GFLOP/s.
    EXPECT_DOUBLE_EQ(record.measured_flop_rate(), 900.0);
  }
  EXPECT_EQ(program_records, 1);
}

TEST(OpStatsToRooflineModelSparseCoreTest,
     AsyncWrappersSkippedAndV2StartDoneDeduplicated) {
  OpStats op_stats;
  op_stats.mutable_run_environment()->set_hardware_type(
      tensorflow::profiler::HardwareType::TPU);
  auto* perf_env = op_stats.mutable_perf_env();
  perf_env->set_peak_sc_tera_flops_per_second(1.525);
  perf_env->set_peak_hbm_bw_giga_bytes_per_second(600.0);

  auto* db = op_stats.mutable_device_op_metrics_db();
  db->set_total_time_ps(10000000);

  auto add_tc_wrapper = [&](const std::string& name, const std::string& cat) {
    auto* wrapper = db->add_metrics_db();
    wrapper->set_name(name);
    wrapper->set_category(cat);
    wrapper->set_core_type(OpMetrics_TpuCoreType_TENSOR_CORE);
    wrapper->set_occurrences(1);
    wrapper->set_time_ps(1000000);
    auto* child = wrapper->mutable_children()->add_metrics_db();
    child->set_name("sc_kernel");
    child->set_category("custom fusion");
    child->set_core_type(OpMetrics_TpuCoreType_SPARSE_CORE);
    child->set_hlo_module_id(42);
    child->set_occurrences(1);
    child->set_time_ps(1000000);
    child->set_self_time_ps(1000000);
    child->set_flops_v2(1000000);
  };
  add_tc_wrapper("sc_kernel.call-start", "async-start");
  add_tc_wrapper("sc_kernel.call-done", "async-done");

  RooflineModelDatabase result_db = ConvertOpStatsToRooflineModel(op_stats);

  ASSERT_EQ(result_db.sc_roofline_model_record_size(), 2);
  EXPECT_EQ(result_db.sc_roofline_model_record(0).hlo_name(), "Program");
  EXPECT_DOUBLE_EQ(result_db.sc_roofline_model_record(0).measured_flop_rate(),
                   1000.0);
  EXPECT_EQ(result_db.sc_roofline_model_record(1).hlo_name(), "sc_kernel");
}

TEST(OpStatsToRooflineModelSparseCoreTest,
     TimeScaleMultiplierDoesNotScaleHbmBw) {
  RooflineModelDatabase db;
  db.set_peak_sc_flop_rate(1525.0);
  db.set_peak_sc_hbm_bw(643.725);

  OpMetrics metrics;
  metrics.set_name("sc_dvfs_op");
  metrics.set_core_type(OpMetrics_TpuCoreType_SPARSE_CORE);
  metrics.set_time_ps(1000000);            // 1 us raw
  metrics.set_normalized_time_ps(500000);  // 0.5 us normalized
  metrics.set_flops_v2(500000);            // 500 GFLOP/s raw, 1000 normalized
  metrics.set_bytes_accessed(1000000);     // ~931 GiB/s raw

  OpStats op_stats;
  RooflineModelOptions options;
  options.apply_time_scale_multiplier = true;
  ScRooflineModelRecord record = ConvertOpMetricsToScRooflineModelRecord(
      op_stats, metrics, RecordType::ALL, /*step_num=*/0,
      /*total_time_ps=*/1000000, db, options);

  EXPECT_DOUBLE_EQ(record.measured_flop_rate(), 1000.0);
  EXPECT_DOUBLE_EQ(record.hbm_bw(),
                   ::tsl::profiler::GibibytesPerSecond(1000000, 1000.0));
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
