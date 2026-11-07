#include <memory>

#include <gtest/gtest.h>
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

TEST(OpStatsToRooflineModelSparseCoreTest, SparseCoreDatabaseInitialization) {
  RooflineModelDatabase db;
  db.set_num_sparse_core_tiles(16);
  db.set_peak_sc_flop_rate(48.0);
  db.set_peak_spmem_read_bw(4096.0);
  db.set_peak_spmem_write_bw(2048.0);

  EXPECT_EQ(db.num_sparse_core_tiles(), 16);
  EXPECT_DOUBLE_EQ(db.peak_sc_flop_rate(), 48.0);
  EXPECT_DOUBLE_EQ(db.peak_spmem_read_bw(), 4096.0);
  EXPECT_DOUBLE_EQ(db.peak_spmem_write_bw(), 2048.0);
}

TEST(OpStatsToRooflineModelSparseCoreTest, SparseCoreHbmBoundRecord) {
  RooflineModelDatabase db;
  db.set_num_sparse_core_tiles(16);
  db.set_peak_sc_flop_rate(48.0);
  db.set_peak_hbm_bw(643.7);
  db.set_peak_spmem_read_bw(4096.0);
  db.set_peak_spmem_write_bw(2048.0);

  OpMetrics metrics;
  metrics.set_name("embedding_lookup_hbm");
  metrics.set_category("sparse_core");
  metrics.set_core_type(OpMetrics_TpuCoreType_SPARSE_CORE);
  metrics.set_time_ps(10000000);          // 10 us
  metrics.set_flops_v2(1000);             // Low FLOPS
  metrics.set_bytes_accessed(100000000);  // High HBM bytes

  auto* mem = metrics.add_memory_accessed_breakdown();
  mem->set_memory_space(1 /* HBM */);
  mem->set_bytes_accessed(100000000);

  OpStats op_stats;
  ScRooflineModelRecord record = ConvertOpMetricsToScRooflineModelRecord(
      op_stats, metrics, RecordType::ALL, /*step_num=*/0,
      /*total_time_ps=*/10000000, db, RooflineModelOptions{});

  EXPECT_EQ(record.hlo_name(), "embedding_lookup_hbm");
  EXPECT_EQ(record.sc_bound_by(), "HBM");
  EXPECT_GT(record.sc_roofline_efficiency(), 0.0);
}

TEST(OpStatsToRooflineModelSparseCoreTest, SparseCoreSpmemBoundRecord) {
  RooflineModelDatabase db;
  db.set_num_sparse_core_tiles(16);
  db.set_peak_sc_flop_rate(48.0);
  db.set_peak_hbm_bw(643.7);
  db.set_peak_spmem_read_bw(4096.0);
  db.set_peak_spmem_write_bw(2048.0);

  OpMetrics metrics;
  metrics.set_name("embedding_lookup_spmem");
  metrics.set_category("sparse_core");
  metrics.set_core_type(OpMetrics_TpuCoreType_SPARSE_CORE);
  metrics.set_time_ps(1000000);  // 1 us
  metrics.set_flops_v2(1000);

  auto* mem_read = metrics.add_memory_accessed_breakdown();
  mem_read->set_memory_space(7 /* SPMEM Read */);
  mem_read->set_operation_type(OpMetrics::MemoryAccessed::READ);
  mem_read->set_bytes_accessed(
      10000000);  // 10MB in 1us -> 10,000 GB/s SPMEM Read BW

  OpStats op_stats;
  ScRooflineModelRecord record = ConvertOpMetricsToScRooflineModelRecord(
      op_stats, metrics, RecordType::ALL, /*step_num=*/0,
      /*total_time_ps=*/1000000, db, RooflineModelOptions{});

  EXPECT_EQ(record.hlo_name(), "embedding_lookup_spmem");
  EXPECT_EQ(record.sc_bound_by(), "SPMEM");
  EXPECT_GT(record.spmem_read_bw(), 0.0);
  EXPECT_GT(record.sc_roofline_efficiency(), 0.0);
}

TEST(OpStatsToRooflineModelSparseCoreTest, SparseCoreTecComputeBoundRecord) {
  RooflineModelDatabase db;
  db.set_num_sparse_core_tiles(16);
  db.set_peak_sc_flop_rate(1.0);  // 1 GFLOP/s low peak
  db.set_peak_hbm_bw(1000.0);     // High peak BW
  db.set_peak_spmem_read_bw(10000.0);
  db.set_peak_spmem_write_bw(10000.0);

  OpMetrics metrics;
  metrics.set_name("tec_compute_heavy_op");
  metrics.set_category("sparse_core");
  metrics.set_core_type(OpMetrics_TpuCoreType_SPARSE_CORE);
  metrics.set_time_ps(1000000);    // 1 us
  metrics.set_flops_v2(10000000);  // 10 GFLOPS/s measured

  OpStats op_stats;
  ScRooflineModelRecord record = ConvertOpMetricsToScRooflineModelRecord(
      op_stats, metrics, RecordType::ALL, /*step_num=*/0,
      /*total_time_ps=*/1000000, db, RooflineModelOptions{});

  EXPECT_EQ(record.hlo_name(), "tec_compute_heavy_op");
  EXPECT_EQ(record.sc_bound_by(), "TEC Compute");
  EXPECT_GT(record.sc_roofline_efficiency(), 0.0);
}

TEST(OpStatsToRooflineModelSparseCoreTest, DataTableExportCustomProperties) {
  RooflineModelDatabase db;
  db.set_num_sparse_core_tiles(32);
  db.set_peak_sc_flop_rate(96.0);
  db.set_peak_hbm_bw(1200.0);
  db.set_peak_spmem_read_bw(8192.0);
  db.set_peak_spmem_write_bw(4096.0);

  std::unique_ptr<DataTable> data_table = GenerateRooflineModelDataTable(db);
  ASSERT_NE(data_table, nullptr);
  bool has_is_sparse_core = false;
  for (const auto& col : data_table->GetColumns()) {
    if (col.id == "is_sparse_core") {
      has_is_sparse_core = true;
      break;
    }
  }
  EXPECT_TRUE(has_is_sparse_core);
}

TEST(OpStatsToRooflineModelSparseCoreTest, NegativeFlopsClamping) {
  RooflineModelDatabase db;
  db.set_num_sparse_core_tiles(16);
  db.set_peak_sc_flop_rate(96.0);
  db.set_peak_hbm_bw(1200.0);
  db.set_peak_spmem_read_bw(8192.0);
  db.set_peak_spmem_write_bw(4096.0);

  OpMetrics metrics;
  metrics.set_name("fusion.592.cloned.1.call-start");
  metrics.set_category("sparse_core");
  metrics.set_core_type(OpMetrics_TpuCoreType_SPARSE_CORE);
  metrics.set_time_ps(1000000);  // 1 us
  metrics.set_flops(-1);         // Negative FLOPs (e.g. from async call-start)
  metrics.set_flops_v2(-1.0);
  metrics.set_model_flops_v2(-1.0);
  metrics.set_bytes_accessed(1668);

  OpStats op_stats;
  ScRooflineModelRecord record = ConvertOpMetricsToScRooflineModelRecord(
      op_stats, metrics, RecordType::ALL, /*step_num=*/0,
      /*total_time_ps=*/1000000, db, RooflineModelOptions{});

  EXPECT_EQ(record.measured_flop_rate(), 0.0);
  EXPECT_EQ(record.model_flop_rate(), 0.0);
  EXPECT_EQ(record.operational_intensity(), 0.0);
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

  perf_env->set_peak_sc_tera_flops_per_second(12.5);
  perf_env->set_num_sparse_core_tiles(4);

  while (perf_env->peak_bws_giga_bytes_per_second_size() <= 8) {
    perf_env->add_peak_bws_giga_bytes_per_second(0.0);
  }
  perf_env->set_peak_bws_giga_bytes_per_second(0, 600.0);
  perf_env->set_peak_bws_giga_bytes_per_second(7, 4000.0);  // SPMEM Read
  perf_env->set_peak_bws_giga_bytes_per_second(8, 2000.0);  // SPMEM Write

  RooflineModelDatabase db =
      InitializeRooflineModelDatabaseFromOpStats(op_stats);

  EXPECT_GT(db.peak_spmem_read_bw(), 0.0);
  EXPECT_GT(db.peak_spmem_write_bw(), 0.0);
  EXPECT_EQ(db.num_sparse_core_tiles(), 4);
  EXPECT_DOUBLE_EQ(db.peak_sc_flop_rate(), 12500.0);
  EXPECT_GT(db.peak_hbm_bw(), 0.0);
}

TEST(OpStatsToRooflineModelSparseCoreTest,
     PerfEnvDefaultSparseCoreTileAndFlopRatePropagation) {
  OpStats op_stats;
  op_stats.mutable_run_environment()->set_hardware_type(
      tensorflow::profiler::HardwareType::TPU);
  op_stats.mutable_run_environment()->set_device_type("TPU v5");
  auto* perf_env = op_stats.mutable_perf_env();
  perf_env->set_peak_tera_flops_per_second(48.0);
  perf_env->set_peak_hbm_bw_giga_bytes_per_second(600.0);

  while (perf_env->peak_bws_giga_bytes_per_second_size() <= 8) {
    perf_env->add_peak_bws_giga_bytes_per_second(0.0);
  }
  perf_env->set_peak_bws_giga_bytes_per_second(7, 4000.0);  // SPMEM Read
  perf_env->set_peak_bws_giga_bytes_per_second(8, 2000.0);  // SPMEM Write

  RooflineModelDatabase db =
      InitializeRooflineModelDatabaseFromOpStats(op_stats);

  EXPECT_GT(db.peak_spmem_read_bw(), 0.0);
  EXPECT_GT(db.peak_spmem_write_bw(), 0.0);
  EXPECT_EQ(db.num_sparse_core_tiles(), 16);
  EXPECT_DOUBLE_EQ(db.peak_sc_flop_rate(), 0.0);
  EXPECT_GT(db.peak_hbm_bw(), 0.0);
}

TEST(OpStatsToRooflineModelSparseCoreTest,
     ZeroSpmemBwDoesNotPopulateSparseCore) {
  OpStats op_stats;
  op_stats.mutable_run_environment()->set_hardware_type(
      tensorflow::profiler::HardwareType::TPU);
  op_stats.mutable_run_environment()->set_device_type("TPU v5");
  auto* perf_env = op_stats.mutable_perf_env();
  perf_env->set_peak_tera_flops_per_second(48.0);
  perf_env->set_peak_hbm_bw_giga_bytes_per_second(600.0);

  while (perf_env->peak_bws_giga_bytes_per_second_size() <= 8) {
    perf_env->add_peak_bws_giga_bytes_per_second(0.0);
  }

  RooflineModelDatabase db =
      InitializeRooflineModelDatabaseFromOpStats(op_stats);

  EXPECT_FALSE(db.has_peak_spmem_read_bw());
  EXPECT_DOUBLE_EQ(db.peak_spmem_read_bw(), 0.0);
  EXPECT_FALSE(db.has_peak_spmem_write_bw());
  EXPECT_DOUBLE_EQ(db.peak_spmem_write_bw(), 0.0);
  EXPECT_FALSE(db.has_num_sparse_core_tiles());
  EXPECT_EQ(db.num_sparse_core_tiles(), 0);
  EXPECT_FALSE(db.has_peak_sc_flop_rate());
  EXPECT_DOUBLE_EQ(db.peak_sc_flop_rate(), 0.0);
}

TEST(OpStatsToRooflineModelSparseCoreTest,
     EndToEndConvertOpStatsToRooflineModel) {
  OpStats op_stats;
  op_stats.mutable_run_environment()->set_hardware_type(
      tensorflow::profiler::HardwareType::TPU);
  op_stats.mutable_run_environment()->set_device_type("TPU v5");
  auto* perf_env = op_stats.mutable_perf_env();
  perf_env->set_peak_tera_flops_per_second(48.0);
  perf_env->set_peak_hbm_bw_giga_bytes_per_second(600.0);
  perf_env->set_peak_sc_tera_flops_per_second(12.5);
  while (perf_env->peak_bws_giga_bytes_per_second_size() <= 8) {
    perf_env->add_peak_bws_giga_bytes_per_second(0.0);
  }
  perf_env->set_peak_bws_giga_bytes_per_second(7, 4000.0);
  perf_env->set_peak_bws_giga_bytes_per_second(8, 2000.0);

  auto* db = op_stats.mutable_device_op_metrics_db();
  db->set_total_op_time_ps(10000000);
  auto* sc_metric = db->add_metrics_db();
  sc_metric->set_name("embedding_lookup_test");
  sc_metric->set_category("sparse_core");
  sc_metric->set_core_type(OpMetrics_TpuCoreType_SPARSE_CORE);
  sc_metric->set_occurrences(1);
  sc_metric->set_time_ps(10000000);
  sc_metric->set_flops_v2(50000);
  sc_metric->set_bytes_accessed(1000000);
  auto* mem = sc_metric->add_memory_accessed_breakdown();
  mem->set_memory_space(7 /* SPMEM Read */);
  mem->set_bytes_accessed(1000000);

  RooflineModelDatabase result_db =
      ConvertOpStatsToRooflineModel(op_stats);

  ASSERT_GE(result_db.sc_roofline_model_record_size(), 2);
  // Verify program summary record at rank 0
  const auto& program_rec = result_db.sc_roofline_model_record(0);
  EXPECT_EQ(program_rec.rank(), 0);
  EXPECT_EQ(program_rec.hlo_category(), "Program");
  EXPECT_EQ(program_rec.hlo_name(), "Program");

  // Verify op record at rank 1
  const auto& op_rec = result_db.sc_roofline_model_record(1);
  EXPECT_EQ(op_rec.rank(), 1);
  EXPECT_EQ(op_rec.hlo_name(), "embedding_lookup_test");
  EXPECT_GT(op_rec.optimal_flop_rate(), 0.0);
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
