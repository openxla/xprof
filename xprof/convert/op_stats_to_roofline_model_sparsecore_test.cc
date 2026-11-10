#include <memory>

#include <gtest/gtest.h>
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
  EXPECT_DOUBLE_EQ(record.bottleneck_operational_intensity(),
                   record.hbm_operational_intensity());
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

  // On SparseCore, the on-chip (VMEM) memory space is the SPMEM.
  auto* mem_read = metrics.add_memory_accessed_breakdown();
  mem_read->set_memory_space(PerformanceInfo::MemoryAccessed::VMEM);
  mem_read->set_operation_type(OpMetrics::MemoryAccessed::READ);
  mem_read->set_bytes_accessed(
      10000000);  // 10MB in 1us -> 10,000 GB/s SPMEM Read BW

  OpStats op_stats;
  ScRooflineModelRecord record = ConvertOpMetricsToScRooflineModelRecord(
      op_stats, metrics, RecordType::ALL, /*step_num=*/0,
      /*total_time_ps=*/1000000, db, RooflineModelOptions{});

  EXPECT_EQ(record.hlo_name(), "embedding_lookup_spmem");
  EXPECT_EQ(record.sc_bound_by(), "SPMEM Read");
  EXPECT_GT(record.spmem_read_bw(), 0.0);
  EXPECT_DOUBLE_EQ(record.spmem_write_bw(), 0.0);
  EXPECT_DOUBLE_EQ(record.hbm_bw(), 0.0);
  EXPECT_GT(record.sc_roofline_efficiency(), 0.0);
  EXPECT_DOUBLE_EQ(record.bottleneck_operational_intensity(),
                   record.spmem_read_operational_intensity());
}

TEST(OpStatsToRooflineModelSparseCoreTest,
     SparseCoreMemorySpaceClassificationIgnoresUnknownAndCmem) {
  RooflineModelDatabase db;
  db.set_peak_sc_flop_rate(48.0);
  db.set_peak_hbm_bw(643.7);
  db.set_peak_spmem_read_bw(4096.0);
  db.set_peak_spmem_write_bw(2048.0);

  OpMetrics metrics;
  metrics.set_name("mixed_memory_spaces");
  metrics.set_core_type(OpMetrics_TpuCoreType_SPARSE_CORE);
  metrics.set_time_ps(1000000);  // 1 us
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
  // UNKNOWN and CMEM accesses must not be attributed to HBM or SPMEM.
  add_access(PerformanceInfo::MemoryAccessed::UNKNOWN,
             OpMetrics::MemoryAccessed::READ, 500);
  add_access(PerformanceInfo::MemoryAccessed::CMEM,
             OpMetrics::MemoryAccessed::WRITE, 500);

  OpStats op_stats;
  ScRooflineModelRecord record = ConvertOpMetricsToScRooflineModelRecord(
      op_stats, metrics, RecordType::ALL, /*step_num=*/0,
      /*total_time_ps=*/1000000, db, RooflineModelOptions{});

  // 1 us == 1000 ns.
  EXPECT_DOUBLE_EQ(record.hbm_bw(),
                   ::tsl::profiler::GibibytesPerSecond(1000, 1000.0));
  EXPECT_DOUBLE_EQ(record.spmem_read_bw(),
                   ::tsl::profiler::GibibytesPerSecond(2000, 1000.0));
  EXPECT_DOUBLE_EQ(record.spmem_write_bw(),
                   ::tsl::profiler::GibibytesPerSecond(3000, 1000.0));
}

TEST(OpStatsToRooflineModelSparseCoreTest,
     SparseCoreWithoutBreakdownTreatsAllBytesAsHbm) {
  RooflineModelDatabase db;
  db.set_peak_sc_flop_rate(48.0);
  db.set_peak_hbm_bw(643.7);

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
  db.set_peak_sc_flop_rate(48.0);
  db.set_peak_hbm_bw(643.7);
  db.set_peak_spmem_read_bw(4096.0);
  db.set_peak_spmem_write_bw(2048.0);

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

TEST(OpStatsToRooflineModelSparseCoreTest, InvalidWrappedFlopsAreIgnored) {
  RooflineModelDatabase db;
  db.set_peak_sc_flop_rate(48.0);
  db.set_peak_hbm_bw(643.7);

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
  // Without megacore, the SparseCore HBM peak equals the TensorCore one.
  EXPECT_DOUBLE_EQ(db.peak_sc_hbm_bw(), db.peak_hbm_bw());

  // With megacore, SparseCores see the HBM bandwidth of both TensorCores.
  perf_env->set_has_megacore(true);
  RooflineModelDatabase megacore_db =
      InitializeRooflineModelDatabaseFromOpStats(op_stats);
  EXPECT_DOUBLE_EQ(megacore_db.peak_sc_hbm_bw(),
                   2 * megacore_db.peak_hbm_bw());
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
  perf_env->set_peak_bws_giga_bytes_per_second(0, 600.0);
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
  mem->set_memory_space(PerformanceInfo::MemoryAccessed::VMEM);
  mem->set_operation_type(OpMetrics::MemoryAccessed::READ);
  mem->set_bytes_accessed(1000000);

  RooflineModelDatabase result_db = ConvertOpStatsToRooflineModel(op_stats);

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

TEST(OpStatsToRooflineModelSparseCoreTest,
     ProgramRecordExcludesNestedAndInvalidFlops) {
  OpStats op_stats;
  op_stats.mutable_run_environment()->set_hardware_type(
      tensorflow::profiler::HardwareType::TPU);
  auto* perf_env = op_stats.mutable_perf_env();
  perf_env->set_peak_tera_flops_per_second(48.0);
  perf_env->set_peak_hbm_bw_giga_bytes_per_second(600.0);
  perf_env->set_peak_sc_tera_flops_per_second(12.5);
  while (perf_env->peak_bws_giga_bytes_per_second_size() <= 8) {
    perf_env->add_peak_bws_giga_bytes_per_second(0.0);
  }
  perf_env->set_peak_bws_giga_bytes_per_second(0, 600.0);
  perf_env->set_peak_bws_giga_bytes_per_second(7, 4000.0);
  perf_env->set_peak_bws_giga_bytes_per_second(8, 2000.0);

  auto* db = op_stats.mutable_device_op_metrics_db();
  db->set_total_time_ps(1000000);  // 1 us

  // A TensorCore offload op whose children are the SparseCore ops.
  auto* tc_op = db->add_metrics_db();
  tc_op->set_name("offload");
  tc_op->set_core_type(OpMetrics_TpuCoreType_TENSOR_CORE);
  tc_op->set_occurrences(1);
  tc_op->set_time_ps(1000000);

  // An outer SparseCore op with a nested SparseCore child. The child's work
  // is already included in the parent and must not be counted twice.
  auto* sc_parent = tc_op->mutable_children()->add_metrics_db();
  sc_parent->set_name("sc_parent");
  sc_parent->set_category("custom fusion");
  sc_parent->set_core_type(OpMetrics_TpuCoreType_SPARSE_CORE);
  sc_parent->set_occurrences(1);
  sc_parent->set_time_ps(500000);
  sc_parent->set_flops_v2(1000);
  auto* sc_child = sc_parent->mutable_children()->add_metrics_db();
  sc_child->set_name("sc_child");
  sc_child->set_category("custom fusion");
  sc_child->set_core_type(OpMetrics_TpuCoreType_SPARSE_CORE);
  sc_child->set_occurrences(1);
  sc_child->set_time_ps(400000);
  sc_child->set_flops_v2(1000);

  // A SparseCore op with an unknown (wrapped) FLOP count.
  auto* sc_invalid = tc_op->mutable_children()->add_metrics_db();
  sc_invalid->set_name("sc_invalid");
  sc_invalid->set_category("custom fusion");
  sc_invalid->set_core_type(OpMetrics_TpuCoreType_SPARSE_CORE);
  sc_invalid->set_occurrences(1);
  sc_invalid->set_time_ps(100000);
  sc_invalid->set_flops_v2(18446744073709551616.0);

  RooflineModelDatabase result_db = ConvertOpStatsToRooflineModel(op_stats);

  int program_records = 0;
  for (const auto& record : result_db.sc_roofline_model_record()) {
    EXPECT_NE(record.hlo_name(), "sc_child");
    if (record.hlo_name() != "Program") continue;
    ++program_records;
    // Only the outer op's 1000 FLOPs over the SparseCore busy time (outermost
    // SparseCore ops: 500 ns + 100 ns), not the device-wide total time.
    EXPECT_DOUBLE_EQ(record.measured_flop_rate(), 1000.0 / 600.0);
  }
  EXPECT_GT(program_records, 0);
}

TEST(OpStatsToRooflineModelSparseCoreTest, AsyncWrappersAreSkipped) {
  OpStats op_stats;
  op_stats.mutable_run_environment()->set_hardware_type(
      tensorflow::profiler::HardwareType::TPU);
  auto* perf_env = op_stats.mutable_perf_env();
  perf_env->set_peak_tera_flops_per_second(48.0);
  perf_env->set_peak_hbm_bw_giga_bytes_per_second(600.0);
  perf_env->set_peak_sc_tera_flops_per_second(12.5);
  while (perf_env->peak_bws_giga_bytes_per_second_size() <= 8) {
    perf_env->add_peak_bws_giga_bytes_per_second(0.0);
  }
  perf_env->set_peak_bws_giga_bytes_per_second(0, 600.0);
  perf_env->set_peak_bws_giga_bytes_per_second(7, 4000.0);
  perf_env->set_peak_bws_giga_bytes_per_second(8, 2000.0);

  auto* db = op_stats.mutable_device_op_metrics_db();
  db->set_total_time_ps(100000000);  // 100 us device-wide.

  // TC-issued launch wrapper tagged as SparseCore. It carries the cost of the
  // whole offloaded computation but only lasts a few ns.
  auto* start = db->add_metrics_db();
  start->set_name("fusion.1.call-start");
  start->set_category("async-start");
  start->set_core_type(OpMetrics_TpuCoreType_SPARSE_CORE);
  start->set_occurrences(1);
  start->set_time_ps(5000);
  start->set_flops_v2(1000000);
  auto* body = start->mutable_children()->add_metrics_db();
  body->set_name("fusion.1");
  body->set_category("custom fusion");
  body->set_core_type(OpMetrics_TpuCoreType_SPARSE_CORE);
  body->set_occurrences(1);
  body->set_time_ps(1000000);  // 1 us
  body->set_flops_v2(1000000);
  auto* done = db->add_metrics_db();
  done->set_name("fusion.1.call-done");
  done->set_category("async-done");
  done->set_core_type(OpMetrics_TpuCoreType_SPARSE_CORE);
  done->set_occurrences(1);
  done->set_time_ps(2000000);

  RooflineModelDatabase result_db = ConvertOpStatsToRooflineModel(op_stats);

  ASSERT_EQ(result_db.sc_roofline_model_record_size(), 2);
  const auto& program_rec = result_db.sc_roofline_model_record(0);
  EXPECT_EQ(program_rec.hlo_name(), "Program");
  // 1e6 FLOPs over 1 us -> 1000 GFLOP/s.
  EXPECT_DOUBLE_EQ(program_rec.measured_flop_rate(), 1000.0);
  EXPECT_DOUBLE_EQ(program_rec.total_time_in_us(), 1.0);
  const auto& op_rec = result_db.sc_roofline_model_record(1);
  EXPECT_EQ(op_rec.hlo_name(), "fusion.1");
  EXPECT_DOUBLE_EQ(op_rec.measured_flop_rate(), 1000.0);
}

TEST(OpStatsToRooflineModelSparseCoreTest, ScHbmUtilizationUsesChipHbmPeak) {
  RooflineModelDatabase db;
  db.set_peak_sc_flop_rate(6100.0);
  db.set_peak_hbm_bw(1000.0);
  EXPECT_DOUBLE_EQ(PeakScHbmBw(db), 1000.0);
  db.set_peak_sc_hbm_bw(2000.0);
  EXPECT_DOUBLE_EQ(PeakScHbmBw(db), 2000.0);

  OpStats op_stats;
  OpMetrics metrics;
  metrics.set_name("sc_op");
  metrics.set_core_type(OpMetrics_TpuCoreType_SPARSE_CORE);
  metrics.set_occurrences(1);
  metrics.set_time_ps(1000000);  // 1 us
  // 1500 GiB/s of HBM traffic.
  metrics.set_bytes_accessed(1500ULL * 1024 * 1024 * 1024 / 1000000);
  auto* mem = metrics.add_memory_accessed_breakdown();
  mem->set_memory_space(PerformanceInfo::MemoryAccessed::HBM);
  mem->set_operation_type(OpMetrics::MemoryAccessed::READ);
  mem->set_bytes_accessed(metrics.bytes_accessed());

  ScRooflineModelRecord record = ConvertOpMetricsToScRooflineModelRecord(
      op_stats, metrics, RecordType::ALL, /*step_num=*/0,
      /*total_time_ps=*/1000000, db, RooflineModelOptions{});
  // Utilization is relative to the chip-level HBM peak (2000 GiB/s), so it
  // must stay below 1.
  EXPECT_NEAR(record.memory_bw_relative_to_hw_limit(), 0.75, 0.01);
  EXPECT_EQ(record.sc_bound_by(), "HBM");
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
