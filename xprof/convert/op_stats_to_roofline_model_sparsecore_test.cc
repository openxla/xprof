#include <memory>
#include <string>

#include "<gtest/gtest.h>"
#include "tensorflow/core/profiler/protobuf/op_metrics.pb.h"
#include "tensorflow/core/profiler/protobuf/op_stats.pb.h"
#include "xprof/convert/op_metrics_to_record.h"
#include "xprof/convert/op_stats_to_roofline_model.h"
#include "plugin/xprof/protobuf/roofline_model.pb.h"

namespace tensorflow {
namespace profiler {
namespace {

using ::tensorflow::profiler::roofline_model::RecordType;
using ::tensorflow::profiler::roofline_model::RooflineModelDatabase;
using ::tensorflow::profiler::roofline_model::RooflineModelRecord;
using ::tensorflow::profiler::roofline_model::ScRooflineModelRecord;

TEST(OpStatsToRooflineModelSparseCoreTest, SparseCoreDatabaseInitialization) {
  RooflineModelDatabase db;
  db.set_num_sparse_core_tiles(16);
  db.set_peak_sc_flop_rate(48.0);
  db.set_peak_sc_hbm_bw(643.7);
  db.set_peak_spmem_read_bw(4096.0);
  db.set_peak_spmem_write_bw(2048.0);

  EXPECT_EQ(db.num_sparse_core_tiles(), 16);
  EXPECT_DOUBLE_EQ(db.peak_sc_flop_rate(), 48.0);
  EXPECT_DOUBLE_EQ(db.peak_sc_hbm_bw(), 643.7);
  EXPECT_DOUBLE_EQ(db.peak_spmem_read_bw(), 4096.0);
  EXPECT_DOUBLE_EQ(db.peak_spmem_write_bw(), 2048.0);
}

TEST(OpStatsToRooflineModelSparseCoreTest, SparseCoreHbmBoundRecord) {
  RooflineModelDatabase db;
  db.set_num_sparse_core_tiles(16);
  db.set_peak_sc_flop_rate(48.0);
  db.set_peak_sc_hbm_bw(643.7);
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
      /*total_time_ps=*/10000000, db, /*include_infeed_outfeed=*/true);

  EXPECT_EQ(record.hlo_name(), "embedding_lookup_hbm");
  EXPECT_EQ(record.sc_bound_by(), "HBM");
  EXPECT_GT(record.sc_roofline_efficiency(), 0.0);
}

TEST(OpStatsToRooflineModelSparseCoreTest, SparseCoreSpmemBoundRecord) {
  RooflineModelDatabase db;
  db.set_num_sparse_core_tiles(16);
  db.set_peak_sc_flop_rate(48.0);
  db.set_peak_sc_hbm_bw(643.7);
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
      /*total_time_ps=*/1000000, db, /*include_infeed_outfeed=*/true);

  EXPECT_EQ(record.hlo_name(), "embedding_lookup_spmem");
  EXPECT_EQ(record.sc_bound_by(), "SPMEM");
  EXPECT_GT(record.spmem_read_bw(), 0.0);
  EXPECT_GT(record.sc_roofline_efficiency(), 0.0);
}

TEST(OpStatsToRooflineModelSparseCoreTest, SparseCoreTecComputeBoundRecord) {
  RooflineModelDatabase db;
  db.set_num_sparse_core_tiles(16);
  db.set_peak_sc_flop_rate(1.0);  // 1 GFLOP/s low peak
  db.set_peak_sc_hbm_bw(1000.0);  // High peak BW
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
      /*total_time_ps=*/1000000, db, /*include_infeed_outfeed=*/true);

  EXPECT_EQ(record.hlo_name(), "tec_compute_heavy_op");
  EXPECT_EQ(record.sc_bound_by(), "TEC Compute");
  EXPECT_GT(record.sc_roofline_efficiency(), 0.0);
}

TEST(OpStatsToRooflineModelSparseCoreTest, DataTableExportCustomProperties) {
  RooflineModelDatabase db;
  db.set_num_sparse_core_tiles(32);
  db.set_peak_sc_flop_rate(96.0);
  db.set_peak_sc_hbm_bw(1200.0);
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
  db.set_peak_sc_hbm_bw(1200.0);
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
      /*total_time_ps=*/1000000, db, /*include_infeed_outfeed=*/true);

  EXPECT_EQ(record.flops_v2(), 0.0);
  EXPECT_EQ(record.measured_flop_rate(), 0.0);
  EXPECT_EQ(record.model_flop_rate(), 0.0);
  EXPECT_EQ(record.operational_intensity(), 0.0);
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
