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

#include "xprof/utils/op_utils.h"

#include <memory>
#include <string>
#include <utility>

#include "<gtest/gtest.h>"
#include "xla/tsl/profiler/convert/xla_op_utils.h"
#include "plugin/xprof/protobuf/op_metrics.pb.h"
#include "plugin/xprof/protobuf/source_info.pb.h"
#include "xprof/utils/performance_info_wrapper.h"

namespace tensorflow {
namespace profiler {
namespace {

TEST(DeviceOpMetricsDbBuilderTest, EnterOpAccumulatesBasicMetrics) {
  OpMetricsDb db;
  DeviceOpMetricsDbBuilder builder(&db);

  DeviceOpMetricsDbBuilder::OpIdentifier op_id = {
      .program_id = 123,
      .name = "test_op",
      .category = "test_cat",
      .provenance = "test_prov",
      .deduplicated_name = "test_dedup",
      .long_name = "long_test_op",
  };

  DeviceOpMetricsDbBuilder::OpData metrics_data1 = {
      .is_eager = false,
      .occurrences = 2,
      .time_ps = 1000,
      .children_time_ps = 100,
      .flops = 10,
      .bytes_accessed = 20,
  };

  builder.EnterOp(op_id, metrics_data1);

  ASSERT_EQ(db.metrics_db_size(), 1);
  const OpMetrics& metrics = db.metrics_db(0);
  EXPECT_EQ(metrics.name(), "test_op");
  EXPECT_EQ(metrics.category(), "test_cat");
  EXPECT_EQ(metrics.provenance(), "test_prov");
  EXPECT_EQ(metrics.deduplicated_name(), "test_dedup");
  EXPECT_EQ(metrics.long_name(), "long_test_op");
  EXPECT_FALSE(metrics.is_eager());
  EXPECT_EQ(metrics.occurrences(), 2);
  EXPECT_EQ(metrics.time_ps(), 1000);
  EXPECT_EQ(metrics.self_time_ps(), 900);
  EXPECT_EQ(metrics.flops(), 20);  // 10 * 2
  EXPECT_DOUBLE_EQ(metrics.flops_v2(), 20.0);
  EXPECT_EQ(metrics.model_flops(), 20);  // Fallback to flops
  EXPECT_DOUBLE_EQ(metrics.model_flops_v2(), 20.0);
  EXPECT_EQ(metrics.bytes_accessed(), 40);  // 20 * 2
}

TEST(DeviceOpMetricsDbBuilderTest, EnterOpAccumulatesMultipleOccurrences) {
  OpMetricsDb db;
  DeviceOpMetricsDbBuilder builder(&db);

  DeviceOpMetricsDbBuilder::OpIdentifier op_id = {
      .program_id = 123,
      .name = "test_op",
  };

  DeviceOpMetricsDbBuilder::OpData metrics_data1 = {
      .is_eager = false,
      .occurrences = 2,
      .time_ps = 1000,
      .children_time_ps = 100,
      .flops = 10,
      .bytes_accessed = 20,
  };

  builder.EnterOp(op_id, metrics_data1);

  DeviceOpMetricsDbBuilder::OpData metrics_data2 = {
      .is_eager = true,  // Now it is eager
      .occurrences = 1,
      .time_ps = 500,
      .children_time_ps = 50,
      .flops = 5,
      .bytes_accessed = 10,
  };

  builder.EnterOp(op_id, metrics_data2);

  ASSERT_EQ(db.metrics_db_size(), 1);
  const OpMetrics& metrics = db.metrics_db(0);
  EXPECT_FALSE(
      metrics
          .is_eager());  // Stays false due to early return in EnterOpMetadata
  EXPECT_EQ(metrics.occurrences(), 3);
  EXPECT_EQ(metrics.time_ps(), 1500);
  EXPECT_EQ(metrics.self_time_ps(), 1350);  // 900 + 450
  EXPECT_EQ(metrics.flops(), 25);           // 20 + 5
  EXPECT_DOUBLE_EQ(metrics.flops_v2(), 25.0);
  EXPECT_EQ(metrics.bytes_accessed(), 50);  // 40 + 10
}

TEST(DeviceOpMetricsDbBuilderTest, EnterOpPreservesEagerFromFirstOccurrence) {
  OpMetricsDb db;
  DeviceOpMetricsDbBuilder builder(&db);

  DeviceOpMetricsDbBuilder::OpIdentifier op_id = {
      .program_id = 123,
      .name = "test_op",
  };

  DeviceOpMetricsDbBuilder::OpData metrics_data1 = {
      .is_eager = true,
      .occurrences = 1,
  };

  builder.EnterOp(op_id, metrics_data1);

  DeviceOpMetricsDbBuilder::OpData metrics_data2 = {
      .is_eager = false,
      .occurrences = 1,
  };

  builder.EnterOp(op_id, metrics_data2);

  ASSERT_EQ(db.metrics_db_size(), 1);
  const OpMetrics& metrics = db.metrics_db(0);
  EXPECT_TRUE(metrics.is_eager());  // Should stay true
}

TEST(DeviceOpMetricsDbBuilderTest, EnterOpHandlesModelFlops) {
  OpMetricsDb db;
  DeviceOpMetricsDbBuilder builder(&db);

  DeviceOpMetricsDbBuilder::OpIdentifier op_id = {
      .program_id = 123,
      .name = "test_op",
  };

  DeviceOpMetricsDbBuilder::OpData metrics_data = {
      .occurrences = 1,
      .flops = 10,
      .model_flops = 15,  // Different from flops
  };

  builder.EnterOp(op_id, metrics_data);

  ASSERT_EQ(db.metrics_db_size(), 1);
  const OpMetrics& metrics = db.metrics_db(0);
  EXPECT_EQ(metrics.flops(), 10);
  EXPECT_EQ(metrics.model_flops(), 15);
  EXPECT_DOUBLE_EQ(metrics.model_flops_v2(), 15.0);
}

TEST(DeviceOpMetricsDbBuilderTest, EnterOpHandlesSourceInfo) {
  OpMetricsDb db;
  DeviceOpMetricsDbBuilder builder(&db);

  tsl::profiler::OpSourceInfo source_info = {
      .source_file = "file.py",
      .source_line = 42,
      .stack_frame = "stack_frame_info",
  };

  DeviceOpMetricsDbBuilder::OpIdentifier op_id = {
      .program_id = 123,
      .name = "test_op",
      .op_source_info = source_info,
  };

  DeviceOpMetricsDbBuilder::OpData metrics_data;

  builder.EnterOp(op_id, metrics_data);

  ASSERT_EQ(db.metrics_db_size(), 1);
  const OpMetrics& metrics = db.metrics_db(0);
  EXPECT_EQ(metrics.source_info().file_name(), "file.py");
  EXPECT_EQ(metrics.source_info().line_number(), 42);
  EXPECT_EQ(metrics.source_info().stack_frame(), "stack_frame_info");
}

TEST(DeviceOpMetricsDbBuilderTest, EnterOpAccumulatesMemoryAccessedBreakdown) {
  OpMetricsDb db;
  DeviceOpMetricsDbBuilder builder(&db);

  DeviceOpMetricsDbBuilder::OpIdentifier op_id = {
      .program_id = 123,
      .name = "test_op",
  };

  DeviceOpMetricsDbBuilder::OpData metrics_data1;
  metrics_data1.occurrences = 1;

  OpMetrics::MemoryAccessed* mem1 =
      metrics_data1.memory_accessed_breakdown.Add();
  mem1->set_memory_space(1);  // e.g. HBM
  mem1->set_operation_type(OpMetrics::MemoryAccessed::READ);
  mem1->set_bytes_accessed(100);

  builder.EnterOp(op_id, metrics_data1);

  ASSERT_EQ(db.metrics_db_size(), 1);
  const OpMetrics& metrics = db.metrics_db(0);
  ASSERT_EQ(metrics.memory_accessed_breakdown_size(), 1);
  EXPECT_EQ(metrics.memory_accessed_breakdown(0).memory_space(), 1);
  EXPECT_EQ(metrics.memory_accessed_breakdown(0).operation_type(),
            OpMetrics::MemoryAccessed::READ);
  EXPECT_EQ(metrics.memory_accessed_breakdown(0).bytes_accessed(), 100);

  // Call again with same memory space/op type to check accumulation.
  DeviceOpMetricsDbBuilder::OpData metrics_data2;
  metrics_data2.occurrences = 1;
  OpMetrics::MemoryAccessed* mem2 =
      metrics_data2.memory_accessed_breakdown.Add();
  mem2->set_memory_space(1);
  mem2->set_operation_type(OpMetrics::MemoryAccessed::READ);
  mem2->set_bytes_accessed(50);

  // And add a new memory space/op type.
  OpMetrics::MemoryAccessed* mem3 =
      metrics_data2.memory_accessed_breakdown.Add();
  mem3->set_memory_space(2);
  mem3->set_operation_type(OpMetrics::MemoryAccessed::WRITE);
  mem3->set_bytes_accessed(200);

  builder.EnterOp(op_id, metrics_data2);

  ASSERT_EQ(metrics.memory_accessed_breakdown_size(), 2);

  // Find the one with memory_space 1.
  const OpMetrics::MemoryAccessed* resolved_mem1 = nullptr;
  const OpMetrics::MemoryAccessed* resolved_mem2 = nullptr;
  for (const OpMetrics::MemoryAccessed& mem :
       metrics.memory_accessed_breakdown()) {
    if (mem.memory_space() == 1) resolved_mem1 = &mem;
    if (mem.memory_space() == 2) resolved_mem2 = &mem;
  }

  ASSERT_NE(resolved_mem1, nullptr);
  EXPECT_EQ(resolved_mem1->bytes_accessed(), 150);  // 100 + 50

  ASSERT_NE(resolved_mem2, nullptr);
  EXPECT_EQ(resolved_mem2->bytes_accessed(), 200);
}

TEST(DeviceOpMetricsDbBuilderTest, EnterOpHandlesGPUFields) {
  OpMetricsDb db;
  DeviceOpMetricsDbBuilder builder(&db);

  DeviceOpMetricsDbBuilder::OpIdentifier op_id = {
      .program_id = 123,
      .name = "gpu_op",
  };

  auto perf_info = std::make_unique<PerformanceInfoWrapper::PerfInfoType>();
  perf_info->set_flops(100);
  perf_info->set_bytes_accessed(200);
  auto* mem = perf_info->mutable_memory_accessed_breakdown()->Add();
  mem->set_memory_space(
      PerformanceInfoWrapper::PerfInfoType::MemoryAccessed::HBM);
  mem->set_is_read(true);
  mem->set_bytes_accessed(50);

  std::unique_ptr<PerformanceInfoWrapper> perf_info_wrapper =
      PerformanceInfoWrapper::Create(std::move(perf_info));

  DeviceOpMetricsDbBuilder::OpData event_data = {
      .occurrences = 2,
      .dma_stall_ps = 500,
      .perf_info = perf_info_wrapper.get(),
  };

  builder.EnterOp(op_id, event_data);

  ASSERT_EQ(db.metrics_db_size(), 1);
  const OpMetrics& metrics = db.metrics_db(0);
  EXPECT_EQ(metrics.flops(), 200);           // 100 * 2
  EXPECT_EQ(metrics.bytes_accessed(), 400);  // 200 * 2
  EXPECT_EQ(metrics.model_flops(),
            200);  // 100 * 2 (DeviceFlops fallback/match)
  EXPECT_EQ(metrics.dma_stall_ps(), 1000);  // 500 * 2
  ASSERT_EQ(metrics.memory_accessed_breakdown_size(), 1);
  EXPECT_EQ(metrics.memory_accessed_breakdown(0).bytes_accessed(),
            100);  // 50 * 2
}

TEST(DeviceOpMetricsDbBuilderTest, EnterOpHandlesGPUNegativeBytesFallback) {
  OpMetricsDb db;
  DeviceOpMetricsDbBuilder builder(&db);

  DeviceOpMetricsDbBuilder::OpIdentifier op_id = {
      .program_id = 123,
      .name = "gpu_op",
  };

  auto perf_info = std::make_unique<PerformanceInfoWrapper::PerfInfoType>();
  auto* mem = perf_info->mutable_memory_accessed_breakdown()->Add();
  mem->set_memory_space(
      PerformanceInfoWrapper::PerfInfoType::MemoryAccessed::HBM);
  mem->set_is_read(true);
  mem->set_bytes_accessed(-50);  // Negative bytes

  std::unique_ptr<PerformanceInfoWrapper> perf_info_wrapper =
      PerformanceInfoWrapper::Create(std::move(perf_info));

  DeviceOpMetricsDbBuilder::OpData event_data = {
      .occurrences = 2,
      .perf_info = perf_info_wrapper.get(),
  };

  builder.EnterOp(op_id, event_data);

  ASSERT_EQ(db.metrics_db_size(), 1);
  const OpMetrics& metrics = db.metrics_db(0);
  ASSERT_EQ(metrics.memory_accessed_breakdown_size(), 1);
  EXPECT_EQ(metrics.memory_accessed_breakdown(0).bytes_accessed(),
            0);  // Should fall back to 0
}

TEST(DeviceOpMetricsDbBuilderTest, EnterOpMetadataPopulatesMetadata) {
  OpMetricsDb db;
  DeviceOpMetricsDbBuilder builder(&db);

  tsl::profiler::OpSourceInfo source_info = {
      .source_file = "file.py",
      .source_line = 42,
      .stack_frame = "stack_frame_info",
  };

  DeviceOpMetricsDbBuilder::OpIdentifier op_id = {
      .program_id = 123,
      .name = "test_op",
      .category = "test_cat",
      .provenance = "test_prov",
      .deduplicated_name = "test_dedup",
      .long_name = "long_test_op",
      .op_source_info = source_info,
  };

  builder.EnterOpMetadata(op_id, /*is_eager=*/true);

  ASSERT_EQ(db.metrics_db_size(), 1);
  const OpMetrics& metrics = db.metrics_db(0);
  EXPECT_EQ(metrics.name(), "test_op");
  EXPECT_EQ(metrics.category(), "test_cat");
  EXPECT_EQ(metrics.provenance(), "test_prov");
  EXPECT_EQ(metrics.deduplicated_name(), "test_dedup");
  EXPECT_EQ(metrics.long_name(), "long_test_op");
  EXPECT_TRUE(metrics.is_eager());
  EXPECT_EQ(metrics.source_info().file_name(), "file.py");
  EXPECT_EQ(metrics.source_info().line_number(), 42);
  EXPECT_EQ(metrics.source_info().stack_frame(), "stack_frame_info");
}

TEST(DeviceOpMetricsDbBuilderTest,
     EnterOpMetadataDoesNotOverwriteIfOccurrencesExist) {
  OpMetricsDb db;
  DeviceOpMetricsDbBuilder builder(&db);

  DeviceOpMetricsDbBuilder::OpIdentifier op_id1 = {
      .program_id = 123,
      .name = "test_op",
      .category = "initial_cat",
      .provenance = "initial_prov",
  };

  DeviceOpMetricsDbBuilder::OpData event_data = {
      .occurrences = 1,
  };

  // Enters OP and sets occurrences to 1, metadata populated
  builder.EnterOp(op_id1, event_data);

  DeviceOpMetricsDbBuilder::OpIdentifier op_id2 = {
      .program_id = 123,
      .name = "test_op",
      .category = "new_cat",
      .provenance = "new_prov",
  };

  // Should not overwrite because occurrences > 0
  builder.EnterOpMetadata(op_id2, /*is_eager=*/true);

  ASSERT_EQ(db.metrics_db_size(), 1);
  const OpMetrics& metrics = db.metrics_db(0);
  EXPECT_EQ(metrics.category(), "initial_cat");
  EXPECT_EQ(metrics.provenance(), "initial_prov");
}

TEST(DeviceOpMetricsDbBuilderTest, EnterOpAccumulatesVddEnergy) {
  OpMetricsDb db;
  DeviceOpMetricsDbBuilder builder(&db);

  DeviceOpMetricsDbBuilder::OpIdentifier op_id = {
      .program_id = 123,
      .name = "test_op",
      .category = "test_cat",
      .provenance = "test_prov",
      .deduplicated_name = "test_dedup",
  };

  DeviceOpMetricsDbBuilder::OpData metrics_data1 = {
      .is_eager = false,
      .occurrences = 2,
      .time_ps = 1000,
      .children_time_ps = 100,
      .flops = 10,
      .bytes_accessed = 20,
      .vdd_energy_j = 1.5,
  };

  // Let's call EnterOp with some energy.
  builder.EnterOp(op_id, metrics_data1);

  ASSERT_EQ(db.metrics_db_size(), 1);
  const OpMetrics& metrics = db.metrics_db(0);
  EXPECT_EQ(metrics.name(), "test_op");
  EXPECT_DOUBLE_EQ(metrics.vdd_energy_j(), 1.5);

  DeviceOpMetricsDbBuilder::OpData metrics_data2 = {
      .is_eager = false,
      .occurrences = 1,
      .time_ps = 500,
      .children_time_ps = 50,
      .flops = 5,
      .bytes_accessed = 10,
      .vdd_energy_j = 2.2,
  };

  // Call EnterOp again on same op to check accumulation.
  builder.EnterOp(op_id, metrics_data2);

  EXPECT_DOUBLE_EQ(metrics.vdd_energy_j(), 3.7);

  DeviceOpMetricsDbBuilder::OpIdentifier other_op_id = {
      .program_id = 123,
      .name = "other_op",
      .category = "test_cat",
      .provenance = "test_prov",
      .deduplicated_name = "test_dedup",
  };

  DeviceOpMetricsDbBuilder::OpData metrics_data3 = {
      .is_eager = false,
      .occurrences = 1,
      .time_ps = 500,
      .children_time_ps = 50,
      .flops = 5,
      .bytes_accessed = 10,
      .vdd_energy_j = 0.0,
  };

  // Call EnterOp with zero vdd_energy_j, it shouldn't change or set it if
  // started from 0.
  builder.EnterOp(other_op_id, metrics_data3);

  ASSERT_EQ(db.metrics_db_size(), 2);
  const OpMetrics& other_metrics = db.metrics_db(1);
  EXPECT_EQ(other_metrics.name(), "other_op");
  EXPECT_FALSE(other_metrics.has_vdd_energy_j());
}

TEST(DeviceFlatOpMetricsDbBuilderTest, EnterOpAccumulatesBasicMetrics) {
  FlatOpMetricsDb db;
  DeviceFlatOpMetricsDbBuilder builder(&db);

  tsl::profiler::OpSourceInfo source_info = {
      .source_file = "file.py",
      .source_line = 42,
      .stack_frame = "stack_frame_info",
  };

  DeviceFlatOpMetricsDbBuilder::OpIdentifier op_id = {
      .program_id = 123,
      .name = "test_op",
      .category = "test_cat",
      .provenance = "test_prov",
      .deduplicated_name = "test_dedup",
      .long_name = "long_test_op",
      .op_source_info = source_info,
  };

  DeviceFlatOpMetricsDbBuilder::OpData metrics_data1 = {
      .is_eager = false,
      .occurrences = 2,
      .time_ps = 1000,
      .children_time_ps = 100,
      .flops = 10,
      .bytes_accessed = 20,
  };

  builder.EnterOp(op_id, metrics_data1);

  ASSERT_EQ(db.op_instances_size(), 1);
  const FlatOpMetrics& metrics = db.op_instances(0);
  EXPECT_EQ(metrics.hlo_name(), "test_op");
  EXPECT_EQ(metrics.category(), "test_cat");
  EXPECT_EQ(metrics.provenance(), "test_prov");
  EXPECT_EQ(metrics.deduplicated_name(), "test_dedup");
  EXPECT_EQ(metrics.long_name(), "long_test_op");
  EXPECT_FALSE(metrics.is_eager());
  EXPECT_EQ(metrics.occurrences(), 2);
  EXPECT_EQ(metrics.time_ps(), 1000);
  EXPECT_EQ(metrics.self_time_ps(), 900);
  EXPECT_DOUBLE_EQ(metrics.flops_v2(), 20.0);
  EXPECT_DOUBLE_EQ(metrics.model_flops_v2(), 20.0);
  EXPECT_EQ(metrics.bytes_accessed(), 40);  // 20 * 2
  EXPECT_EQ(metrics.source_info().file_name(), "file.py");
  EXPECT_EQ(metrics.source_info().line_number(), 42);
  EXPECT_EQ(metrics.source_info().stack_frame(), "stack_frame_info");
}

TEST(DeviceFlatOpMetricsDbBuilderTest, EnterOpMetadataPopulatesMetadata) {
  FlatOpMetricsDb db;
  DeviceFlatOpMetricsDbBuilder builder(&db);

  tsl::profiler::OpSourceInfo source_info = {
      .source_file = "file.py",
      .source_line = 42,
      .stack_frame = "stack_frame_info",
  };

  DeviceFlatOpMetricsDbBuilder::OpIdentifier op_id = {
      .program_id = 123,
      .name = "test_op",
      .category = "test_cat",
      .provenance = "test_prov",
      .deduplicated_name = "test_dedup",
      .long_name = "long_test_op",
      .op_source_info = source_info,
  };

  builder.EnterOpMetadata(op_id, /*is_eager=*/true);

  ASSERT_EQ(db.op_instances_size(), 1);
  const FlatOpMetrics& metrics = db.op_instances(0);
  EXPECT_EQ(metrics.hlo_name(), "test_op");
  EXPECT_EQ(metrics.category(), "test_cat");
  EXPECT_EQ(metrics.provenance(), "test_prov");
  EXPECT_EQ(metrics.deduplicated_name(), "test_dedup");
  EXPECT_EQ(metrics.long_name(), "long_test_op");
  EXPECT_TRUE(metrics.is_eager());
  EXPECT_EQ(metrics.source_info().file_name(), "file.py");
  EXPECT_EQ(metrics.source_info().line_number(), 42);
  EXPECT_EQ(metrics.source_info().stack_frame(), "stack_frame_info");
}

TEST(DeviceFlatOpMetricsDbBuilderTest,
     EnterOpAccumulatesMemoryAccessedBreakdown) {
  FlatOpMetricsDb db;
  DeviceFlatOpMetricsDbBuilder builder(&db);

  DeviceFlatOpMetricsDbBuilder::OpIdentifier op_id = {
      .program_id = 123,
      .name = "test_op",
  };

  DeviceFlatOpMetricsDbBuilder::OpData metrics_data1;
  metrics_data1.occurrences = 1;

  OpMetrics::MemoryAccessed* mem1 =
      metrics_data1.memory_accessed_breakdown.Add();
  mem1->set_memory_space(1);  // e.g. HBM
  mem1->set_operation_type(OpMetrics::MemoryAccessed::READ);
  mem1->set_bytes_accessed(100);

  builder.EnterOp(op_id, metrics_data1);

  ASSERT_EQ(db.op_instances_size(), 1);
  const FlatOpMetrics& metrics = db.op_instances(0);
  ASSERT_EQ(metrics.memory_accessed_breakdown_size(), 1);
  EXPECT_EQ(metrics.memory_accessed_breakdown(0).memory_space(), 1);
  EXPECT_EQ(metrics.memory_accessed_breakdown(0).operation_type(),
            FlatOpMetrics::MemoryAccessed::READ);
  EXPECT_EQ(metrics.memory_accessed_breakdown(0).bytes_accessed(), 100);

  // Call again with same memory space/op type to check accumulation.
  DeviceFlatOpMetricsDbBuilder::OpData metrics_data2;
  metrics_data2.occurrences = 1;
  OpMetrics::MemoryAccessed* mem2 =
      metrics_data2.memory_accessed_breakdown.Add();
  mem2->set_memory_space(1);
  mem2->set_operation_type(OpMetrics::MemoryAccessed::READ);
  mem2->set_bytes_accessed(50);

  // And add a new memory space/op type.
  OpMetrics::MemoryAccessed* mem3 =
      metrics_data2.memory_accessed_breakdown.Add();
  mem3->set_memory_space(2);
  mem3->set_operation_type(OpMetrics::MemoryAccessed::WRITE);
  mem3->set_bytes_accessed(200);

  builder.EnterOp(op_id, metrics_data2);

  ASSERT_EQ(metrics.memory_accessed_breakdown_size(), 2);

  // Find the one with memory_space 1.
  const FlatOpMetrics::MemoryAccessed* resolved_mem1 = nullptr;
  const FlatOpMetrics::MemoryAccessed* resolved_mem2 = nullptr;
  for (const FlatOpMetrics::MemoryAccessed& mem :
       metrics.memory_accessed_breakdown()) {
    if (mem.memory_space() == 1) resolved_mem1 = &mem;
    if (mem.memory_space() == 2) resolved_mem2 = &mem;
  }

  ASSERT_NE(resolved_mem1, nullptr);
  EXPECT_EQ(resolved_mem1->bytes_accessed(), 150);  // 100 + 50

  ASSERT_NE(resolved_mem2, nullptr);
  EXPECT_EQ(resolved_mem2->bytes_accessed(), 200);
}

TEST(HostOpMetricsDbBuilderTest, EnterOpAccumulatesTimings) {
  OpMetricsDb db;
  HostOpMetricsDbBuilder builder(&db);

  builder.EnterOp(/*name=*/"host_op", /*category=*/"host_category",
                  /*is_eager=*/true, /*time_ps=*/1000,
                  /*children_time_ps=*/200, /*id=*/5);

  ASSERT_EQ(db.metrics_db_size(), 1);
  const OpMetrics& metrics = db.metrics_db(0);
  EXPECT_EQ(metrics.name(), "host_op");
  EXPECT_EQ(metrics.category(), "host_category");
  EXPECT_TRUE(metrics.is_eager());
  EXPECT_EQ(metrics.time_ps(), 1000);
  EXPECT_EQ(metrics.self_time_ps(), 800);
  EXPECT_EQ(db.total_op_time_ps(), 800);
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
