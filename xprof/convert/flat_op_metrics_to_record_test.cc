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

#include "xprof/convert/flat_op_metrics_to_record.h"

#include <vector>

#include "net/proto2/contrib/parse_proto/parse_text_proto.h"
#include "<gtest/gtest.h>"
#include "plugin/xprof/protobuf/flat_op_metrics.pb.h"
#include "plugin/xprof/protobuf/roofline_model.pb.h"

namespace tensorflow {
namespace profiler {
namespace {

using ::google::protobuf::contrib::parse_proto::ParseTextProtoOrDie;
using ::tensorflow::profiler::roofline_model::RooflineModelRecord;

TEST(FlatOpMetricsToRecordTest, SortedOpMetricsDb) {
  FlatOpMetricsDb db = ParseTextProtoOrDie(R"pb(
    op_instances { hlo_name: "op1" self_time_ps: 100 core_type: TENSOR_CORE }
    op_instances { hlo_name: "op2" self_time_ps: 200 core_type: TENSOR_CORE }
    op_instances { hlo_name: "op3" self_time_ps: 150 core_type: TENSOR_CORE }
    op_instances { hlo_name: "op4" self_time_ps: 150 core_type: TENSOR_CORE }
    op_instances { hlo_name: "op5" self_time_ps: 300 core_type: SPARSE_CORE }
  )pb");

  // Default max_records (-1)
  {
    std::vector<const FlatOpMetrics*> sorted = SortedOpMetricsDb(db);
    // Sparse core should be excluded.
    ASSERT_EQ(sorted.size(), 4);
    // Ordered by self_time_ps descending, then hlo_name descending.
    EXPECT_EQ(sorted[0]->hlo_name(), "op2");  // 200
    EXPECT_EQ(sorted[1]->hlo_name(), "op4");  // 150 (op4 > op3)
    EXPECT_EQ(sorted[2]->hlo_name(), "op3");  // 150
    EXPECT_EQ(sorted[3]->hlo_name(), "op1");  // 100
  }

  // max_records = 2
  {
    std::vector<const FlatOpMetrics*> sorted = SortedOpMetricsDb(db, 2);
    ASSERT_EQ(sorted.size(), 2);
    EXPECT_EQ(sorted[0]->hlo_name(), "op2");
    EXPECT_EQ(sorted[1]->hlo_name(), "op4");
  }
}

// Minimal PerfEnv and RunEnvironment for testing SetRooflineMetrics.
PerfEnv GetTestPerfEnv() {
  return ParseTextProtoOrDie(R"pb(
    peak_tera_flops_per_second: 100.0
    peak_bws_giga_bytes_per_second: 1000.0  # Index 0 (HBM)
    peak_bws_giga_bytes_per_second: 0.0
    peak_bws_giga_bytes_per_second: 500.0   # Index 2 (Shm/L1 for GPU)
    peak_bws_giga_bytes_per_second: 2000.0  # Index 3 (CMEM Read)
    peak_bws_giga_bytes_per_second: 2000.0  # Index 4 (CMEM Write)
    peak_bws_giga_bytes_per_second: 4000.0  # Index 5 (VMEM Read)
    peak_bws_giga_bytes_per_second: 4000.0  # Index 6 (VMEM Write)
  )pb");
}

RunEnvironment GetTestRunEnv(HardwareType hardware_type) {
  RunEnvironment env;
  env.set_hardware_type(hardware_type);
  return env;
}

TEST(FlatOpMetricsToRecordTest, SetRooflineMetricsComputeBound) {
  FlatOpMetrics metrics = ParseTextProtoOrDie(R"pb(
    time_ps: 1000000000000  # 1s
    flops: 100000000000000  # 100 TFLOPS
    flops_v2: 100000000000000
    bytes_accessed: 1000000000  # 1 GB
  )pb");

  RooflineModelRecord record;
  SetRooflineMetrics(metrics, GetTestPerfEnv(), GetTestRunEnv(TPU), &record);

  // Peak flops = 100 TFLOPS. Measured = 100 TFLOPS. Utilization = 1.0
  // Peak HBM BW = 1000 GB/s. Measured = 1 GB/s. Utilization = 0.001
  EXPECT_EQ(record.bound_by(), "Compute");
}

TEST(FlatOpMetricsToRecordTest, SetRooflineMetricsHbmBound) {
  FlatOpMetrics metrics = ParseTextProtoOrDie(R"pb(
    time_ps: 1000000000000  # 1s
    flops: 1000000000000    # 1 TFLOPS
    flops_v2: 1000000000000
    bytes_accessed: 1000000000000  # 1 TB
  )pb");

  RooflineModelRecord record;
  SetRooflineMetrics(metrics, GetTestPerfEnv(), GetTestRunEnv(TPU), &record);

  // Peak flops = 100 TFLOPS. Measured = 1 TFLOPS. Utilization = 0.01
  // Peak HBM BW = 1000 GB/s. Measured = 1000 GB/s. Utilization = 1.0
  EXPECT_EQ(record.bound_by(), "HBM");
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
