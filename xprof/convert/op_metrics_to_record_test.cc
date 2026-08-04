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

#include "xprof/convert/op_metrics_to_record.h"

#include "<gtest/gtest.h>"
#include "plugin/xprof/protobuf/op_metrics.pb.h"

namespace tensorflow {
namespace profiler {
namespace {

constexpr double kMaxError = 1E-10;

TEST(OpMetricsToRecordTest, GigaFlopsPerSecondPerCoreNormalizedOnDvfs) {
  OpMetrics metrics;
  metrics.set_time_ps(100);
  metrics.set_normalized_time_ps(200);
  metrics.set_flops_v2(1000);
  metrics.set_occurrences(1);
  metrics.set_num_cores(1);

  // GigaFlopsPerSecondPerCore = (flops_v2 / (time_ps / 1000.0)) = 1000 / 0.1 =
  // 10000. Multiplier = time_ps / normalized_time_ps = 100 / 200 = 0.5.
  // Expected normalized GFLOPS = 10000 * 0.5 = 5000.
  EXPECT_NEAR(5000.0, GigaFlopsPerSecondPerCoreNormalizedOnDvfs(metrics),
              kMaxError);
}

TEST(OpMetricsToRecordTest, GigaFlopsPerSecondPerCoreNormalizedOnDvfsFallback) {
  OpMetrics metrics;
  metrics.set_time_ps(100);
  metrics.set_normalized_time_ps(0);
  metrics.set_flops_v2(1000);
  metrics.set_occurrences(1);
  metrics.set_num_cores(1);

  EXPECT_NEAR(10000.0, GigaFlopsPerSecondPerCoreNormalizedOnDvfs(metrics),
              kMaxError);
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
