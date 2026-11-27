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

#include "xprof/convert/compute_inference_latency.h"

#include <gtest/gtest.h>
#include "plugin/xprof/protobuf/inference_stats.pb.h"

namespace tensorflow::profiler {
namespace {

constexpr double kMaxError = 0.0001;

TEST(ComputeInferenceLatencyResult, InferenceLatencyTest) {
  InferenceStats inference_stats;
  auto& model = (*inference_stats.mutable_inference_stats_per_model())[0];

  // Generates requests for testing.
  for (int i = 0; i < 100; i++) {
    RequestDetail request_detail;
    request_detail.set_start_time_ps(0);
    request_detail.set_end_time_ps(i * 10000);
    request_detail.set_device_time_ps(i * 1000);
    request_detail.set_write_to_device_time_ps(i * 1000);
    model.add_request_details()->Swap(&request_detail);
  }

  auto result = ComputeInferenceLatencyResult(inference_stats);

  // 5 percentiles and 1 average, so 6 results in total.
  ASSERT_EQ(result.latency_breakdowns_size(), 6);

  // Verify average result.
  EXPECT_NEAR(result.latency_breakdowns(0).total_latency_us(), 0.495,
              kMaxError);
  EXPECT_NEAR(result.latency_breakdowns(0).host_latency_us(), 0.396, kMaxError);
  EXPECT_NEAR(result.latency_breakdowns(0).device_latency_us(), 0.0495,
              kMaxError);
  EXPECT_NEAR(result.latency_breakdowns(0).communication_latency_us(), 0.0495,
              kMaxError);

  // Verify 50 percentile result.
  EXPECT_NEAR(result.latency_breakdowns(1).total_latency_us(), 0.5, kMaxError);
  EXPECT_NEAR(result.latency_breakdowns(1).host_latency_us(), 0.4, kMaxError);
  EXPECT_NEAR(result.latency_breakdowns(1).device_latency_us(), 0.05,
              kMaxError);
  EXPECT_NEAR(result.latency_breakdowns(1).communication_latency_us(), 0.05,
              kMaxError);

  // Verify 99.9 percentile result.
  EXPECT_NEAR(result.latency_breakdowns(5).total_latency_us(), 0.99, kMaxError);
  EXPECT_NEAR(result.latency_breakdowns(5).host_latency_us(), 0.792, kMaxError);
  EXPECT_NEAR(result.latency_breakdowns(5).device_latency_us(), 0.099,
              kMaxError);
  EXPECT_NEAR(result.latency_breakdowns(5).communication_latency_us(), 0.099,
              kMaxError);
}

TEST(ComputeInferenceLatencyResult, ClampsNegativeHostTimeToZero) {
  InferenceStats inference_stats;
  auto& model = (*inference_stats.mutable_inference_stats_per_model())[0];

  // Request 1: device + communication time exceeds session run duration
  // (15 us > 10 us). Must clamp to 0 instead of cancelling out request 2.
  RequestDetail* req1 = model.add_request_details();
  req1->set_start_time_ps(0);
  req1->set_end_time_ps(10000000);       // 10 us
  req1->set_device_time_ps(12000000);    // 12 us
  req1->set_write_to_device_time_ps(3000000);  // 3 us

  // Request 2: healthy request with host time = 10 - 2 - 2 = 6 us.
  RequestDetail* req2 = model.add_request_details();
  req2->set_start_time_ps(0);
  req2->set_end_time_ps(10000000);       // 10 us
  req2->set_device_time_ps(2000000);     // 2 us
  req2->set_write_to_device_time_ps(2000000);  // 2 us

  auto result = ComputeInferenceLatencyResult(inference_stats);
  ASSERT_EQ(result.latency_breakdowns_size(), 6);
  // Average host time is (0 + 6) / 2 = 3.0 us (unclamped would be (-5 + 6)/2).
  EXPECT_NEAR(result.latency_breakdowns(0).host_latency_us(), 3.0, kMaxError);
  for (int i = 0; i < result.latency_breakdowns_size(); ++i) {
    EXPECT_GE(result.latency_breakdowns(i).host_latency_us(), 0.0);
  }
}

TEST(ComputeInferenceLatencyResult, GuardsReversedRequestTimestampUnderflow) {
  InferenceStats inference_stats;
  auto& model = (*inference_stats.mutable_inference_stats_per_model())[0];
  RequestDetail* req = model.add_request_details();
  req->set_start_time_ps(5000000);
  req->set_end_time_ps(1000000);  // start > end

  auto result = ComputeInferenceLatencyResult(inference_stats);
  ASSERT_EQ(result.latency_breakdowns_size(), 6);
  EXPECT_DOUBLE_EQ(result.latency_breakdowns(0).total_latency_us(), 0.0);
  EXPECT_DOUBLE_EQ(result.latency_breakdowns(0).host_latency_us(), 0.0);
}

}  // namespace
}  // namespace tensorflow::profiler
