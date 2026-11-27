/* Copyright 2026 The OpenXLA Authors. All Rights Reserved.

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

#include "xprof/convert/tpu_generic_utilization_utils.h"

#include <cstdint>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "xprof/convert/tpu_counter_util.h"
#include "xprof/utils/tpu_counter_ids_v6e.h"
#include "xprof/utils/tpu_counter_ids_v7x.h"

namespace xprof {
namespace {

using ::testing::IsEmpty;
using ::testing::SizeIs;

TEST(TpuGenericUtilizationUtilsTest, IciBandwidthSkipsNonPositiveFrequency) {
  absl::flat_hash_map<uint64_t, uint64_t> counters_map = {
      {TpuCounterIdsTpu6e::
           VF_CHIP_TC_TCS_TC_MISC_TCS_STATS_TCS_STATS_COUNTERS_UNPRIVILEGED_COUNT_CYCLES,  // NOLINT
       1750},
      {TpuCounterIdsTpu6e::
           VF_CHIP_OCI_ICR_PERF_COUNTERS_FIFO_STATS_COUNTERS_UNPRIVILEGED_BN_RD_REQ_0_INSERTION_COUNT,  // NOLINT
       10},
  };
  TpuCounterUtil counters(/*host_id=*/0, /*device_id=*/0, /*correlation_id=*/0,
                          counters_map);

  TpuGenericUtilizationOptions zero_freq_options = {
      .num_mxu_per_tensor_core = 2,
      .cycles_per_xlu_instruction = 4,
      .is_tpu6e = true,
      .frequency_hz = 0.0,
      .peak_hbm_bw_bps = 1.0,
      .pstate_normalized_frequency_hz = 0.0,
  };
  UtilizationCounters zero_freq_utilization;
  ComputeTpuGenericIciBandwidthUtilization(counters, zero_freq_options,
                                           &zero_freq_utilization);
  EXPECT_THAT(zero_freq_utilization.metrics, IsEmpty());

  TpuGenericUtilizationOptions neg_freq_options = zero_freq_options;
  neg_freq_options.frequency_hz = -1.75e9;
  UtilizationCounters neg_freq_utilization;
  ComputeTpuGenericIciBandwidthUtilization(counters, neg_freq_options,
                                           &neg_freq_utilization);
  EXPECT_THAT(neg_freq_utilization.metrics, IsEmpty());
}

TEST(TpuGenericUtilizationUtilsTest, IciBandwidthTpuV6ePeakAndAchieved) {
  absl::flat_hash_map<uint64_t, uint64_t> counters_map = {
      {TpuCounterIdsTpu6e::
           VF_CHIP_TC_TCS_TC_MISC_TCS_STATS_TCS_STATS_COUNTERS_UNPRIVILEGED_COUNT_CYCLES,  // NOLINT
       1750},
      {TpuCounterIdsTpu6e::
           VF_CHIP_OCI_ICR_PERF_COUNTERS_FIFO_STATS_COUNTERS_UNPRIVILEGED_BN_RD_REQ_0_INSERTION_COUNT,  // NOLINT
       10},
      {TpuCounterIdsTpu6e::
           VF_CHIP_OCI_ICR_PERF_COUNTERS_FIFO_STATS_COUNTERS_UNPRIVILEGED_BN_WR_REQ_0_INSERTION_COUNT,  // NOLINT
       5},
  };
  TpuCounterUtil counters(/*host_id=*/0, /*device_id=*/0, /*correlation_id=*/0,
                          counters_map);
  TpuGenericUtilizationOptions options = {
      .num_mxu_per_tensor_core = 2,
      .cycles_per_xlu_instruction = 4,
      .is_tpu6e = true,
      .frequency_hz = 1.75e9,
      .peak_hbm_bw_bps = 1.0,
      .pstate_normalized_frequency_hz = 0.0,
  };

  UtilizationCounters utilization;
  ComputeTpuGenericIciBandwidthUtilization(counters, options, &utilization);

  ASSERT_THAT(utilization.metrics, SizeIs(2));
  EXPECT_EQ(utilization.metrics[0].metric, "ICI (Read)");
  EXPECT_DOUBLE_EQ(utilization.metrics[0].achieved, 1280.0);
  EXPECT_NEAR(utilization.metrics[0].peak, 294000.0, 1e-3);
  EXPECT_EQ(utilization.metrics[1].metric, "ICI (Write)");
  EXPECT_DOUBLE_EQ(utilization.metrics[1].achieved, 640.0);
  EXPECT_NEAR(utilization.metrics[1].peak, 294000.0, 1e-3);
}

TEST(TpuGenericUtilizationUtilsTest, IciBandwidthTpuV7xPeakAndAchieved) {
  absl::flat_hash_map<uint64_t, uint64_t> counters_map = {
      {TpuCounterIdsTpu7x::
           VF_CHIP_DIE0_PWRMGR_PWRMGR_TC_THROTTLE_CORE_DEBUG_STATS_UNPRIVILEGED_CYCLE_COUNT,  // NOLINT
       1900},
      {TpuCounterIdsTpu7x::
           VF_CHIP_DIE1_OCI_ICR_0_PERF_COUNTERS_FIFO_STATS_COUNTERS_UNPRIVILEGED_BN_RD_REQ_0_INSERTION_COUNT,  // NOLINT
       10},
      {TpuCounterIdsTpu7x::
           VF_CHIP_DIE1_OCI_ICR_0_PERF_COUNTERS_FIFO_STATS_COUNTERS_UNPRIVILEGED_BN_WR_REQ_0_INSERTION_COUNT,  // NOLINT
       5},
  };
  TpuCounterUtil counters(/*host_id=*/0, /*device_id=*/0, /*correlation_id=*/0,
                          counters_map);
  TpuGenericUtilizationOptions options = {
      .num_mxu_per_tensor_core = 2,
      .cycles_per_xlu_instruction = 1,
      .is_tpu6e = false,
      .frequency_hz = 1.9e9,
      .peak_hbm_bw_bps = 1.0,
      .pstate_normalized_frequency_hz = 0.0,
  };

  UtilizationCounters utilization;
  ComputeTpuGenericIciBandwidthUtilization(counters, options, &utilization);

  ASSERT_THAT(utilization.metrics, SizeIs(2));
  EXPECT_EQ(utilization.metrics[0].metric, "ICI (Read)");
  EXPECT_DOUBLE_EQ(utilization.metrics[0].achieved, 1280.0);
  EXPECT_NEAR(utilization.metrics[0].peak, 560400.0, 1e-3);
  EXPECT_EQ(utilization.metrics[1].metric, "ICI (Write)");
  EXPECT_DOUBLE_EQ(utilization.metrics[1].achieved, 640.0);
  EXPECT_NEAR(utilization.metrics[1].peak, 560400.0, 1e-3);
}

TEST(TpuGenericUtilizationUtilsTest,
     ScUnitUtilizationSkipsZeroCycleSubsystems) {
  TpuCounterUtil empty_counters(/*host_id=*/0, /*device_id=*/0,
                                /*correlation_id=*/0, {});
  UtilizationCounters v6e_idle;
  ComputeTpuv6eScUnitUtilization(empty_counters, /*die=*/0, /*core=*/0,
                                 &v6e_idle);
  EXPECT_THAT(v6e_idle.metrics, IsEmpty());

  UtilizationCounters v7x_idle;
  ComputeTpuv7xScUnitUtilization(empty_counters, /*die=*/0, /*core=*/0,
                                 &v7x_idle);
  EXPECT_THAT(v7x_idle.metrics, IsEmpty());

  // Only SCS cycles non-zero: emits 4 SCS rows and skips all 12 TEC/SC rows.
  TpuCounterUtil v6e_scs_only(
      /*host_id=*/0, /*device_id=*/0, /*correlation_id=*/0,
      {{TpuCounterIdsTpu6e::
            VF_CHIP_SC_0_SCS_SC_STATS_COUNTERS_UNPRIVILEGED_COUNT_CYCLES,  // NOLINT
        100}});
  UtilizationCounters v6e_partial;
  ComputeTpuv6eScUnitUtilization(v6e_scs_only, /*die=*/0, /*core=*/0,
                                 &v6e_partial);
  EXPECT_THAT(v6e_partial.metrics, SizeIs(4));

  TpuCounterUtil v7x_scs_only(
      /*host_id=*/0, /*device_id=*/0, /*correlation_id=*/0,
      {{TpuCounterIdsTpu7x::
            VF_CHIP_DIE0_SC_0_SCS_SC_STATS_COUNTERS_UNPRIVILEGED_COUNT_CYCLES,  // NOLINT
        100}});
  UtilizationCounters v7x_partial;
  ComputeTpuv7xScUnitUtilization(v7x_scs_only, /*die=*/0, /*core=*/0,
                                 &v7x_partial);
  EXPECT_THAT(v7x_partial.metrics, SizeIs(4));
}

}  // namespace
}  // namespace xprof
