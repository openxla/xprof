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

#include "xprof/utils/step_intersection.h"

#include <cstdint>
#include <vector>

#include "testing/base/public/benchmark.h"
#include "testing/base/public/gmock.h"
#include "<gtest/gtest.h>"
#include "absl/container/flat_hash_map.h"

namespace tensorflow {
namespace profiler {
namespace {

using PerHostStepDb =
    absl::flat_hash_map<uint32_t /*=host_id*/, StepDatabaseResult>;
using ::testing::AllOf;
using ::testing::HasSubstr;

constexpr uint64_t kStepDurationPs = 2000000000;
constexpr uint32_t kNumStepsPerHost = 10;
constexpr uint64_t kStepGapPs = 0;
constexpr uint32_t kNumCoresPerHost = 8;

PerCoreStepInfo CreateOneTestStep(uint32_t host_id, uint32_t num_steps,
                                  uint32_t step_idx, uint64_t step_begin_ps) {
  PerCoreStepInfo result;
  uint32_t step_num =
      step_idx * host_id;  // creates the situation where each host has a
                           // different step number for the same step.
  result.set_step_num(step_num);
  StepInfoResult info;
  info.set_step_num(step_num);
  if (host_id == 0 && step_idx == (num_steps - 1)) {
    // Makes the last step on host_id is little bit shorter so that host-0 will
    // be chosen as the chief.
    info.set_duration_ps(kStepDurationPs - 1);
  } else {
    info.set_duration_ps(kStepDurationPs);
  }
  info.set_begin_ps(step_begin_ps);
  // Don't care about the rest of the fields in StepInfoResult.
  for (uint32_t core_id = 0; core_id < kNumCoresPerHost; ++core_id) {
    (*result.mutable_step_info_per_core())[core_id] = info;
    // Don't care about the rest of the fields in PerCoreStepInfo.
  }
  return result;
}

StepDatabaseResult CreateStepDatabaseResultForHost(
    uint32_t host_id, uint32_t num_steps, uint64_t first_step_begin_ps,
    uint64_t step_gap_ps = kStepGapPs) {
  StepDatabaseResult step_db;
  uint64_t step_begin_ps = first_step_begin_ps;
  for (uint32_t step_idx = 0; step_idx < num_steps; ++step_idx) {
    *step_db.add_step_sequence() =
        CreateOneTestStep(host_id, num_steps, step_idx, step_begin_ps);
    step_begin_ps += kStepDurationPs + step_gap_ps;
  }
  return step_db;
}

PerHostStepDb CreateTestSteps(uint32_t num_hosts, uint32_t num_steps_per_host,
                              uint64_t shift_ps) {
  PerHostStepDb result;
  uint64_t first_step_begin_ps = 0;
  for (uint32_t host_id = 0; host_id < num_hosts; ++host_id) {
    result[host_id] = CreateStepDatabaseResultForHost(
        host_id, num_steps_per_host, first_step_begin_ps);
    first_step_begin_ps += shift_ps;
  }
  return result;
}

PerHostStepDb CreateEmptyIntersectTestSteps() {
  PerHostStepDb result;

  uint64_t host_0_num_steps = 10;
  result[0] = CreateStepDatabaseResultForHost(0, host_0_num_steps, 0);

  uint64_t host_1_num_steps = 5;
  result[1] = CreateStepDatabaseResultForHost(
      1, host_1_num_steps,
      (host_0_num_steps - 2) * (kStepDurationPs + kStepGapPs));

  uint64_t host_2_num_steps = 10;
  result[2] = CreateStepDatabaseResultForHost(
      2, host_2_num_steps,
      (host_0_num_steps + host_1_num_steps - 4) *
          (kStepDurationPs + kStepGapPs));
  return result;
}

PerHostStepDb CreateNoStep(uint32_t num_hosts) {
  PerHostStepDb result;
  for (uint32_t host_id = 0; host_id < num_hosts; ++host_id) {
    StepDatabaseResult step_db;
    result[host_id] = step_db;
  }
  return result;
}

absl::flat_hash_map<uint32_t /*=host_id*/, const StepDatabaseResult*> Convert(
    const PerHostStepDb& perhost_stepdb) {
  absl::flat_hash_map<uint32_t /*=host_id*/, const StepDatabaseResult*> result;
  for (const auto& hostid_stepdb : perhost_stepdb) {
    auto host_id = hostid_stepdb.first;
    const auto& step_db = hostid_stepdb.second;
    result[host_id] = &step_db;
  }
  return result;
}

TEST(StepIntersectionTest, EachHostShiftedBy1StepDuration) {
  uint32_t num_hosts = 4;
  uint64_t shift_ps = kStepDurationPs;

  PerHostStepDb perhost_stepdb =
      CreateTestSteps(num_hosts, kNumStepsPerHost, shift_ps);
  StepIntersection intersection =
      StepIntersection(kNumStepsPerHost, Convert(perhost_stepdb));
  EXPECT_EQ(intersection.StepsDropped(), 0);
  uint32_t dst_num_steps = kNumStepsPerHost - num_hosts + 1;
  EXPECT_EQ(intersection.NumSteps(), dst_num_steps);

  uint32_t src_first_step_index = intersection.FirstStepIndex(0);
  EXPECT_EQ(src_first_step_index, num_hosts - 1);
  std::vector<uint32_t> dst_step_numbers = intersection.DstStepNumbers();
  for (uint32_t i = 0; i < dst_num_steps; ++i) {
    EXPECT_EQ(dst_step_numbers[i], i);
  }
}

TEST(StepIntersectionTest, ExactlyNoShift) {
  uint32_t num_hosts = 4;
  uint64_t shift_ps = 0;

  PerHostStepDb perhost_stepdb =
      CreateTestSteps(num_hosts, kNumStepsPerHost, shift_ps);
  StepIntersection intersection =
      StepIntersection(kNumStepsPerHost, Convert(perhost_stepdb));
  EXPECT_EQ(intersection.StepsDropped(), 0);
  uint32_t dst_num_steps = kNumStepsPerHost;
  EXPECT_EQ(intersection.NumSteps(), dst_num_steps);

  std::vector<uint32_t> dst_step_numbers = intersection.DstStepNumbers();
  for (uint32_t i = 0; i < dst_num_steps; ++i) {
    EXPECT_EQ(dst_step_numbers[i], i);
  }
  for (uint32_t host_id = 0; host_id < num_hosts; ++host_id) {
    uint32_t src_first_step_index = intersection.FirstStepIndex(host_id);
    EXPECT_EQ(src_first_step_index, 0);
  }
}

TEST(StepIntersectionTest, EachHostShiftedByJustABit) {
  uint32_t num_hosts = 4;
  uint64_t shift_ps = 100;

  PerHostStepDb perhost_stepdb =
      CreateTestSteps(num_hosts, kNumStepsPerHost, shift_ps);
  StepIntersection intersection =
      StepIntersection(kNumStepsPerHost, Convert(perhost_stepdb));
  EXPECT_EQ(intersection.StepsDropped(), 0);
  uint32_t dst_num_steps = kNumStepsPerHost;
  EXPECT_EQ(intersection.NumSteps(), dst_num_steps);

  std::vector<uint32_t> dst_step_numbers = intersection.DstStepNumbers();
  for (uint32_t i = 0; i < dst_num_steps; ++i) {
    EXPECT_EQ(dst_step_numbers[i], i);
  }
  for (uint32_t host_id = 0; host_id < num_hosts; ++host_id) {
    uint32_t src_first_step_index = intersection.FirstStepIndex(host_id);
    EXPECT_EQ(src_first_step_index, 0);
  }
}

TEST(StepIntersectionTest, SingleHost) {
  uint32_t num_hosts = 1;
  uint64_t shift_ps = 0;

  PerHostStepDb perhost_stepdb =
      CreateTestSteps(num_hosts, kNumStepsPerHost, shift_ps);
  StepIntersection intersection =
      StepIntersection(kNumStepsPerHost, Convert(perhost_stepdb));
  EXPECT_EQ(intersection.StepsDropped(), 0);
  uint32_t dst_num_steps = kNumStepsPerHost;
  EXPECT_EQ(intersection.NumSteps(), dst_num_steps);

  std::vector<uint32_t> dst_step_numbers = intersection.DstStepNumbers();
  for (uint32_t i = 0; i < dst_num_steps; ++i) {
    EXPECT_EQ(dst_step_numbers[i], i);
  }
  for (uint32_t host_id = 0; host_id < num_hosts; ++host_id) {
    uint32_t src_first_step_index = intersection.FirstStepIndex(host_id);
    EXPECT_EQ(src_first_step_index, 0);
  }
}

TEST(StepIntersectionTest, WithMaxSteps) {
  uint32_t num_hosts = 4;
  uint64_t shift_ps = 0;
  uint32_t max_steps = 3;

  PerHostStepDb perhost_stepdb =
      CreateTestSteps(num_hosts, kNumStepsPerHost, shift_ps);
  StepIntersection intersection =
      StepIntersection(max_steps, Convert(perhost_stepdb));
  EXPECT_EQ(intersection.StepsDropped(), kNumStepsPerHost - max_steps);
  EXPECT_EQ(intersection.NumSteps(), max_steps);
}

TEST(StepIntersectionTest, NoStep) {
  uint32_t num_hosts = 4;
  uint32_t max_steps = 100;
  PerHostStepDb perhost_stepdb = CreateNoStep(num_hosts);
  StepIntersection intersection =
      StepIntersection(max_steps, Convert(perhost_stepdb));
  EXPECT_EQ(intersection.NumSteps(), 0);
  EXPECT_FALSE(intersection.EmptyIntersect());
}

TEST(StepIntersectionTest, EmptyIntersection) {
  uint32_t max_steps = 100;
  PerHostStepDb perhost_stepdb = CreateEmptyIntersectTestSteps();
  StepIntersection intersection =
      StepIntersection(max_steps, Convert(perhost_stepdb));
  EXPECT_EQ(intersection.StepsDropped(), 0);
  EXPECT_EQ(intersection.NumSteps(), 0);
  EXPECT_TRUE(intersection.EmptyIntersect());
}

TEST(StepIntersectionTest, UnequalStepCounts) {
  PerHostStepDb perhost_stepdb;
  // Host 0 has 5 steps.
  perhost_stepdb[0] =
      CreateStepDatabaseResultForHost(/*host_id=*/0, /*num_steps=*/5, 0);

  // Host 1 has 10 steps, starting at the same time.
  perhost_stepdb[1] =
      CreateStepDatabaseResultForHost(/*host_id=*/1, /*num_steps=*/10, 0);

  StepIntersection intersection(kNumStepsPerHost, Convert(perhost_stepdb));
  EXPECT_EQ(intersection.StepsDropped(), 0);
  // Should intersect on the common 5 steps.
  EXPECT_EQ(intersection.NumSteps(), 5);
}

TEST(StepIntersectionTest, OneHostEmpty) {
  PerHostStepDb perhost_stepdb;
  // Host 0 has 5 steps.
  perhost_stepdb[0] =
      CreateStepDatabaseResultForHost(/*host_id=*/0, /*num_steps=*/5, 0);

  // Host 1 has 0 steps.
  StepDatabaseResult step_db_1;
  perhost_stepdb[1] = step_db_1;

  StepIntersection intersection(kNumStepsPerHost, Convert(perhost_stepdb));
  // If one host is empty, the resulting intersection has 0 steps.
  // Consistent with NoStep test, EmptyIntersect() remains false as it's not a
  // disjoint set but rather a zero-length valid intersection.
  EXPECT_EQ(intersection.NumSteps(), 0);
  EXPECT_FALSE(intersection.EmptyIntersect());
}

TEST(StepIntersectionTest, VaryingStepDurations) {
  PerHostStepDb perhost_stepdb;

  // Host 0: Steps of duration 2 * kStepDurationPs
  uint64_t step_begin_ps = 0;
  StepDatabaseResult step_db_0;
  for (uint32_t i = 0; i < 5; ++i) {
    PerCoreStepInfo step =
        CreateOneTestStep(/*host_id=*/0, /*num_steps=*/5, i, step_begin_ps);
    // Manually adjust duration for all cores
    for (auto& core_step : *step.mutable_step_info_per_core()) {
      core_step.second.set_duration_ps(2 * kStepDurationPs);
    }
    *step_db_0.add_step_sequence() = step;
    step_begin_ps += 2 * kStepDurationPs;
  }
  perhost_stepdb[0] = step_db_0;

  // Host 1: Standard duration steps, twice as many to cover the same time range
  perhost_stepdb[1] =
      CreateStepDatabaseResultForHost(/*host_id=*/1, /*num_steps=*/10, 0);

  StepIntersection intersection(kNumStepsPerHost, Convert(perhost_stepdb));
  // Intersection logic might align based on overlap.
  // With precise alignment, 5 long steps vs 10 short steps covering same span.
  // The greedy alignment might pick up some overlap.
  // We expect non-zero intersection if alignment works.
  EXPECT_GT(intersection.NumSteps(), 0);
}

TEST(StepIntersectionTest, GapsBetweenSteps) {
  PerHostStepDb perhost_stepdb;

  // Host 0: 5 steps with gaps
  perhost_stepdb[0] = CreateStepDatabaseResultForHost(
      /*host_id=*/0, /*num_steps=*/5, 0, /*step_gap_ps=*/kStepDurationPs);

  // Host 1: 5 steps with SAME gaps
  perhost_stepdb[1] = CreateStepDatabaseResultForHost(
      /*host_id=*/1, /*num_steps=*/5, 0, /*step_gap_ps=*/kStepDurationPs);

  StepIntersection intersection(kNumStepsPerHost, Convert(perhost_stepdb));
  EXPECT_EQ(intersection.StepsDropped(), 0);
  EXPECT_EQ(intersection.NumSteps(), 5);
}

void BM_StepIntersection(benchmark::State& state) {
  uint32_t num_hosts = state.range(0);
  uint32_t num_steps = state.range(1);
  constexpr uint64_t kShiftPs = 100;
  PerHostStepDb perhost_stepdb =
      CreateTestSteps(num_hosts, num_steps, kShiftPs);
  absl::flat_hash_map<uint32_t, const StepDatabaseResult*> perhost_stepdb_ptr =
      Convert(perhost_stepdb);
  for (auto s : state) {
    StepIntersection intersection(num_steps, perhost_stepdb_ptr);
    benchmark::DoNotOptimize(intersection);
  }
}
BENCHMARK(BM_StepIntersection)
    ->Args({16, 100})
    ->Args({64, 2000})
    ->Args({128, 5000});

TEST(StepIntersectionTest, EmptyPerHostStepDb) {
  uint32_t max_steps = 10;
  PerHostStepDb perhost_stepdb;
  StepIntersection intersection(max_steps, Convert(perhost_stepdb));
  EXPECT_EQ(intersection.NumSteps(), 0);
  EXPECT_EQ(intersection.StepsDropped(), 0);
  EXPECT_FALSE(intersection.EmptyIntersect());
}

TEST(StepIntersectionTest, StepWithStepsButNoCores) {
  uint32_t max_steps = 10;
  PerHostStepDb perhost_stepdb;
  StepDatabaseResult step_db;
  PerCoreStepInfo* step_info = step_db.add_step_sequence();
  step_info->set_step_num(0);
  // No per-core info added, so StepTimespan returns empty.

  perhost_stepdb[0] = step_db;

  StepIntersection intersection(max_steps, Convert(perhost_stepdb));
  EXPECT_EQ(intersection.NumSteps(), 1);
  EXPECT_EQ(intersection.StepsDropped(), 0);
  // Even with empty timespan, it aligns with itself.
}

TEST(StepIntersectionTest, DebugString) {
  uint32_t num_hosts = 2;
  uint32_t num_steps = 2;
  PerHostStepDb perhost_stepdb = CreateTestSteps(num_hosts, num_steps, 0);
  StepIntersection intersection(num_steps, Convert(perhost_stepdb));
  EXPECT_THAT(
      intersection.DebugString(),
      AllOf(HasSubstr("chief host id_:"), HasSubstr("begin_chief_idx_:"),
            HasSubstr("DstStepNumbers():"), HasSubstr("perhost_alignment:"),
            HasSubstr("step-alignment:"), HasSubstr("SrcToDstIndexMap():"),
            HasSubstr("src-to-dst-index-map:")));
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
