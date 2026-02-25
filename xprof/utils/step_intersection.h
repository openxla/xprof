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

#ifndef XPROF_UTILS_STEP_INTERSECTION_H_
#define XPROF_UTILS_STEP_INTERSECTION_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "xla/tsl/platform/types.h"
#include "xla/tsl/profiler/utils/timespan.h"
#include "plugin/xprof/protobuf/steps_db.pb.h"

namespace tensorflow {
namespace profiler {

// Describes how two step sequences are aligned.
struct StepsAlignment {
  // The index in the subordinate step sequence where the alignment begins.
  uint32_t begin_subordinate_idx = 0;
  // The index in the chief step sequence where the alignment begins.
  uint32_t begin_chief_idx = 0;
  // The number of steps included in the alignment.
  uint32_t num_steps = 0;
};

// Intersects step sequences from multiple hosts by aligning them against a
// single "chief" host, which is chosen as the host with the shortest total
// step duration. The intersection includes the steps that overlap across all
// hosts.
class StepIntersection {
 public:
  StepIntersection(
      uint32_t max_steps,
      const absl::flat_hash_map</*host_id=*/uint32_t,
                                const StepDatabaseResult*>& perhost_stepdb);

  // Returns the number of steps in the intersection.
  uint32_t NumSteps() const { return end_chief_idx_ - begin_chief_idx_; }

  // Returns the value of empty_intersect_ (see the explanation of
  // empty_intersect_ below).
  bool EmptyIntersect() const { return empty_intersect_; }

  // Returns the step numbers for the destination (i.e. the intersection
  // result).
  std::vector<uint32_t> DstStepNumbers() const;

  // Returns the index to the step in the given host that corresponds to the
  // first step in the intersection.
  uint32_t FirstStepIndex(uint32_t host_id) const;

  // Returns the number of steps dropped due to the max_steps constraint
  // specified in the constructor.
  uint32_t StepsDropped() const { return steps_dropped_; }

  std::string DebugString() const;

 private:
  // Precompute all step timespans and find the chief host.
  struct HostInfo {
    uint32_t host_id = 0;
    const StepDatabaseResult* step_db = nullptr;
    uint32_t timespans_offset = 0;
    uint32_t timespans_count = 0;
    uint64_t total_duration_ps = 0;
  };

  void PrecomputeTimespansAndFindChief(
      const absl::flat_hash_map<uint32_t, const StepDatabaseResult*>&
          perhost_stepdb,
      std::vector<HostInfo>& hosts,
      std::vector<tsl::profiler::Timespan>& all_step_timespans,
      const HostInfo*& chief_host_info);

  void AlignStepsWithChief(
      const std::vector<HostInfo>& hosts,
      const std::vector<tsl::profiler::Timespan>& all_step_timespans,
      const HostInfo* chief_host_info, uint32_t& max_begin_chief_idx,
      uint32_t& min_end_chief_idx);

  void ComputeFinalBounds(uint32_t max_steps, uint32_t max_begin_chief_idx,
                          uint32_t min_end_chief_idx);

  absl::flat_hash_map</*host_id=*/uint32_t, StepsAlignment> perhost_alignment_;
  // The host whose step sequence is selected as the chief.
  uint32_t chief_host_id_;
  // Number of steps dropped.
  uint32_t steps_dropped_;
  // If NumSteps() is 0, empty_intersect indicates one of two possible reasons:
  //   (i) At least one host has some steps, but the intersection over all
  //   hosts is empty. In this case, empty_intersect is true.
  //   (ii) None of the hosts has any steps. In this case, empty_intersect is
  //   false.
  // If NumSteps() > 0, empty_intersect is don't care.
  bool empty_intersect_;
  // The begin and end indices to the chief step sequence for this step
  // intersection. Note that the begin index is inclusive but the end index is
  // exclusive.
  uint32_t begin_chief_idx_;
  uint32_t end_chief_idx_;
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // XPROF_UTILS_STEP_INTERSECTION_H_
