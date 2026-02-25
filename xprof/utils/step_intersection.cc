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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "xla/tsl/lib/gtl/map_util.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/profiler/utils/timespan.h"

namespace tensorflow {
namespace profiler {

namespace {

// Returns the timespan in this step (across all cores).
tsl::profiler::Timespan StepTimespan(const PerCoreStepInfo& percore_stepinfo) {
  const auto& step_info_per_core = percore_stepinfo.step_info_per_core();
  if (step_info_per_core.empty()) return tsl::profiler::Timespan();
  if (step_info_per_core.size() == 1) {
    const auto& stepinfo = step_info_per_core.begin()->second;
    return tsl::profiler::Timespan::FromEndPoints(
        stepinfo.begin_ps(), stepinfo.begin_ps() + stepinfo.duration_ps());
  }
  uint64_t min_ps = std::numeric_limits<uint64_t>::max();
  uint64_t max_ps = 0;
  for (const auto& [unused_core_id, stepinfo] : step_info_per_core) {
    uint64_t begin_ps = stepinfo.begin_ps();
    uint64_t end_ps = begin_ps + stepinfo.duration_ps();
    min_ps = std::min(min_ps, begin_ps);
    max_ps = std::max(max_ps, end_ps);
  }
  return (min_ps < max_ps)
             ? tsl::profiler::Timespan::FromEndPoints(min_ps, max_ps)
             : tsl::profiler::Timespan();
}

void CalculateSimilarities(
    const tsl::profiler::Timespan* subordinate_step_timespans,
    uint32_t subordinate_step_sequence_size,
    const tsl::profiler::Timespan* chief_step_timespans,
    uint32_t chief_step_sequence_size,
    std::vector<uint64_t>& offset_to_similarity) {
  // The offset k is defined as chief_idx - subordinate_idx.
  // The possible range of k is
  // [-(subordinate_step_sequence_size-1), chief_step_sequence_size-1].
  // We use offset_to_similarity to store similarity for each offset k.
  // The index in offset_to_similarity is calculated as:
  // k + (subordinate_step_sequence_size - 1), which maps k to indices
  // [0, subordinate_step_sequence_size + chief_step_sequence_size - 2].

  int subordinate_idx = 0;
  int chief_idx = 0;
  while (subordinate_idx < subordinate_step_sequence_size &&
         chief_idx < chief_step_sequence_size) {
    const tsl::profiler::Timespan& subordinate_timespan =
        subordinate_step_timespans[subordinate_idx];
    if (subordinate_timespan.duration_ps() == 0) {
      ++subordinate_idx;
      continue;
    }
    const tsl::profiler::Timespan& chief_timespan =
        chief_step_timespans[chief_idx];
    if (chief_timespan.duration_ps() == 0) {
      ++chief_idx;
      continue;
    }

    uint64_t subordinate_begin_ps = subordinate_timespan.begin_ps();
    uint64_t subordinate_end_ps = subordinate_timespan.end_ps();
    uint64_t chief_begin_ps = chief_timespan.begin_ps();
    uint64_t chief_end_ps = chief_timespan.end_ps();

    uint64_t start_ps = std::max(subordinate_begin_ps, chief_begin_ps);
    uint64_t end_ps = std::min(subordinate_end_ps, chief_end_ps);
    uint64_t overlap = (start_ps < end_ps) ? (end_ps - start_ps) : 0;
    offset_to_similarity[chief_idx - subordinate_idx +
                         (subordinate_step_sequence_size - 1)] += overlap;

    // Advance the index of the sequence that ends earlier to continue overlap
    // checking.
    if (subordinate_end_ps < chief_end_ps) {
      ++subordinate_idx;
    } else if (chief_end_ps < subordinate_end_ps) {
      ++chief_idx;
    } else {
      ++subordinate_idx;
      ++chief_idx;
    }
  }
}

int FindBestOffset(const std::vector<uint64_t>& offset_to_similarity,
                   uint32_t subordinate_step_sequence_size,
                   uint32_t chief_step_sequence_size) {
  uint64_t max_similarity = 0;
  int best_k = 0;
  bool found = false;

  // Finds offset k that has the maximum similarity score.
  // The original implementation checked k = 0, 1, ..., m-1 first, then
  // k = -1, -2, ..., -(n-1) for tie-breaking. We preserve that order.
  auto update_best = [&](int k) {
    uint64_t similarity =
        offset_to_similarity[k + (subordinate_step_sequence_size - 1)];
    if (!found || similarity > max_similarity) {
      max_similarity = similarity;
      best_k = k;
      found = true;
    }
  };

  for (int k = 0; k < chief_step_sequence_size; ++k) update_best(k);
  for (int k = -1; k >= -(static_cast<int>(subordinate_step_sequence_size) - 1);
       --k) {
    update_best(k);
  }
  return best_k;
}

StepsAlignment FindStepsAlignment(
    const StepDatabaseResult& subordinate,
    const tsl::profiler::Timespan* subordinate_step_timespans,
    const StepDatabaseResult& chief,
    const tsl::profiler::Timespan* chief_step_timespans,
    std::vector<uint64_t>& offset_to_similarity) {
  uint32_t subordinate_step_sequence_size = subordinate.step_sequence_size();
  uint32_t chief_step_sequence_size = chief.step_sequence_size();
  if (subordinate_step_sequence_size == 0 || chief_step_sequence_size == 0) {
    return {.begin_subordinate_idx = 0, .begin_chief_idx = 0, .num_steps = 0};
  }

  CalculateSimilarities(subordinate_step_timespans,
                        subordinate_step_sequence_size, chief_step_timespans,
                        chief_step_sequence_size, offset_to_similarity);

  int best_k = FindBestOffset(offset_to_similarity,
                              subordinate_step_sequence_size,
                              chief_step_sequence_size);

  uint32_t begin_subordinate_idx = best_k >= 0 ? 0 : -best_k;
  uint32_t begin_chief_idx = best_k >= 0 ? best_k : 0;
  uint32_t num_steps =
      std::min(subordinate_step_sequence_size - begin_subordinate_idx,
               chief_step_sequence_size - begin_chief_idx);

  return {.begin_subordinate_idx = begin_subordinate_idx,
          .begin_chief_idx = begin_chief_idx,
          .num_steps = num_steps};
}

std::string StringStepsAlignment(const StepsAlignment& alignment) {
  return absl::StrCat(
      "[begin_subordinate_idx: ", alignment.begin_subordinate_idx,
      ", begin_chief_idx: ", alignment.begin_chief_idx,
      ", num_steps: ", alignment.num_steps, "]");
}

std::string StringDstStepNumbers(const std::vector<uint32_t>& step_numbers) {
  std::string str;
  absl::StrAppend(&str, "[");
  for (int i = 0; i < step_numbers.size(); ++i) {
    if (i > 0) absl::StrAppend(&str, ", ");
    absl::StrAppend(&str, step_numbers[i]);
  }
  absl::StrAppend(&str, "]");
  return str;
}

std::string StringSrcToDstIndexMap(uint32_t src_first_step_idx,
                                   uint32_t num_steps) {
  std::string str;
  absl::StrAppend(&str, "[");
  for (int i = 0; i < num_steps; ++i) {
    if (i > 0) absl::StrAppend(&str, ", ");
    absl::StrAppend(&str, src_first_step_idx + i, ":", i);
  }
  absl::StrAppend(&str, "]");
  return str;
}

}  // namespace

void StepIntersection::PrecomputeTimespansAndFindChief(
    const absl::flat_hash_map<uint32_t, const StepDatabaseResult*>&
        perhost_stepdb,
    std::vector<StepIntersection::HostInfo>& hosts,
    std::vector<tsl::profiler::Timespan>& all_step_timespans,
    const StepIntersection::HostInfo*& chief_host_info) {
  size_t total_steps = 0;
  for (const auto& [host_id, step_db] : perhost_stepdb) {
    total_steps += step_db->step_sequence_size();
  }
  all_step_timespans.reserve(total_steps);

  uint64_t min_duration_ps = std::numeric_limits<uint64_t>::max();

  for (const auto& [host_id, step_db] : perhost_stepdb) {
    hosts.emplace_back();
    HostInfo& info = hosts.back();
    info.host_id = host_id;
    info.step_db = step_db;
    info.timespans_offset = all_step_timespans.size();

    uint64_t min_ps = std::numeric_limits<uint64_t>::max();
    uint64_t max_ps = 0;
    for (const PerCoreStepInfo& step : step_db->step_sequence()) {
      tsl::profiler::Timespan timespan = StepTimespan(step);
      all_step_timespans.push_back(timespan);
      if (timespan.duration_ps() > 0) {
        min_ps = std::min(min_ps, timespan.begin_ps());
        max_ps = std::max(max_ps, timespan.end_ps());
      }
    }
    info.timespans_count = all_step_timespans.size() - info.timespans_offset;
    info.total_duration_ps = (min_ps < max_ps) ? (max_ps - min_ps) : 0;
    if (chief_host_info == nullptr ||
        info.total_duration_ps < min_duration_ps) {
      min_duration_ps = info.total_duration_ps;
      chief_host_info = &info;
    }
  }
}

void StepIntersection::AlignStepsWithChief(
    const std::vector<StepIntersection::HostInfo>& hosts,
    const std::vector<tsl::profiler::Timespan>& all_step_timespans,
    const StepIntersection::HostInfo* chief_host_info,
    uint32_t& max_begin_chief_idx, uint32_t& min_end_chief_idx) {
  std::vector<uint64_t> offset_to_similarity;
  uint32_t chief_timespans_count = chief_host_info->timespans_count;
  const tsl::profiler::Timespan* chief_ptr =
      all_step_timespans.data() + chief_host_info->timespans_offset;

  for (const HostInfo& info : hosts) {
    StepsAlignment& alignment = perhost_alignment_[info.host_id];
    if (info.host_id == chief_host_id_) {
      alignment = {.begin_subordinate_idx = 0,
                   .begin_chief_idx = 0,
                   .num_steps = static_cast<uint32_t>(info.timespans_count)};
    } else {
      uint32_t subordinate_timespans_count = info.timespans_count;
      offset_to_similarity.assign(
          subordinate_timespans_count + chief_timespans_count - 1, 0);
      alignment = FindStepsAlignment(
          *info.step_db, all_step_timespans.data() + info.timespans_offset,
          *chief_host_info->step_db, chief_ptr, offset_to_similarity);
    }
    max_begin_chief_idx =
        std::max(max_begin_chief_idx, alignment.begin_chief_idx);
    min_end_chief_idx = std::min(
        min_end_chief_idx, alignment.begin_chief_idx + alignment.num_steps);
  }
}

void StepIntersection::ComputeFinalBounds(uint32_t max_steps,
                                          uint32_t max_begin_chief_idx,
                                          uint32_t min_end_chief_idx) {
  if (max_begin_chief_idx > min_end_chief_idx) {
    steps_dropped_ = 0;
    begin_chief_idx_ = 0;
    end_chief_idx_ = 0;
    empty_intersect_ = true;
    return;
  }

  begin_chief_idx_ = max_begin_chief_idx;
  uint32_t num_steps = min_end_chief_idx - max_begin_chief_idx;
  if (num_steps > max_steps) {
    steps_dropped_ = num_steps - max_steps;
    end_chief_idx_ = max_begin_chief_idx + max_steps;
  } else {
    steps_dropped_ = 0;
    end_chief_idx_ = min_end_chief_idx;
  }
}

StepIntersection::StepIntersection(
    uint32_t max_steps,
    const absl::flat_hash_map<uint32_t, const StepDatabaseResult*>&
        perhost_stepdb) {
  empty_intersect_ = false;

  if (perhost_stepdb.empty()) {
    steps_dropped_ = 0;
    begin_chief_idx_ = 0;
    end_chief_idx_ = 0;
    chief_host_id_ = std::numeric_limits<uint32_t>::max();
    return;
  }

  std::vector<StepIntersection::HostInfo> hosts;
  hosts.reserve(perhost_stepdb.size());
  std::vector<tsl::profiler::Timespan> all_step_timespans;
  const StepIntersection::HostInfo* chief_host_info = nullptr;

  PrecomputeTimespansAndFindChief(perhost_stepdb, hosts, all_step_timespans,
                                  chief_host_info);

  if (chief_host_info == nullptr || chief_host_info->timespans_count == 0) {
    steps_dropped_ = 0;
    begin_chief_idx_ = 0;
    end_chief_idx_ = 0;
    chief_host_id_ = (chief_host_info != nullptr)
                         ? chief_host_info->host_id
                         : std::numeric_limits<uint32_t>::max();
    return;
  }
  chief_host_id_ = chief_host_info->host_id;

  uint32_t max_begin_chief_idx = 0;
  uint32_t min_end_chief_idx = std::numeric_limits<uint32_t>::max();

  AlignStepsWithChief(hosts, all_step_timespans, chief_host_info,
                      max_begin_chief_idx, min_end_chief_idx);

  ComputeFinalBounds(max_steps, max_begin_chief_idx, min_end_chief_idx);
}

std::vector<uint32_t> StepIntersection::DstStepNumbers() const {
  // TODO(ckluk): Honors training-loop boundaries (if more than one loop
  // sampled).
  std::vector<uint32_t> result(NumSteps());
  std::iota(result.begin(), result.end(), 0);
  return result;
}

uint32_t StepIntersection::FirstStepIndex(uint32_t host_id) const {
  const StepsAlignment* alignment =
      tsl::gtl::FindOrNull(perhost_alignment_, host_id);
  if (alignment == nullptr) return 0;
  DCHECK(alignment->begin_chief_idx <= begin_chief_idx_);
  uint32_t shift = begin_chief_idx_ - alignment->begin_chief_idx;
  uint32_t begin_subordinate_idx = alignment->begin_subordinate_idx + shift;
  return begin_subordinate_idx;
}

std::string StepIntersection::DebugString() const {
  std::string debug_string;
  absl::StrAppend(&debug_string, "chief host id_: ", chief_host_id_, "\n");
  absl::StrAppend(&debug_string, "begin_chief_idx_: ", begin_chief_idx_,
                  ", num_steps: ", NumSteps(), "\n");
  absl::StrAppend(&debug_string,
                  "DstStepNumbers(): ", StringDstStepNumbers(DstStepNumbers()),
                  "\n");

  std::vector<uint32_t> host_ids;
  host_ids.reserve(perhost_alignment_.size());
  for (const auto& [host_id, unused_alignment] : perhost_alignment_) {
    host_ids.push_back(host_id);
  }
  absl::c_sort(host_ids);

  absl::StrAppend(&debug_string, "perhost_alignment:\n");
  for (const uint32_t host_id : host_ids) {
    const StepsAlignment* ptr =
        tsl::gtl::FindOrNull(perhost_alignment_, host_id);
    if (ptr == nullptr) continue;
    absl::StrAppend(&debug_string, "host: ", host_id,
                    ", step-alignment: ", StringStepsAlignment(*ptr), "\n");
  }
  absl::StrAppend(&debug_string, "SrcToDstIndexMap():\n");
  for (const uint32_t host_id : host_ids) {
    absl::StrAppend(
        &debug_string, "host: ", host_id, ", src-to-dst-index-map: ",
        StringSrcToDstIndexMap(FirstStepIndex(host_id), NumSteps()), "\n");
  }
  return debug_string;
}

}  // namespace profiler
}  // namespace tensorflow
