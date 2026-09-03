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

#include "xprof/utils/xplane_hlo_fixer.h"

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/numbers.h"
#include "absl/strings/string_view.h"
#include "re2/re2.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xprof {
namespace {
// We use "HLO Proto" for the legacy HLO Proto stat name.
constexpr absl::string_view kHloStatNameLegacy = "HLO Proto";
// Matches the program ID in parentheses at the end of the name.
static const RE2* kProgIdRegex = new RE2(".*\\((-?\\d+)\\)");

// Returns the stat ID for HLO Proto. If legacy name "HLO Proto" is found,
// it is renamed to the expected name.
std::optional<int64_t> GetHloStatId(tensorflow::profiler::XPlane* plane) {
  std::string hlo_stat_name_expected =
      std::string(tsl::profiler::GetStatTypeStr(tsl::profiler::kHloProto));
  std::optional<int64_t> expected_id;

  for (auto& kv : *plane->mutable_stat_metadata()) {
    if (kv.second.name() == kHloStatNameLegacy) {
      LOG(INFO) << "Found legacy HLO Proto stat, renaming to "
                << hlo_stat_name_expected;
      kv.second.set_name(hlo_stat_name_expected);
      return kv.second.id();
    } else if (kv.second.name() == hlo_stat_name_expected) {
      expected_id = kv.second.id();
    }
  }

  return expected_id;
}

// Fixes event metadata IDs to match program IDs for events with HLO stats.
void FixHloEventMetadataIds(tensorflow::profiler::XPlane* plane,
                            int64_t hlo_stat_id) {
  std::vector<int64_t> event_metadata_ids_to_delete;
  std::map<int64_t, tensorflow::profiler::XEventMetadata>
      program_id_to_event_meta;

  for (auto& kv : *plane->mutable_event_metadata()) {
    auto& event_meta = kv.second;
    bool has_stat = false;
    for (const auto& stat : event_meta.stats()) {
      if (stat.metadata_id() == hlo_stat_id) {
        has_stat = true;
        break;
      }
    }

    if (has_stat) {
      absl::string_view program_id_str;
      if (RE2::FullMatch(event_meta.name(), *kProgIdRegex, &program_id_str)) {
        int64_t program_id = 0;
        if (!absl::SimpleAtoi(program_id_str, &program_id)) {
          uint64_t uprogram_id = 0;
          if (absl::SimpleAtoi(program_id_str, &uprogram_id)) {
            program_id = static_cast<int64_t>(uprogram_id);
          } else {
            continue;
          }
        }
        if (event_meta.id() != program_id) {
          VLOG(1) << "Updating event metadata ID for " << event_meta.name()
                  << " from " << event_meta.id() << " to " << program_id;
          program_id_to_event_meta[program_id] = std::move(event_meta);
          event_metadata_ids_to_delete.push_back(kv.first);
        }
      }
    }
  }

  auto* mutable_event_metadata = plane->mutable_event_metadata();
  for (int64_t key : event_metadata_ids_to_delete) {
    mutable_event_metadata->erase(key);
  }

  for (auto& kv : program_id_to_event_meta) {
    int64_t prog_id = kv.first;
    auto& new_meta = kv.second;
    new_meta.set_id(prog_id);
    (*mutable_event_metadata)[prog_id] = std::move(new_meta);
  }
}
}  // namespace

// Fixes XSpace data by updating HLO Proto metadata and IDs.
// This proprocess script is targeting specific trace data created by
// libtpu=0.0.37. Assumption to identify problematic data is:
// 1. XSpace has a plane with name "metadata".
// 2. The plane has a stat with name "HLO Proto".
void FixHloMetadataInXSpace(tensorflow::profiler::XSpace* space) {
  if (space == nullptr) return;

  for (auto& plane : *space->mutable_planes()) {
    if (plane.name() == tsl::profiler::kMetadataPlaneName) {
      std::optional<int64_t> hlo_stat_id = GetHloStatId(&plane);

      if (hlo_stat_id.has_value()) {
        LOG(INFO) << "XProf HLO fixer triggered for metadata plane.";
        FixHloEventMetadataIds(&plane, *hlo_stat_id);
      }
    }
  }
}
}  // namespace xprof
