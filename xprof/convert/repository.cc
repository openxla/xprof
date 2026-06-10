/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "xprof/convert/repository.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "google/protobuf/arena.h"
#include "riegeli/bytes/string_reader.h"  // from @com_google_riegeli
#include "riegeli/records/record_reader.h"  // from @com_google_riegeli
#include "riegeli/records/skipped_region.h"  // from @com_google_riegeli
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/profiler/utils/file_system_utils.h"
#include "tsl/platform/path.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/file_utils.h"
#include "xprof/convert/unified_session_snapshot.h"
#include "xprof/utils/xplane_hlo_fixer.h"

namespace tensorflow {
namespace profiler {
namespace {

static auto* kHostDataSuffixes =
    new std::vector<std::pair<StoredDataType, const char*>>(
        {{StoredDataType::DCN_COLLECTIVE_STATS, ".dcn_collective_stats.pb"},
         {StoredDataType::OP_STATS, ".op_stats_v2.pb"},
         {StoredDataType::RIEGELI_XSPACE, ".xplane.riegeli"},
         {StoredDataType::SMART_SUGGESTION, ".smart_suggestion.pb"},
         {StoredDataType::TRACE_EVENTS_METADATA_LEVELDB, ".metadata.SSTABLE"},
         {StoredDataType::TRACE_EVENTS_PREFIX_TRIE_LEVELDB, ".trie.SSTABLE"},
         {StoredDataType::TRACE_LEVELDB, ".SSTABLE"}});

void MergeXSpaceTopLevelMetadata(const XSpace& from, XSpace* to) {
  absl::flat_hash_set<absl::string_view> existing_hostnames(
      to->hostnames().begin(), to->hostnames().end());
  for (const std::string& hostname : from.hostnames()) {
    if (existing_hostnames.insert(hostname).second) {
      to->add_hostnames(hostname);
    }
  }

  absl::flat_hash_set<absl::string_view> existing_errors(to->errors().begin(),
                                                         to->errors().end());
  for (const std::string& error : from.errors()) {
    if (existing_errors.insert(error).second) {
      to->add_errors(error);
    }
  }

  absl::flat_hash_set<absl::string_view> existing_warnings(
      to->warnings().begin(), to->warnings().end());
  for (const std::string& warning : from.warnings()) {
    if (existing_warnings.insert(warning).second) {
      to->add_warnings(warning);
    }
  }
}

template <typename MetadataMap>
void BuildSingleMetadataMap(const MetadataMap& from_metadata,
                            MetadataMap* to_metadata,
                            absl::flat_hash_map<int64_t, int64_t>* id_map) {
  absl::flat_hash_map<absl::string_view, int64_t> to_by_name;
  int64_t next_id = 0;
  // NOLINTNEXTLINE
  for (const auto& [id, metadata] : *to_metadata) {
    if (!metadata.name().empty()) {
      to_by_name[metadata.name()] = id;
    }
    next_id = std::max(next_id, id + 1);
  }
  // NOLINTNEXTLINE
  for (const auto& [from_id, from_meta] : from_metadata) {
    if (from_meta.name().empty()) {
      (*id_map)[from_id] = from_id;
      continue;
    }
    auto [it_meta, inserted] =
        to_by_name.try_emplace(from_meta.name(), next_id);
    if (inserted) {
      (*to_metadata)[next_id] = from_meta;
      next_id++;
    }
    (*id_map)[from_id] = it_meta->second;
  }
}

void BuildMetadataIdMaps(const XPlane& from_plane, XPlane* to_plane,
                         absl::flat_hash_map<int64_t, int64_t>* event_map,
                         absl::flat_hash_map<int64_t, int64_t>* stat_map) {
  BuildSingleMetadataMap(from_plane.event_metadata(),
                         to_plane->mutable_event_metadata(), event_map);
  BuildSingleMetadataMap(from_plane.stat_metadata(),
                         to_plane->mutable_stat_metadata(), stat_map);
}

void RemapEventMetadata(XEvent* event,
                        const absl::flat_hash_map<int64_t, int64_t>& event_map,
                        const absl::flat_hash_map<int64_t, int64_t>& stat_map) {
  auto it_remap = event_map.find(event->metadata_id());
  if (it_remap != event_map.end()) {
    event->set_metadata_id(it_remap->second);
  }

  for (XStat& stat : *event->mutable_stats()) {
    auto it_stat = stat_map.find(stat.metadata_id());
    if (it_stat != stat_map.end()) {
      stat.set_metadata_id(it_stat->second);
    }
  }
}

void MergeLine(std::unique_ptr<XLine> from_line, XLine* to_line,
               const absl::flat_hash_map<int64_t, int64_t>& event_map,
               const absl::flat_hash_map<int64_t, int64_t>& stat_map) {
  // Align timestamps
  const int64_t to_start = to_line->timestamp_ns();
  const int64_t from_start = from_line->timestamp_ns();
  if (from_start < to_start) {
    int64_t offset_ps = (to_start - from_start) * 1000;
    for (XEvent& event : *to_line->mutable_events()) {
      event.set_offset_ps(event.offset_ps() + offset_ps);
    }
    to_line->set_timestamp_ns(from_start);
  } else if (from_start > to_start) {
    int64_t offset_ps = (from_start - to_start) * 1000;
    for (XEvent& event : *from_line->mutable_events()) {
      event.set_offset_ps(event.offset_ps() + offset_ps);
    }
  }

  // Move events preserving chronological order by first releasing into a
  // temporary LIFO stack.
  std::vector<XEvent*> events_to_move;
  events_to_move.reserve(from_line->events_size());
  while (!from_line->events().empty()) {
    events_to_move.push_back(from_line->mutable_events()->ReleaseLast());
  }
  for (auto it = events_to_move.rbegin(); it != events_to_move.rend(); ++it) {
    XEvent* event = *it;
    RemapEventMetadata(event, event_map, stat_map);
    to_line->mutable_events()->AddAllocated(event);
  }
  to_line->set_duration_ps(
      std::max(to_line->duration_ps(), from_line->duration_ps()));
}

void MergePlaneLines(XPlane* from_plane, XPlane* to_plane,
                     const absl::flat_hash_map<int64_t, int64_t>& event_map,
                     const absl::flat_hash_map<int64_t, int64_t>& stat_map) {
  absl::flat_hash_map<int64_t, XLine*> to_lines;
  for (int i = 0; i < to_plane->lines_size(); ++i) {
    XLine* l = to_plane->mutable_lines(i);
    to_lines[l->id()] = l;
  }

  while (!from_plane->lines().empty()) {
    std::unique_ptr<XLine> from_line(
        from_plane->mutable_lines()->ReleaseLast());
    auto line_it = to_lines.find(from_line->id());

    if (line_it == to_lines.end()) {
      for (XEvent& event : *from_line->mutable_events()) {
        RemapEventMetadata(&event, event_map, stat_map);
      }
      to_plane->mutable_lines()->AddAllocated(from_line.release());
    } else {
      MergeLine(std::move(from_line), line_it->second, event_map, stat_map);
    }
  }
}

void MergePlane(std::unique_ptr<XPlane> from_plane, XPlane* to_plane) {
  absl::flat_hash_map<int64_t, int64_t> event_map;
  absl::flat_hash_map<int64_t, int64_t> stat_map;

  // Remap metadata IDs
  BuildMetadataIdMaps(*from_plane, to_plane, &event_map, &stat_map);

  // Merge lines and events
  MergePlaneLines(from_plane.get(), to_plane, event_map, stat_map);

  // Merge plane-level stats
  absl::flat_hash_set<int64_t> existing_plane_stats;
  for (const XStat& stat : to_plane->stats()) {
    existing_plane_stats.insert(stat.metadata_id());
  }
  for (const XStat& stat : from_plane->stats()) {
    auto it_stat_meta = stat_map.find(stat.metadata_id());
    int64_t remapped_id = (it_stat_meta != stat_map.end())
                              ? it_stat_meta->second
                              : stat.metadata_id();
    if (existing_plane_stats.insert(remapped_id).second) {
      auto* new_stat = to_plane->add_stats();
      *new_stat = stat;
      new_stat->set_metadata_id(remapped_id);
    }
  }
}

void MergeXSpace(std::unique_ptr<XSpace> from, XSpace* to) {
  if (!from) {
    return;
  }

  // 1. Merge top-level data
  MergeXSpaceTopLevelMetadata(*from, to);

  absl::flat_hash_map<absl::string_view, XPlane*> to_planes;
  for (int i = 0; i < to->planes_size(); ++i) {
    XPlane* p = to->mutable_planes(i);
    to_planes[p->name()] = p;
  }

  // 2. Process planes
  while (!from->planes().empty()) {
    std::unique_ptr<XPlane> from_plane(from->mutable_planes()->ReleaseLast());
    auto it = to_planes.find(from_plane->name());

    if (it == to_planes.end()) {
      to->mutable_planes()->AddAllocated(from_plane.release());
    } else {
      MergePlane(std::move(from_plane), it->second);
    }
  }
}

}  // namespace

std::string SessionSnapshot::GetHostnameByPath(absl::string_view xspace_path) {
  std::string_view file_name = tsl::io::Basename(xspace_path);
  absl::ConsumeSuffix(&file_name, ".pb");
  absl::ConsumeSuffix(&file_name, ".riegeli");
  if (!absl::ConsumeSuffix(&file_name, ".xplane.pb") &&
      !absl::ConsumeSuffix(&file_name, ".xplane.riegeli")) {
    absl::ConsumeSuffix(&file_name, ".xplane");
  }
  return std::string(file_name);
}

absl::StatusOr<SessionSnapshot> SessionSnapshot::Create(
    std::vector<std::string> xspace_paths,
    std::optional<std::vector<std::unique_ptr<XSpace>>> xspaces) {
  if (xspace_paths.empty()) {
    return absl::InvalidArgumentError("Can not find XSpace path.");
  }

  if (xspaces.has_value()) {
    if (xspaces->size() != xspace_paths.size()) {
      return absl::InvalidArgumentError(
          absl::StrCat("The size of the XSpace paths: ", xspace_paths.size(),
                       " is not equal ",
                       "to the size of the XSpace proto: ", xspaces->size()));
    }
    for (size_t i = 0; i < xspace_paths.size(); ++i) {
      auto host_name = GetHostnameByPath(xspace_paths.at(i));
      if (xspaces->at(i)->hostnames_size() > 0 && !host_name.empty()) {
        if (!absl::StrContains(host_name, xspaces->at(i)->hostnames(0))) {
          return absl::InvalidArgumentError(absl::StrCat(
              "The hostname of xspace path and preloaded xpace don't match at "
              "index: ",
              i, ". \nThe host name of xpace path is ", host_name,
              " but the host name of preloaded xpace is ",
              xspaces->at(i)->hostnames(0), "."));
        }
      }
    }
  }

  return SessionSnapshot(std::move(xspace_paths), std::move(xspaces));
}

absl::StatusOr<XSpace*> SessionSnapshot::GetXSpace(size_t index,
                                                   google::protobuf::Arena* arena) const {
  if (index >= xspace_paths_.size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Can not get the ", index, "th XSpace. The total number of XSpace is ",
        xspace_paths_.size()));
  }

  size_t idx = index;
  XSpace* merged_xspace = nullptr;

  // Return the pre-loaded XSpace proto.
  if (xspaces_.has_value()) {
    merged_xspace = xspaces_->at(idx).get();
    if (merged_xspace == nullptr) {
      return absl::InternalError("Preloaded XSpace is null");
    }
  } else {
    const std::string& path = xspace_paths_.at(idx);
    if (absl::EndsWith(path, ".riegeli")) {
      std::string contents;
      TF_RETURN_IF_ERROR(
          tsl::ReadFileToString(tsl::Env::Default(), path, &contents));

      riegeli::RecordReaderBase::Options reader_options;
      reader_options.set_recovery(
          [](const riegeli::SkippedRegion&, riegeli::RecordReaderBase&) {
            return true;
          });

      riegeli::RecordReader<riegeli::StringReader<>> reader{
          riegeli::StringReader<>(std::move(contents)), reader_options};
      XSpace current_record;
      while (reader.ReadRecord(current_record)) {
        if (merged_xspace == nullptr) {
          merged_xspace = google::protobuf::Arena::Create<XSpace>(arena);
          merged_xspace->Swap(&current_record);
        } else {
          auto chunk_up = std::make_unique<XSpace>();
          chunk_up->Swap(&current_record);
          MergeXSpace(std::move(chunk_up), merged_xspace);
        }
      }
      if (!reader.Close()) return reader.status();
      if (merged_xspace != nullptr) {
        xprof::FixHloMetadataInXSpace(merged_xspace);
      }
    } else {
      merged_xspace = google::protobuf::Arena::Create<XSpace>(arena);
      TF_RETURN_IF_ERROR(xprof::ReadBinaryProto(path, merged_xspace));
      xprof::FixHloMetadataInXSpace(merged_xspace);
    }
  }
  return merged_xspace;
}

absl::StatusOr<XSpace*> SessionSnapshot::GetXSpaceByName(
    absl::string_view name, google::protobuf::Arena* arena) const {
  if (auto it = hostname_map_.find(name);
      it != hostname_map_.end()) {
    return GetXSpace(it->second, arena);
  }

  return absl::InvalidArgumentError(
      absl::StrCat("Can not find the XSpace by name: ", name,
                   ". The total number of XSpace is ", xspace_paths_.size()));
}

std::string SessionSnapshot::GetHostname(size_t index) const {
  if (index >= xspace_paths_.size()) return "";
  return GetHostnameByPath(xspace_paths_.at(index));
}

std::optional<std::string> SessionSnapshot::GetFilePath(
    absl::string_view toolname, absl::string_view hostname) const {
  if (!has_accessible_run_dir_) return std::nullopt;
  std::optional<std::string> file_name = std::nullopt;
  if (toolname == "trace_viewer@")
    file_name = MakeHostDataFilePath(StoredDataType::TRACE_LEVELDB, hostname);
  return file_name;
}

std::optional<std::string> SessionSnapshot::MakeHostDataFilePath(
    const StoredDataType data_type, absl::string_view host) const {
  if (!has_accessible_run_dir_) return std::nullopt;
  auto filename = GetHostDataFileName(data_type, std::string(host));
  if (!filename.ok()) return std::nullopt;
  return tsl::io::JoinPath(session_run_dir_, *filename);
}

absl::StatusOr<std::string> SessionSnapshot::GetHostDataFileName(
    const StoredDataType data_type, absl::string_view host) const {
  for (const auto& format : *kHostDataSuffixes) {
    if (data_type == format.first) return absl::StrCat(host, format.second);
  }
  return absl::InternalError(&"Unknown StoredDataType: "[data_type]);
}

absl::StatusOr<std::optional<std::string>> SessionSnapshot::GetHostDataFilePath(
    const StoredDataType data_type, const std::string host) const {
  // Gets all the files in session run directory.
  std::vector<std::string> results;
  TF_RETURN_IF_ERROR(::tsl::Env::Default()->GetChildren(
      std::string(GetSessionRunDir()), &results));

  TF_ASSIGN_OR_RETURN(std::string filename,
                      GetHostDataFileName(data_type, host));

  for (const std::string& path : results) {
    if (absl::EndsWith(path, filename)) {
      return ::tsl::profiler::ProfilerJoinPath(GetSessionRunDir(), filename);
    }
  }

  return std::nullopt;
}

absl::StatusOr<std::pair<bool, std::string>> SessionSnapshot::HasCacheFile(
    const StoredDataType data_type) const {
  std::optional<std::string> filepath;
  TF_ASSIGN_OR_RETURN(filepath,
                      GetHostDataFilePath(data_type, kNoHostIdentifier));
  if (filepath) {
    // cache file is present but file contains no data_type events
    return std::pair<bool, std::string>(true, std::string());
  }

  TF_ASSIGN_OR_RETURN(filepath,
                      GetHostDataFilePath(data_type, kAllHostsIdentifier));
  if (filepath) {
    // cache file is present and file contains data_type events
    return std::pair<bool, std::string>(true, filepath.value());
  }

  // no cache file present
  return std::pair<bool, std::string>(false, std::string());
}

absl::Status SessionSnapshot::ClearCacheFiles() const {
  if (!has_accessible_run_dir_) return absl::OkStatus();

  // Delete all the cache files in session run directory for all cache types
  std::vector<std::string> results;
  TF_RETURN_IF_ERROR(::tsl::Env::Default()->GetChildren(
      std::string(GetSessionRunDir()), &results));

  for (const std::string& path : results) {
    std::string file_path = tsl::io::JoinPath(GetSessionRunDir(), path);
    for (const auto& format : *kHostDataSuffixes) {
      // FIX: Skip RIEGELI_XSPACE as it is source data, not cache.
      if (format.first == StoredDataType::RIEGELI_XSPACE) continue;

      if (absl::EndsWith(path, format.second)) {
        TF_RETURN_IF_ERROR(tsl::Env::Default()->DeleteFile(file_path));
        break;
      }
    }
  }

  return absl::OkStatus();
}

}  // namespace profiler
}  // namespace tensorflow
