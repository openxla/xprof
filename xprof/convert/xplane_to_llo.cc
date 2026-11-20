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

#include "xprof/convert/xplane_to_llo.h"

#include <cstdint>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/arena.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/profiler/utils/xplane_builder.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_utils.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/repository.h"
#include "xprof/convert/unified_session_snapshot.h"

namespace tensorflow {
namespace profiler {
namespace {

// Returns the id of the `StatType::kLloProto` stat metadata of `plane`, or -1
// if the plane has none.
int64_t FindLloProtoStatMetadataId(const XPlane& plane) {
  const absl::string_view llo_proto_stat_name =
      tsl::profiler::GetStatTypeStr(tsl::profiler::StatType::kLloProto);
  for (const auto& [id, stat_metadata] : plane.stat_metadata()) {
    if (stat_metadata.name() == llo_proto_stat_name) {
      return id;
    }
  }
  return -1;
}

// Returns whether `event_metadata` carries an LLO module.
bool HasLloProtoStat(const XEventMetadata& event_metadata,
                     int64_t llo_proto_stat_id) {
  for (const XStat& stat : event_metadata.stats()) {
    if (stat.metadata_id() == llo_proto_stat_id &&
        !stat.bytes_value().empty()) {
      return true;
    }
  }
  return false;
}

}  // namespace

bool HasLloProtoMetadata(const XSpace& xspace) {
  const std::vector<const XPlane*> planes = tsl::profiler::FindPlanesWithNames(
      xspace, {tsl::profiler::kMetadataPlaneName});
  for (const XPlane* plane : planes) {
    if (plane == nullptr) continue;
    const int64_t llo_proto_stat_id = FindLloProtoStatMetadataId(*plane);
    if (llo_proto_stat_id < 0) continue;
    for (const auto& [id, event_metadata] : plane->event_metadata()) {
      if (HasLloProtoStat(event_metadata, llo_proto_stat_id)) return true;
    }
  }
  return false;
}

int ExtractLloXPlane(const XSpace& xspace, XPlane* llo_plane) {
  if (llo_plane == nullptr) return 0;
  llo_plane->Clear();

  const std::vector<const XPlane*> planes = tsl::profiler::FindPlanesWithNames(
      xspace, {tsl::profiler::kMetadataPlaneName});
  if (planes.empty()) return 0;

  tsl::profiler::XPlaneBuilder builder(llo_plane);
  const XStatMetadata* dst_llo_stat_metadata = nullptr;
  int num_modules = 0;

  for (const XPlane* plane : planes) {
    if (plane == nullptr) continue;
    const int64_t src_llo_stat_id = FindLloProtoStatMetadataId(*plane);
    if (src_llo_stat_id < 0) continue;

    std::vector<int64_t> event_ids;
    event_ids.reserve(plane->event_metadata_size());
    for (const auto& [id, event_metadata] : plane->event_metadata()) {
      if (HasLloProtoStat(event_metadata, src_llo_stat_id)) {
        event_ids.push_back(id);
      }
    }
    absl::c_sort(event_ids);

    for (const int64_t id : event_ids) {
      const XEventMetadata& event_metadata = plane->event_metadata().at(id);
      if (dst_llo_stat_metadata == nullptr) {
        builder.SetName(tsl::profiler::kMetadataPlaneName);
        dst_llo_stat_metadata = builder.GetOrCreateStatMetadata(
            tsl::profiler::GetStatTypeStr(tsl::profiler::StatType::kLloProto));
      }
      XEventMetadata* copy = builder.CreateEventMetadata();
      copy->set_name(event_metadata.name());
      copy->set_display_name(event_metadata.display_name());
      for (const XStat& stat : event_metadata.stats()) {
        if (stat.metadata_id() == src_llo_stat_id &&
            !stat.bytes_value().empty()) {
          XStat* dst_stat = copy->add_stats();
          dst_stat->set_metadata_id(dst_llo_stat_metadata->id());
          dst_stat->set_bytes_value(stat.bytes_value());
        }
      }
      ++num_modules;
    }
  }
  return num_modules;
}

absl::Status MaybeCacheLloXPlane(const SessionSnapshot& session_snapshot,
                                 absl::string_view host_name,
                                 const XSpace& xspace) {
  if (!session_snapshot.HasHost(host_name)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Host not found in session: ", host_name));
  }
  if (!session_snapshot.HasAccessibleRunDir()) return absl::OkStatus();
  TF_ASSIGN_OR_RETURN(
      const bool cache_exists,
      HostDataFileExists(session_snapshot, StoredDataType::LLO_XPLANE,
                         host_name));
  if (cache_exists) return absl::OkStatus();
  XPlane llo_plane;
  ExtractLloXPlane(xspace, &llo_plane);
  return WriteBinaryProto(session_snapshot, StoredDataType::LLO_XPLANE,
                          host_name, llo_plane);
}

absl::StatusOr<XPlane> GetLloXPlaneByHostName(
    const SessionSnapshot& session_snapshot, absl::string_view host_name) {
  if (session_snapshot.XSpaceSize() == 0) {
    return absl::NotFoundError("The session has no XSpace.");
  }
  const std::string host = host_name.empty() ? session_snapshot.GetHostname(0)
                                             : std::string(host_name);
  if (!session_snapshot.HasHost(host)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Host not found in session: ", host));
  }

  XPlane llo_plane;
  if (session_snapshot.HasAccessibleRunDir()) {
    absl::Status read_status = ReadBinaryProto(
        session_snapshot, StoredDataType::LLO_XPLANE, host, &llo_plane);
    if (read_status.ok()) {
      return llo_plane;
    }
    LOG_IF(WARNING, !absl::IsNotFound(read_status))
        << "Failed to read the cached LLO modules for host " << host << ": "
        << read_status;
    llo_plane.Clear();
  }
  google::protobuf::Arena arena;
  TF_ASSIGN_OR_RETURN(XSpace * xspace,
                      session_snapshot.GetXSpaceByName(host, &arena));
  ExtractLloXPlane(*xspace, &llo_plane);
  if (session_snapshot.HasAccessibleRunDir()) {
    absl::Status status = WriteBinaryProto(
        session_snapshot, StoredDataType::LLO_XPLANE, host, llo_plane);
    LOG_IF(WARNING, !status.ok()) << "Failed to cache the LLO modules for host "
                                  << host << ": " << status;
  }
  return llo_plane;
}

}  // namespace profiler
}  // namespace tensorflow
