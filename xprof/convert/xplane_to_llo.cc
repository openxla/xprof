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

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/arena.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/profiler/utils/file_system_utils.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_utils.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/file_utils.h"
#include "xprof/convert/repository.h"

namespace tensorflow {
namespace profiler {
namespace {

using tsl::profiler::ProfilerJoinPath;

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

std::string LloXPlaneCachePath(const SessionSnapshot& session_snapshot,
                               absl::string_view host_name) {
  return ProfilerJoinPath(session_snapshot.GetSessionRunDir(),
                          absl::StrCat(host_name, kLloXPlaneSuffix));
}

}  // namespace

bool HasLloProtoMetadata(const XSpace& xspace) {
  const XPlane* plane = tsl::profiler::FindPlaneWithName(
      xspace, tsl::profiler::kMetadataPlaneName);
  if (plane == nullptr) return false;
  const int64_t llo_proto_stat_id = FindLloProtoStatMetadataId(*plane);
  if (llo_proto_stat_id < 0) return false;
  for (const auto& [id, event_metadata] : plane->event_metadata()) {
    if (HasLloProtoStat(event_metadata, llo_proto_stat_id)) return true;
  }
  return false;
}

int ExtractLloXPlane(const XSpace& xspace, XPlane* llo_plane) {
  const XPlane* plane = tsl::profiler::FindPlaneWithName(
      xspace, tsl::profiler::kMetadataPlaneName);
  if (plane == nullptr) return 0;
  const int64_t llo_proto_stat_id = FindLloProtoStatMetadataId(*plane);
  if (llo_proto_stat_id < 0) return 0;

  llo_plane->set_name(plane->name());
  llo_plane->set_id(plane->id());
  (*llo_plane->mutable_stat_metadata())[llo_proto_stat_id] =
      plane->stat_metadata().at(llo_proto_stat_id);
  int num_modules = 0;
  for (const auto& [id, event_metadata] : plane->event_metadata()) {
    if (!HasLloProtoStat(event_metadata, llo_proto_stat_id)) continue;
    XEventMetadata& copy = (*llo_plane->mutable_event_metadata())[id];
    copy.set_id(event_metadata.id());
    copy.set_name(event_metadata.name());
    copy.set_display_name(event_metadata.display_name());
    for (const XStat& stat : event_metadata.stats()) {
      if (stat.metadata_id() == llo_proto_stat_id) {
        *copy.add_stats() = stat;
      }
    }
    ++num_modules;
  }
  return num_modules;
}

absl::Status MaybeCacheLloXPlane(const SessionSnapshot& session_snapshot,
                                 absl::string_view host_name,
                                 const XSpace& xspace) {
  if (!session_snapshot.HasAccessibleRunDir()) return absl::OkStatus();
  const std::string path = LloXPlaneCachePath(session_snapshot, host_name);
  if (tsl::Env::Default()->FileExists(path).ok()) return absl::OkStatus();
  XPlane llo_plane;
  ExtractLloXPlane(xspace, &llo_plane);
  return xprof::WriteBinaryProto(path, llo_plane);
}

absl::StatusOr<XPlane> GetLloXPlaneByHostName(
    const SessionSnapshot& session_snapshot, absl::string_view host_name) {
  if (session_snapshot.XSpaceSize() == 0) {
    return absl::NotFoundError("The session has no XSpace.");
  }
  std::string host = std::string(host_name);
  if (host.empty()) host = session_snapshot.GetHostname(0);

  XPlane llo_plane;
  if (session_snapshot.HasAccessibleRunDir()) {
    const std::string path = LloXPlaneCachePath(session_snapshot, host);
    if (tsl::Env::Default()->FileExists(path).ok()) {
      TF_RETURN_IF_ERROR(xprof::ReadBinaryProto(path, &llo_plane));
      return llo_plane;
    }
  }
  google::protobuf::Arena arena;
  TF_ASSIGN_OR_RETURN(XSpace * xspace,
                      session_snapshot.GetXSpaceByName(host, &arena));
  ExtractLloXPlane(*xspace, &llo_plane);
  if (session_snapshot.HasAccessibleRunDir()) {
    TF_RETURN_IF_ERROR(xprof::WriteBinaryProto(
        LloXPlaneCachePath(session_snapshot, host), llo_plane));
  }
  return llo_plane;
}

}  // namespace profiler
}  // namespace tensorflow
