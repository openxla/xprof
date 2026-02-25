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

#include "xprof/convert/xplane_to_hlo.h"

#include <optional>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "google/protobuf/arena.h"
#include "xla/service/hlo.pb.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/profiler/utils/file_system_utils.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/file_utils.h"
#include "xprof/convert/repository.h"
#include "xprof/convert/tool_options.h"
#include "xprof/utils/hlo_proto_map.h"

namespace tensorflow {
namespace profiler {

namespace {

using tsl::profiler::ProfilerJoinPath;

constexpr char kNoModuleIdentifier[] = "NO_MODULE";
constexpr char kHloProtoSuffix[] = ".hlo_proto.pb";

// Extracts and deduplicates the HLO protos from all the XSpaces.
// Stores the HLO protos as files in the same directory as the xspace files.
absl::StatusOr<bool> GetHloProtoFromMultiXSpaceAndSaveToFile(
    const SessionSnapshot& session_snapshot) {
  // Get all HLO protos from XSpaces and deduplicate.
  HloProtoMap hlo_proto_map;
  for (int i = 0; i < session_snapshot.XSpaceSize(); i++) {
    google::protobuf::Arena arena;
    TF_ASSIGN_OR_RETURN(XSpace* xspace, session_snapshot.GetXSpace(i, &arena));
    hlo_proto_map.AddHloProtosFromXSpace(*xspace);
  }

  std::vector<absl::string_view> module_list = hlo_proto_map.GetModuleList();
  // Write an empty identifier if there is no HLO module.
  if (module_list.empty()) {
    std::string file_name =
        ProfilerJoinPath(session_snapshot.GetSessionRunDir(),
                         absl::StrCat(kNoModuleIdentifier, kHloProtoSuffix));
    xla::HloProto empty_hlo;
    TF_RETURN_IF_ERROR(
        xprof::WriteBinaryProto(file_name, empty_hlo));
    // The profile does not have HLO proto.
    return false;
  }

  // Save HLO protos to session run directory.
  for (const absl::string_view module_name : module_list) {
    auto hlo_proto_or = hlo_proto_map.GetHloProtoByModuleName(module_name);
    if (!hlo_proto_or.ok()) {
      return tsl::errors::Internal(hlo_proto_or.status().message());
    }
    std::string file_name =
        ProfilerJoinPath(session_snapshot.GetSessionRunDir(),
                         absl::StrCat(module_name, kHloProtoSuffix));
    TF_RETURN_IF_ERROR(xprof::WriteBinaryProto(
        file_name, *hlo_proto_or.value()));
  }

  // The profile has HLO proto.
  return true;
}

}  // namespace

absl::StatusOr<xla::HloProto> GetHloProtoByModuleName(
    const SessionSnapshot& session_snapshot,
    const absl::string_view module_name) {
  std::string file_name =
      ProfilerJoinPath(session_snapshot.GetSessionRunDir(),
                       absl::StrCat(module_name, kHloProtoSuffix));
  xla::HloProto hlo_proto;
  TF_RETURN_IF_ERROR(xprof::ReadBinaryProto(file_name, &hlo_proto));
  return hlo_proto;
}

absl::StatusOr<xla::HloProto> GetHloProtoByProgramId(
    const SessionSnapshot& session_snapshot,
    const absl::string_view program_id_str) {
  std::vector<std::string> files;
  TF_RETURN_IF_ERROR(tsl::Env::Default()->GetChildren(
      std::string(session_snapshot.GetSessionRunDir()), &files));

  std::string target_module_name = "";

  for (const std::string& file : files) {
    if (absl::EndsWith(file, kHloProtoSuffix)) {
      absl::string_view module_name = file;
      if (!absl::ConsumeSuffix(&module_name, kHloProtoSuffix)) {
        continue;  // Should not happen based on the EndsWith check
      }

      // Fuzzy search: Check if the module name contains the program_id string.
      if (absl::StrContains(module_name, program_id_str)) {
        // Assuming the first match is the desired one.
        target_module_name = std::string(module_name);
        break;
      }
    }
  }

  if (target_module_name.empty()) {
    return tsl::errors::NotFound(
        absl::StrCat("HLO proto file containing program ID ", program_id_str,
                     " not found in ", session_snapshot.GetSessionRunDir()));
  }

  return GetHloProtoByModuleName(session_snapshot, target_module_name);
}

// TODO(b/471848690): Revisit and consolidate the hlo proto processing logic.
absl::StatusOr<xla::HloProto> GetHloProtoByOptions(
    const SessionSnapshot& session_snapshot, const ToolOptions& options) {
  std::optional<std::string> hlo_module_name =
      GetParam<std::string>(options, tensorflow::profiler::kModuleNameOption);
  std::optional<std::string> program_id =
      GetParam<std::string>(options, tensorflow::profiler::kProgramIdOption);

  if (hlo_module_name.has_value() && !hlo_module_name->empty()) {
    return GetHloProtoByModuleName(session_snapshot, *hlo_module_name);
  } else if (program_id.has_value() && !program_id->empty()) {
    return GetHloProtoByProgramId(session_snapshot, *program_id);
  } else {
    return tsl::errors::InvalidArgument("Can not load hlo proto from options.");
  }
}

absl::StatusOr<bool> ConvertMultiXSpaceToHloProto(
    const SessionSnapshot& session_snapshot) {
  // Gets all the files in session run directory.
  // TODO(profiler): Move this glob to SessionSnapshot and build a map from file
  // type to file paths.
  std::vector<std::string> results;
  TF_RETURN_IF_ERROR(tsl::Env::Default()->GetChildren(
      std::string(session_snapshot.GetSessionRunDir()), &results));

  // If the profiler finds a filename with hlo proto suffix, this means HLO
  // proto was already generated previously.
  for (const std::string& path : results) {
    if (absl::EndsWith(path, kHloProtoSuffix)) {
      if (absl::EndsWith(path,
                         absl::StrCat(kNoModuleIdentifier, kHloProtoSuffix))) {
        return false;
      } else {
        return true;
      }
    }
  }

  // Generate HLO proto.
  // TODO(jiesun): Maybe generate a tag file at profile collection time, so
  // don't need to read XSpace files for checking whether HLO proto exists or
  // not.
  return GetHloProtoFromMultiXSpaceAndSaveToFile(session_snapshot);
}

}  // namespace profiler
}  // namespace tensorflow
