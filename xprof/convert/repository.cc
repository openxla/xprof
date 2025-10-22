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

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "google/protobuf/arena.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/profiler/utils/file_system_utils.h"
#include "tsl/platform/path.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace tensorflow {
namespace profiler {
namespace {
std::string GetHostnameByPath(absl::string_view xspace_path) {
  std::string_view file_name = tsl::io::Basename(xspace_path);
  // Remove suffix from file_name, preserving entire prefix.
  absl::ConsumeSuffix(&file_name, ".xplane.pb");
  return std::string(file_name);
}

static auto* kHostDataSuffixes =
    new std::vector<std::pair<StoredDataType, const char*>>(
        {{StoredDataType::DCN_COLLECTIVE_STATS, ".dcn_collective_stats.pb"},
         {StoredDataType::OP_STATS, ".op_stats.pb"},
         {StoredDataType::TRACE_LEVELDB, ".SSTABLE"},
         {StoredDataType::TRACE_EVENTS_METADATA_LEVELDB, ".metadata.SSTABLE"},
         {StoredDataType::TRACE_EVENTS_PREFIX_TRIE_LEVELDB, ".trie.SSTABLE"}});

}  // namespace

absl::StatusOr<SessionSnapshot> SessionSnapshot::Create(
    std::vector<std::string> xspace_paths,
    std::optional<std::vector<std::unique_ptr<XSpace>>> xspaces,
    std::optional<std::vector<std::string>> all_hosts) {
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

  return SessionSnapshot(std::move(xspace_paths), std::move(xspaces),
                         std::move(all_hosts));
}

SessionSnapshot::SessionSnapshot(
    std::vector<std::string> xspace_paths,
    std::optional<std::vector<std::unique_ptr<XSpace>>> xspaces,
    std::optional<std::vector<std::string>> all_hosts)
    : xspace_paths_(std::move(xspace_paths)),
      all_hosts_(std::move(all_hosts)),
      // If the snapshot was initialized by xspaces, the file path and run dir
      // is a path tensorflow can't read from or write to so any file IO
      // encapsulated in this class will be disabled in this mode.
      has_accessible_run_dir_(!xspaces.has_value()),
      xspaces_(std::move(xspaces)) {
  session_run_dir_ = tsl::io::Dirname(xspace_paths_.at(0));
  for (size_t i = 0; i < xspace_paths_.size(); ++i) {
    std::string host_name = GetHostname(i);
    hostname_map_[host_name] = i;
  }
}

absl::StatusOr<XSpace*> SessionSnapshot::GetXSpace(size_t index,
                                                   google::protobuf::Arena* arena) const {
  if (index >= xspace_paths_.size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Can not get the ", index, "th XSpace. The total number of XSpace is ",
        xspace_paths_.size()));
  }

  // Return the pre-loaded XSpace proto.
  if (xspaces_.has_value()) {
    if (xspaces_->at(index) == nullptr) {
      return tsl::errors::Internal("");
    }
    return xspaces_->at(index).get();
  }

  // Return the XSpace proto from file.
  XSpace* xspace_from_file = google::protobuf::Arena::Create<XSpace>(arena);
  TF_RETURN_IF_ERROR(tsl::ReadBinaryProto(
      tsl::Env::Default(), xspace_paths_.at(index), xspace_from_file));
  return xspace_from_file;
}

absl::StatusOr<XSpace*> SessionSnapshot::GetXSpaceByName(
    absl::string_view name, google::protobuf::Arena* arena) const {
  if (auto it = hostname_map_.find(name); it != hostname_map_.end()) {
    return GetXSpace(it->second, arena);
  }

  return absl::InvalidArgumentError(
      absl::StrCat("Can not find the XSpace by name: ", name,
                   ". The total number of XSpace is ", xspace_paths_.size()));
}

std::string SessionSnapshot::GetHostname(size_t index) const {
  return GetHostnameByPath(xspace_paths_.at(index));
}

std::optional<std::vector<std::string>> SessionSnapshot::GetAllHosts() const {
  return all_hosts_;
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
    const StoredDataType data_type, const std::string host) const {
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
