/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/types.h"
#include "xla/tsl/profiler/rpc/client/capture_profile.h"
#include "xla/tsl/profiler/utils/session_manager.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/repository.h"
#include "xprof/convert/tool_options.h"
#include "xprof/convert/xplane_to_tools_data.h"

namespace xprof {
namespace pywrap {

using ::tensorflow::profiler::ConvertMultiXSpacesToToolData;
using ::tensorflow::profiler::GetParam;
using ::tensorflow::profiler::SessionSnapshot;
using ::tensorflow::profiler::ToolOptions;
using ::tensorflow::profiler::XSpace;

absl::StatusOr<std::pair<std::string, bool>> SessionSnapshotToToolsData(
    const absl::StatusOr<SessionSnapshot>& status_or_session_snapshot,
    const std::string& tool_name, const ToolOptions& tool_options) {
  if (!status_or_session_snapshot.ok()) {
    LOG(ERROR) << status_or_session_snapshot.status().message();
    return std::make_pair("", false);
  }

  // If use_saved_result is False, clear the cache files before converting to
  // tool data.
  std::optional<bool> use_saved_result =
      GetParam<bool>(tool_options, "use_saved_result");
  if (use_saved_result.has_value() && !use_saved_result.value()) {
    TF_RETURN_IF_ERROR(status_or_session_snapshot->ClearCacheFiles());
  }

  absl::StatusOr<std::string> status_or_tool_data =
      ConvertMultiXSpacesToToolData(status_or_session_snapshot.value(),
                                    tool_name, tool_options);
  if (!status_or_tool_data.ok()) {
    LOG(ERROR) << status_or_tool_data.status().message();
    return std::make_pair(std::string(status_or_tool_data.status().message()),
                          false);
  }
  return std::make_pair(status_or_tool_data.value(), true);
}

absl::Status Monitor(const char* service_addr, int duration_ms,
                     int monitoring_level, bool display_timestamp,
                     tsl::string* result) {
  TF_RETURN_IF_ERROR(tsl::profiler::ValidateHostPortPair(service_addr));
  {
    TF_RETURN_IF_ERROR(tsl::profiler::Monitor(service_addr, duration_ms,
                                              monitoring_level,
                                              display_timestamp, result));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::pair<std::string, bool>> XSpaceToToolsData(
    std::vector<std::string> xspace_paths, const std::string& tool_name,
    const ToolOptions& tool_options) {
  auto status_or_session_snapshot = SessionSnapshot::Create(
      std::move(xspace_paths), /*xspaces=*/std::nullopt);
  return SessionSnapshotToToolsData(status_or_session_snapshot, tool_name,
                                    tool_options);
}

absl::StatusOr<std::pair<std::string, bool>> XSpaceToToolsDataFromByteString(
    std::vector<std::string> xspace_strings,
    std::vector<std::string> xspace_paths, const std::string& tool_name,
    const ToolOptions& tool_options) {
  std::vector<std::unique_ptr<XSpace>> xspaces;
  xspaces.reserve(xspace_strings.size());

  for (const auto& xspace_string : xspace_strings) {
    auto xspace = std::make_unique<XSpace>();
    if (!xspace->ParseFromString(xspace_string)) {
      return std::make_pair("", false);
    }

    for (int i = 0; i < xspace->hostnames_size(); ++i) {
      std::string hostname = xspace->hostnames(i);
      std::replace(hostname.begin(), hostname.end(), ':', '_');
      xspace->mutable_hostnames(i)->swap(hostname);
    }
    xspaces.push_back(std::move(xspace));
  }

  auto status_or_session_snapshot =
      SessionSnapshot::Create(std::move(xspace_paths), std::move(xspaces));
  return SessionSnapshotToToolsData(status_or_session_snapshot, tool_name,
                                    tool_options);
}

}  // namespace pywrap
}  // namespace xprof
