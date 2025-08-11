/* Copyright 2024 The OpenXLA Authors. All Rights Reserved.

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
#ifndef THIRD_PARTY_XPROF_CONVERT_PROFILE_PROCESSOR_H_
#define THIRD_PARTY_XPROF_CONVERT_PROFILE_PROCESSOR_H_

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/repository.h"
#include "xprof/convert/tool_options.h"
#include "plugin/xprof/protobuf/op_stats.pb.h"

namespace xprof {

class ProfileProcessor {
 public:
  virtual ~ProfileProcessor() = default;

  // Processes a single host's XSpace data and returns the path to the output
  // file.
  virtual absl::StatusOr<std::string> Map(
      const tensorflow::profiler::SessionSnapshot& session_snapshot,
      const std::string& hostname,
      const tensorflow::profiler::XSpace& xspace) = 0;

  // Combines the results from multiple Map calls.
  virtual absl::Status Reduce(
      const tensorflow::profiler::SessionSnapshot& session_snapshot,
      const std::vector<std::string>& map_output_files) = 0;

  // Indicates whether this tool can be distributed across multiple workers.
  virtual bool ShouldUseWorkerService(
      const tensorflow::profiler::SessionSnapshot& session_snapshot) const {
    return false;
  }

  // Processes the entire session at once, without map/reduce.
  virtual absl::Status ProcessSession(
      const tensorflow::profiler::SessionSnapshot& session_snapshot,
      const tensorflow::profiler::ToolOptions& options) {
    return absl::UnimplementedError("ProcessSession not implemented");
  }

  // Sets the final output data and content type.
  void SetOutput(absl::string_view data, absl::string_view content_type) {
    data_ = data;
    content_type_ = content_type;
  }

  const std::string& GetData() const { return data_; }
  const std::string& GetContentType() const { return content_type_; }

 protected:
  std::string data_;
  std::string content_type_;
};

}  // namespace xprof

#endif  // THIRD_PARTY_XPROF_CONVERT_PROFILE_PROCESSOR_H_
