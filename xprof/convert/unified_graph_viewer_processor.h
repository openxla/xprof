/* Copyright 2026 The OpenXLA Authors. All Rights Reserved.

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

#ifndef THIRD_PARTY_XPROF_CONVERT_UNIFIED_GRAPH_VIEWER_PROCESSOR_H_
#define THIRD_PARTY_XPROF_CONVERT_UNIFIED_GRAPH_VIEWER_PROCESSOR_H_

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/service/hlo.pb.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/base_hlo_processor.h"
#include "xprof/convert/tool_options.h"
#include "xprof/convert/unified_session_snapshot.h"

namespace xprof {

absl::StatusOr<std::string> ConvertHloProtoToGraphViewer(
    const xla::HloProto& hlo_proto,
    const tensorflow::profiler::ToolOptions& options);

class UnifiedGraphViewerProcessor : public BaseHloProcessor {
 public:
  explicit UnifiedGraphViewerProcessor(
      const tensorflow::profiler::ToolOptions& options)
      : BaseHloProcessor(options) {}
  ~UnifiedGraphViewerProcessor() override = default;

  absl::Status ProcessSession(
      const XprofSessionSnapshot& session_snapshot,
      const tensorflow::profiler::ToolOptions& options) override;

  absl::Status ProcessHlo(
      const XprofSessionSnapshot& session_snapshot,
      const xla::HloProto& hlo_proto,
      const tensorflow::profiler::ToolOptions& options) override;
};

}  // namespace xprof

#endif  // THIRD_PARTY_XPROF_CONVERT_UNIFIED_GRAPH_VIEWER_PROCESSOR_H_
