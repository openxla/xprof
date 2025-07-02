/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#ifndef THIRD_PARTY_XPROF_CONVERT_TOOL_CONVERTERS_H_
#define THIRD_PARTY_XPROF_CONVERT_TOOL_CONVERTERS_H_

#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xprof/convert/repository.h"
#include "xprof/convert/tool_options.h"

namespace tensorflow {
namespace profiler {

// Converts XSpace to Trace Events JSON for the trace viewer tool.
absl::StatusOr<std::string> ConvertXSpaceToTraceEvents(
    const SessionSnapshot& session_snapshot, absl::string_view tool_name,
    const ToolOptions& options);

// Converts multiple XSpaces to Overview Page data.
absl::StatusOr<std::string> ConvertMultiXSpacesToOverviewPage(
    const SessionSnapshot& session_snapshot);

// Converts multiple XSpaces to Input Pipeline Analysis data.
absl::StatusOr<std::string> ConvertMultiXSpacesToInputPipeline(
    const SessionSnapshot& session_snapshot);

// Converts multiple XSpaces to TF Stats data.
absl::StatusOr<std::string> ConvertMultiXSpacesToTfStats(
    const SessionSnapshot& session_snapshot);

// Converts multiple XSpaces to Kernel Stats data.
absl::StatusOr<std::string> ConvertMultiXSpacesToKernelStats(
    const SessionSnapshot& session_snapshot);

// Converts XSpace to Memory Profile JSON.
absl::StatusOr<std::string> ConvertXSpaceToMemoryProfile(
    const SessionSnapshot& session_snapshot);

// Converts multiple XSpaces to Pod Viewer data.
absl::StatusOr<std::string> ConvertMultiXSpacesToPodViewer(
    const SessionSnapshot& session_snapshot);

// Converts multiple XSpaces to TF Data Bottleneck Analysis data.
absl::StatusOr<std::string> ConvertMultiXSpacesToTfDataBottleneckAnalysis(
    const SessionSnapshot& session_snapshot);

// Converts multiple XSpaces to HLO Stats data.
absl::StatusOr<std::string> ConvertMultiXSpacesToHloStats(
    const SessionSnapshot& session_snapshot);

// Converts multiple XSpaces to Roofline Model data.
absl::StatusOr<std::string> ConvertMultiXSpacesToRooflineModel(
    const SessionSnapshot& session_snapshot);

// Converts multiple XSpaces to Op Profile Viewer data.
absl::StatusOr<std::string> ConvertMultiXSpacesToOpProfileViewer(
    const SessionSnapshot& session_snapshot);

// Preprocesses XSpace data.
absl::StatusOr<std::string> PreprocessXSpace(
    const SessionSnapshot& session_snapshot);

// Converts DCN Collective Stats to Tool Data.
absl::StatusOr<std::string> ConvertDcnCollectiveStatsToToolData(
    const SessionSnapshot& session_snapshot, const ToolOptions& options);

// Converts multiple XSpaces to Inference Stats data.
absl::StatusOr<std::string> ConvertMultiXSpacesToInferenceStats(
    const SessionSnapshot& session_snapshot, const ToolOptions& options);

}  // namespace profiler
}  // namespace tensorflow

#endif  // THIRD_PARTY_XPROF_CONVERT_TOOL_CONVERTERS_H_
