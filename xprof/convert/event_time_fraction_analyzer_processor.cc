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

#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>
#include <optional>

#include "xprof/convert/event_time_fraction_analyzer_processor.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "google/protobuf/arena.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/preprocess_single_host_xplane.h"
#include "xprof/convert/profile_processor.h"
#include "xprof/convert/profile_processor_factory.h"
#include "xprof/convert/repository.h"
#include "xprof/convert/tool_options.h"
#include "xprof/convert/xspace_to_event_time_fraction_analyzer.h"
#include "plugin/xprof/protobuf/event_time_fraction_analyzer.pb.h"

namespace xprof {
namespace {

using ::tensorflow::profiler::ConvertXSpaceToEventTimeFractionAnalyzerResult;
using ::tensorflow::profiler::EventTimeFractionAnalyzerResult;
using ::tensorflow::profiler::PreprocessSingleHostXSpace;
using ::tensorflow::profiler::SessionSnapshot;
using ::tensorflow::profiler::ToolOptions;
using ::tensorflow::profiler::XSpace;

std::string GetTargetEventName(const ToolOptions& options) {
  for (const auto& [key, value] : options) {
    if (key == "event_name" && std::holds_alternative<std::string>(value)) {
      return std::get<std::string>(value);
    }
  }
  return "";
}

std::unique_ptr<ProfileProcessor> CreateEventTimeFractionAnalyzerProcessor(
    const ToolOptions& options) {
  return std::make_unique<EventTimeFractionAnalyzerProcessor>(options);
}

RegisterProfileProcessor event_time_fraction_analyzer_processor_registration(
    "event_time_fraction_analyzer", CreateEventTimeFractionAnalyzerProcessor);

void AccumulateEventTimeFractionAnalyzerResult(
    const EventTimeFractionAnalyzerResult& result,
    EventTimeFractionAnalyzerResult& combined_result) {
  for (const auto& [chip_id, fractions] : result.chip_event_time_fractions()) {
    if (combined_result.chip_event_time_fractions().contains(chip_id)) {
      auto& existing_fractions =
          (*combined_result.mutable_chip_event_time_fractions())[chip_id];
      existing_fractions.mutable_event_time_fractions()->Add(
          fractions.event_time_fractions().begin(),
          fractions.event_time_fractions().end());
    } else {
      combined_result.mutable_chip_event_time_fractions()->insert(
          {chip_id, fractions});
    }
  }
  for (const auto& [hostname, fractions] : result.host_event_time_fractions()) {
    combined_result.mutable_host_event_time_fractions()->insert(
        {hostname, fractions});
  }
}

}  // namespace

absl::StatusOr<std::string> EventTimeFractionAnalyzerProcessor::Map(
    const SessionSnapshot& session_snapshot, const std::string& hostname,
    const XSpace& xspace) {
  // We need to copy XSpace because PreprocessSingleHostXSpace modifies it.
  XSpace xspace_copy = xspace;
  if (xspace_copy.hostnames().empty()) {
    xspace_copy.add_hostnames(hostname);
  }
  PreprocessSingleHostXSpace(&xspace_copy, /*step_grouping=*/true,
                             /*derived_timeline=*/true);
  std::string target_event_name = GetTargetEventName(options_);
  TF_ASSIGN_OR_RETURN(EventTimeFractionAnalyzerResult result,
                      ConvertXSpaceToEventTimeFractionAnalyzerResult(
                          xspace_copy, target_event_name));
  return result.SerializeAsString();
}

absl::StatusOr<std::string> EventTimeFractionAnalyzerProcessor::Map(
    const std::string& xspace_path) {
  std::vector<std::string> xspace_paths = {xspace_path};
  TF_ASSIGN_OR_RETURN(
      SessionSnapshot session_snapshot,
      SessionSnapshot::Create(xspace_paths, /*xspaces=*/std::nullopt));

  EventTimeFractionAnalyzerResult combined_result;
  for (int i = 0; i < session_snapshot.XSpaceSize(); ++i) {
    google::protobuf::Arena arena;
    TF_ASSIGN_OR_RETURN(XSpace * xspace, session_snapshot.GetXSpace(i, &arena));
    std::string hostname = session_snapshot.GetHostname(i);
    TF_ASSIGN_OR_RETURN(std::string serialized_result,
                        Map(session_snapshot, hostname, *xspace));
    EventTimeFractionAnalyzerResult result;
    if (!result.ParseFromString(serialized_result)) {
      return absl::InternalError(
          "Failed to parse EventTimeFractionAnalyzerResult from map output");
    }
    AccumulateEventTimeFractionAnalyzerResult(result, combined_result);
  }

  if (combined_result.SerializeAsString().empty()) {
    return absl::InternalError(
        "Failed to serialize EventTimeFractionAnalyzerResult");
  }
  return combined_result.SerializeAsString();
}

absl::Status EventTimeFractionAnalyzerProcessor::Reduce(
    const SessionSnapshot& session_snapshot,
    const std::vector<std::string>& map_output_files) {
  EventTimeFractionAnalyzerResult combined_result;

  for (const auto& map_output_file : map_output_files) {
    EventTimeFractionAnalyzerResult result;
    if (!result.ParseFromString(map_output_file)) {
      return absl::InternalError(
          "Failed to parse EventTimeFractionAnalyzerResult from map output");
    }
    AccumulateEventTimeFractionAnalyzerResult(result, combined_result);
  }

  data_ = combined_result.SerializeAsString();
  return absl::OkStatus();
}

absl::Status EventTimeFractionAnalyzerProcessor::ProcessSession(
    const SessionSnapshot& session_snapshot, const ToolOptions& options) {
  EventTimeFractionAnalyzerResult combined_result;
  for (int i = 0; i < session_snapshot.XSpaceSize(); i++) {
    google::protobuf::Arena arena;
    TF_ASSIGN_OR_RETURN(XSpace * xspace, session_snapshot.GetXSpace(i, &arena));
    std::string hostname = session_snapshot.GetHostname(i);
    TF_ASSIGN_OR_RETURN(std::string serialized_result,
                        Map(session_snapshot, hostname, *xspace));
    EventTimeFractionAnalyzerResult result;
    if (!result.ParseFromString(serialized_result)) {
      return absl::InternalError(
          "Failed to parse EventTimeFractionAnalyzerResult from map output");
    }
    AccumulateEventTimeFractionAnalyzerResult(result, combined_result);
  }
  data_ = combined_result.SerializeAsString();
  return absl::OkStatus();
}

bool EventTimeFractionAnalyzerProcessor::ShouldUseWorkerService(
    const SessionSnapshot& session_snapshot, const ToolOptions& options) const {
  return session_snapshot.HasAccessibleRunDir() &&
         session_snapshot.XSpaceSize() > 1;
}

}  // namespace xprof
