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

#include "xprof/convert/event_time_fraction_analyzer_processor.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_split.h"
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

using ::tensorflow::profiler::ConvertXSpaceToEventTimeFractionAnalyzerResults;
using ::tensorflow::profiler::EventTimeFractionAnalyzerResult;
using ::tensorflow::profiler::EventTimeFractionAnalyzerResults;
using ::tensorflow::profiler::PreprocessSingleHostXSpace;
using ::tensorflow::profiler::SessionSnapshot;
using ::tensorflow::profiler::ToolOptions;
using ::tensorflow::profiler::XSpace;

std::vector<std::string> GetTargetEventNames(const ToolOptions& options) {
  for (const auto& [key, value] : options) {
    if (key == "event_name" && std::holds_alternative<std::string>(value)) {
      return absl::StrSplit(std::get<std::string>(value), ',');
    }
  }
  return {};
}

std::unique_ptr<ProfileProcessor> CreateEventTimeFractionAnalyzerProcessor(
    const ToolOptions& options) {
  return std::make_unique<EventTimeFractionAnalyzerProcessor>(options);
}

RegisterProfileProcessor event_time_fraction_analyzer_processor_registration(
    "event_time_fraction_analyzer", CreateEventTimeFractionAnalyzerProcessor);

void AccumulateEventTimeFractionAnalyzerResults(
    const EventTimeFractionAnalyzerResults& results,
    EventTimeFractionAnalyzerResults& combined_results) {
  for (const auto& [target_event_name, result] : results.results()) {
    auto* combined_result =
        &(*combined_results.mutable_results())[target_event_name];
    for (const auto& [chip_id, fractions] :
         result.chip_event_time_fractions()) {
      auto& combined_fractions =
          (*combined_result->mutable_chip_event_time_fractions())[chip_id];
      combined_fractions.set_id(chip_id);
      combined_fractions.mutable_event_time_fractions()->Add(
          fractions.event_time_fractions().begin(),
          fractions.event_time_fractions().end());
    }
    for (const auto& [host_name, fractions] :
         result.host_event_time_fractions()) {
      auto& combined_fractions =
          (*combined_result->mutable_host_event_time_fractions())[host_name];
      combined_fractions.set_hostname(host_name);
      combined_fractions.mutable_event_time_fractions()->Add(
          fractions.event_time_fractions().begin(),
          fractions.event_time_fractions().end());
    }
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
  std::vector<std::string> target_event_names = GetTargetEventNames(options_);
  TF_ASSIGN_OR_RETURN(EventTimeFractionAnalyzerResults results,
                      ConvertXSpaceToEventTimeFractionAnalyzerResults(
                          xspace_copy, target_event_names));
  return results.SerializeAsString();
}

absl::StatusOr<std::string> EventTimeFractionAnalyzerProcessor::Map(
    const std::string& xspace_path) {
  std::vector<std::string> xspace_paths = {xspace_path};
  TF_ASSIGN_OR_RETURN(
      SessionSnapshot session_snapshot,
      SessionSnapshot::Create(xspace_paths, /*xspaces=*/std::nullopt));

  EventTimeFractionAnalyzerResults combined_results;
  for (int i = 0; i < session_snapshot.XSpaceSize(); ++i) {
    google::protobuf::Arena arena;
    TF_ASSIGN_OR_RETURN(XSpace * xspace, session_snapshot.GetXSpace(i, &arena));
    std::string hostname = session_snapshot.GetHostname(i);
    TF_ASSIGN_OR_RETURN(std::string serialized_results,
                        Map(session_snapshot, hostname, *xspace));
    EventTimeFractionAnalyzerResults results;
    if (!results.ParseFromString(serialized_results)) {
      return absl::InternalError(
          "Failed to parse EventTimeFractionAnalyzerResults from map output");
    }
    AccumulateEventTimeFractionAnalyzerResults(results, combined_results);
  }

  if (combined_results.SerializeAsString().empty()) {
    return absl::InternalError(
        "Failed to serialize EventTimeFractionAnalyzerResults");
  }
  return combined_results.SerializeAsString();
}

absl::Status EventTimeFractionAnalyzerProcessor::Reduce(
    const SessionSnapshot& session_snapshot,
    const std::vector<std::string>& map_output_files) {
  EventTimeFractionAnalyzerResults combined_results;

  for (const auto& map_output_file : map_output_files) {
    EventTimeFractionAnalyzerResults results;
    if (!results.ParseFromString(map_output_file)) {
      return absl::InternalError(
          "Failed to parse EventTimeFractionAnalyzerResults from map output");
    }
    AccumulateEventTimeFractionAnalyzerResults(results, combined_results);
  }

  data_ = combined_results.SerializeAsString();
  return absl::OkStatus();
}

absl::Status EventTimeFractionAnalyzerProcessor::ProcessSession(
    const SessionSnapshot& session_snapshot, const ToolOptions& options) {
  EventTimeFractionAnalyzerResults combined_results;
  for (int i = 0; i < session_snapshot.XSpaceSize(); i++) {
    google::protobuf::Arena arena;
    TF_ASSIGN_OR_RETURN(XSpace * xspace, session_snapshot.GetXSpace(i, &arena));
    std::string hostname = session_snapshot.GetHostname(i);
    TF_ASSIGN_OR_RETURN(std::string serialized_results,
                        Map(session_snapshot, hostname, *xspace));
    EventTimeFractionAnalyzerResults results;
    if (!results.ParseFromString(serialized_results)) {
      return absl::InternalError(
          "Failed to parse EventTimeFractionAnalyzerResults from map output");
    }
    AccumulateEventTimeFractionAnalyzerResults(results, combined_results);
  }
  data_ = combined_results.SerializeAsString();
  return absl::OkStatus();
}

bool EventTimeFractionAnalyzerProcessor::ShouldUseWorkerService(
    const SessionSnapshot& session_snapshot, const ToolOptions& options) const {
  return session_snapshot.HasAccessibleRunDir() &&
         session_snapshot.XSpaceSize() > 1;
}

}  // namespace xprof
