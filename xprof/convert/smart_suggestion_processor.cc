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

#include "xprof/convert/smart_suggestion_processor.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/json/json.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/profile_processor.h"
#include "xprof/convert/profile_processor_factory.h"
#include "xprof/convert/repository.h"
#include "xprof/convert/smart_suggestion/all_rules.h"
#include "xprof/convert/smart_suggestion/signal_provider.h"
#include "xprof/convert/smart_suggestion/smart_suggestion_engine.h"
#include "xprof/convert/smart_suggestion/smart_suggestion_rule_factory.h"
#include "xprof/convert/smart_suggestion/tool_data_provider_impl.h"
#include "xprof/convert/tool_options.h"
#include "plugin/xprof/protobuf/smart_suggestion.pb.h"

namespace xprof {
namespace {

using ::tensorflow::profiler::RegisterAllRules;
using ::tensorflow::profiler::SessionSnapshot;
using ::tensorflow::profiler::SignalProvider;
using ::tensorflow::profiler::SmartSuggestionEngine;
using ::tensorflow::profiler::SmartSuggestionReport;
using ::tensorflow::profiler::SmartSuggestionRuleFactory;
using ::tensorflow::profiler::ToolDataProviderImpl;
using ::tensorflow::profiler::ToolOptions;
using ::tensorflow::profiler::XSpace;

std::unique_ptr<ProfileProcessor> CreateSmartSuggestionProcessor(
    const ToolOptions& options) {
  return std::make_unique<SmartSuggestionProcessor>(options);
}

RegisterProfileProcessor smart_suggestion_processor_registration(
    "smart_suggestion", CreateSmartSuggestionProcessor);

}  // namespace

absl::StatusOr<std::string> SmartSuggestionProcessor::Map(
    const SessionSnapshot& session_snapshot, const std::string& hostname,
    const XSpace& xspace) {
  return absl::UnimplementedError(
      "SmartSuggestionProcessor::Map is not implemented yet.");
}

absl::Status SmartSuggestionProcessor::Reduce(
    const SessionSnapshot& session_snapshot,
    const std::vector<std::string>& map_output_files) {
  return absl::UnimplementedError(
      "SmartSuggestionProcessor::Reduce is not implemented yet.");
}

absl::Status SmartSuggestionProcessor::ProcessSession(
    const SessionSnapshot& session_snapshot, const ToolOptions& options) {
  SmartSuggestionEngine engine;
  SmartSuggestionRuleFactory rule_factory;
  RegisterAllRulesFor3P(&rule_factory);

  auto tool_data_provider =
      std::make_unique<ToolDataProviderImpl>(session_snapshot);
  SignalProvider signal_provider(std::move(tool_data_provider));

  TF_ASSIGN_OR_RETURN(SmartSuggestionReport report,
                      engine.Run(signal_provider, rule_factory));

  std::string json_output;
  tsl::protobuf::util::JsonPrintOptions opts;
  opts.always_print_fields_with_no_presence = true;
  auto encode_status =
      tsl::protobuf::util::MessageToJsonString(report, &json_output, opts);
  if (!encode_status.ok()) {
    const auto& error_message = encode_status.message();
    return tsl::errors::Internal(
        "Could not convert smart suggestion report to json. Error: ",
        absl::string_view(error_message.data(), error_message.length()));
  }

  data_ = json_output;
  return absl::OkStatus();
}

}  // namespace xprof
