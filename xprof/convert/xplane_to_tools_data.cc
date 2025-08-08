/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "xprof/convert/xplane_to_tools_data.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/file_system.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/profiler/convert/xplane_to_trace_events.h"
#include "xla/tsl/profiler/utils/timespan.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_utils.h"
#include "tsl/platform/protobuf.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/compute_inference_latency.h"
#include "xprof/convert/hlo_to_tools_data.h"
#include "xprof/convert/multi_xplanes_to_op_stats.h"
#include "xprof/convert/multi_xspace_to_inference_stats.h"
#include "xprof/convert/op_stats_to_hlo_stats.h"
#include "xprof/convert/op_stats_to_input_pipeline_analysis.h"
#include "xprof/convert/op_stats_to_op_profile.h"
#include "xprof/convert/op_stats_to_overview_page.h"
#include "xprof/convert/op_stats_to_pod_viewer.h"
#include "xprof/convert/op_stats_to_roofline_model.h"
#include "xprof/convert/op_stats_to_tf_stats.h"
#include "xprof/convert/preprocess_single_host_xplane.h"
#include "xprof/convert/process_megascale_dcn.h"
#include "xprof/convert/profile_processor.h"
#include "xprof/convert/profile_processor_factory.h"
#include "xprof/convert/repository.h"
#include "xprof/convert/smart_suggestion/all_rules.h"
#include "xprof/convert/smart_suggestion/signal_provider.h"
#include "xprof/convert/smart_suggestion/smart_suggestion_engine.h"
#include "xprof/convert/smart_suggestion/smart_suggestion_rule_factory.h"
#include "xprof/convert/smart_suggestion/tool_data_provider_impl.h"
#include "xprof/convert/tool_options.h"
#include "xprof/convert/trace_viewer/trace_events.h"
#include "xprof/convert/trace_viewer/trace_events_to_json.h"
#include "xprof/convert/trace_viewer/trace_options.h"
#include "xprof/convert/trace_viewer/trace_viewer_visibility.h"
#include "xprof/convert/xplane_to_dcn_collective_stats.h"
#include "xprof/convert/xplane_to_hlo.h"
#include "xprof/convert/xplane_to_kernel_stats_db.h"
#include "xprof/convert/xplane_to_memory_profile.h"
#include "xprof/convert/xplane_to_tf_data_stats.h"
#include "xprof/convert/xplane_to_tool_names.h"
#include "xprof/convert/xplane_to_trace_container.h"
#include "plugin/xprof/protobuf/dcn_slack_analysis.pb.h"
#include "plugin/xprof/protobuf/hardware_types.pb.h"
#include "plugin/xprof/protobuf/inference_stats.pb.h"
#include "plugin/xprof/protobuf/input_pipeline.pb.h"
#include "plugin/xprof/protobuf/kernel_stats.pb.h"
#include "plugin/xprof/protobuf/op_profile.pb.h"
#include "plugin/xprof/protobuf/op_stats.pb.h"
#include "plugin/xprof/protobuf/overview_page.pb.h"
#include "plugin/xprof/protobuf/roofline_model.pb.h"
#include "plugin/xprof/protobuf/tf_data_stats.pb.h"
#include "plugin/xprof/protobuf/tf_stats.pb.h"
#include "xprof/utils/hardware_type_utils.h"

namespace tensorflow {
namespace profiler {

namespace {

struct TraceViewOption {
  uint64_t resolution = 0;
  double start_time_ms = 0.0;
  double end_time_ms = 0.0;
};

absl::StatusOr<TraceViewOption> GetTraceViewOption(const ToolOptions& options) {
  TraceViewOption trace_options;
  auto start_time_ms_opt =
      GetParamWithDefault<std::string>(options, "start_time_ms", "0.0");
  auto end_time_ms_opt =
      GetParamWithDefault<std::string>(options, "end_time_ms", "0.0");
  auto resolution_opt =
      GetParamWithDefault<std::string>(options, "resolution", "0");

  if (!absl::SimpleAtoi(resolution_opt, &trace_options.resolution) ||
      !absl::SimpleAtod(start_time_ms_opt, &trace_options.start_time_ms) ||
      !absl::SimpleAtod(end_time_ms_opt, &trace_options.end_time_ms)) {
    return tsl::errors::InvalidArgument("wrong arguments");
  }
  return trace_options;
}

absl::StatusOr<std::string> ConvertXSpaceToTraceEvents(
    const SessionSnapshot& session_snapshot, const absl::string_view tool_name,
    const ToolOptions& options) {
  if (session_snapshot.XSpaceSize() != 1) {
    return tsl::errors::InvalidArgument(
        "Trace events tool expects only 1 XSpace path but gets ",
        session_snapshot.XSpaceSize());
  }

  google::protobuf::Arena arena;
  TF_ASSIGN_OR_RETURN(XSpace* xspace, session_snapshot.GetXSpace(0, &arena));
  PreprocessSingleHostXSpace(xspace, /*step_grouping=*/true,
                             /*derived_timeline=*/true);
  std::string content;
  if (tool_name == "trace_viewer") {
    tsl::profiler::ConvertXSpaceToTraceEventsString(*xspace, &content);
    return content;
  } else {  // streaming trace viewer.
    std::string host_name = session_snapshot.GetHostname(0);
    auto sstable_path = session_snapshot.GetFilePath(tool_name, host_name);
    if (!sstable_path) {
      return tsl::errors::Unimplemented(
          "streaming trace viewer hasn't been supported in Cloud AI");
    }
    if (!tsl::Env::Default()->FileExists(*sstable_path).ok()) {
      ProcessMegascaleDcn(xspace);
      TraceEventsContainer trace_container;
      ConvertXSpaceToTraceEventsContainer(host_name, *xspace, &trace_container);
      std::unique_ptr<tsl::WritableFile> file;
      TF_RETURN_IF_ERROR(
          tsl::Env::Default()->NewWritableFile(*sstable_path, &file));
      TF_RETURN_IF_ERROR(trace_container.StoreAsLevelDbTable(std::move(file)));
    }
    TF_ASSIGN_OR_RETURN(TraceViewOption trace_option,
                        GetTraceViewOption(options));
    tensorflow::profiler::TraceOptions profiler_trace_options =
        TraceOptionsFromToolOptions(options);
    auto visibility_filter = std::make_unique<TraceVisibilityFilter>(
        tsl::profiler::MilliSpan(trace_option.start_time_ms,
                                 trace_option.end_time_ms),
        trace_option.resolution, profiler_trace_options);
    TraceEventsContainer trace_container;
    // Trace smaller than threshold will be disabled from streaming.
    constexpr int64_t kDisableStreamingThreshold = 500000;
    auto trace_events_filter =
        CreateTraceEventsFilterFromTraceOptions(profiler_trace_options);
    TraceEventsLevelDbFilePaths file_paths;
    file_paths.trace_events_file_path = *sstable_path;
    TF_RETURN_IF_ERROR(trace_container.LoadFromLevelDbTable(
        file_paths, std::move(trace_events_filter),
        std::move(visibility_filter), kDisableStreamingThreshold));
    JsonTraceOptions json_trace_options;

    tensorflow::profiler::TraceDeviceType device_type =
        tensorflow::profiler::TraceDeviceType::kUnknownDevice;
    if (IsTpuTrace(trace_container.trace())) {
      device_type = TraceDeviceType::kTpu;
    }
    json_trace_options.details =
        TraceOptionsToDetails(device_type, profiler_trace_options);
    IOBufferAdapter adapter(&content);
    TraceEventsToJson<IOBufferAdapter, TraceEventsContainer, RawData>(
        json_trace_options, trace_container, &adapter);
    return content;
  }
}

absl::StatusOr<std::string> ConvertMultiXSpacesToOverviewPage(
    const SessionSnapshot& session_snapshot) {
  OpStats combined_op_stats;
  TF_RETURN_IF_ERROR(ConvertMultiXSpaceToCombinedOpStatsWithCache(
      session_snapshot, &combined_op_stats));
  OverviewPage overview_page = ConvertOpStatsToOverviewPage(combined_op_stats);
  if (!combined_op_stats.run_environment().is_training()) {
    InferenceStats inference_stats;
    TF_RETURN_IF_ERROR(ConvertMultiXSpaceToInferenceStats(
        session_snapshot, "", "", &inference_stats));
    *overview_page.mutable_inference_latency() =
        ComputeInferenceLatencyResult(inference_stats);
  }
  return OverviewPageToJson(overview_page);
}

absl::StatusOr<std::string> ConvertMultiXSpacesToInputPipeline(
    const SessionSnapshot& session_snapshot) {
  OpStats combined_op_stats;
  TF_RETURN_IF_ERROR(ConvertMultiXSpaceToCombinedOpStatsWithCache(
      session_snapshot, &combined_op_stats));
  InputPipelineAnalysisResult result =
      ConvertOpStatsToInputPipelineAnalysis(combined_op_stats);
  return InputPipelineAnalysisResultToDataTableJson(result);
}

absl::StatusOr<std::string> ConvertMultiXSpacesToTfStats(
    const SessionSnapshot& session_snapshot) {
  OpStats combined_op_stats;
  TF_RETURN_IF_ERROR(ConvertMultiXSpaceToCombinedOpStatsWithCache(
      session_snapshot, &combined_op_stats));
  TfStatsDatabase tf_stats_db = ConvertOpStatsToTfStats(combined_op_stats);
  return TfStatsToDataTableJson(tf_stats_db);
}

absl::StatusOr<std::string> ConvertMultiXSpacesToKernelStats(
    const SessionSnapshot& session_snapshot) {
  OpStats combined_op_stats;
  TF_RETURN_IF_ERROR(ConvertMultiXSpaceToCombinedOpStatsWithCache(
      session_snapshot, &combined_op_stats));
  return KernelStatsToDataTableJson(combined_op_stats.kernel_stats_db());
}

absl::StatusOr<std::string> ConvertXSpaceToMemoryProfile(
    const SessionSnapshot& session_snapshot) {
  if (session_snapshot.XSpaceSize() != 1) {
    return tsl::errors::InvalidArgument(
        "Memory profile tool expects only 1 XSpace path but gets ",
        session_snapshot.XSpaceSize());
  }

  std::string json_output;
  google::protobuf::Arena arena;
  TF_ASSIGN_OR_RETURN(XSpace* xspace, session_snapshot.GetXSpace(0, &arena));
  PreprocessSingleHostXSpace(xspace, /*step_grouping=*/true,
                             /*derived_timeline=*/false);
  TF_RETURN_IF_ERROR(ConvertXSpaceToMemoryProfileJson(*xspace, &json_output));
  return json_output;
}

absl::StatusOr<std::string> ConvertMultiXSpacesToPodViewer(
    const SessionSnapshot& session_snapshot) {
  OpStats combined_op_stats;
  TF_RETURN_IF_ERROR(ConvertMultiXSpaceToCombinedOpStatsWithCache(
      session_snapshot, &combined_op_stats));

  std::string json_output;
  tsl::protobuf::util::JsonPrintOptions opts;
  opts.always_print_fields_with_no_presence = true;
  auto encode_status = tsl::protobuf::util::MessageToJsonString(
      ConvertOpStatsToPodViewer(combined_op_stats), &json_output, opts);
  if (!encode_status.ok()) {
    const auto& error_message = encode_status.message();
    return tsl::errors::Internal(
        "Could not convert pod viewer to json. Error: ",
        absl::string_view(error_message.data(), error_message.length()));
  }
  return json_output;
}

absl::StatusOr<std::string> ConvertMultiXSpacesToTfDataBottleneckAnalysis(
    const SessionSnapshot& session_snapshot) {
  CombinedTfDataStats combined_tf_data_stats;
  CombinedTfDataStatsBuilder builder(&combined_tf_data_stats);

  for (int idx = 0; idx < session_snapshot.XSpaceSize(); ++idx) {
    google::protobuf::Arena arena;
    TF_ASSIGN_OR_RETURN(XSpace* xspace,
                        session_snapshot.GetXSpace(idx, &arena));

    PreprocessSingleHostXSpace(xspace, /*step_grouping=*/true,
                               /*derived_timeline=*/false);
    XPlane* host_plane = tsl::profiler::FindMutablePlaneWithName(
        xspace, tsl::profiler::kHostThreadsPlaneName);
    std::string host_name_from_file = session_snapshot.GetHostname(idx);
    if (host_plane == nullptr) {
      return tsl::errors::InvalidArgument(
          "Could not find host XPlane for tf data stats: ",
          host_name_from_file);
    }
    absl::string_view host_name =
        xspace->hostnames_size() ? xspace->hostnames(0) : host_name_from_file;
    builder.Add(host_name, host_plane);
  }
  builder.Finalize();
  return combined_tf_data_stats.SerializeAsString();
}

absl::StatusOr<std::string> ConvertMultiXSpacesToHloStats(
    const SessionSnapshot& session_snapshot) {
  OpStats combined_op_stats;
  TF_RETURN_IF_ERROR(ConvertMultiXSpaceToCombinedOpStatsWithCache(
      session_snapshot, &combined_op_stats));
  hlo_stats::HloStatsDatabase hlo_stats_db =
      ConvertOpStatsToHloStats(combined_op_stats);
  return HloStatsToDataTableJson(hlo_stats_db);
}

absl::StatusOr<std::string> ConvertMultiXSpacesToRooflineModel(
    const SessionSnapshot& session_snapshot) {
  OpStats combined_op_stats;
  TF_RETURN_IF_ERROR(ConvertMultiXSpaceToCombinedOpStatsWithCache(
      session_snapshot, &combined_op_stats));
  RooflineModelDatabase result =
      ConvertOpStatsToRooflineModel(combined_op_stats, true);
  RooflineModelDatabase result_without_infeed_outfeed =
      ConvertOpStatsToRooflineModel(combined_op_stats, false);
  result.mutable_roofline_model_record()->MergeFrom(
      result_without_infeed_outfeed.roofline_model_record());
  return RooflineModelToDataTableJson(result);
}

absl::StatusOr<std::string> ConvertMultiXSpacesToOpProfileViewer(
    const SessionSnapshot& session_snapshot) {
  OpStats combined_op_stats;
  TF_RETURN_IF_ERROR(ConvertMultiXSpaceToCombinedOpStatsWithCache(
      session_snapshot, &combined_op_stats));

  tensorflow::profiler::op_profile::Profile profile;
  ConvertOpStatsToOpProfile(
      combined_op_stats,
      ParseHardwareType(combined_op_stats.run_environment().device_type()),
      profile);
  std::string json_output;
  tsl::protobuf::util::JsonPrintOptions opts;
  opts.always_print_fields_with_no_presence = true;

  auto encode_status =
      tsl::protobuf::util::MessageToJsonString(profile, &json_output, opts);
  if (!encode_status.ok()) {
    const auto& error_message = encode_status.message();
    return tsl::errors::Internal(
        "Could not convert op profile proto to json. Error: ",
        absl::string_view(error_message.data(), error_message.length()));
  }
  return json_output;
}

absl::StatusOr<std::string> PreprocessXSpace(
    const SessionSnapshot& session_snapshot) {
  if (session_snapshot.XSpaceSize() != 1) {
    return tsl::errors::InvalidArgument(
        "PreprocessXSpace tool expects only 1 XSpace path but gets ",
        session_snapshot.XSpaceSize());
  }

  google::protobuf::Arena arena;
  TF_ASSIGN_OR_RETURN(XSpace* xspace, session_snapshot.GetXSpace(0, &arena));
  PreprocessSingleHostXSpace(xspace, /*step_grouping=*/true,
                             /*derived_timeline=*/true);
  return xspace->SerializeAsString();
}

absl::StatusOr<std::string> ConvertDcnCollectiveStatsToToolData(
    const SessionSnapshot& session_snapshot, const ToolOptions& options) {
  // <options> must provide a host_name field.
  std::optional<std::string> hostname =
      GetParam<std::string>(options, "host_name");
  if (!hostname.has_value() || hostname->empty()) {
    return absl::InvalidArgumentError(
        "Cannot find host_name from options for megascale_stats tool.");
  }

  // Load DcnSlackAnalysis for a host.
  TF_ASSIGN_OR_RETURN(
      DcnSlackAnalysis dcnSlackAnalysis,
      GetDcnSlackAnalysisByHostName(session_snapshot, hostname.value()));

  return GenerateMegaScaleJson(dcnSlackAnalysis);
}

absl::StatusOr<std::string> ConvertMultiXSpacesToInferenceStats(
    const SessionSnapshot& session_snapshot, const ToolOptions& options) {
  InferenceStats inference_stats;
  std::string request_column =
      GetParamWithDefault<std::string>(options, "request_column", "");
  std::string batch_column =
      GetParamWithDefault<std::string>(options, "batch_column", "");
  TF_RETURN_IF_ERROR(ConvertMultiXSpaceToInferenceStats(
      session_snapshot, request_column, batch_column, &inference_stats));
  return InferenceStatsToDataTableJson(inference_stats);
}

absl::Status RunMapReduce(xprof::ProfileProcessor* processor,
                          const SessionSnapshot& session_snapshot) {
  std::vector<std::string> map_output_files;
  map_output_files.reserve(session_snapshot.XSpaceSize());
  for (int i = 0; i < session_snapshot.XSpaceSize(); ++i) {
    std::string hostname = session_snapshot.GetHostname(i);
    google::protobuf::Arena arena;
    TF_ASSIGN_OR_RETURN(XSpace * xspace, session_snapshot.GetXSpace(i, &arena));
    TF_ASSIGN_OR_RETURN(std::string map_output_file,
                        processor->Map(session_snapshot, hostname, *xspace));
    map_output_files.push_back(map_output_file);
  }
  return processor->Reduce(session_snapshot, map_output_files);
}

absl::StatusOr<std::string> ConvertMultiXSpacesToSmartSuggestion(
    const SessionSnapshot& session_snapshot) {
  SmartSuggestionEngine engine;
  SmartSuggestionRuleFactory rule_factory;
  RegisterAllRules(&rule_factory);

  auto tool_data_provider =
      std::make_unique<ToolDataProviderImpl>(session_snapshot);
  SignalProvider signal_provider(std::move(tool_data_provider));

  TF_ASSIGN_OR_RETURN(SmartSuggestionReport report,
                      engine.Run(signal_provider, rule_factory));
  std::string json_output;
  tsl::protobuf::util::JsonPrintOptions opts;
  opts.always_print_fields_with_no_presence = true;
  // Perform the Proto to JSON conversion.
  auto encode_status =
      tsl::protobuf::util::MessageToJsonString(report, &json_output, opts);
  if (!encode_status.ok()) {
    const auto& error_message = encode_status.message();
    return tsl::errors::Internal(
        "Could not convert smart suggestion report to json. Error: ",
        absl::string_view(error_message.data(), error_message.length()));
  }
  // Return the generated JSON string.
  return json_output;
}

}  // namespace

absl::StatusOr<std::string> ConvertMultiXSpacesToToolData(
    const SessionSnapshot& session_snapshot, const absl::string_view tool_name,
    const ToolOptions& options) {
  LOG(INFO) << "serving tool: " << tool_name
            << " with options: " << DebugString(options);
  if (tool_name == "trace_viewer" || tool_name == "trace_viewer@") {
    return ConvertXSpaceToTraceEvents(session_snapshot, tool_name, options);
  } else if (tool_name == "overview_page") {
    return ConvertMultiXSpacesToOverviewPage(session_snapshot);
  } else if (tool_name == "input_pipeline_analyzer") {
    return ConvertMultiXSpacesToInputPipeline(session_snapshot);
  } else if (tool_name == "framework_op_stats") {
    return ConvertMultiXSpacesToTfStats(session_snapshot);
  } else if (tool_name == "kernel_stats") {
    return ConvertMultiXSpacesToKernelStats(session_snapshot);
  } else if (tool_name == "memory_profile") {
    return ConvertXSpaceToMemoryProfile(session_snapshot);
  } else if (tool_name == "pod_viewer") {
    return ConvertMultiXSpacesToPodViewer(session_snapshot);
  } else if (tool_name == "op_profile") {
    return ConvertMultiXSpacesToOpProfileViewer(session_snapshot);
  } else if (tool_name == "hlo_stats") {
    return ConvertMultiXSpacesToHloStats(session_snapshot);
  } else if (tool_name == "roofline_model") {
    return ConvertMultiXSpacesToRooflineModel(session_snapshot);
  } else if (tool_name == "memory_viewer" || tool_name == "graph_viewer") {
    return ConvertHloProtoToToolData(session_snapshot, tool_name, options);
  } else if (tool_name == "megascale_stats") {
    return ConvertDcnCollectiveStatsToToolData(session_snapshot, options);
  } else if (tool_name == "tool_names") {
    // Generate the proto cache for hlo_proto tool.
    // This is needed for getting the module list.
    // TODO - b/378923777: Create only when needed.
    TF_ASSIGN_OR_RETURN(bool hlo_proto_status,
                        ConvertMultiXSpaceToHloProto(session_snapshot));
    LOG_IF(WARNING, !hlo_proto_status)
        << "No HLO proto found in XSpace.";
    return GetAvailableToolNames(session_snapshot);
  } else if (tool_name == "_xplane.pb") {  // internal test only.
    return PreprocessXSpace(session_snapshot);
  } else if (tool_name == "inference_profile") {
    return ConvertMultiXSpacesToInferenceStats(session_snapshot, options);
  } else if (tool_name == "smart_suggestion") {
    return ConvertMultiXSpacesToSmartSuggestion(session_snapshot);
  } else {
    return tsl::errors::InvalidArgument(
        "Can not find tool: ", tool_name,
        ". Please update to the latest version of Tensorflow.");
  }
}

absl::StatusOr<std::string> ConvertMultiXSpacesToToolDataWithProfileProcessor(
    const SessionSnapshot& session_snapshot, const absl::string_view tool_name,
    const ToolOptions& options) {
  LOG(INFO) << "serving tool: " << tool_name
            << " with options: " << DebugString(options)
            << " using ProfileProcessor";

  auto processor =
      xprof::ProfileProcessorFactory::GetInstance().Create(tool_name, options);
  if (!processor) {
    return tsl::errors::InvalidArgument(
        "Can not find tool: ", tool_name,
        ". Please update to the latest version of Tensorflow.");
  }

  if (processor->ShouldUseWorkerService(session_snapshot)) {
    // This branch is for the Map/Reduce flow, potentially distributed in the
    // future.
    TF_RETURN_IF_ERROR(RunMapReduce(processor.get(), session_snapshot));
  } else {
    // This branch is for processing the session directly.
    TF_RETURN_IF_ERROR(processor->ProcessSession(session_snapshot));
  }
  return processor->GetData();
}

}  // namespace profiler
}  // namespace tensorflow
