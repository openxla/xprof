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

#include "xprof/convert/xplane_to_tools_data.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "google/protobuf/arena.h"
#include "google/protobuf/json/json.h"
#include "xla/service/hlo.pb.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/file_system.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/profiler/convert/xplane_to_trace_events.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/framework_op_stats_processor.h"
#include "xprof/convert/hlo_stats_processor.h"
#include "xprof/convert/hlo_to_tools_data.h"
#include "xprof/convert/input_pipeline_processor.h"
#include "xprof/convert/kernel_stats_processor.h"
#include "xprof/convert/multi_xplanes_to_op_stats.h"
#include "xprof/convert/multi_xspace_to_inference_stats.h"
#include "xprof/convert/op_stats_to_op_profile.h"
#include "xprof/convert/overview_page_processor.h"
#include "xprof/convert/pod_viewer_processor.h"
#include "xprof/convert/preprocess_single_host_xplane.h"
#include "xprof/convert/process_megascale_dcn.h"
#include "xprof/convert/repository.h"
#include "xprof/convert/roofline_model_processor.h"
#include "xprof/convert/smart_suggestion_processor.h"
#include "xprof/convert/tool_options.h"
#include "xprof/convert/trace_view_options.h"
#include "xprof/convert/trace_viewer/trace_events.h"
#include "xprof/convert/trace_viewer/trace_events_to_json.h"
#include "xprof/convert/trace_viewer/trace_options.h"
#include "xprof/convert/unified_session_snapshot.h"
#include "xprof/convert/xplane_to_dcn_collective_stats.h"
#include "xprof/convert/xplane_to_hlo.h"
#include "xprof/convert/xplane_to_memory_profile.h"
#include "xprof/convert/xplane_to_perf_counters.h"
#include "xprof/convert/xplane_to_tool_names.h"
#include "xprof/convert/xplane_to_trace_container.h"
#include "xprof/convert/xplane_to_utilization_viewer.h"
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

absl::StatusOr<std::string> ConvertXSpaceToTraceEvents(
    const SessionSnapshot& session_snapshot, const absl::string_view tool_name,
    const ToolOptions& options) {
  if (session_snapshot.XSpaceSize() != 1) {
    return absl::InvalidArgumentError(
        absl::StrCat("Trace events tool expects only 1 XSpace path but gets ",
                     session_snapshot.XSpaceSize()));
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
    TF_ASSIGN_OR_RETURN(TraceViewOption trace_option,
                        GetTraceViewOption(options));
    tensorflow::profiler::TraceOptions profiler_trace_options =
        TraceOptionsFromToolOptions(options);
    std::string host_name = session_snapshot.GetHostname(0);
    std::optional<std::string> trace_events_sstable_path =
        session_snapshot.MakeHostDataFilePath(StoredDataType::TRACE_LEVELDB,
                                              host_name);
    std::optional<std::string> trace_events_metadata_sstable_path =
        session_snapshot.MakeHostDataFilePath(
            StoredDataType::TRACE_EVENTS_METADATA_LEVELDB, host_name);
    std::optional<std::string> trace_events_prefix_trie_sstable_path =
        session_snapshot.MakeHostDataFilePath(
            StoredDataType::TRACE_EVENTS_PREFIX_TRIE_LEVELDB, host_name);
    if (!trace_events_sstable_path || !trace_events_metadata_sstable_path ||
        !trace_events_prefix_trie_sstable_path) {
      return absl::UnimplementedError(
          "streaming trace viewer hasn't been supported in Cloud AI");
    }
    if (!tsl::Env::Default()->FileExists(*trace_events_sstable_path).ok()) {
      if (profiler_trace_options.enable_legacy_dcn) {
        ProcessMegascaleDcn(xspace);
      }
      TraceEventsContainer trace_container;
      // No-op method which will be deprecated in the future, thus added
      // /*host_id=*/1 as a placeholder for now.
      ConvertXSpaceToTraceEventsContainer(host_name, *xspace, &trace_container);
      std::unique_ptr<tsl::WritableFile> trace_events_file;
      TF_RETURN_IF_ERROR(tsl::Env::Default()->NewWritableFile(
          *trace_events_sstable_path, &trace_events_file));
      std::unique_ptr<tsl::WritableFile> trace_events_metadata_file;
      TF_RETURN_IF_ERROR(tsl::Env::Default()->NewWritableFile(
          *trace_events_metadata_sstable_path, &trace_events_metadata_file));
      std::unique_ptr<tsl::WritableFile> trace_events_prefix_trie_file;
      TF_RETURN_IF_ERROR(tsl::Env::Default()->NewWritableFile(
          *trace_events_prefix_trie_sstable_path,
          &trace_events_prefix_trie_file));
      TF_RETURN_IF_ERROR(trace_container.StoreAsLevelDbTables(
          std::move(trace_events_file),
          std::move(trace_events_metadata_file),
          std::move(trace_events_prefix_trie_file)
      ));
    }
    TraceEventsLevelDbFilePaths file_paths;
    file_paths.trace_events_file_path = *trace_events_sstable_path;
    file_paths.trace_events_metadata_file_path =
        *trace_events_metadata_sstable_path;
    file_paths.trace_events_prefix_trie_file_path =
        *trace_events_prefix_trie_sstable_path;
    TraceEventsContainer trace_container;
    TF_RETURN_IF_ERROR(LoadTraceEventsContainer(
        file_paths, trace_option, profiler_trace_options, &trace_container));
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

// TODO(b/442320796) - Remove this once ProfileProcessor is the default.
absl::StatusOr<std::string> ConvertMultiXSpacesToOverviewPage(
    const SessionSnapshot& session_snapshot) {
  xprof::OverviewPageProcessor processor({});
  TF_RETURN_IF_ERROR(processor.ProcessSession(session_snapshot, {}));
  return processor.GetData();
}

absl::StatusOr<std::string> ConvertMultiXSpacesToInputPipeline(
    const SessionSnapshot& session_snapshot) {
  xprof::InputPipelineProcessor processor({});
  TF_RETURN_IF_ERROR(processor.ProcessSession(session_snapshot, {}));
  return processor.GetData();
}

absl::StatusOr<std::string> ConvertMultiXSpacesToTfStats(
    const SessionSnapshot& session_snapshot) {
  xprof::FrameworkOpStatsProcessor processor({});
  TF_RETURN_IF_ERROR(processor.ProcessSession(session_snapshot, {}));
  return processor.GetData();
}

absl::StatusOr<std::string> ConvertMultiXSpacesToKernelStats(
    const SessionSnapshot& session_snapshot) {
  xprof::KernelStatsProcessor processor({});
  TF_RETURN_IF_ERROR(processor.ProcessSession(session_snapshot, {}));
  return processor.GetData();
}

absl::StatusOr<std::string> ConvertXSpaceToMemoryProfile(
    const SessionSnapshot& session_snapshot) {
  if (session_snapshot.XSpaceSize() != 1) {
    return absl::InvalidArgumentError(
        absl::StrCat("Memory profile tool expects only 1 XSpace path but gets ",
                     session_snapshot.XSpaceSize()));
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
  xprof::PodViewerProcessor processor({});
  TF_RETURN_IF_ERROR(processor.ProcessSession(session_snapshot, {}));
  return processor.GetData();
}

absl::StatusOr<std::string> ConvertMultiXSpacesToHloStats(
    const SessionSnapshot& session_snapshot) {
  xprof::HloStatsProcessor processor({});
  TF_RETURN_IF_ERROR(processor.ProcessSession(session_snapshot, {}));
  return processor.GetData();
}

absl::StatusOr<std::string> ConvertMultiXSpacesToRooflineModel(
    const SessionSnapshot& session_snapshot) {
  xprof::RooflineModelProcessor processor({});
  TF_RETURN_IF_ERROR(processor.ProcessSession(session_snapshot, {}));
  return processor.GetData();
}

absl::StatusOr<std::string> ConvertMultiXSpacesToOpProfileViewer(
    const SessionSnapshot& session_snapshot, const ToolOptions& options) {
  OpStats combined_op_stats;
  TF_RETURN_IF_ERROR(ConvertMultiXSpaceToCombinedOpStatsWithCache(
      session_snapshot, &combined_op_stats));

  tensorflow::profiler::op_profile::Profile profile;
  auto group_by = tensorflow::profiler::GetOpProfileGrouping(options);
  ConvertOpStatsToOpProfile(
      combined_op_stats,
      ParseHardwareType(combined_op_stats.run_environment().device_type()),
      profile, /*op_profile_limit=*/100, group_by);
  std::string json_output;
  tsl::protobuf::util::JsonPrintOptions opts;
  opts.always_print_fields_with_no_presence = true;

  auto encode_status =
      tsl::protobuf::util::MessageToJsonString(profile, &json_output, opts);
  if (!encode_status.ok()) {
    const auto& error_message = encode_status.message();
    return absl::InternalError(absl::StrCat(
        "Could not convert op profile proto to json. Error: ", error_message));
  }
  return json_output;
}

absl::StatusOr<std::string> PreprocessXSpace(
    const SessionSnapshot& session_snapshot) {
  if (session_snapshot.XSpaceSize() != 1) {
    return absl::InvalidArgumentError(absl::StrCat(
        "PreprocessXSpace tool expects only 1 XSpace path but gets ",
        session_snapshot.XSpaceSize()));
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

absl::StatusOr<std::string> ConvertMultiXSpacesToSmartSuggestion(
    const SessionSnapshot& session_snapshot, const ToolOptions& options) {
  xprof::SmartSuggestionProcessor processor(options);
  TF_RETURN_IF_ERROR(processor.ProcessSession(session_snapshot, options));
  return processor.GetData();
}

}  // namespace

absl::StatusOr<std::string> ConvertMultiXSpacesToToolData(
    const SessionSnapshot& session_snapshot, const absl::string_view tool_name,
    const ToolOptions& options) {
  absl::string_view session_id = session_snapshot.GetSessionRunDir();
  LOG(INFO) << "serving tool: " << tool_name
            << " with options: " << DebugString(options)
            << " session_id: " << session_id;
  absl::Time start_time = absl::Now();

  absl::StatusOr<std::string> tool_data;
  if (tool_name == "trace_viewer" || tool_name == "trace_viewer@") {
    tool_data =
        ConvertXSpaceToTraceEvents(session_snapshot, tool_name, options);
  } else if (tool_name == "overview_page") {
    tool_data = ConvertMultiXSpacesToOverviewPage(session_snapshot);
  } else if (tool_name == "input_pipeline_analyzer") {
    tool_data = ConvertMultiXSpacesToInputPipeline(session_snapshot);
  } else if (tool_name == "framework_op_stats") {
    tool_data = ConvertMultiXSpacesToTfStats(session_snapshot);
  } else if (tool_name == "kernel_stats") {
    tool_data = ConvertMultiXSpacesToKernelStats(session_snapshot);
  } else if (tool_name == "memory_profile") {
    tool_data = ConvertXSpaceToMemoryProfile(session_snapshot);
  } else if (tool_name == "pod_viewer") {
    tool_data = ConvertMultiXSpacesToPodViewer(session_snapshot);
  } else if (tool_name == "op_profile") {
    tool_data = ConvertMultiXSpacesToOpProfileViewer(session_snapshot, options);
  } else if (tool_name == "hlo_stats") {
    tool_data = ConvertMultiXSpacesToHloStats(session_snapshot);
  } else if (tool_name == "roofline_model") {
    tool_data = ConvertMultiXSpacesToRooflineModel(session_snapshot);
  } else if (tool_name == "memory_viewer" || tool_name == "graph_viewer") {
    tool_data = ConvertHloProtoToToolData(session_snapshot, tool_name, options);
  } else if (tool_name == "megascale_stats") {
    tool_data = ConvertDcnCollectiveStatsToToolData(session_snapshot, options);
  } else if (tool_name == "perf_counters") {
    tool_data = ConvertMultiXSpacesToPerfCounters(session_snapshot);
  } else if (tool_name == "tool_names") {
    // Generate the proto cache for hlo_proto tool.
    // This is needed for getting the module list.
    // TODO - b/378923777: Create only when needed.
    TF_ASSIGN_OR_RETURN(bool hlo_proto_status,
                        ConvertMultiXSpaceToHloProto(session_snapshot));
    LOG_IF(WARNING, !hlo_proto_status)
        << "No HLO proto found in XSpace.";
    tool_data = GetAvailableToolNames(session_snapshot);
  } else if (tool_name == "_xplane.pb") {  // internal test only.
    tool_data = PreprocessXSpace(session_snapshot);
  } else if (tool_name == "inference_profile") {
    tool_data = ConvertMultiXSpacesToInferenceStats(session_snapshot, options);
  } else if (tool_name == "smart_suggestion") {
    tool_data = ConvertMultiXSpacesToSmartSuggestion(session_snapshot, options);
  } else if (tool_name == "utilization_viewer") {
    if (session_snapshot.XSpaceSize() != 1) {
      tool_data = absl::InvalidArgumentError(absl::StrCat(
          "Utilization viewer tool expects only 1 XSpace path but gets ",
          session_snapshot.XSpaceSize()));
    } else {
      google::protobuf::Arena arena;
      auto xspace = session_snapshot.GetXSpace(0, &arena);
      if (xspace.ok()) {
        PreprocessSingleHostXSpace(*xspace, /*step_grouping=*/true,
                                   /*derived_timeline=*/true);
        tool_data = xprof::ConvertXSpaceToUtilizationViewer(**xspace);
      } else {
        tool_data = xspace.status();
      }
    }
  } else {
    tool_data = absl::InvalidArgumentError(
        absl::StrCat("Can not find tool: ", tool_name,
                     ". Please update to the latest version of Tensorflow."));
  }

  LOG(INFO) << "serving tool: " << tool_name << " session_id: " << session_id
            << " duration: " << absl::Now() - start_time;
  return tool_data;
}

}  // namespace profiler
}  // namespace tensorflow
