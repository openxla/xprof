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
#include "xprof/convert/multi_xspace_to_inference_stats.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/arena.h"
#include "google/protobuf/repeated_ptr_field.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/profiler/utils/device_utils.h"
#include "xla/tsl/profiler/utils/group_events.h"
#include "xla/tsl/profiler/utils/math_utils.h"
#include "xla/tsl/profiler/utils/tf_xplane_visitor.h"
#include "xla/tsl/profiler/utils/tpu_xplane_utils.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_utils.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/data_table_utils.h"
#include "xprof/convert/inference_stats.h"
#include "xprof/convert/inference_stats_combiner.h"
#include "xprof/convert/inference_stats_grouping.h"
#include "xprof/convert/inference_stats_sampler.h"
#include "xprof/convert/preprocess_single_host_xplane.h"
#include "xprof/convert/repository.h"
#include "xprof/convert/url_utils.h"
#include "xprof/convert/xplane_to_step_events.h"
#include "plugin/xprof/protobuf/inference_stats.pb.h"
#include "xprof/utils/event_span.h"

namespace tensorflow::profiler {

namespace {
using ::tensorflow::profiler::DataTable;
using tsl::profiler::FindMutablePlanesWithPrefix;
using tsl::profiler::FindMutablePlaneWithName;
using tsl::profiler::PicoToMilli;

SampledInferenceStatsProto GetSampledInferenceStatsProto(
    const InferenceStats& inference_stats, absl::string_view request_column,
    absl::string_view batch_column) {
  SampledInferenceStatsProto result;
  SampledInferenceStatsProto sampled_stats =
      SampleInferenceStats(request_column, batch_column, inference_stats);
  for (const auto& [model_index, samples] :
       sampled_stats.sampled_inference_stats_per_model()) {
    SampledPerModelInferenceStatsProto per_model_stats;
    for (const auto& request : samples.sampled_requests()) {
      RequestDetail request_detail = request;
      request_detail.set_percentile(request.percentile());
      *per_model_stats.add_sampled_requests() = request_detail;
    }
    for (const auto& batch : samples.sampled_batches()) {
      BatchDetail batch_detail = batch;
      batch_detail.set_percentile(batch.percentile());
      *per_model_stats.add_sampled_batches() = batch_detail;
    }
    result.mutable_sampled_inference_stats_per_model()->insert(
        {model_index, per_model_stats});
  }
  return result;
}
}  // namespace

StepEvents GetNonOverlappedStepEvents(XSpace* xspace) {
  StepEvents non_overlapped_step_events;

  std::vector<XPlane*> device_traces =
      FindMutablePlanesWithPrefix(xspace, tsl::profiler::kGpuPlanePrefix);
  if (device_traces.empty()) return non_overlapped_step_events;

  StepEvents device_step_events;
  StepEvents host_step_events;
  for (XPlane* device_trace : device_traces) {
    StepEvents events = ConvertDeviceTraceXPlaneToStepEvents(*device_trace);
    UnionCombineStepEvents(events, &device_step_events);
  }

  XPlane* host_plane =
      FindMutablePlaneWithName(xspace, tsl::profiler::kHostThreadsPlaneName);
  if (host_plane != nullptr) {
    host_step_events =
        ConvertHostThreadsXPlaneToStepEvents(*host_plane, &device_step_events);
  }
  StepEvents overlapped_step_events;
  UnionCombineStepEvents(device_step_events, &overlapped_step_events);
  UnionCombineStepEvents(host_step_events, &overlapped_step_events);
  non_overlapped_step_events =
      ToNonOverlappedStepEvents(overlapped_step_events);
  return non_overlapped_step_events;
}

absl::Status ConvertMultiXSpaceToInferenceStats(
    const SessionSnapshot& session_snapshot, absl::string_view request_column,
    absl::string_view batch_column, InferenceStats* inference_stats) {
  for (int i = 0; i < session_snapshot.XSpaceSize(); ++i) {
    google::protobuf::Arena arena;
    TF_ASSIGN_OR_RETURN(XSpace* xspace, session_snapshot.GetXSpace(i, &arena));
    tsl::profiler::GroupMetadataMap metadata_map;
    InferenceStats inference_stats_per_host;
    std::vector<XPlane*> device_traces =
        tsl::profiler::FindMutableTensorCorePlanes(xspace);
    PreprocessSingleHostXSpace(xspace, /*step_grouping=*/true,
                               /*derived_timeline=*/false, &metadata_map);
    StepEvents non_overlapped_step_events = GetNonOverlappedStepEvents(xspace);
    GenerateInferenceStats(
        device_traces, non_overlapped_step_events, metadata_map, *xspace,
        tsl::profiler::DeviceType::kTpu, i, &inference_stats_per_host);
    CombineInferenceStatsResult(i, inference_stats_per_host, inference_stats);
  }
  RegroupInferenceStatsByModel(inference_stats);
  *inference_stats->mutable_sampled_inference_stats() =
      GetSampledInferenceStatsProto(*inference_stats, request_column,
                                    batch_column);
  return absl::OkStatus();
}

// Calculate batching efficiency.
double CalculateBatchingEfficiency(
    const tensorflow::profiler::BatchDetail& batch) {
  return tsl::profiler::SafeDivide(
      static_cast<double>(batch.batch_size_after_padding() -
                          batch.padding_amount()),
      static_cast<double>(batch.batch_size_after_padding()));
}

// Generates a string that shows the percentile and time spent on linearize and
// delinearize a tensor pattern.
std::string GenerateTensorPatternPercentileText(
    const google::protobuf::RepeatedPtrField<
        tensorflow::profiler::TensorTransferAggregatedResult::
            TensorPatternResult::PercentileTime>& percentiles) {
  return absl::StrJoin(
      percentiles, "<br>",
      [](std::string* out, const auto& percentile_and_time) {
        out->append(absl::StrFormat("%.2f", percentile_and_time.percentile()));
        out->append("%: ");
        out->append(std::to_string(PicoToMilli(percentile_and_time.time_ps())));
        out->append("ms");
      });
}

// Converts tensor patterns from proto format to data table.
DataTable CreateTensorPatternDataTable(
    const tensorflow::profiler::PerModelInferenceStats& per_model_stats,
    const tensorflow::profiler::TensorPatternDatabase& tensor_pattern_db) {
  DataTable tensor_pattern_data_table;
  tensor_pattern_data_table.AddColumn(TableColumn("id", "number", "ID"));
  tensor_pattern_data_table.AddColumn(
      TableColumn("tensor_pattern", "string", "Tensor Pattern"));
  tensor_pattern_data_table.AddColumn(
      TableColumn("count", "number", "Number of Occurrence"));
  tensor_pattern_data_table.AddColumn(
      TableColumn("percentile", "string", "Linearize/Delinearize latency"));
  const auto& aggregated_results =
      per_model_stats.tensor_transfer_aggregated_result();
  for (int i = 0;
       i < static_cast<int>(aggregated_results.tensor_pattern_results_size());
       i++) {
    const auto& result = aggregated_results.tensor_pattern_results(i);
    if (result.tensor_pattern_index() >=
        tensor_pattern_db.tensor_pattern_size()) {
      LOG(ERROR) << "Can not find tensor pattern: "
                 << result.tensor_pattern_index();
      continue;
    }
    TableRow* row = tensor_pattern_data_table.AddRow();
    row->AddNumberCell(i);
    row->AddTextCell(
        tensor_pattern_db.tensor_pattern(result.tensor_pattern_index()));
    row->AddNumberCell(result.count());
    row->AddTextCell(GenerateTensorPatternPercentileText(
        result.linearize_delinearize_percentile_time()));
  }
  return tensor_pattern_data_table;
}

void AddRequestDetails(const tensorflow::profiler::RequestDetail& request,
                       const absl::string_view session_id,
                       std::string percentile, std::string request_id,
                       bool has_batching, bool is_tpu,
                       const absl::string_view throughput,
                       DataTable* request_data_table) {
  TableRow* row = request_data_table->AddRow();
  row->AddTextCell(std::move(percentile));
  row->AddTextCell(std::move(request_id));
  row->AddNumberCell(
      PicoToMilli(request.end_time_ps() - request.start_time_ps()));

  if (has_batching) {
    row->AddNumberCell(request.batching_request_size());
    row->AddNumberCell(PicoToMilli(request.batching_request_delay_ps()));
    row->AddTextCell(throughput);
  }

  if (is_tpu) {
    row->AddNumberCell(PicoToMilli(request.host_preprocessing_ps()));
    row->AddNumberCell(PicoToMilli(request.host_runtime_ps()));
    row->AddNumberCell(PicoToMilli(request.write_to_device_time_ps()));
    row->AddNumberCell(PicoToMilli(request.read_from_device_time_ps()));
    row->AddNumberCell(PicoToMilli(request.device_time_ps()));
    row->AddNumberCell(PicoToMilli(request.host_postprocessing_ps()));
    row->AddNumberCell(PicoToMilli(request.idle_time_ps()));
  }
  row->AddTextCell(Linkify(
      GenerateTraceViewerUrl(session_id, request.request_id(),
                             request.host_id(), request.related_batch_ids()),
      "link"));
}

// Convert requests from proto format to data table.
// If <has_batching> is true, add the columns that only makes sense in batching
// mode.
DataTable CreateRequestDataTable(
    const tensorflow::profiler::PerModelInferenceStats& per_model_stats,
    const SampledPerModelInferenceStatsProto& sampled_per_model_stats,
    const absl::string_view session_id, bool has_batching, bool is_tpu) {
  DataTable request_data_table;
  request_data_table.AddCustomProperty(
      "throughput",
      absl::StrFormat("%.1lf", per_model_stats.request_throughput()));
  request_data_table.AddCustomProperty(
      "averageLatencyMs",
      absl::StrFormat("%.3lf",
                      tsl::profiler::MicroToMilli(
                          per_model_stats.request_average_latency_us())));

  // Percentile and request ID columns are text column because we want to
  // display "Average" and "N/A" in the average record.
  request_data_table.AddColumn(
      TableColumn("percentile", "string", "Percentile"));
  request_data_table.AddColumn(
      TableColumn("request_id", "string", "Request ID"));
  request_data_table.AddColumn(TableColumn("latency_ms", "number", "Latency"));
  if (has_batching) {
    request_data_table.AddColumn(
        TableColumn("batching_request_size", "number", "Request size"));
    request_data_table.AddColumn(
        TableColumn("host_batch_formation", "number", "Host batch formation"));
    request_data_table.AddColumn(
        TableColumn("throughput", "string", "Throughput"));
  }
  if (is_tpu) {
    request_data_table.AddColumn(
        TableColumn("host_preprocessing", "number", "Host preprocess"));
    request_data_table.AddColumn(
        TableColumn("host_runtime", "number", "Host runtime"));
    request_data_table.AddColumn(
        TableColumn("data_transfer_h2d", "number", "Data transfer H2D"));
    request_data_table.AddColumn(
        TableColumn("data_transfer_d2h", "number", "Data transfer D2H"));
    request_data_table.AddColumn(
        TableColumn("device_compute", "number", "Device compute"));
    request_data_table.AddColumn(
        TableColumn("host_postprocess", "number", "Host postprocess"));
    request_data_table.AddColumn(
        TableColumn("idle time", "number", "Idle time"));
  }
  request_data_table.AddColumn(
      TableColumn("trace_viewer_url", "string", "Trace Viewer URL"));
  for (const auto& request_details :
       sampled_per_model_stats.sampled_requests()) {
    AddRequestDetails(request_details, session_id,
                      absl::StrFormat("%.3f", request_details.percentile()),
                      absl::StrCat(request_details.request_id()), has_batching,
                      is_tpu, "N/A", &request_data_table);
  }
  // TODO: remove batch size aggregation from request table.
  if (per_model_stats.per_batch_size_aggregated_result_size()) {
    for (const auto& per_batch_size_result :
         per_model_stats.per_batch_size_aggregated_result()) {
      AddRequestDetails(
          per_batch_size_result.aggregated_request_result(), session_id,
          absl::StrCat("Batch size ", per_batch_size_result.batch_size()),
          "N/A", has_batching, is_tpu,
          absl::StrFormat("%.1f", per_batch_size_result.request_throughput()),
          &request_data_table);
    }
  }
  if (per_model_stats.has_aggregated_request_detail()) {
    AddRequestDetails(
        per_model_stats.aggregated_request_detail(), session_id, "Average",
        "N/A", has_batching, is_tpu,
        absl::StrFormat("%.1f", per_model_stats.request_throughput()),
        &request_data_table);
  }
  return request_data_table;
}

void AddBatchDetails(const tensorflow::profiler::BatchDetail& batch,
                     const absl::string_view session_id, std::string percentile,
                     std::string batch_id, const absl::string_view throughput,
                     DataTable* batch_data_table) {
  TableRow* row = batch_data_table->AddRow();
  row->AddTextCell(std::move(percentile));
  row->AddTextCell(std::move(batch_id));
  row->AddNumberCell(PicoToMilli(batch.end_time_ps() - batch.start_time_ps()));
  row->AddNumberCell(batch.padding_amount());
  row->AddNumberCell(batch.batch_size_after_padding());
  row->AddNumberCell(tensorflow::profiler::CalculateBatchingEfficiency(batch));
  row->AddNumberCell(PicoToMilli(batch.batch_delay_ps()));
  row->AddTextCell(throughput);
  row->AddNumberCell(PicoToMilli(batch.device_time_ps()));
  row->AddTextCell(Linkify(
      GenerateTraceViewerUrl(session_id, batch.batch_id(), batch.host_id(),
                             batch.related_request_ids()),
      "link"));
}

// Convert batches from proto format to data table.
DataTable CreateBatchDataTable(
    const tensorflow::profiler::PerModelInferenceStats& per_model_stats,
    const SampledPerModelInferenceStatsProto& sampled_per_model_stats,
    const tensorflow::profiler::ModelIdDatabase& model_id_db,
    absl::string_view model_id, const absl::string_view session_id) {
  DataTable batch_data_table;

  // Percentile and batch ID columns are text column because we want to display
  // "Average" and "N/A" in the average record.
  std::vector<std::vector<std::string>> kColumns = {
      {"percentile", "string", "Percentile"},
      {"batch_id", "string", "Batch ID"},
      {"latency", "number", "Latency"},
      {"padding_amount", "number", "Padding amount"},
      {"batch_size_after_padding", "number", "Batch size after padding"},
      {"batching_efficiency", "number", "Batching efficiency"},
      {"batching_delay_us", "number", "Batching delay"},
      {"throughput", "string", "Throughput"},
      {"device_compute", "number", "Device compute"},
      {"trace_viewer_url", "string", "Trace Viewer URL"}};

  for (const auto& column : kColumns) {
    batch_data_table.AddColumn(TableColumn(column[0], column[1], column[2]));
  }

  for (const auto& batch : sampled_per_model_stats.sampled_batches()) {
    AddBatchDetails(batch, session_id,
                    absl::StrFormat("%.3f", batch.percentile()),
                    absl::StrCat(batch.batch_id()), "N/A", &batch_data_table);
  }
  if (per_model_stats.per_batch_size_aggregated_result_size()) {
    for (const auto& per_batch_size_result :
         per_model_stats.per_batch_size_aggregated_result()) {
      AddBatchDetails(
          per_batch_size_result.aggregated_batch_result(), session_id,
          absl::StrCat("Batch size ", per_batch_size_result.batch_size()),
          "N/A",
          absl::StrFormat("%.1f", per_batch_size_result.batch_throughput()),
          &batch_data_table);
    }
  }
  if (per_model_stats.has_aggregated_batch_detail()) {
    AddBatchDetails(per_model_stats.aggregated_batch_detail(), session_id,
                    "Aggregated", "N/A",
                    absl::StrFormat("%.1f", per_model_stats.batch_throughput()),
                    &batch_data_table);
  }

  batch_data_table.AddCustomProperty(
      "throughput",
      absl::StrFormat("%.1lf", per_model_stats.batch_throughput()));
  batch_data_table.AddCustomProperty(
      "averageLatencyMs",
      absl::StrFormat("%.3lf",
                      tsl::profiler::MicroToMilli(
                          per_model_stats.batch_average_latency_us())));
  const auto params_it = model_id_db.id_to_batching_params().find(model_id);
  if (params_it != model_id_db.id_to_batching_params().end()) {
    const auto& params = params_it->second;
    batch_data_table.AddCustomProperty("hasBatchingParam", "true");
    batch_data_table.AddCustomProperty(
        "batchingParamNumBatchThreads",
        absl::StrCat(params.num_batch_threads()));
    batch_data_table.AddCustomProperty("batchingParamMaxBatchSize",
                                       absl::StrCat(params.max_batch_size()));
    batch_data_table.AddCustomProperty(
        "batchingParamBatchTimeoutMicros",
        absl::StrCat(params.batch_timeout_micros()));
    batch_data_table.AddCustomProperty(
        "batchingParamMaxEnqueuedBatches",
        absl::StrCat(params.max_enqueued_batches()));
    batch_data_table.AddCustomProperty("batchingParamAllowedBatchSizes",
                                       params.allowed_batch_sizes());
  } else {
    batch_data_table.AddCustomProperty("hasBatchingParam", "false");
  }

  return batch_data_table;
}

void GeneratePerModelInferenceDataTables(
    const InferenceStats& inference_stats,
    const SampledInferenceStatsProto& result,
    std::vector<std::string>& sorted_model_ids,
    std::vector<DataTable>& per_model_inference_tables,
    const bool& has_batching, const bool& has_tensor_pattern,
    absl::string_view session_id, bool is_tpu) {
  for (const std::string& model_id : sorted_model_ids) {
    const auto model_index_it =
        inference_stats.model_id_db().id_to_index().find(model_id);
    if (model_index_it == inference_stats.model_id_db().id_to_index().end()) {
      LOG(ERROR) << "Failed to find model id: " << model_id << " in the "
                 << "ModelIdDatabase";
      continue;
    }
    const auto& model_index = model_index_it->second;

    const auto all_stats_it =
        inference_stats.inference_stats_per_model().find(model_index);
    if (all_stats_it == inference_stats.inference_stats_per_model().end()) {
      LOG(ERROR) << "Failed to find inference stats for model index: "
                 << model_index << ", model id: " << model_id;
      continue;
    }
    const auto& all_stats = all_stats_it->second;
    const auto sampled_stats_it =
        result.sampled_inference_stats_per_model().find(model_index);
    if (sampled_stats_it == result.sampled_inference_stats_per_model().end()) {
      LOG(ERROR) << "Failed to find sampled inference stats for model index: "
                 << model_index << ", model id: " << model_id;
      continue;
    }
    const auto& sampled_stats = sampled_stats_it->second;

    per_model_inference_tables.push_back(CreateRequestDataTable(
        all_stats, sampled_stats, session_id, has_batching, is_tpu));

    // If batching is used for this inference job, add a second table for
    // batching.
    if (has_batching) {
      per_model_inference_tables.push_back(CreateBatchDataTable(
          all_stats, sampled_stats, inference_stats.model_id_db(), model_id,
          session_id));
    }

    // If there are tensor patterns recorded in InferenceStats, add a third
    // table for tensor pattern analysis.
    if (has_tensor_pattern) {
      per_model_inference_tables.push_back(CreateTensorPatternDataTable(
          all_stats, inference_stats.tensor_pattern_db()));
    }
  }
}

std::unique_ptr<DataTable> GenerateMetaDataTable(
    const InferenceStats& inference_stats,
    std::vector<std::string>& sorted_model_ids, bool& has_batching,
    bool& has_tensor_pattern) {
  // Record whether this inference profile has batching.
  has_batching = false;
  for (const auto& per_model_stats :
       inference_stats.inference_stats_per_model()) {
    if (!per_model_stats.second.batch_details().empty()) {
      has_batching = true;
      break;
    }
  }
  auto data_table = std::make_unique<DataTable>();
  data_table->AddCustomProperty("hasBatching", has_batching ? "true" : "false");
  has_tensor_pattern =
      !inference_stats.tensor_pattern_db().tensor_pattern().empty();
  data_table->AddCustomProperty("hasTensorPattern", "false");
  data_table->AddColumn(TableColumn("model_name", "string", "Model Name"));
  for (const auto& model_id : sorted_model_ids) {
    TableRow* row = data_table->AddRow();
    row->AddTextCell(model_id);
  }
  return data_table;
}

// Sorts the model_ids in alphabetic order so that the model_id '' will appear
// as the first one.
void SortModelIds(const InferenceStats& inference_stats,
                  std::vector<std::string>& sorted_model_ids) {
  if (!sorted_model_ids.empty()) return;
  for (const std::string& model_id : inference_stats.model_id_db().ids()) {
    sorted_model_ids.push_back(model_id);
  }
  absl::c_sort(sorted_model_ids);
}

std::string InferenceStatsToDataTableJson(
    const InferenceStats& inference_stats) {
  std::vector<std::string> sorted_model_ids;
  SortModelIds(inference_stats, sorted_model_ids);

  bool has_batching, has_tensor_pattern;
  std::unique_ptr<DataTable> meta_table = GenerateMetaDataTable(
      inference_stats, sorted_model_ids, has_batching, has_tensor_pattern);

  std::vector<DataTable> per_model_inference_tables;
  GeneratePerModelInferenceDataTables(
      inference_stats, inference_stats.sampled_inference_stats(),
      sorted_model_ids, per_model_inference_tables, has_batching,
      has_tensor_pattern, "", true);
  std::vector<std::string> data_table_json_vec;
  data_table_json_vec.push_back(meta_table->ToJson());
  for (auto& table : per_model_inference_tables) {
    data_table_json_vec.push_back(table.ToJson());
  }
  std::string data_table_json = absl::StrJoin(data_table_json_vec, ",");
  return absl::StrCat("[", data_table_json, "]");
}
}  // namespace tensorflow::profiler
