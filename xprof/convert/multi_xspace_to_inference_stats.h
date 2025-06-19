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
#ifndef XPROF_CONVERT_MULTI_XSPACE_TO_INFERENCE_STATS_H_
#define XPROF_CONVERT_MULTI_XSPACE_TO_INFERENCE_STATS_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xprof/convert/data_table_utils.h"
#include "xprof/convert/repository.h"
#include "plugin/xprof/protobuf/inference_stats.pb.h"
#include "xprof/utils/event_span.h"

namespace tensorflow::profiler {
// Get non overlapped step events from xspace for GPU.
StepEvents GetNonOverlappedStepEvents(XSpace* xspace);

absl::Status ConvertMultiXSpaceToInferenceStats(
    const SessionSnapshot& session_snapshot, absl::string_view request_column,
    absl::string_view batch_column, InferenceStats* inference_stats);

void SortModelIds(const InferenceStats& inference_stats,
                  std::vector<std::string>& sorted_model_ids);

double CalculateBatchingEfficiency(
    const tensorflow::profiler::BatchDetail& batch);

std::unique_ptr<DataTable> GenerateMetaDataTable(
    const InferenceStats& inference_stats,
    std::vector<std::string>& sorted_model_ids, bool& has_batching,
    bool& has_tensor_pattern);

void GeneratePerModelInferenceDataTables(
    const InferenceStats& inference_stats,
    const SampledInferenceStatsProto& result,
    std::vector<std::string>& sorted_model_ids,
    std::vector<DataTable>& per_model_inference_tables,
    const bool& has_batching, const bool& has_tensor_pattern,
    absl::string_view session_id, bool is_tpu);

DataTable CreateTensorPatternDataTable(
    const tensorflow::profiler::PerModelInferenceStats& per_model_stats,
    const tensorflow::profiler::TensorPatternDatabase& tensor_pattern_db);

std::string InferenceStatsToDataTableJson(
    const InferenceStats& inference_stats);

}  // namespace tensorflow::profiler

#endif  // XPROF_CONVERT_MULTI_XSPACE_TO_INFERENCE_STATS_H_
