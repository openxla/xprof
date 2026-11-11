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
#ifndef THIRD_PARTY_XPROF_CONVERT_INFERENCE_STATS_PROGRAM_METADATA_H_
#define THIRD_PARTY_XPROF_CONVERT_INFERENCE_STATS_PROGRAM_METADATA_H_

#include "xprof/convert/op_profile_builder.h"
#include "plugin/xprof/protobuf/inference_stats.pb.h"

namespace tensorflow::profiler {

// Builds a map from HLO program_id to its ProgramServingMetadata (model_id,
// batch_size, and num_batches) from InferenceStats.
//
// Batches and unbatched requests (RequestDetail entries not linked to any
// BatchDetail) with a non-empty model_id and batch_size >= 0 are counted;
// batches with batch_size <= 0 are logged in a summary warning. Only
// batch_size > 0 is treated as a valid batch size: when a program has any
// entry with batch_size > 0, batch_size == 0 entries are ignored so the
// program falls back to model-only (batch_size = 0) only when no positive
// batch size was recorded. If a program has multiple candidate (model_id,
// batch_size) pairs, keeps the pair with the highest num_batches (tie-breaking
// by model_id, then batch_size) and logs a warning.
ProgramServingMetadataMap BuildProgramServingMetadata(
    const InferenceStats& inference_stats);

}  // namespace tensorflow::profiler

#endif  // THIRD_PARTY_XPROF_CONVERT_INFERENCE_STATS_PROGRAM_METADATA_H_
