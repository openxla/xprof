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
#include "xprof/convert/inference_stats_combiner.h"

#include <cstdint>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/lib/gtl/map_util.h"

namespace tensorflow::profiler {
namespace {
// Returns the index of `model_id` in `model_id_db`, inserting `model_id` into
// both `ids` and `id_to_index` if it is not already present. Returns -1 if
// `model_id` is empty (unknown model).
int32_t AssignIndexToModelId(const std::string& model_id,
                             ModelIdDatabase* model_id_db) {
  if (model_id.empty()) {
    return -1;
  }
  auto [iter, was_inserted] = model_id_db->mutable_id_to_index()->insert(
      {model_id, model_id_db->ids_size()});
  if (was_inserted) {
    model_id_db->add_ids(model_id);
  }
  return iter->second;
}

// Combines two ModelIdDatabases. Returns true if this combination requires
// updating model_id_index in the incoming host's details because any entry in
// src maps to a different index in dst (including empty model IDs mapping to
// -1).
bool CombineModelIdDatabases(const ModelIdDatabase& src, ModelIdDatabase* dst) {
  if (dst->ids_size() == 0) {
    // dst is empty. Simply copy src to dst. This avoids rebuilding
    // dst from src from scratch, which may change the name-to-index mapping.
    *dst = src;
    return false;
  }
  // Ensure every non-empty entry already in dst->ids() is indexed in
  // dst->id_to_index() before merging src.
  for (int32_t i = 0; i < dst->ids_size(); ++i) {
    if (!dst->ids(i).empty()) {
      dst->mutable_id_to_index()->insert({dst->ids(i), i});
    }
  }
  // TODO(tianrun): For now, assume a model is always served with the same
  // parameter on different hosts. In the future, we might consider the case
  // when the same model are served with different batching parameters on
  // different hosts.
  for (const auto& id_and_param : src.id_to_batching_params()) {
    dst->mutable_id_to_batching_params()->insert(id_and_param);
  }
  bool need_update = false;
  for (int32_t i = 0; i < src.ids_size(); ++i) {
    if (AssignIndexToModelId(src.ids(i), dst) != i) {
      need_update = true;
    }
  }
  return need_update;
}

// Combines two TensorPatternDatabase. Returns true if this combination requires
// updating the tensor_pattern_index. This will be the case if: (1) Src has a
// tensor pattern that doesn't exist in dst; or (2) Src has a tensor pattern
// that does exist in dst but has a different index.
bool CombineTensorPatternDatabase(
    const TensorPatternDatabase& src, TensorPatternDatabase* dst,
    absl::flat_hash_map<absl::string_view, int>* dst_pattern_to_index) {
  if (dst->tensor_pattern().empty()) {
    *dst = src;
    return false;
  }

  bool need_update = false;
  for (int i = 0; i < static_cast<int>(src.tensor_pattern_size()); i++) {
    auto [iter, inserted] = dst_pattern_to_index->insert(
        {src.tensor_pattern(i), dst_pattern_to_index->size()});
    if (inserted) {
      // Src has a tensor pattern that doesn't exist in dst.
      dst->add_tensor_pattern(src.tensor_pattern(i));
      need_update = true;
    } else if (iter->second != i) {
      // Src has a tensor pattern with different index than dst.
      need_update = true;
    }
  }
  return need_update;
}

void UpdateTensorPatternIndex(
    const TensorPatternDatabase& src,
    const absl::flat_hash_map<absl::string_view, int>& dst_pattern_to_index,
    TensorEventDetail* detail) {
  const int32_t pattern_index = detail->tensor_pattern_index();
  if (pattern_index < 0 || pattern_index >= src.tensor_pattern_size()) {
    LOG(WARNING) << "Invalid tensor_pattern_index " << pattern_index
                 << " for src.tensor_pattern_db() of size "
                 << src.tensor_pattern_size();
    return;
  }
  absl::string_view tensor_pattern = src.tensor_pattern(pattern_index);
  if (const int* new_index =
          tsl::gtl::FindOrNull(dst_pattern_to_index, tensor_pattern)) {
    detail->set_tensor_pattern_index(*new_index);
  } else {
    LOG(WARNING) << "Tensor pattern " << tensor_pattern
                 << " is not found in dst->tensor_pattern_db()";
  }
}

// Remaps model_id_index from src_db to dst_db to align model indices across
// hosts, adding the model to dst_db if it is missing. Returns -1 (no model)
// if src_index is out of range or refers to an empty model ID.
int32_t RemapModelIdIndex(const ModelIdDatabase& src_db,
                          ModelIdDatabase* dst_db, int32_t src_index) {
  if (src_index < 0 || src_index >= src_db.ids_size()) {
    LOG(WARNING) << "Invalid model_id_index " << src_index
                 << " for src.model_id_db() of size " << src_db.ids_size();
    return -1;
  }
  return AssignIndexToModelId(src_db.ids(src_index), dst_db);
}
}  // namespace

void CombineInferenceStatsResult(int src_host_id, const InferenceStats& src,
                                 InferenceStats* dst) {
  // There should be one key-value pair inside src.inference_stats_per_host(),
  // because the src comes from one XprofResponse (i.e., one host).
  DCHECK_LE(src.inference_stats_per_host_size(), 1);
  // Check for a duplicate host before merging the databases, so that a
  // rejected host does not add its models or tensor patterns to dst.
  if (dst->inference_stats_per_host().contains(src_host_id)) {
    LOG(INFO) << "Duplicate host_id: " << src_host_id;
    return;
  }
  bool need_update_model_id =
      CombineModelIdDatabases(src.model_id_db(), dst->mutable_model_id_db());
  absl::flat_hash_map<absl::string_view, int> dst_pattern_to_index;
  for (int i = 0;
       i < static_cast<int>(dst->tensor_pattern_db().tensor_pattern_size());
       i++) {
    dst_pattern_to_index[dst->tensor_pattern_db().tensor_pattern(i)] = i;
  }
  bool need_update_tensor_pattern = CombineTensorPatternDatabase(
      src.tensor_pattern_db(), dst->mutable_tensor_pattern_db(),
      &dst_pattern_to_index);
  for (const auto& [host_id, inf_stats] : src.inference_stats_per_host()) {
    auto [iter, was_inserted] = dst->mutable_inference_stats_per_host()->insert(
        {src_host_id, inf_stats});
    if (!was_inserted) {
      LOG(INFO) << "Duplicate host_id: " << iter->first;
      continue;
    }
    if (!need_update_model_id && !need_update_tensor_pattern) {
      continue;
    }
    PerHostInferenceStats& dst_inference_stats = iter->second;
    for (RequestDetail& request_detail :
         *dst_inference_stats.mutable_request_details()) {
      if (need_update_model_id && request_detail.model_id_index() != -1) {
        request_detail.set_model_id_index(
            RemapModelIdIndex(src.model_id_db(), dst->mutable_model_id_db(),
                              request_detail.model_id_index()));
      }
      if (need_update_tensor_pattern) {
        for (TensorEventDetail& tensor_event_details :
             *request_detail.mutable_tensor_event_details()) {
          UpdateTensorPatternIndex(src.tensor_pattern_db(),
                                   dst_pattern_to_index, &tensor_event_details);
        }
      }
    }
    for (BatchDetail& batch_detail :
         *dst_inference_stats.mutable_batch_details()) {
      if (need_update_model_id && batch_detail.model_id_index() != -1) {
        batch_detail.set_model_id_index(
            RemapModelIdIndex(src.model_id_db(), dst->mutable_model_id_db(),
                              batch_detail.model_id_index()));
      }
      if (need_update_tensor_pattern &&
          batch_detail.has_tensor_event_detail()) {
        UpdateTensorPatternIndex(src.tensor_pattern_db(), dst_pattern_to_index,
                                 batch_detail.mutable_tensor_event_detail());
      }
    }
  }
}
}  // namespace tensorflow::profiler
