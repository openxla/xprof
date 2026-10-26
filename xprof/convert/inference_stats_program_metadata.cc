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
#include "xprof/convert/inference_stats_program_metadata.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "xprof/convert/op_profile_builder.h"
#include "plugin/xprof/protobuf/inference_stats.pb.h"
#include "plugin/xprof/protobuf/op_profile.pb.h"

namespace tensorflow::profiler {
namespace {

struct BatchRecord {
  const BatchDetail* batch = nullptr;
  int32_t host_id = -1;
};

struct RequestRecord {
  const RequestDetail* request = nullptr;
  int32_t host_id = -1;
};

struct ModelBatchEntry {
  std::string model_id;
  int32_t batch_size = 0;
  int32_t num_batches = 0;
};

std::string ResolveModelId(const ModelIdDatabase& model_id_db,
                           int32_t model_id_index) {
  if (model_id_index < 0 || model_id_index >= model_id_db.ids_size()) {
    return "";
  }
  absl::string_view model_id = model_id_db.ids(model_id_index);
  if (model_id == "ALL") {
    return "";
  }
  return std::string(model_id);
}

}  // namespace

ProgramServingMetadataMap BuildProgramServingMetadata(
    const InferenceStats& inference_stats) {
  std::vector<BatchRecord> batches;
  std::vector<RequestRecord> requests;
  absl::flat_hash_set<int32_t> distinct_hosts;

  // Prefer per-model stats when present; otherwise fall back to per-host stats.
  // Never read both to avoid double counting.
  if (!inference_stats.inference_stats_per_model().empty()) {
    for (const auto& [model_index, per_model] :
         inference_stats.inference_stats_per_model()) {
      for (const BatchDetail& batch : per_model.batch_details()) {
        int32_t host_id = batch.has_host_id() ? batch.host_id() : -1;
        if (host_id >= 0) {
          distinct_hosts.insert(host_id);
        }
        batches.push_back({&batch, host_id});
      }
      for (const RequestDetail& request : per_model.request_details()) {
        int32_t host_id = request.has_host_id() ? request.host_id() : -1;
        if (host_id >= 0) {
          distinct_hosts.insert(host_id);
        }
        requests.push_back({&request, host_id});
      }
    }
  } else {
    for (const auto& [host_key, per_host] :
         inference_stats.inference_stats_per_host()) {
      // A negative host id is invalid, so fall back to the map key.
      for (const BatchDetail& batch : per_host.batch_details()) {
        int32_t host_id = (batch.has_host_id() && batch.host_id() >= 0)
                              ? batch.host_id()
                              : host_key;
        if (host_id >= 0) {
          distinct_hosts.insert(host_id);
        }
        batches.push_back({&batch, host_id});
      }
      for (const RequestDetail& request : per_host.request_details()) {
        int32_t host_id = (request.has_host_id() && request.host_id() >= 0)
                              ? request.host_id()
                              : host_key;
        if (host_id >= 0) {
          distinct_hosts.insert(host_id);
        }
        requests.push_back({&request, host_id});
      }
    }
  }

  ProgramServingMetadataMap result;
  if (batches.empty()) {
    return result;
  }

  const ModelIdDatabase& model_id_db = inference_stats.model_id_db();
  // BatchDetail::model_id_index is directly trusted when all data comes from a
  // single source or when the session has at most one model. When multiple
  // sources and multiple models exist (or when batch.model_id_index() is
  // unset), resolve the batch's model through its related requests, so
  // sessions cached before the combiner remapped batch model indices still
  // report the right model.
  const bool trust_batch_model_index =
      (distinct_hosts.size() <= 1) || (model_id_db.ids_size() <= 1);

  absl::flat_hash_map<std::pair<int32_t, int64_t>, int32_t>
      request_model_index_by_host_and_id;
  request_model_index_by_host_and_id.reserve(requests.size());
  for (const RequestRecord& record : requests) {
    if (record.host_id >= 0 && record.request->request_id() != -1) {
      request_model_index_by_host_and_id[{record.host_id,
                                          record.request->request_id()}] =
          record.request->model_id_index();
    }
  }

  using ModelBatchKey = std::pair<std::string, int32_t>;
  absl::flat_hash_map<uint64_t, absl::flat_hash_map<ModelBatchKey, int32_t>>
      counts_by_program;
  int64_t skipped_non_positive_batch_size_batches = 0;
  absl::flat_hash_set<uint64_t> programs_with_non_positive_batch_size;

  for (const BatchRecord& record : batches) {
    const BatchDetail& batch = *record.batch;
    if (batch.program_ids().empty()) {
      continue;
    }

    absl::flat_hash_set<uint64_t> seen_program_ids;
    for (uint64_t program_id : batch.program_ids()) {
      if (program_id != 0) {
        seen_program_ids.insert(program_id);
      }
    }
    if (seen_program_ids.empty()) {
      continue;
    }

    const int32_t batch_size = batch.batch_size_after_padding();
    if (batch_size <= 0) {
      ++skipped_non_positive_batch_size_batches;
      programs_with_non_positive_batch_size.insert(seen_program_ids.begin(),
                                                   seen_program_ids.end());
      continue;
    }

    std::string model_id;
    if (trust_batch_model_index) {
      model_id = ResolveModelId(model_id_db, batch.model_id_index());
    }
    if (model_id.empty() && record.host_id >= 0) {
      for (int64_t request_id : batch.related_request_ids()) {
        auto it = request_model_index_by_host_and_id.find(
            {record.host_id, request_id});
        if (it != request_model_index_by_host_and_id.end()) {
          model_id = ResolveModelId(model_id_db, it->second);
          if (!model_id.empty()) {
            break;
          }
        }
      }
    }
    if (model_id.empty()) {
      continue;
    }

    for (uint64_t program_id : seen_program_ids) {
      counts_by_program[program_id][{model_id, batch_size}]++;
    }
  }

  if (skipped_non_positive_batch_size_batches > 0) {
    LOG(WARNING) << "BuildProgramServingMetadata: skipped "
                 << skipped_non_positive_batch_size_batches
                 << " batches with batch_size_after_padding <= 0 across "
                 << programs_with_non_positive_batch_size.size()
                 << " HLO programs.";
  }

  result.reserve(counts_by_program.size());
  for (const auto& [program_id, pair_counts] : counts_by_program) {
    std::vector<ModelBatchEntry> entries;
    entries.reserve(pair_counts.size());
    for (const auto& [key, count] : pair_counts) {
      entries.push_back({key.first, key.second, count});
    }
    absl::c_sort(entries,
                 [](const ModelBatchEntry& a, const ModelBatchEntry& b) {
                   if (a.num_batches != b.num_batches) {
                     return a.num_batches > b.num_batches;
                   }
                   if (a.model_id != b.model_id) {
                     return a.model_id < b.model_id;
                   }
                   return a.batch_size < b.batch_size;
                 });
    if (entries.empty()) {
      continue;
    }
    if (entries.size() > 1) {
      std::vector<std::string> formatted_pairs;
      formatted_pairs.reserve(entries.size());
      for (const ModelBatchEntry& entry : entries) {
        formatted_pairs.push_back(absl::StrCat(
            "(model_id=", entry.model_id, ", batch_size=", entry.batch_size,
            ", num_batches=", entry.num_batches, ")"));
      }
      LOG(WARNING) << "BuildProgramServingMetadata: HLO program " << program_id
                   << " has multiple (model_id, batch_size) pairs; keeping "
                   << formatted_pairs[0] << " out of ["
                   << absl::StrJoin(formatted_pairs, ", ") << "].";
    }

    op_profile::Node::ProgramServingMetadata metadata;
    metadata.set_model_id(entries[0].model_id);
    metadata.set_batch_size(entries[0].batch_size);
    metadata.set_num_batches(entries[0].num_batches);
    result[program_id] = std::move(metadata);
  }

  return result;
}

}  // namespace tensorflow::profiler
