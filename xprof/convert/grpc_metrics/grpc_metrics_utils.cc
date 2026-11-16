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

#include "xprof/convert/grpc_metrics/grpc_metrics_utils.h"

#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "tsl/platform/protobuf.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/grpc_metrics/database_populator.h"
#include "xprof/convert/grpc_metrics/grpc_metrics_types.h"
#include "xprof/convert/grpc_metrics/xspace_extractor.h"
#include "plugin/xprof/protobuf/grpc_metrics.pb.h"
#include "plugin/xprof/protobuf/topology.pb.h"

namespace xprof::grpc_metrics {

GrpcMetricsParseResults ProcessXSpaceToGrpcMetrics(
    const tensorflow::profiler::XSpace& xspace, absl::string_view host_name,
    const tensorflow::profiler::Topology* topology, bool is_megacore,
    xprof::GrpcMetricsDatabase* db) {
  XSpaceExtractor extractor;
  RawMetricsData data =
      extractor.Extract(xspace, host_name, topology, is_megacore);

  DatabasePopulator populator;
  return populator.Populate(data, db);
}

void FinalizeHostGrpcMetrics(GrpcMetricsParseResults results,
                             xprof::GrpcMetricsDatabase* host_db) {
  host_db->set_total_network_stall_ms(results.total_network_stall_ms);
  host_db->set_total_compute_overlap_ms(results.total_compute_overlap_ms);

  // Server latencies of flows that have a straggler on this host.
  const absl::flat_hash_map<std::string, double> host_server_latencies =
      std::move(results.server_latencies);
  absl::flat_hash_set<std::string> matched_flow_ids;
  for (xprof::GrpcStraggler& straggler : *host_db->mutable_top_stragglers()) {
    const auto latency_it = host_server_latencies.find(straggler.chunk_id());
    if (latency_it != host_server_latencies.end()) {
      straggler.set_grpc_server_latency_ms(latency_it->second);
      matched_flow_ids.insert(straggler.chunk_id());
    }
  }

  // The other server latencies are matched against the stragglers of the
  // other hosts by FinalizeMergedGrpcMetrics.
  auto* unmatched_server_latencies =
      host_db->mutable_unmatched_server_latencies();
  for (const auto& [flow_id, latency_ms] : host_server_latencies) {
    if (!matched_flow_ids.contains(flow_id)) {
      (*unmatched_server_latencies)[flow_id] = latency_ms;
    }
  }

  if (host_db->top_stragglers_size() > kMaxLocalStragglers) {
    host_db->mutable_top_stragglers()->DeleteSubrange(
        kMaxLocalStragglers,
        host_db->top_stragglers_size() - kMaxLocalStragglers);
  }
}

void MergeHostGrpcMetrics(xprof::GrpcMetricsDatabase host_db,
                          xprof::GrpcMetricsDatabase* merged_db) {
  merged_db->set_total_network_stall_ms(merged_db->total_network_stall_ms() +
                                        host_db.total_network_stall_ms());
  merged_db->set_total_compute_overlap_ms(
      merged_db->total_compute_overlap_ms() +
      host_db.total_compute_overlap_ms());
  // Cleared so that MergeFrom does not overwrite the sums.
  host_db.clear_total_network_stall_ms();
  host_db.clear_total_compute_overlap_ms();
  merged_db->MergeFrom(host_db);
}

void FinalizeMergedGrpcMetrics(xprof::GrpcMetricsDatabase* merged_db) {
  // Cross-host pass: match stragglers against the server latencies that no
  // straggler of their own host used.
  const auto& unmatched_server_latencies =
      merged_db->unmatched_server_latencies();
  for (xprof::GrpcStraggler& straggler : *merged_db->mutable_top_stragglers()) {
    if (straggler.grpc_server_latency_ms() > 0) continue;
    const auto latency_it =
        unmatched_server_latencies.find(straggler.chunk_id());
    if (latency_it != unmatched_server_latencies.end()) {
      straggler.set_grpc_server_latency_ms(latency_it->second);
    }
  }
  merged_db->mutable_unmatched_server_latencies()->clear();

  if (merged_db->top_stragglers_size() > kMaxGlobalStragglers) {
    merged_db->mutable_top_stragglers()->DeleteSubrange(
        kMaxGlobalStragglers,
        merged_db->top_stragglers_size() - kMaxGlobalStragglers);
  }
}

absl::Status ConvertGrpcMetricsToJson(const xprof::GrpcMetricsDatabase& db,
                                      std::string* json) {
  // All other options keep their defaults, so fields with default values are
  // left out.
  tsl::protobuf::util::JsonPrintOptions options;
  options.preserve_proto_field_names = true;
  return tsl::protobuf::util::MessageToJsonString(db, json, options);
}

}  // namespace xprof::grpc_metrics
