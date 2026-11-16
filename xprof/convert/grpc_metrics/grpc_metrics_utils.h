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

#ifndef XPROF_CONVERT_GRPC_METRICS_GRPC_METRICS_UTILS_H_
#define XPROF_CONVERT_GRPC_METRICS_GRPC_METRICS_UTILS_H_

#include <string>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/grpc_metrics/grpc_metrics_types.h"
#include "plugin/xprof/protobuf/grpc_metrics.pb.h"
#include "plugin/xprof/protobuf/topology.pb.h"

namespace xprof::grpc_metrics {

// Maximum number of stragglers kept for one host.
inline constexpr int kMaxLocalStragglers = 100000;

// Maximum number of stragglers kept after the hosts are merged.
inline constexpr int kMaxGlobalStragglers = 1000000;

// Parses a single host's XSpace and populates the GrpcMetricsDatabase.
GrpcMetricsParseResults ProcessXSpaceToGrpcMetrics(
    const tensorflow::profiler::XSpace& xspace, absl::string_view host_name,
    const tensorflow::profiler::Topology* topology, bool is_megacore,
    xprof::GrpcMetricsDatabase* db);

// Finishes the database of one host after ProcessXSpaceToGrpcMetrics: stores
// the stall and overlap totals of `results`, gives each straggler the server
// latency that this host recorded for its flow, keeps the server latencies
// that match no straggler of this host in `unmatched_server_latencies` (for
// FinalizeMergedGrpcMetrics), and keeps at most kMaxLocalStragglers
// stragglers.
void FinalizeHostGrpcMetrics(GrpcMetricsParseResults results,
                             xprof::GrpcMetricsDatabase* host_db);

// Adds the database of one host, or of a group of hosts, to `merged_db`: sums
// the stall and overlap totals and appends everything else (stragglers, time
// series, unmatched server latencies and trace events). A server latency of a
// flow that is already in `merged_db` is replaced.
void MergeHostGrpcMetrics(xprof::GrpcMetricsDatabase host_db,
                          xprof::GrpcMetricsDatabase* merged_db);

// Finishes a database merged from all hosts: gives each straggler that has no
// server latency the latency that another host recorded for its flow, drops
// the unmatched server latencies, and keeps at most kMaxGlobalStragglers
// stragglers.
void FinalizeMergedGrpcMetrics(xprof::GrpcMetricsDatabase* merged_db);

// Writes the JSON of `db` that the DCN Analyzer page reads to `json`: fields
// use their proto names and fields with default values are left out. Returns
// the status of the protobuf JSON printer.
absl::Status ConvertGrpcMetricsToJson(const xprof::GrpcMetricsDatabase& db,
                                      std::string* json);

}  // namespace xprof::grpc_metrics

#endif  // XPROF_CONVERT_GRPC_METRICS_GRPC_METRICS_UTILS_H_
