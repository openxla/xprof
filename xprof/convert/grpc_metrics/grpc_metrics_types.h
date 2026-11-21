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

#ifndef XPROF_CONVERT_GRPC_METRICS_GRPC_METRICS_TYPES_H_
#define XPROF_CONVERT_GRPC_METRICS_GRPC_METRICS_TYPES_H_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"

namespace xprof::grpc_metrics {

struct FlowInfo {
  std::string flow_id;
  int64_t start_ps;
  int64_t duration_ps;
  std::string host_name;
  std::string endpoint_id;
};

struct PendingPoint {
  double timestamp_rel_sec;
  double value;
  std::string host_name;
  std::string endpoint_id;
};

struct TcpMetrics {
  std::optional<double> min_rtt;
  std::optional<double> packet_retx;
  std::optional<double> congestion_window;
  std::optional<double> delivery_rate;
  std::optional<double> data_sent;
  std::optional<double> data_retx;
  std::optional<double> pacing_rate;
  std::optional<double> packet_sent;
  std::optional<double> packet_delivered;
  std::optional<double> packet_delivered_ce;
  std::optional<double> busy_usec;
  std::optional<double> rwnd_limited_usec;
  std::optional<double> sndbuf_limited_usec;
  std::optional<double> is_app_limited;
  std::optional<double> data_not_sent;
  std::optional<double> srtt;
  std::optional<double> net_rtt_usec;
  std::optional<int64_t> fathom_timestamp;
  int64_t timestamp_ps;
  std::string type;
  double byte_offset;
  std::string endpoint_id;
  std::vector<std::string> flow_ids;
  // Fathom `byte_offset` in bytes from the start of the write: last byte sent
  // (SENDMSG) or acknowledged (ACKED).
  std::optional<int64_t> write_byte_offset;
};

struct GrpcCallStats {
  double client_latency_ms = 0;
  double server_latency_ms = 0;
  int64_t client_start_ps = -1;
  int64_t server_start_ps = -1;
  int status = -1;
};

struct GrpcMetricsParseResults {
  double total_network_stall_ms = 0.0;
  double total_compute_overlap_ms = 0.0;
  absl::flat_hash_map<std::string, double> server_latencies;
};

}  // namespace xprof::grpc_metrics

#endif  // XPROF_CONVERT_GRPC_METRICS_GRPC_METRICS_TYPES_H_
