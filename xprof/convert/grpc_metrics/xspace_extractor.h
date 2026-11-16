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

#ifndef XPROF_CONVERT_GRPC_METRICS_XSPACE_EXTRACTOR_H_
#define XPROF_CONVERT_GRPC_METRICS_XSPACE_EXTRACTOR_H_

#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/profiler/utils/xplane_visitor.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/grpc_metrics/grpc_metrics_types.h"
#include "plugin/xprof/protobuf/dcn_slack_analysis.pb.h"
#include "plugin/xprof/protobuf/topology.pb.h"

namespace xprof::grpc_metrics {

struct DcnTransferDetails {
  std::string dcn_source_device_id;
  std::string dcn_destination_device_id;
  std::string dcn_launch_id;
  std::string dcn_loop_index;
  std::string dcn_chunk_index;
  std::string dcn_shard_range;
  std::string dcn_collective_label;
  std::string raw_transfer_key;
};

struct RawMetricsData {
  std::string host_name;
  absl::flat_hash_map<std::string, std::pair<int64_t, std::string>>
      flow_start_times;
  absl::flat_hash_map<std::string, int64_t> flow_end_times;
  absl::flat_hash_map<std::string, std::string> flow_to_endpoint;
  absl::flat_hash_map<std::string, std::vector<TcpMetrics>>
      endpoint_tcp_metrics;
  absl::flat_hash_map<std::string, GrpcCallStats> flow_grpc_stats;
  absl::flat_hash_map<std::string, std::vector<PendingPoint>> points_by_series;
  std::vector<tensorflow::profiler::DcnSlackSummary> dcn_slack_summaries;
  absl::flat_hash_map<std::string, std::string> flow_to_rendezvous;
  absl::flat_hash_map<std::string, DcnTransferDetails> flow_to_dcn_details;
  // MXLA payload size of each chunk (SendRpc `buffer_size`), in bytes.
  absl::flat_hash_map<std::string, int64_t> flow_buffer_sizes;

  absl::flat_hash_map<std::string, double> endpoint_constant_offsets;

  int64_t min_timestamp_ps = std::numeric_limits<int64_t>::max();
  int64_t max_timestamp_ps = std::numeric_limits<int64_t>::min();

  double total_network_stall_ms = 0.0;
  double total_compute_overlap_ms = 0.0;

  // Serialized JSON of filtered trace events.
  std::vector<std::string> trace_events_json;
};

class XSpaceExtractor {
 public:
  RawMetricsData Extract(const tensorflow::profiler::XSpace& xspace,
                         absl::string_view host_name,
                         const tensorflow::profiler::Topology* topology,
                         bool is_megacore);

 private:
  struct EventMetadataIds {
    int64_t send_rpc_id = -1;
    int64_t send_rpc_done_id = -1;
    int64_t endpoint_write_id = -1;
    int64_t client_call_id = -1;
    int64_t server_call_id = -1;
    int64_t system_config_id = -1;
  };

  void ParseDcnSlack(const tensorflow::profiler::XSpace& xspace,
                     const tensorflow::profiler::Topology* topology,
                     bool is_megacore);

  EventMetadataIds ResolveMetadataIds(
      const tsl::profiler::XPlaneVisitor& plane_visitor);

  void ResolveStatMetadataIds(const tensorflow::profiler::XPlane* plane,
                              int64_t& flow_stat_id, int64_t& key_stat_id,
                              int64_t& event_args_stat_id);

  void CalculateTimestampBounds(
      const tsl::profiler::XPlaneVisitor& plane_visitor,
      const EventMetadataIds& ids);

  void ProcessSendRpcEvent(const tsl::profiler::XEventVisitor& event,
                           int64_t flow_stat_id, int64_t key_stat_id,
                           absl::string_view host_name);

  void ProcessSendRpcDoneEvent(const tsl::profiler::XEventVisitor& event,
                               int64_t flow_stat_id);

  void ProcessGrpcFlowWriteEvent(const tsl::profiler::XEventVisitor& event,
                                 absl::string_view args_str,
                                 double rel_time_sec,
                                 absl::string_view host_name);

  void ProcessClientCallEvent(const tsl::profiler::XEventVisitor& event,
                              absl::string_view args_str, double rel_time_sec,
                              absl::string_view host_name);

  void ProcessServerCallEvent(const tsl::profiler::XEventVisitor& event,
                              absl::string_view args_str, double rel_time_sec,
                              absl::string_view host_name);

  RawMetricsData data_;
};

}  // namespace xprof::grpc_metrics

#endif  // XPROF_CONVERT_GRPC_METRICS_XSPACE_EXTRACTOR_H_
