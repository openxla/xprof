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

#include "xprof/convert/grpc_metrics/database_populator.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xprof/convert/grpc_metrics/grpc_metrics_types.h"
#include "xprof/convert/grpc_metrics/xspace_extractor.h"
#include "plugin/xprof/protobuf/dcn_slack_analysis.pb.h"
#include "plugin/xprof/protobuf/grpc_metrics.pb.h"

namespace xprof::grpc_metrics {

namespace {

// Fathom timestamps are epoch ns; scaling them to ps overflows int64_t.
int64_t GetTimestampNs(const TcpMetrics& m) {
  return m.fathom_timestamp.has_value() ? *m.fathom_timestamp
                                        : m.timestamp_ps / 1000;
}

// Why the TCP latency of a flow is missing (first two) or incomplete (rest).
enum class TcpLatencyRejection {
  kNone,
  kMissingEvents,
  kNoAckAfterFirstSendmsg,
  // The last ACK is not after the last SENDMSG.
  kAckBeforeSendmsg,
  // The last ACK covers fewer bytes than the last SENDMSG wrote.
  kFinalAckMissing,
  // The last SENDMSG ends before the end of the write.
  kWriteNotFinished,
};

// A TCP write and the RPCs in it. Chaotic good sends all frames queued on a
// data connection in one write, and each of them gets the write's events.
struct TcpWrite {
  absl::flat_hash_set<absl::string_view> flow_ids;
  // Sum of the payload sizes of those RPCs (a lower bound of the write size).
  int64_t payload_bytes = 0;
};

// (connection, Fathom time, byte offset) of a SENDMSG identifies its write.
using TcpWriteKey = std::tuple<absl::string_view, int64_t, int64_t>;
using TcpWriteIndex = absl::flat_hash_map<TcpWriteKey, TcpWrite>;

std::optional<TcpWriteKey> GetTcpWriteKey(const TcpMetrics& metric) {
  if (metric.type != "SENDMSG" || !metric.fathom_timestamp.has_value() ||
      !metric.write_byte_offset.has_value()) {
    return std::nullopt;
  }
  return TcpWriteKey(metric.endpoint_id, *metric.fathom_timestamp,
                     *metric.write_byte_offset);
}

// Groups SENDMSG events by write. Keys point into `endpoint_tcp_metrics`.
TcpWriteIndex BuildTcpWriteIndex(
    const absl::flat_hash_map<std::string, std::vector<TcpMetrics>>&
        endpoint_tcp_metrics,
    const absl::flat_hash_map<std::string, int64_t>& flow_buffer_sizes) {
  TcpWriteIndex index;
  for (const auto& [endpoint_id, metrics] : endpoint_tcp_metrics) {
    for (const TcpMetrics& metric : metrics) {
      const std::optional<TcpWriteKey> key = GetTcpWriteKey(metric);
      if (!key.has_value()) continue;
      TcpWrite& write = index[*key];
      for (const std::string& flow_id : metric.flow_ids) {
        if (!write.flow_ids.insert(flow_id).second) continue;
        const auto size_it = flow_buffer_sizes.find(flow_id);
        if (size_it != flow_buffer_sizes.end()) {
          write.payload_bytes += size_it->second;
        }
      }
    }
  }
  return index;
}

struct TcpLatencyResult {
  const TcpMetrics* first_sendmsg = nullptr;
  const TcpMetrics* last_sendmsg = nullptr;
  const TcpMetrics* last_acked = nullptr;
  std::optional<double> ack_to_sendmsg_ms;
  TcpLatencyRejection rejection = TcpLatencyRejection::kNone;
  // RPCs, including this one, in the write of the last SENDMSG.
  int rpcs_in_write = 1;

  bool IsIncomplete() const {
    return ack_to_sendmsg_ms.has_value() &&
           rejection != TcpLatencyRejection::kNone;
  }
};

// Later by time; equal times are broken by the larger byte offset.
bool IsLaterTcpEvent(const TcpMetrics& candidate, const TcpMetrics* current) {
  if (current == nullptr) return true;
  const int64_t candidate_ns = GetTimestampNs(candidate);
  const int64_t current_ns = GetTimestampNs(*current);
  if (candidate_ns != current_ns) return candidate_ns > current_ns;
  return candidate.write_byte_offset.value_or(-1) >
         current->write_byte_offset.value_or(-1);
}

// TCP latency = first SENDMSG -> last ACK of the flow on `endpoint_id`. A write
// can need several sendmsg() calls, and the ACK of its last byte can be lost;
// the latency is then reported as incomplete. `chunk_bytes` is
// the chunk's payload size (0 if unknown). Kernel and profiler times are never
// mixed: events without a Fathom time are ignored if the flow has any.
TcpLatencyResult ComputeTcpLatency(
    absl::Span<const TcpMetrics* const> matched_metrics,
    absl::string_view endpoint_id, int64_t chunk_bytes,
    const TcpWriteIndex& tcp_writes) {
  bool has_fathom_timestamps = false;
  for (const TcpMetrics* metric : matched_metrics) {
    if (metric->endpoint_id == endpoint_id &&
        metric->fathom_timestamp.has_value()) {
      has_fathom_timestamps = true;
      break;
    }
  }

  TcpLatencyResult result;
  for (const TcpMetrics* metric : matched_metrics) {
    if (metric->endpoint_id != endpoint_id ||
        (has_fathom_timestamps && !metric->fathom_timestamp.has_value())) {
      continue;
    }
    if (metric->type == "SENDMSG") {
      if (result.first_sendmsg == nullptr ||
          GetTimestampNs(*metric) < GetTimestampNs(*result.first_sendmsg)) {
        result.first_sendmsg = metric;
      }
      if (IsLaterTcpEvent(*metric, result.last_sendmsg)) {
        result.last_sendmsg = metric;
      }
    } else if (metric->type == "ACKED") {
      if (IsLaterTcpEvent(*metric, result.last_acked)) {
        result.last_acked = metric;
      }
    }
  }

  if (result.first_sendmsg == nullptr || result.last_acked == nullptr) {
    result.rejection = TcpLatencyRejection::kMissingEvents;
    return result;
  }

  // A shared write ends after all of its RPCs, not at the end of this chunk.
  int64_t write_bytes = chunk_bytes;
  if (const std::optional<TcpWriteKey> key =
          GetTcpWriteKey(*result.last_sendmsg);
      key.has_value()) {
    const auto write_it = tcp_writes.find(*key);
    if (write_it != tcp_writes.end()) {
      result.rpcs_in_write = static_cast<int>(write_it->second.flow_ids.size());
      write_bytes = std::max(write_bytes, write_it->second.payload_bytes);
    }
  }

  const int64_t first_sendmsg_ns = GetTimestampNs(*result.first_sendmsg);
  const int64_t last_sendmsg_ns = GetTimestampNs(*result.last_sendmsg);
  const int64_t last_acked_ns = GetTimestampNs(*result.last_acked);
  if (last_acked_ns <= first_sendmsg_ns) {
    result.rejection = TcpLatencyRejection::kNoAckAfterFirstSendmsg;
    return result;
  }
  result.ack_to_sendmsg_ms =
      static_cast<double>(last_acked_ns - first_sendmsg_ns) / 1e6;

  if (last_acked_ns <= last_sendmsg_ns) {
    result.rejection = TcpLatencyRejection::kAckBeforeSendmsg;
    return result;
  }
  const std::optional<int64_t> last_sendmsg_offset =
      result.last_sendmsg->write_byte_offset;
  if (result.last_acked->write_byte_offset.has_value() &&
      last_sendmsg_offset.has_value() &&
      *result.last_acked->write_byte_offset < *last_sendmsg_offset) {
    result.rejection = TcpLatencyRejection::kFinalAckMissing;
    return result;
  }
  if (write_bytes > 0 && last_sendmsg_offset.has_value() &&
      *last_sendmsg_offset + 1 < write_bytes) {
    result.rejection = TcpLatencyRejection::kWriteNotFinished;
  }
  return result;
}

}  // namespace

GrpcMetricsParseResults DatabasePopulator::Populate(
    const RawMetricsData& data, xprof::GrpcMetricsDatabase* db) {
  for (const auto& json : data.trace_events_json) {
    db->add_trace_events_json(json);
  }

  LOG_EVERY_N(INFO, 10000) << "PopulateDatabase: host=" << data.host_name
                           << " flows_size=" << flows_.size()
                           << " active_sockets_size="
                           << data.flow_to_endpoint.size();
  for (const auto& [flow, ep] : data.flow_to_endpoint) {
    LOG_EVERY_N(INFO, 10000) << "  Flow " << flow << " on socket: " << ep;
  }

  int total_flows = 0;
  int added_to_db = 0;

  // 1. Populate flows_ sequentially from flow_start_times and flow_end_times.
  absl::flat_hash_set<absl::string_view> known_flows;
  for (const auto& [flow_id, start_info] : data.flow_start_times) {
    known_flows.insert(flow_id);
    total_flows++;
    int64_t start_ps = start_info.first;
    const std::string& host_name = start_info.second;
    int64_t duration_ps = -1;
    auto end_it = data.flow_end_times.find(flow_id);
    if (end_it != data.flow_end_times.end()) {
      duration_ps = end_it->second - start_ps;
    } else {
      duration_ps = data.max_timestamp_ps - start_ps;
    }
    if (duration_ps < 0) duration_ps = 0;
    flows_.push_back({flow_id, start_ps, duration_ps, host_name, ""});

    // We will append to db later, but we need to keep track of points.
    // Actually we can just mutate a copy of points_by_series.
  }

  // We need to copy points_by_series because we need to sort and modify it.
  auto points_by_series = data.points_by_series;

  for (const auto& [flow_id, start_info] : data.flow_start_times) {
    int64_t start_ps = start_info.first;
    const std::string& host_name = start_info.second;
    int64_t duration_ps = -1;
    auto end_it = data.flow_end_times.find(flow_id);
    if (end_it != data.flow_end_times.end()) {
      duration_ps = end_it->second - start_ps;
    } else {
      duration_ps = data.max_timestamp_ps - start_ps;
    }
    if (duration_ps < 0) duration_ps = 0;
    double rel_time_sec =
        static_cast<double>(start_ps - data.min_timestamp_ps) / 1e12;
    points_by_series["chunk_latency"].push_back(
        {rel_time_sec, static_cast<double>(duration_ps) / 1e9, host_name, ""});
  }

  for (const auto& [flow_id, end_ps] : data.flow_end_times) {
    if (known_flows.insert(flow_id).second) {
      total_flows++;
      int64_t duration_ps = end_ps - data.min_timestamp_ps;
      if (duration_ps < 0) duration_ps = 0;
      flows_.push_back(
          {flow_id, data.min_timestamp_ps, duration_ps, data.host_name, ""});

      points_by_series["chunk_latency"].push_back(
          {0.0, static_cast<double>(duration_ps) / 1e9, data.host_name, ""});
    }
  }

  for (const auto& [flow_id, stats] : data.flow_grpc_stats) {
    if (stats.client_latency_ms > 0 && known_flows.insert(flow_id).second) {
      total_flows++;
      int64_t start_ps = stats.client_start_ps > 0 ? stats.client_start_ps
                                                   : data.min_timestamp_ps;
      int64_t duration_ps = stats.client_latency_ms * 1e9;
      flows_.push_back({flow_id, start_ps, duration_ps, data.host_name, ""});
      double rel_time_sec =
          static_cast<double>(start_ps - data.min_timestamp_ps) / 1e12;
      points_by_series["chunk_latency"].push_back(
          {rel_time_sec, static_cast<double>(duration_ps) / 1e9, data.host_name,
           ""});
    }
  }

  // Sort endpoint_tcp_metrics by timestamp. We need a mutable copy of the map.
  endpoint_tcp_metrics_ = data.endpoint_tcp_metrics;
  for (auto& [endpoint_id, metrics_vector] : endpoint_tcp_metrics_) {
    std::sort(metrics_vector.begin(), metrics_vector.end(),
              [](const TcpMetrics& a, const TcpMetrics& b) {
                return GetTimestampNs(a) < GetTimestampNs(b);
              });
  }

  // Build flow_id to TcpMetrics index for efficient matching.
  flow_to_tcp_metrics_.clear();
  for (const auto& [endpoint_id, metrics_list] : endpoint_tcp_metrics_) {
    for (const auto& metric : metrics_list) {
      for (const auto& fid : metric.flow_ids) {
        flow_to_tcp_metrics_[fid].push_back(&metric);
      }
    }
  }
  // Points into endpoint_tcp_metrics_, which is not modified below.
  const TcpWriteIndex tcp_writes =
      BuildTcpWriteIndex(endpoint_tcp_metrics_, data.flow_buffer_sizes);

  for (const auto& [flow_id, metrics_list] : flow_to_tcp_metrics_) {
    if (known_flows.insert(flow_id).second) {
      total_flows++;
      // Profiler times (same base as min_timestamp_ps); the list is sorted by
      // Fathom time.
      int64_t start_ps = data.min_timestamp_ps;
      int64_t duration_ps = 0;
      if (!metrics_list.empty()) {
        start_ps = metrics_list.front()->timestamp_ps;
        int64_t end_ps = start_ps;
        for (const TcpMetrics* metric : metrics_list) {
          start_ps = std::min(start_ps, metric->timestamp_ps);
          end_ps = std::max(end_ps, metric->timestamp_ps);
        }
        duration_ps = end_ps - start_ps;
      }
      flows_.push_back({flow_id, start_ps, duration_ps, data.host_name, ""});
      double rel_time_sec =
          static_cast<double>(start_ps - data.min_timestamp_ps) / 1e12;
      points_by_series["chunk_latency"].push_back(
          {rel_time_sec, static_cast<double>(duration_ps) / 1e9, data.host_name,
           ""});
    }
  }

  std::sort(flows_.begin(), flows_.end(),
            [](const FlowInfo& a, const FlowInfo& b) {
              return a.start_ps < b.start_ps;
            });

  // 2. Populate endpoint_id mapping for all flows directly.
  for (auto& flow : flows_) {
    auto ep_it = data.flow_to_endpoint.find(flow.flow_id);
    if (ep_it != data.flow_to_endpoint.end()) {
      flow.endpoint_id = ep_it->second;
    }
  }

  if (data.endpoint_tcp_metrics.empty()) {
    LOG(WARNING) << "NO FATHOM TCP METRICS DETECTED FOR HOST: "
                 << data.host_name
                 << " (Fathom telemetry was likely disabled or failed)";
  }

  absl::flat_hash_map<std::string, const tensorflow::profiler::DcnSlackSummary*>
      slack_map;
  for (const auto& summary : data.dcn_slack_summaries) {
    slack_map[summary.rendezvous()] = &summary;
  }

  int total_flows_with_endpoint = 0;
  int total_matched_tcp_index = 0;
  int total_unmatched_tcp_index = 0;
  int total_first_sendmsg_matched = 0;
  int total_last_acked_matched = 0;
  int total_failed_last_acked = 0;
  int total_incomplete_ack_before_sendmsg = 0;
  int total_incomplete_final_ack_missing = 0;
  int total_incomplete_write_not_finished = 0;
  int total_shared_tcp_writes = 0;

  for (auto flow_it = flows_.begin(); flow_it != flows_.end(); ++flow_it) {
    const auto& flow = *flow_it;
    auto* straggler = db->add_top_stragglers();
    straggler->set_chunk_id(flow.flow_id);
    straggler->set_tpu_vm(flow.host_name);
    if (!flow.endpoint_id.empty()) {
      straggler->set_peer_address(flow.endpoint_id);
    }
    straggler->set_latency_ms(static_cast<double>(flow.duration_ps) / 1e9);
    if (data.flow_grpc_stats.contains(flow.flow_id)) {
      const auto& stats = data.flow_grpc_stats.at(flow.flow_id);
      straggler->set_grpc_client_latency_ms(stats.client_latency_ms);
      straggler->set_grpc_server_latency_ms(stats.server_latency_ms);
      straggler->set_grpc_status(stats.status);
    }

    if (!flow.endpoint_id.empty()) {
      straggler->add_endpoint_ids(flow.endpoint_id);
    }

    if (data.flow_to_dcn_details.contains(flow.flow_id)) {
      const auto& dcn_info = data.flow_to_dcn_details.at(flow.flow_id);
      straggler->set_dcn_source_device_id(dcn_info.dcn_source_device_id);
      straggler->set_dcn_destination_device_id(
          dcn_info.dcn_destination_device_id);
      straggler->set_dcn_launch_id(dcn_info.dcn_launch_id);
      straggler->set_dcn_loop_index(dcn_info.dcn_loop_index);
      straggler->set_dcn_chunk_index(dcn_info.dcn_chunk_index);
      straggler->set_dcn_shard_range(dcn_info.dcn_shard_range);
      straggler->set_dcn_collective_label(dcn_info.dcn_collective_label);
      straggler->set_raw_transfer_key(dcn_info.raw_transfer_key);
    }

    // Check if flow has server call stats for rendezvous.
    auto r_it = data.flow_to_rendezvous.find(flow.flow_id);
    if (r_it != data.flow_to_rendezvous.end()) {
      const std::string& r_name = r_it->second;
      straggler->set_hlo_collective_name(r_name);
      const tensorflow::profiler::DcnSlackSummary* matched_summary = nullptr;
      auto slack_it = slack_map.find(r_name);
      if (slack_it != slack_map.end()) {
        matched_summary = slack_it->second;
      } else {
        // Fallback loop to find the longest matching prefix key.
        size_t longest_key_len = 0;
        for (const auto& [key, summary_ptr] : slack_map) {
          if (absl::StartsWith(r_name, key + "_") &&
              key.length() > longest_key_len) {
            longest_key_len = key.length();
            matched_summary = summary_ptr;
          }
        }
      }
      if (matched_summary != nullptr) {
        const auto& summary = *matched_summary;
        straggler->set_slack_time_ms(summary.slack_us() / 1000.0);
        straggler->set_tpu_stall_time_ms(summary.stall_duration_us() / 1000.0);
        straggler->set_communication_overlap_ms(summary.slack_us() / 1000.0);
        if (!summary.recv_op_name().empty()) {
          straggler->set_hlo_collective_name(summary.recv_op_name());
        }
        if (!summary.rendezvous().empty()) {
          straggler->set_dcn_collective_label(summary.rendezvous());
        }
      }
    }

    if (!flow.endpoint_id.empty()) {
      total_flows_with_endpoint++;
      // Execute exact matching using the flow_id index.
      auto tcp_it = flow_to_tcp_metrics_.find(flow.flow_id);
      if (tcp_it != flow_to_tcp_metrics_.end()) {
        total_matched_tcp_index++;
        // 0 when unknown.
        const auto size_it = data.flow_buffer_sizes.find(flow.flow_id);
        const int64_t chunk_bytes =
            size_it == data.flow_buffer_sizes.end() ? 0 : size_it->second;
        const TcpLatencyResult tcp_latency = ComputeTcpLatency(
            tcp_it->second, flow.endpoint_id, chunk_bytes, tcp_writes);
        if (tcp_latency.rpcs_in_write > 1) {
          total_shared_tcp_writes++;
          straggler->set_tcp_write_shared(true);
        }

        if (tcp_latency.first_sendmsg != nullptr) {
          total_first_sendmsg_matched++;
        }
        if (tcp_latency.last_acked != nullptr) {
          total_last_acked_matched++;
        } else {
          total_failed_last_acked++;
        }

        if (tcp_latency.ack_to_sendmsg_ms.has_value()) {
          straggler->set_ack_to_sendmsg_ms(*tcp_latency.ack_to_sendmsg_ms);
          if (tcp_latency.IsIncomplete()) {
            straggler->set_tcp_latency_incomplete(true);
          }
        }
        switch (tcp_latency.rejection) {
          case TcpLatencyRejection::kNone:
            added_to_db++;
            break;
          case TcpLatencyRejection::kMissingEvents:
          case TcpLatencyRejection::kNoAckAfterFirstSendmsg:
            break;
          case TcpLatencyRejection::kAckBeforeSendmsg:
            total_incomplete_ack_before_sendmsg++;
            break;
          case TcpLatencyRejection::kFinalAckMissing:
            total_incomplete_final_ack_missing++;
            break;
          case TcpLatencyRejection::kWriteNotFinished:
            total_incomplete_write_not_finished++;
            break;
        }
      } else {
        total_unmatched_tcp_index++;
      }
    }
  }

  LOG(INFO) << "PopulateDatabase: host=" << data.host_name
            << " merged flows=" << added_to_db << "/" << flows_.size()
            << " (flows_with_endpoint=" << total_flows_with_endpoint
            << ", matched_tcp_index=" << total_matched_tcp_index
            << ", unmatched_tcp_index=" << total_unmatched_tcp_index
            << ", first_sendmsg_matched=" << total_first_sendmsg_matched
            << ", last_acked_matched=" << total_last_acked_matched
            << ", failed_last_acked=" << total_failed_last_acked
            << ", incomplete_ack_before_sendmsg="
            << total_incomplete_ack_before_sendmsg
            << ", incomplete_final_ack_missing="
            << total_incomplete_final_ack_missing
            << ", incomplete_write_not_finished="
            << total_incomplete_write_not_finished
            << ", in_shared_tcp_write=" << total_shared_tcp_writes << ")";

  std::sort(db->mutable_top_stragglers()->pointer_begin(),
            db->mutable_top_stragglers()->pointer_end(),
            [](const xprof::GrpcStraggler* a, const xprof::GrpcStraggler* b) {
              return a->latency_ms() > b->latency_ms();
            });

  if (db->top_stragglers_size() > 100000) {
    db->mutable_top_stragglers()->DeleteSubrange(
        100000, db->top_stragglers_size() - 100000);
  }

  auto append_pt = [&](auto* series, const PendingPoint& pt) {
    auto* new_pt = series->Add();
    new_pt->set_timestamp_rel_sec(pt.timestamp_rel_sec);
    new_pt->set_value(pt.value);
    new_pt->set_tpu_vm(pt.host_name);
    if (!pt.endpoint_id.empty()) {
      new_pt->set_endpoint_id(pt.endpoint_id);
    }
  };

  for (auto& pair : points_by_series) {
    const std::string& series_name = pair.first;
    std::vector<PendingPoint>& points = pair.second;
    std::sort(points.begin(), points.end(),
              [](const PendingPoint& a, const PendingPoint& b) {
                if (a.host_name != b.host_name)
                  return a.host_name < b.host_name;
                return a.timestamp_rel_sec < b.timestamp_rel_sec;
              });

    for (size_t i = 0; i < points.size();) {
      const std::string& current_host = points[i].host_name;
      size_t host_start = i;
      while (i < points.size() && points[i].host_name == current_host) {
        i++;
      }

      for (size_t j = host_start; j < i; j++) {
        const PendingPoint& pt = points[j];
        if (series_name == "tcp_min_rtt") {
          append_pt(db->mutable_tcp_min_rtt_series(), pt);
        } else if (series_name == "tcp_packet_retx") {
          append_pt(db->mutable_tcp_packet_retx_series(), pt);
        } else if (series_name == "tcp_data_sent") {
          append_pt(db->mutable_tcp_data_sent_series(), pt);
        } else if (series_name == "tcp_pacing_rate") {
          append_pt(db->mutable_tcp_pacing_rate_series(), pt);
        } else if (series_name == "tcp_delivery_rate") {
          append_pt(db->mutable_tcp_delivery_rate_series(), pt);
        } else if (series_name == "tcp_ssthresh") {
          append_pt(db->mutable_tcp_ssthresh_series(), pt);
        } else if (series_name == "tcp_is_delivery_rate_app_limited") {
          append_pt(db->mutable_tcp_is_delivery_rate_app_limited_series(), pt);
        } else if (series_name == "tcp_byte_offset") {
          append_pt(db->mutable_tcp_byte_offset_series(), pt);
        } else if (series_name == "tcp_data_notsent") {
          append_pt(db->mutable_tcp_data_notsent_series(), pt);
        } else if (series_name == "tcp_packet_spurious_retx") {
          append_pt(db->mutable_tcp_packet_spurious_retx_series(), pt);
        } else if (series_name == "tcp_packet_delivered_ce") {
          append_pt(db->mutable_tcp_packet_delivered_ce_series(), pt);
        } else if (series_name == "tcp_reordering") {
          append_pt(db->mutable_tcp_reordering_series(), pt);
        } else if (series_name == "tcp_recurring_retrans") {
          append_pt(db->mutable_tcp_recurring_retrans_series(), pt);
        } else if (series_name == "tcp_busy_usec") {
          append_pt(db->mutable_tcp_busy_usec_series(), pt);
        } else if (series_name == "tcp_rwnd_limited_usec") {
          append_pt(db->mutable_tcp_rwnd_limited_usec_series(), pt);
        } else if (series_name == "tcp_sndbuf_limited_usec") {
          append_pt(db->mutable_tcp_sndbuf_limited_usec_series(), pt);
        } else if (series_name == "tcp_congestion_window") {
          append_pt(db->mutable_tcp_congestion_window_series(), pt);
        } else if (series_name == "tcp_packet_sent") {
          append_pt(db->mutable_tcp_packet_sent_series(), pt);
        } else if (series_name == "tcp_packet_delivered") {
          append_pt(db->mutable_tcp_packet_delivered_series(), pt);
        } else if (series_name == "grpc_client_latency") {
          append_pt(db->mutable_grpc_client_latency_series(), pt);
        } else if (series_name == "grpc_server_latency") {
          append_pt(db->mutable_grpc_server_latency_series(), pt);
        } else if (series_name == "chunk_latency") {
          auto* new_pt = db->add_chunk_latency_series();
          new_pt->set_timestamp_rel_sec(pt.timestamp_rel_sec);
          new_pt->set_value(pt.value);
          new_pt->set_tpu_vm(pt.host_name);
        }
      }
    }
  }

  GrpcMetricsParseResults results;
  results.total_network_stall_ms = data.total_network_stall_ms;
  results.total_compute_overlap_ms = data.total_compute_overlap_ms;
  for (const auto& [flow, stats] : data.flow_grpc_stats) {
    if (stats.server_latency_ms > 0) {
      results.server_latencies[flow] = stats.server_latency_ms;
    }
  }
  return results;
}

}  // namespace xprof::grpc_metrics
