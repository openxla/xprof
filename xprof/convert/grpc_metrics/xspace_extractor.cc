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

#include "xprof/convert/grpc_metrics/xspace_extractor.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "xla/tsl/profiler/utils/tf_xplane_visitor.h"
#include "xla/tsl/profiler/utils/xplane_utils.h"
#include "xla/tsl/profiler/utils/xplane_visitor.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/grpc_metrics/grpc_metrics_types.h"
#include "xprof/convert/grpc_metrics/json_utils.h"
#include "xprof/convert/grpc_metrics/transfer_key_parser.h"
#include "xprof/convert/xspace_to_dcn_slack_analysis.h"
#include "plugin/xprof/protobuf/dcn_slack_analysis.pb.h"
#include "plugin/xprof/protobuf/topology.pb.h"

namespace xprof::grpc_metrics {

namespace {

using ::tsl::profiler::FindPlaneWithName;
using ::tsl::profiler::XPlane;
using ::tsl::profiler::XPlaneVisitor;
using ::tsl::profiler::XSpace;

// Name of the XPlane that holds the MegaScale (DCN) trace events.
constexpr absl::string_view kMegascaleTracePlaneName =
    "/device:CUSTOM:Megascale Trace";

constexpr char kSendRpcEvent[] = "Megascale_gRPC:SendRpc";
constexpr char kSendRpcDoneEvent[] = "Megascale_gRPC:SendRpcDone";
constexpr char kGrpcFlowWriteMetricsTrace[] = "GrpcFlowWriteMetricsTrace";
constexpr char kClientCallMetrics[] = "ClientCallMetrics";
constexpr char kServerCallMetrics[] = "ServerCallMetrics";
constexpr char kSystemConfigEvent[] = "GrpcSystemConfigTrace";

constexpr char kFlowStat[] = "flow";
constexpr char kKeyStat[] = "key";
constexpr char kBufferSizeStat[] = "buffer_size";
constexpr char kEventArgsStat[] = "event_args";
constexpr char kTraceArgsStat[] = "trace_args";
constexpr char kArgsStat[] = "args";

inline double BytesToBits(double bytes) { return bytes * 8.0; }

}  // namespace

RawMetricsData XSpaceExtractor::Extract(
    const XSpace& xspace, absl::string_view host_name,
    const tensorflow::profiler::Topology* topology, bool is_megacore) {
  data_ = RawMetricsData();  // Reset.
  data_.host_name = std::string(host_name);

  for (const auto& p : xspace.planes()) {
    VLOG(1) << "Plane seen in xspace: " << p.name();
  }
  const XPlane* plane = FindPlaneWithName(xspace, "/host:CPU");
  if (!plane) return std::move(data_);

  ParseDcnSlack(xspace, topology, is_megacore);

  XPlaneVisitor plane_visitor = tsl::profiler::CreateTfXPlaneVisitor(plane);
  EventMetadataIds ids = ResolveMetadataIds(plane_visitor);

  int64_t flow_stat_id = -1;
  int64_t key_stat_id = -1;
  int64_t event_args_stat_id = -1;
  ResolveStatMetadataIds(plane, flow_stat_id, key_stat_id, event_args_stat_id);

  CalculateTimestampBounds(plane_visitor, ids);

  plane_visitor.ForEachLine([&](const tsl::profiler::XLineVisitor& line) {
    line.ForEachEvent([&](const tsl::profiler::XEventVisitor& event) {
      if (event.Id() == ids.send_rpc_id) {
        ProcessSendRpcEvent(event, flow_stat_id, key_stat_id, host_name);
      } else if (event.Id() == ids.send_rpc_done_id) {
        ProcessSendRpcDoneEvent(event, flow_stat_id);
      } else {
        double rel_time_sec =
            static_cast<double>(event.TimestampPs() - data_.min_timestamp_ps) /
            1e12;

        if (event.Id() == ids.endpoint_write_id ||
            event.Id() == ids.client_call_id ||
            event.Id() == ids.server_call_id ||
            event.Id() == ids.system_config_id) {
          std::string args_str;
          event.ForEachStat([&](const tsl::profiler::XStatVisitor& stat) {
            if (stat.Id() == event_args_stat_id)
              args_str = std::string(stat.StrOrRefValue());
          });
          if (args_str.empty()) return;

          if (event.Id() == ids.endpoint_write_id) {
            ProcessGrpcFlowWriteEvent(event, args_str, rel_time_sec, host_name);
            data_.trace_events_json.push_back(absl::StrCat(
                "{\"name\":\"", event.Name(), "\",\"args\":", args_str, "}"));
          } else if (event.Id() == ids.client_call_id) {
            ProcessClientCallEvent(event, args_str, rel_time_sec, host_name);
            data_.trace_events_json.push_back(absl::StrCat(
                "{\"name\":\"grpc_core::XorClientFilter::ClientCallMetrics\","
                "\"args\":",
                args_str, "}"));
          } else if (event.Id() == ids.server_call_id) {
            ProcessServerCallEvent(event, args_str, rel_time_sec, host_name);
            data_.trace_events_json.push_back(absl::StrCat(
                "{\"name\":\"grpc_core::XorServerFilter::ServerCallMetrics\","
                "\"args\":",
                args_str, "}"));
          } else if (event.Id() == ids.system_config_id) {
            std::string modified_args = args_str;
            // Inject the host name as 'tpu_vm' into the JSON object.
            if (!modified_args.empty() && modified_args.back() == '}') {
              modified_args.pop_back();
              absl::StrAppend(&modified_args, R"(,"tpu_vm":")", host_name,
                              R"("})");
            } else {
              modified_args = args_str;
            }
            data_.trace_events_json.push_back(absl::StrCat(
                "{\"name\":\"GrpcSystemConfigTrace\",\"args\":", modified_args,
                "}"));
          }
        }
      }
    });
  });

  return std::move(data_);
}

void XSpaceExtractor::ParseDcnSlack(
    const XSpace& xspace, const tensorflow::profiler::Topology* topology,
    bool is_megacore) {
  const XPlane* dcn_plane = FindPlaneWithName(xspace, kMegascaleTracePlaneName);
  if (dcn_plane) {
    if (topology == nullptr) {
      LOG(WARNING)
          << "Topology is null, but running DCN slack analysis anyway.";
    }
    tensorflow::profiler::DcnSlackAnalysis dcn_slack_analysis =
        tensorflow::profiler::ConvertXSpaceToDcnSlackAnalysis(
            xspace, dcn_plane, topology, is_megacore);
    LOG_EVERY_N(INFO, 10000) << "Dcn slack summary size: "
                             << dcn_slack_analysis.dcn_slack_summary_size();
    for (const auto& summary : dcn_slack_analysis.dcn_slack_summary()) {
      VLOG(1) << "Summary name: " << summary.rendezvous()
              << " stall: " << summary.stall_duration_us()
              << " slack: " << summary.slack_us();
      data_.total_network_stall_ms +=
          (summary.stall_duration_us() * summary.occurrences()) / 1000.0;
      data_.total_compute_overlap_ms +=
          (summary.slack_us() * summary.occurrences()) / 1000.0;

      data_.dcn_slack_summaries.push_back(summary);
    }
  }
}

XSpaceExtractor::EventMetadataIds XSpaceExtractor::ResolveMetadataIds(
    const XPlaneVisitor& plane_visitor) {
  EventMetadataIds ids;
  plane_visitor.ForEachEventMetadata(
      [&](const tsl::profiler::XEventMetadataVisitor& meta) {
        if (meta.Name() == kSendRpcEvent)
          ids.send_rpc_id = meta.Id();
        else if (meta.Name() == kSendRpcDoneEvent)
          ids.send_rpc_done_id = meta.Id();
        else if (absl::StrContains(meta.Name(), kGrpcFlowWriteMetricsTrace))
          ids.endpoint_write_id = meta.Id();
        else if (absl::StrContains(meta.Name(), kClientCallMetrics))
          ids.client_call_id = meta.Id();
        else if (absl::StrContains(meta.Name(), kServerCallMetrics))
          ids.server_call_id = meta.Id();
        else if (absl::StrContains(meta.Name(), kSystemConfigEvent))
          ids.system_config_id = meta.Id();
      });
  return ids;
}

void XSpaceExtractor::ResolveStatMetadataIds(const XPlane* plane,
                                             int64_t& flow_stat_id,
                                             int64_t& key_stat_id,
                                             int64_t& event_args_stat_id) {
  for (const auto& [stat_id, meta] : plane->stat_metadata()) {
    if (meta.name() == kFlowStat)
      flow_stat_id = stat_id;
    else if (meta.name() == kKeyStat)
      key_stat_id = stat_id;
    else if (meta.name() == kEventArgsStat || meta.name() == kTraceArgsStat ||
             meta.name() == kArgsStat)
      event_args_stat_id = stat_id;
  }
}

void XSpaceExtractor::CalculateTimestampBounds(
    const XPlaneVisitor& plane_visitor, const EventMetadataIds& ids) {
  plane_visitor.ForEachLine([&](const tsl::profiler::XLineVisitor& line) {
    line.ForEachEvent([&](const tsl::profiler::XEventVisitor& event) {
      if (event.Id() == ids.send_rpc_id || event.Id() == ids.send_rpc_done_id ||
          event.Id() == ids.endpoint_write_id ||
          event.Id() == ids.client_call_id ||
          event.Id() == ids.server_call_id ||
          event.Id() == ids.system_config_id) {
        data_.min_timestamp_ps =
            std::min(data_.min_timestamp_ps, event.TimestampPs());
        data_.max_timestamp_ps =
            std::max(data_.max_timestamp_ps, event.TimestampPs());
      }
    });
  });
}

void XSpaceExtractor::ProcessSendRpcEvent(
    const tsl::profiler::XEventVisitor& event, int64_t flow_stat_id,
    int64_t key_stat_id, absl::string_view host_name) {
  event.ForEachStat([&](const tsl::profiler::XStatVisitor& stat) {
    if (stat.Id() == flow_stat_id) {
      data_.flow_start_times[std::to_string(stat.IntOrUintValue())] = {
          event.TimestampPs(), std::string(host_name)};
    } else if (stat.Id() == key_stat_id) {
      auto key_str = stat.StrOrRefValue();
      std::string rendezvous_name = std::string(key_str);
      const std::optional<TransferKeyFields> transfer_key =
          ParseTransferKey(key_str);
      DcnTransferDetails dcn_details;
      dcn_details.raw_transfer_key = std::string(key_str);

      if (transfer_key.has_value()) {
        const TransferKeyFields& tk = *transfer_key;
        rendezvous_name = tk.label;
        dcn_details.dcn_source_device_id =
            std::to_string(tk.source_megascale_id);
        dcn_details.dcn_destination_device_id =
            std::to_string(tk.destination_megascale_id);
        if (tk.launch_id.has_value()) {
          dcn_details.dcn_launch_id = std::to_string(tk.launch_id.value());
        }
        if (tk.loop_index.has_value()) {
          dcn_details.dcn_loop_index = std::to_string(tk.loop_index.value());
        }
        if (tk.chunk.has_value()) {
          dcn_details.dcn_chunk_index = std::to_string(tk.chunk.value());
        }
        dcn_details.dcn_collective_label = tk.label;
      }
      event.ForEachStat([&](const tsl::profiler::XStatVisitor& inner_stat) {
        if (inner_stat.Id() == flow_stat_id) {
          std::string flow_id = std::to_string(inner_stat.IntOrUintValue());
          data_.flow_to_rendezvous[flow_id] = rendezvous_name;
          data_.flow_to_dcn_details[flow_id] = dcn_details;
        }
      });
    }
  });

  // Chunk size, to check that the TCP events cover the whole write.
  std::string flow_id;
  int64_t buffer_size = 0;
  event.ForEachStat([&](const tsl::profiler::XStatVisitor& stat) {
    if (stat.Id() == flow_stat_id) {
      flow_id = std::to_string(stat.IntOrUintValue());
    } else if (stat.Name() == kBufferSizeStat) {
      buffer_size = stat.IntOrUintValue();
    }
  });
  if (!flow_id.empty() && buffer_size > 0) {
    data_.flow_buffer_sizes[flow_id] = buffer_size;
  }
}

void XSpaceExtractor::ProcessSendRpcDoneEvent(
    const tsl::profiler::XEventVisitor& event, int64_t flow_stat_id) {
  event.ForEachStat([&](const tsl::profiler::XStatVisitor& stat) {
    if (stat.Id() == flow_stat_id)
      data_.flow_end_times[std::to_string(stat.IntOrUintValue())] =
          event.TimestampPs();
  });
}

void XSpaceExtractor::ProcessGrpcFlowWriteEvent(
    const tsl::profiler::XEventVisitor& event, absl::string_view args_str,
    double rel_time_sec, absl::string_view host_name) {
  std::string endpoint_id = GetIdFromJson(args_str, "endpoint_id");
  if (endpoint_id.empty()) {
    std::string local_addr = GetIdFromJson(args_str, "local_addr");
    std::string peer_addr = GetIdFromJson(args_str, "peer_addr");
    if (!local_addr.empty() && !peer_addr.empty()) {
      endpoint_id = absl::StrCat(local_addr, "-", peer_addr);
    } else if (!peer_addr.empty()) {
      endpoint_id = peer_addr;
    }
  }
  std::string flow_id = GetIdFromJson(args_str, "flow_id");
  if (flow_id.empty()) {
    flow_id = GetIdFromJson(args_str, "chunk_id");
  }

  if (!endpoint_id.empty() && !flow_id.empty()) {
    std::string qualified_endpoint_id =
        absl::StrCat(host_name, "_", endpoint_id);
    data_.flow_to_endpoint[flow_id] = qualified_endpoint_id;
    std::vector<std::string> flow_ids = {flow_id};
    auto get_metric_val = [&](absl::string_view base_name) -> double {
      return GetNumFromJson(args_str, base_name);
    };

    std::string metadata_type = GetIdFromJson(args_str, "metadata_type");
    std::string normalized_type;
    if (absl::StrContains(metadata_type, "SENDMSG") ||
        absl::StrContains(metadata_type, "SEND_MSG")) {
      normalized_type = "SENDMSG";
    } else if (absl::StrContains(metadata_type, "SCHEDULED")) {
      normalized_type = "SCHEDULED";
    } else if (absl::StrContains(metadata_type, "ACKED")) {
      normalized_type = "ACKED";
    } else if (absl::StrContains(metadata_type, "SENT")) {
      normalized_type = "SENT";
    } else if (absl::StrContains(metadata_type, "CLOSED")) {
      normalized_type = "CLOSED";
    }

    double byte_offset = -1.0;
    bool has_data_sent = !ExtractJsonValue(args_str, "data_sent").empty();
    if (has_data_sent) {
      double raw_data_sent = get_metric_val("data_sent");
      double data_sent_bits = BytesToBits(raw_data_sent);
      auto it = data_.endpoint_constant_offsets.find(qualified_endpoint_id);
      if (it == data_.endpoint_constant_offsets.end()) {
        if (normalized_type == "SENDMSG") {
          it = data_.endpoint_constant_offsets
                   .insert({qualified_endpoint_id, data_sent_bits})
                   .first;
        }
      }
      if (it != data_.endpoint_constant_offsets.end()) {
        byte_offset = data_sent_bits - it->second;
      }
    } else {
      double raw_byte_offset = get_metric_val("byte_offset");
      byte_offset = BytesToBits(raw_byte_offset);
    }

    {
      auto get_optional_metric = [&](absl::string_view key,
                                     bool scale_bits =
                                         false) -> std::optional<double> {
        if (ExtractJsonValue(args_str, key).empty()) {
          return std::nullopt;
        }
        double val = GetNumFromJson(args_str, key);
        if (scale_bits) {
          return BytesToBits(val);
        }
        return val;
      };

      std::optional<double> min_rtt = get_optional_metric("min_rtt");
      std::optional<double> packet_retx = get_optional_metric("packet_retx");
      std::optional<double> cwnd = get_optional_metric("congestion_window");
      std::optional<double> drate = get_optional_metric("delivery_rate", true);
      std::optional<double> data_sent = get_optional_metric("data_sent", false);
      std::optional<double> data_retx = get_optional_metric("data_retx", false);
      std::optional<double> pacing_rate =
          get_optional_metric("pacing_rate", true);
      std::optional<double> packet_sent = get_optional_metric("packet_sent");
      std::optional<double> packet_delivered =
          get_optional_metric("packet_delivered");
      std::optional<double> packet_delivered_ce =
          get_optional_metric("packet_delivered_ce");
      std::optional<double> busy_usec = get_optional_metric("busy_usec");
      std::optional<double> rwnd_limited_usec =
          get_optional_metric("rwnd_limited_usec");
      std::optional<double> sndbuf_limited_usec =
          get_optional_metric("sndbuf_limited_usec");
      std::optional<double> is_delivery_rate_app_limited =
          get_optional_metric("is_delivery_rate_app_limited");
      // Fathom reports the slow start threshold as "snd_ssthresh". The kernel's
      // TCP_INFINITE_SSTHRESH (INT32_MAX) means it is not set yet.
      std::optional<double> ssthresh = get_optional_metric("snd_ssthresh");
      if (ssthresh.has_value() &&
          *ssthresh >= std::numeric_limits<int32_t>::max()) {
        ssthresh = std::nullopt;
      }
      std::optional<double> data_notsent =
          get_optional_metric("data_notsent", false);
      std::optional<double> packet_spurious_retx =
          get_optional_metric("packet_spurious_retx");
      std::optional<double> reordering = get_optional_metric("reordering");
      std::optional<double> recurring_retrans =
          get_optional_metric("recurring_retrans");
      std::optional<double> srtt = get_optional_metric("srtt");
      std::optional<double> net_rtt_usec = get_optional_metric("net_rtt_usec");
      // Raw byte offset; `byte_offset` above is in bits for the charts.
      std::optional<int64_t> write_byte_offset = std::nullopt;
      int64_t raw_write_byte_offset = 0;
      if (absl::SimpleAtoi(ExtractJsonValue(args_str, "byte_offset"),
                           &raw_write_byte_offset)) {
        write_byte_offset = raw_write_byte_offset;
      }
      std::optional<int64_t> fathom_timestamp = std::nullopt;
      absl::string_view timestamp_str =
          ExtractJsonValue(args_str, "fathom_timestamp");
      if (!timestamp_str.empty()) {
        if (timestamp_str.front() == '"' && timestamp_str.back() == '"') {
          timestamp_str = timestamp_str.substr(1, timestamp_str.length() - 2);
        }
        absl::Time time;
        std::string err;
        int64_t nanos;
        if (absl::ParseTime(absl::RFC3339_full, timestamp_str, &time, &err)) {
          fathom_timestamp = absl::ToUnixNanos(time);
        } else if (absl::SimpleAtoi(timestamp_str, &nanos)) {
          fathom_timestamp = nanos;
        }
      }

      data_.endpoint_tcp_metrics[qualified_endpoint_id].push_back(
          TcpMetrics{min_rtt,
                     packet_retx,
                     cwnd,
                     drate,
                     data_sent,
                     data_retx,
                     pacing_rate,
                     packet_sent,
                     packet_delivered,
                     packet_delivered_ce,
                     busy_usec,
                     rwnd_limited_usec,
                     sndbuf_limited_usec,
                     is_delivery_rate_app_limited,
                     data_notsent,
                     srtt,
                     net_rtt_usec,
                     fathom_timestamp,
                     event.TimestampPs(),
                     normalized_type,
                     byte_offset,
                     qualified_endpoint_id,
                     std::move(flow_ids),
                     write_byte_offset});

      auto add_pt = [&](absl::string_view series_name, double value) {
        data_.points_by_series[series_name].push_back({rel_time_sec, value,
                                                       std::string(host_name),
                                                       qualified_endpoint_id});
      };

      if (min_rtt) add_pt("tcp_min_rtt", *min_rtt);
      if (pacing_rate) add_pt("tcp_pacing_rate", *pacing_rate);
      if (drate) add_pt("tcp_delivery_rate", *drate);
      if (ssthresh) add_pt("tcp_ssthresh", *ssthresh);
      if (is_delivery_rate_app_limited) {
        add_pt("tcp_is_delivery_rate_app_limited",
               *is_delivery_rate_app_limited);
      }
      add_pt("tcp_byte_offset", byte_offset);
      if (data_notsent) add_pt("tcp_data_notsent", BytesToBits(*data_notsent));
      if (busy_usec) add_pt("tcp_busy_usec", *busy_usec);
      if (rwnd_limited_usec) {
        add_pt("tcp_rwnd_limited_usec", *rwnd_limited_usec);
      }
      if (sndbuf_limited_usec) {
        add_pt("tcp_sndbuf_limited_usec", *sndbuf_limited_usec);
      }
      if (cwnd) add_pt("tcp_congestion_window", *cwnd);

      if (packet_retx) add_pt("tcp_packet_retx", *packet_retx);
      if (packet_spurious_retx)
        add_pt("tcp_packet_spurious_retx", *packet_spurious_retx);
      if (packet_delivered_ce)
        add_pt("tcp_packet_delivered_ce", *packet_delivered_ce);
      if (reordering) add_pt("tcp_reordering", *reordering);
      if (recurring_retrans)
        add_pt("tcp_recurring_retrans", *recurring_retrans);
      if (packet_sent) add_pt("tcp_packet_sent", *packet_sent);
      if (packet_delivered) add_pt("tcp_packet_delivered", *packet_delivered);
      if (data_sent) add_pt("tcp_data_sent", BytesToBits(*data_sent));
    }
  }
}

void XSpaceExtractor::ProcessClientCallEvent(
    const tsl::profiler::XEventVisitor& event, absl::string_view args_str,
    double rel_time_sec, absl::string_view host_name) {
  std::string flow_id = GetIdFromJson(args_str, "flow_id");
  if (!flow_id.empty()) {
    double raw_latency = GetNumFromJson(args_str, "client_latency_ns");
    // client_latency_ns is always emitted in nanoseconds by the gRPC XOR
    // filter, so convert to milliseconds unconditionally. Treating
    // sub-millisecond values as already-milliseconds corrupts genuine sub-ms
    // latencies.
    double latency_ms = raw_latency / 1e6;
    double latency_ns = raw_latency;
    data_.flow_grpc_stats[flow_id].client_latency_ms = latency_ms;
    data_.flow_grpc_stats[flow_id].client_start_ps =
        event.TimestampPs() -
        static_cast<int64_t>(latency_ns * 1e3);  // ns -> ps.
    data_.flow_grpc_stats[flow_id].status = GetNumFromJson(args_str, "status");

    data_.points_by_series["grpc_client_latency"].push_back(
        {rel_time_sec, latency_ms, std::string(host_name), ""});
  }
}

void XSpaceExtractor::ProcessServerCallEvent(
    const tsl::profiler::XEventVisitor& event, absl::string_view args_str,
    double rel_time_sec, absl::string_view host_name) {
  std::string flow_id = GetIdFromJson(args_str, "flow_id");
  if (!flow_id.empty()) {
    double raw_latency = GetNumFromJson(args_str, "server_latency_ns");
    // server_latency_ns is always emitted in nanoseconds; convert to
    // milliseconds unconditionally.
    double server_latency_ms = raw_latency / 1e6;
    double server_latency_ns = raw_latency;
    data_.flow_grpc_stats[flow_id].server_latency_ms = server_latency_ms;
    data_.flow_grpc_stats[flow_id].server_start_ps =
        event.TimestampPs() - static_cast<int64_t>(server_latency_ns * 1e3);

    data_.points_by_series["grpc_server_latency"].push_back(
        {rel_time_sec, server_latency_ms, std::string(host_name), ""});
  }
}

bool HasGrpcMetricsEvents(const XSpace& xspace) {
  const XPlane* plane = FindPlaneWithName(xspace, "/host:CPU");
  if (plane == nullptr) return false;
  // Same name rules as XSpaceExtractor::ResolveMetadataIds.
  for (const auto& [metadata_id, metadata] : plane->event_metadata()) {
    const absl::string_view event_name = metadata.name();
    if (event_name == kSendRpcEvent || event_name == kSendRpcDoneEvent ||
        absl::StrContains(event_name, kGrpcFlowWriteMetricsTrace) ||
        absl::StrContains(event_name, kClientCallMetrics) ||
        absl::StrContains(event_name, kServerCallMetrics) ||
        absl::StrContains(event_name, kSystemConfigEvent)) {
      return true;
    }
  }
  return false;
}

}  // namespace xprof::grpc_metrics
