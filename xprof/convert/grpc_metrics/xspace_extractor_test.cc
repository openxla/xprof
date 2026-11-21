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

#include <cstdint>

#include <gtest/gtest.h>
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/profiler/utils/xplane_builder.h"

namespace xprof::grpc_metrics {
namespace {

TEST(XSpaceExtractorTest, ProcessesSendRpcEventDcnDetails) {
  tsl::profiler::XSpace xspace;
  tsl::profiler::XPlaneBuilder host_plane_builder(xspace.add_planes());
  host_plane_builder.SetName("/host:CPU");

  tsl::profiler::XLineBuilder line_builder =
      host_plane_builder.GetOrCreateLine(1);
  tsl::profiler::XEventBuilder event_builder = line_builder.AddEvent(
      *host_plane_builder.GetOrCreateEventMetadata("Megascale_gRPC:SendRpc"));
  event_builder.SetTimestampNs(100);

  // Transfer key with label "test_collective", source 100001, destination
  // 100002, chunk 5, launch id 3 and loop index 4.
  event_builder.AddStatValue(*host_plane_builder.GetOrCreateStatMetadata("key"),
                             "test_collective|s100001->100002d|$c5$l3^i4");

  event_builder.AddStatValue(
      *host_plane_builder.GetOrCreateStatMetadata("flow"), 12345);

  XSpaceExtractor extractor;
  RawMetricsData data = extractor.Extract(xspace, "test_host", nullptr, false);

  ASSERT_TRUE(data.flow_to_dcn_details.contains("12345"));
  const auto& details = data.flow_to_dcn_details.at("12345");

  EXPECT_EQ(details.dcn_source_device_id, "100001");
  EXPECT_EQ(details.dcn_destination_device_id, "100002");
  EXPECT_EQ(details.dcn_launch_id, "3");
  EXPECT_EQ(details.dcn_loop_index, "4");
  EXPECT_EQ(details.dcn_chunk_index, "5");
  EXPECT_EQ(details.dcn_collective_label, "test_collective");
}

// The SendRpc `buffer_size` (chunk size) is recorded per flow.
TEST(XSpaceExtractorTest, RecordsSendRpcBufferSize) {
  tsl::profiler::XSpace xspace;
  tsl::profiler::XPlaneBuilder host_plane_builder(xspace.add_planes());
  host_plane_builder.SetName("/host:CPU");
  tsl::profiler::XLineBuilder line_builder =
      host_plane_builder.GetOrCreateLine(1);
  tsl::profiler::XEventBuilder event_builder = line_builder.AddEvent(
      *host_plane_builder.GetOrCreateEventMetadata("Megascale_gRPC:SendRpc"));
  event_builder.SetTimestampNs(100);
  event_builder.AddStatValue(
      *host_plane_builder.GetOrCreateStatMetadata("buffer_size"), 8428544);
  event_builder.AddStatValue(
      *host_plane_builder.GetOrCreateStatMetadata("flow"), 12345);

  XSpaceExtractor extractor;
  const RawMetricsData data =
      extractor.Extract(xspace, "test_host", nullptr, false);

  ASSERT_TRUE(data.flow_buffer_sizes.contains("12345"));
  EXPECT_EQ(data.flow_buffer_sizes.at("12345"), 8428544);
}

// Verifies that a sub-millisecond client latency (in nanoseconds) is converted
// to milliseconds unconditionally. The old heuristic treated any value below
// 1e6 ns as if it were already milliseconds, inflating genuine sub-ms latencies
// by ~1000x.
TEST(XSpaceExtractorTest, ConvertsSubMillisecondClientLatencyNsToMs) {
  tsl::profiler::XSpace xspace;
  tsl::profiler::XPlaneBuilder host_plane_builder(xspace.add_planes());
  host_plane_builder.SetName("/host:CPU");

  tsl::profiler::XLineBuilder line_builder =
      host_plane_builder.GetOrCreateLine(1);
  tsl::profiler::XEventBuilder event_builder =
      line_builder.AddEvent(*host_plane_builder.GetOrCreateEventMetadata(
          "grpc_core::XorClientFilter::ClientCallMetrics"));
  event_builder.SetTimestampNs(1000);

  event_builder.AddStatValue(
      *host_plane_builder.GetOrCreateStatMetadata("event_args"),
      R"({"flow_id":123,"client_latency_ns":800000,"status":0})");

  XSpaceExtractor extractor;
  RawMetricsData data = extractor.Extract(xspace, "test_host", nullptr, false);

  ASSERT_TRUE(data.flow_grpc_stats.contains("123"));
  // 800000 ns is 0.8 ms, not 800000 ms.
  EXPECT_NEAR(data.flow_grpc_stats.at("123").client_latency_ms, 0.8, 1e-9);
}

TEST(XSpaceExtractorTest,
     ProcessesGrpcSystemConfigTraceEventWithNetworkInterfaces) {
  tsl::profiler::XSpace xspace;
  tsl::profiler::XPlaneBuilder host_plane_builder(xspace.add_planes());
  host_plane_builder.SetName("/host:CPU");

  tsl::profiler::XLineBuilder line_builder =
      host_plane_builder.GetOrCreateLine(1);
  tsl::profiler::XEventBuilder event_builder = line_builder.AddEvent(
      *host_plane_builder.GetOrCreateEventMetadata("GrpcSystemConfigTrace"));
  event_builder.SetTimestampNs(100);

  event_builder.AddStatValue(
      *host_plane_builder.GetOrCreateStatMetadata("event_args"),
      R"({"network_interfaces":"[{\"carrier\":1,\"delta_rx_bytes\":1024,\"delta_rx_dropped\":0,\"delta_rx_errors\":0,\"delta_rx_packets\":10,\"delta_tx_bytes\":2048,\"delta_tx_dropped\":0,\"delta_tx_errors\":0,\"delta_tx_packets\":20,\"duplex\":\"full\",\"iface\":\"eth0\",\"is_physical\":true,\"mtu\":1500,\"numa_node\":0,\"operstate\":\"up\",\"speed_mbps\":200000,\"start_rx_bytes\":0,\"stop_rx_bytes\":1024,\"start_tx_bytes\":0,\"stop_tx_bytes\":2048}]"})");

  XSpaceExtractor extractor;
  RawMetricsData data = extractor.Extract(xspace, "test_host", nullptr, false);

  ASSERT_EQ(data.trace_events_json.size(), 1);
  EXPECT_TRUE(
      absl::StrContains(data.trace_events_json[0], "GrpcSystemConfigTrace"));
  EXPECT_TRUE(
      absl::StrContains(data.trace_events_json[0], "network_interfaces"));
  EXPECT_TRUE(absl::StrContains(data.trace_events_json[0], "is_physical"));
  EXPECT_TRUE(absl::StrContains(data.trace_events_json[0], "numa_node"));
  EXPECT_TRUE(absl::StrContains(data.trace_events_json[0], "delta_rx_bytes"));
  EXPECT_TRUE(absl::StrContains(data.trace_events_json[0], "delta_rx_dropped"));
  EXPECT_TRUE(absl::StrContains(data.trace_events_json[0], "test_host"));
}

// Fathom's key is "snd_ssthresh"; 2147483647 means not set yet.
TEST(XSpaceExtractorTest, PlotsSndSsthresh) {
  tsl::profiler::XSpace xspace;
  tsl::profiler::XPlaneBuilder host_plane_builder(xspace.add_planes());
  host_plane_builder.SetName("/host:CPU");
  tsl::profiler::XLineBuilder line_builder =
      host_plane_builder.GetOrCreateLine(1);
  const auto add_tcp_event = [&](int64_t timestamp_ns,
                                 absl::string_view ssthresh) {
    tsl::profiler::XEventBuilder event_builder =
        line_builder.AddEvent(*host_plane_builder.GetOrCreateEventMetadata(
            "grpc_core::GrpcFlowWriteMetricsTrace"));
    event_builder.SetTimestampNs(timestamp_ns);
    event_builder.AddStatValue(
        *host_plane_builder.GetOrCreateStatMetadata("event_args"),
        absl::StrCat(R"({"endpoint_id":"7","flow_id":123,)",
                     R"("metadata_type":"Endpoint Write: ACKED",)",
                     R"("snd_ssthresh":)", ssthresh, "}"));
  };
  add_tcp_event(100, "120");
  add_tcp_event(200, "2147483647");

  XSpaceExtractor extractor;
  const RawMetricsData data =
      extractor.Extract(xspace, "test_host", nullptr, false);

  ASSERT_TRUE(data.points_by_series.contains("tcp_ssthresh"));
  const auto& points = data.points_by_series.at("tcp_ssthresh");
  ASSERT_EQ(points.size(), 1);
  EXPECT_EQ(points[0].value, 120);
}
}  // namespace
}  // namespace xprof::grpc_metrics
