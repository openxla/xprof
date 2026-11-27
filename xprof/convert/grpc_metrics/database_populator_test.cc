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

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xprof/convert/grpc_metrics/grpc_metrics_types.h"
#include "xprof/convert/grpc_metrics/xspace_extractor.h"
#include "plugin/xprof/protobuf/dcn_slack_analysis.pb.h"
#include "plugin/xprof/protobuf/grpc_metrics.pb.h"

namespace xprof::grpc_metrics {
namespace {

TEST(DatabasePopulatorTest, PopulatesDcnTransferDetails) {
  RawMetricsData data;
  data.host_name = "test_host";

  // Set up a flow.
  data.flow_start_times["123"] = {1000, "test_host"};
  data.flow_end_times["123"] = 2000;

  // Set up DCN details.
  DcnTransferDetails dcn_details;
  dcn_details.dcn_source_device_id = "100001";
  dcn_details.dcn_destination_device_id = "100002";
  dcn_details.dcn_launch_id = "3";
  dcn_details.dcn_loop_index = "4";
  dcn_details.dcn_chunk_index = "5";
  dcn_details.dcn_shard_range = "6-7";
  dcn_details.dcn_collective_label = "test_collective";
  dcn_details.raw_transfer_key = "test_key";

  data.flow_to_dcn_details["123"] = dcn_details;

  xprof::GrpcMetricsDatabase db;
  DatabasePopulator populator;
  populator.Populate(data, &db);

  ASSERT_EQ(db.top_stragglers_size(), 1);
  EXPECT_EQ(db.top_stragglers(0).dcn_source_device_id(), "100001");
  EXPECT_EQ(db.top_stragglers(0).dcn_destination_device_id(), "100002");
  EXPECT_EQ(db.top_stragglers(0).dcn_launch_id(), "3");
  EXPECT_EQ(db.top_stragglers(0).dcn_loop_index(), "4");
  EXPECT_EQ(db.top_stragglers(0).dcn_chunk_index(), "5");
  EXPECT_EQ(db.top_stragglers(0).dcn_shard_range(), "6-7");
  EXPECT_EQ(db.top_stragglers(0).dcn_collective_label(), "test_collective");
  EXPECT_EQ(db.top_stragglers(0).raw_transfer_key(), "test_key");
}

// Tests prefix matching fallback for DCN slack summaries when rendezvous name
// has suffixes. Verifies that a base collective name in slack_map matches a
// suffixed rendezvous name.
TEST(DatabasePopulatorTest, PopulatesDcnSlackSummaryWithPrefixMatching) {
  // Setup.
  RawMetricsData data;
  data.host_name = "test_host";
  data.flow_start_times["123"] = {1000, "test_host"};
  data.flow_end_times["123"] = 2000;
  data.flow_to_rendezvous["123"] = "all-reduce.76_0_rs_0_sr_0_1";

  tensorflow::profiler::DcnSlackSummary summary;
  summary.set_rendezvous("all-reduce.76");
  summary.set_recv_op_name("all-reduce-recv");
  summary.set_slack_us(5000);
  summary.set_stall_duration_us(2000);
  data.dcn_slack_summaries.push_back(summary);

  // Execution.
  xprof::GrpcMetricsDatabase db;
  DatabasePopulator populator;
  populator.Populate(data, &db);

  // Verification.
  ASSERT_EQ(db.top_stragglers_size(), 1);
  EXPECT_EQ(db.top_stragglers(0).slack_time_ms(), 5.0);
  EXPECT_EQ(db.top_stragglers(0).tpu_stall_time_ms(), 2.0);
  EXPECT_EQ(db.top_stragglers(0).communication_overlap_ms(), 5.0);
  EXPECT_EQ(db.top_stragglers(0).hlo_collective_name(), "all-reduce-recv");
  EXPECT_EQ(db.top_stragglers(0).dcn_collective_label(), "all-reduce.76");
}

// TCP latency tests. The times come from a real profile where an
// 8,428,736-byte write needed two sendmsg() calls.
constexpr char kTcpFlowId[] = "282674346809085379";
constexpr char kTcpEndpointId[] = "test_host_[::ffff:192.0.2.1]:38044";
constexpr int64_t kTcpBaseNs = 1792002492620818970;
constexpr int64_t kNsPerMs = 1000000;

TcpMetrics MakeTcpEvent(absl::string_view type, int64_t offset_from_base_ns,
                        int64_t write_byte_offset,
                        absl::string_view flow_id = kTcpFlowId) {
  TcpMetrics metric{};
  metric.type = std::string(type);
  metric.fathom_timestamp = kTcpBaseNs + offset_from_base_ns;
  metric.timestamp_ps = offset_from_base_ns * 1000;
  metric.write_byte_offset = write_byte_offset;
  metric.endpoint_id = kTcpEndpointId;
  metric.flow_ids = {std::string(flow_id)};
  return metric;
}

// Adds a flow with `chunk_bytes` (0 = unknown) to `data`.
void AddTcpFlow(absl::string_view flow_id, int64_t chunk_bytes,
                RawMetricsData& data) {
  const std::string id(flow_id);
  data.host_name = "test_host";
  data.flow_start_times[id] = {0, "test_host"};
  data.flow_end_times[id] = 100 * kNsPerMs * 1000;
  data.flow_to_endpoint[id] = kTcpEndpointId;
  if (chunk_bytes > 0) data.flow_buffer_sizes[id] = chunk_bytes;
}

xprof::GrpcStraggler FindStraggler(const xprof::GrpcMetricsDatabase& db,
                                   absl::string_view chunk_id) {
  for (const xprof::GrpcStraggler& straggler : db.top_stragglers()) {
    if (straggler.chunk_id() == chunk_id) return straggler;
  }
  ADD_FAILURE() << "No straggler for chunk " << chunk_id;
  return xprof::GrpcStraggler();
}

xprof::GrpcMetricsDatabase Populate(const RawMetricsData& data) {
  xprof::GrpcMetricsDatabase db;
  DatabasePopulator populator;
  populator.Populate(data, &db);
  return db;
}

struct TcpLatencyCase {
  std::string name;
  std::vector<TcpMetrics> events;
  int64_t chunk_bytes;
  // Unset: no TCP latency.
  std::optional<double> latency_ms;
  bool incomplete;
};

class TcpLatencyTest : public testing::TestWithParam<TcpLatencyCase> {};

// Each case checks the TCP latency (first SENDMSG -> last ACK) and whether it
// is flagged as incomplete.
TEST_P(TcpLatencyTest, ComputesLatencyAndIncompleteFlag) {
  RawMetricsData data;
  AddTcpFlow(kTcpFlowId, GetParam().chunk_bytes, data);
  data.endpoint_tcp_metrics[kTcpEndpointId] = GetParam().events;

  const xprof::GrpcStraggler straggler =
      FindStraggler(Populate(data), kTcpFlowId);

  ASSERT_EQ(straggler.has_ack_to_sendmsg_ms(),
            GetParam().latency_ms.has_value());
  if (GetParam().latency_ms.has_value()) {
    EXPECT_DOUBLE_EQ(straggler.ack_to_sendmsg_ms(), *GetParam().latency_ms);
  }
  EXPECT_EQ(straggler.tcp_latency_incomplete(), GetParam().incomplete);
  EXPECT_FALSE(straggler.tcp_write_shared());
}

TcpMetrics WithoutFathomTime(TcpMetrics metric) {
  metric.fathom_timestamp.reset();
  return metric;
}

INSTANTIATE_TEST_SUITE_P(
    DatabasePopulatorTest, TcpLatencyTest,
    testing::ValuesIn(std::vector<TcpLatencyCase>{
        {"Complete",
         {MakeTcpEvent("SENDMSG", 0, 2064575),
          MakeTcpEvent("SCHEDULED", 10000, 63),
          MakeTcpEvent("ACKED", 5 * kNsPerMs, 2064575)},
         0,
         5.0,
         false},
        // Unsorted, two sendmsg() calls, both ACKs recorded.
        {"FirstSendmsgToLastAck",
         {MakeTcpEvent("ACKED", 60 * kNsPerMs, 8428735),
          MakeTcpEvent("SENDMSG", 44 * kNsPerMs, 8428735),
          MakeTcpEvent("ACKED", 28 * kNsPerMs, 2971647),
          MakeTcpEvent("SENDMSG", 0, 2971647)},
         0,
         60.0,
         false},
        // From the real profile: no ACK after the second SENDMSG.
        {"AckBeforeLastSendmsg",
         {MakeTcpEvent("SENDMSG", 51918, 2971647),
          MakeTcpEvent("ACKED", 28345434, 2971647),
          MakeTcpEvent("SENDMSG", 44475093, 8428735)},
         0,
         28.293516,
         true},
        {"AckDoesNotCoverLastByte",
         {MakeTcpEvent("SENDMSG", 0, 1000),
          MakeTcpEvent("SENDMSG", 1 * kNsPerMs, 2000),
          MakeTcpEvent("ACKED", 5 * kNsPerMs, 1000)},
         0,
         5.0,
         true},
        // The rest of the write went out without timestamps.
        {"WriteNotFinished",
         {MakeTcpEvent("SENDMSG", 0, 7862379),
          MakeTcpEvent("ACKED", 5 * kNsPerMs, 7862379)},
         8428544,
         5.0,
         true},
        {"WriteFinished",
         {MakeTcpEvent("SENDMSG", 0, 8428735),
          MakeTcpEvent("ACKED", 6 * kNsPerMs, 8428735)},
         8428544,
         6.0,
         false},
        // From the real profile: the only ACK is before the SENDMSG.
        {"NoAckAfterSendmsg",
         {MakeTcpEvent("ACKED", 16575145, 3369627),
          MakeTcpEvent("SENDMSG", 20310328, 8428735)},
         0,
         std::nullopt,
         false},
        // Kernel and profiler times are never subtracted.
        {"IgnoresEventsWithoutFathomTime",
         {WithoutFathomTime(MakeTcpEvent("SENDMSG", 0, 8428735)),
          MakeTcpEvent("ACKED", 5 * kNsPerMs, 8428735)},
         0,
         std::nullopt,
         false},
    }),
    [](const testing::TestParamInfo<TcpLatencyCase>& info) {
      return info.param.name;
    });

// RPCs A (3 MiB) and B (5 MiB) went out in one write; C alone in a later one.
// Every RPC in a write gets the write's events.
RawMetricsData MakeSharedWriteData(int64_t write_last_byte) {
  constexpr int64_t kMiB = 1024 * 1024;
  RawMetricsData data;
  AddTcpFlow("111", 3 * kMiB, data);
  AddTcpFlow("222", 5 * kMiB, data);
  AddTcpFlow("333", 1 * kMiB, data);
  std::vector<TcpMetrics>& events = data.endpoint_tcp_metrics[kTcpEndpointId];
  for (const absl::string_view flow_id : {"111", "222"}) {
    events.push_back(MakeTcpEvent("SENDMSG", 0, write_last_byte, flow_id));
    events.push_back(
        MakeTcpEvent("ACKED", 6 * kNsPerMs, write_last_byte, flow_id));
  }
  events.push_back(MakeTcpEvent("SENDMSG", 20 * kNsPerMs, 1048639, "333"));
  events.push_back(MakeTcpEvent("ACKED", 22 * kNsPerMs, 1048639, "333"));
  return data;
}

// RPCs in one write are flagged as shared, and the end-of-write check uses the
// size of all of them, not of one chunk.
TEST(DatabasePopulatorTest, TcpLatencyOfSharedWrites) {
  // The write is acked to its last byte (8 MiB plus two 64-byte headers).
  const xprof::GrpcMetricsDatabase complete =
      Populate(MakeSharedWriteData(8 * 1024 * 1024 + 2 * 64 - 1));
  // The only SENDMSG ends at 6,000,000 bytes, past either chunk alone.
  const xprof::GrpcMetricsDatabase unfinished =
      Populate(MakeSharedWriteData(5999999));

  for (const absl::string_view flow_id : {"111", "222"}) {
    const xprof::GrpcStraggler straggler = FindStraggler(complete, flow_id);
    EXPECT_DOUBLE_EQ(straggler.ack_to_sendmsg_ms(), 6.0) << flow_id;
    EXPECT_TRUE(straggler.tcp_write_shared()) << flow_id;
    EXPECT_FALSE(straggler.tcp_latency_incomplete()) << flow_id;
    EXPECT_TRUE(FindStraggler(unfinished, flow_id).tcp_latency_incomplete())
        << flow_id;
  }
  const xprof::GrpcStraggler other = FindStraggler(complete, "333");
  EXPECT_DOUBLE_EQ(other.ack_to_sendmsg_ms(), 2.0);
  EXPECT_FALSE(other.tcp_write_shared());
}

}  // namespace
}  // namespace xprof::grpc_metrics
