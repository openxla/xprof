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

#include "xprof/convert/grpc_metrics/xplane_to_grpc_metrics.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/profiler/utils/xplane_builder.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_utils.h"
#include "tsl/platform/protobuf.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/grpc_metrics/xspace_extractor.h"
#include "xprof/convert/repository.h"
#include "plugin/xprof/protobuf/grpc_metrics.pb.h"
#include "plugin/xprof/protobuf/topology.pb.h"

namespace xprof::grpc_metrics {
namespace {

using ::tensorflow::profiler::SessionSnapshot;
using ::tensorflow::profiler::Topology;
using ::tensorflow::profiler::XSpace;
using ::tsl::profiler::XEventBuilder;
using ::tsl::profiler::XPlaneBuilder;

// Name of the host plane that the DCN Analyzer reads.
constexpr absl::string_view kHostPlaneName = "/host:CPU";

// Adds an event named `event_name` at `timestamp_ps` to line 0 of `plane`.
XEventBuilder AddEvent(XPlaneBuilder* plane, absl::string_view event_name,
                       int64_t timestamp_ps) {
  XEventBuilder event = plane->GetOrCreateLine(0).AddEvent(
      *plane->GetOrCreateEventMetadata(event_name));
  event.SetTimestampPs(timestamp_ps);
  return event;
}

// Returns an XSpace whose host plane has one event named `event_name`.
XSpace XSpaceWithHostEvent(absl::string_view event_name) {
  XSpace xspace;
  XPlaneBuilder host_plane(
      tsl::profiler::FindOrAddMutablePlaneWithName(&xspace, kHostPlaneName));
  AddEvent(&host_plane, event_name, 1000);
  return xspace;
}

// Why: the tool list shows the DCN Analyzer only for profiles that have its
// events.
// What: checks the events that count and that other events do not.
TEST(HasGrpcMetricsEventsTest, DetectsGrpcAndTcpFlowEvents) {
  // Setup.
  const XSpace with_send_rpc = XSpaceWithHostEvent("Megascale_gRPC:SendRpc");
  const XSpace with_tcp_write =
      XSpaceWithHostEvent("GrpcFlowWriteMetricsTrace");
  const XSpace with_other_event = XSpaceWithHostEvent("SomeOtherEvent");
  const XSpace empty_xspace;

  // Execution and verification.
  EXPECT_TRUE(HasGrpcMetricsEvents(with_send_rpc));
  EXPECT_TRUE(HasGrpcMetricsEvents(with_tcp_write));
  EXPECT_FALSE(HasGrpcMetricsEvents(with_other_event));
  EXPECT_FALSE(HasGrpcMetricsEvents(empty_xspace));
}

// Why: the DCN slack analysis uses the system topology when the profile has
// one.
// What: checks that the topology is read from the Task Environment plane and
// that a profile without it gives std::nullopt.
TEST(GetSystemTopologyTest, ReadsTopologyStatWhenPresent) {
  // Setup.
  Topology expected_topology;
  expected_topology.mutable_host_bounds()->set_x(2);
  expected_topology.mutable_chips_per_host_bounds()->set_y(4);
  XSpace with_topology;
  XPlaneBuilder task_env_plane(tsl::profiler::FindOrAddMutablePlaneWithName(
      &with_topology, tsl::profiler::kTaskEnvPlaneName));
  task_env_plane.AddStatValue(
      *task_env_plane.GetOrCreateStatMetadata("system_topology"),
      expected_topology);
  const XSpace without_topology;

  // Execution.
  const std::optional<Topology> topology = GetSystemTopology(with_topology);

  // Verification.
  ASSERT_TRUE(topology.has_value());
  EXPECT_EQ(topology->host_bounds().x(), 2);
  EXPECT_EQ(topology->chips_per_host_bounds().y(), 4);
  EXPECT_FALSE(GetSystemTopology(without_topology).has_value());
}

// Why: megacore changes how the DCN slack analysis counts cores.
// What: checks the uses_megacore stat of a TensorCore plane.
TEST(UsesMegacoreTest, ReadsTensorCorePlaneStat) {
  // Setup.
  XSpace megacore_xspace;
  XPlaneBuilder megacore_plane(tsl::profiler::FindOrAddMutablePlaneWithName(
      &megacore_xspace, tsl::profiler::TpuPlaneName(0)));
  megacore_plane.AddStatValue(
      *megacore_plane.GetOrCreateStatMetadata(tsl::profiler::GetStatTypeStr(
          tsl::profiler::StatType::kUsesMegaCore)),
      true);
  XSpace plain_xspace;
  tsl::profiler::FindOrAddMutablePlaneWithName(&plain_xspace,
                                               tsl::profiler::TpuPlaneName(0));

  // Execution and verification.
  EXPECT_TRUE(UsesMegacore(megacore_xspace));
  EXPECT_FALSE(UsesMegacore(plain_xspace));
}

// Why: the tool must finish each host, merge the hosts and then match
// stragglers across hosts.
// What: host A sends chunk 123 and host B records its server latency; the
// merged JSON has one straggler from host A with host B's server latency.
TEST(ConvertMultiXSpacesToGrpcMetricsJsonTest, MergesHostsAndMatchesLatency) {
  // Setup.
  auto host_a = std::make_unique<XSpace>();
  XPlaneBuilder host_a_plane(tsl::profiler::FindOrAddMutablePlaneWithName(
      host_a.get(), kHostPlaneName));
  const tsl::profiler::XStatMetadata& flow_stat =
      *host_a_plane.GetOrCreateStatMetadata("flow");
  AddEvent(&host_a_plane, "Megascale_gRPC:SendRpc", 1000000000)
      .AddStatValue(flow_stat, uint64_t{123});
  AddEvent(&host_a_plane, "Megascale_gRPC:SendRpcDone", 3000000000)
      .AddStatValue(flow_stat, uint64_t{123});

  auto host_b = std::make_unique<XSpace>();
  XPlaneBuilder host_b_plane(tsl::profiler::FindOrAddMutablePlaneWithName(
      host_b.get(), kHostPlaneName));
  AddEvent(&host_b_plane, "grpc_core::XorServerFilter::ServerCallMetrics",
           1100000000)
      .AddStatValue(*host_b_plane.GetOrCreateStatMetadata("args"),
                    R"({"flow_id":123, "server_latency_ns":5000000})");

  std::vector<std::unique_ptr<XSpace>> xspaces;
  xspaces.push_back(std::move(host_a));
  xspaces.push_back(std::move(host_b));
  absl::StatusOr<SessionSnapshot> session_snapshot =
      SessionSnapshot::Create({"ram://grpc_metrics_test/host_a.xplane.pb",
                               "ram://grpc_metrics_test/host_b.xplane.pb"},
                              std::move(xspaces));
  ASSERT_TRUE(session_snapshot.ok()) << session_snapshot.status();

  // Execution.
  const absl::StatusOr<std::string> json =
      ConvertMultiXSpacesToGrpcMetricsJson(*session_snapshot);

  // Verification.
  ASSERT_TRUE(json.ok()) << json.status();
  xprof::GrpcMetricsDatabase db;
  ASSERT_TRUE(tsl::protobuf::util::JsonStringToMessage(*json, &db).ok());
  ASSERT_EQ(db.top_stragglers_size(), 1);
  EXPECT_EQ(db.top_stragglers(0).chunk_id(), "123");
  EXPECT_EQ(db.top_stragglers(0).tpu_vm(), "host_a");
  EXPECT_DOUBLE_EQ(db.top_stragglers(0).latency_ms(), 2.0);
  EXPECT_DOUBLE_EQ(db.top_stragglers(0).grpc_server_latency_ms(), 5.0);
  EXPECT_EQ(db.unmatched_server_latencies_size(), 0);
}

}  // namespace
}  // namespace xprof::grpc_metrics
