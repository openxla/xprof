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

#include <optional>
#include <string>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/arena.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/profiler/utils/tf_xplane_visitor.h"
#include "xla/tsl/profiler/utils/tpu_xplane_utils.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_utils.h"
#include "xla/tsl/profiler/utils/xplane_visitor.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/grpc_metrics/grpc_metrics_types.h"
#include "xprof/convert/grpc_metrics/grpc_metrics_utils.h"
#include "xprof/convert/repository.h"
#include "plugin/xprof/protobuf/grpc_metrics.pb.h"
#include "plugin/xprof/protobuf/topology.pb.h"
#include "xprof/utils/xplane_hlo_fixer.h"

namespace xprof::grpc_metrics {

namespace {

using ::tensorflow::profiler::SessionSnapshot;
using ::tensorflow::profiler::Topology;
using ::tensorflow::profiler::XPlane;
using ::tensorflow::profiler::XSpace;

// Name of the Task Environment plane stat that holds a serialized Topology.
constexpr absl::string_view kSystemTopologyStatName = "system_topology";

}  // namespace

std::optional<Topology> GetSystemTopology(const XSpace& xspace) {
  const XPlane* task_env_plane = tsl::profiler::FindPlaneWithName(
      xspace, tsl::profiler::kTaskEnvPlaneName);
  if (task_env_plane == nullptr) return std::nullopt;

  std::optional<Topology> topology;
  const tsl::profiler::XPlaneVisitor task_env_visitor(task_env_plane);
  task_env_visitor.ForEachStat([&](const tsl::profiler::XStatVisitor& stat) {
    if (stat.Name() != kSystemTopologyStatName) return;
    Topology parsed_topology;
    if (parsed_topology.ParseFromString(stat.BytesValue())) {
      topology = std::move(parsed_topology);
    }
  });
  return topology;
}

bool UsesMegacore(const XSpace& xspace) {
  for (const XPlane* tensor_core_plane :
       tsl::profiler::FindTensorCorePlanes(xspace)) {
    const tsl::profiler::XPlaneVisitor plane_visitor =
        tsl::profiler::CreateTfXPlaneVisitor(tensor_core_plane);
    const std::optional<tsl::profiler::XStatVisitor> megacore_stat =
        plane_visitor.GetStat(tsl::profiler::StatType::kUsesMegaCore);
    if (megacore_stat.has_value() && megacore_stat->BoolValue()) return true;
  }
  return false;
}

absl::StatusOr<std::string> ConvertMultiXSpacesToGrpcMetricsJson(
    const SessionSnapshot& session_snapshot) {
  xprof::GrpcMetricsDatabase merged_db;
  for (int host_index = 0; host_index < session_snapshot.XSpaceSize();
       ++host_index) {
    const std::string host_name = session_snapshot.GetHostname(host_index);
    google::protobuf::Arena arena;
    const absl::StatusOr<XSpace*> xspace =
        session_snapshot.GetXSpace(host_index, &arena);
    if (!xspace.ok()) {
      // A host that cannot be read is left out instead of failing the whole
      // tool.
      LOG(WARNING) << "Skipping host " << host_name
                   << " in the DCN Analyzer: " << xspace.status();
      continue;
    }
    ::xprof::FixHloMetadataInXSpace(*xspace);

    // Per-host step: parse the events, then finish the host's database.
    const std::optional<Topology> topology = GetSystemTopology(**xspace);
    xprof::GrpcMetricsDatabase host_db;
    GrpcMetricsParseResults results = ProcessXSpaceToGrpcMetrics(
        **xspace, host_name, topology.has_value() ? &*topology : nullptr,
        UsesMegacore(**xspace), &host_db);
    FinalizeHostGrpcMetrics(std::move(results), &host_db);
    MergeHostGrpcMetrics(std::move(host_db), &merged_db);
  }

  // Cross-host step, then the JSON that the page reads.
  FinalizeMergedGrpcMetrics(&merged_db);
  std::string json;
  TF_RETURN_IF_ERROR(ConvertGrpcMetricsToJson(merged_db, &json));
  return json;
}

}  // namespace xprof::grpc_metrics
