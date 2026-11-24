/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "xprof/convert/xplane_to_tool_names.h"

#include <string>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "google/protobuf/arena.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/profiler/utils/tf_xplane_visitor.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_utils.h"
#include "xla/tsl/profiler/utils/xplane_visitor.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/kernel_viewer_registry.h"
#include "xprof/convert/repository.h"
#include "xprof/convert/xplane_to_dcn_collective_stats.h"
#include "xprof/convert/xplane_to_llo.h"
#include "xprof/convert/xplane_to_utilization_viewer.h"

namespace tensorflow {
namespace profiler {
namespace {
bool HasHloProtoMetadata(const XSpace& xspace) {
  std::vector<const XPlane*> planes = tsl::profiler::FindPlanesWithNames(
      xspace, {tsl::profiler::kMetadataPlaneName});
  for (const XPlane* raw_plane : planes) {
    if (raw_plane != nullptr) {
      tsl::profiler::XPlaneVisitor plane =
          tsl::profiler::CreateTfXPlaneVisitor(raw_plane);
      const XStatMetadata* hlo_proto_stat_metadata =
          plane.GetStatMetadataByType(tsl::profiler::StatType::kHloProto);
      if (hlo_proto_stat_metadata != nullptr) {
        return true;
      }
    }
  }
  return false;
}
}  // namespace

absl::StatusOr<std::string> GetAvailableToolNames(
    const SessionSnapshot& session_snapshot) {
  std::vector<std::string> tools;
  bool is_cloud_vertex_ai = !session_snapshot.HasAccessibleRunDir();
  if (session_snapshot.XSpaceSize() != 0) {
    tools.reserve(11);
    tools.push_back(is_cloud_vertex_ai ? "trace_viewer" : "trace_viewer@");
    tools.push_back("overview_page");
    tools.push_back("input_pipeline_analyzer");
    tools.push_back("framework_op_stats");
    tools.push_back("memory_profile");
    // TODO(sannidhya): deprecate the pod_viewer.
    // b/414293137
    // tools.push_back("pod_viewer");
    tools.push_back("op_profile");
    // TODO(b/410070223): Re-enable inference_profile when it is ready.
    // tools.push_back("inference_profile");
    tools.push_back("hlo_stats");
    tools.push_back("roofline_model");

    bool has_kernel_stats = false;
    bool has_hlo = false;
    bool has_dcn_collective_stats = false;
    bool has_perf_counters = false;
    bool has_supported_utilization_device = false;
    bool has_kernel_viewer = false;

    // Use only the first host, as the sessions would consist of similar
    // devices, and the tool list can be generated from the first host itself.
    // TODO(b/413686163): Create mechanism to cache the tools list.
    // Current optimization should benefits most profiles captured in 3P
    google::protobuf::Arena arena;
    TF_ASSIGN_OR_RETURN(XSpace * xspace, session_snapshot.GetXSpace(0, &arena));

    has_kernel_stats =
        has_kernel_stats || !tsl::profiler::FindPlanesWithPrefix(
                                 *xspace, tsl::profiler::kGpuPlanePrefix)
                                 .empty();

    has_hlo = has_hlo || HasHloProtoMetadata(*xspace);

    // The Kernel Viewer needs LLO modules and a build that converts them; cache
    // the modules while the XSpace is loaded so that requests do not reload it.
    if (IsKernelViewerRegistered() && HasLloProtoMetadata(*xspace)) {
      has_kernel_viewer = true;
      absl::Status cache_status = MaybeCacheLloXPlane(
          session_snapshot, session_snapshot.GetHostname(0), *xspace);
      LOG_IF(WARNING, !cache_status.ok())
          << "Failed to cache the LLO modules: " << cache_status;
    }

    has_dcn_collective_stats =
        has_dcn_collective_stats || HasDcnCollectiveStatsInXSpace(*xspace);

    std::vector<const XPlane*> tpu_planes = tsl::profiler::FindPlanesWithPrefix(
        *xspace, tsl::profiler::kTpuPlanePrefix);
    for (const XPlane* plane : tpu_planes) {
      tsl::profiler::XPlaneVisitor visitor =
          tsl::profiler::CreateTfXPlaneVisitor(plane);
      bool plane_has_perf_counters = false;
      visitor.ForEachLine([&](const tsl::profiler::XLineVisitor& line) {
        line.ForEachEvent([&](const tsl::profiler::XEventVisitor& event) {
          if (event.GetStat(tsl::profiler::StatType::kCounterValue)) {
            plane_has_perf_counters = true;
            return;
          }
        });
        if (plane_has_perf_counters) return;
      });
      if (plane_has_perf_counters) {
        has_perf_counters = true;
        if (!has_supported_utilization_device) {
          if (auto device_type_stat =
                  visitor.GetStat(tsl::profiler::StatType::kDeviceTypeString);
              device_type_stat.has_value() &&
              xprof::ShouldProcessDevice(device_type_stat->StrOrRefValue())) {
            has_supported_utilization_device = true;
          }
        }
      }
      if (has_perf_counters && has_supported_utilization_device) break;
    }

    if (has_kernel_stats) {
      tools.push_back("kernel_stats");
    }

    if (has_hlo) {
      tools.push_back("memory_viewer");
      tools.push_back("graph_viewer");
    }

    if (has_dcn_collective_stats) {
      tools.push_back("megascale_stats");
    }

    if (has_perf_counters) {
      tools.push_back("perf_counters");
      if (has_supported_utilization_device) {
        tools.push_back("utilization_viewer");
        tools.push_back("kernel_utilization");
      }
    }

    if (has_kernel_viewer) {
      tools.push_back("kernel_viewer");
    }
  }

  return absl::StrJoin(tools, ",");
}

}  // namespace profiler
}  // namespace tensorflow
