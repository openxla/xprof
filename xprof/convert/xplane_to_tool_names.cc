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

#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/arena.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/profiler/utils/tf_xplane_visitor.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_utils.h"
#include "xla/tsl/profiler/utils/xplane_visitor.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/repository.h"
#include "xprof/convert/xplane_to_dcn_collective_stats.h"

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

// Returns true if any plane whose name starts with `plane_prefix` has an event
// with a counter value stat.
bool HasCounterValueInPlanes(const XSpace& xspace,
                             absl::string_view plane_prefix) {
  for (const XPlane* plane :
       tsl::profiler::FindPlanesWithPrefix(xspace, plane_prefix)) {
    tsl::profiler::XPlaneVisitor visitor =
        tsl::profiler::CreateTfXPlaneVisitor(plane);
    const XStatMetadata* counter_value_metadata =
        visitor.GetStatMetadataByType(tsl::profiler::StatType::kCounterValue);
    if (counter_value_metadata == nullptr) continue;
    for (const auto& line : plane->lines()) {
      for (const auto& event : line.events()) {
        for (const auto& stat : event.stats()) {
          if (stat.metadata_id() == counter_value_metadata->id()) return true;
        }
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
    bool has_tpu_perf_counters = false;
    bool has_gpu_perf_counters = false;

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

    has_dcn_collective_stats =
        has_dcn_collective_stats || HasDcnCollectiveStatsInXSpace(*xspace);

    has_tpu_perf_counters =
        HasCounterValueInPlanes(*xspace, tsl::profiler::kTpuPlanePrefix);
    // The perf counter converter reads both TPU and GPU planes. The
    // utilization viewer and the kernel utilization tools decode TPU counter
    // IDs only, so do not list them for GPU counters.
    has_gpu_perf_counters =
        !has_tpu_perf_counters &&
        HasCounterValueInPlanes(*xspace, tsl::profiler::kGpuPlanePrefix);

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

    if (has_tpu_perf_counters) {
      tools.push_back("perf_counters");
      tools.push_back("utilization_viewer");
      tools.push_back("kernel_utilization");
    } else if (has_gpu_perf_counters) {
      tools.push_back("perf_counters");
    }
  }

  return absl::StrJoin(tools, ",");
}

}  // namespace profiler
}  // namespace tensorflow
