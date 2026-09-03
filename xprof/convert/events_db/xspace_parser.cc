/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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

#include "xprof/convert/events_db/xspace_parser.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional_ref.h"
#include "xla/service/hlo.pb.h"
#include "xla/tsl/profiler/utils/device_utils.h"
#include "xla/tsl/profiler/utils/group_events.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/events_db/custom_trace_parser.h"
#include "xprof/convert/events_db/event_utils.h"
#include "xprof/convert/events_db/gpu_trace_parser.h"
#include "xprof/convert/events_db/host_trace_parser.h"
#include "xprof/convert/events_db/record_consumer.h"
#include "xprof/convert/events_db/schema.h"
#include "xprof/convert/events_db/tpu_trace_parser.h"
#include "xprof/convert/executor.h"
#include "xprof/convert/executor_factory.h"
#include "xprof/convert/file_utils.h"
#include "xprof/utils/hlo_module_map.h"
#include "xprof/utils/hlo_proto_map.h"

namespace xprof::events_db {
namespace {

struct ClassifiedPlanes {
  const tensorflow::profiler::XPlane* host_trace = nullptr;
  std::vector<const tensorflow::profiler::XPlane*> device_traces;
  std::vector<const tensorflow::profiler::XPlane*> custom_traces;
};

ClassifiedPlanes ClassifyPlanes(const tensorflow::profiler::XSpace& space) {
  ClassifiedPlanes result;
  for (const tensorflow::profiler::XPlane& plane : space.planes()) {
    if (plane.name() == tsl::profiler::kHostThreadsPlaneName) {
      result.host_trace = &plane;
    } else if (absl::StartsWith(plane.name(),
                                tsl::profiler::kCustomPlanePrefix)) {
      result.custom_traces.push_back(&plane);
    } else if (absl::StartsWith(plane.name(), tsl::profiler::kTpuPlanePrefix) ||
               absl::StartsWith(plane.name(), tsl::profiler::kGpuPlanePrefix)) {
      result.device_traces.push_back(&plane);
    }
  }
  return result;
}

tsl::profiler::GroupMetadataMap BuildGroupMetadataMap(
    tensorflow::profiler::XSpace& space) {
  tsl::profiler::EventForest event_forest;
  tsl::profiler::GroupTfEvents(&space, &event_forest);
  return event_forest.GetGroupMetadataMap();
}

tensorflow::profiler::HloModuleMap BuildHloModuleMap(
    const tensorflow::profiler::XSpace& space) {
  tensorflow::profiler::HloModuleMap hlo_module_map;
  absl::flat_hash_map<uint64_t, std::unique_ptr<xla::HloProto>> hlo_protos =
      tensorflow::profiler::ParseHloProtosFromXSpace(space);
  for (const auto& [program_id, hlo_proto] : hlo_protos) {
    tensorflow::profiler::AddHloProto(hlo_module_map, program_id, *hlo_proto,
                                      /*cost_analysis=*/nullptr);
  }
  return hlo_module_map;
}

absl::StatusOr<ParseStatus> ParseDeviceTrace(
    const tensorflow::profiler::XPlane& plane,
    const tensorflow::profiler::HloModuleMap& hlo_module_map,
    const tsl::profiler::GroupMetadataMap& group_metadata_map,
    const internal::FieldIndices& indices, RecordConsumerRef consumer) {
  const tsl::profiler::DeviceType device_type =
      tsl::profiler::GetDeviceType(plane);
  if (device_type == tsl::profiler::DeviceType::kGpu) {
    return internal::ParseGpuTrace(plane, hlo_module_map, group_metadata_map,
                                   indices, consumer);
  }
  if (device_type == tsl::profiler::DeviceType::kTpu) {
    return internal::ParseTpuTrace(plane, hlo_module_map, indices, consumer);
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Unsupported or unexpected device type for device plane: ",
                   plane.name()));
}

}  // namespace

absl::StatusOr<ParseStatus> ParseXSpace(
    const tensorflow::profiler::XSpace& xspace,
    const tsl::profiler::GroupMetadataMap& group_metadata_map, Schema& schema,
    RecordConsumerRef consumer,
    absl::optional_ref<const tensorflow::profiler::HloModuleMap> hlo_module_map,
    tensorflow::profiler::ExecutorFactoryRef executor_factory) {
  const internal::FieldIndices indices(schema);
  const ClassifiedPlanes planes = ClassifyPlanes(xspace);

  const tensorflow::profiler::HloModuleMap local_hlo_map =
      hlo_module_map.has_value() ? tensorflow::profiler::HloModuleMap()
                                 : BuildHloModuleMap(xspace);
  const tensorflow::profiler::HloModuleMap& active_hlo_map =
      hlo_module_map.has_value() ? *hlo_module_map : local_hlo_map;

  std::unique_ptr<tensorflow::profiler::Executor> executor = executor_factory();
  if (executor == nullptr) {
    return absl::InvalidArgumentError("executor_factory returned nullptr.");
  }

  std::atomic_flag stopped_early;
  std::atomic_flag failure_occurred;
  absl::Status failure_status = absl::OkStatus();

  const auto wrapped_consumer =
      [&](Record& record) -> absl::StatusOr<StepControl> {
    if (stopped_early.test(std::memory_order_relaxed)) {
      return StepControl::kStop;
    }
    return consumer(record);
  };

  const auto run_plane =
      [&](absl::FunctionRef<absl::StatusOr<ParseStatus>()> parse_fn) {
        if (stopped_early.test(std::memory_order_relaxed)) return;
        absl::StatusOr<ParseStatus> status = parse_fn();
        if (!status.ok() || *status == ParseStatus::kStoppedEarly) {
          stopped_early.test_and_set(std::memory_order_relaxed);
        }
        if (!status.ok() &&
            !failure_occurred.test_and_set(std::memory_order_relaxed)) {
          failure_status.Update(status.status());
        }
      };

  if (planes.host_trace != nullptr) {
    executor->Execute([&] {
      run_plane([&] {
        return internal::ParseHostTrace(*planes.host_trace, group_metadata_map,
                                        indices, wrapped_consumer);
      });
    });
  }

  for (const tensorflow::profiler::XPlane* plane : planes.custom_traces) {
    executor->Execute([&, plane] {
      run_plane([&] {
        return internal::ParseCustomTrace(*plane, indices, wrapped_consumer);
      });
    });
  }

  for (const tensorflow::profiler::XPlane* plane : planes.device_traces) {
    executor->Execute([&, plane] {
      run_plane([&] {
        return ParseDeviceTrace(*plane, active_hlo_map, group_metadata_map,
                                indices, wrapped_consumer);
      });
    });
  }

  executor->JoinAll();

  absl::StatusOr<ParseStatus> result;
  if (failure_occurred.test(std::memory_order_relaxed)) {
    result = std::move(failure_status);
  } else if (stopped_early.test(std::memory_order_relaxed)) {
    result = ParseStatus::kStoppedEarly;
  } else {
    result = ParseStatus::kComplete;
  }

  RETURN_IF_ERROR(consumer.Finalize(result));
  return result;
}

absl::StatusOr<ParseStatus> ParseXSpace(
    tensorflow::profiler::XSpace& xspace, Schema& schema,
    RecordConsumerRef consumer,
    absl::optional_ref<const tensorflow::profiler::HloModuleMap> hlo_module_map,
    absl::optional_ref<const tsl::profiler::GroupMetadataMap>
        group_metadata_map,
    tensorflow::profiler::ExecutorFactoryRef executor_factory) {
  if (group_metadata_map.has_value()) {
    return ParseXSpace(xspace, *group_metadata_map, schema, consumer,
                       hlo_module_map, executor_factory);
  }
  return ParseXSpace(xspace, BuildGroupMetadataMap(xspace), schema, consumer,
                     hlo_module_map, executor_factory);
}

absl::StatusOr<ParseStatus> ParseXSpace(
    absl::string_view file_path, Schema& schema,
    RecordConsumerRef consumer,
    absl::optional_ref<const tensorflow::profiler::HloModuleMap> hlo_module_map,
    absl::optional_ref<const tsl::profiler::GroupMetadataMap>
        group_metadata_map,
    tensorflow::profiler::ExecutorFactoryRef executor_factory) {
  tensorflow::profiler::XSpace xspace;
  RETURN_IF_ERROR(xprof::ReadBinaryProto(file_path, &xspace));
  return ParseXSpace(xspace, schema, consumer, hlo_module_map,
                     group_metadata_map, executor_factory);
}

}  // namespace xprof::events_db
