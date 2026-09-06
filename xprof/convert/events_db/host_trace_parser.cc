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

#include "xprof/convert/events_db/host_trace_parser.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/profiler/utils/group_events.h"
#include "xla/tsl/profiler/utils/tf_op_utils.h"
#include "xla/tsl/profiler/utils/tf_xplane_visitor.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_visitor.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/events_db/event_utils.h"
#include "xprof/convert/events_db/record_consumer.h"

namespace xprof::events_db::internal {
namespace {

void ExtractInfo(const tsl::profiler::XLineVisitor& line,
                 const tsl::profiler::XEventVisitor& event,
                 const tsl::profiler::GroupMetadataMap& group_metadata_map,
                 const FieldIndices& indices, Record& output) {
  output[indices.thread_id] = static_cast<uint32_t>(line.Id());
  output[indices.thread_name] = line.Name();
  output[indices.start_ns] = static_cast<uint64_t>(event.TimestampNs());
  output[indices.end_ns] = static_cast<uint64_t>(event.EndTimestampNs());
  output[indices.self_time_ns] = static_cast<uint64_t>(event.DurationNs());
  absl::string_view name = event.Name();
  output[indices.kernel_name] = name;
  tsl::profiler::TfOp tf_op = tsl::profiler::ParseTfOpFullname(name);
  if (tf_op.name != name) {
    output[indices.tf_op_name] = tf_op.name;
    output[indices.tf_op_type] = tf_op.type;
  }

  std::vector<std::pair<absl::string_view, std::string>> key_value_pairs;
  // Duplicated keys are allowed in the final `trace_args` field.
  auto for_each_stat = [&](const tsl::profiler::XStatVisitor& stat) {
    if (stat.Type().has_value()) {
      // If recognized types are duplicated, the last one wins.
      switch (stat.Type().value()) {
        case tsl::profiler::StatType::kGroupId: {
          auto it = group_metadata_map.find(stat.IntValue());
          if (it != group_metadata_map.end()) {
            output[indices.step] = it->second.name;
          }
          return;
        }
        case tsl::profiler::StatType::kCorrelationId:
          output[indices.correlation_id] = stat.UintValue();
          return;
        case tsl::profiler::StatType::kFlow:
          output[indices.flow] = stat.IntOrUintValue();
          return;
        case tsl::profiler::StatType::kSourceInfo:
          output[indices.source_line] = stat.StrOrRefValue();
          return;
      }
    }

    key_value_pairs.emplace_back(stat.Name(), stat.ToString());
  };
  event.ForEachStat(for_each_stat);
  if (!key_value_pairs.empty()) {
    output[indices.trace_args] = internal::FormatTraceArgs(key_value_pairs);
  }
}

}  // namespace

absl::StatusOr<ParseStatus> ParseHostTrace(
    const tensorflow::profiler::XPlane& host_trace,
    const tsl::profiler::GroupMetadataMap& group_metadata_map,
    const FieldIndices& indices, RecordConsumerRef consumer) {
  const tsl::profiler::XPlaneVisitor plane =
      tsl::profiler::CreateTfXPlaneVisitor(&host_trace);

  Record record;
  for (const tensorflow::profiler::XLine& line : host_trace.lines()) {
    const tsl::profiler::XLineVisitor line_visitor(&plane, &line);
    for (const tensorflow::profiler::XEvent& event : line.events()) {
      const tsl::profiler::XEventVisitor event_visitor(&plane, &line, &event);
      record.clear();
      record[indices.device] = "cpu:0";
      record[indices.category] = "host";
      ExtractInfo(line_visitor, event_visitor, group_metadata_map, indices,
                  record);
      internal::ExtractDcnEvent(event_visitor, indices, record);

      TF_ASSIGN_OR_RETURN(const StepControl control, consumer(record));
      if (control == StepControl::kStop) return ParseStatus::kStoppedEarly;
    }
  }

  return ParseStatus::kComplete;
}

}  // namespace xprof::events_db::internal
