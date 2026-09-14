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

#include "xprof/convert/events_db/event_utils.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/strings/numbers.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "absl/types/span.h"
#include "xla/tsl/profiler/utils/tf_op_utils.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_visitor.h"
#include "xprof/convert/dcn_utils.h"
#include "xprof/convert/events_db/schema.h"

namespace xprof::events_db::internal {

static_assert(std::is_trivially_destructible_v<FieldIndices>);
static_assert(std::is_trivially_copyable_v<FieldIndices>);

FieldIndices::FieldIndices(Schema& schema) {
  schema.Register(device, kDevice);
  schema.Register(stream_id, kStreamId);
  schema.Register(thread_id, kThreadId);
  schema.Register(thread_name, kThreadName);
  schema.Register(correlation_id, kCorrelationId);
  schema.Register(kernel_name, kKernelName);
  schema.Register(kernel_details, kKernelDetails);
  schema.Register(tf_op_name, kTfOpName);
  schema.Register(tf_op_type, kTfOpType);
  schema.Register(hlo_op, kHloOp);
  schema.Register(hlo_module, kHloModule);
  schema.Register(start_ns, kStartNs);
  schema.Register(end_ns, kEndNs);
  schema.Register(self_time_ns, kSelfTimeNs);
  schema.Register(step, kStep);
  schema.Register(category, kCategory);
  schema.Register(flops, kFlops);
  schema.Register(memory_accessed, kMemoryAccessed);
  schema.Register(input_tensors, kInputTensors);
  schema.Register(output_tensors, kOutputTensors);
  schema.Register(trace_args, kTraceArgs);
  schema.Register(hlo_fingerprint, kHloFingerprint);
  schema.Register(flow, kFlow);
  schema.Register(source_line, kSourceLine);
  schema.Register(dcn_src_slice_id, kDcnSrcSliceId);
  schema.Register(dcn_dst_slice_id, kDcnDstSliceId);
  schema.Register(dcn_src_logical_device_id, kDcnSrcLogicalDeviceId);
  schema.Register(dcn_dst_logical_device_id, kDcnDstLogicalDeviceId);
  schema.Register(dcn_collective_name, kDcnCollectiveName);
  schema.Register(dcn_duration_us, kDcnDurationUs);
  schema.Register(dcn_payload_size_bytes, kDcnPayloadSizeBytes);
}

absl::string_view GetTpuDeviceName(absl::string_view plane_name) {
  absl::string_view device_name = plane_name;
  absl::ConsumePrefix(&device_name, "/device:");
  return device_name;
}

std::optional<uint64_t> GetProgramIdFromHloModuleName(
    absl::string_view hlo_module_name) {
  // HLO module name is in the format "<module-name>(<program-id>)". We use
  // direct string search and `absl::SimpleAtoi` instead of
  // `RE2::FullMatch(..., ".*\\((\\d+)\\)")` for significantly better
  // performance in hot trace-parsing paths, avoiding regex compilation and
  // backtracking overhead.
  if (!absl::ConsumeSuffix(&hlo_module_name, ")")) return std::nullopt;
  const size_t open_paren = hlo_module_name.rfind('(');
  if (open_paren == absl::string_view::npos) return std::nullopt;
  uint64_t program_id = 0;
  if (absl::SimpleAtoi(hlo_module_name.substr(open_paren + 1), &program_id)) {
    return program_id;
  }
  return std::nullopt;
}

std::string FormatTraceArgs(
    absl::Span<const std::pair<absl::string_view, std::string>>
        key_value_pairs) {
  return absl::StrJoin(key_value_pairs, ",", absl::PairFormatter("="));
}

void ExtractDcnEvent(const tsl::profiler::XEventVisitor& event,
                     const FieldIndices& indices, Record& output) {
  if (!tensorflow::profiler::IsDcnEvent(event)) return;
  tensorflow::profiler::DcnMessage message =
      tensorflow::profiler::GetDcnMessageFromXEvent(event);
  if (message.validity_info != tensorflow::profiler::DCN_MESSAGE_VALID &&
      message.validity_info !=
          tensorflow::profiler::DCN_MESSAGE_VALID_LOOPBACK) {
    return;
  }
  output[indices.dcn_src_slice_id] = message.slice_src;
  output[indices.dcn_dst_slice_id] = message.slice_dst;
  output[indices.dcn_src_logical_device_id] = message.tpu_src;
  output[indices.dcn_dst_logical_device_id] = message.tpu_dst;
  output[indices.dcn_collective_name] = std::move(message.collective_name);
  output[indices.dcn_payload_size_bytes] = message.size_bytes;
  output[indices.dcn_duration_us] = message.duration_us;
}

void ExtractCommonInfo(absl::string_view device_name,
                       const tsl::profiler::XLineVisitor& line,
                       const tsl::profiler::XEventVisitor& event,
                       const FieldIndices& indices, Record& output) {
  output[indices.device] = device_name;
  output[indices.category] = line.Name();
  output[indices.stream_id] = static_cast<uint32_t>(line.Id());
  output[indices.start_ns] = static_cast<uint64_t>(event.TimestampNs());
  output[indices.end_ns] = static_cast<uint64_t>(event.EndTimestampNs());
  output[indices.self_time_ns] = static_cast<uint64_t>(event.DurationNs());

  std::vector<std::pair<absl::string_view, std::string>> key_value_pairs;
  // Duplicated keys are allowed in the final `trace_args` field.
  auto for_each_stat = [&](const tsl::profiler::XStatVisitor& stat) {
    if (stat.Type().has_value()) {
      // If recognized types are duplicated, the last one wins.
      switch (stat.Type().value()) {
        case tsl::profiler::StatType::kFlops:
          output[indices.flops] = stat.IntOrUintValue();
          return;
        case tsl::profiler::StatType::kBytesAccessed:
          output[indices.memory_accessed] = stat.IntOrUintValue();
          return;
        case tsl::profiler::StatType::kTfOp: {
          tsl::profiler::TfOp tf_op =
              tsl::profiler::ParseTfOpFullname(stat.StrOrRefValue());
          output[indices.tf_op_name] = tf_op.name;
          output[indices.tf_op_type] = tf_op.type;
          return;
        }
        case tsl::profiler::StatType::kSourceInfo:
          output[indices.source_line] = stat.StrOrRefValue();
          return;
      }
    }

    key_value_pairs.emplace_back(stat.Name(), stat.ToString());
  };

  event.Metadata().ForEachStat(for_each_stat);
  event.ForEachStat(for_each_stat);

  if (!key_value_pairs.empty()) {
    output[indices.trace_args] = FormatTraceArgs(key_value_pairs);
  }
}

}  // namespace xprof::events_db::internal
