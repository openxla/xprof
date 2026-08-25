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

#include "absl/strings/numbers.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "absl/types/span.h"
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

}  // namespace xprof::events_db::internal
