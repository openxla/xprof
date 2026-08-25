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

#ifndef THIRD_PARTY_XPROF_CONVERT_EVENTS_DB_EVENT_UTILS_H_
#define THIRD_PARTY_XPROF_CONVERT_EVENTS_DB_EVENT_UTILS_H_

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xprof/convert/events_db/schema.h"

namespace xprof::events_db::internal {

// Standard field name constants in the Events DB schema.
inline constexpr absl::string_view kDevice = "device";
inline constexpr absl::string_view kStreamId = "stream_id";
inline constexpr absl::string_view kThreadId = "thread_id";
inline constexpr absl::string_view kThreadName = "thread_name";
inline constexpr absl::string_view kCorrelationId = "correlation_id";
inline constexpr absl::string_view kKernelName = "kernel_name";
inline constexpr absl::string_view kKernelDetails = "kernel_details";
inline constexpr absl::string_view kTfOpName = "tf_op_name";
inline constexpr absl::string_view kTfOpType = "tf_op_type";
inline constexpr absl::string_view kHloOp = "hlo_op";
inline constexpr absl::string_view kHloModule = "hlo_module";
inline constexpr absl::string_view kStartNs = "start_ns";
inline constexpr absl::string_view kEndNs = "end_ns";
inline constexpr absl::string_view kSelfTimeNs = "self_time_ns";
inline constexpr absl::string_view kStep = "step";
inline constexpr absl::string_view kCategory = "category";
inline constexpr absl::string_view kFlops = "flops";
inline constexpr absl::string_view kMemoryAccessed = "memory_accessed";
inline constexpr absl::string_view kInputTensors = "input_tensors";
inline constexpr absl::string_view kOutputTensors = "output_tensors";
inline constexpr absl::string_view kTraceArgs = "trace_args";
inline constexpr absl::string_view kHloFingerprint = "hlo_fingerprint";
inline constexpr absl::string_view kFlow = "flow";
inline constexpr absl::string_view kSourceLine = "source_line";
inline constexpr absl::string_view kDcnSrcSliceId = "dcn_src_slice_id";
inline constexpr absl::string_view kDcnDstSliceId = "dcn_dst_slice_id";
inline constexpr absl::string_view kDcnSrcLogicalDeviceId =
    "dcn_src_logical_device_id";
inline constexpr absl::string_view kDcnDstLogicalDeviceId =
    "dcn_dst_logical_device_id";
inline constexpr absl::string_view kDcnCollectiveName = "dcn_collective_name";
inline constexpr absl::string_view kDcnDurationUs = "dcn_duration_us";
inline constexpr absl::string_view kDcnPayloadSizeBytes =
    "dcn_payload_size_bytes";

// Pre-registered FieldIndex cache for rapid O(1) field assignment.
struct FieldIndices final {
  explicit FieldIndices(Schema& schema);

  TypedFieldIndex<std::string> device;
  TypedFieldIndex<uint32_t> stream_id;
  TypedFieldIndex<uint32_t> thread_id;
  TypedFieldIndex<std::string> thread_name;
  TypedFieldIndex<uint32_t> correlation_id;
  TypedFieldIndex<std::string> kernel_name;
  TypedFieldIndex<std::string> kernel_details;
  TypedFieldIndex<std::string> tf_op_name;
  TypedFieldIndex<std::string> tf_op_type;
  TypedFieldIndex<std::string> hlo_op;
  TypedFieldIndex<std::string> hlo_module;
  TypedFieldIndex<uint64_t> start_ns;
  TypedFieldIndex<uint64_t> end_ns;
  TypedFieldIndex<uint64_t> self_time_ns;
  TypedFieldIndex<std::string> step;
  TypedFieldIndex<std::string> category;
  TypedFieldIndex<uint64_t> flops;
  TypedFieldIndex<uint64_t> memory_accessed;
  TypedFieldIndex<std::vector<std::string>> input_tensors;
  TypedFieldIndex<std::vector<std::string>> output_tensors;
  TypedFieldIndex<std::string> trace_args;
  TypedFieldIndex<uint64_t> hlo_fingerprint;
  TypedFieldIndex<uint64_t> flow;
  TypedFieldIndex<std::string> source_line;
  TypedFieldIndex<int64_t> dcn_src_slice_id;
  TypedFieldIndex<int64_t> dcn_dst_slice_id;
  TypedFieldIndex<int64_t> dcn_src_logical_device_id;
  TypedFieldIndex<int64_t> dcn_dst_logical_device_id;
  TypedFieldIndex<std::string> dcn_collective_name;
  TypedFieldIndex<uint64_t> dcn_duration_us;
  TypedFieldIndex<uint64_t> dcn_payload_size_bytes;
};

// Returns the TPU device name by stripping the leading "/device:" prefix
// from `plane_name` if present (e.g. "/device:TPU:0" -> "TPU:0",
// "/device:TPU_SPARSE_CORE:0" -> "TPU_SPARSE_CORE:0").
//
// If `plane_name` does not start with "/device:" (e.g. "TPU:0", "Host", or
// empty string), prefix stripping is a no-op and the function returns
// `plane_name` unchanged.
absl::string_view GetTpuDeviceName(absl::string_view plane_name);

// Extracts the numerical 64-bit program ID from an HLO module name string
// formatted as "<module_name>(<program_id>)" (e.g. "jit_model_eval(12345)" ->
// 12345, "train_step(0)" -> 0).
//
// If `hlo_module_name` does not conform to the pattern `.*\((\d+)\)` (e.g.
// "module_without_id", "model()", "model(abc)", or empty string), or if the
// captured digits overflow a uint64_t, parsing fails and `std::nullopt` is
// returned.
std::optional<uint64_t> GetProgramIdFromHloModuleName(
    absl::string_view hlo_module_name);

// Formats key-value pairs into a comma-separated string of "key=value" pairs
// for the `trace_args` schema field. Returns an empty string if
// `key_value_pairs` is empty.
std::string FormatTraceArgs(
    absl::Span<const std::pair<absl::string_view, std::string>>
        key_value_pairs);

}  // namespace xprof::events_db::internal

#endif  // THIRD_PARTY_XPROF_CONVERT_EVENTS_DB_EVENT_UTILS_H_
