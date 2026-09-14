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

#ifndef THIRD_PARTY_XPROF_CONVERT_EVENTS_DB_HOST_TRACE_PARSER_H_
#define THIRD_PARTY_XPROF_CONVERT_EVENTS_DB_HOST_TRACE_PARSER_H_

#include "absl/status/statusor.h"
#include "xla/tsl/profiler/utils/group_events.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/events_db/event_utils.h"
#include "xprof/convert/events_db/record_consumer.h"

namespace xprof::events_db::internal {

// Ingests host CPU trace events from `host_trace` (e.g. "Host Threads" or
// "/host:CPU") and streams each parsed `Record` to `consumer`.
absl::StatusOr<ParseStatus> ParseHostTrace(
    const tensorflow::profiler::XPlane& host_trace,
    const tsl::profiler::GroupMetadataMap& group_metadata_map,
    const FieldIndices& indices, RecordConsumerRef consumer);

}  // namespace xprof::events_db::internal

#endif  // THIRD_PARTY_XPROF_CONVERT_EVENTS_DB_HOST_TRACE_PARSER_H_
