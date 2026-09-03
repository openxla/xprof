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

#ifndef THIRD_PARTY_XPROF_CONVERT_EVENTS_DB_XSPACE_PARSER_H_
#define THIRD_PARTY_XPROF_CONVERT_EVENTS_DB_XSPACE_PARSER_H_

#include <optional>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional_ref.h"
#include "xla/tsl/profiler/utils/group_events.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/events_db/record_consumer.h"
#include "xprof/convert/events_db/schema.h"
#include "xprof/convert/executor_factory.h"
#include "xprof/utils/hlo_module_map.h"

namespace xprof::events_db {

// Ingests profiler trace events (host CPU, GPU/TPU devices, custom traces) from
// `xspace` and streams each parsed `Record` to `consumer`.
//
// Discovered column names and metadata are registered in `schema`.
//
// `group_metadata_map` maps step/group IDs to step metadata (such as step
// names) to associate events with their corresponding execution steps.
//
// `hlo_module_map` (optional) provides pre-computed HLO module definitions. If
// omitted (std::nullopt), it is parsed and constructed from `xspace`.
//
// Concurrency is managed via `executor_factory`, which defaults to
// `DefaultExecutorFactory` (multi-threaded thread pool).
//
// Thread-safety & Early Termination:
// When a multi-threaded executor is used, planes are parsed concurrently across
// multiple worker threads. `consumer` must be thread-safe. If `consumer`
// returns `StepControl::kStop` or an error status, in-flight worker threads
// will immediately terminate upon their next consumer invocation.
absl::StatusOr<ParseStatus> ParseXSpace(
    const tensorflow::profiler::XSpace& xspace,
    const tsl::profiler::GroupMetadataMap& group_metadata_map, Schema& schema,
    RecordConsumerRef consumer,
    absl::optional_ref<const tensorflow::profiler::HloModuleMap>
        hlo_module_map = std::nullopt,
    tensorflow::profiler::ExecutorFactoryRef executor_factory =
        tensorflow::profiler::DefaultExecutorFactory);

// Ingests profiler trace events from `xspace` and streams each parsed `Record`
// to `consumer`.
//
// Convenience overload that automatically groups TensorFlow/XLA events in
// `xspace` when `group_metadata_map` is omitted (std::nullopt). Note that
// `xspace` is modified in-place during event grouping. If `group_metadata_map`
// is provided, in-place event grouping is skipped.
//
// See the overload above for details on `hlo_module_map`, concurrency,
// thread-safety, early termination, and return values.
absl::StatusOr<ParseStatus> ParseXSpace(
    tensorflow::profiler::XSpace& xspace, Schema& schema,
    RecordConsumerRef consumer,
    absl::optional_ref<const tensorflow::profiler::HloModuleMap>
        hlo_module_map = std::nullopt,
    absl::optional_ref<const tsl::profiler::GroupMetadataMap>
        group_metadata_map = std::nullopt,
    tensorflow::profiler::ExecutorFactoryRef executor_factory =
        tensorflow::profiler::DefaultExecutorFactory);

// Ingests profiler trace events from the binary XSpace protobuf file at
// `file_path` and streams each parsed `Record` to `consumer`.
//
// See the overloads above for details on `hlo_module_map`,
// `group_metadata_map`, concurrency, thread-safety, early termination, and
// return values.
absl::StatusOr<ParseStatus> ParseXSpace(
    absl::string_view file_path, Schema& schema, RecordConsumerRef consumer,
    absl::optional_ref<const tensorflow::profiler::HloModuleMap>
        hlo_module_map = std::nullopt,
    absl::optional_ref<const tsl::profiler::GroupMetadataMap>
        group_metadata_map = std::nullopt,
    tensorflow::profiler::ExecutorFactoryRef executor_factory =
        tensorflow::profiler::DefaultExecutorFactory);

}  // namespace xprof::events_db

#endif  // THIRD_PARTY_XPROF_CONVERT_EVENTS_DB_XSPACE_PARSER_H_
