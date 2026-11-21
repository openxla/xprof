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
#ifndef THIRD_PARTY_XPROF_CONVERT_TRACE_VIEWER_TRACE_EVENTS_TEST_UTILS_H_
#define THIRD_PARTY_XPROF_CONVERT_TRACE_VIEWER_TRACE_EVENTS_TEST_UTILS_H_

#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/file_system.h"
#include "xprof/convert/trace_viewer/trace_events.h"

namespace tensorflow {
namespace profiler {

// Writes `container` to LevelDB event, metadata, and prefix-trie table files
// under `temp_dir` prefixed with `file_prefix`, returning the resulting paths.
template <typename ContainerT>
absl::StatusOr<TraceEventsLevelDbFilePaths> StoreTraceContainerToLevelDbTables(
    absl::string_view temp_dir, absl::string_view file_prefix,
    const ContainerT& container) {
  const std::string base = absl::StrCat(temp_dir, "/", file_prefix);
  TraceEventsLevelDbFilePaths file_paths = {
      .trace_events_file_path = absl::StrCat(base, "_events.ldb"),
      .trace_events_metadata_file_path = absl::StrCat(base, "_metadata.ldb"),
      .trace_events_prefix_trie_file_path = absl::StrCat(base, "_trie.ldb"),
  };

  std::unique_ptr<tsl::WritableFile> events_file;
  std::unique_ptr<tsl::WritableFile> metadata_file;
  std::unique_ptr<tsl::WritableFile> trie_file;
  if (absl::Status status = tsl::Env::Default()->NewWritableFile(
          file_paths.trace_events_file_path, &events_file);
      !status.ok()) {
    return status;
  }
  if (absl::Status status = tsl::Env::Default()->NewWritableFile(
          file_paths.trace_events_metadata_file_path, &metadata_file);
      !status.ok()) {
    return status;
  }
  if (absl::Status status = tsl::Env::Default()->NewWritableFile(
          file_paths.trace_events_prefix_trie_file_path, &trie_file);
      !status.ok()) {
    return status;
  }
  if (absl::Status status = container.StoreAsLevelDbTables(
          std::move(events_file), std::move(metadata_file),
          std::move(trie_file));
      !status.ok()) {
    return status;
  }
  return file_paths;
}

}  // namespace profiler
}  // namespace tensorflow

#endif  // THIRD_PARTY_XPROF_CONVERT_TRACE_VIEWER_TRACE_EVENTS_TEST_UTILS_H_
