/* Copyright 2026 The OpenXLA Authors. All Rights Reserved.

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

#ifndef THIRD_PARTY_XPROF_CONVERT_UNIFIED_SESSION_SNAPSHOT_H_
#define THIRD_PARTY_XPROF_CONVERT_UNIFIED_SESSION_SNAPSHOT_H_

#include <cstddef>
#include <string>

#include "absl/base/attributes.h"
#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/arena.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace tensorflow {
namespace profiler {

enum StoredDataType {
  DCN_COLLECTIVE_STATS,
  OP_STATS,
  TRACE_LEVELDB,
  TRACE_EVENTS_METADATA_LEVELDB,
  TRACE_EVENTS_PREFIX_TRIE_LEVELDB,
  SMART_SUGGESTION,
  RIEGELI_XSPACE,
};

}  // namespace profiler
}  // namespace tensorflow

namespace xprof {

class XprofSessionSnapshot {
 public:
  virtual ~XprofSessionSnapshot() = default;

  virtual size_t XSpaceSize() const = 0;

  virtual absl::StatusOr<tensorflow::profiler::XSpace* absl_nonnull> GetXSpace(
      size_t index,
      google::protobuf::Arena* absl_nonnull arena ABSL_ATTRIBUTE_LIFETIME_BOUND) const
      ABSL_ATTRIBUTE_LIFETIME_BOUND = 0;

  virtual std::string GetHostname(size_t index) const = 0;

  virtual absl::string_view GetSessionRunDir() const
      ABSL_ATTRIBUTE_LIFETIME_BOUND = 0;

  virtual absl::StatusOr<std::string> GetHostDataFileName(
      tensorflow::profiler::StoredDataType data_type,
      absl::string_view host) const = 0;
};

}  // namespace xprof

#endif  // THIRD_PARTY_XPROF_CONVERT_UNIFIED_SESSION_SNAPSHOT_H_
